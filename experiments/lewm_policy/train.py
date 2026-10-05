"""Behavior-clone a successful visual demo, then fine-tune with PPO in MuJoCo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from sim.training_config import DEFAULT_CONFIG_PATH, load_training_config

from .env import BallCupVisionEnv
from .features import VisionProprioFeatures
from .record_demo import record_demo


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEWM_REPO = Path(
    "/Users/rohan/Documents/Python-Projects/machine-learning/Le-World-Model-Implementation"
)


def _demonstration_features(model: PPO, camera_images, robot_state, batch_size=32):
    device = model.device
    images = torch.as_tensor(camera_images, device=device).permute(0, 3, 1, 2)
    states = torch.as_tensor(robot_state, dtype=torch.float32, device=device)
    features = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            observation = {
                "image": images[start : start + batch_size],
                "proprio": states[start : start + batch_size],
            }
            features.append(model.policy.extract_features(observation).detach())
    return torch.cat(features), states


def behavior_clone_actor(
    model: PPO,
    camera_images: np.ndarray,
    robot_state: np.ndarray,
    actions: np.ndarray,
    *,
    epochs: int = 60,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
) -> float:
    """Initialize the PPO actor from successful demonstrations, with frozen vision."""
    features, _ = _demonstration_features(model, camera_images, robot_state)
    action_tensor = torch.as_tensor(actions, dtype=torch.float32, device=model.device)
    policy = model.policy
    trainable = list(policy.mlp_extractor.policy_net.parameters()) + list(
        policy.action_net.parameters()
    )
    optimizer = torch.optim.Adam(trainable, lr=learning_rate)
    final_mse = float("nan")
    for _ in range(int(epochs)):
        order = torch.randperm(len(action_tensor), device=model.device)
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            latent_pi, _ = policy.mlp_extractor(features[indices])
            mean_actions = policy.action_net(latent_pi)
            loss = F.mse_loss(mean_actions, action_tensor[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_mse = float(loss.detach().cpu())
    return final_mse


def evaluate(model: PPO, *, stage: int, episodes: int, seed: int) -> dict:
    env = BallCupVisionEnv(stage=stage, seed=seed)
    rows = []
    try:
        for episode in range(int(episodes)):
            observation, _ = env.reset(seed=seed + episode)
            done = False
            total_reward = 0.0
            info = {}
            length = 0
            while not done:
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                length += 1
                done = bool(terminated or truncated)
            metrics = info.get("episode_metrics", info)
            rows.append(
                {
                    "episode": episode + 1,
                    "seed": seed + episode,
                    "return": total_reward,
                    "length": length,
                    **{
                        name: bool(metrics.get(name, info.get(name, False)))
                        for name in (
                            "is_success",
                            "reach_success",
                            "grasp_success",
                            "lift_success",
                            "transport_success",
                            "above_cup_success",
                            "correct_release",
                            "placement_success",
                        )
                    },
                }
            )
    finally:
        env.close()
    phases = (
        "reach_success",
        "grasp_success",
        "lift_success",
        "transport_success",
        "above_cup_success",
        "correct_release",
        "placement_success",
    )
    return {
        "stage": int(stage),
        "seed": int(seed),
        "episodes": len(rows),
        "successes": sum(row["is_success"] for row in rows),
        "success_rate": float(np.mean([row["is_success"] for row in rows])),
        "mean_return": float(np.mean([row["return"] for row in rows])),
        "phase_rates": {
            phase: float(np.mean([row[phase] for row in rows]) if rows else 0.0)
            for phase in phases
        },
        "episode_rows": rows,
    }


class VisionEvaluationCallback(BaseCallback):
    def __init__(self, output_dir: Path, *, stage: int, eval_freq: int, episodes: int, seed: int):
        super().__init__(verbose=0)
        self.output_dir = output_dir
        self.stage = stage
        self.eval_freq = eval_freq
        self.episodes = episodes
        self.seed = seed
        self.next_eval = eval_freq
        self.best_success_rate = -1.0
        self.eval_index = 0

    def _on_step(self) -> bool:
        if self.num_timesteps < self.next_eval:
            return True
        self.eval_index += 1
        result = evaluate(
            self.model,
            stage=self.stage,
            episodes=self.episodes,
            seed=self.seed + self.eval_index * self.episodes,
        )
        result.update(
            {
                "evaluation_index": self.eval_index,
                "global_env_steps": int(self.num_timesteps),
            }
        )
        with (self.output_dir / "evaluations.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result, sort_keys=True) + "\n")
        print(
            f"Visual policy evaluation {self.eval_index}: "
            f"{result['successes']}/{result['episodes']} successes "
            f"({result['success_rate']:.1%}); phases={result['phase_rates']}"
        )
        self.model.save(str(self.output_dir / "latest_model"))
        if result["success_rate"] > self.best_success_rate:
            self.best_success_rate = result["success_rate"]
            self.model.save(str(self.output_dir / "best_model"))
        self.next_eval += self.eval_freq
        return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-backbone", choices=("lewm", "resnet18"), default="lewm")
    parser.add_argument("--lewm-repo", type=Path, default=DEFAULT_LEWM_REPO)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--demo", type=Path, default=None)
    parser.add_argument("--stage", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--timesteps", type=int, default=250_000)
    parser.add_argument("--eval-freq", type=int, default=25_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--bc-epochs", type=int, default=60)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--random-resnet", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args(argv)

    config = load_training_config(args.config)
    if int(config["task"]["control_hz"]) != 20 or float(config["task"]["arm_delta_degrees"]) != 2.0:
        raise ValueError("This policy path expects the configured 20 Hz / ±2 degree controls")

    checkpoint = args.checkpoint or args.lewm_repo / "lewm_seq_projectors.pt"
    demo_path = args.demo or PROJECT_ROOT / "outputs" / "lewm_policy" / "expert_demo.npz"
    if not demo_path.is_file():
        if args.visual_backbone == "lewm" and not checkpoint.is_file():
            raise FileNotFoundError(f"LeWM checkpoint not found: {checkpoint}")
        print(f"Recording successful scripted demonstration to {demo_path}")
        record_demo(args.config, demo_path, seed=int(config["seed"]))

    with np.load(demo_path, allow_pickle=False) as demo:
        camera_images = demo["camera_images"].copy()
        robot_state = demo["robot_state"].copy()
        demo_actions = demo["actions"].copy()
    if not (len(camera_images) == len(robot_state) == len(demo_actions)):
        raise ValueError("Demonstration camera, state, and action counts do not match")

    output_dir = args.output_dir or (
        PROJECT_ROOT / "outputs" / "lewm_policy" / args.visual_backbone / f"stage_{args.stage:02d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    set_random_seed(int(config["seed"]))

    def make_training_env():
        env = BallCupVisionEnv(stage=args.stage, seed=int(config["seed"]), config_path=args.config)
        return Monitor(env)

    vector_env = DummyVecEnv([make_training_env])
    ppo = config["ppo"]
    checkpoint_kwargs = {}
    if args.visual_backbone == "lewm":
        checkpoint_kwargs = {
            "lewm_repo": str(args.lewm_repo.resolve()),
            "lewm_checkpoint": str(checkpoint.resolve()),
        }
    policy_kwargs = {
        "features_extractor_class": VisionProprioFeatures,
        "features_extractor_kwargs": {
            "visual_backbone": args.visual_backbone,
            "imagenet_weights": not args.random_resnet,
            **checkpoint_kwargs,
            "features_dim": 256,
        },
        "normalize_images": False,
        "net_arch": {"pi": [512, 512], "vf": [512, 512]},
        "log_std_init": float(np.log(float(ppo["initial_action_std"]))),
    }
    model = PPO(
        "MultiInputPolicy",
        vector_env,
        learning_rate=float(ppo["learning_rate"]),
        n_steps=int(ppo["rollout_steps"]),
        batch_size=int(ppo["batch_size"]),
        n_epochs=int(ppo["epochs"]),
        gamma=float(ppo["gamma"]),
        gae_lambda=float(ppo["gae_lambda"]),
        clip_range=float(ppo["clip_range"]),
        ent_coef=float(ppo["entropy_coefficient"]),
        target_kl=float(ppo["target_kl"]),
        max_grad_norm=float(ppo["max_gradient_norm"]),
        policy_kwargs=policy_kwargs,
        seed=int(config["seed"]),
        device=args.device,
        verbose=1,
        tensorboard_log=str(output_dir / "tensorboard"),
    )

    bc_mse = behavior_clone_actor(
        model,
        camera_images,
        robot_state,
        demo_actions,
        epochs=args.bc_epochs,
    )
    model.save(str(output_dir / "behavior_cloned_model"))
    warm_start = evaluate(
        model,
        stage=1,
        episodes=max(5, min(10, args.eval_episodes)),
        seed=5000,
    )
    (output_dir / "initialization.json").write_text(
        json.dumps(
            {
                "visual_backbone": args.visual_backbone,
                "lewm_checkpoint": str(checkpoint.resolve()) if args.visual_backbone == "lewm" else None,
                "demo_path": str(demo_path.resolve()),
                "demonstration_steps": len(demo_actions),
                "behavior_cloning_action_mse": bc_mse,
                "fixed_scene_warm_start": {
                    "successes": warm_start["successes"],
                    "episodes": warm_start["episodes"],
                    "success_rate": warm_start["success_rate"],
                    "phase_rates": warm_start["phase_rates"],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"BC action MSE={bc_mse:.6g}; fixed-scene warm start "
        f"{warm_start['successes']}/{warm_start['episodes']}"
    )

    callback = VisionEvaluationCallback(
        output_dir,
        stage=args.stage,
        eval_freq=args.eval_freq,
        episodes=args.eval_episodes,
        seed=10_000,
    )
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            reset_num_timesteps=False,
            progress_bar=False,
        )
    finally:
        model.save(str(output_dir / "final_model"))
        vector_env.close()
    summary = {
        "visual_backbone": args.visual_backbone,
        "stage": args.stage,
        "timesteps": int(model.num_timesteps),
        "best_evaluation_success_rate": callback.best_success_rate,
        "last_checkpoint": str(output_dir / "latest_model.zip"),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

