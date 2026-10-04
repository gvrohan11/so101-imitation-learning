"""Train the SO-101 ball-in-cup policy with Stable-Baselines3 PPO."""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize
from torch import nn
import torch

from experiments.ppo_common import (
    make_vec_env,
    run_deterministic_evaluation,
    save_bundle,
)
from experiments.ppo_curriculum import CurriculumStageCallback
from sim.expert_demo import generate_demonstration, save_demonstration
from sim.rl_env import CLOCK_OBSERVATION_START
from sim.training_config import DEFAULT_CONFIG_PATH, load_training_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _make_policy_kwargs(config, *, smoke=False, horizon=None):
    layers = [32, 32] if smoke else list(config["policy_layers"])
    if smoke:
        actor_layers = layers
        critic_layers = layers
    else:
        if not layers:
            raise ValueError("policy_layers must contain a positive hidden width")
        clock_width = int(horizon or config.get("episode_steps", 0))
        actor_layers = [max(int(layers[0]), clock_width)]
        critic_layers = layers
    return {
        # The actor's first hidden layer has one unit per policy timestep. The
        # demonstration initializer maps the one-hot clock to its exact action
        # with this layer; PPO can then learn state feedback in those same units.
        "net_arch": {"pi": actor_layers, "vf": critic_layers},
        "activation_fn": nn.Tanh,
        "log_std_init": float(np.log(config["initial_action_std"])),
    }


def initialize_actor_from_demonstration(model, observations, actions, config):
    """Initialize a one-hidden-layer PPO actor to replay a retimed demonstration."""
    policy = model.policy
    vec_normalize = model.get_vec_normalize_env()
    observations = np.asarray(observations, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    if vec_normalize is not None:
        vec_normalize.obs_rms.update(np.asarray(observations, dtype=np.float64))
        normalized_observations = vec_normalize.normalize_obs(observations)
        mean = np.asarray(vec_normalize.obs_rms.mean, dtype=np.float64)
        std = np.sqrt(
            np.asarray(vec_normalize.obs_rms.var, dtype=np.float64)
            + float(vec_normalize.epsilon)
        )
        clip = float(vec_normalize.clip_obs)
    else:
        normalized_observations = observations.copy()
        mean = np.zeros(observations.shape[1], dtype=np.float64)
        std = np.ones(observations.shape[1], dtype=np.float64)
        clip = float("inf")

    # rl_env observation layout: 22 physical state values, one scalar time
    # fraction, then one-hot timestep. The cloned action schedule depends only
    # on this clock; PPO is free to learn state feedback after initialization.
    clock_start = CLOCK_OBSERVATION_START
    horizon = observations.shape[1] - clock_start
    if horizon <= 0 or len(actions) > horizon:
        raise ValueError(
            f"Expected clock features after observation index {clock_start}; "
            f"got observation shape {observations.shape} and {len(actions)} actions"
        )
    hidden = policy.mlp_extractor.policy_net
    if len(hidden) != 2 or not isinstance(hidden[0], nn.Linear):
        raise ValueError("Demonstration actor requires one Linear+Tanh hidden layer")
    clock_width = hidden[0].out_features
    if clock_width < horizon:
        raise ValueError(
            f"Actor width {clock_width} is too small for the {horizon}-step clock"
        )
    if not isinstance(policy.action_net, nn.Linear):
        raise ValueError("Demonstration actor requires a linear PPO action head")

    raw_inactive = np.zeros(horizon, dtype=np.float64)
    raw_active = np.eye(horizon, dtype=np.float64)
    clock_mean = mean[clock_start:clock_start + horizon]
    clock_std = std[clock_start:clock_start + horizon]
    inactive = np.clip((raw_inactive - clock_mean) / clock_std, -clip, clip)
    active = np.clip((raw_active - clock_mean) / clock_std, -clip, clip)
    activation_delta = active.diagonal() - inactive
    if np.any(np.abs(activation_delta) < 1e-6):
        raise ValueError("Normalized one-hot clock features are not distinguishable")

    # A tanh unit marks each clock index: inactive clock bits sit below its
    # midpoint and the active bit sits above it. The linear action head then
    # maps each clock index to its recorded six-dimensional action exactly.
    first = hidden[0]
    action_head = policy.action_net
    target_actions = np.zeros((horizon, actions.shape[1]), dtype=np.float32)
    target_actions[:len(actions)] = actions
    with torch.no_grad():
        first.weight.zero_()
        first.bias.zero_()
        indices = torch.arange(horizon, device=model.device)
        clock_columns = torch.arange(
            clock_start, clock_start + horizon, device=model.device
        )
        first.weight[indices, clock_columns] = 1.0
        midpoint = 0.5 * (inactive + active.diagonal())
        first.bias[indices] = torch.as_tensor(
            -midpoint, dtype=first.bias.dtype, device=model.device
        )

        inactive_hidden = np.tanh(inactive - midpoint)
        active_hidden = np.tanh(active.diagonal() - midpoint)
        hidden_delta = active_hidden - inactive_hidden
        lookup_weights = target_actions.T / hidden_delta[None, :]
        action_head.weight.zero_()
        action_head.bias.zero_()
        action_head.weight[:, :horizon] = torch.as_tensor(
            lookup_weights, dtype=action_head.weight.dtype, device=model.device
        )
        action_head.bias.copy_(-action_head.weight[:, :horizon] @ torch.as_tensor(
            inactive_hidden, dtype=action_head.weight.dtype, device=model.device
        ))

    policy.set_training_mode(False)
    with torch.no_grad():
        expert_tensor = torch.as_tensor(
            normalized_observations, dtype=torch.float32, device=model.device
        )
        predicted = policy.get_distribution(expert_tensor).distribution.mean
        final_mse = float(
            torch.mean((predicted - torch.as_tensor(
                actions, dtype=torch.float32, device=model.device
            )) ** 2).cpu()
        )
    return {
        "method": "one_hidden_layer_clock_lookup",
        "demonstration_action_mse": final_mse,
        "state_feedback_initialized_from_demo": False,
        "initial_action_std": float(config["initial_action_std"]),
        "clock_steps_encoded": int(horizon),
        "demonstration_steps": int(len(actions)),
    }


def run_smoke_test(config, config_path):
    """Run two tiny PPO updates and a deterministic evaluation; no long run."""
    with tempfile.TemporaryDirectory(prefix="so101-ppo-smoke-") as tmp:
        output_dir = Path(tmp)
        vec_env = make_vec_env(
            stage=1,
            seed=int(config["seed"]),
            output_dir=output_dir,
            horizon=50,
            normalize=bool(config["ppo"]["normalize_observations"]),
            config_path=config_path,
        )
        ppo = config["ppo"]
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=float(ppo["learning_rate"]),
            n_steps=64,
            batch_size=32,
            n_epochs=1,
            gamma=float(ppo["gamma"]),
            gae_lambda=float(ppo["gae_lambda"]),
            clip_range=float(ppo["clip_range"]),
            ent_coef=float(ppo["entropy_coefficient"]),
            vf_coef=float(ppo["value_coefficient"]),
            max_grad_norm=float(ppo["max_gradient_norm"]),
            policy_kwargs=_make_policy_kwargs(ppo, smoke=True, horizon=50),
            seed=int(config["seed"]),
            verbose=0,
            device="cpu",
        )
        model.set_logger(configure(str(output_dir / "sb3_logs"), ["csv"]))
        model.learn(total_timesteps=128, progress_bar=False)
        save_bundle(model, output_dir, "smoke_model")
        eval_base_env = make_vec_env(
            stage=1,
            seed=int(config["seed"]) + 50_000,
            output_dir=output_dir / "eval",
            horizon=50,
            normalize=False,
            config_path=config_path,
        )
        eval_vec_env = VecNormalize.load(
            str(output_dir / "smoke_model_vecnormalize.pkl"), eval_base_env
        )
        eval_vec_env.training = False
        eval_vec_env.norm_reward = False
        loaded = PPO.load(
            str(output_dir / "smoke_model.zip"), env=eval_vec_env, device="cpu"
        )
        try:
            result = run_deterministic_evaluation(
                loaded,
                loaded.get_vec_normalize_env(),
                stage=1,
                episodes=2,
                seed=int(config["seed"]) + 50_000,
                horizon=50,
                config_path=config_path,
            )
        finally:
            eval_vec_env.close()
            vec_env.close()
        if int(model.num_timesteps) < 128 or len(result["episode_rows"]) != 2:
            raise RuntimeError("PPO smoke test did not complete its train/evaluate path")
        print(json.dumps({
            "smoke_test": "passed",
            "training_steps": int(model.num_timesteps),
            "deterministic_evaluation_episodes": len(result["episode_rows"]),
            "successes": result["successes"],
            "success_rate": result["success_rate"],
            "policy_checkpoint_reload": "passed",
            "observation_normalization_stats_reload": "passed",
        }, indent=2))


def run_training(config, config_path, output_dir, *, initialize_only=False):
    curriculum = config["curriculum"]
    ppo = config["ppo"]
    seed = int(config["seed"])
    normalize = bool(ppo["normalize_observations"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating a full pick-and-place demonstration at the PPO control rate...")
    demo_observations, demo_actions, demo_metadata = generate_demonstration(
        config_path, seed=seed
    )
    if not demo_metadata["success"] or not all(
        demo_metadata["phase_success"].values()
    ):
        raise RuntimeError("Refusing to train from an unsuccessful expert demonstration")
    demo_path = save_demonstration(
        output_dir / "demonstration_stage01.npz",
        demo_observations,
        demo_actions,
        demo_metadata,
    )
    print(
        f"Saved successful {demo_metadata['steps']}-step, "
        f"{demo_metadata['control_hz']} Hz demonstration to {demo_path}; "
        "all arm commands respect the configured 2-degree limit."
    )

    first_env = make_vec_env(
        stage=1,
        seed=seed,
        output_dir=output_dir,
        normalize=normalize,
        config_path=config_path,
    )
    model = PPO(
        "MlpPolicy",
        first_env,
        learning_rate=float(ppo["learning_rate"]),
        n_steps=int(ppo["rollout_steps"]),
        batch_size=int(ppo["batch_size"]),
        n_epochs=int(ppo["epochs"]),
        gamma=float(ppo["gamma"]),
        gae_lambda=float(ppo["gae_lambda"]),
        clip_range=float(ppo["clip_range"]),
        ent_coef=float(ppo["entropy_coefficient"]),
        vf_coef=float(ppo["value_coefficient"]),
        max_grad_norm=float(ppo["max_gradient_norm"]),
        policy_kwargs=_make_policy_kwargs(
            ppo, horizon=int(config["task"]["episode_steps"])
        ),
        seed=seed,
        verbose=1,
        device="auto",
    )
    model.set_logger(configure(str(output_dir / "sb3_logs"), ["stdout", "csv"]))

    initialization_metrics = initialize_actor_from_demonstration(
        model, demo_observations, demo_actions, ppo
    )
    save_bundle(model, output_dir, "demonstration_initialized")
    warm_start_evaluation = run_deterministic_evaluation(
        model,
        model.get_vec_normalize_env(),
        stage=1,
        episodes=int(ppo["warm_start_evaluation_episodes"]),
        seed=seed + 5_000_000,
        config_path=config_path,
    )
    warm_start_summary = {
        "demonstration_path": str(demo_path),
        "demonstration": demo_metadata,
        "actor_initialization": initialization_metrics,
        "deterministic_stage_1_evaluation": warm_start_evaluation,
    }
    (output_dir / "demonstration_warm_start.json").write_text(
        json.dumps(warm_start_summary, indent=2), encoding="utf-8"
    )
    print(
        "Demonstration-initialized actor: "
        f"demonstration action MSE "
        f"{initialization_metrics['demonstration_action_mse']:.2e}; "
        f"stage-1 success {warm_start_evaluation['successes']}/"
        f"{warm_start_evaluation['episodes']} "
        f"({warm_start_evaluation['success_rate']:.0%})."
    )
    minimum_warm_start_rate = float(ppo["minimum_warm_start_success_rate"])
    if warm_start_evaluation["success_rate"] < minimum_warm_start_rate:
        first_env.close()
        raise RuntimeError(
            "Demonstration-initialized policy failed its pre-training gate: "
            f"{warm_start_evaluation['success_rate']:.0%} success is below "
            f"the configured {minimum_warm_start_rate:.0%} minimum. "
            "PPO was not started. Inspect demonstration_warm_start.json."
        )
    if initialize_only:
        first_env.close()
        print(
            "Initialization-only run passed. The demonstration and initialized policy, "
            "normalization statistics, and warm-start evaluation are saved; "
            "the PPO curriculum was not started."
        )
        return 0

    global_best = {"success_rate": -1.0, "stage": 0}
    stage_results = []
    for stage in range(1, 6):
        if stage > 1:
            previous_env = model.get_env()
            previous_vec_normalize = model.get_vec_normalize_env()
            previous_obs_rms = (
                copy.deepcopy(previous_vec_normalize.obs_rms)
                if previous_vec_normalize is not None
                else None
            )
            stage_env = make_vec_env(
                stage=stage,
                seed=seed + stage * 100_000,
                output_dir=output_dir,
                normalize=normalize,
                config_path=config_path,
            )
            if previous_obs_rms is not None:
                stage_env.obs_rms = previous_obs_rms
            model.set_env(stage_env, force_reset=True)
            previous_env.close()

        callback = CurriculumStageCallback(
            stage=stage,
            output_dir=output_dir,
            seed=seed,
            minimum_steps=int(curriculum["minimum_steps_per_stage"]),
            evaluation_interval=int(curriculum["evaluation_interval_steps"]),
            evaluation_episodes=int(curriculum["evaluation_episodes"]),
            required_success_rate=float(curriculum["required_success_rate"]),
            required_consecutive=int(curriculum["consecutive_passing_evaluations"]),
            maximum_steps=int(curriculum["maximum_steps_per_stage"]),
            checkpoint_interval=int(config["logging"]["checkpoint_interval_steps"]),
            global_best=global_best,
            config_path=config_path,
        )
        stage_start = int(model.num_timesteps)
        print(
            f"\nStarting curriculum stage {stage}/5; "
            f"minimum={curriculum['minimum_steps_per_stage']:,} steps, "
            f"evaluation={curriculum['evaluation_episodes']} episodes every "
            f"{curriculum['evaluation_interval_steps']:,} steps, "
            f"maximum={curriculum['maximum_steps_per_stage']:,} steps."
        )
        model.learn(
            total_timesteps=int(curriculum["maximum_steps_per_stage"]),
            callback=callback,
            reset_num_timesteps=(stage == 1),
            progress_bar=False,
            tb_log_name=f"stage_{stage:02d}",
        )
        stage_steps = int(model.num_timesteps - stage_start)
        save_bundle(model, output_dir, "latest_model")
        stage_results.append({
            "stage": stage,
            "outcome": callback.outcome,
            "steps": stage_steps,
            "consecutive_passing_evaluations": callback.consecutive_passes,
            "best_evaluation_success_rate": callback.best_stage_success_rate,
        })
        (output_dir / "training_summary.json").write_text(
            json.dumps({
                "stages": stage_results,
                "global_best": global_best,
                "final_global_steps": int(model.num_timesteps),
            }, indent=2),
            encoding="utf-8",
        )
        print(
            f"Stage {stage} ended: {callback.outcome} after {stage_steps:,} "
            "training environment steps."
        )
        if callback.outcome != "passed":
            model.get_env().close()
            return 2

    model.get_env().close()
    save_bundle(model, output_dir, "final_model")
    print("All five curriculum stages passed their independent evaluation criteria.")
    print(f"Latest checkpoint: {output_dir / 'latest_model.zip'}")
    print(f"Best checkpoint:   {output_dir / 'best_model.zip'}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH,
        help="JSON training configuration (default: config/ppo_training.json)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Override the configured output directory",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run 128 PPO steps plus a two-episode check, never the curriculum",
    )
    parser.add_argument(
        "--initialize-only", action="store_true",
        help="Generate and clone the expert demo, evaluate it, and stop before PPO",
    )
    args = parser.parse_args(argv)
    config = load_training_config(args.config)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if args.smoke_test:
        run_smoke_test(config, config_path)
        return 0
    output_dir = args.output_dir or (PROJECT_ROOT / config["logging"]["output_directory"])
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    return run_training(
        config, config_path, output_dir, initialize_only=args.initialize_only
    )


if __name__ == "__main__":
    raise SystemExit(main())
