"""Train the SO-101 ball-in-cup policy with Stable-Baselines3 PPO."""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize
from torch import nn

from experiments.ppo_common import (
    make_vec_env,
    run_deterministic_evaluation,
    save_bundle,
)
from experiments.ppo_curriculum import CurriculumStageCallback
from sim.training_config import DEFAULT_CONFIG_PATH, load_training_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _make_policy_kwargs(config, *, smoke=False):
    layers = [32, 32] if smoke else list(config["policy_layers"])
    if not layers or any(int(width) <= 0 for width in layers):
        raise ValueError("policy_layers must contain positive hidden widths")
    return {
        # Both networks learn directly from physical state and PPO rewards.
        "net_arch": {"pi": layers, "vf": layers},
        "activation_fn": nn.Tanh,
        "log_std_init": float(np.log(config["initial_action_std"])),
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
            target_kl=float(ppo["target_kl"]),
            ent_coef=float(ppo["entropy_coefficient"]),
            vf_coef=float(ppo["value_coefficient"]),
            max_grad_norm=float(ppo["max_gradient_norm"]),
            policy_kwargs=_make_policy_kwargs(ppo, smoke=True),
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


def run_training(config, config_path, output_dir):
    curriculum = config["curriculum"]
    ppo = config["ppo"]
    seed = int(config["seed"])
    normalize = bool(ppo["normalize_observations"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to mix a fresh PPO run with existing output files: {output_dir}. "
            "Choose a new --output-dir or move the old run first."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run artifacts: {output_dir}")
    print("Starting PPO from a random policy; demonstrations are not used for training.")
    run_metadata = {
        "training_mode": "ppo_from_scratch",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "stage_order": [1, 2, 3, 4, 5],
        "observation_size": 25,
        "policy_layers": list(ppo["policy_layers"]),
        "initial_action_std": float(ppo["initial_action_std"]),
        "learning_rate": float(ppo["learning_rate"]),
        "scripted_demonstration_used": False,
        "pretrained_policy_used": False,
        "demonstration_action_anchor_used": False,
        "open_loop_clock_lookup_used": False,
        "config": config,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2), encoding="utf-8"
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
        target_kl=float(ppo["target_kl"]),
        ent_coef=float(ppo["entropy_coefficient"]),
        vf_coef=float(ppo["value_coefficient"]),
        max_grad_norm=float(ppo["max_gradient_norm"]),
        policy_kwargs=_make_policy_kwargs(ppo),
        seed=seed,
        verbose=1,
        device="cpu",
    )
    model.set_logger(configure(str(output_dir / "sb3_logs"), ["stdout", "csv"]))
    save_bundle(model, output_dir, "initial_model")
    global_best = {"success_rate": 0.0, "stage": 1, "global_env_steps": 0}
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
            maximum_consecutive_zero_grasp_evaluations=int(
                0
                if stage == 1
                else curriculum["maximum_consecutive_zero_grasp_evaluations"]
            ),
            config_path=config_path,
        )
        stage_start = int(model.num_timesteps)
        print(
            f"\nStarting curriculum stage {stage}/5; "
            "training the state-feedback PPO actor from rewards only; "
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
                "training_mode": "ppo_from_scratch",
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
    args = parser.parse_args(argv)
    config = load_training_config(args.config)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if args.smoke_test:
        run_smoke_test(config, config_path)
        return 0
    output_dir = args.output_dir
    if output_dir is None:
        run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S_%f_utc")
        output_dir = PROJECT_ROOT / config["logging"]["output_directory"] / run_id
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    return run_training(config, config_path, output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
