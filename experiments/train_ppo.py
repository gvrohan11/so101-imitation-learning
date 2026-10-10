"""Train the SO-101 ball-in-cup policy with Stable-Baselines3 PPO."""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import pickle
import platform
import subprocess
import sys
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
from sim.rl_env import OBSERVATION_SIZE
from sim.training_config import DEFAULT_CONFIG_PATH, load_training_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _observation_index_map(source_size: int) -> dict[int, int]:
    """Map old state features into the current observation layout.

    The original 25-value observation contained the first 24 physical values
    followed by elapsed episode fraction. The 37-value layout kept those
    first values in place and added relative positions and task flags before
    elapsed fraction. The current 38-value layout appends the commanded
    gripper target.
    """
    if source_size == 25:
        return {**{index: index for index in range(24)}, 24: 36}
    if source_size == 37:
        return {index: index for index in range(37)}
    if source_size == OBSERVATION_SIZE:
        return {index: index for index in range(OBSERVATION_SIZE)}
    raise ValueError(
        f"Cannot resume observation size {source_size}; supported sizes are "
        f"25, 37, and {OBSERVATION_SIZE}."
    )


def _make_migrated_vec_normalize(source_path, env, source_size: int):
    """Carry old observation statistics into a VecNormalize wrapper."""
    stats_path = Path(source_path).with_name(
        f"{Path(source_path).stem}_vecnormalize.pkl"
    )
    if not stats_path.is_file():
        raise FileNotFoundError(
            f"Resume checkpoint is missing its observation-normalization file: "
            f"{stats_path}"
        )
    with stats_path.open("rb") as stats_file:
        source_vec_normalize = pickle.load(stats_file)
    source_rms = source_vec_normalize.obs_rms
    if source_rms.mean.shape != (source_size,):
        raise ValueError(
            f"Checkpoint observation statistics have shape {source_rms.mean.shape}, "
            f"but its policy expects ({source_size},)."
        )

    vec_env = VecNormalize(
        env,
        training=True,
        norm_obs=True,
        norm_reward=False,
        clip_obs=float(source_vec_normalize.clip_obs),
        epsilon=float(source_vec_normalize.epsilon),
    )
    target_rms = vec_env.obs_rms
    target_rms.mean.fill(0.0)
    target_rms.var.fill(1.0)
    target_rms.var[24:30] = 0.01
    target_rms.mean[37] = 1.0
    target_rms.var[37] = 0.25
    for source_index, target_index in _observation_index_map(source_size).items():
        target_rms.mean[target_index] = source_rms.mean[source_index]
        target_rms.var[target_index] = source_rms.var[source_index]
    # Retain the source count so mapped features keep their learned
    # normalization. Relative positions start at a 10 cm scale and the jaw
    # target around its range midpoint. The new policy input weights are zero.
    target_rms.count = float(source_rms.count)
    return vec_env


def _make_model(
    config,
    env,
    *,
    seed: int,
    verbose: int = 1,
    use_sde: bool | None = None,
    sde_sample_freq: int | None = None,
):
    ppo = config["ppo"]
    if use_sde is None:
        use_sde = bool(ppo["use_state_dependent_exploration"])
    if sde_sample_freq is None:
        sde_sample_freq = int(ppo["state_dependent_exploration_sample_freq"])
    return PPO(
        "MlpPolicy",
        env,
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
        use_sde=bool(use_sde),
        sde_sample_freq=int(sde_sample_freq),
        policy_kwargs=_make_policy_kwargs(ppo),
        seed=seed,
        verbose=verbose,
        device="cpu",
    )


def _migrate_policy(source_model: PPO, target_model: PPO, source_size: int) -> None:
    """Copy a PPO policy into the current observation space, zero-extending inputs."""
    source_state = source_model.policy.state_dict()
    target_state = target_model.policy.state_dict()
    index_map = _observation_index_map(source_size)
    for name, target_tensor in target_state.items():
        source_tensor = source_state.get(name)
        if source_tensor is None:
            raise ValueError(f"Checkpoint policy is missing parameter {name!r}.")
        if source_tensor.shape == target_tensor.shape:
            target_state[name] = source_tensor.detach().clone()
            continue
        if (
            source_tensor.ndim == 2
            and target_tensor.ndim == 2
            and source_tensor.shape[0] == target_tensor.shape[0]
            and source_tensor.shape[1] == source_size
            and target_tensor.shape[1] == OBSERVATION_SIZE
            and name.endswith("weight")
        ):
            expanded = target_tensor.new_zeros(target_tensor.shape)
            for source_index, target_index in index_map.items():
                expanded[:, target_index] = source_tensor[:, source_index]
            target_state[name] = expanded
            continue
        raise ValueError(
            f"Cannot migrate policy parameter {name!r}: source shape "
            f"{tuple(source_tensor.shape)}, target shape {tuple(target_tensor.shape)}."
        )
    target_model.policy.load_state_dict(target_state)
    target_model.num_timesteps = int(source_model.num_timesteps)
    target_model._n_updates = int(source_model._n_updates)


def _runtime_metadata():
    packages = {}
    for distribution in (
        "mujoco",
        "stable-baselines3",
        "gymnasium",
        "numpy",
        "torch",
    ):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    try:
        git_revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        git_dirty = bool(subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        git_revision = None
        git_dirty = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
        "git_revision": git_revision,
        "git_dirty": git_dirty,
    }


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
            use_sde=bool(ppo["use_state_dependent_exploration"]),
            sde_sample_freq=int(
                ppo["state_dependent_exploration_sample_freq"]
            ),
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


def run_training(config, config_path, output_dir, resume_from=None):
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

    resume_normalization = None
    source_model = None
    source_observation_size = None
    policy_migration = False
    if resume_from is not None:
        resume_from = Path(resume_from).expanduser().resolve()
        if not resume_from.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_from}")
        source_model = PPO.load(str(resume_from), device="cpu")
        source_observation_size = int(source_model.observation_space.shape[0])
        _observation_index_map(source_observation_size)
        policy_migration = source_observation_size != OBSERVATION_SIZE
        resume_normalization = resume_from.with_name(
            f"{resume_from.stem}_vecnormalize.pkl"
        )
        if normalize and not resume_normalization.is_file():
            raise FileNotFoundError(
                "Resume checkpoint is missing its observation-normalization "
                f"file: {resume_normalization}"
            )
        if not normalize and resume_normalization.is_file():
            raise ValueError(
                "This checkpoint was saved with observation normalization, but "
                "the selected config disables it. Enable ppo.normalize_observations "
                "to preserve the checkpoint's policy inputs."
            )

    print(f"Run artifacts: {output_dir}")
    if resume_from is None:
        print("Starting PPO from a random policy; demonstrations are not used for training.")
    elif policy_migration:
        print(
            f"Migrating learned PPO weights from {source_observation_size} "
            f"observations to {OBSERVATION_SIZE}; no scripted demonstration is used."
        )
    else:
        print(f"Resuming PPO weights and observation statistics from {resume_from}")
    if resume_from is None:
        training_mode = "ppo_from_scratch"
    elif policy_migration:
        training_mode = "ppo_policy_migration_finetune"
    else:
        training_mode = "ppo_resumed_finetune"
    run_metadata = {
        "training_mode": training_mode,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "stage_order": [1, 2, 3, 4, 5],
        "observation_size": OBSERVATION_SIZE,
        "source_observation_size": source_observation_size,
        "policy_migration": policy_migration,
        "use_state_dependent_exploration": (
            bool(source_model.use_sde)
            if source_model is not None
            else bool(ppo["use_state_dependent_exploration"])
        ),
        "policy_layers": list(ppo["policy_layers"]),
        "initial_action_std": float(ppo["initial_action_std"]),
        "learning_rate": float(ppo["learning_rate"]),
        "scripted_demonstration_used": False,
        "pretrained_policy_used": resume_from is not None,
        "demonstration_action_anchor_used": False,
        "open_loop_clock_lookup_used": False,
        "resumed_from": str(resume_from) if resume_from is not None else None,
        "runtime": _runtime_metadata(),
        "config": config,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2), encoding="utf-8"
    )

    first_env = make_vec_env(
        stage=1,
        seed=seed,
        output_dir=output_dir,
        normalize=False if resume_from is not None else normalize,
        config_path=config_path,
    )
    if resume_from is not None and normalize:
        if policy_migration:
            first_env = _make_migrated_vec_normalize(
                resume_from, first_env, source_observation_size
            )
        else:
            first_env = VecNormalize.load(str(resume_normalization), first_env)
        first_env.training = True
        first_env.norm_reward = False
    if resume_from is None:
        model = _make_model(config, first_env, seed=seed)
    elif policy_migration:
        model = _make_model(
            config,
            first_env,
            seed=seed,
            use_sde=bool(source_model.use_sde),
            sde_sample_freq=int(source_model.sde_sample_freq),
        )
        _migrate_policy(source_model, model, source_observation_size)
    else:
        model = source_model
        model.set_env(first_env, force_reset=True)
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
                curriculum["maximum_consecutive_zero_grasp_evaluations"]
                if stage > 1 or resume_from is not None
                else 0
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
            reset_num_timesteps=(stage == 1 and resume_from is None),
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
                "training_mode": training_mode,
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
        "--resume-from", type=Path, default=None,
        help="Resume PPO weights and VecNormalize stats from a saved checkpoint",
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
    return run_training(config, config_path, output_dir, resume_from=args.resume_from)


if __name__ == "__main__":
    raise SystemExit(main())
