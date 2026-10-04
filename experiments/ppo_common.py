"""Shared PPO environment, evaluation, and checkpoint utilities."""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sim.rl_env import BallCupTrainingEnv


EPISODE_INFO_KEYS = (
    "is_success",
    "curriculum_stage",
    "reach_success",
    "grasp_success",
    "lift_success",
    "transport_success",
    "above_cup_success",
    "correct_release",
    "placement_success",
)


def make_vec_env(
    stage, seed, output_dir, *, horizon=None, normalize=True, config_path=None
):
    """Create one reproducible monitored environment for the current stage."""
    output_dir = Path(output_dir)
    monitor_dir = output_dir / "episodes" / f"stage_{stage:02d}"
    monitor_dir.mkdir(parents=True, exist_ok=True)

    def construct():
        env = BallCupTrainingEnv(
            stage=stage,
            horizon=horizon,
            seed=seed,
            config_path=config_path,
        )
        return Monitor(
            env,
            filename=str(monitor_dir / "rollout"),
            info_keywords=EPISODE_INFO_KEYS,
        )

    vec_env = DummyVecEnv([construct])
    if normalize:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
        )
    vec_env.seed(seed)
    return vec_env


def save_bundle(model: PPO, output_dir: Path, name: str) -> None:
    """Save policy and observation normalization statistics together."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(output_dir / name))
    vec_normalize = model.get_vec_normalize_env()
    if vec_normalize is not None:
        vec_normalize.save(str(output_dir / f"{name}_vecnormalize.pkl"))


def run_deterministic_evaluation(
    model: PPO,
    vec_normalize: VecNormalize | None,
    *,
    stage: int,
    episodes: int,
    seed: int,
    video_path: str | Path | None = None,
    video_stride: int = 2,
    horizon: int | None = None,
    config_path=None,
) -> dict:
    """Run frozen deterministic episodes over reproducible scene variations."""
    render_mode = "rgb_array" if video_path is not None else None
    env = BallCupTrainingEnv(
        stage=stage,
        horizon=horizon,
        seed=seed,
        config_path=config_path,
        render_mode=render_mode,
    )
    writer = None
    if video_path is not None:
        video_path = Path(video_path)
        if video_path.suffix.lower() != ".gif":
            raise ValueError("Video output currently supports .gif files")
        video_path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(video_path, mode="I", duration=0.1, loop=0)

    episode_rows = []
    try:
        for episode_index in range(int(episodes)):
            observation, _ = env.reset(seed=int(seed) + episode_index)
            episode_return = 0.0
            episode_length = 0
            done = False
            if writer is not None and episode_index == 0:
                writer.append_data(env.render())
            while not done:
                policy_observation = observation
                if vec_normalize is not None:
                    policy_observation = vec_normalize.normalize_obs(
                        np.asarray(observation, dtype=np.float32)[None, :]
                    )
                action, _ = model.predict(policy_observation, deterministic=True)
                action = np.asarray(action).reshape(6)
                observation, reward, terminated, truncated, info = env.step(action)
                episode_return += float(reward)
                episode_length += 1
                done = bool(terminated or truncated)
                if (
                    writer is not None
                    and episode_index == 0
                    and (episode_length % max(1, video_stride) == 0 or done)
                ):
                    writer.append_data(env.render())
            final_metrics = info.get("episode_metrics", {})
            episode_rows.append({
                "episode": episode_index + 1,
                "seed": int(seed) + episode_index,
                "stage": int(stage),
                "return": float(episode_return),
                "length": int(episode_length),
                "is_success": bool(final_metrics.get("is_success", info["is_success"])),
                "reach_success": bool(final_metrics.get("reach_success", info["reach_success"])),
                "grasp_success": bool(final_metrics.get("grasp_success", info["grasp_success"])),
                "lift_success": bool(final_metrics.get("lift_success", info["lift_success"])),
                "transport_success": bool(final_metrics.get("transport_success", info["transport_success"])),
                "above_cup_success": bool(final_metrics.get("above_cup_success", info["above_cup_success"])),
                "correct_release": bool(final_metrics.get("correct_release", info["correct_release"])),
                "placement_success": bool(final_metrics.get("placement_success", info["placement_success"])),
                "ball_radius_m": float(info["ball_radius_m"]),
                "cup_scale": float(info["cup_scale"]),
            })
    finally:
        if writer is not None:
            writer.close()
        env.close()

    return {
        "stage": int(stage),
        "seed": int(seed),
        "episodes": int(episodes),
        "successes": sum(row["is_success"] for row in episode_rows),
        "success_rate": float(np.mean([row["is_success"] for row in episode_rows])),
        "mean_return": float(np.mean([row["return"] for row in episode_rows])),
        "phase_rates": {
            key: float(np.mean([row[key] for row in episode_rows]))
            for key in (
                "reach_success", "grasp_success", "lift_success",
                "transport_success", "above_cup_success", "correct_release",
                "placement_success",
            )
        },
        "episode_rows": episode_rows,
        "video_path": str(video_path) if video_path is not None else None,
    }


def write_json_line(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(row, sort_keys=True) + "\n")
