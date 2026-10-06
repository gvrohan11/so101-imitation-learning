"""Record camera, robot state, and actions from the verified scripted placement."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import mujoco
import numpy as np

from sim.expert_demo import _RetimedExpert
from sim.rl_env import BallCupTrainingEnv
from sim.search_grasp_poses import main as search_safe_pinch_candidates
from sim.training_config import DEFAULT_CONFIG_PATH, load_training_config

from .env import IMAGE_SIZE, proprioception_from_state


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class CameraRecordingExpert(_RetimedExpert):
    def __init__(self, env, candidate, *, execution_action_noise_std=0.0):
        super().__init__(
            env,
            candidate,
            execution_action_noise_std=execution_action_noise_std,
        )
        self.images: list[np.ndarray] = []
        self.proprio: list[np.ndarray] = []

    def _step(self, target_q, gripper_target):
        raw_state = self.env._observation().copy()
        frame = self.env.render()
        if frame is None or frame.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
            raise RuntimeError("Expected a 224x224 RGB camera frame")
        self.images.append(frame.astype(np.uint8, copy=True))
        self.proprio.append(proprioception_from_state(raw_state, self.env))
        return super()._step(target_q, gripper_target)


def record_demo(
    config_path: str | Path,
    output_path: str | Path,
    seed: int | None = None,
    *,
    episodes: int = 8,
    execution_action_noise_std: float = 0.1,
):
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if execution_action_noise_std < 0.0:
        raise ValueError("execution_action_noise_std must be non-negative")
    config = load_training_config(config_path)
    effective_seed = int(config["seed"] if seed is None else seed)
    with contextlib.redirect_stdout(io.StringIO()):
        candidates = search_safe_pinch_candidates()
    if not candidates:
        raise RuntimeError("The grasp-pose search found no valid scripted demonstration")
    candidate = candidates[0]

    expected_offset = np.asarray(config["task"]["grasp_site_offset_m"])
    if not np.allclose(candidate[3], expected_offset, atol=1e-9):
        raise RuntimeError("Demonstration grasp pose differs from the training configuration")
    episode_images = []
    episode_proprio = []
    episode_actions = []
    episode_metadata = []
    for episode_index in range(int(episodes)):
        episode_seed = effective_seed + episode_index
        env = BallCupTrainingEnv(
            stage=1,
            seed=episode_seed,
            config_path=config_path,
            render_mode="rgb_array",
        )
        env._renderer = mujoco.Renderer(env.model, height=IMAGE_SIZE, width=IMAGE_SIZE)
        try:
            expert = CameraRecordingExpert(
                env,
                candidate,
                execution_action_noise_std=execution_action_noise_std,
            )
            observations, actions, metadata = expert.run(episode_seed)
            images = np.asarray(expert.images, dtype=np.uint8)
            proprio = np.asarray(expert.proprio, dtype=np.float32)
        finally:
            env.close()

        if not (len(images) == len(proprio) == len(actions) == len(observations)):
            raise RuntimeError(
                f"Demonstration {episode_index + 1} camera/state/action samples are not aligned"
            )
        if not metadata.get("success") or not metadata["phase_success"].get("placement"):
            raise RuntimeError(
                f"Refusing to save unsuccessful demonstration {episode_index + 1}"
            )
        episode_images.append(images)
        episode_proprio.append(proprio)
        episode_actions.append(np.asarray(actions, dtype=np.float32))
        episode_metadata.append(metadata)

    images = np.concatenate(episode_images, axis=0)
    proprio = np.concatenate(episode_proprio, axis=0)
    actions = np.concatenate(episode_actions, axis=0)
    metadata = {
        "success": True,
        "episode_count": len(episode_metadata),
        "steps": int(len(actions)),
        "execution_action_noise_std": float(execution_action_noise_std),
        "episodes": episode_metadata,
        "phase_success": {
            phase: all(row["phase_success"][phase] for row in episode_metadata)
            for phase in episode_metadata[0]["phase_success"]
        },
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.update(
        {
            "camera": "front",
            "image_size": IMAGE_SIZE,
            "state_fields": ["normalized_joint_positions_6", "joint_velocities_6", "time_fraction"],
            "actions_are_normalized": True,
            "control_hz": 20,
        }
    )
    np.savez_compressed(
        output_path,
        camera_images=images,
        robot_state=proprio,
        actions=np.asarray(actions, dtype=np.float32),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return output_path, metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "lewm_policy" / "expert_demo.npz",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--execution-action-noise-std", type=float, default=0.1)
    args = parser.parse_args(argv)
    output, metadata = record_demo(
        args.config,
        args.output,
        args.seed,
        episodes=args.episodes,
        execution_action_noise_std=args.execution_action_noise_std,
    )
    print(json.dumps({"demo_path": str(output), **metadata}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
