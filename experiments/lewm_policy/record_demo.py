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
    def __init__(self, env, candidate):
        super().__init__(env, candidate)
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


def record_demo(config_path: str | Path, output_path: str | Path, seed: int | None = None):
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
    env = BallCupTrainingEnv(
        stage=1,
        seed=effective_seed,
        config_path=config_path,
        render_mode="rgb_array",
    )
    env._renderer = mujoco.Renderer(env.model, height=IMAGE_SIZE, width=IMAGE_SIZE)
    try:
        expert = CameraRecordingExpert(env, candidate)
        observations, actions, metadata = expert.run(effective_seed)
        images = np.asarray(expert.images, dtype=np.uint8)
        proprio = np.asarray(expert.proprio, dtype=np.float32)
    finally:
        env.close()

    if not (len(images) == len(proprio) == len(actions) == len(observations)):
        raise RuntimeError("Recorded camera/state/action samples are not aligned")
    if not metadata.get("success") or not metadata["phase_success"].get("placement"):
        raise RuntimeError("Refusing to save an unsuccessful pick-and-place demonstration")

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
    args = parser.parse_args(argv)
    output, metadata = record_demo(args.config, args.output, args.seed)
    print(json.dumps({"demo_path": str(output), **metadata}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

