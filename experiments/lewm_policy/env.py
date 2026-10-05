"""Image and proprioception wrapper for the SO-101 ball-in-cup task."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from gymnasium import Env, spaces

# MuJoCo's EGL backend is the usual headless renderer on Linux/RunPod. Respect
# an explicit user choice such as MUJOCO_GL=osmesa.
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco

from sim.rl_env import BallCupTrainingEnv


IMAGE_SIZE = 224
RAW_STATE_TIME_INDEX = 24


def proprioception_from_state(raw_observation: np.ndarray, base_env) -> np.ndarray:
    """Return only robot-measurable joint state and episode elapsed fraction."""
    raw = np.asarray(raw_observation, dtype=np.float32)
    qpos = raw[:6]
    qvel = raw[6:12]

    arm_limits = np.maximum(np.max(np.abs(base_env.joint_ranges), axis=1), 1e-3)
    gripper_range = base_env.model.actuator_ctrlrange[base_env.sim.actuator_ids[-1]]
    gripper_limit = max(float(np.max(np.abs(gripper_range))), 1e-3)
    qpos_scale = np.concatenate((arm_limits, [gripper_limit])).astype(np.float32)

    # Joint velocities are divided by a broad rad/s scale and clipped so that
    # occasional reset/contact transients cannot dominate the policy input.
    normalized = np.concatenate(
        (qpos / qpos_scale, np.clip(qvel / 10.0, -5.0, 5.0), [raw[RAW_STATE_TIME_INDEX]])
    )
    return normalized.astype(np.float32)


class BallCupVisionEnv(Env):
    """Expose a front RGB frame and robot proprioception; hide object coordinates."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        *,
        stage: int = 1,
        seed: int | None = None,
        config_path: str | Path | None = None,
    ):
        super().__init__()
        self.base = BallCupTrainingEnv(
            stage=stage,
            seed=seed,
            config_path=config_path,
            render_mode="rgb_array",
        )
        self.action_space = self.base.action_space
        self.observation_space = spaces.Dict(
            {
                # CHW is explicit so Stable-Baselines3 will not guess a layout.
                "image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(3, IMAGE_SIZE, IMAGE_SIZE),
                    dtype=np.uint8,
                ),
                "proprio": spaces.Box(
                    low=-5.0,
                    high=5.0,
                    shape=(13,),
                    dtype=np.float32,
                ),
            }
        )
        if self.base._renderer is not None:
            self.base._renderer.close()
        self.base._renderer = mujoco.Renderer(
            self.base.model, height=IMAGE_SIZE, width=IMAGE_SIZE
        )

    def _observe(self, raw_observation: np.ndarray) -> dict[str, np.ndarray]:
        rgb = self.base.render()
        if rgb is None or rgb.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
            raise RuntimeError("MuJoCo did not return a 224x224 RGB camera frame")
        return {
            "image": np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.uint8),
            "proprio": proprioception_from_state(raw_observation, self.base),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        raw_observation, info = self.base.reset(seed=seed, options=options)
        return self._observe(raw_observation), info

    def step(self, action):
        raw_observation, reward, terminated, truncated, info = self.base.step(action)
        return (
            self._observe(raw_observation),
            reward,
            terminated,
            truncated,
            info,
        )

    def render(self):
        return self.base.render()

    def close(self):
        self.base.close()

