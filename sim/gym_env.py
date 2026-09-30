import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sim.ball_cup_env import BallCupEnv


class StateOnlyBallCupEnv(gym.Env):
    """Gymnasium interface exposing only simulator state observations."""

    metadata = {"render_modes": []}

    def __init__(self, frame_skip=20, horizon=300, seed=None):
        super().__init__()
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32
        )
        self.env = BallCupEnv(
            frame_skip=frame_skip,
            horizon=horizon,
            seed=seed,
            render_images=False,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        observation, info = self.env.reset(seed=seed)
        return observation["state"], info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation["state"], reward, terminated, truncated, info

    def close(self):
        self.env.close()