import numpy as np

from sim.ball_cup_env import BallCupEnv

env = BallCupEnv(seed=7)
try:
    obs, info = env.reset(seed=7)
    print("image:", obs["image"].shape, obs["image"].dtype)
    print("joints:", obs["joint_positions"].shape)
    print("reset info:", info)

    obs, reward, terminated, truncated, info = env.step(
        np.zeros(6, dtype=np.float32)
    )
    print("step image:", obs["image"].shape)
    print("reward/terminated/truncated:", reward, terminated, truncated)
    print("step info:", info)
finally:
    env.close()