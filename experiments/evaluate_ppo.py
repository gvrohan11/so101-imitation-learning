"""Evaluate a saved PPO policy, optionally saving one rendered GIF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from experiments.ppo_common import run_deterministic_evaluation
from sim.rl_env import BallCupTrainingEnv
from sim.training_config import DEFAULT_CONFIG_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path,
        default=PROJECT_ROOT / "outputs/ball_cup_ppo/final_model.zip",
    )
    parser.add_argument("--stage", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=70_000)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--video", type=Path, default=None,
        help="Save the first evaluation episode as a GIF",
    )
    args = parser.parse_args(argv)
    model_path = args.model.expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"PPO checkpoint not found: {model_path}")
    model = PPO.load(str(model_path), device="auto")

    vecnormalize_path = model_path.with_name(
        f"{model_path.stem}_vecnormalize.pkl"
    )
    vec_normalize = None
    dummy_vec_env = None
    if vecnormalize_path.exists():
        dummy_vec_env = DummyVecEnv([
            lambda: BallCupTrainingEnv(
                stage=args.stage, seed=args.seed, config_path=args.config
            )
        ])
        vec_normalize = VecNormalize.load(
            str(vecnormalize_path), dummy_vec_env
        )
        vec_normalize.training = False
        vec_normalize.norm_reward = False

    try:
        result = run_deterministic_evaluation(
            model,
            vec_normalize,
            stage=args.stage,
            episodes=args.episodes,
            seed=args.seed,
            video_path=args.video,
            config_path=args.config,
        )
    finally:
        if dummy_vec_env is not None:
            dummy_vec_env.close()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
