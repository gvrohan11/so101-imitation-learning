"""Short environment checks that must pass before starting PPO."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from gymnasium.utils.env_checker import check_env

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sim.rl_env import BallCupTrainingEnv
from sim.training_config import load_training_config


def assert_finite_observation(observation):
    assert observation.shape == (25,), observation.shape
    assert np.all(np.isfinite(observation))


def validate_action_mapping(env):
    observation, _ = env.reset(seed=19)
    assert_finite_observation(observation)
    current = env.data.qpos[env.joint_qpos[:5]].copy()
    targets, gripper, _ = env.map_action_to_targets(np.ones(6, dtype=np.float32))
    expected = np.minimum(
        current + np.deg2rad(2.0), env.joint_ranges[:, 1]
    )
    np.testing.assert_allclose(targets, expected, atol=1e-7)
    assert gripper == env.open_gripper_target

    _, closed, _ = env.map_action_to_targets(np.array([0, 0, 0, 0, 0, -1]))
    _, held, _ = env.map_action_to_targets(np.zeros(6))
    _, opened, _ = env.map_action_to_targets(np.array([0, 0, 0, 0, 0, 1]))
    assert closed == env.close_gripper_target
    assert held == closed
    assert opened == env.open_gripper_target

    high = env.joint_ranges[0, 1]
    env.data.qpos[env.joint_qpos[0]] = high - 1e-4
    mujoco.mj_forward(env.model, env.data)
    clipped, _, _ = env.map_action_to_targets(np.array([1, 0, 0, 0, 0, 0]))
    assert clipped[0] <= high + 1e-10
    env.reset(seed=20)


def validate_gripper_motion(env):
    env.reset(seed=21)
    gripper_qpos = env.joint_qpos[-1]
    opened = float(env.data.qpos[gripper_qpos])
    close_action = np.array([0, 0, 0, 0, 0, -1], dtype=np.float32)
    for _ in range(5):
        _, _, terminated, truncated, _ = env.step(close_action)
        assert not terminated and not truncated
    closed = float(env.data.qpos[gripper_qpos])
    assert closed < opened - 0.05, (opened, closed)

    open_action = np.array([0, 0, 0, 0, 0, 1], dtype=np.float32)
    for _ in range(5):
        _, _, terminated, truncated, _ = env.step(open_action)
        assert not terminated and not truncated
    reopened = float(env.data.qpos[gripper_qpos])
    assert reopened > closed + 0.05, (closed, reopened)


def validate_success_detector_and_physical_cup(env):
    # A fully contained, released, already lifted ball must become success only
    # after its 0.25 s settling window. Touching/holding must disqualify it.
    _, reset_info = env.reset(seed=31)
    env.grasp_success = True
    env.lift_success = True
    cup = env.data.xpos[env.sim.cup_body].copy()
    _, floor_top, _ = env._cup_dimensions()
    env.data.qpos[env.sim.ball_qpos:env.sim.ball_qpos + 3] = [
        cup[0], cup[1], floor_top + env.ball_radius
    ]
    env.data.qvel[env.sim.ball_qvel:env.sim.ball_qvel + 6] = 0.0
    mujoco.mj_forward(env.model, env.data)
    assert env._ball_inside_cup_volume(safety_margin=0.0)
    env.holding_ball = True
    assert not env._is_settled_placement()
    env.holding_ball = False
    for step in range(6):
        _, _, terminated, truncated, info = env.step(np.array([0, 0, 0, 0, 0, 1]))
        if terminated:
            assert info["is_success"] is True
            assert info["correct_release"] is True
            break
        assert not truncated
    else:
        raise AssertionError("A settled, released ball inside the cup was not success")

    # Drop the nominal training ball through the actual contact geometry in
    # both the nominal and smallest randomized cup.
    for trial, (radius, cup_scale) in enumerate(((0.020, 1.0), (0.020, 0.9))):
        env.reset(seed=32 + trial)
        if radius != env.ball_radius or cup_scale != env.cup_scale:
            env.ball_radius = radius
            env.cup_scale = cup_scale
            cup_xy = env.data.xpos[env.sim.cup_body, :2].copy()
            env._activate_scene_variants(radius, cup_scale, cup_xy)
            mujoco.mj_forward(env.model, env.data)
        env.grasp_success = True
        env.lift_success = True
        cup = env.data.xpos[env.sim.cup_body].copy()
        _, _, rim_top = env._cup_dimensions()
        env.data.qpos[env.sim.ball_qpos:env.sim.ball_qpos + 3] = [
            cup[0], cup[1], rim_top + env.ball_radius + 0.02
        ]
        env.data.qvel[env.sim.ball_qvel:env.sim.ball_qvel + 6] = 0.0
        mujoco.mj_forward(env.model, env.data)
        did_settle = False
        for _ in range(100):
            _, _, terminated, truncated, info = env.step(
                np.array([0, 0, 0, 0, 0, 1], dtype=np.float32)
            )
            if terminated:
                did_settle = bool(info["is_success"])
                break
            if truncated:
                break
        assert did_settle, (
            f"The radius {radius:.3f} m ball failed to settle in the "
            f"scale {cup_scale:.1f} cup"
        )


def validate_timeout_and_nan(env):
    short_env = BallCupTrainingEnv(stage=1, horizon=3, seed=41)
    try:
        short_env.reset(seed=41)
        for index in range(3):
            _, _, terminated, truncated, _ = short_env.step(np.zeros(6))
            if index < 2:
                assert not terminated and not truncated
            else:
                assert not terminated and truncated
        try:
            short_env.step(np.array([np.nan, 0, 0, 0, 0, 0]))
        except ValueError:
            pass
        else:
            raise AssertionError("NaN action was not rejected")
    finally:
        short_env.close()


def validate_curriculum(config):
    env = BallCupTrainingEnv(stage=1, seed=7)
    try:
        for radius in config["task"]["ball_radius_variants_m"]:
            for scale in config["task"]["cup_scale_variants"]:
                ratio = (0.09 * scale) / (2 * radius)
                assert ratio >= config["task"]["minimum_cup_opening_to_ball_diameter_ratio"]
        for stage in range(1, 6):
            env.set_stage(stage)
            for trial in range(8):
                observation, info = env.reset(seed=10_000 * stage + trial)
                assert_finite_observation(observation)
                ball = info["ball_position"]
                cup = info["cup_position"]
                start = info["start_joint_positions"]
                assert np.all(np.isfinite(ball)) and np.all(np.isfinite(cup))
                assert np.all(start >= env.joint_ranges[:, 0] - 1e-8)
                assert np.all(start <= env.joint_ranges[:, 1] + 1e-8)
                assert env._start_pose_is_clear()
                if stage == 1:
                    np.testing.assert_allclose(ball[:2], config["task"]["fixed_ball_xy"])
                    np.testing.assert_allclose(cup[:2], config["task"]["fixed_cup_xy"])
                if stage >= 2:
                    bounds = np.asarray(config["task"]["ball_xy_bounds"])
                    assert np.all(ball[:2] >= bounds[:, 0] - 1e-9)
                    assert np.all(ball[:2] <= bounds[:, 1] + 1e-9)
                if stage >= 3:
                    bounds = np.asarray(config["task"]["cup_xy_bounds"])
                    assert np.all(cup[:2] >= bounds[:, 0] - 1e-9)
                    assert np.all(cup[:2] <= bounds[:, 1] + 1e-9)
                if stage >= 4:
                    jitter = np.deg2rad(config["task"]["start_joint_jitter_degrees"])
                    home = np.asarray(config["task"]["fixed_arm_start_rad"])
                    assert np.all(np.abs(start - home) <= jitter + 1e-9)
                if stage == 5:
                    assert env.ball_radius in config["task"]["ball_radius_variants_m"]
                    assert env.cup_scale in config["task"]["cup_scale_variants"]
                    ratio = (0.09 * env.cup_scale) / (2 * env.ball_radius)
                    assert ratio >= config["task"]["minimum_cup_opening_to_ball_diameter_ratio"]
        return 5 * 8
    finally:
        env.close()


def validate_random_rollouts():
    env = BallCupTrainingEnv(stage=5, horizon=80, seed=71)
    rng = np.random.default_rng(71)
    completed_episodes = 0
    total_steps = 0
    try:
        for episode in range(10):
            observation, _ = env.reset(seed=71_000 + episode)
            assert_finite_observation(observation)
            while True:
                action = rng.uniform(-1.0, 1.0, size=6).astype(np.float32)
                observation, reward, terminated, truncated, info = env.step(action)
                assert_finite_observation(observation)
                assert np.isfinite(reward)
                total_steps += 1
                if terminated or truncated:
                    assert isinstance(info["is_success"], bool)
                    completed_episodes += 1
                    break
    finally:
        env.close()
    assert completed_episodes == 10
    return {"episodes": completed_episodes, "policy_steps": total_steps}


def main():
    config = load_training_config()
    env = BallCupTrainingEnv(stage=1, horizon=100, seed=5)
    try:
        check_env(env, skip_render_check=True)
        observation, _ = env.reset(seed=5)
        assert_finite_observation(observation)
        validate_action_mapping(env)
        validate_gripper_motion(env)
        validate_success_detector_and_physical_cup(env)
    finally:
        env.close()

    validate_timeout_and_nan(env)
    stage_reset_count = validate_curriculum(config)
    random_result = validate_random_rollouts()
    print(json.dumps({
        "gymnasium_env_checker": "passed",
        "observation_shape": [25],
        "action_shape": [6],
        "action_mapping_and_joint_limits": "passed",
        "gripper_open_close_motion": "passed",
        "geometric_release_success_and_physical_cup_drop": "passed",
        "timeout_and_nan_action_checks": "passed",
        "curriculum_stage_resets": stage_reset_count,
        "random_rollouts": random_result,
    }, indent=2))


if __name__ == "__main__":
    main()
