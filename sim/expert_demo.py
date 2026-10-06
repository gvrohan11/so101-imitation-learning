"""Generate a successful, policy-rate demonstration from the verified grasp pose."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import mujoco
import numpy as np

from sim.rl_env import BallCupTrainingEnv
from sim.search_grasp_poses import main as search_safe_pinch_candidates
from sim.training_config import DEFAULT_CONFIG_PATH, load_training_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _rotation_vector(target_rotation, current_rotation):
    relative = target_rotation @ current_rotation.T
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    angle = float(np.arccos(cosine))
    skew = np.array(
        [
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ],
        dtype=np.float64,
    )
    if angle < 1e-7:
        return 0.5 * skew
    return (angle / (2.0 * np.sin(angle))) * skew


class _RetimedExpert:
    """Waypoint IK teacher whose every action passes through the PPO wrapper."""

    def __init__(self, env: BallCupTrainingEnv, candidate, *, execution_action_noise_std=0.0):
        self.env = env
        self.model, self.data = env.model, env.data
        self.candidate = candidate
        self.observations: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.max_requested_arm_delta = 0.0
        self.execution_action_noise_std = float(execution_action_noise_std)
        if self.execution_action_noise_std < 0.0:
            raise ValueError("execution_action_noise_std must be non-negative")
        self.qpos_ids = env.joint_qpos.copy()
        self.dof_ids = env.joint_qvel.copy()
        self.site_id = env.sim.gripper_site
        self.arm_hold_target = self.data.qpos[self.qpos_ids[:5]].copy()
        self.fixed_geoms = set(env.sim.fixed_finger_geoms)
        self.moving_geoms = set(env.sim.moving_finger_geoms)
        self.finger_geoms = self.fixed_geoms | self.moving_geoms

    def _gripper_action(self, target: float) -> float:
        current = float(self.data.qpos[self.qpos_ids[5]])
        return float(
            np.clip(
                (target - current) / self.env.gripper_action_delta,
                -1.0,
                1.0,
            )
        )

    def _step(self, target_q, gripper_target):
        before = self.env._observation().copy()
        current = self.data.qpos[self.qpos_ids[:5]].copy()
        action = np.zeros(6, dtype=np.float32)
        action[:5] = np.clip(
            (np.asarray(target_q[:5]) - current) / self.env.max_arm_delta,
            -1.0,
            1.0,
        )
        action[5] = np.clip(
            self._gripper_action(gripper_target), -1.0, 1.0
        )
        previous_gripper_target = self.env._gripper_target
        targets, _, _ = self.env.map_action_to_targets(action)
        self.env._gripper_target = previous_gripper_target
        requested_delta = np.abs(targets - current)
        self.max_requested_arm_delta = max(
            self.max_requested_arm_delta, float(np.max(requested_delta))
        )
        if np.any(requested_delta > self.env.max_arm_delta + 1e-8):
            raise RuntimeError("Expert exceeded the policy's per-step arm limit")
        self.arm_hold_target = targets.copy()

        # Keep the clean expert command as the learning label, but optionally
        # perturb the executed arm command. The expert replans from the resulting
        # state on the next control tick, which supplies corrective labels over
        # states a perfectly replayed single trajectory would never visit.
        executed_action = action.copy()
        if self.execution_action_noise_std:
            executed_action[:5] = np.clip(
                executed_action[:5]
                + self.env.rng.normal(0.0, self.execution_action_noise_std, size=5),
                -1.0,
                1.0,
            )
        observation, _, terminated, truncated, info = self.env.step(executed_action)
        self.observations.append(before)
        self.actions.append(action.copy())
        if terminated or truncated:
            if not info.get("is_success", False):
                raise RuntimeError(
                    "The retimed expert episode ended before successful placement: "
                    f"{info.get('failure_reason', 'time limit')}"
                )
        return observation, info

    def _solve_ik(
        self,
        target_xyz,
        seed_q,
        wrist_gripper,
        *,
        target_rotation=None,
        target_axis=None,
    ):
        saved_qpos = self.data.qpos.copy()
        self.data.qpos[self.qpos_ids] = seed_q
        if target_rotation is None:
            self.data.qpos[self.qpos_ids[3:6]] = wrist_gripper
        else:
            self.data.qpos[self.qpos_ids[5]] = wrist_gripper[2]

        try:
            for _ in range(150):
                mujoco.mj_forward(self.model, self.data)
                position_error = target_xyz - self.data.site_xpos[self.site_id]
                jac_pos = np.zeros((3, self.model.nv))
                jac_rot = np.zeros((3, self.model.nv))
                mujoco.mj_jacSite(
                    self.model, self.data, jac_pos, jac_rot, self.site_id
                )
                if target_rotation is None:
                    if np.linalg.norm(position_error) < 0.0008:
                        break
                    jacobian = jac_pos[:, self.dof_ids[:3]]
                    delta = jacobian.T @ np.linalg.solve(
                        jacobian @ jacobian.T + 0.05**2 * np.eye(3),
                        position_error,
                    )
                    controlled_joint_count = 3
                else:
                    rotation_error = _rotation_vector(
                        target_rotation,
                        self.data.site_xmat[self.site_id].reshape(3, 3),
                    )
                    rotation_projector = np.eye(3)
                    if target_axis is not None:
                        rotation_projector -= np.outer(target_axis, target_axis)
                    rotation_error = rotation_projector @ rotation_error
                    if (
                        np.linalg.norm(position_error) < 0.0008
                        and np.linalg.norm(rotation_error) < 0.008
                    ):
                        break
                    rotation_weight = 0.08
                    jacobian = np.vstack(
                        (
                            jac_pos[:, self.dof_ids[:5]],
                            rotation_weight
                            * rotation_projector
                            @ jac_rot[:, self.dof_ids[:5]],
                        )
                    )
                    error = np.concatenate(
                        (position_error, rotation_weight * rotation_error)
                    )
                    delta = jacobian.T @ np.linalg.solve(
                        jacobian @ jacobian.T + 0.05**2 * np.eye(6), error
                    )
                    controlled_joint_count = 5

                length = float(np.linalg.norm(delta))
                if length > 0.05:
                    delta *= 0.05 / length
                for joint_index in range(controlled_joint_count):
                    qpos_index = self.qpos_ids[joint_index]
                    # The env's ordered joint ranges are already validated;
                    # use them directly rather than depending on model order.
                    low, high = self.env.joint_ranges[joint_index]
                    self.data.qpos[qpos_index] = np.clip(
                        self.data.qpos[qpos_index] + delta[joint_index], low, high
                    )

            mujoco.mj_forward(self.model, self.data)
            solution = self.data.qpos[self.qpos_ids].copy()
            position_residual = float(
                np.linalg.norm(target_xyz - self.data.site_xpos[self.site_id])
            )
            if target_rotation is None:
                residual = position_residual
            else:
                rotation_residual = _rotation_vector(
                    target_rotation,
                    self.data.site_xmat[self.site_id].reshape(3, 3),
                )
                if target_axis is not None:
                    projector = np.eye(3) - np.outer(target_axis, target_axis)
                    rotation_residual = projector @ rotation_residual
                residual = max(position_residual, 0.08 * float(np.linalg.norm(rotation_residual)))
            return solution, float(residual)
        finally:
            self.data.qpos[:] = saved_qpos
            mujoco.mj_forward(self.model, self.data)

    def _move_segment(
        self,
        start_xyz,
        target_xyz,
        start_wrist,
        target_wrist,
        waypoint_count,
        gripper_target,
        *,
        target_rotation=None,
        target_axis=None,
        max_actions_per_waypoint=20,
    ):
        start_xyz = np.asarray(start_xyz, dtype=np.float64)
        target_xyz = np.asarray(target_xyz, dtype=np.float64)
        start_wrist = np.asarray(start_wrist, dtype=np.float64)
        target_wrist = np.asarray(target_wrist, dtype=np.float64)
        for waypoint_index in range(1, waypoint_count + 1):
            fraction = waypoint_index / waypoint_count
            waypoint_xyz = start_xyz + fraction * (target_xyz - start_xyz)
            waypoint_wrist = start_wrist + fraction * (target_wrist - start_wrist)
            for _ in range(max_actions_per_waypoint):
                current_q = self.data.qpos[self.qpos_ids].copy()
                ik_q, residual = self._solve_ik(
                    waypoint_xyz,
                    current_q,
                    waypoint_wrist,
                    target_rotation=target_rotation,
                    target_axis=target_axis,
                )
                if residual > 0.03:
                    raise RuntimeError(
                        "Expert IK failed at Cartesian waypoint "
                        f"{waypoint_index}/{waypoint_count} ({residual * 1000:.1f} mm); "
                        f"target={np.round(waypoint_xyz, 5)}, "
                        f"actual={np.round(self.data.site_xpos[self.site_id], 5)}, "
                        f"q={np.round(self.data.qpos[self.qpos_ids], 4)}"
                    )
                position_error = float(
                    np.linalg.norm(waypoint_xyz - self.data.site_xpos[self.site_id])
                )
                wrist_error = float(
                    np.linalg.norm(current_q[3:5] - waypoint_wrist[:2])
                ) if target_rotation is None else 0.0
                orientation_error = 0.0
                if target_rotation is not None:
                    orientation_error = _rotation_vector(
                        target_rotation,
                        self.data.site_xmat[self.site_id].reshape(3, 3),
                    )
                    if target_axis is not None:
                        projector = np.eye(3) - np.outer(target_axis, target_axis)
                        orientation_error = projector @ orientation_error
                    orientation_error = float(np.linalg.norm(orientation_error))
                if (
                    position_error < 0.0025
                    and wrist_error < 0.02
                    and orientation_error < 0.06
                ):
                    break
                self._step(ik_q, gripper_target)
                if self.env.steps >= self.env.horizon:
                    raise RuntimeError("Retimed expert exceeded the episode horizon")
            else:
                raise RuntimeError(
                    "Retimed expert could not reach waypoint "
                    f"{waypoint_index}/{waypoint_count}; "
                    f"position_error={position_error * 1000:.1f} mm, "
                    f"orientation_error={orientation_error:.3f} rad, "
                    f"target={np.round(waypoint_xyz, 5)}, "
                    f"actual={np.round(self.data.site_xpos[self.site_id], 5)}"
                )

    def _hold(self, steps, gripper_target):
        for _ in range(int(steps)):
            target_q = np.concatenate(
                (self.arm_hold_target, [self.data.qpos[self.qpos_ids[5]]])
            )
            self._step(target_q, gripper_target)

    def _contact_point(self, geoms):
        points = [
            self.data.contact[index].pos.copy()
            for index in range(self.data.ncon)
            if self.env.sim.ball_geom
            in (self.data.contact[index].geom1, self.data.contact[index].geom2)
            and geoms.intersection(
                {
                    int(self.data.contact[index].geom1),
                    int(self.data.contact[index].geom2),
                }
            )
            and self.data.contact[index].dist <= 0.0
        ]
        if not points:
            raise RuntimeError("Expert pinch is missing one physical fingertip contact")
        return np.mean(points, axis=0)

    def run(self, seed: int):
        observation, reset_info = self.env.reset(seed=seed)
        del observation
        offset = np.asarray(self.candidate[3], dtype=np.float64)
        wrist_flex = float(self.candidate[1])
        wrist_roll = float(self.candidate[2])
        contact_q = np.asarray(self.candidate[6], dtype=np.float64)
        ball_start = reset_info["ball_position"].copy()
        ball_contact = ball_start + offset
        wrist_start = self.data.qpos[self.qpos_ids[3:6]].copy()
        open_gripper = float(self.env.open_gripper_target)
        pinch_gripper = float(contact_q[-1])

        start_site = self.data.site_xpos[self.site_id].copy()
        clearance_site = start_site + np.asarray([0.0, 0.0, 0.08])
        transit_site = ball_contact + np.asarray([0.0, 0.0, 0.09])
        transit_site[2] = max(transit_site[2], clearance_site[2])
        side_site = ball_contact + np.asarray([0.0, 0.0, 0.03])
        wrist_target = np.asarray([wrist_flex, wrist_roll, open_gripper])

        self._move_segment(
            start_site, clearance_site, wrist_start, wrist_start,
            8, open_gripper,
        )
        current_site = self.data.site_xpos[self.site_id].copy()
        current_wrist = self.data.qpos[self.qpos_ids[3:6]].copy()
        self._move_segment(
            current_site, transit_site, current_wrist, wrist_target,
            40, open_gripper,
        )
        current_site = self.data.site_xpos[self.site_id].copy()
        self._move_segment(
            current_site, side_site, wrist_target, wrist_target,
            32, open_gripper,
        )
        current_site = self.data.site_xpos[self.site_id].copy()
        self._move_segment(
            current_site, ball_contact, wrist_target, wrist_target,
            20, open_gripper,
        )

        # Close to the first validated physical two-finger pinch, then preload
        # gradually. A continuous gripper target is essential here; snapping to
        # the hard stop does not reproduce this grasp.
        close_steps = max(
            1,
            int(
                np.ceil(
                    (open_gripper - pinch_gripper)
                    / self.env.gripper_action_delta
                )
            ),
        )
        for step_index in range(1, close_steps + 1):
            target = open_gripper + step_index / close_steps * (
                pinch_gripper - open_gripper
            )
            target_q = np.concatenate(
                (self.arm_hold_target, [self.data.qpos[self.qpos_ids[5]]])
            )
            self._step(target_q, target)
        if not self.env.sim._is_pinched():
            raise RuntimeError(
                "Demonstration failed to make bilateral mesh contact; "
                f"site={np.round(self.data.site_xpos[self.site_id], 5)}, "
                f"ball={np.round(self.data.xpos[self.env.sim.ball_body], 5)}, "
                f"q={np.round(self.data.qpos[self.qpos_ids], 4)}, "
                f"pinch_target={pinch_gripper:.4f}, "
                f"contacts={[(int(self.data.contact[i].geom1), int(self.data.contact[i].geom2), round(float(self.data.contact[i].dist), 5)) for i in range(self.data.ncon) if self.env.sim.ball_geom in (self.data.contact[i].geom1, self.data.contact[i].geom2)]}"
            )
        for step_index in range(1, 6):
            target = pinch_gripper - 0.02 * step_index / 5.0
            target_q = np.concatenate(
                (self.arm_hold_target, [self.data.qpos[self.qpos_ids[5]]])
            )
            self._step(target_q, target)
        pinch_target = pinch_gripper - 0.02
        self._hold(5, pinch_target)
        if not self.env.grasp_success or not self.env.sim._is_pinched():
            raise RuntimeError("Demonstration pinch did not remain stable for 0.1 s")

        lift_start_site = self.data.site_xpos[self.site_id].copy()
        rotation_at_pinch = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
        fixed_contact = self._contact_point(self.fixed_geoms)
        moving_contact = self._contact_point(self.moving_geoms)
        jaw_axis = fixed_contact - moving_contact
        jaw_axis /= np.linalg.norm(jaw_axis)
        lift_target = lift_start_site + np.asarray([0.0, 0.0, 0.025])
        wrist_hold = self.data.qpos[self.qpos_ids[3:6]].copy()
        self._move_segment(
            lift_start_site, lift_target, wrist_hold, wrist_hold,
            32, pinch_target,
            target_rotation=rotation_at_pinch, target_axis=jaw_axis,
        )
        if not self.env.sim._is_pinched():
            raise RuntimeError("Demonstration lost the ball during its initial lift")

        site_to_ball = (
            self.data.site_xpos[self.site_id].copy()
            - self.data.xpos[self.env.sim.ball_body].copy()
        )
        rotation = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
        # The verified carry trajectory preserves the grasp plane's pitch and
        # roll while leaving world-up yaw as the arm's redundant DOF.
        carry_axis = np.asarray([0.0, 0.0, 1.0])
        cup = self.data.xpos[self.env.sim.cup_body].copy()
        _, _, rim_top = self.env._cup_dimensions()
        carry_height = max(
            float(self.data.xpos[self.env.sim.ball_body, 2]),
            rim_top + self.env.ball_radius + 0.025,
        )
        lifted_ball_target = self.data.xpos[self.env.sim.ball_body].copy()
        lifted_ball_target[2] = carry_height
        lifted_site_target = lifted_ball_target + site_to_ball
        current_site = self.data.site_xpos[self.site_id].copy()
        self._move_segment(
            current_site, lifted_site_target, wrist_hold, wrist_hold,
            30, pinch_target,
            target_rotation=rotation, target_axis=carry_axis,
        )

        carry_ball_target = np.asarray([cup[0], cup[1], carry_height])
        carry_site_target = carry_ball_target + site_to_ball
        current_site = self.data.site_xpos[self.site_id].copy()
        self._move_segment(
            current_site, carry_site_target, wrist_hold, wrist_hold,
            96, pinch_target,
            target_rotation=rotation, target_axis=carry_axis,
        )
        if not self.env.sim._is_pinched():
            raise RuntimeError("Demonstration lost bilateral contact during cup transfer")

        # Joint tracking and compliant contacts create a small ball-to-tool
        # lag. Center the real ball over the cup before opening instead of
        # assuming the nominal tool transform remained exact during carry.
        for correction_index in range(3):
            ball_now = self.data.xpos[self.env.sim.ball_body].copy()
            cup_error = cup[:2] - ball_now[:2]
            if np.linalg.norm(cup_error) <= 0.006:
                break
            current_site = self.data.site_xpos[self.site_id].copy()
            corrected_site = current_site + np.asarray(
                [cup_error[0], cup_error[1], 0.0]
            )
            correction_waypoints = max(
                4, int(np.ceil(np.linalg.norm(cup_error) / 0.004))
            )
            self._move_segment(
                current_site,
                corrected_site,
                wrist_hold,
                wrist_hold,
                correction_waypoints,
                pinch_target,
                target_rotation=rotation,
                target_axis=carry_axis,
            )
            if not self.env.sim._is_pinched():
                raise RuntimeError(
                    "Demonstration lost bilateral contact while centering over the cup"
                )
        final_center_error = float(
            np.linalg.norm(
                self.data.xpos[self.env.sim.ball_body, :2] - cup[:2]
            )
        )
        if final_center_error > 0.012:
            raise RuntimeError(
                "Demonstration could not center the held ball above the cup; "
                f"error={final_center_error * 1000:.1f} mm"
            )
        self._hold(20, pinch_target)

        self._hold(20, open_gripper)
        if self.env.holding_ball:
            raise RuntimeError("Gripper did not release the ball over the cup")
        for _ in range(100):
            self._hold(1, open_gripper)
            if self.env.placement_success:
                break
        if not self.env.placement_success:
            raise RuntimeError(
                "Retimed demonstration did not settle the released ball in the cup; "
                f"ball={np.round(self.data.xpos[self.env.sim.ball_body], 4)}, "
                f"cup={np.round(self.data.xpos[self.env.sim.cup_body], 4)}"
            )

        actions = np.asarray(self.actions, dtype=np.float32)
        observations = np.asarray(self.observations, dtype=np.float32)
        if len(actions) != len(observations) or not len(actions):
            raise RuntimeError("Expert demonstration recording is empty or misaligned")
        if np.max(np.abs(actions[:, :5])) > 1.0 + 1e-7:
            raise RuntimeError("Recorded expert action violates normalized action bounds")
        if self.max_requested_arm_delta > self.env.max_arm_delta + 1e-8:
            raise RuntimeError("Recorded demonstration exceeded the 2-degree arm limit")
        return observations, actions, {
            "success": True,
            "seed": int(seed),
            "steps": int(len(actions)),
            "control_hz": int(self.env.control_hz),
            "max_arm_delta_degrees": float(np.rad2deg(self.max_requested_arm_delta)),
            "grasp_site_offset_m": offset.tolist(),
            "wrist_flex_rad": wrist_flex,
            "wrist_roll_rad": wrist_roll,
            "ball_final_position_m": self.data.xpos[self.env.sim.ball_body].tolist(),
            "cup_position_m": self.data.xpos[self.env.sim.cup_body].tolist(),
            "phase_success": {
                "grasp": bool(self.env.grasp_success),
                "lift": bool(self.env.lift_success),
                "transport": bool(self.env.transport_success),
                "above_cup": bool(self.env.above_cup_success),
                "correct_release": bool(self.env.correct_release),
                "placement": bool(self.env.placement_success),
            },
        }


def generate_demonstration(
    config_path=None, *, seed=None, execution_action_noise_std=0.0
):
    config = load_training_config(config_path)
    ball_radius = 0.020
    with contextlib.redirect_stdout(io.StringIO()):
        candidates = search_safe_pinch_candidates()
    if not candidates:
        raise RuntimeError("Grasp pose search found no valid physical pinch candidate")
    candidate = candidates[0]
    expected_offset = np.asarray(config["task"]["grasp_site_offset_m"], dtype=np.float64)
    if not np.allclose(candidate[3], expected_offset, atol=1e-9):
        raise RuntimeError(
            "Training grasp target and validated pose search disagree: "
            f"{expected_offset} vs {candidate[3]}"
        )
    if not np.isclose(float(candidate[1]), config["task"]["grasp_wrist_flex_rad"]):
        raise RuntimeError("Training wrist-flex target disagrees with the validated grasp")
    if not np.isclose(float(candidate[2]), config["task"]["grasp_wrist_roll_rad"]):
        raise RuntimeError("Training wrist-roll target disagrees with the validated grasp")

    env = BallCupTrainingEnv(
        stage=1,
        seed=int(config["seed"] if seed is None else seed),
        config_path=config_path,
    )
    try:
        teacher = _RetimedExpert(
            env,
            candidate,
            execution_action_noise_std=execution_action_noise_std,
        )
        observations, actions, metadata = teacher.run(
            int(config["seed"] if seed is None else seed)
        )
    finally:
        env.close()
    metadata["ball_radius_m"] = ball_radius
    metadata["execution_action_noise_std"] = float(execution_action_noise_std)
    return observations, actions, metadata


def save_demonstration(path, observations, actions, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "ball_cup_ppo" / "demonstration_stage01.npz",
    )
    parser.add_argument("--execution-action-noise-std", type=float, default=0.0)
    args = parser.parse_args(argv)
    observations, actions, metadata = generate_demonstration(
        args.config,
        execution_action_noise_std=args.execution_action_noise_std,
    )
    output = save_demonstration(args.output, observations, actions, metadata)
    print(json.dumps({"demonstration": str(output), **metadata}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
