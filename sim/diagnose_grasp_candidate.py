from pathlib import Path

import os

import mujoco
import numpy as np
from PIL import Image

from sim.ball_cup_env import (
    JOINT_NAMES,
    BallCupEnv,
)
from sim.collision_check import JawTableCollisionChecker
from sim.search_grasp_poses import main as search_safe_pinch_candidates

# Test the best static lower-contact candidate selected by the pose search.
MAX_CANDIDATES = 10
DIAGNOSTIC_SEED = int(os.environ.get("SO101_GRASP_SEED", "0"))

# The ball starts near z=0.020 m. Lift the gripper 25 mm so a successful
# grasp has room to raise the ball above the required z=0.035 m threshold.
LIFT_METERS = 0.025
LIFT_WAYPOINTS = 25
FRAMES_PER_LIFT_WAYPOINT = 30
MIN_FRAMES_PER_PLACE_WAYPOINT = 100
CUP_SETTLE_FRAMES = 1800
CUP_STABLE_FRAMES = 150

# Abort an approach if it pushes the ball too far.
MAX_APPROACH_BALL_SHIFT = 0.012  # meters
MAX_LIFT_LATERAL_SHIFT = 0.025  # meters
APPROACH_CLEARANCE_METERS = 0.08
APPROACH_WAYPOINTS = 20
MIN_FRAMES_PER_APPROACH_WAYPOINT = 50
RADIANS_PER_APPROACH_FRAME = 0.009
MAX_WAYPOINT_SETTLE_FRAMES = 200
PINCH_COMPRESSION_FRAMES = 200
GRIPPER_CLOSE_FRAMES = 1000
PINCH_PRELOAD_RADIANS = 0.02
PINCH_SETTLE_FRAMES = 20
MIN_GRIP_NORMAL_FORCE = 0.01
GRIPPER_FORCE_CLOSE_STEP = 0.0001
MAX_FORCE_CONTROL_PRELOAD = 0.08
MAX_GRIP_RECOVERY_FRAMES = 500
RECORD_CANDIDATE_VIDEO = (
    os.environ.get("SO101_RECORD_CANDIDATE_VIDEO", "0") == "1"
)
VIDEO_CANDIDATE_INDEX = int(
    os.environ.get("SO101_VIDEO_CANDIDATE_INDEX", "1")
)


def main():
    candidates = search_safe_pinch_candidates()
    if not candidates:
        print("No static pinch candidates found; no dynamic tests attempted.")
        return 1

    candidates = candidates[:MAX_CANDIDATES]
    print(f"\nDynamically testing {len(candidates)} candidates")

    # The full placement path can exceed 10,000 control steps. Keep a real
    # diagnostic run alive through transfer and release.
    ball_radius = float(os.environ.get("SO101_DIAGNOSTIC_BALL_RADIUS", "0.020"))
    env = BallCupEnv(
        render_images=False,
        frame_skip=1,
        horizon=100000,
        ball_radius=ball_radius,
    )
    model, data = env.model, env.data
    qpos_ids = np.array(
        [env.joint_qpos[name] for name in JOINT_NAMES], dtype=np.int32
    )
    dof_ids = np.array(
        [
            model.jnt_dofadr[
                mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT, name
                )
            ]
            for name in JOINT_NAMES
        ],
        dtype=np.int32,
    )
    checker = JawTableCollisionChecker(env)

    def action_for(joint_positions):
        action = []
        for target, actuator_id in zip(joint_positions, env.actuator_ids):
            low, high = model.actuator_ctrlrange[actuator_id]
            action.append(2.0 * (target - low) / (high - low) - 1.0)
        return np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

    def ball_position():
        return data.xpos[env.ball_body].copy()

    def both_finger_meshes_touch_ball():
        contacted = set()
        group_for_geom = {
            geom: group_index
            for group_index, geoms in enumerate(env.finger_geom_groups)
            for geom in geoms
        }
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            if env.ball_geom not in (contact.geom1, contact.geom2):
                continue
            other_geom = (
                int(contact.geom2)
                if contact.geom1 == env.ball_geom
                else int(contact.geom1)
            )
            group_index = group_for_geom.get(other_geom)
            if group_index is not None and contact.dist <= 0.0:
                contacted.add(group_index)
        return len(contacted) == len(env.finger_geom_groups)

    def print_finger_contact_state(stage):
        print(f"  {stage}: ball={np.round(ball_position(), 5)}")
        group_for_geom = {
            geom: group_index
            for group_index, geoms in enumerate(env.finger_geom_groups)
            for geom in geoms
        }
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            if env.ball_geom not in (contact.geom1, contact.geom2):
                continue
            finger_geom = int(
                contact.geom2
                if contact.geom1 == env.ball_geom
                else contact.geom1
            )
            group_index = group_for_geom.get(finger_geom)
            if group_index is None:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, contact_index, force)
            finger_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, finger_geom
            )
            print(
                f"    group={group_index} {finger_name}: "
                f"distance={contact.dist:.6f} m, "
                f"force(normal,tangent)={np.round(force[:3], 5)}, "
                f"point={np.round(contact.pos, 5)}"
            )

    def finger_mesh_normal_forces():
        forces = {index: 0.0 for index in range(len(env.finger_geom_groups))}
        group_for_geom = {
            geom: group_index
            for group_index, geoms in enumerate(env.finger_geom_groups)
            for geom in geoms
        }
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            if env.ball_geom not in (contact.geom1, contact.geom2):
                continue
            finger = int(
                contact.geom2
                if contact.geom1 == env.ball_geom
                else contact.geom1
            )
            group_index = group_for_geom.get(finger)
            if group_index is None or contact.dist > 0.0:
                continue
            contact_force = np.zeros(6)
            mujoco.mj_contactForce(
                model, data, contact_index, contact_force
            )
            forces[group_index] += float(contact_force[0])
        return forces

    def actual_contact_point(geoms):
        points = [
            data.contact[index].pos.copy()
            for index in range(data.ncon)
            if env.ball_geom
            in (data.contact[index].geom1, data.contact[index].geom2)
            and any(
                geom in (data.contact[index].geom1, data.contact[index].geom2)
                for geom in geoms
            )
            and data.contact[index].dist <= 0.0
        ]
        if not points:
            raise RuntimeError("Expected a real ball contact on each finger")
        return np.mean(points, axis=0)

    def gripper_force_is_sufficient():
        forces = finger_mesh_normal_forces()
        return min(forces.values()) >= MIN_GRIP_NORMAL_FORCE

    def maintain_gripper_force(hold_state):
        if hold_state is None or hold_state["target"] is None:
            return
        forces = finger_mesh_normal_forces()
        weakest = min(forces.values())
        if weakest < MIN_GRIP_NORMAL_FORCE:
            hold_state["target"] = max(
                hold_state["floor"],
                hold_state["target"] - GRIPPER_FORCE_CLOSE_STEP,
            )

    # The cup diagnostic only passes if bilateral contact is uninterrupted
    # from the start of the lift through release.
    contact_integrity = {"lost": False}

    def move_to(
        target_positions,
        frame_count,
        stage,
        ball_start,
        capture,
        stop_when_pinched=False,
        check_ball_shift=True,
        require_two_finger_contact=False,
        contact_tracking=None,
        gripper_hold=None,
    ):
        """Move to a pose, freezing the jaw at first stable pinch contact."""
        start_positions = data.qpos[qpos_ids].copy()
        pinch_frame = None
        pinch_start_positions = None
        initial_frame_count = frame_count
        frame = 0
        grip_recovery_frames = 0
        arm_hold_target = start_positions.copy()

        def command_at_fraction(fraction):
            if contact_tracking is None:
                return start_positions + fraction * (
                    target_positions - start_positions
                )

            nominal_site = contact_tracking["start_site"] + fraction * (
                contact_tracking["target_site"]
                - contact_tracking["start_site"]
            )
            nominal_ball = nominal_site - contact_tracking["site_to_ball"]
            ball_error = ball_position() - nominal_ball
            # Follow the prescribed gripper trajectory. Correcting the target
            # from the ball's measured lag can chase a slipping object and
            # unload the real finger contacts instead of maintaining the
            # closed-jaw pose.
            correction = np.zeros(3)
            correction_norm = float(np.linalg.norm(correction))
            if correction_norm > 0.020:
                correction *= 0.020 / correction_norm
            tracked_site = nominal_site + correction
            commanded, ik_error = solve_cartesian_waypoint(
                tracked_site,
                data.qpos[qpos_ids].copy(),
                target_positions[3:6],
                target_rotation=contact_tracking["target_rotation"],
                target_axis=contact_tracking["target_axis"],
            )
            if ik_error > 0.012:
                raise RuntimeError(
                    f"contact-follow IK error {ik_error * 1000:.1f} mm; "
                    f"ball error={np.round(ball_error, 5)}"
                )
            if gripper_hold is not None:
                commanded[-1] = gripper_hold["target"]
            return commanded

        while frame < frame_count:
            grip_ready = True
            if require_two_finger_contact:
                grip_ready = gripper_force_is_sufficient()
                if not grip_ready:
                    maintain_gripper_force(gripper_hold)

            if not grip_ready:
                # Keep the last commanded arm pose during jaw recovery. Using
                # measured joints as a new target each frame lets gravity
                # pull the arm away from the ball.
                commanded = arm_hold_target.copy()
                commanded[-1] = gripper_hold["target"]
            elif pinch_start_positions is None:
                fraction = (frame + 1) / initial_frame_count
                try:
                    commanded = command_at_fraction(fraction)
                except RuntimeError as error:
                    print(f"  abort: {stage}: {error}")
                    return False
                if contact_tracking is None and gripper_hold is not None:
                    commanded[-1] = gripper_hold["target"]
            else:
                preload_fraction = min(
                    (frame + 1 - pinch_frame) / PINCH_COMPRESSION_FRAMES,
                    1.0,
                )
                # Continue to the statically validated first-contact target.
                # Holding the measured joint angle here lets the position
                # actuator relax open while the jaw is still moving.
                commanded = target_positions.copy()
                commanded[-1] -= (
                    preload_fraction * PINCH_PRELOAD_RADIANS
                )
            if grip_ready:
                arm_hold_target = commanded.copy()
            observation, _, terminated, truncated, _ = env.step(
                action_for(commanded)
            )
            capture(observation)

            actual = data.qpos[qpos_ids].copy()
            if not checker.pose_is_collision_free(actual):
                print(f"  abort: jaw/table collision during {stage}")
                return False

            shift = np.linalg.norm(ball_position() - ball_start)
            if check_ball_shift and shift > MAX_APPROACH_BALL_SHIFT:
                print(
                    f"  abort: ball shifted {shift * 1000:.1f} mm "
                    f"during {stage}"
                )
                for contact_index in range(data.ncon):
                    contact = data.contact[contact_index]
                    if env.ball_geom not in (
                        contact.geom1, contact.geom2
                    ):
                        continue

                    other_geom = (
                        int(contact.geom2)
                        if contact.geom1 == env.ball_geom
                        else int(contact.geom1)
                    )
                    contact_force = np.zeros(6, dtype=np.float64)
                    mujoco.mj_contactForce(
                        model, data, contact_index, contact_force
                    )
                    other_body = int(model.geom_bodyid[other_geom])
                    body_rotation = data.xmat[other_body].reshape(3, 3)
                    local_point = body_rotation.T @ (
                        contact.pos - data.xpos[other_body]
                    )
                    world_normal = contact.frame[:3].copy()
                    local_normal = body_rotation.T @ world_normal
                    print(
                        "  ball contact at abort:",
                        mujoco.mj_id2name(
                            model, mujoco.mjtObj.mjOBJ_GEOM, other_geom
                        ),
                        "distance=",
                        round(float(contact.dist), 6),
                        "normal_force=",
                        round(float(contact_force[0]), 5),
                        "point=",
                        np.round(contact.pos, 4),
                        "body-local point=",
                        np.round(local_point, 5),
                        "body-local normal=",
                        np.round(local_normal, 4),
                    )
                return False

            if require_two_finger_contact and not both_finger_meshes_touch_ball():
                contact_integrity["lost"] = True
                grip_recovery_frames += 1
                if grip_recovery_frames == 1:
                    print(
                        f"  contact slipped during {stage}; holding the arm "
                        "and closing the real jaws to recover bilateral contact"
                    )
                if grip_recovery_frames >= MAX_GRIP_RECOVERY_FRAMES:
                    print(
                        f"  abort: actual finger-mesh contact could not be "
                        f"recovered during {stage}; site="
                        f"{np.round(data.site_xpos[env.gripper_site], 5)}, "
                        f"q={np.round(actual, 4)}"
                    )
                    print_finger_contact_state("remaining real finger contacts")
                    return False
                maintain_gripper_force(gripper_hold)
                continue
            if require_two_finger_contact:
                maintain_gripper_force(gripper_hold)

            if stop_when_pinched:
                currently_pinched = env._is_pinched()
                if currently_pinched and pinch_frame is None:
                    pinch_frame = frame + 1
                    pinch_start_positions = actual.copy()
                    frame_count = max(
                        frame_count,
                        pinch_frame + PINCH_COMPRESSION_FRAMES,
                    )
                    print(
                        f"  two-finger pinch detected after {pinch_frame} "
                        "close frames; "
                        f"q={actual[-1]:.4f}, "
                        f"target={commanded[-1]:.4f}, "
                        f"qvel={data.qvel[dof_ids[-1]]:.4f}; "
                        "continuing to the pinch target"
                    )
                elif (
                    pinch_frame is not None
                    and not both_finger_meshes_touch_ball()
                ):
                    print("  abort: actual finger-mesh contact was lost while closing")
                    print(
                        "  closure slip state: ball=",
                        np.round(ball_position(), 5),
                        "jaw=",
                        np.round(data.qpos[qpos_ids], 4),
                    )
                    for contact_index in range(data.ncon):
                        contact = data.contact[contact_index]
                        pair = {int(contact.geom1), int(contact.geom2)}
                        if env.ball_geom in pair and pair.intersection(
                            set().union(*env.finger_geom_groups)
                        ):
                            print(
                                "  closure contact:",
                                np.round(contact.pos, 5),
                                "distance=",
                                round(float(contact.dist), 6),
                            )
                    return False
                elif (
                    pinch_frame is not None
                    and frame + 1 - pinch_frame >= PINCH_COMPRESSION_FRAMES
                ):
                    print(
                        "  completed pinch preload over frames:",
                        PINCH_COMPRESSION_FRAMES,
                    )
                    return True

            if terminated or truncated:
                print(f"  abort: episode ended during {stage}")
                return False

            if require_two_finger_contact and not grip_ready:
                grip_recovery_frames += 1
                if grip_recovery_frames >= MAX_GRIP_RECOVERY_FRAMES:
                    print(
                        f"  abort: {stage}: bilateral finger force stayed "
                        f"below {MIN_GRIP_NORMAL_FORCE:.2f} N for "
                        f"{grip_recovery_frames} recovery frames; arm held "
                        "position and both actual finger contacts remained "
                        "required"
                    )
                    return False
                # Do not advance the arm path until both real finger meshes
                # have recovered the minimum pinch force.
                continue
            grip_recovery_frames = 0

            frame += 1

        if contact_tracking is None:
            settle_target = target_positions
            error = float(
                np.max(np.abs(data.qpos[qpos_ids] - settle_target))
            )
        else:
            try:
                settle_target = command_at_fraction(1.0)
            except RuntimeError as error:
                print(f"  abort: {stage}: {error}")
                return False
            error = float(
                np.max(np.abs(data.qpos[qpos_ids] - settle_target))
            )
        settle_frames = 0
        grip_recovery_frames = 0
        desired_settle_target = settle_target.copy()
        while (
            error > 0.03
            and settle_frames < MAX_WAYPOINT_SETTLE_FRAMES
        ):
            grip_ready = True
            step_target = desired_settle_target
            if require_two_finger_contact:
                grip_ready = gripper_force_is_sufficient()
                if not grip_ready:
                    maintain_gripper_force(gripper_hold)
                    step_target = data.qpos[qpos_ids].copy()
                    step_target[-1] = gripper_hold["target"]
            observation, _, terminated, truncated, _ = env.step(
                action_for(step_target)
            )
            capture(observation)
            settle_frames += 1

            actual = data.qpos[qpos_ids].copy()
            if not checker.pose_is_collision_free(actual):
                print(f"  abort: jaw/table collision while settling {stage}")
                return False

            if require_two_finger_contact and not both_finger_meshes_touch_ball():
                grip_recovery_frames += 1
                if grip_recovery_frames >= MAX_GRIP_RECOVERY_FRAMES:
                    print(
                        f"  abort: actual finger-mesh contact could not be "
                        f"recovered while settling {stage}"
                    )
                    print_finger_contact_state("remaining real finger contacts")
                    return False
                maintain_gripper_force(gripper_hold)
                continue
            if require_two_finger_contact:
                maintain_gripper_force(gripper_hold)

            shift = np.linalg.norm(ball_position() - ball_start)
            if check_ball_shift and shift > MAX_APPROACH_BALL_SHIFT:
                print(
                    f"  abort: ball shifted {shift * 1000:.1f} mm "
                    f"while settling {stage}"
                )
                return False

            if terminated or truncated:
                print(f"  abort: episode ended while settling {stage}")
                return False

            if require_two_finger_contact and not grip_ready:
                grip_recovery_frames += 1
                if grip_recovery_frames >= MAX_GRIP_RECOVERY_FRAMES:
                    print(
                        f"  abort: {stage}: bilateral finger force stayed "
                        f"below {MIN_GRIP_NORMAL_FORCE:.2f} N while "
                        "settling; arm held position"
                    )
                    return False
                continue
            grip_recovery_frames = 0

            if contact_tracking is None:
                settle_target = target_positions
            else:
                try:
                    settle_target = command_at_fraction(1.0)
                except RuntimeError as error:
                    print(f"  abort: {stage}: {error}")
                    return False
            error = float(np.max(np.abs(actual - settle_target)))
            desired_settle_target = settle_target.copy()

        if error > 0.03:
            print(
                f"  abort: {stage} joint tracking error="
                f"{error:.3f} rad after {settle_frames} settle frames"
            )
            return False
        return True

    def rotation_vector(target_rotation, current_rotation):
        relative = target_rotation @ current_rotation.T
        cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
        angle = float(np.arccos(cosine))
        skew = np.array(
            [
                relative[2, 1] - relative[1, 2],
                relative[0, 2] - relative[2, 0],
                relative[1, 0] - relative[0, 1],
            ]
        )
        if angle < 1e-7:
            return 0.5 * skew
        return (angle / (2.0 * np.sin(angle))) * skew

    def solve_cartesian_waypoint(
        target_xyz,
        seed_positions,
        fixed_wrist_gripper,
        target_rotation=None,
        target_axis=None,
    ):
        """Solve XYZ, optionally preserving the gripper's world orientation."""
        saved_qpos = data.qpos.copy()
        data.qpos[qpos_ids] = seed_positions
        if target_rotation is None:
            data.qpos[qpos_ids[3:6]] = fixed_wrist_gripper
        else:
            data.qpos[qpos_ids[5]] = fixed_wrist_gripper[2]

        try:
            for _ in range(150):
                mujoco.mj_forward(model, data)
                error = target_xyz - data.site_xpos[env.gripper_site]
                jac_pos = np.zeros((3, model.nv))
                jac_rot = np.zeros((3, model.nv))
                mujoco.mj_jacSite(
                    model, data, jac_pos, jac_rot, env.gripper_site
                )
                if target_rotation is None:
                    if np.linalg.norm(error) < 0.001:
                        break
                    jac = jac_pos[:, dof_ids[:3]]
                    delta = jac.T @ np.linalg.solve(
                        jac @ jac.T + 0.05**2 * np.eye(3), error
                    )
                    controlled_joint_count = 3
                else:
                    rotation_error = rotation_vector(
                        target_rotation,
                        data.site_xmat[env.gripper_site].reshape(3, 3),
                    )
                    rotation_projector = np.eye(3)
                    if target_axis is not None:
                        rotation_projector -= np.outer(
                            target_axis, target_axis
                        )
                    rotation_error = rotation_projector @ rotation_error
                    if (
                        np.linalg.norm(error) < 0.001
                        and np.linalg.norm(rotation_error) < 0.01
                    ):
                        break

                    rotation_weight = 0.08
                    task_jacobian = np.vstack(
                        (
                            jac_pos[:, dof_ids[:5]],
                            rotation_weight
                            * rotation_projector
                            @ jac_rot[:, dof_ids[:5]],
                        )
                    )
                    task_error = np.concatenate(
                        (error, rotation_weight * rotation_error)
                    )
                    delta = task_jacobian.T @ np.linalg.solve(
                        task_jacobian @ task_jacobian.T
                        + 0.05**2 * np.eye(6),
                        task_error,
                    )
                    controlled_joint_count = 5

                length = np.linalg.norm(delta)
                if length > 0.05:
                    delta *= 0.05 / length

                for joint_index in range(controlled_joint_count):
                    qpos_index = qpos_ids[joint_index]
                    joint_id = mujoco.mj_name2id(
                        model,
                        mujoco.mjtObj.mjOBJ_JOINT,
                        JOINT_NAMES[joint_index],
                    )
                    low, high = model.jnt_range[joint_id]
                    data.qpos[qpos_index] = np.clip(
                        data.qpos[qpos_index] + delta[joint_index],
                        low,
                        high,
                    )

            mujoco.mj_forward(model, data)
            solution = data.qpos[qpos_ids].copy()
            position_residual = np.linalg.norm(
                target_xyz - data.site_xpos[env.gripper_site]
            )
            if target_rotation is None:
                residual = float(position_residual)
            else:
                orientation_residual = np.linalg.norm(
                    rotation_vector(
                        target_rotation,
                        data.site_xmat[env.gripper_site].reshape(3, 3),
                    )
                )
                if target_axis is not None:
                    orientation_residual = np.linalg.norm(
                        (np.eye(3) - np.outer(target_axis, target_axis))
                        @ rotation_vector(
                            target_rotation,
                            data.site_xmat[env.gripper_site].reshape(3, 3),
                        )
                    )
                residual = float(
                    max(position_residual, 0.08 * orientation_residual)
                )
            return solution, residual
        finally:
            data.qpos[:] = saved_qpos
            mujoco.mj_forward(model, data)

    def test_candidate(candidate_index, candidate):
        (
            total_error,
            wrist_flex,
            wrist_roll,
            offset,
            pregrasp_q,
            contact_open_q,
            contact_closed_q,
            contact_opposition,
            contact_heights,
        ) = candidate

        # Candidate poses were searched using seed 0, so reset to seed 0 for
        # every attempt: same scene, fresh simulation state.
        if (
            candidate_index == VIDEO_CANDIDATE_INDEX
            and RECORD_CANDIDATE_VIDEO
            and env.renderer is None
        ):
            env.renderer = mujoco.Renderer(model, height=224, width=224)

        observation, _ = env.reset(seed=DIAGNOSTIC_SEED)
        contact_integrity["lost"] = False
        frames = []
        captured_steps = 0

        def capture(observation):
            nonlocal captured_steps
            if (
                candidate_index != VIDEO_CANDIDATE_INDEX
                or not RECORD_CANDIDATE_VIDEO
                or observation["image"] is None
            ):
                return
            captured_steps += 1
            if captured_steps % 20 == 0 or not frames:
                frames.append(observation["image"].copy())

        capture(observation)
        mujoco.mj_forward(model, data)
        ball_start = ball_position()

        print(
            f"\nCandidate {candidate_index}: "
            f"wrist_flex={wrist_flex:+.3f}, "
            f"wrist_roll={wrist_roll:+.3f}, "
            f"offset={np.round(offset, 3)}, "
            f"IK error sum={total_error * 1000:.1f} mm, "
            f"static contact opposition={contact_opposition:.3f}, "
            f"contact z offsets={np.round(contact_heights, 4)}"
        )
        print("  ball start:", np.round(ball_start, 4))

        result = {
            "success": False,
            "pick_success": False,
            "place_success": False,
            "reason": "not completed",
            "ball_displacement_mm": 0.0,
            "ball_lift_mm": 0.0,
            "ball_z": float(ball_start[2]),
        }
        gripper_hold = {"target": None, "floor": None}

        ball_lift_reference = ball_start

        def record_metrics():
            ball_end = ball_position()
            result["ball_displacement_mm"] = float(
                np.linalg.norm(ball_end - ball_start) * 1000.0
            )
            result["ball_lift_mm"] = float(
                (ball_end[2] - ball_lift_reference[2]) * 1000.0
            )
            result["ball_z"] = float(ball_end[2])

        start_q = data.qpos[qpos_ids].copy()
        if not checker.candidate_is_safe(contact_open_q, contact_closed_q):
            result["reason"] = "closing path failed collision check"
            record_metrics()
            return result

        def move_cartesian_path(
            start_xyz,
            target_xyz,
            fixed_wrist_gripper,
            waypoint_count,
            stage,
            start_wrist_gripper=None,
        ):
            for waypoint_index in range(1, waypoint_count + 1):
                fraction = waypoint_index / waypoint_count
                waypoint_xyz = start_xyz + fraction * (
                    target_xyz - start_xyz
                )
                waypoint_wrist_gripper = fixed_wrist_gripper
                if start_wrist_gripper is not None:
                    waypoint_wrist_gripper = start_wrist_gripper + fraction * (
                        fixed_wrist_gripper - start_wrist_gripper
                    )
                actual_start = data.qpos[qpos_ids].copy()
                waypoint_q, ik_error = solve_cartesian_waypoint(
                    waypoint_xyz, actual_start, waypoint_wrist_gripper
                )
                if ik_error > 0.012:
                    print(
                        f"  abort: {stage} IK error at waypoint "
                        f"{waypoint_index} is {ik_error * 1000:.1f} mm"
                    )
                    return False
                if not checker.candidate_is_safe(actual_start, waypoint_q):
                    print(
                        f"  abort: jaw/table path check failed during "
                        f"{stage} at waypoint {waypoint_index}; "
                        f"contact={checker.last_collision}, "
                        f"q={np.round(checker.last_collision_qpos, 4) if checker.last_collision_qpos is not None else None}"
                    )
                    return False
                max_joint_change = float(
                    np.max(np.abs(waypoint_q - actual_start))
                )
                frame_count = max(
                    MIN_FRAMES_PER_APPROACH_WAYPOINT,
                    int(np.ceil(
                        max_joint_change / RADIANS_PER_APPROACH_FRAME
                    )),
                )
                if not move_to(
                    waypoint_q,
                    frame_count,
                    f"{stage} waypoint {waypoint_index}",
                    ball_start,
                    capture,
                ):
                    print(
                        f"  failed waypoint target_xyz="
                        f"{np.round(waypoint_xyz, 5)}, "
                        f"IK_error={ik_error * 1000:.1f} mm, "
                        f"target_q={np.round(waypoint_q, 4)}, "
                        f"actual_q={np.round(data.qpos[qpos_ids], 4)}, "
                        f"actual_site={np.round(data.site_xpos[env.gripper_site], 5)}"
                    )
                    return False
            return True

        def move_grasped_ball_path(
            start_xyz,
            target_xyz,
            fixed_wrist_gripper,
            target_rotation,
            target_axis,
            waypoint_count,
            stage,
            require_two_finger_contact,
        ):
            site_to_ball = start_xyz - ball_position()
            for waypoint_index in range(1, waypoint_count + 1):
                fraction = waypoint_index / waypoint_count
                waypoint_xyz = start_xyz + fraction * (
                    target_xyz - start_xyz
                )
                actual_start = data.qpos[qpos_ids].copy()
                waypoint_q, ik_error = solve_cartesian_waypoint(
                    waypoint_xyz,
                    actual_start,
                    fixed_wrist_gripper,
                    target_rotation=target_rotation,
                    target_axis=target_axis,
                )
                if ik_error > 0.003:
                    print(
                        f"  abort: {stage} IK error at waypoint "
                        f"{waypoint_index} is {ik_error * 1000:.1f} mm"
                    )
                    return False
                if not checker.candidate_is_safe(actual_start, waypoint_q):
                    print(
                        f"  abort: jaw/table path check failed during "
                        f"{stage} at waypoint {waypoint_index}; "
                        f"contact={checker.last_collision}, "
                        f"q={np.round(checker.last_collision_qpos, 4) if checker.last_collision_qpos is not None else None}"
                    )
                    return False

                max_joint_change = float(
                    np.max(np.abs(waypoint_q - actual_start))
                )
                frame_count = max(
                    MIN_FRAMES_PER_PLACE_WAYPOINT,
                    int(np.ceil(
                        max_joint_change / RADIANS_PER_APPROACH_FRAME
                    )),
                )
                if not move_to(
                    waypoint_q,
                    frame_count,
                    f"{stage} waypoint {waypoint_index}",
                    ball_start,
                    capture,
                    check_ball_shift=False,
                    require_two_finger_contact=require_two_finger_contact,
                    contact_tracking={
                        "start_site": data.site_xpos[env.gripper_site].copy(),
                        "target_site": waypoint_xyz.copy(),
                        "site_to_ball": site_to_ball.copy(),
                        "target_rotation": target_rotation,
                        "target_axis": target_axis,
                    },
                    gripper_hold=(
                        gripper_hold if require_two_finger_contact else None
                    ),
                ):
                    return False
            return True

        fixed_open_wrist_gripper = pregrasp_q[3:6].copy()
        reset_wrist_gripper = data.qpos[qpos_ids[3:6]].copy()
        start_site = data.site_xpos[env.gripper_site].copy()
        vertical_clearance = start_site + np.array(
            [0.0, 0.0, APPROACH_CLEARANCE_METERS]
        )
        contact_xyz = ball_start + offset
        side_approach_xyz = contact_xyz + np.array([0.0, 0.0, 0.03])
        transit_xyz = side_approach_xyz + np.array([0.0, 0.0, 0.06])
        transit_xyz[2] = max(transit_xyz[2], vertical_clearance[2])

        if not move_cartesian_path(
            start_site,
            vertical_clearance,
            reset_wrist_gripper,
            8,
            "vertical clearance",
        ):
            result["reason"] = "vertical clearance motion failed"
            record_metrics()
            return result
        current_site = data.site_xpos[env.gripper_site].copy()
        if not move_cartesian_path(
            current_site,
            transit_xyz,
            fixed_open_wrist_gripper,
            APPROACH_WAYPOINTS,
            "high transit",
            start_wrist_gripper=reset_wrist_gripper,
        ):
            result["reason"] = "high transit motion failed"
            record_metrics()
            return result
        current_site = data.site_xpos[env.gripper_site].copy()
        if not move_cartesian_path(
            current_site,
            side_approach_xyz,
            fixed_open_wrist_gripper,
            APPROACH_WAYPOINTS,
            "vertical descent above ball",
        ):
            result["reason"] = "pregrasp descent failed"
            record_metrics()
            return result
        current_site = data.site_xpos[env.gripper_site].copy()
        if not move_cartesian_path(
            current_site,
            side_approach_xyz,
            contact_open_q[3:6],
            1,
            "pre-close beside ball",
            start_wrist_gripper=fixed_open_wrist_gripper,
        ):
            result["reason"] = "side pre-close failed"
            record_metrics()
            return result
        current_site = data.site_xpos[env.gripper_site].copy()
        if not move_cartesian_path(
            current_site,
            contact_xyz,
            contact_open_q[3:6],
            APPROACH_WAYPOINTS,
            "final vertical open-jaw approach",
        ):
            result["reason"] = "contact descent failed"
            record_metrics()
            return result
        print(
            "  at contact before closure: ball=",
            np.round(ball_position(), 5),
            "site=",
            np.round(data.site_xpos[env.gripper_site], 5),
            "fixed_tip=",
            np.round(data.geom_xpos[env.fixed_finger_geom], 5),
            "moving_tip=",
            np.round(data.geom_xpos[env.moving_finger_geom], 5),
        )
        if not move_to(
            contact_closed_q,
            GRIPPER_CLOSE_FRAMES,
            "gripper close",
            ball_start,
            capture,
            stop_when_pinched=True,
        ):
            result["reason"] = "closing motion failed"
            record_metrics()
            return result

        contact_closed_q = data.qpos[qpos_ids].copy()
        # The ball can stop the jaw before its requested preload angle. Keep
        # the actuator target so the gripper continues pressing during lift.
        contact_closed_q[-1] = data.ctrl[env.actuator_ids[-1]]
        gripper_hold["target"] = float(contact_closed_q[-1])
        gripper_hold["floor"] = max(
            float(model.actuator_ctrlrange[env.actuator_ids[-1], 0]),
            gripper_hold["target"] - MAX_FORCE_CONTROL_PRELOAD,
        )
        ball_after_close = ball_position()
        fixed_finger_geom = env.fixed_finger_geom
        moving_finger_geom = env.moving_finger_geom
        finger_geoms = set().union(*env.finger_geom_groups)
        contacted_fingers = set()

        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]

            if env.ball_geom == contact.geom1:
                other_geom = int(contact.geom2)
            elif env.ball_geom == contact.geom2:
                other_geom = int(contact.geom1)
            else:
                continue

            if other_geom in finger_geoms:
                contacted_fingers.add(other_geom)

                contact_force = np.zeros(6)
                mujoco.mj_contactForce(
                    model, data, contact_index, contact_force
                )
                contact_rotation = contact.frame.reshape(3, 3)
                world_contact_force = contact_rotation.T @ contact_force[:3]
                finger_name = mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_GEOM, other_geom
                )
                print(
                    f"  finger-mesh contact {finger_name}: "
                    f"distance={contact.dist:.6f} m, "
                    f"normal_force={contact_force[0]:.4f} N, "
                    f"point={np.round(contact.pos, 4)}, "
                    f"normal={np.round(contact.frame[:3], 3)}, "
                    f"world_force_on_ball={np.round(world_contact_force, 4)}, "
                    f"pair=({mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1))}, "
                    f"{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2))})"
                )

        pinched_after_close = bool(env._is_pinched())
        ball_lift_reference = ball_after_close
        print(
            "  after close:",
            f"pinched={pinched_after_close}",
            f"ball={np.round(ball_after_close, 4)}",
        )
        print(
            "  ball velocity/acceleration:",
            np.round(data.qvel[env.ball_qvel:env.ball_qvel + 3], 5),
            np.round(data.qacc[env.ball_qvel:env.ball_qvel + 3], 4),
        )
        print(
            "  ball constraint force:",
            np.round(data.qfrc_constraint[env.ball_qvel:env.ball_qvel + 3], 5),
        )
        table_geom = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "table_top"
        )
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if env.ball_geom in pair and table_geom in pair:
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, contact_index, force)
                print(
                    "  ball/table contact:",
                    "distance=",
                    round(float(contact.dist), 6),
                    "normal_force=",
                    round(float(force[0]), 5),
                )

        if not pinched_after_close:
            result["reason"] = "no two-jaw pinch after closing"
            record_metrics()
            return result

        if not move_to(
            contact_closed_q,
            PINCH_SETTLE_FRAMES,
            "pinch settle",
            ball_start,
            capture,
            check_ball_shift=False,
            require_two_finger_contact=True,
            gripper_hold=gripper_hold,
        ):
            result["reason"] = "pinch did not settle before lift"
            record_metrics()
            return result
        print(
            "  after pinch settle: ball=",
            np.round(ball_position(), 5),
            "velocity=",
            np.round(data.qvel[env.ball_qvel:env.ball_qvel + 3], 5),
            "gripper_q/target=",
            round(float(data.qpos[qpos_ids[-1]]), 5),
            round(float(data.ctrl[env.actuator_ids[-1]]), 5),
        )
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if env.ball_geom not in pair or not pair.intersection(finger_geoms):
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, contact_index, force)
            print(
                "  settled mesh contact:",
                [
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
                    for geom in (contact.geom1, contact.geom2)
                ],
                "distance=", round(float(contact.dist), 6),
                "normal_force=", round(float(force[0]), 5),
                "point=", np.round(contact.pos, 5),
            )

        lift_start_site = data.site_xpos[env.gripper_site].copy()
        lift_start_ball = ball_position()
        fixed_wrist_gripper = contact_closed_q[3:6].copy()
        fixed_wrist_gripper[-1] = gripper_hold["target"]
        target_gripper_rotation = data.site_xmat[env.gripper_site].reshape(
            3, 3
        ).copy()
        fixed_contact = actual_contact_point(env.fixed_finger_geoms)
        moving_contact = actual_contact_point(env.moving_finger_geoms)
        jaw_axis = fixed_contact - moving_contact
        jaw_axis /= np.linalg.norm(jaw_axis)
        fixed_finger_axis = data.geom_xmat[env.fixed_finger_geom].reshape(
            3, 3
        )[:, 2]
        moving_finger_axis = data.geom_xmat[env.moving_finger_geom].reshape(
            3, 3
        )[:, 2]
        print(
            "  world jaw/fixed-finger/moving-finger axes:",
            np.round(jaw_axis, 4),
            np.round(fixed_finger_axis, 4),
            np.round(moving_finger_axis, 4),
        )
        pinch_persisted = True
        lift_aborted = False

        for waypoint_index in range(1, LIFT_WAYPOINTS + 1):
            target_xyz = lift_start_site + np.array(
                [0.0, 0.0, LIFT_METERS * waypoint_index / LIFT_WAYPOINTS]
            )
            actual_start = data.qpos[qpos_ids].copy()
            waypoint_q, ik_error = solve_cartesian_waypoint(
                target_xyz,
                actual_start,
                fixed_wrist_gripper,
                target_rotation=target_gripper_rotation,
                target_axis=jaw_axis,
            )

            if ik_error > 0.003:
                print(
                    f"  abort: lift IK error at waypoint {waypoint_index} "
                    f"is {ik_error * 1000:.1f} mm"
                )
                result["reason"] = "lift IK failed"
                lift_aborted = True
                break

            if not checker.candidate_is_safe(actual_start, waypoint_q):
                print(
                    f"  abort: collision check failed at lift waypoint "
                    f"{waypoint_index}; contact={checker.last_collision}, "
                    f"q={np.round(checker.last_collision_qpos, 4) if checker.last_collision_qpos is not None else None}"
                )
                result["reason"] = "lift collision check failed"
                lift_aborted = True
                break

            substep = 0
            grip_recovery_frames = 0
            contact_recovery_frames = 0
            arm_hold_target = data.qpos[qpos_ids].copy()
            while substep < FRAMES_PER_LIFT_WAYPOINT:
                grip_ready = gripper_force_is_sufficient()
                if not grip_ready:
                    maintain_gripper_force(gripper_hold)
                fraction = (substep + 1) / FRAMES_PER_LIFT_WAYPOINT
                lift_progress = (
                    waypoint_index - 1 + fraction
                ) / LIFT_WAYPOINTS
                nominal_target = lift_start_site + np.array(
                    [0.0, 0.0, LIFT_METERS * lift_progress]
                )
                nominal_ball = lift_start_ball + np.array(
                    [0.0, 0.0, LIFT_METERS * lift_progress]
                )
                actual_now = data.qpos[qpos_ids].copy()
                if grip_ready:
                    ball_error = ball_position() - nominal_ball
                    # Keep the lift trajectory fixed to the arm's planned
                    # motion. A correction from the freely moving ball can
                    # relax the pinch while the gripper is trying to lift.
                    correction = np.zeros(3)
                    correction_norm = float(np.linalg.norm(correction))
                    if correction_norm > 0.020:
                        correction *= 0.020 / correction_norm
                    commanded, tracking_error = solve_cartesian_waypoint(
                        nominal_target + correction,
                        actual_now,
                        fixed_wrist_gripper,
                        target_rotation=target_gripper_rotation,
                        target_axis=jaw_axis,
                    )
                    if tracking_error > 0.006:
                        print(
                            f"  abort: contact-follow lift IK error="
                            f"{tracking_error * 1000:.1f} mm at waypoint "
                            f"{waypoint_index}, substep {substep}"
                        )
                        result["reason"] = "contact-follow lift IK failed"
                        lift_aborted = True
                        break
                else:
                    # Hold the last arm target while the jaws recover. A
                    # measured-joint target would follow any gravity droop.
                    commanded = arm_hold_target.copy()
                commanded[-1] = gripper_hold["target"]
                if grip_ready:
                    arm_hold_target = commanded.copy()
                observation, _, terminated, truncated, _ = env.step(
                    action_for(commanded)
                )
                capture(observation)

                if not checker.pose_is_collision_free(data.qpos[qpos_ids]):
                    result["reason"] = "jaw/table collision during lift"
                    lift_aborted = True
                    break

                # If one real jaw surface slips, stop arm motion and squeeze
                # gradually until both physical sides touch again. Do not
                # advance the lift path while a contact is missing.
                if not both_finger_meshes_touch_ball():
                    contact_integrity["lost"] = True
                    contact_recovery_frames += 1
                    if contact_recovery_frames == 1:
                        print(
                            f"  contact slipped at lift waypoint "
                            f"{waypoint_index}; holding the arm and closing "
                            "the real gripper to recover both sides"
                        )
                    maintain_gripper_force(gripper_hold)
                    if contact_recovery_frames >= MAX_GRIP_RECOVERY_FRAMES:
                        print(
                            "  slip state: ball=",
                            np.round(ball_position(), 5),
                            "jaw=",
                            np.round(data.qpos[qpos_ids], 4),
                            "site=",
                            np.round(data.site_xpos[env.gripper_site], 5),
                            "target=",
                            np.round(target_xyz, 5),
                        )
                        print_finger_contact_state("remaining real contacts")
                        result["reason"] = (
                            "actual finger-mesh contact could not be "
                            "recovered during lift"
                        )
                        pinch_persisted = False
                        lift_aborted = True
                        break
                    continue
                contact_recovery_frames = 0
                maintain_gripper_force(gripper_hold)
                fixed_wrist_gripper[-1] = gripper_hold["target"]

                if terminated or truncated:
                    result["reason"] = "episode ended during lift"
                    lift_aborted = True
                    break

                ball_now = ball_position()
                lateral_shift = np.linalg.norm(
                    ball_now[:2] - ball_after_close[:2]
                )
                if lateral_shift > MAX_LIFT_LATERAL_SHIFT:
                    print(
                        f"  abort: lateral ball shift during lift="
                        f"{lateral_shift * 1000:.1f} mm at waypoint "
                        f"{waypoint_index}, substep {substep}; "
                        f"ball={np.round(ball_now, 5)}, "
                        f"site={np.round(data.site_xpos[env.gripper_site], 5)}, "
                        f"target={np.round(target_xyz, 5)}, "
                        f"q={np.round(data.qpos[qpos_ids], 4)}"
                    )
                    for contact_index in range(data.ncon):
                        contact = data.contact[contact_index]
                        pair = {int(contact.geom1), int(contact.geom2)}
                        if env.ball_geom not in pair or not pair.intersection(
                            finger_geoms
                        ):
                            continue
                        force = np.zeros(6)
                        mujoco.mj_contactForce(
                            model, data, contact_index, force
                        )
                        print(
                            "  lateral-slip contact:",
                            np.round(contact.pos, 5),
                            "distance=", round(float(contact.dist), 6),
                            "force(normal, tangent)=", np.round(force[:3], 5),
                        )
                    result["reason"] = "ball slid sideways during lift"
                    lift_aborted = True
                    break

                if not grip_ready:
                    grip_recovery_frames += 1
                    if grip_recovery_frames >= MAX_GRIP_RECOVERY_FRAMES:
                        result["reason"] = (
                            "bilateral finger force did not recover while "
                            "the arm was held during lift"
                        )
                        print(f"  abort: {result['reason']}")
                        lift_aborted = True
                        break
                    continue
                grip_recovery_frames = 0
                substep += 1

            if lift_aborted:
                break

        ball_end = ball_position()
        result["ball_displacement_mm"] = float(
            np.linalg.norm(ball_end - ball_start) * 1000.0
        )
        result["ball_lift_mm"] = float(
            (ball_end[2] - ball_after_close[2]) * 1000.0
        )
        result["ball_z"] = float(ball_end[2])
        pick_success = bool(
            not lift_aborted
            and pinch_persisted
            and env._is_pinched()
            and ball_end[2] > 0.035
        )
        result["pick_success"] = pick_success

        if pick_success:
            cup_position = data.xpos[env.cup_body].copy()
            cup_wall_geom = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, "cup_wall_0"
            )
            cup_top_z = float(
                cup_position[2]
                + model.geom_pos[cup_wall_geom, 2]
                + model.geom_size[cup_wall_geom, 2]
            )
            ball_radius = float(model.geom_size[env.ball_geom, 0])
            site_to_ball = (
                data.site_xpos[env.gripper_site].copy() - ball_position()
            )
            carry_rotation = data.site_xmat[env.gripper_site].reshape(
                3, 3
            ).copy()
            carry_fixed_contact = actual_contact_point(env.fixed_finger_geoms)
            carry_moving_contact = actual_contact_point(env.moving_finger_geoms)
            # Keep the grasp plane upright and allow only world-up yaw as the
            # redundant orientation DOF through the carry path.
            carry_axis = np.array([0.0, 0.0, 1.0])
            # Clear the cup rim by 25 mm. The previous 1 mm clearance was
            # less than the ball's tracking lag and let the rim strike it.
            carry_height = max(
                ball_position()[2], cup_top_z + ball_radius + 0.025
            )
            carry_ball_target = np.array(
                [cup_position[0], cup_position[1], carry_height]
            )
            lifted_ball_target = np.array(
                [ball_position()[0], ball_position()[1], carry_height]
            )
            # Release just above the rim after a verified carry. Lowering the
            # ball while pinched makes it slide down the tapered finger faces.
            release_ball_target = carry_ball_target.copy()
            # Preserve the commanded pinch preload.  The measured jaw angle
            # can lag this target while squeezing the ball; using that lower
            # measured value here would relax the gripper during transport.
            hold_wrist_gripper = fixed_wrist_gripper.copy()
            place_path_ok = True
            release_completed = False

            print(
                "  carrying ball to cup:",
                "cup=", np.round(cup_position, 4),
                f"rim_z={cup_top_z:.4f}",
                "ball target=", np.round(carry_ball_target, 4),
            )
            current_site = data.site_xpos[env.gripper_site].copy()
            if not move_grasped_ball_path(
                current_site,
                lifted_ball_target + site_to_ball,
                hold_wrist_gripper,
                carry_rotation,
                carry_axis,
                20,
                "raise for cup transfer",
                require_two_finger_contact=True,
            ):
                place_path_ok = False
            else:
                current_site = data.site_xpos[env.gripper_site].copy()
                if not move_grasped_ball_path(
                    current_site,
                    carry_ball_target + site_to_ball,
                    hold_wrist_gripper,
                    carry_rotation,
                    carry_axis,
                    48,
                    "horizontal transfer to cup",
                    require_two_finger_contact=True,
                ):
                    place_path_ok = False

            if place_path_ok:
                # Let ball motion settle while the actual jaws remain closed
                # and in bilateral contact before opening over the cup.
                if not move_to(
                    data.qpos[qpos_ids].copy(),
                    100,
                    "settle held ball over cup",
                    ball_start,
                    capture,
                    check_ball_shift=False,
                    require_two_finger_contact=True,
                    gripper_hold=gripper_hold,
                ):
                    place_path_ok = False

            if place_path_ok:
                ball_at_release = ball_position()
                print(
                    "  at release pose: ball=",
                    np.round(ball_at_release, 5),
                    "target=",
                    np.round(release_ball_target, 5),
                    "cup_xy_error_mm=",
                    round(
                        float(
                            np.linalg.norm(
                                ball_at_release[:2] - cup_position[:2]
                            )
                            * 1000.0
                        ),
                        1,
                    ),
                )

                open_q = data.qpos[qpos_ids].copy()
                open_q[-1] = model.actuator_ctrlrange[
                    env.actuator_ids[-1], 1
                ]
                if not move_to(
                    open_q,
                    200,
                    "open gripper over cup",
                    ball_start,
                    capture,
                    check_ball_shift=False,
                ):
                    place_path_ok = False
                else:
                    release_completed = True

            # The ball may land and begin accumulating stable frames while
            # the 200-frame gripper-opening command is still running.
            stable_cup_frames = env.stable_steps if release_completed else 0
            settle_limit = CUP_SETTLE_FRAMES if release_completed else 0
            for _ in range(settle_limit):
                observation, _, terminated, truncated, _ = env.step(
                    action_for(data.qpos[qpos_ids])
                )
                capture(observation)
                ball_speed = float(
                    np.linalg.norm(
                        data.qvel[env.ball_qvel:env.ball_qvel + 3]
                    )
                )
                stable_cup_frames = env.stable_steps
                if stable_cup_frames >= CUP_STABLE_FRAMES:
                    break
                if terminated or truncated:
                    break

            ball_end = ball_position()
            ball_speed = float(
                np.linalg.norm(data.qvel[env.ball_qvel:env.ball_qvel + 3])
            )
            result["place_success"] = bool(
                place_path_ok
                and not contact_integrity["lost"]
                and env._ball_in_cup()
                and ball_speed < 0.05
                and stable_cup_frames >= CUP_STABLE_FRAMES
            )
            result["success"] = result["place_success"]
            result["ball_displacement_mm"] = float(
                np.linalg.norm(ball_end - ball_start) * 1000.0
            )
            result["ball_z"] = float(ball_end[2])
            print(
                "  after release settle: ball=",
                np.round(ball_end, 5),
                f"speed={ball_speed:.4f} m/s",
                f"in_cup={env._ball_in_cup()}",
                f"stable_frames={stable_cup_frames}",
                f"continuous_finger_contact={not contact_integrity['lost']}",
            )
            if result["success"]:
                result["reason"] = (
                    "passed: ball was released and settled in the cup"
                )
            elif not place_path_ok:
                result["reason"] = "cup approach or release motion failed"
            elif not env._ball_in_cup():
                result["reason"] = "ball did not settle inside the cup"
            elif contact_integrity["lost"]:
                result["reason"] = "bilateral gripper contact was interrupted"
            else:
                result["reason"] = "ball did not settle in the cup"
        else:
            result["place_success"] = False
            result["success"] = False
            if result["reason"] == "not completed":
                result["reason"] = "grasp lift did not pass"

        if (
            candidate_index == VIDEO_CANDIDATE_INDEX
            and RECORD_CANDIDATE_VIDEO
            and frames
        ):
            output = Path(
                f"results/gripper_pick_place_candidate_{candidate_index:02d}.gif"
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            images = [Image.fromarray(frame) for frame in frames]
            images[0].save(
                output,
                save_all=True,
                append_images=images[1:],
                duration=50,
                loop=0,
            )
            print("Saved video:", output.resolve())
        return result

    try:
        results = []
        for candidate_index, candidate in enumerate(candidates, start=1):
            result = test_candidate(candidate_index, candidate)
            results.append((candidate_index, result))

            print(
                f"  RESULT candidate {candidate_index}: "
                f"pick_success={result['pick_success']} | "
                f"cup_success={result['place_success']} | "
                f"success={result['success']} | "
                f"ball displacement={result['ball_displacement_mm']:.1f} mm | "
                f"ball lift={result['ball_lift_mm']:.1f} mm | "
                f"final ball z={result['ball_z']:.4f} m | "
                f"{result['reason']}"
            )

            if result["success"]:
                print(f"\nFirst successful candidate: {candidate_index}")
                break
        else:
            print(
                f"\nNo candidate passed in the bounded batch "
                f"of {len(candidates)}."
            )

        print("\nAttempt summary:")
        for candidate_index, result in results:
            print(
                f"  candidate {candidate_index}: "
                f"pick_success={result['pick_success']}, "
                f"cup_success={result['place_success']}, "
                f"success={result['success']}, "
                f"displacement={result['ball_displacement_mm']:.1f} mm, "
                f"lift={result['ball_lift_mm']:.1f} mm, "
                f"final_z={result['ball_z']:.4f} m, "
                f"reason={result['reason']}"
            )

        return 0 if any(result["success"] for _, result in results) else 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
