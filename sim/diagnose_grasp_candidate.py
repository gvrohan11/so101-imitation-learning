from pathlib import Path

import os

import mujoco
import numpy as np
from PIL import Image

from sim.ball_cup_env import JOINT_NAMES, BallCupEnv
from sim.collision_check import JawTableCollisionChecker
from sim.search_grasp_poses import main as search_safe_pinch_candidates

# Test the best static lower-contact candidate selected by the pose search.
MAX_CANDIDATES = 1
DIAGNOSTIC_SEED = int(os.environ.get("SO101_GRASP_SEED", "0"))

# The ball starts near z=0.020 m. Lift the gripper 25 mm so a successful
# grasp has room to raise the ball above the required z=0.035 m threshold.
LIFT_METERS = 0.025
LIFT_WAYPOINTS = 25
FRAMES_PER_LIFT_WAYPOINT = 30

# Abort an approach if it pushes the ball too far.
MAX_APPROACH_BALL_SHIFT = 0.012  # meters
MAX_LIFT_LATERAL_SHIFT = 0.025  # meters
APPROACH_CLEARANCE_METERS = 0.08
APPROACH_WAYPOINTS = 20
MIN_FRAMES_PER_APPROACH_WAYPOINT = 50
RADIANS_PER_APPROACH_FRAME = 0.009
MAX_WAYPOINT_SETTLE_FRAMES = 200
PINCH_COMPRESSION_FRAMES = 40
PINCH_PRELOAD_RADIANS = 0.06
PINCH_SETTLE_FRAMES = 300
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

    env = BallCupEnv(render_images=False, frame_skip=1, horizon=10000)
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

    def move_to(
        target_positions,
        frame_count,
        stage,
        ball_start,
        capture,
        stop_when_pinched=False,
    ):
        """Move to a pose, freezing the jaw at first stable pinch contact."""
        start_positions = data.qpos[qpos_ids].copy()
        pinch_frame = None
        pinch_start_positions = None
        initial_frame_count = frame_count
        frame = 0

        while frame < frame_count:
            if pinch_start_positions is None:
                fraction = (frame + 1) / initial_frame_count
                commanded = start_positions + fraction * (
                    target_positions - start_positions
                )
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
            observation, _, terminated, truncated, _ = env.step(
                action_for(commanded)
            )
            capture(observation)

            actual = data.qpos[qpos_ids].copy()
            if not checker.pose_is_collision_free(actual):
                print(f"  abort: jaw/table collision during {stage}")
                return False

            shift = np.linalg.norm(ball_position() - ball_start)
            if shift > MAX_APPROACH_BALL_SHIFT:
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
                elif pinch_frame is not None and not currently_pinched:
                    print("  abort: pinch was lost while holding contact")
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
                            {env.fixed_finger_geom, env.moving_finger_geom}
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

            frame += 1

        error = float(
            np.max(np.abs(data.qpos[qpos_ids] - target_positions))
        )
        settle_frames = 0
        while (
            error > 0.03
            and settle_frames < MAX_WAYPOINT_SETTLE_FRAMES
        ):
            observation, _, terminated, truncated, _ = env.step(
                action_for(target_positions)
            )
            capture(observation)
            settle_frames += 1

            actual = data.qpos[qpos_ids].copy()
            if not checker.pose_is_collision_free(actual):
                print(f"  abort: jaw/table collision while settling {stage}")
                return False

            shift = np.linalg.norm(ball_position() - ball_start)
            if shift > MAX_APPROACH_BALL_SHIFT:
                print(
                    f"  abort: ball shifted {shift * 1000:.1f} mm "
                    f"while settling {stage}"
                )
                return False

            if terminated or truncated:
                print(f"  abort: episode ended while settling {stage}")
                return False

            error = float(
                np.max(np.abs(actual - target_positions))
            )

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
            if captured_steps % 5 == 0 or not frames:
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
            "reason": "not completed",
            "ball_displacement_mm": 0.0,
            "ball_lift_mm": 0.0,
            "ball_z": float(ball_start[2]),
        }

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
                    return False
            return True

        fixed_open_wrist_gripper = pregrasp_q[3:6].copy()
        reset_wrist_gripper = data.qpos[qpos_ids[3:6]].copy()
        start_site = data.site_xpos[env.gripper_site].copy()
        vertical_clearance = start_site + np.array(
            [0.0, 0.0, APPROACH_CLEARANCE_METERS]
        )
        contact_xyz = ball_start + offset
        side_approach_xyz = contact_xyz + np.array([0.0, 0.05, 0.0])
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
            "descent beside ball",
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
            "horizontal open-jaw approach",
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
            300,
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
        ball_after_close = ball_position()
        fixed_finger_geom = env.fixed_finger_geom
        moving_finger_geom = env.moving_finger_geom
        finger_geoms = {fixed_finger_geom, moving_finger_geom}
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
        ):
            result["reason"] = "pinch did not settle before lift"
            record_metrics()
            return result
        print(
            "  after pinch settle: ball=",
            np.round(ball_position(), 5),
            "velocity=",
            np.round(data.qvel[env.ball_qvel:env.ball_qvel + 3], 5),
        )

        lift_start_site = data.site_xpos[env.gripper_site].copy()
        fixed_wrist_gripper = contact_closed_q[3:6].copy()
        target_gripper_rotation = data.site_xmat[env.gripper_site].reshape(
            3, 3
        ).copy()
        fixed_contact = next(
            data.contact[index].pos.copy()
            for index in range(data.ncon)
            if env.ball_geom in (
                data.contact[index].geom1,
                data.contact[index].geom2,
            )
            and env.fixed_finger_geom in (
                data.contact[index].geom1,
                data.contact[index].geom2,
            )
        )
        moving_contact = next(
            data.contact[index].pos.copy()
            for index in range(data.ncon)
            if env.ball_geom in (
                data.contact[index].geom1,
                data.contact[index].geom2,
            )
            and env.moving_finger_geom in (
                data.contact[index].geom1,
                data.contact[index].geom2,
            )
        )
        jaw_axis = fixed_contact - moving_contact
        jaw_axis /= np.linalg.norm(jaw_axis)
        horizontal_jaw_axis = jaw_axis.copy()
        horizontal_jaw_axis[2] = 0.0
        horizontal_jaw_axis /= np.linalg.norm(horizontal_jaw_axis)
        alignment_axis = np.cross(jaw_axis, horizontal_jaw_axis)
        alignment_sine = np.linalg.norm(alignment_axis)
        alignment_cosine = float(np.clip(jaw_axis @ horizontal_jaw_axis, -1, 1))
        if alignment_sine > 1e-8:
            alignment_axis /= alignment_sine
            skew = np.array(
                [
                    [0.0, -alignment_axis[2], alignment_axis[1]],
                    [alignment_axis[2], 0.0, -alignment_axis[0]],
                    [-alignment_axis[1], alignment_axis[0], 0.0],
                ]
            )
            alignment = (
                np.eye(3)
                + alignment_sine * skew
                + (1.0 - alignment_cosine) * (skew @ skew)
            )
            target_gripper_rotation = alignment @ target_gripper_rotation
        fixed_pad_axis = data.geom_xmat[env.fixed_finger_geom].reshape(
            3, 3
        )[:, 2]
        moving_pad_axis = data.geom_xmat[env.moving_finger_geom].reshape(
            3, 3
        )[:, 2]
        print(
            "  world jaw/fixed-pad/moving-pad axes:",
            np.round(jaw_axis, 4),
            np.round(fixed_pad_axis, 4),
            np.round(moving_pad_axis, 4),
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
                target_axis=horizontal_jaw_axis,
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
                    f"{waypoint_index}"
                )
                result["reason"] = "lift collision check failed"
                lift_aborted = True
                break

            for substep in range(FRAMES_PER_LIFT_WAYPOINT):
                fraction = (substep + 1) / FRAMES_PER_LIFT_WAYPOINT
                commanded = actual_start + fraction * (
                    waypoint_q - actual_start
                )
                observation, _, terminated, truncated, _ = env.step(
                    action_for(commanded)
                )
                capture(observation)

                if not checker.pose_is_collision_free(data.qpos[qpos_ids]):
                    result["reason"] = "jaw/table collision during lift"
                    lift_aborted = True
                    break

                # Require both jaw contacts to remain present throughout lift.
                if not env._is_pinched():
                    print(
                        f"  pinch lost at lift waypoint {waypoint_index}, "
                        f"substep {substep}"
                    )
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
                            "  slip contact:",
                            np.round(contact.pos, 5),
                            "distance=",
                            round(float(contact.dist), 6),
                            "force(normal, tangent)=",
                            np.round(force[:3], 5),
                        )
                    result["reason"] = "two-jaw pinch lost during lift"
                    pinch_persisted = False
                    lift_aborted = True
                    break

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
                        f"{waypoint_index}, substep {substep}"
                    )
                    result["reason"] = "ball slid sideways during lift"
                    lift_aborted = True
                    break

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

        result["success"] = bool(
            not lift_aborted
            and pinch_persisted
            and env._is_pinched()
            and ball_end[2] > 0.035
        )
        if (
            candidate_index == VIDEO_CANDIDATE_INDEX
            and RECORD_CANDIDATE_VIDEO
            and frames
        ):
            output = Path(
                f"results/finger_mesh_candidate_{candidate_index:02d}.gif"
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
        if result["success"]:
            result["reason"] = "passed: pinch persisted and ball rose above 0.035 m"
        elif result["reason"] == "not completed":
            result["reason"] = "ball did not rise above 0.035 m"

        record_metrics()
        return result

    try:
        results = []
        for candidate_index, candidate in enumerate(candidates, start=1):
            result = test_candidate(candidate_index, candidate)
            results.append((candidate_index, result))

            print(
                f"  RESULT candidate {candidate_index}: "
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
