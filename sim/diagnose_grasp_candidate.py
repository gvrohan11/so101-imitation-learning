import mujoco
import numpy as np

from sim.ball_cup_env import JOINT_NAMES, BallCupEnv
from sim.collision_check import JawTableCollisionChecker
from sim.search_grasp_poses import main as search_safe_pinch_candidates


def main():
    candidates = search_safe_pinch_candidates()
    if not candidates:
        print("No jaw-table-clear static pinch candidate; no motion attempted.")
        return 1

    env = BallCupEnv(render_images=False, frame_skip=1, horizon=5000)
    model, data = env.model, env.data
    try:
        env.reset(seed=0)
        ball_start = data.xpos[env.ball_body].copy()
        checker = JawTableCollisionChecker(env)
        qpos_ids = np.array(
            [env.joint_qpos[name] for name in JOINT_NAMES], dtype=np.int32
        )
        (
            total_error,
            wrist_flex,
            wrist_roll,
            offset,
            pregrasp_q,
            contact_open_q,
            contact_closed_q,
        ) = candidates[0]

        def action_for(joint_positions):
            action = []
            for target, actuator_id in zip(
                joint_positions, env.actuator_ids
            ):
                low, high = model.actuator_ctrlrange[actuator_id]
                action.append(2.0 * (target - low) / (high - low) - 1.0)
            return np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        def move_to(target_positions, frame_count, stage):
            start_positions = data.qpos[qpos_ids].copy()

            for frame in range(frame_count):
                fraction = (frame + 1) / frame_count
                commanded = start_positions + fraction * (
                    target_positions - start_positions
                )
                _, _, terminated, truncated, _ = env.step(
                    action_for(commanded)
                )
                actual_positions = data.qpos[qpos_ids].copy()

                if not checker.pose_is_collision_free(actual_positions):
                    print(
                        f"ABORT: moving jaw/table collision during {stage}."
                    )
                    return False

                ball_shift = np.linalg.norm(
                    data.xpos[env.ball_body] - ball_start
                )
                if ball_shift > 0.012:
                    print(
                        f"ABORT: ball moved {ball_shift * 1000:.1f} mm "
                        f"during {stage}."
                    )
                    return False

                if terminated or truncated:
                    print(f"ABORT: episode ended during {stage}.")
                    return False

            actual_positions = data.qpos[qpos_ids].copy()
            max_joint_error = np.max(
                np.abs(actual_positions - target_positions)
            )
            print(
                f"{stage}: max joint target error="
                f"{max_joint_error:.3f} rad"
            )

            if max_joint_error > 0.03:
                print(f"ABORT: joints did not reach the target for {stage}.")
                return False

            return True

        start_q = data.qpos[qpos_ids].copy()
        if not checker.candidate_is_safe(start_q, pregrasp_q):
            print("ABORT: selected pregrasp pose or path failed collision check.")
            return 1
        if not checker.candidate_is_safe(pregrasp_q, contact_open_q):
            print("ABORT: selected open descent failed collision check.")
            return 1
        if not checker.candidate_is_safe(contact_open_q, contact_closed_q):
            print("ABORT: selected closing motion failed collision check.")
            return 1

        print("Selected jaw-table-clear static candidate:")
        print(f"  wrist_flex={wrist_flex:+.3f} rad")
        print(f"  wrist_roll={wrist_roll:+.3f} rad")
        print(f"  offset={np.round(offset, 4)} m")
        print(f"  IK error sum={total_error * 1000:.1f} mm")

        if not move_to(pregrasp_q, 400, "pregrasp approach"):
            return 1
        if not move_to(contact_open_q, 400, "open-jaw descent"):
            return 1
        if not move_to(contact_closed_q, 300, "gripper closing"):
            return 1

        print("ball contacts immediately after gripper close:")
        found_contact = False
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            if env.ball_geom not in (contact.geom1, contact.geom2):
                continue

            other_geom = (
                contact.geom2
                if contact.geom1 == env.ball_geom
                else contact.geom1
            )
            other_body = int(model.geom_bodyid[other_geom])
            print(
                " ",
                "geom=",
                mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_GEOM, other_geom
                ),
                "body=",
                mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_BODY, other_body
                ),
                "distance=",
                round(float(contact.dist), 5),
            )
            found_contact = True

        if not found_contact:
            print("  no ball contacts")

        print("currently pinched:", env._is_pinched())
        print("ball position after closing:", np.round(data.xpos[env.ball_body], 4))
        print("gripper site after closing:", np.round(
            data.site_xpos[env.gripper_site], 4
        ))

        if not env._is_pinched():
            print("ABORT: ball is not pinched; skipping lift.")
            return 1

        ball_after_close = data.xpos[env.ball_body].copy()
        lift_start = data.qpos[qpos_ids].copy()
        lift_start_site = data.site_xpos[env.gripper_site].copy()
        lift_target = lift_start.copy()
        lift_target[3:6] = contact_closed_q[3:6]
        lift_dof_ids = np.array(
            [model.jnt_dofadr[mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, name
            )] for name in JOINT_NAMES],
            dtype=np.int32,
        )
        fixed_wrist_and_gripper = contact_closed_q[3:6].copy()
        waypoint_count = 60
        frames_per_waypoint = 7
        lost_pinch_steps = 0
        lift_aborted = False

        print("starting cautious 6 cm lift with gripper closed")

        def solve_lift_waypoint(target_xyz, seed_positions):
            saved_qpos = data.qpos.copy()
            data.qpos[qpos_ids] = seed_positions
            data.qpos[qpos_ids[3:6]] = fixed_wrist_and_gripper

            try:
                for _ in range(120):
                    mujoco.mj_forward(model, data)
                    error = target_xyz - data.site_xpos[env.gripper_site]
                    if np.linalg.norm(error) < 0.001:
                        break

                    jac_pos = np.zeros((3, model.nv))
                    jac_rot = np.zeros((3, model.nv))
                    mujoco.mj_jacSite(
                        model,
                        data,
                        jac_pos,
                        jac_rot,
                        env.gripper_site,
                    )
                    jac = jac_pos[:, lift_dof_ids[:3]]
                    delta = jac.T @ np.linalg.solve(
                        jac @ jac.T + 0.05**2 * np.eye(3), error
                    )
                    delta_length = np.linalg.norm(delta)
                    if delta_length > 0.05:
                        delta *= 0.05 / delta_length

                    for joint_index in range(3):
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
                residual = float(
                    np.linalg.norm(
                        target_xyz - data.site_xpos[env.gripper_site]
                    )
                )
                return solution, residual
            finally:
                data.qpos[:] = saved_qpos
                mujoco.mj_forward(model, data)

        for waypoint_index in range(1, waypoint_count + 1):
            waypoint_xyz = lift_start_site + np.array(
                [0.0, 0.0, 0.06 * waypoint_index / waypoint_count]
            )
            actual_start = data.qpos[qpos_ids].copy()
            waypoint_q, ik_error = solve_lift_waypoint(
                waypoint_xyz, actual_start
            )
            if ik_error > 0.003:
                print(
                    f"ABORT: Cartesian lift IK error at waypoint "
                    f"{waypoint_index}: {ik_error * 1000:.1f} mm."
                )
                lift_aborted = True
                break
            if not checker.candidate_is_safe(actual_start, waypoint_q):
                print(
                    f"ABORT: jaw/table check failed at lift waypoint "
                    f"{waypoint_index}."
                )
                lift_aborted = True
                break

            for substep in range(frames_per_waypoint):
                fraction = (substep + 1) / frames_per_waypoint
                commanded = actual_start + fraction * (
                    waypoint_q - actual_start
                )
                _, _, terminated, truncated, _ = env.step(
                    action_for(commanded)
                )

                actual_positions = data.qpos[qpos_ids].copy()
                ball_now = data.xpos[env.ball_body].copy()
                if not checker.pose_is_collision_free(actual_positions):
                    print(
                        f"ABORT: jaw/table collision at lift waypoint "
                        f"{waypoint_index}, substep {substep}."
                    )
                    lift_aborted = True
                    break

                if terminated or truncated:
                    print(
                        f"ABORT: episode ended at lift waypoint "
                        f"{waypoint_index}, substep {substep}."
                    )
                    lift_aborted = True
                    break

                sideways_shift = np.linalg.norm(
                    ball_now[:2] - ball_after_close[:2]
                )
                if sideways_shift > 0.012:
                    print(
                        f"ABORT: ball slid sideways "
                        f"{sideways_shift * 1000:.1f} mm during lift."
                    )
                    lift_aborted = True
                    break

                if env._is_pinched():
                    lost_pinch_steps = 0
                else:
                    lost_pinch_steps += 1
                    if lost_pinch_steps >= 5:
                        print(
                            f"ABORT: pinch lost for 5 frames; "
                            f"ball z={ball_now[2]:.4f} m."
                        )
                        lift_aborted = True
                        break

            if lift_aborted:
                break

            waypoint_joint_error = np.max(
                np.abs(data.qpos[qpos_ids] - waypoint_q)
            )
            if waypoint_joint_error > 0.03:
                print(
                    f"ABORT: joint target error at lift waypoint "
                    f"{waypoint_index} is {waypoint_joint_error:.3f} rad."
                )
                lift_aborted = True
                break
            lift_target = waypoint_q

        ball_after_lift = data.xpos[env.ball_body].copy()
        rise = ball_after_lift[2] - ball_after_close[2]

        print("ball after close:", np.round(ball_after_close, 4))
        print("ball after lift:", np.round(ball_after_lift, 4))
        print(f"ball rise: {rise * 1000:.1f} mm")
        print("pinched at end:", bool(env._is_pinched()))
        print("grasped at end:", bool(env._is_grasped()))
        lift_joint_error = np.max(
            np.abs(data.qpos[qpos_ids] - lift_target)
        )
        lift_success = (
            not lift_aborted
            and lift_joint_error <= 0.03
            and env._is_grasped()
        )

        print(f"lift joint error: {lift_joint_error:.3f} rad")
        print("lift success:", lift_success)
        return 0 if lift_success else 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
