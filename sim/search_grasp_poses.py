from itertools import product

import mujoco
import numpy as np

from sim.ball_cup_env import JOINT_NAMES, BallCupEnv
from sim.collision_check import JawTableCollisionChecker


def main():
    env = BallCupEnv(render_images=False)
    model, data = env.model, env.data

    try:
        env.reset(seed=0)
        checker = JawTableCollisionChecker(env)

        joint_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in JOINT_NAMES
        ]
        qpos_ids = [model.jnt_qposadr[j] for j in joint_ids]
        dof_ids = [model.jnt_dofadr[j] for j in joint_ids]
        site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe"
        )

        start_q = np.array([data.qpos[q] for q in qpos_ids])
        ball = data.xpos[env.ball_body].copy()
        gripper_open = float(
            model.actuator_ctrlrange[env.actuator_ids[-1]][1]
        )
        gripper_closed = float(
            model.actuator_ctrlrange[env.actuator_ids[-1]][0]
        )

        wrist_flex_range = model.jnt_range[joint_ids[3]]
        wrist_roll_range = model.jnt_range[joint_ids[4]]
        wrist_flex_values = np.linspace(*wrist_flex_range, 9)
        wrist_roll_values = np.linspace(*wrist_roll_range, 13)

        offsets = [
            np.array([dx, dy, dz])
            for dx, dy, dz in product(
                (-0.010, 0.0, 0.010),
                (-0.030, -0.015, 0.0, 0.015, 0.030),
                (0.005, 0.010, 0.015),
            )
        ]

        def solve_site(target, wrist_flex, wrist_roll, gripper_angle):
            saved_qpos = data.qpos.copy()
            data.qpos[qpos_ids[3]] = wrist_flex
            data.qpos[qpos_ids[4]] = wrist_roll
            data.qpos[qpos_ids[5]] = gripper_angle

            for _ in range(120):
                mujoco.mj_forward(model, data)
                error = target - data.site_xpos[site_id]
                if np.linalg.norm(error) < 0.003:
                    break

                jac_pos = np.zeros((3, model.nv))
                jac_rot = np.zeros((3, model.nv))
                mujoco.mj_jacSite(
                    model, data, jac_pos, jac_rot, site_id
                )
                jac = jac_pos[:, dof_ids[:3]]
                delta = jac.T @ np.linalg.solve(
                    jac @ jac.T + 0.05**2 * np.eye(3), error
                )

                length = np.linalg.norm(delta)
                if length > 0.10:
                    delta *= 0.10 / length

                for qid, jid, change in zip(
                    qpos_ids[:3], joint_ids[:3], delta
                ):
                    low, high = model.jnt_range[jid]
                    data.qpos[qid] = np.clip(
                        data.qpos[qid] + change, low, high
                    )

            mujoco.mj_forward(model, data)
            residual = float(
                np.linalg.norm(target - data.site_xpos[site_id])
            )
            solution = np.array([data.qpos[q] for q in qpos_ids])

            data.qpos[:] = saved_qpos
            mujoco.mj_forward(model, data)
            return solution, residual

        def pinches_ball(joint_positions):
            saved_qpos = data.qpos.copy()
            data.qpos[qpos_ids] = joint_positions
            mujoco.mj_forward(model, data)

            pad_contacts = {}
            for i in range(data.ncon):
                contact = data.contact[i]

                if contact.geom1 == env.ball_geom:
                    pad_id = int(contact.geom2)
                elif contact.geom2 == env.ball_geom:
                    pad_id = int(contact.geom1)
                else:
                    continue

                if pad_id not in (
                    env.fixed_finger_pad_geom,
                    env.moving_finger_pad_geom,
                ):
                    continue

                old = pad_contacts.get(pad_id)
                if old is None or contact.dist < old[0]:
                    pad_contacts[pad_id] = (
                        float(contact.dist),
                        contact.pos.copy(),
                    )

            opposition = None
            required = (
                env.fixed_finger_pad_geom,
                env.moving_finger_pad_geom,
            )
            if all(pad in pad_contacts for pad in required):
                center = data.xpos[env.ball_body]
                directions = []
                for pad in required:
                    direction = pad_contacts[pad][1] - center
                    length = np.linalg.norm(direction)
                    if length > 1e-8:
                        direction = direction / length
                    directions.append(direction)
                if all(np.linalg.norm(direction) > 0.0 for direction in directions):
                    opposition = float(
                        np.dot(directions[0], directions[1])
                    )

            is_pinched = bool(env._is_pinched())
            data.qpos[:] = saved_qpos
            mujoco.mj_forward(model, data)
            return is_pinched, opposition

        ik_solutions = 0
        pinches = 0
        safe_paths = []
        safe_pinch_candidates = []
        best_opposition = None
        best_pose = None

        for wrist_flex, wrist_roll, offset in product(
            wrist_flex_values, wrist_roll_values, offsets
        ):
            contact_target = ball + offset
            pregrasp_target = contact_target + np.array([0.0, 0.0, 0.06])

            pregrasp_q, pregrasp_error = solve_site(
                pregrasp_target, wrist_flex, wrist_roll, gripper_open
            )
            contact_q, contact_error = solve_site(
                contact_target, wrist_flex, wrist_roll, gripper_closed
            )
            contact_open_q = contact_q.copy()
            contact_open_q[-1] = gripper_open

            if pregrasp_error > 0.005 or contact_error > 0.005:
                continue
            ik_solutions += 1

            # Require both endpoints and the whole joint-interpolated
            # approach paths to be clear of moving-jaw/table contact.
            if not checker.candidate_is_safe(start_q, pregrasp_q):
                continue
            if not checker.candidate_is_safe(pregrasp_q, contact_open_q):
                continue
            if not checker.candidate_is_safe(contact_open_q, contact_q):
                continue
            safe_paths.append(
                (pregrasp_error + contact_error, wrist_flex,
                 wrist_roll, offset, pregrasp_q, contact_open_q, contact_q)
            )

            is_pinched, opposition = pinches_ball(contact_q)

            if opposition is not None:
                if best_opposition is None or opposition < best_opposition:
                    best_opposition = opposition
                    best_pose = (wrist_flex, wrist_roll, offset.copy())

            if is_pinched:
                pinches += 1
                safe_pinch_candidates.append(safe_paths[-1])

        print("ball:", np.round(ball, 4))
        print("IK-reachable candidates:", ik_solutions)
        print("paths clear of moving-jaw/table contact:", len(safe_paths))
        print("static two-jaw contact candidates:", pinches)

        if best_opposition is not None:
            print(
                "Best two-pad contact dot product:",
                round(best_opposition, 3),
                "| pose:",
                best_pose,
            )
        else:
            print("No candidate had simultaneous contact with both pads.")

        if pinches == 0:
            print(
                "No candidate passed all checks. Do not execute the old pose; "
                "the next issue is pose/scene geometry, not PPO."
            )
            return []

        safe_paths.sort(key=lambda item: item[0])
        safe_pinch_candidates.sort(key=lambda item: item[0])
        print("\nFirst jaw-table-clear static two-jaw candidates:")
        shown = 0
        for (
            total_error,
            wrist_flex,
            wrist_roll,
            offset,
            _,
            _,
            _,
        ) in safe_pinch_candidates:
            pregrasp = ball + offset + np.array([0.0, 0.0, 0.06])
            contact = ball + offset
            print(
                f"  wrist_flex={wrist_flex:+.3f}, "
                f"wrist_roll={wrist_roll:+.3f}, "
                f"offset={np.round(offset, 3)}, "
                f"pregrasp={np.round(pregrasp, 3)}, "
                f"contact={np.round(contact, 3)}, "
                f"IK-error-sum={total_error*1000:.1f} mm"
            )
            shown += 1
            if shown == 10:
                break
        return safe_pinch_candidates

    finally:
        env.close()


if __name__ == "__main__":
    main()