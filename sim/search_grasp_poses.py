from itertools import product

import mujoco
import os
import numpy as np

from sim.ball_cup_env import (
    JOINT_NAMES,
    BallCupEnv,
)
from sim.collision_check import JawTableCollisionChecker

def main():
    env = BallCupEnv(render_images=False)
    model, data = env.model, env.data

    try:
        env.reset(seed=int(os.environ.get("SO101_GRASP_SEED", "0")))
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
        # Keep the gripper fully open during the side approach, then close at
        # the contact pose after the low fingertips are beside the ball.
        # The jaw must be partly closed before descending: its fully open
        # fingertip sweeps below the tabletop during the closing arc.
        gripper_approach = 0.70
        gripper_contact_open = 0.70

        # The sphere proxies used an offset well below the gripper frame.
        # Search the wrist orientation and frame offset that put the real
        # fingertips on opposite lower sides of the ball.
        # Include the slightly tucked-wrist family: the SO101 moving jaw
        # otherwise closes with a large vertical sweep and drives the ball
        # along the fingertip taper before it can support a lift.
        wrist_flex_values = np.array([-0.3, -0.2, -0.1, 0.1, 0.3, 0.5])
        # Scan the whole reachable wrist-roll range. The earlier narrow bands
        # put the moving fingertip's thin CAD axis across the ball's lateral
        # slip direction, so a static opposed pinch could not carry it.
        wrist_roll_values = np.array(
            sorted(
                {
                    round(float(value), 6)
                    for value in np.concatenate(
                        (
                            np.arange(-2.7, 2.81, 0.3),
                            np.arange(-2.7, -2.29, 0.05),
                        )
                    )
                }
            )
        )
        offsets = [
            np.array(offset, dtype=np.float64)
            for offset in product(
                (-0.020, -0.012, 0.012, 0.020),
                (-0.020, -0.012, -0.008, 0.012, 0.020),
                (-0.010, -0.006, -0.002),
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

            finger_contacts = {}
            for i in range(data.ncon):
                contact = data.contact[i]

                if contact.geom1 == env.ball_geom:
                    finger_geom = int(contact.geom2)
                elif contact.geom2 == env.ball_geom:
                    finger_geom = int(contact.geom1)
                else:
                    continue

                group_index = next(
                    (
                        index
                        for index, geoms in enumerate(env.finger_geom_groups)
                        if finger_geom in geoms
                    ),
                    None,
                )
                if group_index is None:
                    continue

                old = finger_contacts.get(group_index)
                if old is None or contact.dist < old[0]:
                    finger_contacts[group_index] = (
                        float(contact.dist),
                        contact.pos.copy(),
                    )

            opposition = None
            contact_heights = None
            required = range(len(env.finger_geom_groups))
            if all(finger in finger_contacts for finger in required):
                center = data.xpos[env.ball_body]
                directions = []
                for group_index in required:
                    direction = finger_contacts[group_index][1] - center
                    length = np.linalg.norm(direction)
                    if length > 1e-8:
                        direction = direction / length
                    directions.append(direction)
                if all(np.linalg.norm(direction) > 0.0 for direction in directions):
                    opposition = float(
                        np.dot(directions[0], directions[1])
                    )
                    contact_heights = tuple(
                        float(
                            finger_contacts[group_index][1][2] - center[2]
                        )
                        for group_index in required
                    )

            is_pinched = bool(env._is_pinched())
            data.qpos[:] = saved_qpos
            mujoco.mj_forward(model, data)
            return is_pinched, opposition, contact_heights

        def first_safe_pinch(joint_positions):
            """Find the first shallow, opposed contact while closing the jaw."""
            previous_angle = gripper_contact_open
            for angle in np.linspace(
                gripper_contact_open, gripper_closed, 64
            )[1:]:
                candidate = joint_positions.copy()
                candidate[-1] = angle
                is_pinched, opposition, contact_heights = pinches_ball(candidate)
                if not is_pinched:
                    previous_angle = angle
                    continue

                not_pinched_angle = previous_angle
                pinched_angle = angle
                for _ in range(7):
                    midpoint = 0.5 * (not_pinched_angle + pinched_angle)
                    candidate[-1] = midpoint
                    midpoint_pinched, _, _ = pinches_ball(candidate)
                    if midpoint_pinched:
                        pinched_angle = midpoint
                    else:
                        not_pinched_angle = midpoint

                candidate[-1] = pinched_angle
                is_pinched, opposition, contact_heights = pinches_ball(candidate)
                if (
                    is_pinched
                    and opposition is not None
                    and contact_heights is not None
                ):
                    return candidate, opposition, contact_heights
                return None
            return None

        def minimum_finger_ball_distance(joint_positions):
            """Measure ball clearance from both finger collision meshes."""
            saved_qpos = data.qpos.copy()
            data.qpos[qpos_ids] = joint_positions
            mujoco.mj_forward(model, data)
            distances = []
            required = set().union(*env.finger_geom_groups)
            for i in range(data.ncon):
                contact = data.contact[i]
                if env.ball_geom not in (contact.geom1, contact.geom2):
                    continue
                other = (
                    int(contact.geom2)
                    if contact.geom1 == env.ball_geom
                    else int(contact.geom1)
                )
                if other in required:
                    distances.append(float(contact.dist))
            data.qpos[:] = saved_qpos
            mujoco.mj_forward(model, data)
            return min(distances, default=float("inf"))

        ik_solutions = 0
        pinches = 0
        open_jaw_collisions = 0
        start_path_collisions = 0
        descent_collisions = 0
        no_safe_pinch = 0
        closing_collisions = 0
        collision_samples = {}
        safe_paths = []
        safe_pinch_candidates = []
        best_opposition = None
        best_pose = None
        best_contact_q = None

        for wrist_flex, wrist_roll, offset in product(
            wrist_flex_values, wrist_roll_values, offsets
        ):
            contact_target = ball + offset
            pregrasp_target = contact_target + np.array([0.0, 0.0, 0.09])

            pregrasp_q, pregrasp_error = solve_site(
                pregrasp_target, wrist_flex, wrist_roll, gripper_approach
            )
            contact_q, contact_error = solve_site(
                contact_target, wrist_flex, wrist_roll, gripper_closed
            )
            contact_open_q = contact_q.copy()
            contact_open_q[-1] = gripper_contact_open

            if pregrasp_error > 0.005 or contact_error > 0.005:
                continue
            ik_solutions += 1

            # Keep the pre-closed approach clear, then stop the closure at the first
            # shallow, opposed two-finger contact instead of driving to the
            # mechanically closed end stop.
            # The dynamic diagnostic moves through Cartesian waypoints. A
            # straight joint-space chord can dip the jaw through the table
            # even when that executed Cartesian path does not, so only reject
            # colliding endpoints here; the diagnostic checks the real route.
            if not checker.pose_is_collision_free(start_q) or not checker.pose_is_collision_free(pregrasp_q):
                start_path_collisions += 1
                collision_samples.setdefault(
                    "start",
                    (checker.last_collision, checker.last_collision_qpos),
                )
                continue
            if not checker.pose_is_collision_free(contact_open_q):
                descent_collisions += 1
                collision_samples.setdefault(
                    "descent",
                    (checker.last_collision, checker.last_collision_qpos),
                )
                continue
            if minimum_finger_ball_distance(contact_open_q) < -0.001:
                open_jaw_collisions += 1
                continue

            first_pinch = first_safe_pinch(contact_q)
            if first_pinch is None:
                no_safe_pinch += 1
                continue
            contact_pinch_q, opposition, contact_heights = first_pinch
            contact_height_span = abs(
                contact_heights[0] - contact_heights[1]
            )
            contact_height_mean = abs(float(np.mean(contact_heights)))
            if contact_height_span > 0.006 or contact_height_mean > 0.008:
                no_safe_pinch += 1
                continue
            if not checker.candidate_is_safe(
                contact_open_q, contact_pinch_q
            ):
                closing_collisions += 1
                collision_samples.setdefault(
                    "closing",
                    (checker.last_collision, checker.last_collision_qpos),
                )
                continue

            pinches += 1
            candidate_path = (
                pregrasp_error + contact_error,
                float(pregrasp_q[3]),
                float(pregrasp_q[4]),
                offset,
                pregrasp_q,
                contact_open_q,
                contact_pinch_q,
            )
            safe_paths.append(candidate_path)
            safe_pinch_candidates.append(
                (*candidate_path, opposition, contact_heights)
            )

            if best_opposition is None or opposition < best_opposition:
                best_opposition = opposition
                best_pose = (
                    float(pregrasp_q[3]),
                    float(pregrasp_q[4]),
                    offset.copy(),
                )
                best_contact_q = contact_pinch_q.copy()

        print("ball:", np.round(ball, 4))
        print("IK-reachable candidates:", ik_solutions)
        print("approach, shallow-pinch paths clear of jaw/table contact:", len(safe_paths))
        print("first-contact stable pinch candidates:", pinches)
        print("rejected for open-jaw ball overlap:", open_jaw_collisions)
        print(
            "path rejects (start, descent, pinch, close):",
            start_path_collisions,
            descent_collisions,
            no_safe_pinch,
            closing_collisions,
        )
        if collision_samples:
            print("first table-contact rejects:", collision_samples)

        if best_opposition is not None:
            print(
                "Best two-finger contact dot product:",
                round(best_opposition, 3),
                "| pose:",
                best_pose,
            )

            data.qpos[qpos_ids] = best_contact_q
            mujoco.mj_forward(model, data)
            ball_center = data.xpos[env.ball_body].copy()
            print("Ball center:", np.round(ball_center, 4))

            for label, finger_geoms in (
                ("fixed", env.fixed_finger_geoms),
                ("moving", env.moving_finger_geoms),
            ):
                for i in range(data.ncon):
                    contact = data.contact[i]
                    pair = {int(contact.geom1), int(contact.geom2)}
                    if env.ball_geom in pair and pair.intersection(finger_geoms):
                        finger_geom = min(pair.intersection(finger_geoms))
                        point = contact.pos.copy()
                        print(
                            f"{label} finger mesh: geom origin="
                            f"{np.round(data.geom_xpos[finger_geom], 4)}, "
                            f"contact point={np.round(point, 4)}, "
                            f"from ball center={np.round(point - ball_center, 4)}"
                        )
                        break

            closed_centers = {
                "fixed": data.geom_xpos[env.fixed_finger_geom].copy(),
                "moving": np.mean(
                    data.geom_xpos[list(env.moving_finger_geoms)], axis=0
                ),
            }

            data.qpos[qpos_ids[-1]] = gripper_open
            mujoco.mj_forward(model, data)

            print(
                "Fixed finger mesh origin when open:",
                np.round(data.geom_xpos[env.fixed_finger_geom], 4),
            )
            print(
                "Moving finger mesh origin when open:",
                np.round(
                    np.mean(
                        data.geom_xpos[list(env.moving_finger_geoms)], axis=0
                    ),
                    4,
                ),
            )
            opening_travel = (
                np.mean(
                    data.geom_xpos[list(env.moving_finger_geoms)], axis=0
                )
                - closed_centers["moving"]
            )
            print("Moving-finger opening travel:", np.round(opening_travel, 4))
        else:
            print("No candidate had simultaneous contact with both finger meshes.")

        if pinches == 0:
            print(
                "No candidate passed all checks. Do not execute the old pose; "
                "the next issue is pose/scene geometry, not PPO."
            )
            return []

        safe_paths.sort(key=lambda item: item[0])
        roll_joint_low, roll_joint_high = model.jnt_range[joint_ids[4]]
        safe_pinch_candidates.sort(
            key=lambda item: (
                # This actual-fingertip pose passed the dynamic carry test;
                # try it first instead of preferring a static score that
                # ignores whether the ball survives transport.
                abs(item[1] - 0.5)
                + abs(item[2] + 2.7)
                + 100.0 * float(
                    np.linalg.norm(
                        item[3] - np.array([0.020, -0.008, -0.010])
                    )
                ),
                abs(item[8][0] - item[8][1]),
                abs(float(np.mean(item[8])) + 0.005),
                item[7],
                -min(
                    item[4][4] - roll_joint_low,
                    roll_joint_high - item[4][4],
                ),
                max(item[8]),
                item[0],
            )
        )
        print("\nFirst jaw-table-clear static two-jaw candidates:")
        shown = 0
        for (
            total_error,
            wrist_flex,
            wrist_roll,
            offset,
            _,
            _,
            contact_pinch_q,
            contact_opposition,
            contact_heights,
        ) in safe_pinch_candidates:
            pregrasp = ball + offset + np.array([0.0, 0.0, 0.06])
            contact = ball + offset
            print(
                f"  wrist_flex={wrist_flex:+.3f}, "
                f"wrist_roll={wrist_roll:+.3f}, "
                f"gripper_contact={contact_pinch_q[-1]:+.3f} rad, "
                f"offset={np.round(offset, 3)}, "
                f"pregrasp={np.round(pregrasp, 3)}, "
                f"contact={np.round(contact, 3)}, "
                f"IK-error-sum={total_error*1000:.1f} mm, "
                f"contact-opposition={contact_opposition:.3f}, "
                f"contact-z-offsets={np.round(contact_heights, 4)}"
            )
            shown += 1
            if shown == 10:
                break
        return safe_pinch_candidates

    finally:
        env.close()


if __name__ == "__main__":
    main()
