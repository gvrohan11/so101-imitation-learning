import mujoco
import numpy as np

from sim.ball_cup_env import JOINT_NAMES, BallCupEnv
from sim.collision_check import JawTableCollisionChecker
from sim.search_grasp_poses import main as search_safe_pinch_candidates

# Keep the sweep bounded. Candidates are sorted by IK error by the search script.
MAX_CANDIDATES = 20

# The ball starts near z=0.020 m. Lift the gripper 25 mm so a successful
# grasp has room to raise the ball above the required z=0.035 m threshold.
LIFT_METERS = 0.025
LIFT_WAYPOINTS = 25
FRAMES_PER_LIFT_WAYPOINT = 7

# Abort an approach if it pushes the ball too far.
MAX_APPROACH_BALL_SHIFT = 0.012  # meters


def main():
    candidates = search_safe_pinch_candidates()
    if not candidates:
        print("No static pinch candidates found; no dynamic tests attempted.")
        return 1

    candidates = candidates[:MAX_CANDIDATES]
    print(f"\nDynamically testing {len(candidates)} candidates")

    env = BallCupEnv(render_images=False, frame_skip=1, horizon=5000)
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

    def move_to(target_positions, frame_count, stage, ball_start):
        """Interpolate joint targets and stop on collision or excessive ball shift."""
        start_positions = data.qpos[qpos_ids].copy()

        for frame in range(frame_count):
            fraction = (frame + 1) / frame_count
            commanded = start_positions + fraction * (
                target_positions - start_positions
            )
            _, _, terminated, truncated, _ = env.step(action_for(commanded))

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
                return False

            if terminated or truncated:
                print(f"  abort: episode ended during {stage}")
                return False

        error = np.max(np.abs(data.qpos[qpos_ids] - target_positions))
        if error > 0.03:
            print(f"  abort: {stage} joint tracking error={error:.3f} rad")
            return False
        return True

    def solve_lift_waypoint(target_xyz, seed_positions, fixed_wrist_gripper):
        """Solve Cartesian XYZ using the first three arm joints."""
        saved_qpos = data.qpos.copy()
        data.qpos[qpos_ids] = seed_positions
        data.qpos[qpos_ids[3:6]] = fixed_wrist_gripper

        try:
            for _ in range(150):
                mujoco.mj_forward(model, data)
                error = target_xyz - data.site_xpos[env.gripper_site]
                if np.linalg.norm(error) < 0.001:
                    break

                jac_pos = np.zeros((3, model.nv))
                jac_rot = np.zeros((3, model.nv))
                mujoco.mj_jacSite(
                    model, data, jac_pos, jac_rot, env.gripper_site
                )
                jac = jac_pos[:, dof_ids[:3]]
                delta = jac.T @ np.linalg.solve(
                    jac @ jac.T + 0.05**2 * np.eye(3), error
                )

                length = np.linalg.norm(delta)
                if length > 0.05:
                    delta *= 0.05 / length

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

    def test_candidate(candidate_index, candidate):
        (
            total_error,
            wrist_flex,
            wrist_roll,
            offset,
            pregrasp_q,
            contact_open_q,
            contact_closed_q,
        ) = candidate

        # Candidate poses were searched using seed 0, so reset to seed 0 for
        # every attempt: same scene, fresh simulation state.
        env.reset(seed=0)
        mujoco.mj_forward(model, data)
        ball_start = ball_position()

        print(
            f"\nCandidate {candidate_index}: "
            f"wrist_flex={wrist_flex:+.3f}, "
            f"wrist_roll={wrist_roll:+.3f}, "
            f"offset={np.round(offset, 3)}, "
            f"IK error sum={total_error * 1000:.1f} mm"
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
        if not checker.candidate_is_safe(start_q, pregrasp_q):
            result["reason"] = "pregrasp path failed collision check"
            record_metrics()
            return result
        if not checker.candidate_is_safe(pregrasp_q, contact_open_q):
            result["reason"] = "open approach failed collision check"
            record_metrics()
            return result
        if not checker.candidate_is_safe(contact_open_q, contact_closed_q):
            result["reason"] = "closing path failed collision check"
            record_metrics()
            return result

        if not move_to(pregrasp_q, 400, "pregrasp", ball_start):
            result["reason"] = "pregrasp motion failed"
            record_metrics()
            return result
        if not move_to(contact_open_q, 400, "open-jaw approach", ball_start):
            result["reason"] = "approach motion failed"
            record_metrics()
            return result
        if not move_to(contact_closed_q, 300, "gripper close", ball_start):
            result["reason"] = "closing motion failed"
            record_metrics()
            return result

        ball_after_close = ball_position()
        fixed_pad_geom = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "fixed_finger_pad"
        )
        moving_pad_geom = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "moving_finger_pad"
        )
        pad_geoms = {fixed_pad_geom, moving_pad_geom}
        contacted_pads = set()

        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]

            if env.ball_geom == contact.geom1:
                other_geom = int(contact.geom2)
            elif env.ball_geom == contact.geom2:
                other_geom = int(contact.geom1)
            else:
                continue

            if other_geom in pad_geoms:
                contacted_pads.add(other_geom)

        pinched_after_close = pad_geoms.issubset(contacted_pads)
        ball_lift_reference = ball_after_close
        print(
            "  after close:",
            f"pinched={pinched_after_close}",
            f"ball={np.round(ball_after_close, 4)}",
        )

        if not pinched_after_close:
            result["reason"] = "no two-jaw pinch after closing"
            record_metrics()
            return result

        lift_start_site = data.site_xpos[env.gripper_site].copy()
        fixed_wrist_gripper = contact_closed_q[3:6].copy()
        pinch_persisted = True
        lift_aborted = False

        for waypoint_index in range(1, LIFT_WAYPOINTS + 1):
            target_xyz = lift_start_site + np.array(
                [0.0, 0.0, LIFT_METERS * waypoint_index / LIFT_WAYPOINTS]
            )
            actual_start = data.qpos[qpos_ids].copy()
            waypoint_q, ik_error = solve_lift_waypoint(
                target_xyz, actual_start, fixed_wrist_gripper
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
                _, _, terminated, truncated, _ = env.step(
                    action_for(commanded)
                )

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
                if lateral_shift > MAX_APPROACH_BALL_SHIFT:
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