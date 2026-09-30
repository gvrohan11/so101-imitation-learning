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
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
