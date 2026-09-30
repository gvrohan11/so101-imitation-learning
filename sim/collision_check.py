import math

import mujoco
import numpy as np

from sim.ball_cup_env import JOINT_NAMES, BallCupEnv


class JawTableCollisionChecker:
    """Reject arm poses and paths where the moving jaw hits the table."""

    def __init__(self, env: BallCupEnv):
        self.env = env
        self.model = env.model
        self.data = env.data
        self.joint_qpos = np.array(
            [env.joint_qpos[name] for name in JOINT_NAMES], dtype=np.int32
        )
        self.moving_jaw_body = env.moving_jaw_body
        self.table_geom = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "table_top"
        )
        if self.table_geom < 0:
            raise ValueError("MuJoCo model is missing 'table_top'")

    def _has_jaw_table_collision(self, joint_positions):
        qpos = np.asarray(joint_positions, dtype=np.float64)
        if qpos.shape != (len(JOINT_NAMES),):
            raise ValueError(
                f"Expected {len(JOINT_NAMES)} joint positions, got {qpos.shape}"
            )
        if not np.all(np.isfinite(qpos)):
            raise ValueError("Joint positions contain NaN or infinity")

        saved_qpos = self.data.qpos.copy()
        try:
            self.data.qpos[self.joint_qpos] = qpos
            mujoco.mj_forward(self.model, self.data)
            for contact_index in range(self.data.ncon):
                contact = self.data.contact[contact_index]
                geom1, geom2 = int(contact.geom1), int(contact.geom2)
                if geom1 == self.table_geom:
                    other_geom = geom2
                elif geom2 == self.table_geom:
                    other_geom = geom1
                else:
                    continue

                other_body = int(self.model.geom_bodyid[other_geom])
                if (
                    other_body == self.moving_jaw_body
                    and contact.dist <= 0.0
                ):
                    return True
            return False
        finally:
            self.data.qpos[:] = saved_qpos
            mujoco.mj_forward(self.model, self.data)

    def pose_is_collision_free(self, joint_positions):
        """Return false if the specified six-joint pose hits the table."""
        return not self._has_jaw_table_collision(joint_positions)

    def trajectory_is_collision_free(
        self, start_positions, goal_positions, max_joint_step=0.02
    ):
        """Check each interpolated pose, including the start and goal."""
        start = np.asarray(start_positions, dtype=np.float64)
        goal = np.asarray(goal_positions, dtype=np.float64)
        expected_shape = (len(JOINT_NAMES),)
        if start.shape != expected_shape or goal.shape != expected_shape:
            raise ValueError(
                f"Start and goal must both have shape {expected_shape}"
            )
        if not np.all(np.isfinite(start)) or not np.all(np.isfinite(goal)):
            raise ValueError("Trajectory positions contain NaN or infinity")
        if not math.isfinite(max_joint_step) or max_joint_step <= 0.0:
            raise ValueError("max_joint_step must be a finite positive number")

        segment_count = max(
            1,
            math.ceil(np.max(np.abs(goal - start)) / max_joint_step),
        )
        for segment_index in range(segment_count + 1):
            fraction = segment_index / segment_count
            pose = start + fraction * (goal - start)
            if not self.pose_is_collision_free(pose):
                return False
        return True