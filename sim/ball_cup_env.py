from pathlib import Path

import mujoco
import numpy as np

JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper"
)

class BallCupEnv:
    def __init__(self, image_size=224, frame_skip=20, horizon=300, seed=None):
        root = Path(__file__).resolve().parents[1]
        scene = root / "assets" / "SO101" / "ball_cup_scene.xml"
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.frame_skip = frame_skip
        self.horizon = horizon
        self.steps = 0
        self.touched_ball = False
        self.stable_steps = 0

        def get_id(kind, name):
            idx = mujoco.mj_name2id(self.model, kind, name)
            if idx < 0:
                raise ValueError(f"MuJoCo model is missing {name!r}")
            return idx

        self.joint_qpos = {
            name: self.model.jnt_qposadr[
                get_id(mujoco.mjtObj.mjOBJ_JOINT, name)
            ]
            for name in JOINT_NAMES
        }
        self.actuator_ids = [
            get_id(mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in JOINT_NAMES
        ]
        self.ball_joint = get_id(
            mujoco.mjtObj.mjOBJ_JOINT, "ball_free"
        )
        self.ball_qpos = self.model.jnt_qposadr[self.ball_joint]
        self.ball_qvel = self.model.jnt_dofadr[self.ball_joint]
        self.ball_body = get_id(mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.ball_geom = get_id(mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
        self.cup_body = get_id(mujoco.mjtObj.mjOBJ_BODY, "cup")

        self.gripper_bodies = {
            get_id(mujoco.mjtObj.mjOBJ_BODY, "gripper"),
            get_id(mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1"),
        }

        self.renderer = mujoco.Renderer(
            self.model, height=image_size, width=image_size
        )

    def _ball_in_cup(self):
        ball = self.data.xpos[self.ball_body]
        cup = self.model.body_pos[self.cup_body]
        radius_from_center = np.linalg.norm(ball[:2] - cup[:2])
        return (
            radius_from_center < 0.025
            and 0.025 <= ball[2] <= 0.075
        )

    def _gripper_touching_ball(self):
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1, g2 = int(contact.geom1), int(contact.geom2)

            if g1 == self.ball_geom:
                other_geom = g2
            elif g2 == self.ball_geom:
                other_geom = g1
            else:
                continue

            other_body = int(self.model.geom_bodyid[other_geom])
            if other_body in self.gripper_bodies:
                return True

        return False

    def _observation(self):
        self.renderer.update_scene(self.data, camera="front")
        image = self.renderer.render().copy()

        joints = np.array(
            [self.data.qpos[self.joint_qpos[name]] for name in JOINT_NAMES],
            dtype=np.float32,
        )
        return {"image": image, "joint_positions": joints}

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)

        # Small randomized cup and ball positions, both on the tabletop.
        cup_x = self.rng.uniform(0.23, 0.27)
        cup_y = self.rng.uniform(0.11, 0.15)
        self.model.body_pos[self.cup_body] = [cup_x, cup_y, 0.0]

        ball_x = self.rng.uniform(0.18, 0.32)
        ball_y = self.rng.uniform(-0.20, -0.08)
        self.data.qpos[self.ball_qpos:self.ball_qpos + 3] = [
            ball_x, ball_y, 0.020
        ]
        self.data.qpos[self.ball_qpos + 3:self.ball_qpos + 7] = [
            1.0, 0.0, 0.0, 0.0
        ]

        self.steps = 0
        self.touched_ball = False
        self.stable_steps = 0
        mujoco.mj_forward(self.model, self.data)

        return self._observation(), {
            "ball_in_cup": False,
            "gripper_touched_ball": False,
        }

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (6,):
            raise ValueError(f"Expected 6 joint actions, got {action.shape}")
        if not np.all(np.isfinite(action)):
            raise ValueError("Action contains NaN or infinity")

        action = np.clip(action, -1.0, 1.0)
        for i, actuator_id in enumerate(self.actuator_ids):
            low, high = self.model.actuator_ctrlrange[actuator_id]
            self.data.ctrl[actuator_id] = low + (action[i] + 1) * 0.5 * (
                high - low
            )

        self.steps += 1
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

            if self._gripper_touching_ball():
                self.touched_ball = True

            ball_speed = np.linalg.norm(
                self.data.qvel[self.ball_qvel:self.ball_qvel + 3]
            )
            settled_in_cup = self._ball_in_cup() and ball_speed < 0.05

            if self.touched_ball and settled_in_cup:
                self.stable_steps += 1
            else:
                self.stable_steps = 0

        success = self.stable_steps >= 150
        truncated = self.steps >= self.horizon

        info = {
            "ball_in_cup": self._ball_in_cup(),
            "gripper_touched_ball": self.touched_ball,
            "is_success": success,
        }
        return self._observation(), float(success), success, truncated, info

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None