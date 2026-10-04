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

MIN_HORIZONTAL_PINCH_OPPOSITION = -0.75
MAX_PINCH_PENETRATION = 0.004

class BallCupEnv:
    def __init__(
        self,
        image_size=224,
        frame_skip=20,
        horizon=300,
        seed=None,
        render_images=True,
        ball_radius=0.020,
    ):
        root = Path(__file__).resolve().parents[1]
        scene = root / "assets" / "SO101" / "ball_cup_scene.xml"
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed) # random number generator
        self.frame_skip = frame_skip
        self.horizon = horizon
        self.steps = 0
        self.stable_steps = 0
        self.currently_pinched = False
        self.pinched_ball = False
        self.grasped_ball = False
        self.currently_grasped = False
        self.render_images = render_images
        self.previous_gripper_ball_distance = None
        self.previous_ball_cup_distance = None

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
        self.ball_variants = {
            0.020: self._ball_variant_ids("ball", "ball_geom", "ball_free"),
            0.0195: self._ball_variant_ids(
                "ball_s0195", "ball_geom_s0195", "ball_s0195_free", "ball_s0195_parked"
            ),
            0.0205: self._ball_variant_ids(
                "ball_s0205", "ball_geom_s0205", "ball_s0205_free", "ball_s0205_parked"
            ),
        }
        self._ball_variant_rgba = {
            variant["geom"]: self.model.geom_rgba[variant["geom"]].copy()
            for variant in self.ball_variants.values()
        }
        self.ball_radius = float(ball_radius)
        self._select_ball_variant(self.ball_radius)
        self.fixed_finger_geom = get_id(
            mujoco.mjtObj.mjOBJ_GEOM, "fixed_finger_collision"
        )
        self.fixed_finger_geoms = frozenset(
            geom_id
            for geom_id in range(self.model.ngeom)
            if (name := mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
            )) is not None
            and (
                name == "fixed_finger_collision"
                or name.startswith("fixed_finger_collision_section_")
            )
        )
        self.moving_finger_geom = get_id(
            mujoco.mjtObj.mjOBJ_GEOM, "moving_finger_collision"
        )
        self.moving_finger_geoms = frozenset(
            geom_id
            for geom_id in range(self.model.ngeom)
            if (name := mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
            )) is not None
            and (
                name == "moving_finger_collision"
                or name.startswith("moving_finger_collision_")
            )
        )
        self.finger_geom_groups = (
            self.fixed_finger_geoms,
            self.moving_finger_geoms,
        )
        self.cup_body = get_id(mujoco.mjtObj.mjOBJ_BODY, "cup")
        self.cup_mocap_id = int(self.model.body_mocapid[self.cup_body])

        self.fixed_gripper_body = get_id(
            mujoco.mjtObj.mjOBJ_BODY, "gripper"
        )
        self.moving_jaw_body = get_id(
            mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1"
        )
        self.gripper_bodies = {
            self.fixed_gripper_body,
            self.moving_jaw_body,
        }

        self.renderer = (
            mujoco.Renderer(self.model, height=image_size, width=image_size)
            if render_images
            else None
        )

        self.gripper_site = get_id(
            mujoco.mjtObj.mjOBJ_SITE, "gripperframe"
        )

    def _ball_in_cup(self):
        ball = self.data.xpos[self.ball_body]
        cup = self.data.xpos[self.cup_body]
        radius_from_center = np.linalg.norm(ball[:2] - cup[:2])
        return (
            radius_from_center < 0.025
            and 0.025 <= ball[2] <= 0.075
        )

    def _ball_variant_ids(self, body_name, geom_name, joint_name, equality_name=None):
        def get_id(kind, name):
            value = mujoco.mj_name2id(self.model, kind, name)
            if value < 0:
                raise ValueError(f"MuJoCo model is missing {name!r}")
            return int(value)

        body_id = get_id(mujoco.mjtObj.mjOBJ_BODY, body_name)
        geom_id = get_id(mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        joint_id = get_id(mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        equality_id = -1
        if equality_name is not None:
            equality_id = get_id(mujoco.mjtObj.mjOBJ_EQUALITY, equality_name)
        return {
            "body": body_id,
            "geom": geom_id,
            "joint": joint_id,
            "qpos": int(self.model.jnt_qposadr[joint_id]),
            "qvel": int(self.model.jnt_dofadr[joint_id]),
            "equality": equality_id,
        }

    def _select_ball_variant(self, radius):
        radius = float(radius)
        if radius not in self.ball_variants:
            raise ValueError(f"Unsupported precompiled ball radius: {radius}")
        active = self.ball_variants[radius]
        for variant_radius, variant in self.ball_variants.items():
            geom_id = variant["geom"]
            self.model.geom_rgba[geom_id] = self._ball_variant_rgba[geom_id]
            self.model.geom_rgba[geom_id, 3] = 1.0 if variant_radius == radius else 0.0
            if variant["equality"] >= 0:
                self.data.eq_active[variant["equality"]] = variant_radius != radius
        self.ball_body = active["body"]
        self.ball_geom = active["geom"]
        self.ball_joint = active["joint"]
        self.ball_qpos = active["qpos"]
        self.ball_qvel = active["qvel"]
        self.ball_radius = radius

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
        image = None
        if self.renderer is not None:
            self.renderer.update_scene(self.data, camera="front")
            image = self.renderer.render().copy()

        joints = np.array(
            [self.data.qpos[self.joint_qpos[name]] for name in JOINT_NAMES],
            dtype=np.float32,
        )

        state = np.concatenate(
            [
                self.data.site_xpos[self.gripper_site], # gripper xyz
                self.data.xpos[self.ball_body], # ball xyz
                self.data.xpos[self.cup_body], # cup xyz
                self.data.qvel[self.ball_qvel:self.ball_qvel + 3], # ball velocity
                joints
            ]
        ).astype(np.float32)

        return {
            "image": image, 
            "joint_positions": joints,
            "state": state
        }

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)
        self._select_ball_variant(self.ball_radius)

        gripper_actuator = self.actuator_ids[-1]
        open_angle = self.model.actuator_ctrlrange[gripper_actuator, 1]
        self.data.qpos[self.joint_qpos["gripper"]] = open_angle
        self.data.ctrl[gripper_actuator] = open_angle

        # Small randomized cup and ball positions, both on the tabletop.
        cup_x = self.rng.uniform(0.23, 0.27)
        cup_y = self.rng.uniform(0.11, 0.15)
        cup_position = np.array([cup_x, cup_y, 0.0], dtype=np.float64)
        if self.cup_mocap_id >= 0:
            self.data.mocap_pos[self.cup_mocap_id] = cup_position
        else:
            self.model.body_pos[self.cup_body] = cup_position

        ball_x = self.rng.uniform(0.18, 0.32)
        ball_y = self.rng.uniform(-0.20, -0.08)
        self.data.qpos[self.ball_qpos:self.ball_qpos + 3] = [
            ball_x, ball_y, self.ball_radius
        ]
        self.data.qpos[self.ball_qpos + 3:self.ball_qpos + 7] = [
            1.0, 0.0, 0.0, 0.0
        ]

        self.steps = 0
        self.currently_pinched = False
        self.pinched_ball = False
        self.grasped_ball = False
        self.currently_grasped = False
        self.stable_steps = 0
        mujoco.mj_forward(self.model, self.data)

        gripper = self.data.site_xpos[self.gripper_site]
        ball = self.data.xpos[self.ball_body]
        cup = self.data.xpos[self.cup_body]
        self.previous_gripper_ball_distance = np.linalg.norm(gripper - ball)
        self.previous_ball_cup_distance = np.linalg.norm(ball[:2] - cup[:2])

        return self._observation(), {
            "ball_in_cup": False,
            "currently_pinched": self.currently_pinched,
            "pinched_ball": self.pinched_ball,
            "currently_grasped": self.currently_grasped,
            "grasped_ball": self.grasped_ball,
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
        had_pinched_ball = self.pinched_ball
        had_grasped_ball = self.grasped_ball
        lift_progress = 0.0
        previous_ball_height = float(self.data.xpos[self.ball_body][2])
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

            self.currently_pinched = self._is_pinched()
            if self.currently_pinched:
                self.pinched_ball = True
                current_ball_height = float(self.data.xpos[self.ball_body][2])
                lift_progress += np.clip(
                    current_ball_height - previous_ball_height,
                    -0.005,
                    0.005,
                )
            previous_ball_height = float(self.data.xpos[self.ball_body][2])

            self.currently_grasped = self._is_grasped()
            if self.currently_grasped:
                self.grasped_ball = True

            ball_speed = np.linalg.norm(
                self.data.qvel[self.ball_qvel:self.ball_qvel + 3]
            )
            settled_in_cup = self._ball_in_cup() and ball_speed < 0.05

            if self.grasped_ball and settled_in_cup:
                self.stable_steps += 1
            else:
                self.stable_steps = 0

        success = self.stable_steps >= 150
        truncated = self.steps >= self.horizon

        gripper = self.data.site_xpos[self.gripper_site]
        ball = self.data.xpos[self.ball_body]
        cup = self.data.xpos[self.cup_body]
        gripper_ball_distance = np.linalg.norm(gripper - ball)
        ball_cup_distance = np.linalg.norm(ball[:2] - cup[:2])

        reward_approach = 0.0
        reward_pinch = 0.0
        reward_lift = 0.0
        reward_grasp = 0.0
        reward_delivery = 0.0
        reward_success = 0.0
        if not had_grasped_ball:
            progress = self.previous_gripper_ball_distance - gripper_ball_distance
            reward_approach = 10.0 * np.clip(progress, -0.05, 0.05)
            if self.pinched_ball and not had_pinched_ball:
                reward_pinch = 1.0
            if self.currently_pinched:
                reward_lift = 40.0 * lift_progress
            if self.grasped_ball:
                reward_grasp = 5.0 # grasp reward given only after ball has risen past threshold
        else:
            if self.currently_grasped:
                progress = self.previous_ball_cup_distance - ball_cup_distance
                reward_delivery = 10.0 * np.clip(progress, -0.05, 0.05)

        if success:
            reward_success = 10.0

        reward = float(
            reward_approach
            + reward_pinch
            + reward_lift
            + reward_grasp
            + reward_delivery
            + reward_success
        )
        self.previous_gripper_ball_distance = gripper_ball_distance
        self.previous_ball_cup_distance = ball_cup_distance

        info = {
            "ball_in_cup": self._ball_in_cup(),
            "currently_pinched": self.currently_pinched,
            "pinched_ball": self.pinched_ball,
            "currently_grasped": self.currently_grasped,
            "grasped_ball": self.grasped_ball,
            "is_success": success,
            "reward_approach": float(reward_approach),
            "reward_pinch": float(reward_pinch),
            "reward_lift": float(reward_lift),
            "reward_grasp": float(reward_grasp),
            "reward_delivery": float(reward_delivery),
            "reward_success": float(reward_success),
        }
        return self._observation(), reward, success, truncated, info

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def _ball_contact_bodies(self):
        bodies = set()

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if self.ball_geom not in (contact.geom1, contact.geom2):
                continue

            other_geom = (
                contact.geom2
                if contact.geom1 == self.ball_geom
                else contact.geom1
            )
            body_id = int(self.model.geom_bodyid[other_geom])
            if body_id in self.gripper_bodies:
                bodies.add(body_id)

        return bodies


    def _is_pinched(self):
        finger_group = {
            geom: group_index
            for group_index, geoms in enumerate(self.finger_geom_groups)
            for geom in geoms
        }
        best_contact = {}

        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            if self.ball_geom == contact.geom1:
                finger_geom = int(contact.geom2)
            elif self.ball_geom == contact.geom2:
                finger_geom = int(contact.geom1)
            else:
                continue

            group_index = finger_group.get(finger_geom)
            if group_index is None:
                continue

            # MuJoCo can keep a contact record for separated shapes inside
            # the collision margin. A pinch requires the real finger surface
            # to touch or penetrate the ball, not just be nearby.
            if contact.dist > 0.0:
                continue

            previous = best_contact.get(group_index)
            if previous is None or contact.dist < previous[0]:
                best_contact[group_index] = (
                    float(contact.dist),
                    contact.pos.copy(),
                )

        required_groups = range(len(self.finger_geom_groups))
        if not all(group in best_contact for group in required_groups):
            return False

        if any(
            best_contact[group][0] < -MAX_PINCH_PENETRATION
            for group in required_groups
        ):
            return False

        ball_center = self.data.xpos[self.ball_body]
        directions = []

        for group in required_groups:
            direction = best_contact[group][1] - ball_center
            length = np.linalg.norm(direction)
            if length < 1e-8:
                return False
            directions.append(direction / length)

        # The two actual finger surfaces must contact opposite sides of the
        # ball. Do not require a below-center contact: the physical SO101
        # fingertips contact near the ball's equator, where friction carries
        # its weight during a lift.
        horizontal = []
        for direction in directions:
            horizontal_direction = direction[:2]
            horizontal_length = np.linalg.norm(horizontal_direction)
            if horizontal_length < 1e-8:
                return False
            horizontal.append(horizontal_direction / horizontal_length)

        return (
            float(np.dot(horizontal[0], horizontal[1]))
            <= MIN_HORIZONTAL_PINCH_OPPOSITION
        )

    def _is_grasped(self):
        ball_is_lifted = self.data.xpos[self.ball_body][2] > 0.035
        return self._is_pinched() and ball_is_lifted
