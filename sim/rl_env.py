"""Gymnasium task environment for PPO pick-and-place training."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from sim.ball_cup_env import JOINT_NAMES, BallCupEnv
from sim.training_config import load_training_config


ARM_JOINT_NAMES = JOINT_NAMES[:5]
CONTROL_HZ = 20
PHYSICS_HZ = 500
OBSERVATION_SIZE = 25


class BallCupTrainingEnv(gym.Env):
    """State-based, relative-position control task around the SO-101 model.

    The underlying MuJoCo model and actual CAD fingertip contacts come from
    ``BallCupEnv``. This wrapper owns RL actions, curriculum randomization,
    observations, placement success, and the PPO reward.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": CONTROL_HZ}

    def __init__(
        self,
        stage: int = 1,
        horizon: int | None = None,
        seed: int | None = None,
        config_path: str | Path | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()
        if stage not in range(1, 6):
            raise ValueError("stage must be between 1 and 5")
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or 'rgb_array'")

        self.config = load_training_config(config_path)
        self.task = self.config["task"]
        self.reward_config = self.config["reward"]
        self.control_hz = int(self.task["control_hz"])
        self.horizon = int(horizon or self.task["episode_steps"])
        self.render_mode = render_mode
        self.stage = int(stage)
        self.rng = np.random.default_rng(seed)
        self.sim = BallCupEnv(
            frame_skip=1,
            horizon=self.horizon,
            seed=seed,
            render_images=False,
        )
        self.model = self.sim.model
        self.data = self.sim.data
        timestep = float(self.model.opt.timestep)
        self._physics_hz = int(round(1.0 / timestep))
        if not np.isclose(timestep, 0.002, atol=1e-9):
            raise ValueError(
                f"Expected the existing 2 ms model timestep, found {timestep:g} s"
            )
        if self.control_hz <= 0 or self._physics_hz % self.control_hz:
            raise ValueError("control_hz must divide the MuJoCo physics rate")
        self.frame_skip = self._physics_hz // self.control_hz
        self.sim.frame_skip = self.frame_skip
        self._renderer = None

        self.joint_qpos = np.asarray(
            [self.sim.joint_qpos[name] for name in JOINT_NAMES], dtype=np.int32
        )
        self.joint_qvel = np.asarray(
            [
                self.model.jnt_dofadr[
                    mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_JOINT, name
                    )
                ]
                for name in JOINT_NAMES
            ],
            dtype=np.int32,
        )
        self.arm_ctrl_ranges = self.model.actuator_ctrlrange[
            self.sim.actuator_ids[:5]
        ].copy()
        self.joint_ranges = np.asarray(
            [
                self.model.jnt_range[
                    mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_JOINT, name
                    )
                ]
                for name in JOINT_NAMES[:5]
            ],
            dtype=np.float64,
        )
        start_pose = np.asarray(self.task["fixed_arm_start_rad"], dtype=np.float64)
        if np.any(start_pose < self.joint_ranges[:, 0]) or np.any(
            start_pose > self.joint_ranges[:, 1]
        ):
            raise ValueError("fixed_arm_start_rad violates the model's arm joint limits")
        model_ball_mass = float(self.model.body_mass[self.sim.ball_body])
        configured_ball_mass = float(self.task["ball_mass_kg"])
        if not np.isclose(model_ball_mass, configured_ball_mass, atol=1e-8):
            raise ValueError(
                "ball_mass_kg must match the mass compiled into the MuJoCo model "
                f"({model_ball_mass:g} kg)"
            )
        self.open_gripper_target = float(
            self.model.actuator_ctrlrange[self.sim.actuator_ids[-1], 1]
        )
        self.close_gripper_target = float(
            self.model.actuator_ctrlrange[self.sim.actuator_ids[-1], 0]
        )
        self.gripper_policy_close_target = float(
            self.task["gripper_policy_close_target_rad"]
        )
        self.gripper_action_delta = float(self.task["gripper_action_delta_rad"])
        if not (
            self.close_gripper_target
            <= self.gripper_policy_close_target
            < self.open_gripper_target
        ):
            raise ValueError("gripper_policy_close_target_rad is outside the actuator range")
        if self.gripper_action_delta <= 0.0:
            raise ValueError("gripper_action_delta_rad must be positive")
        self.max_arm_delta = float(
            np.deg2rad(self.task["arm_delta_degrees"])
        )

        self.cup_geom_groups = {
            1.0: self._geom_ids(
                ["cup_bottom"] + [f"cup_wall_{i}" for i in range(8)]
            ),
            0.9: self._geom_ids(
                ["cup_bottom_s090"]
                + [f"cup_wall_s090_{i}" for i in range(8)]
            ),
            1.1: self._geom_ids(
                ["cup_bottom_s110"]
                + [f"cup_wall_s110_{i}" for i in range(8)]
            ),
        }
        self.ball_geom_groups = {
            0.0195: self._geom_ids(["ball_geom_s0195"]),
            0.020: self._geom_ids(["ball_geom"]),
            0.0205: self._geom_ids(["ball_geom_s0205"]),
        }
        self.cup_body_groups = {
            1.0: (self.sim.cup_body, int(self.model.body_mocapid[self.sim.cup_body])),
            0.9: self._body_and_mocap_id("cup_s090"),
            1.1: self._body_and_mocap_id("cup_s110"),
        }
        self.ball_variant_info = {
            0.020: self._ball_variant("ball", "ball_geom", "ball_free"),
            0.0195: self._ball_variant(
                "ball_s0195", "ball_geom_s0195", "ball_s0195_free", "ball_s0195_parked"
            ),
            0.0205: self._ball_variant(
                "ball_s0205", "ball_geom_s0205", "ball_s0205_free", "ball_s0205_parked"
            ),
        }
        supported_radii = set(self.ball_geom_groups)
        supported_cup_scales = set(self.cup_geom_groups)
        if set(map(float, self.task["ball_radius_variants_m"])) - supported_radii:
            raise ValueError("ball_radius_variants_m must match precompiled scene geometries")
        if set(map(float, self.task["cup_scale_variants"])) - supported_cup_scales:
            raise ValueError("cup_scale_variants must match precompiled scene geometries")
        self._all_variant_geoms = np.asarray(
            [
                geom_id
                for group in (*self.cup_geom_groups.values(), *self.ball_geom_groups.values())
                for geom_id in group
            ],
            dtype=np.int32,
        )
        self._base_rgba = {
            geom_id: self.model.geom_rgba[geom_id].copy()
            for geom_id in self._all_variant_geoms
        }
        self.table_geom = self._geom_id("table_top")
        self.cup_mocap_id = int(self.model.body_mocapid[self.sim.cup_body])
        if self.cup_mocap_id < 0:
            raise ValueError("The cup body must be a mocap body for curriculum placement")

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )
        # 6 measured joint positions + 6 velocities + EE/ball/cup positions (9)
        # + ball linear velocity (3) + elapsed episode fraction (1).
        # The physical state is sufficient for feedback; no scripted clock is used.
        float32_limit = np.finfo(np.float32).max
        self.observation_space = spaces.Box(
            low=-float32_limit,
            high=float32_limit,
            shape=(OBSERVATION_SIZE,),
            dtype=np.float32,
        )
        self.steps = 0
        self._episode_number = 0
        self._reset_episode_state()

    def _geom_id(self, name: str) -> int:
        geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, name
        )
        if geom_id < 0:
            raise ValueError(f"MuJoCo model is missing geometry {name!r}")
        return int(geom_id)

    def _geom_ids(self, names: list[str]) -> tuple[int, ...]:
        return tuple(self._geom_id(name) for name in names)

    def _body_and_mocap_id(self, name: str) -> tuple[int, int]:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"MuJoCo model is missing body {name!r}")
        mocap_id = int(self.model.body_mocapid[body_id])
        if mocap_id < 0:
            raise ValueError(f"MuJoCo body {name!r} must be a mocap body")
        return int(body_id), mocap_id

    def _ball_variant(self, body_name, geom_name, joint_name, equality_name=None):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        geom_id = self._geom_id(geom_name)
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if body_id < 0 or joint_id < 0:
            raise ValueError(f"MuJoCo model is missing ball variant {body_name!r}")
        equality_id = -1
        if equality_name is not None:
            equality_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_EQUALITY, equality_name
            )
            if equality_id < 0:
                raise ValueError(f"MuJoCo model is missing equality {equality_name!r}")
        return {
            "body": int(body_id),
            "geom": int(geom_id),
            "joint": int(joint_id),
            "qpos": int(self.model.jnt_qposadr[joint_id]),
            "qvel": int(self.model.jnt_dofadr[joint_id]),
            "equality": int(equality_id),
        }

    def set_stage(self, stage: int) -> None:
        """Select the stage used by the next reset."""
        stage = int(stage)
        if stage not in range(1, 6):
            raise ValueError("stage must be between 1 and 5")
        self.stage = stage

    def _activate_scene_variants(self, ball_radius: float, cup_scale: float, cup_xy):
        """Select only compiled contact bodies; leave collision masks untouched."""
        active_ball = self.ball_variant_info[ball_radius]
        for radius, variant in self.ball_variant_info.items():
            geom_id = variant["geom"]
            self.model.geom_rgba[geom_id] = self._base_rgba[geom_id]
            self.model.geom_rgba[geom_id, 3] = 1.0 if radius == ball_radius else 0.0
            if variant["equality"] >= 0:
                self.data.eq_active[variant["equality"]] = radius != ball_radius

        active_cup_body, active_cup_mocap = self.cup_body_groups[cup_scale]
        for index, (scale, (body_id, mocap_id)) in enumerate(self.cup_body_groups.items()):
            self.data.mocap_pos[mocap_id] = (
                [cup_xy[0], cup_xy[1], 0.0]
                if scale == cup_scale
                else [10.0 + 2.0 * index, 10.0, 0.0]
            )
            self.data.mocap_quat[mocap_id] = [1.0, 0.0, 0.0, 0.0]
            for geom_id in self.cup_geom_groups[scale]:
                self.model.geom_rgba[geom_id] = self._base_rgba[geom_id]
                self.model.geom_rgba[geom_id, 3] = 1.0 if scale == cup_scale else 0.0

        self.sim.ball_body = active_ball["body"]
        self.sim.ball_geom = active_ball["geom"]
        self.sim.ball_joint = active_ball["joint"]
        self.sim.ball_qpos = active_ball["qpos"]
        self.sim.ball_qvel = active_ball["qvel"]
        self.sim.cup_body = active_cup_body
        self.sim.cup_mocap_id = active_cup_mocap
        self.cup_mocap_id = active_cup_mocap

    def _select_scene_sizes(self) -> tuple[float, float]:
        if self.stage < 5:
            return 0.020, 1.0
        radius = float(self.rng.choice(self.task["ball_radius_variants_m"]))
        cup_scale = float(self.rng.choice(self.task["cup_scale_variants"]))
        ratio = (0.09 * cup_scale) / (2.0 * radius)
        minimum = float(
            self.task["minimum_cup_opening_to_ball_diameter_ratio"]
        )
        if ratio < minimum:
            raise ValueError(
                f"Invalid size pair: cup/ball opening ratio {ratio:.3f} < {minimum}"
            )
        return radius, cup_scale

    def _sample_positions(self) -> tuple[np.ndarray, np.ndarray]:
        if self.stage == 1:
            ball_xy = np.asarray(self.task["fixed_ball_xy"], dtype=np.float64)
            cup_xy = np.asarray(self.task["fixed_cup_xy"], dtype=np.float64)
        else:
            ball_bounds = np.asarray(self.task["ball_xy_bounds"], dtype=np.float64)
            ball_xy = self.rng.uniform(ball_bounds[:, 0], ball_bounds[:, 1])
            if self.stage == 2:
                cup_xy = np.asarray(self.task["fixed_cup_xy"], dtype=np.float64)
            else:
                cup_bounds = np.asarray(self.task["cup_xy_bounds"], dtype=np.float64)
                cup_xy = self.rng.uniform(cup_bounds[:, 0], cup_bounds[:, 1])
        return ball_xy, cup_xy

    def _start_pose_is_clear(self) -> bool:
        """Reject sampled starts with arm contacts against the table/objects."""
        mujoco.mj_forward(self.model, self.data)
        ball_body = self.sim.ball_body
        cup_body = self.sim.cup_body
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if contact.dist > 0.0:
                continue
            geom_a, geom_b = int(contact.geom1), int(contact.geom2)
            body_a = int(self.model.geom_bodyid[geom_a])
            body_b = int(self.model.geom_bodyid[geom_b])
            if self.table_geom in (geom_a, geom_b):
                other_body = body_b if geom_a == self.table_geom else body_a
                if other_body not in (0, ball_body, cup_body):
                    return False
            object_body = ball_body if ball_body in (body_a, body_b) else (
                cup_body if cup_body in (body_a, body_b) else None
            )
            if object_body is not None:
                other_body = body_b if body_a == object_body else body_a
                if other_body not in (0, ball_body, cup_body):
                    return False
        return True

    def _sample_start_joints(self) -> np.ndarray:
        start = np.asarray(
            self.task["fixed_arm_start_rad"], dtype=np.float64
        )
        if start.shape != (5,):
            raise ValueError("fixed_arm_start_rad must contain five joint positions")
        if self.stage < 4:
            return start
        jitter = np.deg2rad(float(self.task["start_joint_jitter_degrees"]))
        for _ in range(64):
            candidate = self.rng.uniform(start - jitter, start + jitter, size=5)
            self.data.qpos[self.joint_qpos[:5]] = candidate
            self.data.qpos[self.joint_qpos[5]] = self.open_gripper_target
            if self._start_pose_is_clear():
                return candidate
        self.data.qpos[self.joint_qpos[:5]] = start
        self.data.qpos[self.joint_qpos[5]] = self.open_gripper_target
        if not self._start_pose_is_clear():
            raise RuntimeError("No collision-free starting configuration was found")
        return start

    def _reset_episode_state(self) -> None:
        self.steps = 0
        self._pinch_stability_frames = 0
        self._stable_placement_steps = 0
        self._gripper_target = self.open_gripper_target
        self.currently_pinched = False
        self.holding_ball = False
        self.pinched_ball = False
        self.reach_success = False
        self.grasp_success = False
        self.lift_success = False
        self.transport_success = False
        self.above_cup_success = False
        self.correct_release = False
        self.placement_success = False
        self.dropped_outside_cup = False
        self.unsafe_contact_seen = False
        self._drop_penalty_given = False
        self._unsafe_penalty_given = False
        self.failure_reason = ""
        self.reward_totals = {
            name: 0.0
            for name in (
                "reach_progress", "grasp", "lift", "transport_progress",
                "above_cup", "correct_release", "success",
                "dropped_outside_cup", "unsafe_contact", "joint_limit",
                "time", "failure",
            )
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._episode_number += 1
        mujoco.mj_resetData(self.model, self.data)

        ball_xy, cup_xy = self._sample_positions()
        ball_radius, cup_scale = self._select_scene_sizes()
        self.ball_radius = ball_radius
        self.cup_scale = cup_scale
        self._activate_scene_variants(ball_radius, cup_scale, cup_xy)
        self.data.qpos[self.sim.ball_qpos:self.sim.ball_qpos + 3] = [
            ball_xy[0], ball_xy[1], ball_radius
        ]
        self.data.qpos[self.sim.ball_qpos + 3:self.sim.ball_qpos + 7] = [
            1.0, 0.0, 0.0, 0.0
        ]
        self.data.qvel[self.sim.ball_qvel:self.sim.ball_qvel + 6] = 0.0
        start_joints = self._sample_start_joints()
        self.data.qpos[self.joint_qpos[:5]] = start_joints
        self.data.qpos[self.joint_qpos[5]] = self.open_gripper_target
        self.data.qvel[self.joint_qvel] = 0.0
        self.data.ctrl[self.sim.actuator_ids[:5]] = start_joints
        self.data.ctrl[self.sim.actuator_ids[5]] = self.open_gripper_target
        mujoco.mj_forward(self.model, self.data)
        if not self._start_pose_is_clear():
            raise RuntimeError("Reset produced a colliding robot start")

        self._reset_episode_state()
        self._previous_approach_distance = self._approach_path_distance()[0]
        self._best_transport_distance = float(
            np.linalg.norm(self.data.xpos[self.sim.ball_body, :2] - self.data.xpos[self.sim.cup_body, :2])
        )
        obs = self._observation()
        info = self._episode_info(is_success=False)
        info.update({
            "ball_radius_m": self.ball_radius,
            "cup_scale": self.cup_scale,
            "ball_position": self.data.xpos[self.sim.ball_body].copy(),
            "cup_position": self.data.xpos[self.sim.cup_body].copy(),
            "start_joint_positions": self.data.qpos[self.joint_qpos[:5]].copy(),
        })
        return obs, info

    def map_action_to_targets(self, action: np.ndarray):
        """Map arm actions to bounded joint steps and gripper action to position."""
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (6,):
            raise ValueError(f"Expected six actions, got shape {action.shape}")
        if not np.all(np.isfinite(action)):
            raise ValueError("Action contains NaN or infinity")
        action = np.clip(action, -1.0, 1.0)
        current = self.data.qpos[self.joint_qpos[:5]].copy()
        requested = current + action[:5] * self.max_arm_delta
        low = np.maximum(self.joint_ranges[:, 0], self.arm_ctrl_ranges[:, 0])
        high = np.minimum(self.joint_ranges[:, 1], self.arm_ctrl_ranges[:, 1])
        targets = np.clip(requested, low, high)
        clipped_fraction = float(
            np.sum(np.abs(requested - targets) / max(self.max_arm_delta, 1e-8))
        )
        gripper_action = float(action[5])
        current_gripper = float(self.data.qpos[self.joint_qpos[5]])
        self._gripper_target = float(
            np.clip(
                current_gripper + gripper_action * self.gripper_action_delta,
                self.gripper_policy_close_target,
                self.open_gripper_target,
            )
        )
        return targets, self._gripper_target, clipped_fraction

    def _approach_path_distance(self):
        """Distance left along the demonstrated pregrasp-to-pinch approach."""
        ee = self.data.site_xpos[self.sim.gripper_site]
        ball = self.data.xpos[self.sim.ball_body]
        offset = np.asarray(self.task["grasp_site_offset_m"], dtype=np.float64)
        contact = ball + offset
        clearance = float(self.task["grasp_approach_clearance_m"])
        pregrasp = contact + np.asarray([0.0, 0.0, clearance])
        approach_vector = contact - pregrasp
        projection = float(
            np.dot(ee - pregrasp, approach_vector)
            / max(float(np.dot(approach_vector, approach_vector)), 1e-12)
        )
        if projection < 0.0:
            path_distance = float(np.linalg.norm(ee - pregrasp) + clearance)
        else:
            path_distance = float(np.linalg.norm(ee - contact))

        wrist_error = np.asarray(
            [
                self.data.qpos[self.joint_qpos[3]]
                - float(self.task["grasp_wrist_flex_rad"]),
                self.data.qpos[self.joint_qpos[4]]
                - float(self.task["grasp_wrist_roll_rad"]),
            ],
            dtype=np.float64,
        )
        orientation_distance = float(
            self.task["grasp_orientation_distance_scale_m_per_rad"]
            * np.linalg.norm(wrist_error)
        )
        position_error = float(np.linalg.norm(ee - contact))
        return path_distance + orientation_distance, position_error, float(
            np.linalg.norm(wrist_error)
        )

    def _cup_dimensions(self):
        # Geometry comes from the compiled octagonal cup: 50 mm wall-center
        # radius, 5 mm half-thickness, and a 5..90 mm interior z span.
        return 0.045 * self.cup_scale, 0.010 * self.cup_scale, 0.090 * self.cup_scale

    def _inside_cup_xy(self, extra_margin: float = 0.0) -> bool:
        ball = self.data.xpos[self.sim.ball_body]
        cup = self.data.xpos[self.sim.cup_body]
        inner_apothem, _, _ = self._cup_dimensions()
        radius = self.ball_radius + extra_margin
        if inner_apothem <= radius:
            return False
        for index in range(8):
            angle = index * np.pi / 4.0
            normal = np.asarray([np.cos(angle), np.sin(angle)])
            signed_distance = float(np.dot(ball[:2] - cup[:2], normal))
            if signed_distance > inner_apothem - radius:
                return False
        return True

    def _ball_inside_cup_volume(self, safety_margin: float | None = None) -> bool:
        if safety_margin is None:
            safety_margin = float(
                self.task["evaluation_success_xy_margin_m"]
            )
        if not self._inside_cup_xy(extra_margin=safety_margin):
            return False
        ball_z = float(self.data.xpos[self.sim.ball_body, 2])
        _, floor_top, rim_top = self._cup_dimensions()
        tolerance = 0.002
        return (
            ball_z >= floor_top + self.ball_radius - tolerance
            and ball_z <= rim_top - self.ball_radius + tolerance
        )

    def _is_settled_placement(self) -> bool:
        if (
            not self.grasp_success
            or not self.lift_success
            or self.holding_ball
            or not self._ball_inside_cup_volume()
        ):
            return False
        linear = self.data.qvel[self.sim.ball_qvel:self.sim.ball_qvel + 3]
        angular = self.data.qvel[self.sim.ball_qvel + 3:self.sim.ball_qvel + 6]
        return (
            np.linalg.norm(linear) < 0.03
            and np.linalg.norm(angular) < 1.0
        )

    def _observation(self) -> np.ndarray:
        positions = self.data.qpos[self.joint_qpos]
        velocities = self.data.qvel[self.joint_qvel]
        ee = self.data.site_xpos[self.sim.gripper_site]
        ball = self.data.xpos[self.sim.ball_body]
        cup = self.data.xpos[self.sim.cup_body]
        ball_velocity = self.data.qvel[self.sim.ball_qvel:self.sim.ball_qvel + 3]
        observation = np.concatenate(
            [
                positions,
                velocities,
                ee,
                ball,
                cup,
                ball_velocity,
                [self.steps / self.horizon],
            ]
        ).astype(np.float32)
        if observation.shape != (OBSERVATION_SIZE,):
            raise RuntimeError(f"Internal observation has shape {observation.shape}")
        return observation

    def _has_unsafe_contact(self) -> bool:
        allowed_finger_geoms = self.sim.fixed_finger_geoms | self.sim.moving_finger_geoms
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if contact.dist > -0.001:
                continue
            geom_a, geom_b = int(contact.geom1), int(contact.geom2)
            body_a = int(self.model.geom_bodyid[geom_a])
            body_b = int(self.model.geom_bodyid[geom_b])
            if self.table_geom in (geom_a, geom_b):
                other_geom = geom_b if geom_a == self.table_geom else geom_a
                other_body = int(self.model.geom_bodyid[other_geom])
                if other_body not in (0, self.sim.ball_body, self.sim.cup_body):
                    return True
            for object_body in (self.sim.ball_body, self.sim.cup_body):
                if object_body in (body_a, body_b):
                    other_geom = geom_b if body_a == object_body else geom_a
                    other_body = int(self.model.geom_bodyid[other_geom])
                    if (
                        other_body not in (0, self.sim.ball_body, self.sim.cup_body)
                        and other_geom not in allowed_finger_geoms
                    ):
                        return True
        return False

    def _episode_info(self, *, is_success: bool) -> dict:
        approach_distance, grasp_position_error, grasp_wrist_error = (
            self._approach_path_distance()
        )
        return {
            "is_success": bool(is_success),
            "curriculum_stage": int(self.stage),
            "reach_success": bool(self.reach_success),
            "grasp_success": bool(self.grasp_success),
            "lift_success": bool(self.lift_success),
            "transport_success": bool(self.transport_success),
            "above_cup_success": bool(self.above_cup_success),
            "correct_release": bool(self.correct_release),
            "placement_success": bool(self.placement_success),
            "currently_pinched": bool(self.currently_pinched),
            "holding_ball": bool(self.holding_ball),
            "failure_reason": self.failure_reason,
            "approach_path_distance_m": float(approach_distance),
            "grasp_position_error_m": float(grasp_position_error),
            "grasp_wrist_error_rad": float(grasp_wrist_error),
        }

    def step(self, action):
        targets, gripper_target, clipped_fraction = self.map_action_to_targets(action)
        for joint_index, actuator_id in enumerate(self.sim.actuator_ids[:5]):
            self.data.ctrl[actuator_id] = targets[joint_index]
        self.data.ctrl[self.sim.actuator_ids[5]] = gripper_target

        previously_grasped = self.grasp_success
        previously_lifted = self.lift_success
        previously_above_cup = self.above_cup_success
        previously_correct_release = self.correct_release
        self.steps += 1
        invalid_state = False
        pinch_frames_required = int(
            round(float(self.task["pinch_stability_seconds"]) * self._physics_hz)
        )
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            if not (
                np.all(np.isfinite(self.data.qpos))
                and np.all(np.isfinite(self.data.qvel))
                and np.all(np.isfinite(self.data.xpos))
            ):
                invalid_state = True
                break
            self.currently_pinched = self.sim._is_pinched()
            if self.currently_pinched:
                self._pinch_stability_frames += 1
                self.pinched_ball = True
            else:
                self._pinch_stability_frames = 0
            if self._pinch_stability_frames >= pinch_frames_required:
                self.grasp_success = True
            self.holding_ball = self.sim._gripper_touching_ball()
            if self._has_unsafe_contact():
                self.unsafe_contact_seen = True

        if invalid_state:
            self.failure_reason = "invalid_simulation_state"
            self.holding_ball = False
            self.currently_pinched = False
        else:
            ee = self.data.site_xpos[self.sim.gripper_site]
            ball = self.data.xpos[self.sim.ball_body]
            cup = self.data.xpos[self.sim.cup_body]
            _, grasp_position_error, grasp_wrist_error = self._approach_path_distance()
            self.reach_success = self.reach_success or (
                grasp_position_error
                <= float(self.task["grasp_pose_reach_threshold_m"])
                and grasp_wrist_error
                <= float(self.task["grasp_wrist_reach_tolerance_rad"])
            )
            if self.grasp_success:
                self.lift_success = self.lift_success or (
                    float(ball[2]) >= float(self.task["lift_threshold_z_m"])
                )
            _, _, rim_top = self._cup_dimensions()
            if self.lift_success and self.holding_ball and self._inside_cup_xy():
                self.transport_success = True
            if (
                self.lift_success
                and self.currently_pinched
                and self._inside_cup_xy()
                and float(ball[2]) >= rim_top + self.ball_radius
            ):
                self.above_cup_success = True
            if (
                self.grasp_success
                and not self.holding_ball
                and self._ball_inside_cup_volume()
            ):
                self.correct_release = True

            if (
                self.grasp_success
                and not self.holding_ball
                and not self._ball_inside_cup_volume(safety_margin=0.0)
                and float(ball[2]) <= self.ball_radius + 0.004
            ):
                self.dropped_outside_cup = True

            if self._is_settled_placement():
                self._stable_placement_steps += 1
            else:
                self._stable_placement_steps = 0
            required_settle_steps = int(
                np.ceil(float(self.task["placement_stability_seconds"]) * self.control_hz)
            )
            self.placement_success = (
                self._stable_placement_steps >= required_settle_steps
            )

            if float(ball[2]) < float(self.task["ball_unrecoverable_z_m"]):
                self.failure_reason = "ball_fell_below_recovery_height"

        terminated_success = bool(self.placement_success and not invalid_state)
        terminated_failure = bool(self.failure_reason)
        terminated = terminated_success or terminated_failure
        truncated = bool(self.steps >= self.horizon and not terminated)

        reward_terms = {name: 0.0 for name in self.reward_totals}
        if not invalid_state:
            approach_distance = self._approach_path_distance()[0]
            if not previously_grasped and not self.grasp_success:
                # Potential follows the demonstrated above-ball waypoint and
                # final fingertip contact pose, including wrist orientation.
                reward_terms["reach_progress"] = float(
                    self.reward_config["reach_progress_per_meter"]
                    * (self._previous_approach_distance - approach_distance)
                )
            self._previous_approach_distance = approach_distance

            if self.grasp_success and not previously_grasped:
                reward_terms["grasp"] = float(self.reward_config["grasp_once"])
            if self.lift_success and not previously_lifted:
                reward_terms["lift"] = float(self.reward_config["lift_once"])

            ball = self.data.xpos[self.sim.ball_body]
            cup = self.data.xpos[self.sim.cup_body]
            current_distance = float(np.linalg.norm(ball[:2] - cup[:2]))
            if self.lift_success and self.currently_pinched:
                best_delta = max(0.0, self._best_transport_distance - current_distance)
                reward_terms["transport_progress"] = float(
                    self.reward_config["transport_progress_per_meter"] * best_delta
                )
                self._best_transport_distance = min(
                    self._best_transport_distance, current_distance
                )
            if self.above_cup_success and not previously_above_cup:
                reward_terms["above_cup"] = float(self.reward_config["above_cup_once"])
            if self.correct_release and not previously_correct_release:
                reward_terms["correct_release"] = float(
                    self.reward_config["correct_release_once"]
                )
            if terminated_success:
                reward_terms["success"] = float(self.reward_config["terminal_success"])
            if self.dropped_outside_cup and not getattr(self, "_drop_penalty_given", False):
                reward_terms["dropped_outside_cup"] = -float(
                    self.reward_config["dropped_outside_cup_once"]
                )
                self._drop_penalty_given = True
            if self.unsafe_contact_seen and not getattr(self, "_unsafe_penalty_given", False):
                reward_terms["unsafe_contact"] = -float(
                    self.reward_config["unsafe_contact_once"]
                )
                self._unsafe_penalty_given = True
            reward_terms["joint_limit"] = -float(
                self.reward_config["joint_limit_clip_per_normalized_action_unit"]
                * clipped_fraction
            )
            reward_terms["time"] = -float(self.reward_config["per_step_time_cost"])
        if terminated_failure:
            reward_terms["failure"] = -float(
                self.reward_config["unrecoverable_failure"]
            )
        for name, value in reward_terms.items():
            self.reward_totals[name] += float(value)
        reward = float(sum(reward_terms.values()))

        info = self._episode_info(is_success=terminated_success)
        info.update({
            "ball_in_cup": bool(self._ball_inside_cup_volume(safety_margin=0.0)),
            "episode_length": int(self.steps),
            "reward_terms": reward_terms,
            "episode_reward_terms": self.reward_totals.copy(),
            "ball_radius_m": float(self.ball_radius),
            "cup_scale": float(self.cup_scale),
        })
        if terminated or truncated:
            info["episode_metrics"] = {
                "is_success": bool(terminated_success),
                "reach_success": bool(self.reach_success),
                "grasp_success": bool(self.grasp_success),
                "lift_success": bool(self.lift_success),
                "transport_success": bool(self.transport_success),
                "above_cup_success": bool(self.above_cup_success),
                "correct_release": bool(self.correct_release),
                "placement_success": bool(self.placement_success),
                "episode_length": int(self.steps),
            }
        observation = self._observation()
        if not np.all(np.isfinite(observation)):
            observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera="front")
        return self._renderer.render().copy()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self.sim.close()
