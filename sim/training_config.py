"""Load the single source of truth for the state-policy training setup."""

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "ppo_training.json"


def load_training_config(path=None):
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    required_sections = ("task", "reward", "curriculum", "ppo", "logging")
    missing = [section for section in required_sections if section not in config]
    if missing:
        raise ValueError(f"Training config is missing sections: {missing}")

    task = config["task"]
    curriculum = config["curriculum"]
    if int(task["control_hz"]) <= 0 or int(task["episode_steps"]) <= 0:
        raise ValueError("control_hz and episode_steps must be positive")
    if len(task["fixed_ball_xy"]) != 2 or len(task["fixed_cup_xy"]) != 2:
        raise ValueError("Fixed ball and cup positions must be XY pairs")
    if len(task["fixed_arm_start_rad"]) != 5:
        raise ValueError("fixed_arm_start_rad must contain five arm joints")
    for name in ("ball_xy_bounds", "cup_xy_bounds"):
        bounds = task[name]
        if len(bounds) != 2 or any(len(axis) != 2 or axis[0] >= axis[1] for axis in bounds):
            raise ValueError(f"{name} must contain ordered [min, max] XY bounds")
    for value_name, bounds_name in (
        ("fixed_ball_xy", "ball_xy_bounds"),
        ("fixed_cup_xy", "cup_xy_bounds"),
    ):
        point = task[value_name]
        bounds = task[bounds_name]
        if any(not (axis[0] <= coordinate <= axis[1]) for coordinate, axis in zip(point, bounds)):
            raise ValueError(f"{value_name} must lie within {bounds_name}")

    min_opening_ratio = float(task["minimum_cup_opening_to_ball_diameter_ratio"])
    for radius in task["ball_radius_variants_m"]:
        for scale in task["cup_scale_variants"]:
            opening_ratio = (0.09 * float(scale)) / (2.0 * float(radius))
            if opening_ratio < min_opening_ratio:
                raise ValueError(
                    "Curriculum contains a cup/ball size pair below the "
                    f"minimum opening ratio: scale={scale}, radius={radius}"
                )

    if (
        int(curriculum["minimum_steps_per_stage"]) <= 0
        or int(curriculum["maximum_steps_per_stage"])
        < int(curriculum["minimum_steps_per_stage"])
        or int(curriculum["evaluation_interval_steps"]) <= 0
        or int(curriculum["evaluation_episodes"]) <= 0
        or int(curriculum["consecutive_passing_evaluations"]) <= 0
        or not 0.0 < float(curriculum["required_success_rate"]) <= 1.0
    ):
        raise ValueError("Invalid curriculum evaluation or stopping settings")
    return config
