from pathlib import Path
import math

import mujoco

scene = (
    Path(__file__).resolve().parents[1]
    / "assets" / "SO101" / "ball_cup_scene.xml"
)

model = mujoco.MjModel.from_xml_path(str(scene))
data = mujoco.MjData(model)

# Start the ball directly above the cup.
ball_joint = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free"
)
ball_qpos = model.jnt_qposadr[ball_joint]
data.qpos[ball_qpos:ball_qpos + 3] = [0.25, 0.13, 0.18]
data.qpos[ball_qpos + 3:ball_qpos + 7] = [1, 0, 0, 0]

# Let it fall and settle.
for _ in range(3000):
    mujoco.mj_step(model, data)

ball_body = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, "ball"
)
x, y, z = data.xpos[ball_body]

inside = math.hypot(x - 0.25, y - 0.13) < 0.025 and 0.025 <= z <= 0.075
print(f"ball position: x={x:.3f}, y={y:.3f}, z={z:.3f}")
print(f"ball_in_cup: {inside}")
print(f"model: nq={model.nq}, nu={model.nu}")