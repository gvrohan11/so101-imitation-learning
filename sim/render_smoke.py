from pathlib import Path

import mujoco
from PIL import Image

root = Path(__file__).resolve().parents[1]
scene = root / "assets" / "SO101" / "ball_cup_scene.xml"
output = root / "results" / "ball_cup_camera.png"

model = mujoco.MjModel.from_xml_path(str(scene))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

renderer = mujoco.Renderer(model, height=224, width=224)
renderer.update_scene(data, camera="front")
frame = renderer.render()

Image.fromarray(frame).save(output)
print(f"saved camera frame: {output}")
print(f"frame shape: {frame.shape}")