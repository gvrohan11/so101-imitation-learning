from pathlib import Path

import mujoco

model_path = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "SO101"
    / "scene.xml"
)

model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)

for _ in range(1000):
    mujoco.mj_step(model, data)

print(f"Loaded SO-101 model: nq={model.nq}, nu={model.nu}")
for i in range(model.nu):
    name = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, i
    )
    print(f"actuator {i}: {name}")
print(f"Simulation time after smoke test: {data.time:.2f}s")