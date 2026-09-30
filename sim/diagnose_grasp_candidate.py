print("ball contacts after close:")
found_contact = False

for i in range(data.ncon):
    contact = data.contact[i]
    if env.ball_geom not in (contact.geom1, contact.geom2):
        continue

    other_geom = (
        contact.geom2 if contact.geom1 == env.ball_geom
        else contact.geom1
    )
    other_body = model.geom_bodyid[other_geom]

    print(
        " ",
        "geom=",
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other_geom),
        "body=",
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other_body),
        "distance=",
        round(float(contact.dist), 5),
    )
    found_contact = True

if not found_contact:
    print("  no ball contacts")