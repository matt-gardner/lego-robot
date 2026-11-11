"""Check robot geometry against the actual table plane, including visual meshes."""
from __future__ import annotations

import mujoco
import numpy as np


def robot_body_ids(model: mujoco.MjModel) -> set[int]:
    bodies = {model.body("base").id}
    for body in range(1, model.nbody):
        if int(model.body_parentid[body]) in bodies:
            bodies.add(body)
    return bodies


def robot_table_clearance(model: mujoco.MjModel, data: mujoco.MjData) -> dict:
    table = model.geom("table").id
    if model.geom_type[table] != mujoco.mjtGeom.mjGEOM_PLANE:
        raise ValueError("Robot table-clearance check requires the native table plane")
    normal = data.geom_xmat[table].reshape(3, 3)[:, 2]
    origin = data.geom_xpos[table]
    bodies = robot_body_ids(model)
    minimum, limiting = float("inf"), None
    count = 0
    for geom in range(model.ngeom):
        if int(model.geom_bodyid[geom]) not in bodies:
            continue
        if not (model.geom_contype[geom] or model.geom_conaffinity[geom]
                or model.geom_group[geom] == 2):
            continue
        rotation = data.geom_xmat[geom].reshape(3, 3)
        direction = normal @ rotation
        size = model.geom_size[geom]
        kind = model.geom_type[geom]
        center = float(normal @ (data.geom_xpos[geom] - origin))
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[geom]
            start = model.mesh_vertadr[mesh]
            vertices = model.mesh_vert[start:start + model.mesh_vertnum[mesh]]
            gap = center + float((vertices @ direction).min())
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            gap = center - float(np.abs(direction) @ size)
        elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            gap = center - float(size[0])
        elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            gap = center - float(np.linalg.norm(direction * size))
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            gap = center - float(size[0] + size[1] * abs(direction[2]))
        elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            gap = center - float(size[1] * abs(direction[2])
                                 + size[0] * np.linalg.norm(direction[:2]))
        else:
            raise ValueError(f"Unsupported robot geometry for table clearance: {kind}")
        count += 1
        if gap < minimum:
            minimum, limiting = gap, geom
    if not count or not np.isfinite(minimum):
        raise ValueError("Robot table-clearance check found no finite robot geometry")
    return {
        "minimum_clearance_m": minimum,
        "limiting_geom_id": limiting,
        "limiting_body": model.body(int(model.geom_bodyid[limiting])).name,
        "checked_geometry_count": count,
        "includes_visual_meshes": True,
    }


def require_robot_above_table(model: mujoco.MjModel, data: mujoco.MjData) -> dict:
    clearance = robot_table_clearance(model, data)
    # One nanometre permits floating-point noise, not physical penetration.
    if clearance["minimum_clearance_m"] < -1e-9:
        raise ValueError(
            "Robot intersects table: "
            f"{clearance['limiting_body']} clearance "
            f"{clearance['minimum_clearance_m']:.9f}m"
        )
    return clearance
