"""Physical reset invariants for published, independently constructed assistance."""
from __future__ import annotations

import mujoco
import numpy as np

from so101.robot_control import JOINTS
from so101.robot_table_clearance import require_robot_above_table, robot_body_ids


PROFILE = "static-other-task-assistance-v1"
LEVELS = tuple(i / 20 for i in range(20))


def scene_penetrations(sim, baseline: np.ndarray, target: int,
                       moved: set[int]) -> dict:
    """Distinguish new intersections from unchanged admitted piece compliance."""
    model, data = sim.model, sim.data
    robot = robot_body_ids(model)
    relevant = robot | {target} | moved
    pieces = set(sim.registry.values())
    jaws = {model.body(n).id for n in ("gripper", "moving_jaw_so101_v1")}
    table = model.geom("table").id
    original_depths = None
    inherited = {}
    maximum = 0.

    def unchanged_piece(body):
        joint = int(model.body_jntadr[body])
        if (body not in pieces or model.body_parentid[body] != 0 or joint < 0
                or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE):
            return False
        start = int(model.jnt_qposadr[joint])
        return np.array_equal(data.qpos[start:start + 7], baseline[start:start + 7])

    for contact in data.contact:
        pair = tuple(sorted((int(contact.geom1), int(contact.geom2))))
        bodies = {int(model.geom_bodyid[g]) for g in pair}
        jaw_pair = target in bodies and bool(bodies & jaws)
        if not bodies & relevant or table in pair or jaw_pair:
            continue
        depth = max(0., -float(contact.dist))
        if not depth:
            continue
        if not bodies & robot and all(unchanged_piece(body) for body in bodies):
            if original_depths is None:
                reference = mujoco.MjData(model)
                reference.qpos[:] = baseline
                mujoco.mj_forward(model, reference)
                original_depths = {}
                for original in reference.contact:
                    key = tuple(sorted((int(original.geom1), int(original.geom2))))
                    original_depths[key] = max(
                        original_depths.get(key, 0.), -float(original.dist))
            original_depth = original_depths.get(pair, 0.)
            # Source pile admission permits at most 5mm; never worsen that contact.
            if 0 < original_depth <= .005 and depth <= original_depth + 1e-9:
                inherited[pair] = max(inherited.get(pair, 0.), depth)
                continue
        maximum = max(maximum, depth)
    return {
        "maximum_unintended_penetration_m": maximum,
        "maximum_preserved_source_penetration_m": max(inherited.values(), default=0.),
        "preserved_source_contacts": [
            {"geom_ids": list(pair), "depth_m": depth}
            for pair, depth in sorted(inherited.items())],
    }


def require_outside_push(sim, measured: dict) -> float:
    model, data = sim.model, sim.data
    target = measured["target_body"]
    collision = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    pieces = np.flatnonzero(collision & (model.geom_bodyid == target))
    moving = np.flatnonzero(collision & (
        model.geom_bodyid == model.body("moving_jaw_so101_v1").id))
    if not len(pieces) or not len(moving):
        raise ValueError("PUSH requires target and opposing-jaw collision geometry")
    gap = min(mujoco.mj_geomDistance(model, data, int(a), int(b), .2, None)
              for a in moving for b in pieces)
    gripper = model.body("gripper").id
    axis = data.xmat[gripper].reshape(3, 3)[:, 0]
    tip = model.geom("fixed_jaw_sph_tip1").id
    outside_limit = float(data.geom_xpos[tip] @ axis) + .001
    for geom in pieces:
        direction = axis @ data.geom_xmat[geom].reshape(3, 3)
        if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[geom]
            start = model.mesh_vertadr[mesh]
            support = (model.mesh_vert[start:start + model.mesh_vertnum[mesh]] @ direction).max()
        elif model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX:
            support = np.abs(direction) @ model.geom_size[geom]
        else:
            raise ValueError("Unsupported PUSH target collision geometry")
        if float(data.geom_xpos[geom] @ axis + support) > outside_limit:
            raise ValueError("PUSH target extends into the jaw aperture")
    forces = measured["jaw_normal_forces_n"]
    if gap < .015 or forces.get("moving", 0) > 1e-5 or forces.get("fixed", 0) <= 1e-5:
        raise ValueError("PUSH requires loaded outside fixed-jaw-only contact")
    return float(gap)


def validate_reset(sim, problem: dict, parent: dict, measured: dict, contact_state) -> None:
    model, data = sim.model, sim.data
    task, original = problem["task"], parent["task"]
    assistance = task["assistance"]
    value = assistance.get("value")
    kind = task["task_type"]
    if (kind not in ("place", "push", "view") or original.get("parent_task_id")
            or assistance.get("schema_version") != 1
            or type(value) not in (int, float) or value not in LEVELS
            or task["parent_task_id"] != original["task_id"]):
        raise ValueError("Invalid other-task assistance parent, schema or value")
    for key in ("task_type", "target_instance_id", "goal", "grader_version",
                "initial_target_position"):
        if task.get(key) != original.get(key):
            raise ValueError(f"Assistance changed original task field: {key}")
    for key in ("runtime_model_sha256", "model_xml_sha256", "physics_version"):
        if task["initial_state"][key] != original["initial_state"][key]:
            raise ValueError(f"Assistance changed original physics identity: {key}")
    expected_progress = kind == "place" and value > .5
    if task["initial_task_state"] != {"lifted_once": expected_progress}:
        raise ValueError("Other-task progress disagrees with constructed completion")
    if (not all(np.isfinite(a).all() for a in (data.qpos, data.qvel, data.qacc, data.ctrl))
            or any(w.number for w in data.warning)):
        raise ValueError("Invalid other-task native physics state")
    require_robot_above_table(model, data)
    if max(measured["jaw_normal_forces_n"].values(), default=0) > 1:
        raise ValueError("Other-task assistance reset exceeds per-jaw 1N preload")
    joints = np.array([model.joint(name).id for name in JOINTS])
    qadr = model.jnt_qposadr[joints]
    target = sim.registry[task["target_instance_id"]]

    def address(body):
        joint = int(model.body_jntadr[body])
        if joint < 0 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("Assistance piece does not have a free joint")
        return int(model.jnt_qposadr[joint])

    baseline = np.asarray(original["initial_state"]["qpos"])
    if baseline.shape != data.qpos.shape:
        raise ValueError("Assistance parent/model qpos mismatch")
    target_address = address(target)
    ignored = set(map(int, qadr))
    if (kind == "push" and value > 0) or (kind == "place" and value > .5):
        ignored.update(range(target_address, target_address + 7))
    displaced = assistance.get("independently_displaced_other_pieces")
    if not isinstance(displaced, list) or (displaced and (kind != "push" or value == 0)):
        raise ValueError("Invalid non-target displacement policy")
    moved = set()
    for item in displaced:
        name = item["instance"]
        if name not in sim.registry or name == task["target_instance_id"] or name in moved:
            raise ValueError("Invalid or repeated displaced instance")
        moved.add(name)
        start = address(sim.registry[name])
        if (not np.array_equal(baseline[start:start + 7], item["original_qpos"])
                or not np.array_equal(data.qpos[start:start + 7], item["constructed_qpos"])
                or not contact_state(sim, name)["table_supported"]):
            raise ValueError("Invalid displaced-piece pose provenance or table support")
        ignored.update(range(start, start + 7))
    unchanged = [i for i in range(model.nq) if i not in ignored]
    if not np.array_equal(data.qpos[unchanged], baseline[unchanged]):
        raise ValueError("Other-task assistance changed an unapproved original pose")
    if value == 0 and not np.array_equal(data.qpos, baseline):
        raise ValueError("Zero assistance differs from the original reset")
    penetration = scene_penetrations(sim, baseline, target, {sim.registry[name] for name in moved})
    if penetration["maximum_unintended_penetration_m"] > 1e-6:
        raise ValueError("Other-task reset has an unintended scene intersection")
    if kind == "place":
        if .5 <= value <= .9 and not measured["bilateral_grip"]:
            raise ValueError("PLACE reset lacks bilateral loaded grip")
        if .5 < value <= .9 and measured["table_supported"]:
            raise ValueError("Post-lift PLACE reset is still table-supported")
        if value == .95:
            error = np.linalg.norm(sim.position(task["target_instance_id"])[:2]
                                   - np.asarray(task["goal"]["destination_xy"]))
            if (not measured["table_supported"] or measured["jaw_normal_forces_n"]
                    or error > task["goal"]["tolerance_m"]):
                raise ValueError("Released PLACE reset lacks original-goal support/release")
    elif kind == "push":
        if measured["bilateral_grip"] or (value > 0 and not measured["table_supported"]):
            raise ValueError("PUSH reset is grasped or lacks table support")
        if value > 0:
            limits = model.jnt_range[joints[-1]]
            opened = limits[0] + .85 * np.ptp(limits)
            if not np.isclose(data.qpos[qadr[-1]], opened, atol=1e-12, rtol=0):
                raise ValueError("PUSH reset does not preserve open jaws")
        if .5 <= value <= .9:
            require_outside_push(sim, {**measured, "target_body": target})
        if value == .95 and measured["jaw_normal_forces_n"]:
            raise ValueError("Withdrawn PUSH reset still contacts target with jaws")
    elif assistance.get("native_view_qualified") is not True:
        raise ValueError("VIEW assistance requires native wrist segmentation qualification")
