"""Assemble and settle catalog scenes without a policy or expert demonstration."""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from so101.cloud_checkpoint import sha256
from so101.lego_curriculum import Catalog


REFERENCES = {
    "name", "mesh", "material", "texture", "class", "childclass",
    "joint", "joint1", "joint2", "body", "body1", "body2", "geom",
    "geom1", "geom2", "site", "site1", "site2", "target",
}
RGBA = {
    "gray": [.38, .39, .40, 1.], "green": [.04, .48, .12, 1.],
    "orange": [.93, .34, .04, 1.], "navy": [.03, .10, .32, 1.],
    "teal": [.02, .55, .58, 1.], "brown": [.36, .13, .05, 1.],
}


def _namespace(
    element: ET.Element, prefix: str, directory: Path, allowed_assets: dict[Path, str],
) -> None:
    for node in element.iter():
        for key in REFERENCES & node.attrib.keys():
            node.set(key, " ".join(prefix + value for value in node.attrib[key].split()))
        if "file" in node.attrib:
            relative = Path(node.attrib["file"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Part template asset path must be contained and relative")
            asset = directory / relative
            if (not asset.is_file() or asset.is_symlink()
                    or asset.resolve() not in allowed_assets):
                raise ValueError(f"Missing or invalid part template asset: {asset}")
            if sha256(asset) != allowed_assets[asset.resolve()]:
                raise ValueError(f"Part asset changed after catalog validation: {asset}")
            node.set("file", str(asset.resolve()))


def assemble_scene(
    base_xml: str, catalog: Catalog, plan: dict,
    *, scene_origin_m: tuple[float, float, float],
) -> str:
    """Import qualified single-body templates into a caller-owned base scene."""
    if (plan.get("catalog_sha256") != catalog.manifest_sha256
            or plan.get("catalog_id") != catalog.catalog_id
            or plan.get("status") != "proposed"):
        raise ValueError("Scene proposal does not match verified catalog")
    origin = np.asarray(scene_origin_m, dtype=float)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("Scene origin must contain three finite metre coordinates")
    root = ET.fromstring(base_xml)
    if root.tag != "mujoco":
        raise ValueError("Expected base MuJoCo XML")
    base_compiler = root.find("compiler")
    if base_compiler is None or base_compiler.get("angle", "degree") != "radian":
        raise ValueError("Base scene must explicitly use radian compiler angles")
    world = root.find("worldbody")
    if world is None:
        raise ValueError("Base scene has no worldbody")
    assets = root.find("asset")
    if assets is None:
        assets = ET.SubElement(root, "asset")
    parts = {part.part_id: part for part in catalog.parts}
    objects = plan.get("objects")
    if not isinstance(objects, list) or not objects:
        raise ValueError("Scene proposal has no objects")
    names = {node.get("name") for node in root.iter() if node.get("name")}
    instance_ids = set()
    for obj in objects:
        instance = obj["instance_id"]
        if (not isinstance(instance, str) or not instance
                or not all(character.isalnum() or character == "_" for character in instance)
                or instance in instance_ids):
            raise ValueError("Invalid or duplicate object instance identity")
        instance_ids.add(instance)
        if obj["part_id"] not in parts:
            raise ValueError(f"Unverified part in scene: {obj['part_id']}")
        part = parts[obj["part_id"]]
        template_path = catalog.root / part.mjcf
        allowed_assets = {catalog.root / name: digest for name, digest in part.assets}
        if sha256(template_path) != allowed_assets[template_path]:
            raise ValueError("Part template changed after catalog validation")
        template = ET.fromstring(template_path.read_bytes())
        if template.tag != "mujoco" or any(
            child.tag not in {"compiler", "asset", "worldbody"} for child in template
        ):
            raise ValueError("Part template must not introduce global settings or defaults")
        compiler = template.find("compiler")
        if compiler is not None and (
            compiler.get("angle", "degree") != "radian"
            or any(key not in {"angle", "autolimits"} for key in compiler.attrib)
        ):
            raise ValueError("Unsupported part compiler settings")
        template_world = template.find("worldbody")
        if (template_world is None or len(template_world) != 1
                or template_world[0].tag != "body"):
            raise ValueError("Part template needs exactly one dynamic body")
        body = copy.deepcopy(template_world[0])
        if body.findall("body") or len(body.findall("freejoint")) != 1 or body.findall("joint"):
            raise ValueError("Part template needs one free body without nested bodies")
        prefix = instance + "__"
        _namespace(body, prefix, template_path.parent, allowed_assets)
        body.set("name", instance)
        if any(node.get("name") in names for node in body.iter() if node.get("name")):
            raise ValueError("Part instance conflicts with base scene names")
        position = np.asarray(obj["position_relative_to_scene_m"], dtype=float)
        yaw = obj["yaw_rad"]
        if (position.shape != (3,) or not np.isfinite(position).all()
                or type(yaw) not in (int, float) or not math.isfinite(yaw)):
            raise ValueError("Invalid object spawn pose")
        body.set("pos", " ".join(map(str, position + origin)))
        body.set("quat", f"{math.cos(yaw / 2)} 0 0 {math.sin(yaw / 2)}")
        if obj.get("color") not in RGBA:
            raise ValueError("Unknown LEGO color")
        for geom in body.findall("geom"):
            if geom.get("contype") == "0" and geom.get("conaffinity") == "0":
                geom.set("rgba", " ".join(map(str, RGBA[obj["color"]])))
        template_assets = template.find("asset")
        if template_assets is not None:
            for source_asset in template_assets:
                imported = copy.deepcopy(source_asset)
                _namespace(imported, prefix, template_path.parent, allowed_assets)
                if any(node.get("name") in names for node in imported.iter() if node.get("name")):
                    raise ValueError("Part asset conflicts with base scene names")
                assets.append(imported)
                names.update(node.get("name") for node in imported.iter() if node.get("name"))
        world.append(body)
        names.update(node.get("name") for node in body.iter() if node.get("name"))
    return ET.tostring(root, encoding="unicode")


def settle_scene(
    xml: str, plan: dict, *, maximum_seconds: float = 10.,
    stationary_seconds: float = .5, velocity_tolerance: float = .005,
    contact_limit_m: float = .001,
) -> tuple[mujoco.MjModel, mujoco.MjData, dict]:
    """Require finite, stationary physics with acceptable interval penetration."""
    limits = (maximum_seconds, stationary_seconds, velocity_tolerance, contact_limit_m)
    if any(not math.isfinite(value) or value <= 0 for value in limits):
        raise ValueError("Settling parameters must be positive and finite")
    model = mujoco.MjModel.from_xml_string(xml)
    if not math.isfinite(model.opt.timestep) or model.opt.timestep <= 0:
        raise ValueError("Invalid physics timestep")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    addresses = []
    for obj in plan["objects"]:
        body = model.body(obj["instance_id"]).id
        joint = int(model.body_jntadr[body])
        if joint < 0 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("Catalog scene object is not a free dynamic body")
        addresses.append((obj, int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])))
    maximum_penetration = 0.
    stationary_since = None
    while data.time < maximum_seconds:
        penetration = max((max(0., -float(contact.dist)) for contact in data.contact), default=0.)
        maximum_penetration = max(maximum_penetration, penetration)
        if penetration > contact_limit_m:
            raise ValueError(f"Scene contact penetration exceeded limit: {penetration:.6f}m")
        if not all(np.isfinite(value).all() for value in (data.qpos, data.qvel, data.qacc)):
            raise ValueError("Non-finite catalog scene physics")
        if any(warning.number for warning in data.warning):
            raise ValueError("MuJoCo emitted a warning while settling catalog scene")
        moving = any(np.max(np.abs(data.qvel[v:v + 6])) > velocity_tolerance
                     for _, _, v in addresses)
        stationary_since = None if moving else (
            data.time if stationary_since is None else stationary_since
        )
        if stationary_since is not None and data.time - stationary_since >= stationary_seconds:
            break
        mujoco.mj_step(model, data)
    else:
        raise ValueError("Catalog scene did not settle before physics timeout")
    objects = [
        {"instance_id": obj["instance_id"], "part_id": obj["part_id"],
         "qpos": data.qpos[q:q + 7].tolist(), "qvel": data.qvel[v:v + 6].tolist()}
        for obj, q, v in addresses
    ]
    snapshot = {
        "schema_version": 1, "scene_id": plan["scene_id"], "status": "settled",
        "catalog_sha256": plan["catalog_sha256"],
        "model_xml_sha256": hashlib.sha256(xml.encode()).hexdigest(),
        "physics_version": mujoco.__version__, "time": float(data.time),
        "maximum_penetration_m": maximum_penetration, "objects": objects,
        "qpos": data.qpos.tolist(), "qvel": data.qvel.tolist(),
        "act": data.act.tolist(), "ctrl": data.ctrl.tolist(),
    }
    specification = mujoco.mjtState.mjSTATE_INTEGRATION
    integration = np.empty(mujoco.mj_stateSize(model, specification))
    mujoco.mj_getState(model, data, integration, specification)
    snapshot["integration_state"] = integration.tolist()
    return model, data, snapshot


def restore_scene(model: mujoco.MjModel, data: mujoco.MjData, snapshot: dict, xml: str) -> None:
    """Restore a settled state only into its exact compiled model."""
    if snapshot.get("model_xml_sha256") != hashlib.sha256(xml.encode()).hexdigest():
        raise ValueError("Snapshot belongs to a different scene model")
    if snapshot.get("physics_version") != mujoco.__version__:
        raise ValueError("Snapshot uses a different physics version")
    values = {}
    for key in ("qpos", "qvel", "act", "ctrl"):
        value = np.asarray(snapshot[key], dtype=float)
        if value.shape != getattr(data, key).shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid snapshot {key}")
        values[key] = value
    time = snapshot["time"]
    if type(time) not in (int, float) or not math.isfinite(time) or time < 0:
        raise ValueError("Invalid snapshot time")
    specification = mujoco.mjtState.mjSTATE_INTEGRATION
    integration = np.asarray(snapshot["integration_state"], dtype=float)
    if (integration.shape != (mujoco.mj_stateSize(model, specification),)
            or not np.isfinite(integration).all()):
        raise ValueError("Invalid snapshot integration state")
    restored = mujoco.MjData(model)
    mujoco.mj_setState(model, restored, integration, specification)
    if restored.time != time or any(
        not np.array_equal(getattr(restored, key), value) for key, value in values.items()
    ):
        raise ValueError("Snapshot integration state disagrees with recorded state")
    mujoco.mj_resetData(model, data)
    mujoco.mj_setState(model, data, integration, specification)
    mujoco.mj_forward(model, data)
