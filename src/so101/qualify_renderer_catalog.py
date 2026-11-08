"""Render verified molded parts with explicit visual-only cable composition."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np
from OpenGL import GL

from so101.lego_curriculum import LDRAW_TO_MUJOCO, Part, _asset, load_catalog
from so101.lego_scene import _namespace, assemble_scene
from so101.qualify_renderer import table_point
from so101.renderer_appearance_v2 import WristAppearanceV2
from so101.renderer_appearance_v3 import WristAppearanceV3
from so101.robot_sim import MODEL_ROOT, OBJECT_NAMES, TABLE_Z, BeamSimulation, scene_xml


def qualify_catalog(manifest: Path, output: Path, *, reference_assembly: bool = False,
                    appearance: WristAppearanceV2 | None = None,
                    diagnostic_unqualified_separator: bool = False) -> None:
    catalog = load_catalog(manifest)
    parts = {part.part_id: part for part in catalog.parts}
    required = {"3003", "96874", "3702"} if reference_assembly else {"3003", "3022", "3700"}
    diagnostic = None
    if diagnostic_unqualified_separator:
        if not reference_assembly:
            raise ValueError("Unqualified diagnostic geometry requires explicit reference assembly")
        entry = next(part for part in json.loads(manifest.read_bytes())["parts"] if part["part_id"] == "96874")
        if (entry["status"] != "unavailable" or entry.get("failure", {}).get("stage") != "qualification"
                or entry["geometry"]["frame_ldraw_to_mujoco"] != LDRAW_TO_MUJOCO
                or entry["geometry"]["metres_per_ldu"] != .0004):
            raise ValueError("Diagnostic separator requires declared failed physics and correct visual frame")
        assets = []
        for asset in entry["assets"]:
            name, path = _asset(catalog.root, asset["path"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != asset["sha256"]:
                raise ValueError(f"Diagnostic visual asset hash mismatch: {name}")
            assets.append((name, asset["sha256"]))
        if entry["mjcf"] not in {name for name, _ in assets}:
            raise ValueError("Diagnostic MJCF is not covered by declared assets")
        parts["96874"] = Part("96874", entry["family"], tuple(entry["dimensions_m"]),
                              entry["mjcf"], tuple(assets))
        diagnostic = {"part_id": "96874", "status": "unavailable-static-visual-only",
                      "failure": entry["failure"], "assets": entry["assets"],
                      "authors": entry["authors"], "licenses": entry["licenses"]}
    if not required <= parts.keys():
        raise ValueError(f"Appearance fixture requires verified parts: {sorted(required - parts.keys())}")
    output.mkdir(parents=True, exist_ok=False)
    profile = appearance if appearance is not None else WristAppearanceV2()
    baseline = BeamSimulation(render=False)
    try:
        baseline.reset(10)
        grasp = baseline.data.site_xpos[baseline.site].copy()
        grasp[2] = .015 if reference_assembly else .010
        arm = baseline.ik(grasp, 0, rng=np.random.default_rng(10))
        arm[-1] = baseline.limits[-1, 0] + .3 * np.ptp(baseline.limits[-1])
        selected = (("orange_brick", "3003", "orange"),
                    ("teal_separator", "96874", "teal"),
                    ("gray_beam", "3702", "gray"),
                    ("gray_extension", "3702", "gray"),
                    ("gray_neighbor", "3702", "gray")) if reference_assembly else (
                        ("orange_brick", "3003", "orange"),
                        ("orange_plate", "3022", "orange"),
                        ("gray_technic", "3700", "gray"),
                    )
        objects = [{"instance_id": name, "part_id": part_id, "color": color,
                    "position_relative_to_scene_m": [.8, .5 + index * .1, .1],
                    "yaw_rad": 0} for index, (name, part_id, color) in enumerate(selected)]
        plan = {
            "status": "proposed", "catalog_id": catalog.catalog_id,
            "catalog_sha256": catalog.manifest_sha256,
            "objects": [obj for obj in objects if not (
                diagnostic is not None and obj["part_id"] == "96874"
            )],
        }
        xml = assemble_scene(scene_xml(MODEL_ROOT), catalog, plan, scene_origin_m=(0, 0, 0))
        root = ET.fromstring(profile.apply_xml(xml))
        if diagnostic is not None:
            part = parts["96874"]
            template_path = catalog.root / part.mjcf
            template = ET.fromstring(template_path.read_bytes())
            source_body = template.find("worldbody/body")
            if source_body is None or source_body.findall("body"):
                raise ValueError("Unsupported diagnostic visual body")
            body = copy.deepcopy(source_body)
            for child in list(body):
                if child.tag != "geom" or child.get("contype") != "0" or child.get("conaffinity") != "0":
                    body.remove(child)
                else:
                    child.set("rgba", ".015 .40 .43 1")
                    child.set("mass", "0")
            if not body.findall("geom"):
                raise ValueError("Diagnostic separator has no explicit non-contact visual geometry")
            meshes = {geom.get("mesh") for geom in body.findall("geom")}
            allowed = {catalog.root / name: digest for name, digest in part.assets}
            for mesh in template.findall("asset/mesh"):
                if mesh.get("name") in meshes:
                    imported = copy.deepcopy(mesh)
                    _namespace(imported, "teal_separator__", template_path.parent, allowed)
                    root.find("asset").append(imported)
            _namespace(body, "teal_separator__", template_path.parent, allowed)
            body.set("name", "teal_separator")
            root.find("worldbody").append(body)
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        profile.apply_model(model)
        sim = BeamSimulation(model=model, render=False)
        try:
            sim.appearance = profile
            sim.data.qpos[sim.qadr] = arm
            sim.data.ctrl[sim.actuators] = arm
            sim.initial = {"seed": 10, "renderer": profile.identity()}
            for index, name in enumerate(OBJECT_NAMES):
                address = sim.model.joint(f"{name}_free").qposadr[0]
                sim.data.qpos[address:address + 3] = [.8, .5 + index * .2, .1]
            mujoco.mj_forward(sim.model, sim.data)
            centers = ((.30, .02), (.50, .76), (.31, .76), (.69, .76), (.93, .52)) if reference_assembly else (
                (.30, .02), (.99, .39), (.52, .76),
            )
            for obj, center in zip(objects, centers, strict=True):
                name = obj["instance_id"]
                part = parts[obj["part_id"]]
                template = ET.fromstring((catalog.root / part.mjcf).read_bytes())
                vertices = []
                for mesh in template.findall("asset/mesh"):
                    values = np.array(list(map(float, mesh.get("vertex", "").split())))
                    if values.size:
                        vertices.extend(values.reshape(-1, 3))
                if not vertices:
                    raise ValueError(f"Fixture requires explicit verified mesh vertices: {part.part_id}")
                bounds = np.array(vertices)
                low, high = bounds.min(axis=0), bounds.max(axis=0)
                yaw = 0.
                surface = TABLE_Z
                if reference_assembly:
                    camera = sim.model.camera("wrist_cam").id
                    camera_rotation = sim.data.cam_xmat[camera].reshape(3, 3)
                    yaw = float(np.arctan2(camera_rotation[1, 0], camera_rotation[0, 0]))
                    if part.dimensions_m[1] > part.dimensions_m[0]:
                        yaw -= np.pi / 2
                    if name == "gray_neighbor":
                        yaw += .6
                    if name == "teal_separator":
                        surface += .0096
                        yaw += np.pi - .7
                xyz = table_point(sim, center, surface - low[2])
                if reference_assembly:
                    xyz = table_point(sim, center, surface + (high[2] - low[2]) / 2)
                    xyz[2] = surface - low[2]
                    offset = (low + high) / 2
                    xyz[:2] -= np.array([[np.cos(yaw), -np.sin(yaw)],
                                         [np.sin(yaw), np.cos(yaw)]]) @ offset[:2]
                body = sim.model.body(name).id
                joint = int(sim.model.body_jntadr[body])
                quaternion = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
                if joint < 0:
                    if diagnostic is None or name != "teal_separator":
                        raise ValueError(f"Unexpected non-dynamic fixture object: {name}")
                    sim.model.body_pos[body] = xyz
                    sim.model.body_quat[body] = quaternion
                    body_element = root.find(f"worldbody/body[@name='{name}']")
                    body_element.set("pos", " ".join(map(str, xyz)))
                    body_element.set("quat", " ".join(map(str, quaternion)))
                else:
                    address = int(sim.model.jnt_qposadr[joint])
                    sim.data.qpos[address:address + 7] = [*xyz, *quaternion]
                sim.beam_geoms[name] = np.flatnonzero(sim.model.geom_bodyid == body)
            mujoco.mj_forward(sim.model, sim.data)
            points = [table_point(sim, center, TABLE_Z + .0015)
                      for center in ((-.05, .90), (.02, .89), (.08, .87), (.14, .87),
                                     (.18, .90), (.20, .94))]
            final_xml = profile.add_visual_cable(ET.tostring(root, encoding="unicode"), np.array(points))
            final_model = mujoco.MjModel.from_xml_string(final_xml)
            profile.apply_model(final_model)
            if diagnostic is not None:
                separator_body = final_model.body("teal_separator").id
                separator_geoms = np.flatnonzero(final_model.geom_bodyid == separator_body)
                if (final_model.body_jntnum[separator_body] != 0
                        or final_model.geom_contype[separator_geoms].any()
                        or final_model.geom_conaffinity[separator_geoms].any()):
                    raise ValueError("Diagnostic separator must remain static and non-contact")
                diagnostic["compiled_static_noncontact_verified"] = True
            data = mujoco.MjData(final_model)
            data.qpos[:] = sim.data.qpos
            data.ctrl[:] = sim.data.ctrl
            mujoco.mj_forward(final_model, data)
            renderer = mujoco.Renderer(final_model, height=240, width=320)
            try:
                backend = GL.glGetString(GL.GL_RENDERER).decode()
                if "llvmpipe" not in backend.lower() and "softpipe" not in backend.lower():
                    raise RuntimeError(f"CPU fixture requires software renderer, got {backend}")
                image = profile.process_rgb(profile.render_rgb(renderer, final_model, data), seed=10, frame=0)
                if not cv2.imwrite(str(output / "molded_parts_and_cable.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
                    raise OSError("Failed to save molded part fixture")
                renderer.update_scene(data, camera="wrist_cam")
                renderer.enable_segmentation_rendering()
                try:
                    segmentation = renderer.render().copy()
                finally:
                    renderer.disable_segmentation_rendering()
                renderer.enable_depth_rendering()
                try:
                    renderer.update_scene(data, camera="wrist_cam")
                    depth = renderer.render().copy()
                finally:
                    renderer.disable_depth_rendering()
                np.savez_compressed(output / "synchronized_labels.npz",
                                    segmentation=segmentation, depth_m=depth)
                labels = {}
                for obj in objects:
                    body = final_model.body(obj["instance_id"]).id
                    mask = np.isin(segmentation[:, :, 0], np.flatnonzero(final_model.geom_bodyid == body))
                    labels[obj["instance_id"]] = {
                        "part_id": obj["part_id"], "visible_pixels": int(mask.sum()),
                        "status": ("unavailable-static-visual-only" if diagnostic is not None
                                   and obj["part_id"] == "96874" else
                                   "catalog-verified geometry, render-only pose"),
                    }
                report = {"renderer": profile.identity(), "catalog_id": catalog.catalog_id,
                          "catalog_sha256": catalog.manifest_sha256,
                          "assembled_xml_sha256": hashlib.sha256(final_xml.encode()).hexdigest(),
                          "mujoco_version": mujoco.__version__, "gl_renderer": backend,
                          "qpos": data.qpos.tolist(), "labels": labels,
                          "reference_assembly_hypothesis": reference_assembly,
                          "reference_composition_notes": (
                              "Three distinct 3702 instances supply gray context without scaling "
                              "or pretending a single verified longer part. "
                              + ("Teal diagnostic RGB is .015/.40/.43. "
                                 if diagnostic is not None else "Teal uses the scene palette. ")
                              + "No calibrated reflectance claim."
                              if reference_assembly else None),
                          "diagnostic_unqualified_geometry": diagnostic,
                          "qualification": ("Verified molded parts plus explicitly UNQUALIFIED static separator visual; "
                                            if diagnostic is not None else "Actual verified molded parts; ")
                          + "photo part identities are hypotheses, not established. "
                          "Hand-composed render-only poses; no settling/expert success "
                          "claim. Cable is explicitly non-contact visual context. Jaw outer housings "
                          "are uncalibrated cosmetic geometry; original inner contacts unchanged."}
                reference = Path("data/real_robot/wrist_camera/target_teal_current_wrist.jpg")
                if reference_assembly:
                    reference_image = cv2.imread(str(reference))
                    if reference_image is None:
                        raise OSError(f"Cannot read reference image: {reference}")
                    comparison = np.concatenate((
                        cv2.resize(reference_image, (320, 240)),
                        cv2.cvtColor(image, cv2.COLOR_RGB2BGR)), axis=1)
                    if not cv2.imwrite(str(output / "reference_vs_simulation.png"), comparison):
                        raise OSError("Failed to save reference comparison")
                    report["reference_sha256"] = hashlib.sha256(reference.read_bytes()).hexdigest()
                (output / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")
            finally:
                renderer.close()
        finally:
            sim.close()
    finally:
        baseline.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-assembly", action="store_true")
    parser.add_argument("--appearance", choices=("v2", "v3"), default="v2")
    parser.add_argument("--diagnostic-unqualified-separator", action="store_true")
    args = parser.parse_args()
    qualify_catalog(args.catalog, args.output, reference_assembly=args.reference_assembly,
                    appearance=WristAppearanceV3() if args.appearance == "v3" else None,
                    diagnostic_unqualified_separator=args.diagnostic_unqualified_separator)


if __name__ == "__main__":
    main()
