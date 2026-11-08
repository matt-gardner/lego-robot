"""Render verified molded parts with explicit visual-only cable composition."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np
from OpenGL import GL

from so101.lego_curriculum import load_catalog
from so101.lego_scene import assemble_scene
from so101.qualify_renderer import table_point
from so101.renderer_appearance_v2 import WristAppearanceV2
from so101.robot_sim import MODEL_ROOT, OBJECT_NAMES, TABLE_Z, BeamSimulation, scene_xml


def qualify_catalog(manifest: Path, output: Path) -> None:
    catalog = load_catalog(manifest)
    parts = {part.part_id: part for part in catalog.parts}
    if not {"3003", "3022", "3700"} <= parts.keys():
        raise ValueError("Molded appearance fixtures require verified 3003, 3022 and 3700")
    output.mkdir(parents=True, exist_ok=False)
    profile = WristAppearanceV2()
    baseline = BeamSimulation(render=False)
    try:
        baseline.reset(10)
        grasp = baseline.data.site_xpos[baseline.site].copy()
        grasp[2] = .010
        arm = baseline.ik(grasp, 0, rng=np.random.default_rng(10))
        arm[-1] = baseline.limits[-1, 0] + .3 * np.ptp(baseline.limits[-1])
        plan = {
            "status": "proposed", "catalog_id": catalog.catalog_id,
            "catalog_sha256": catalog.manifest_sha256,
            "objects": [{"instance_id": name, "part_id": part_id, "color": color,
                         "position_relative_to_scene_m": [.8, .5 + index * .1, .1],
                         "yaw_rad": 0}
                        for index, (name, part_id, color) in enumerate((
                            ("orange_brick", "3003", "orange"),
                            ("orange_plate", "3022", "orange"),
                            ("gray_technic", "3700", "gray"),
                        ))],
        }
        xml = assemble_scene(scene_xml(MODEL_ROOT), catalog, plan, scene_origin_m=(0, 0, 0))
        root = ET.fromstring(profile.apply_xml(xml))
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
            for obj, center in zip(plan["objects"], ((.30, .02), (.99, .39), (.52, .76)), strict=True):
                name = obj["instance_id"]
                part = parts[obj["part_id"]]
                template = ET.fromstring((catalog.root / part.mjcf).read_bytes())
                bounds = []
                for mesh in template.findall("asset/mesh"):
                    vertices = np.array(list(map(float, mesh.get("vertex", "").split())))
                    if vertices.size:
                        bounds.extend(vertices.reshape(-1, 3)[:, 2])
                if not bounds:
                    raise ValueError(f"Fixture requires explicit verified mesh vertices: {part.part_id}")
                xyz = table_point(sim, center, TABLE_Z - min(bounds))
                body = sim.model.body(name).id
                joint = int(sim.model.body_jntadr[body])
                address = int(sim.model.jnt_qposadr[joint])
                sim.data.qpos[address:address + 7] = [*xyz, 1, 0, 0, 0]
                sim.beam_geoms[name] = np.flatnonzero(sim.model.geom_bodyid == body)
            mujoco.mj_forward(sim.model, sim.data)
            points = [table_point(sim, center, TABLE_Z + .0015)
                      for center in ((-.05, .90), (.02, .89), (.08, .87), (.14, .87),
                                     (.18, .90), (.20, .94))]
            final_xml = profile.add_visual_cable(ET.tostring(root, encoding="unicode"), np.array(points))
            final_model = mujoco.MjModel.from_xml_string(final_xml)
            profile.apply_model(final_model)
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
                labels = {}
                for obj in plan["objects"]:
                    body = final_model.body(obj["instance_id"]).id
                    mask = np.isin(segmentation[:, :, 0], np.flatnonzero(final_model.geom_bodyid == body))
                    labels[obj["instance_id"]] = {
                        "part_id": obj["part_id"], "visible_pixels": int(mask.sum()),
                        "status": "catalog-verified geometry, render-only pose",
                    }
                report = {"renderer": profile.identity(), "catalog_id": catalog.catalog_id,
                          "catalog_sha256": catalog.manifest_sha256,
                          "assembled_xml_sha256": hashlib.sha256(final_xml.encode()).hexdigest(),
                          "mujoco_version": mujoco.__version__, "gl_renderer": backend,
                          "qpos": data.qpos.tolist(), "labels": labels,
                          "qualification": "Actual verified molded parts, NOT the identified real "
                          "teal assembly. Hand-composed render-only poses; no settling/expert success "
                          "claim. Cable is explicitly non-contact visual context. Jaw outer housings "
                          "are uncalibrated cosmetic geometry; original inner contacts unchanged."}
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
    args = parser.parse_args()
    qualify_catalog(args.catalog, args.output)


if __name__ == "__main__":
    main()
