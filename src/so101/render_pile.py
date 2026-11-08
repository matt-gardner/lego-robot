"""Render a saved pile with appearance-only changes; never advance its physics."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np

from so101.lego_scene import runtime_model_sha256


DEFAULT_PROFILE = Path(__file__).with_name("renderer_profiles") / "pile_photo_v1.json"


def smooth_mesh_normals(mesh: ET.Element, crease_angle_deg: float) -> np.ndarray:
    """Smooth shallow faceting while retaining hard molded edges and all vertices."""
    if not 0 < crease_angle_deg < 90:
        raise ValueError("Normal crease angle must lie between zero and 90 degrees")
    vertices = np.fromstring(mesh.attrib["vertex"], sep=" ").reshape(-1, 3)
    faces = np.fromstring(mesh.attrib["face"], sep=" ", dtype=int).reshape(-1, 3)
    triangles = vertices[faces]
    weighted = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    magnitudes = np.linalg.norm(weighted, axis=1)
    if np.any(magnitudes == 0):
        raise ValueError("Cannot shade a degenerate mesh triangle")
    face_normals = weighted / magnitudes[:, None]
    _, groups = np.unique(np.round(vertices, 12), axis=0, return_inverse=True)
    adjacent: dict[int, set[int]] = {}
    for face_index, face in enumerate(faces):
        for vertex in face:
            adjacent.setdefault(int(groups[vertex]), set()).add(face_index)
    normals = []
    cosine = np.cos(np.deg2rad(crease_angle_deg))
    for face_index, face in enumerate(faces):
        for vertex in face:
            candidates = np.array(sorted(adjacent[int(groups[vertex])]))
            candidates = candidates[face_normals[candidates] @ face_normals[face_index] >= cosine]
            normal = weighted[candidates].sum(axis=0)
            normal /= np.linalg.norm(normal)
            normals.append(normal)
    return np.asarray(normals)


def render_pile(source: Path, output: Path, profile: Path = DEFAULT_PROFILE) -> dict:
    config_bytes = profile.read_bytes()
    config = json.loads(config_bytes)
    if config["schema_version"] != 1:
        raise ValueError("Unsupported pile appearance profile")
    width, height, scale = (config[key] for key in
                            ("image_width", "image_height", "supersampling"))
    if any(type(value) is not int or value <= 0 for value in (width, height, scale)):
        raise ValueError("Image dimensions and supersampling must be positive integers")
    occupancy = config["frame_occupancy"]
    if not 0 < occupancy < 1:
        raise ValueError("Frame occupancy must lie between zero and one")
    light_samples = config["light_samples"]
    if type(light_samples) is not int or light_samples <= 0:
        raise ValueError("Light sample count must be a positive integer")
    original_xml = (source / "model.xml").read_bytes()
    original = mujoco.MjModel.from_xml_string(original_xml)
    root = ET.fromstring(original_xml)
    asset = root.find("asset")
    visual = root.find("visual")
    world = root.find("worldbody")
    if asset is None or visual is None or world is None:
        raise ValueError("Pile model lacks assets, visual settings or worldbody")
    global_settings = visual.find("global")
    if global_settings is None:
        global_settings = ET.SubElement(visual, "global")
    global_settings.set("offwidth", str(width * scale))
    global_settings.set("offheight", str(height * scale))
    headlight = visual.find("headlight")
    if headlight is None:
        headlight = ET.SubElement(visual, "headlight")
    headlight.set("ambient", ".25 .25 .25")
    headlight.set("diffuse", ".15 .15 .15")
    headlight.set("specular", "0 0 0")
    material = asset.find("material[@name='table_material']")
    if material is None:
        raise ValueError("Pile model lacks table material")
    material.set("texrepeat", "6 6")
    material.set("texuniform", "true")
    material.set("specular", ".04")
    material.set("shininess", ".1")
    ET.SubElement(asset, "material", name="pile_photo_plastic", rgba="1 1 1 1",
                  specular=str(config["plastic_specular"]),
                  shininess=str(config["plastic_shininess"]))
    for geom in world.findall(".//geom"):
        if geom.get("type") == "mesh" and geom.get("rgba", "").split()[-1:] != ["0"]:
            geom.set("material", "pile_photo_plastic")
    normal_overrides = {}
    for mesh in asset.findall("mesh"):
        if "vertex" in mesh.attrib and "face" in mesh.attrib:
            normal_overrides[mesh.attrib["name"]] = smooth_mesh_normals(
                mesh, config["normal_crease_angle_deg"],
            )
    for light in list(world.findall("light")):
        world.remove(light)
    ET.SubElement(world, "light", name="pile_photo_key", pos=" ".join(map(str, config["light_position_m"])),
                  dir=" ".join(str(-value) for value in config["light_position_m"]),
                  cutoff="90", exponent="1", ambient=" ".join(map(str, config["light_ambient"])),
                  diffuse=" ".join(map(str, config["light_diffuse"])),
                  specular=" ".join(map(str, config["light_specular"])), castshadow="true")
    xml = ET.tostring(root, encoding="unicode")
    specification = mujoco.MjSpec.from_string(xml)
    for mesh in specification.meshes:
        if mesh.name in normal_overrides:
            normals = normal_overrides[mesh.name]
            mesh.usernormal = normals.ravel().tolist()
            mesh.userfacenormal = list(range(len(normals)))
    model = specification.compile()
    if runtime_model_sha256(model) != runtime_model_sha256(original):
        raise ValueError("Appearance changes altered the pile's physical model")
    data = mujoco.MjData(model)
    with np.load(source / "settled-state.npz") as state:
        data.qpos[:] = state["qpos"]
        data.qvel[:] = state["qvel"]
    saved_qpos, saved_qvel = data.qpos.copy(), data.qvel.copy()
    mujoco.mj_forward(model, data)
    texture = model.texture("table_texture").id
    texture_height, texture_width = int(model.tex_height[texture]), int(model.tex_width[texture])
    rng = np.random.default_rng(config["texture_seed"])
    grain = rng.normal(0, config["texture_sigma_u8"], (texture_height, texture_width)).astype(np.float32)
    grain += cv2.GaussianBlur(grain, (0, 0), sigmaX=12, sigmaY=2) * 3
    texture_rgb = np.clip(np.array(config["table_rgb"]) * 255 + grain[..., None], 0, 255).astype(np.uint8)
    start = int(model.tex_adr[texture])
    model.tex_data[start:start + texture_rgb.size] = texture_rgb.ravel()
    positions = data.xpos[1:]
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (positions.min(axis=0) + positions.max(axis=0)) / 2
    camera.distance = max(.3, float(np.linalg.norm(np.ptp(positions, axis=0))) * 1.5)
    camera.azimuth = config["camera_azimuth"]
    camera.elevation = config["camera_elevation"]
    model.vis.quality.shadowsize = 4096
    light_positions = model.light_pos.copy()
    light_shadows = model.light_castshadow.copy()
    samples = []
    with mujoco.Renderer(model, height=height * scale, width=width * scale) as renderer:
        for _ in range(4):
            renderer.enable_segmentation_rendering()
            try:
                renderer.update_scene(data, camera=camera)
                segmentation = renderer.render()
            finally:
                renderer.disable_segmentation_rendering()
            ids = segmentation[..., 0]
            valid = (ids >= 0) & (ids < model.ngeom)
            mask = np.zeros(ids.shape, dtype=bool)
            mask[valid] = model.geom_bodyid[ids[valid]] > 0
            y, x = np.nonzero(mask)
            if not x.size:
                raise ValueError("Camera sees no pile geometry")
            fraction = max((x.max() - x.min() + 1) / (width * scale),
                           (y.max() - y.min() + 1) / (height * scale))
            view = renderer.scene.camera[0]
            right = np.cross(view.forward, view.up)
            view_height = 2 * camera.distance * np.tan(np.deg2rad(model.vis.global_.fovy) / 2)
            offset_x = (x.min() + x.max() + 1) / (2 * width * scale) - .5
            offset_y = (y.min() + y.max() + 1) / (2 * height * scale) - .5
            camera.lookat[:] += (right * offset_x * view_height * width / height
                                - np.asarray(view.up) * offset_y * view_height)
            camera.distance *= fraction / occupancy
        try:
            model.light_castshadow[:] = False
            renderer.update_scene(data, camera=camera)
            unshadowed = renderer.render().astype(np.float32)
            model.light_castshadow[:] = True
            spread = config["light_spread_m"]
            sampling = np.random.default_rng(config["texture_seed"] + 1)
            cells = math.ceil(math.sqrt(light_samples))
            for index in range(light_samples):
                dx, dy = ((np.array([index % cells, index // cells])
                           + sampling.uniform(size=2)) / cells * 2 - 1)
                model.light_pos[:] = light_positions
                model.light_pos[:, :2] += [dx * spread, dy * spread]
                mujoco.mj_camlight(model, data)
                renderer.update_scene(data, camera=camera)
                samples.append(renderer.render().astype(np.float32))
        finally:
            model.light_pos[:] = light_positions
            model.light_castshadow[:] = light_shadows
            mujoco.mj_camlight(model, data)
    strength = config["shadow_strength"]
    rgb = unshadowed * (1 - strength) + np.mean(samples, axis=0) * strength
    rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    rgb = cv2.GaussianBlur(rgb, (0, 0), config["blur_sigma_px"])
    rgb = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    with mujoco.Renderer(model, height=height, width=width) as label_renderer:
        label_renderer.enable_segmentation_rendering()
        try:
            label_renderer.update_scene(data, camera=camera)
            segmentation = label_renderer.render()
        finally:
            label_renderer.disable_segmentation_rendering()
        label_renderer.enable_depth_rendering()
        try:
            label_renderer.update_scene(data, camera=camera)
            depth = label_renderer.render()
        finally:
            label_renderer.disable_depth_rendering()
    ids = segmentation[..., 0]
    valid = (ids >= 0) & (ids < model.ngeom)
    mask = np.zeros(ids.shape, dtype=bool)
    mask[valid] = model.geom_bodyid[ids[valid]] > 0
    y, x = np.nonzero(mask)
    bbox = [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1]
    if bbox[0] == 0 or bbox[1] == 0 or bbox[2] == width or bbox[3] == height:
        raise ValueError("Pile framing clips object geometry at the image boundary")
    if not np.isfinite(depth).all():
        raise ValueError("Rendered depth contains nonfinite values")
    if not np.array_equal(saved_qpos, data.qpos) or not np.array_equal(saved_qvel, data.qvel):
        raise ValueError("Rendering changed the saved pile state")
    output.mkdir(parents=True, exist_ok=False)
    if not cv2.imwrite(str(output / "pile.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise OSError("Cannot save pile render")
    np.savez_compressed(output / "synchronized_labels.npz", segmentation=segmentation, depth_m=depth)
    report = {
        "profile_id": config["profile_id"], "profile_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_xml_sha256": hashlib.sha256(original_xml).hexdigest(),
        "source_state_sha256": hashlib.sha256((source / "settled-state.npz").read_bytes()).hexdigest(),
        "physical_model_sha256": runtime_model_sha256(model),
        "physics_and_state_unchanged": True, "light_samples": len(samples),
        "camera_distance_m": camera.distance, "qualification": (
            "Appearance-only rerender of saved convex-hull pile demo; not native-cavity "
            "physics, camera calibration or real-robot transfer evidence."
        ),
        "object_bbox_px": bbox,
        "frame_occupancy": max((bbox[2] - bbox[0]) / width, (bbox[3] - bbox[1]) / height),
    }
    (output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    args = parser.parse_args()
    print(json.dumps(render_pile(args.source, args.output, args.profile)), flush=True)


if __name__ == "__main__":
    main()
