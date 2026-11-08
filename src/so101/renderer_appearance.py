"""Opt-in appearance only; no camera calibration or transfer claim.

For BeamSimulation, pass ``appearance=RendererAppearance()``. For an assembled
SO101/catalog scene: apply_xml before MjModel compilation, apply_model before
Renderer creation, and apply_lighting after reset. Use render_rgb/process_rgb
for RGB only; segmentation and metric depth use the unchanged geometric camera.
Persist identity() with every initial state, and reject replay hash mismatches.
Published profile and implementation versions must not be edited in place.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np

DEFAULT_PROFILE = Path(__file__).with_name("renderer_profiles") / "wrist_soft_v1.json"


class RendererAppearance:
    def __init__(self, path: Path = DEFAULT_PROFILE):
        self.path = Path(path)
        source = self.path.read_bytes()
        self.config = json.loads(source)
        if self.config["schema_version"] != 1:
            raise ValueError("Unsupported renderer profile schema")
        self.sha256 = hashlib.sha256(source).hexdigest()
        self.implementation_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def identity(self) -> dict:
        return {"profile_id": self.config["profile_id"], "sha256": self.sha256,
                "implementation_sha256": self.implementation_sha256}

    def apply_xml(self, xml: str) -> str:
        """Apply to assembled MJCF before compilation; retain all contact geometry."""
        root = ET.fromstring(xml)
        camera = self.config["camera"]
        element = root.find(f".//camera[@name='{camera['name']}']")
        if element is None:
            raise ValueError(f"Missing appearance camera: {camera['name']}")
        for key in ("pos", "quat"):
            element.set(key, " ".join(map(str, camera[key])))
        element.set("fovy", str(camera["fovy"]))
        asset = root.find("asset")
        if asset is None:
            asset = ET.SubElement(root, "asset")
        ET.SubElement(asset, "texture", name="wrist_table_texture", type="2d",
                      builtin="flat", rgb1=".66 .67 .65", width="512", height="512")
        ET.SubElement(asset, "material", name="wrist_table_material",
                      texture="wrist_table_texture", texrepeat="8 8",
                      reflectance="0", specular=".04", shininess=".05")
        table = root.find(".//geom[@name='table']")
        if table is None:
            raise ValueError("Missing appearance table")
        table.set("material", "wrist_table_material")
        return ET.tostring(root, encoding="unicode")

    def apply_model(self, model) -> None:
        """Initialize deterministic texture before creating the renderer context."""
        texture = model.texture("wrist_table_texture").id
        start = int(model.tex_adr[texture])
        count = int(model.tex_height[texture] * model.tex_width[texture] * 3)
        rng = np.random.default_rng(self.config["texture_seed"])
        noise = rng.normal(0, self.config["texture_sigma"], (512, 512, 1))
        rgb = np.array(self.config["table_rgb"]) * 255 + noise
        model.tex_data[start:start + count] = np.clip(rgb, 0, 255).astype(np.uint8).ravel()
        self.apply_lighting(model)

    def apply_lighting(self, model) -> None:
        for field in ("ambient", "diffuse", "specular"):
            getattr(model, f"light_{field}")[:] = self.config[f"light_{field}"]
        model.light_castshadow[:] = True
        model.vis.quality.shadowsize = 2048
        model.vis.map.shadowscale = self.config["shadow_scale"]
        model.geom_rgba[model.geom("table").id] = [1, 1, 1, 1]
        for name in ("wrist_roll_follower_so101_v1_material", "moving_jaw_so101_v1_material"):
            material = model.material(name).id
            model.mat_rgba[material] = [*self.config["jaw_rgb"], 1]
            model.mat_specular[material] = .04
            model.mat_shininess[material] = .08
            model.mat_emission[material] = self.config["jaw_emission"]

    def process_rgb(self, image: np.ndarray, *, seed: int, frame: int) -> np.ndarray:
        """Photometric processing only: no warps/crops of segmentation or metric depth."""
        image = cv2.GaussianBlur(image, (0, 0), self.config["blur_sigma_px"]).astype(np.float32)
        rng = np.random.default_rng(np.random.SeedSequence([seed, frame, 1108]))
        image += rng.normal(0, self.config["noise_sigma_u8"], image.shape)
        return np.clip(np.rint(image), 0, 255).astype(np.uint8)

    def render_rgb(self, renderer, model, data) -> np.ndarray:
        """Average area-light samples; labels/depth retain the same geometric camera."""
        positions = model.light_pos.copy()
        shadows = model.light_castshadow.copy()
        samples = []
        spread = self.config["light_spread_m"]
        try:
            model.light_castshadow[:] = False
            renderer.update_scene(data, camera=self.config["camera"]["name"])
            unshadowed = renderer.render().astype(np.float32)
            model.light_castshadow[:] = True
            for x, y in ((0, 0), (-spread, -spread), (-spread, spread),
                         (spread, -spread), (spread, spread)):
                model.light_pos[:] = positions
                model.light_pos[:, :2] += [x, y]
                mujoco.mj_camlight(model, data)
                renderer.update_scene(data, camera=self.config["camera"]["name"])
                samples.append(renderer.render().astype(np.float32))
        finally:
            model.light_pos[:] = positions
            model.light_castshadow[:] = shadows
            mujoco.mj_camlight(model, data)
        strength = self.config["shadow_strength"]
        return np.rint(unshadowed * (1 - strength) + np.mean(samples, axis=0) * strength).astype(np.uint8)
