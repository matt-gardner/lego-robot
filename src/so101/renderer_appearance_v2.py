"""Soft white jaw appearance hypotheses; all contact geometry remains upstream.

The tapered cosmetic housings widen outward, not into the grasp gap. Their
dimensions are visual hypotheses, not measurements of the physical gripper.
Use WristAppearanceV2(), retaining the separately pinned v1 implementation.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np

from so101.renderer_appearance import RendererAppearance

DEFAULT_PROFILE_V2 = Path(__file__).with_name("renderer_profiles") / "wrist_soft_v2.json"


class WristAppearanceV2(RendererAppearance):
    def __init__(self, path: Path = DEFAULT_PROFILE_V2):
        super().__init__(path)
        self.parent_implementation_sha256 = self.implementation_sha256
        self.implementation_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def identity(self) -> dict:
        return {**super().identity(),
                "parent_implementation_sha256": self.parent_implementation_sha256}

    def apply_xml(self, xml: str) -> str:
        root = ET.fromstring(super().apply_xml(xml))
        asset = root.find("asset")
        width = self.config["jaw_root_half_width_m"]
        tip_width = self.config["jaw_tip_half_width_m"]
        thickness = self.config["jaw_half_thickness_m"]
        for body_name, mesh_name, material_name, moving in (
            ("gripper", "wrist_fixed_sleeve", "wrist_roll_follower_so101_v1_material", False),
            ("moving_jaw_so101_v1", "wrist_moving_sleeve", "moving_jaw_so101_v1_material", True),
        ):
            body = root.find(f".//body[@name='{body_name}']")
            if body is None or asset is None:
                raise ValueError(f"Missing jaw visual body: {body_name}")
            visual = body.find(f"geom[@material='{material_name}']")
            if visual is None or visual.get("class") != "visual":
                raise ValueError(f"Missing original non-contact jaw visual: {body_name}")
            vertices = []
            for tip in (False, True):
                half_width = tip_width if tip else width
                for side in (-1, 1):
                    for face in (-1, 1):
                        if moving:
                            point = [-.012 if tip else -.002, -.078 if tip else -.028,
                                     .019 + side * half_width]
                        else:
                            point = [-.009 if tip else -.015, side * half_width,
                                     -.103 if tip else -.045]
                        outward = (moving and face == 1) or (not moving and face == -1)
                        extent = (self.config["jaw_root_outward_extent_m"]
                                  if outward and not tip else thickness)
                        point[0] += face * extent
                        vertices.append(point)
            ET.SubElement(asset, "mesh", name=mesh_name,
                          vertex=" ".join(str(value) for point in vertices for value in point))
            visual.set("mesh", mesh_name)
            visual.set("pos", "0 0 0")
            visual.set("quat", "1 0 0 0")
        return ET.tostring(root, encoding="unicode")

    def apply_lighting(self, model) -> None:
        super().apply_lighting(model)
        for name in ("wrist_roll_follower_so101_v1_material", "moving_jaw_so101_v1_material"):
            material = model.material(name).id
            model.mat_specular[material] = 0
            model.mat_shininess[material] = 0

    def add_visual_cable(self, xml: str, points_m: np.ndarray, *, name: str = "appearance_cable") -> str:
        """Optional fixed world-space visual context, not a physical obstacle.

        Include these points in the scene/MJCF fingerprint. Do not attach the
        table cable to the wrist or use this non-contact prop as contact evidence.
        """
        points = np.asarray(points_m, dtype=float)
        if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 2
                or not np.isfinite(points).all() or np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) == 0)):
            raise ValueError("Visual cable requires distinct finite world-space metre points")
        root = ET.fromstring(xml)
        world = root.find("worldbody")
        if world is None or any(node.get("name") == name for node in root.iter()):
            raise ValueError("Missing worldbody or conflicting cable name")
        cable = ET.SubElement(world, "body", name=name)
        for index, (start, end) in enumerate(zip(points, points[1:])):
            ET.SubElement(cable, "geom", name=f"{name}_{index}",
                          type="capsule", size=str(self.config["visual_cable_radius_m"]),
                          fromto=" ".join(map(str, [*start, *end])),
                          rgba=" ".join(map(str, [*self.config["visual_cable_rgb"], 1])),
                          contype="0", conaffinity="0", mass="0", group="2")
        return ET.tostring(root, encoding="unicode")

    def render_rgb(self, renderer, model, data) -> np.ndarray:
        image = super().render_rgb(renderer, model, data)
        renderer.update_scene(data, camera=self.config["camera"]["name"])
        renderer.enable_segmentation_rendering()
        try:
            segmentation = renderer.render().copy()
        finally:
            renderer.disable_segmentation_rendering()
        mesh_ids = [model.mesh(name).id for name in ("wrist_fixed_sleeve", "wrist_moving_sleeve")]
        jaw_geoms = np.flatnonzero(np.isin(model.geom_dataid, mesh_ids)
                                  & (model.geom_type == mujoco.mjtGeom.mjGEOM_MESH))
        mask = np.isin(segmentation[:, :, 0], jaw_geoms).astype(np.float32)
        mask = cv2.GaussianBlur(mask, (0, 0), self.config["jaw_blur_sigma_px"])[:, :, None]
        softened = cv2.GaussianBlur(image, (0, 0), self.config["jaw_blur_sigma_px"])
        return np.rint(image * (1 - mask) + softened * mask).astype(np.uint8)

    def process_rgb(self, image: np.ndarray, *, seed: int, frame: int) -> np.ndarray:
        image = image.astype(np.float32)
        luminance = np.sum(image * np.array([.2126, .7152, .0722]), axis=2, keepdims=True)
        image = luminance + self.config["saturation_scale"] * (image - luminance)
        return super().process_rgb(np.clip(np.rint(image), 0, 255).astype(np.uint8),
                                   seed=seed, frame=frame)
