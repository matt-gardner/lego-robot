"""Rounded outward jaw surfaces; retain original physical contacts and pinned parents."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from so101.renderer_appearance_v2 import DEFAULT_PROFILE_V2, WristAppearanceV2

DEFAULT_PROFILE_V3 = Path(__file__).with_name("renderer_profiles") / "wrist_soft_v3.json"


class WristAppearanceV3(WristAppearanceV2):
    def __init__(self, path: Path = DEFAULT_PROFILE_V3):
        super().__init__(DEFAULT_PROFILE_V2)
        self.parent_profile_sha256 = self.sha256
        self.parent_v2_implementation_sha256 = self.implementation_sha256
        source = Path(path).read_bytes()
        config = json.loads(source)
        if config["schema_version"] != 1 or config["parent_profile_sha256"] != self.sha256:
            raise ValueError("Rounded jaw profile requires its exact pinned v2 parent")
        self.config.update(config)
        self.path = Path(path)
        self.sha256 = hashlib.sha256(source).hexdigest()
        self.implementation_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def identity(self) -> dict:
        return {**super().identity(), "parent_profile_sha256": self.parent_profile_sha256,
                "parent_v2_implementation_sha256": self.parent_v2_implementation_sha256}

    def apply_xml(self, xml: str) -> str:
        root = ET.fromstring(super().apply_xml(xml))
        for mesh_name, moving in (("wrist_fixed_sleeve", False), ("wrist_moving_sleeve", True)):
            vertices = []
            for distance, width_fraction, outward_fraction in self.config["jaw_sections"]:
                center_x = ((-.002 * (1 - distance) - .012 * distance) if moving
                            else (-.015 * (1 - distance) - .009 * distance))
                thickness = self.config["jaw_half_thickness_m"]
                outward = self.config["jaw_root_outward_extent_m"] * outward_fraction
                low = center_x - (thickness if moving else outward)
                high = center_x + (outward if moving else thickness)
                width = self.config["jaw_root_half_width_m"] * width_fraction
                for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
                    x = (low + high) / 2 + (high - low) / 2 * np.cos(angle)
                    transverse = width * np.sin(angle)
                    if moving:
                        point = [x, -.028 * (1 - distance) - .078 * distance, .019 + transverse]
                    else:
                        point = [x, transverse, -.045 * (1 - distance) - .103 * distance]
                    vertices.append(point)
            mesh = root.find(f"asset/mesh[@name='{mesh_name}']")
            if mesh is None:
                raise ValueError(f"Missing inherited jaw visual mesh: {mesh_name}")
            mesh.set("vertex", " ".join(str(value) for point in vertices for value in point))
        return ET.tostring(root, encoding="unicode")
