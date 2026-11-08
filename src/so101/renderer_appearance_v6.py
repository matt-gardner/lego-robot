"""Original robot CAD jaws with the pinned v3 camera and image processing."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from so101.renderer_appearance import RendererAppearance
from so101.renderer_appearance_v3 import DEFAULT_PROFILE_V3, WristAppearanceV3


DEFAULT_PROFILE_V6 = Path(__file__).with_name("renderer_profiles") / "wrist_cad_v6.json"


class WristAppearanceV6(WristAppearanceV3):
    def __init__(self, path: Path = DEFAULT_PROFILE_V6):
        super().__init__(DEFAULT_PROFILE_V3)
        self.parent_v3_profile_sha256 = self.sha256
        self.parent_v3_implementation_sha256 = self.implementation_sha256
        source = Path(path).read_bytes()
        config = json.loads(source)
        if (config["schema_version"] != 1
                or config["parent_v3_profile_sha256"] != self.sha256
                or config["jaw_visual_geometry"] != "original-source-cad"):
            raise ValueError("CAD jaw profile requires its pinned v3 parent and original geometry")
        self.config.update(config)
        self.path = Path(path)
        self.sha256 = hashlib.sha256(source).hexdigest()
        self.implementation_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def identity(self) -> dict:
        return {
            **super().identity(),
            "parent_v3_profile_sha256": self.parent_v3_profile_sha256,
            "parent_v3_implementation_sha256": self.parent_v3_implementation_sha256,
        }

    def apply_xml(self, xml: str) -> str:
        original = ET.fromstring(xml)
        selectors = [
            f".//body[@name='{body}']/geom[@material='{material}']"
            for body, material in (
                ("gripper", "wrist_roll_follower_so101_v1_material"),
                ("moving_jaw_so101_v1", "moving_jaw_so101_v1_material"),
            )
        ]
        visuals = []
        for selector in selectors:
            geom = original.find(selector)
            if geom is None or geom.get("class") != "visual":
                raise ValueError("Missing original non-contact robot CAD jaw")
            visuals.append(dict(geom.attrib))
        root = ET.fromstring(super().apply_xml(xml))
        for selector, attributes in zip(selectors, visuals):
            geom = root.find(selector)
            if geom is None:
                raise ValueError("Missing styled jaw when restoring original CAD")
            geom.attrib.clear()
            geom.attrib.update(attributes)
        return ET.tostring(root, encoding="unicode")

    def render_rgb(self, renderer, model, data):
        return RendererAppearance.render_rgb(self, renderer, model, data)
