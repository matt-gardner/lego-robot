"""Wider wrist-camera composition and sharper cosmetic tips with pinned parents."""
from __future__ import annotations

import hashlib
from pathlib import Path

from so101.renderer_appearance_v3 import DEFAULT_PROFILE_V3, WristAppearanceV3

DEFAULT_PROFILE_V4 = Path(__file__).with_name("renderer_profiles") / "wrist_soft_v4.json"


class WristAppearanceV4(WristAppearanceV3):
    def __init__(self, path: Path = DEFAULT_PROFILE_V4):
        super().__init__(path)
        self.parent_v3_profile_sha256 = hashlib.sha256(DEFAULT_PROFILE_V3.read_bytes()).hexdigest()
        if self.config["parent_v3_profile_sha256"] != self.parent_v3_profile_sha256:
            raise ValueError("Wrist v4 requires its exact pinned v3 parent profile")
        self.parent_v3_implementation_sha256 = self.implementation_sha256
        self.implementation_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def identity(self) -> dict:
        return {
            **super().identity(),
            "parent_v3_profile_sha256": self.parent_v3_profile_sha256,
            "parent_v3_implementation_sha256": self.parent_v3_implementation_sha256,
        }
