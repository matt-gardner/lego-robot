"""Spatially correlated luminance noise; inherit the exact v4 geometric camera."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from so101.renderer_appearance_v4 import DEFAULT_PROFILE_V4, WristAppearanceV4

DEFAULT_PROFILE_V5 = Path(__file__).with_name("renderer_profiles") / "wrist_soft_v5.json"


class WristAppearanceV5(WristAppearanceV4):
    def __init__(self, path: Path = DEFAULT_PROFILE_V5):
        super().__init__(DEFAULT_PROFILE_V4)
        self.parent_v4_profile_sha256 = self.sha256
        self.parent_v4_implementation_sha256 = self.implementation_sha256
        source = Path(path).read_bytes()
        config = json.loads(source)
        if (config["schema_version"] != 1
                or config["parent_v4_profile_sha256"] != self.parent_v4_profile_sha256):
            raise ValueError("Wrist v5 requires its exact pinned v4 parent profile")
        if (config["noise_sigma_u8"] != 0
                or not np.isfinite(config["noise_correlation_sigma_px"])
                or config["noise_correlation_sigma_px"] <= 0
                or not np.isfinite(config["correlated_noise_sigma_u8"])
                or config["correlated_noise_sigma_u8"] < 0):
            raise ValueError("Wrist v5 requires finite correlated noise and no independent noise")
        self.config.update(config)
        self.path = Path(path)
        self.sha256 = hashlib.sha256(source).hexdigest()
        self.implementation_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def identity(self) -> dict:
        return {
            **super().identity(),
            "parent_v4_profile_sha256": self.parent_v4_profile_sha256,
            "parent_v4_implementation_sha256": self.parent_v4_implementation_sha256,
        }

    def process_rgb(self, image: np.ndarray, *, seed: int, frame: int) -> np.ndarray:
        image = super().process_rgb(image, seed=seed, frame=frame).astype(np.float32)
        sigma = self.config["noise_correlation_sigma_px"]
        kernel = cv2.getGaussianKernel(5, sigma)
        gain = float(np.square(kernel).sum())
        rng = np.random.default_rng(np.random.SeedSequence([seed, frame, 1108, 5]))
        noise = rng.normal(0, self.config["correlated_noise_sigma_u8"] / gain, image.shape[:2])
        noise = cv2.GaussianBlur(noise, (5, 5), sigma)
        return np.clip(np.rint(image + noise[:, :, None]), 0, 255).astype(np.uint8)
