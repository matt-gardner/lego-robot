"""Descriptive settled-pose envelopes, not collision or difficulty qualification."""
from __future__ import annotations

from itertools import combinations

import numpy as np


def configuration_metrics(objects: list[dict]) -> dict:
    envelopes = {}
    for obj in objects:
        position = np.asarray(obj["position_m"], dtype=float)
        rotation = np.asarray(obj["rotation"], dtype=float)
        dimensions = np.asarray(obj["dimensions_m"], dtype=float)
        if (position.shape != (3,) or rotation.shape != (3, 3)
                or dimensions.shape != (3,) or not np.isfinite(position).all()
                or not np.isfinite(rotation).all() or not np.isfinite(dimensions).all()
                or np.any(dimensions <= 0)):
            raise ValueError("Invalid settled object pose/dimensions")
        name = obj["instance_id"]
        if name in envelopes:
            raise ValueError("Duplicate settled instance identity")
        half = np.abs(rotation) @ (dimensions / 2)
        envelopes[name] = {"position": position, "lower": position - half, "upper": position + half}
    pairs = []
    for first, second in combinations(sorted(envelopes), 2):
        a, b = envelopes[first], envelopes[second]
        overlap = np.minimum(a["upper"], b["upper"]) - np.maximum(a["lower"], b["lower"])
        pairs.append({
            "instances": [first, second],
            "center_distance_m": float(np.linalg.norm(a["position"] - b["position"])),
            "horizontal_envelope_overlap_m": overlap[:2].tolist(),
            "vertical_center_separation_m": float(abs(a["position"][2] - b["position"][2])),
        })
    return {
        "measurement": "world-axis envelopes of nominal part dimensions at actual settled body poses",
        "not_qualified": "envelopes are not exact source geometry, contact clearance or occlusion",
        "objects": [
            {"instance_id": name, "position_m": value["position"].tolist(),
             "envelope_lower_m": value["lower"].tolist(), "envelope_upper_m": value["upper"].tolist()}
            for name, value in sorted(envelopes.items())
        ],
        "pairs": pairs,
    }
