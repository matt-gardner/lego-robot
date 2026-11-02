"""Detect a requested colored piece in an SO-101 wrist-camera image."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


HSV_RANGES = {
    "teal": ((78, 70, 45), (105, 255, 255)),
    "green": ((42, 55, 35), (78, 255, 255)),
    "orange": ((5, 100, 80), (25, 255, 255)),
    "navy": ((105, 60, 20), (135, 255, 150)),
    "brown": ((0, 55, 25), (20, 220, 150)),
    "gray": ((0, 0, 35), (179, 55, 210)),
}
MIN_TARGET_CONFIDENCE = .2


@dataclass(frozen=True)
class TargetObservation:
    color: str
    centroid: tuple[float, float]
    angle_deg: float
    area_pixels: float
    area_fraction: float
    bbox: tuple[int, int, int, int]
    confidence: float


def target_color(description: str) -> str:
    words = set(description.lower().replace("-", " ").split())
    matches = [color for color in HSV_RANGES if color in words]
    if len(matches) != 1:
        raise ValueError(
            f"Target description must contain exactly one supported color: "
            f"{', '.join(HSV_RANGES)}"
        )
    return matches[0]


def detect_target(image: np.ndarray, description: str) -> tuple[TargetObservation, np.ndarray]:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a BGR image with shape HxWx3")
    color = target_color(description)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    low, high = HSV_RANGES[color]
    mask = cv2.inRange(hsv, np.array(low, np.uint8), np.array(high, np.uint8))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    minimum_area = image.shape[0] * image.shape[1] * .002
    candidates = []
    image_area = image.shape[0] * image.shape[1]
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < minimum_area:
            continue
        solidity = area / max(float(cv2.contourArea(cv2.convexHull(contour))), 1)
        area_fraction = area / image_area
        confidence = float(np.clip(solidity * min(area_fraction / .03, 1), 0, 1))
        if confidence >= MIN_TARGET_CONFIDENCE:
            candidates.append((contour, area, area_fraction, confidence))
    if not candidates:
        raise ValueError(f"No confident {color} target found")
    contour, area, area_fraction, confidence = max(candidates, key=lambda item: item[1])
    moments = cv2.moments(contour)
    centroid = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
    (_, _), (width, height), angle = cv2.minAreaRect(contour)
    if width < height:
        angle += 90
    while angle >= 90:
        angle -= 180
    while angle < -90:
        angle += 180
    x, y, w, h = cv2.boundingRect(contour)
    observation = TargetObservation(
        color=color,
        centroid=(float(centroid[0]), float(centroid[1])),
        angle_deg=float(angle),
        area_pixels=area,
        area_fraction=float(area_fraction),
        bbox=(x, y, w, h),
        confidence=confidence,
    )
    annotated = image.copy()
    box = np.int32(cv2.boxPoints(cv2.minAreaRect(contour)))
    cv2.drawContours(annotated, [box], 0, (255, 255, 255), 2)
    cv2.circle(annotated, (round(centroid[0]), round(centroid[1])), 5, (255, 255, 255), -1)
    cv2.putText(
        annotated, f"{color} {confidence:.2f} {angle:.1f}deg", (x, max(18, y - 6)),
        cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return observation, annotated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--annotated", type=Path)
    args = parser.parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(args.image)
    observation, annotated = detect_target(image, args.target)
    print(json.dumps(asdict(observation), indent=2))
    if args.annotated is not None:
        args.annotated.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.annotated), annotated):
            raise RuntimeError(f"Failed to write {args.annotated}")


if __name__ == "__main__":
    main()
