from __future__ import annotations

import cv2
import numpy as np


ORIENTED_BOX_DIM = 5


def oriented_shape_features(
    extent_x: float, extent_y: float, angle: float
) -> tuple[float, float, float]:
    if not all(np.isfinite((extent_x, extent_y, angle))) or min(extent_x, extent_y) < 0:
        raise ValueError("Extents must be finite and non-negative and angle must be finite")
    mean_extent = (extent_x + extent_y) / 2
    half_difference = (extent_x - extent_y) / 2
    return (
        mean_extent,
        float(half_difference * np.cos(2 * angle)),
        float(half_difference * np.sin(2 * angle)),
    )


def oriented_box_from_mask(mask: np.ndarray) -> tuple[float, float, float, float, float, bool]:
    """Return center, mean extent, anisotropy vector, and visibility."""
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.ndim != 2:
        raise ValueError("mask must be a two-dimensional array")
    height, width = mask.shape
    if width <= 0 or height <= 0:
        raise ValueError("mask must have positive dimensions")
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return (0.0, 0.0, 0.0, 0.0, 0.0, False)
    points = np.concatenate(contours, axis=0).astype(np.float32)
    (center_x, center_y), (extent_x, extent_y), angle_degrees = cv2.minAreaRect(points)
    angle = np.deg2rad(angle_degrees)
    diagonal = float(np.hypot(width, height))
    extent_x = min((float(extent_x) + 1) / diagonal, 1.0)
    extent_y = min((float(extent_y) + 1) / diagonal, 1.0)
    mean_extent, anisotropy_x, anisotropy_y = oriented_shape_features(
        extent_x, extent_y, angle
    )
    return (
        float(center_x) / width,
        float(center_y) / height,
        mean_extent,
        anisotropy_x,
        anisotropy_y,
        True,
    )


def transform_normalized_box(
    box_xyxy: tuple[float, float, float, float],
    image_size: tuple[int, int],
    *,
    crop_xyxy: tuple[int, int, int, int] | None = None,
    flip_horizontal: bool = False,
) -> tuple[float, float, float, float, bool]:
    """Transform a normalized box through crop and optional horizontal flip."""
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image_size must be positive")
    values = np.asarray(box_xyxy, dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("box_xyxy must contain four finite values")
    x1, y1, x2, y2 = values
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError("box_xyxy must be a non-empty normalized box")

    pixels = values * np.array([width, height, width, height], dtype=np.float64)
    if crop_xyxy is None:
        crop = np.array([0, 0, width, height], dtype=np.float64)
    else:
        crop = np.asarray(crop_xyxy, dtype=np.float64)
        if (
            crop.shape != (4,)
            or not np.isfinite(crop).all()
            or not (0 <= crop[0] < crop[2] <= width)
            or not (0 <= crop[1] < crop[3] <= height)
        ):
            raise ValueError("crop_xyxy must be a non-empty crop inside the image")

    clipped = np.array([
        max(pixels[0], crop[0]),
        max(pixels[1], crop[1]),
        min(pixels[2], crop[2]),
        min(pixels[3], crop[3]),
    ])
    if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
        return (0.0, 0.0, 0.0, 0.0, False)

    crop_width = crop[2] - crop[0]
    crop_height = crop[3] - crop[1]
    normalized = np.array([
        (clipped[0] - crop[0]) / crop_width,
        (clipped[1] - crop[1]) / crop_height,
        (clipped[2] - crop[0]) / crop_width,
        (clipped[3] - crop[1]) / crop_height,
    ])
    if flip_horizontal:
        normalized[[0, 2]] = 1 - normalized[[2, 0]]
    return (*normalized.tolist(), True)


def future_box_chunk(
    boxes_oriented_visibility: np.ndarray,
    start_index: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return boxes after each future action and a mask for padded steps."""
    boxes = np.asarray(boxes_oriented_visibility, dtype=np.float32)
    if (
        boxes.ndim != 2
        or boxes.shape[1] != ORIENTED_BOX_DIM + 1
        or not np.isfinite(boxes).all()
    ):
        raise ValueError(
            "boxes_oriented_visibility must have shape (frames, 7) with finite values"
        )
    if len(boxes) == 0:
        raise ValueError("boxes_oriented_visibility must not be empty")
    if not 0 <= start_index < len(boxes):
        raise ValueError("start_index is outside the trajectory")
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    indices = np.arange(start_index + 1, start_index + horizon + 1)
    is_pad = indices >= len(boxes)
    indices = np.minimum(indices, len(boxes) - 1)
    return boxes[indices].copy(), is_pad
