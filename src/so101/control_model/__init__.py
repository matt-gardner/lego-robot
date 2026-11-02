"""Bounding-box-conditioned robot control components."""

from .command import BaseCommand, CommandEntity
from .dataset import (
    future_box_chunk,
    oriented_box_from_mask,
    oriented_shape_features,
    transform_normalized_box,
)

__all__ = [
    "BaseCommand", "CommandEntity", "future_box_chunk", "oriented_box_from_mask",
    "oriented_shape_features", "transform_normalized_box"
]
