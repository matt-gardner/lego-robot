import cv2
import numpy as np
import pytest

from so101.target_perception import detect_target, target_color


def test_target_color_requires_one_supported_color():
    assert target_color("Pick up the teal beam.") == "teal"
    with pytest.raises(ValueError):
        target_color("Pick up the beam")
    with pytest.raises(ValueError):
        target_color("Pick up the teal or orange beam")


def test_detect_target_finds_rotated_colored_piece():
    image = np.full((240, 320, 3), 235, dtype=np.uint8)
    rectangle = ((190, 120), (100, 24), 28)
    box = np.int32(cv2.boxPoints(rectangle))
    cv2.fillConvexPoly(image, box, (150, 120, 15))
    observation, annotated = detect_target(image, "teal piece")
    assert observation.color == "teal"
    assert observation.centroid == pytest.approx((190, 120), abs=2)
    assert observation.angle_deg == pytest.approx(28, abs=2)
    assert observation.confidence > .8
    assert annotated.shape == image.shape


def test_detect_target_rejects_missing_color():
    image = np.full((240, 320, 3), 235, dtype=np.uint8)
    with pytest.raises(ValueError, match="No confident teal"):
        detect_target(image, "teal beam")


def test_detect_target_rejects_small_low_confidence_color_patch():
    image = np.full((240, 320, 3), 235, dtype=np.uint8)
    cv2.rectangle(image, (100, 220), (210, 224), (0, 100, 220), -1)
    with pytest.raises(ValueError, match="No confident orange"):
        detect_target(image, "orange beam")
