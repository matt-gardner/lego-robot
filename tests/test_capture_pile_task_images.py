import numpy as np
import pytest

from so101.capture_pile_task_images import target_annotation


def test_visible_target_box_uses_pixel_exclusive_bounds_and_preserves_zero_visibility():
    mask = np.zeros((3, 5), dtype=bool)
    mask[1:3, 2:5] = True
    annotation = target_annotation(mask)
    assert annotation["target_bbox_xyxy"] == [2, 1, 5, 3]
    assert annotation["target_visible_pixels"] == 6
    assert annotation["target_bbox_status"] == "visible"
    annotation = target_annotation(np.zeros_like(mask))
    assert annotation["target_bbox_xyxy"] is None
    assert annotation["target_visible_pixels"] == 0
    assert annotation["target_bbox_status"] == "not_visible"


def test_target_box_rejects_nonsegmentation_arrays():
    with pytest.raises(ValueError, match="boolean segmentation"):
        target_annotation(np.ones((3, 5), dtype=float))
