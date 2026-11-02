import json

import numpy as np
import pytest

from so101.control_model.command import BaseCommand, CommandEntity
from so101.control_model.dataset import (
    future_box_chunk,
    oriented_box_from_mask,
    oriented_shape_features,
    transform_normalized_box,
)


def command(**overrides):
    values = {
        "command_id": "command-1",
        "scene_id": "scene-1",
        "source_camera": "wrist",
        "source_timestamp": 12.5,
        "phrase": "grasp",
        "entities": (
            CommandEntity("piece-1", "object", (0.4, 0.5, 0.4, 0.2, .3), .18),
        ),
    }
    values.update(overrides)
    return BaseCommand(**values)


def test_command_round_trip_and_center_representation():
    original = command()
    expected = (0.4, 0.5, 0.3, 0.1 * np.cos(.6), 0.1 * np.sin(.6))
    assert original.entities[0].box_features == pytest.approx(expected)
    restored = BaseCommand.from_dict(json.loads(json.dumps(original.to_dict())))
    assert restored == original


@pytest.mark.parametrize(
    "overrides",
    [
        {"command_id": ""},
        {"source_timestamp": -1},
        {"phrase": "drop"},
        {"entities": ()},
        {"entities": (
            CommandEntity("piece-1", "destination", (.4, .5, .4, .2, 0)),
        )},
    ],
)
def test_command_rejects_invalid_values(overrides):
    with pytest.raises(ValueError):
        command(**overrides)


def test_command_rejects_stale_scene_or_wrong_camera():
    current = command()
    current.require_scene("scene-1", "wrist")
    with pytest.raises(ValueError, match="current scene"):
        current.require_scene("scene-2")
    with pytest.raises(ValueError, match="camera"):
        current.require_scene("scene-1", "overview")


def test_push_requires_object_and_destination_entities():
    push = command(
        phrase="push",
        entities=(
            CommandEntity("piece-1", "object", (.2, .3, .2, .1, 0)),
            CommandEntity("goal-1", "destination", (.7, .6, .2, .1, 0)),
        ),
    )
    assert [entity.role for entity in push.entities] == ["object", "destination"]
    with pytest.raises(ValueError, match="push requires"):
        command(phrase="push")


def test_box_transform_tracks_crop_and_horizontal_flip():
    transformed = transform_normalized_box(
        (0.25, 0.25, 0.75, 0.75),
        (640, 480),
        crop_xyxy=(160, 0, 640, 480),
        flip_horizontal=True,
    )
    assert transformed == pytest.approx((1 / 3, 0.25, 1, 0.75, True))


def test_box_transform_marks_fully_cropped_target_invalid():
    assert transform_normalized_box(
        (0.0, 0.0, 0.1, 0.1),
        (640, 480),
        crop_xyxy=(320, 0, 640, 480),
    ) == (0, 0, 0, 0, False)


def test_future_box_chunk_uses_post_action_frames_and_masks_padding():
    boxes = np.array([
        [0.1, 0.2, 0.2, 0.1, 0, 1],
        [0.2, 0.2, 0.2, 0.1, 0, 1],
        [0.3, 0.2, 0.2, 0.1, 0, 0],
    ], dtype=np.float32)
    chunk, is_pad = future_box_chunk(boxes, start_index=1, horizon=3)
    np.testing.assert_allclose(chunk, [boxes[2], boxes[2], boxes[2]])
    np.testing.assert_array_equal(is_pad, [False, True, True])


def test_oriented_box_from_mask_encodes_center_extents_and_doubled_angle():
    mask = np.zeros((100, 200), dtype=np.uint8)
    points = np.array([[60, 30], [140, 50], [136, 66], [56, 46]], dtype=np.int32)
    import cv2

    cv2.fillConvexPoly(mask, points, 1)
    center_x, center_y, mean_extent, anisotropy_x, anisotropy_y, visible = (
        oriented_box_from_mask(mask)
    )
    assert visible
    assert center_x == pytest.approx(.49, abs=.02)
    assert center_y == pytest.approx(.48, abs=.03)
    assert mean_extent > np.hypot(anisotropy_x, anisotropy_y) > 0


def test_oriented_box_from_empty_mask_marks_entity_invisible():
    assert oriented_box_from_mask(np.zeros((10, 20), dtype=np.uint8)) == (
        0, 0, 0, 0, 0, False
    )


def test_oriented_shape_features_are_invariant_to_axis_swap():
    original = oriented_shape_features(.4, .2, .3)
    swapped = oriented_shape_features(.2, .4, .3 + np.pi / 2)
    assert swapped == pytest.approx(original)
