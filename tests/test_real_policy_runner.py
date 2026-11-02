import numpy as np
import pytest

from so101.real_policy_runner import (
    COLOR_SPECIALIST_CHECKPOINTS,
    GENERAL_TARGET_CHECKPOINT,
    apply_contact_phase_guard,
    apply_target_flex_servo,
    apply_target_pan_servo,
    bounded_action,
    orient_wrist_image,
    pickup_task,
    resolve_policy_checkpoint,
    update_contact_detection,
)
from so101.robot_control import JOINTS


BOUNDS = np.array([[-100, 100]] * len(JOINTS), dtype=float)


def test_pickup_task_accepts_specific_object_descriptions():
    assert pickup_task("teal beam") == "Pick up the teal beam."
    with pytest.raises(ValueError):
        pickup_task("  ")


def test_policy_routing_uses_color_specialists_with_general_fallback():
    assert resolve_policy_checkpoint("orange hollow beam") == COLOR_SPECIALIST_CHECKPOINTS["orange"]
    assert resolve_policy_checkpoint("teal beam") == COLOR_SPECIALIST_CHECKPOINTS["teal"]
    assert resolve_policy_checkpoint("orange hollow beam", fallback=True) == GENERAL_TARGET_CHECKPOINT
    assert resolve_policy_checkpoint("uncolored object") == GENERAL_TARGET_CHECKPOINT


def test_wrist_image_can_be_horizontally_flipped_for_policy_coordinates():
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    np.testing.assert_array_equal(orient_wrist_image(image, False), image)
    np.testing.assert_array_equal(orient_wrist_image(image, True), image[:, ::-1])


def test_target_pan_servo_uses_measured_image_direction_and_limits_step():
    target = np.zeros(len(JOINTS))
    current = np.zeros(len(JOINTS))
    adjusted = apply_target_pan_servo(
        target, current, BOUNDS, centroid_x=250, desired_x=320,
        pixels_per_degree=-5, max_step=2,
    )
    assert adjusted[JOINTS.index("shoulder_pan")] == -2


def test_target_flex_servo_uses_measured_image_direction_and_limits_step():
    target = np.zeros(len(JOINTS))
    current = np.zeros(len(JOINTS))
    adjusted = apply_target_flex_servo(
        target, current, BOUNDS, centroid_y=300, desired_y=200,
        pixels_per_degree=-5, max_step=3,
    )
    assert adjusted[JOINTS.index("wrist_flex")] == 3


def test_bounded_action_maps_sim_to_real_and_limits_per_tick_motion():
    current = np.zeros(len(JOINTS))
    offset = np.array([0, -90, 90, 0, 0, 0], dtype=float)
    action = np.array([20, -70, 110, 5, -5, 10], dtype=float)
    target, projection = bounded_action(action, current, offset, BOUNDS, 12, 10)
    np.testing.assert_allclose(target, [12, 12, 12, 5, -5, 10])
    assert projection == 0


def test_bounded_action_projects_small_boundary_noise():
    action = np.array([102, 0, 0, 0, 0, 0], dtype=float)
    target, projection = bounded_action(
        action, np.full(len(JOINTS), 95.0), np.zeros(6), BOUNDS, 12, 10
    )
    assert target[0] == 100
    assert projection == 2


def test_bounded_action_projects_boundary_noise_that_previously_aborted_rollout():
    action = np.array([105.35, 0, 0, 0, 0, 0], dtype=float)
    target, projection = bounded_action(
        action, np.full(len(JOINTS), 95.0), np.zeros(6), BOUNDS, 12, 10
    )
    assert target[0] == 100
    assert projection == pytest.approx(5.35)


@pytest.mark.parametrize(
    "action",
    [np.zeros(5), np.array([np.nan, 0, 0, 0, 0, 0]), np.array([111, 0, 0, 0, 0, 0])],
)
def test_bounded_action_rejects_invalid_or_large_out_of_bounds_actions(action):
    with pytest.raises(ValueError):
        bounded_action(action, np.zeros(len(JOINTS)), np.zeros(6), BOUNDS, 12, 10)


def test_contact_phase_guard_holds_shoulder_until_gripper_contact():
    current = np.zeros(len(JOINTS))
    current[JOINTS.index("shoulder_lift")] = 120
    current[JOINTS.index("gripper")] = 20
    bounds = np.array([[-200, 200]] * len(JOINTS), dtype=float)
    guarded = apply_contact_phase_guard(
        np.zeros(len(JOINTS)), current, bounds, 12, 124, 8, 0, 0
    )
    assert guarded[JOINTS.index("shoulder_lift")] == 124
    assert guarded[JOINTS.index("gripper")] == 8

    current[JOINTS.index("gripper")] = 8
    unguarded = apply_contact_phase_guard(
        np.zeros(len(JOINTS)), current, bounds, 12, 124, 8, 0
    )
    assert unguarded[JOINTS.index("shoulder_lift")] == 0

    target = np.zeros(len(JOINTS))
    target[JOINTS.index("gripper")] = 12
    held = apply_contact_phase_guard(target, current, bounds, 12, 124, 8, 0, 0)
    assert held[JOINTS.index("gripper")] == 0

    lifted = apply_contact_phase_guard(
        target, current, bounds, 12, 124, 8, 0, 0, 105
    )
    assert lifted[JOINTS.index("shoulder_lift")] == 108


def test_contact_phase_guard_requires_both_options():
    with pytest.raises(ValueError):
        apply_contact_phase_guard(
            np.zeros(len(JOINTS)), np.zeros(len(JOINTS)), BOUNDS, 12, 124, None
        )


def test_contact_phase_guard_does_not_lift_for_empty_closure():
    current = np.zeros(len(JOINTS))
    current[JOINTS.index("shoulder_lift")] = 124
    current[JOINTS.index("gripper")] = 0.8
    guarded = apply_contact_phase_guard(
        np.zeros(len(JOINTS)),
        current,
        np.array([[-200, 200]] * len(JOINTS), dtype=float),
        12,
        124,
        8,
        0,
        0,
        105,
        False,
    )
    assert guarded[JOINTS.index("shoulder_lift")] == 124


def test_contact_detection_requires_consecutive_samples_and_clears_empty_closure():
    detected = False
    samples = 0
    for position in (6.2, 0.8, 6.1, 6.0):
        detected, samples = update_contact_detection(
            position, detected, samples, 8, 5, 3
        )
        assert not detected

    detected, samples = update_contact_detection(6.3, detected, samples, 8, 5, 3)
    assert detected
    assert samples == 3

    detected, samples = update_contact_detection(0.8, detected, samples, 8, 5, 3)
    assert not detected
    assert samples == 0


def test_contact_detection_remains_latched_without_empty_closure_minimum():
    detected, samples = update_contact_detection(6, False, 2, 8, None, 3)
    assert detected

    detected, samples = update_contact_detection(0.8, detected, samples, 8, None, 3)
    assert detected
    assert samples == 3


def test_contact_detection_rejects_nonpositive_persistence():
    with pytest.raises(ValueError):
        update_contact_detection(6, False, 0, 8, 5, 0)
