from dataclasses import replace

import pytest

from so101.trajectory_progress import (
    ProgressState, progress_potential, progress_reward, reward_percentage, short_horizon,
)


def state(x, *, grip=False, z=0.):
    return ProgressState((x, 0., 0.), (1., 0., 0., 0., 1., 0., 0., 0., 1.),
                         0. if grip else .5, (0., 0., z), grip, 0.)


@pytest.fixture
def reference():
    return (state(.04), state(.03), state(.02), state(.01), state(0., grip=True))


def test_passing_next_milestone_is_better_not_worse(reference):
    start = reference[0]
    next_band = progress_reward([start, reference[1]], reference)
    overshoot = progress_reward([start, reference[3]], reference)
    assert overshoot["reward"] > next_band["reward"] > 0


def test_secure_lift_beyond_reference_is_not_penalized(reference):
    closure = reference[-1]
    lifted = replace(closure, relative_position=(.1, 0., 0.), target_position=(0., 0., .06))
    assert progress_potential(lifted, reference) > progress_potential(closure, reference)


def test_empty_gripper_cannot_get_contact_milestone(reference):
    empty = replace(reference[-1], bilateral_grip=False)
    assert progress_potential(empty, reference) < progress_potential(reference[-1], reference)


def test_progress_does_not_pay_repeatedly_for_oscillation(reference):
    once = progress_reward([reference[0], reference[2]], reference)
    oscillating = progress_reward(
        [reference[0], reference[2], reference[0], reference[2]], reference)
    assert once["reward"] == oscillating["reward"]


def test_drop_and_invalid_command_have_explicit_penalties(reference):
    held = reference[-1]
    dropped = replace(held, bilateral_grip=False)
    report = progress_reward([reference[0], held, dropped], reference, invalid=True)
    assert report["lost_grip"]
    assert report["penalty"] == .5


def test_stationary_short_rollout_gets_zero(reference):
    assert progress_reward([reference[0], reference[0]], reference)["reward"] == 0
    assert short_horizon(24) == 3
    assert short_horizon(100) == 8
    with pytest.raises(ValueError):
        short_horizon(1)


def test_regressing_rollouts_still_have_a_ranked_learning_signal(reference):
    start = reference[0]
    slight_retreat = replace(start, relative_position=(.06, 0., 0.))
    large_retreat = replace(start, relative_position=(.08, 0., 0.))
    slight = progress_reward([start, slight_retreat, large_retreat], reference)
    large = progress_reward([start, large_retreat, large_retreat], reference)
    assert large["reward"] < slight["reward"] < 0


def test_reward_percentage_uses_remaining_ceiling_without_hiding_retreat(reference):
    start = reference[0]
    lifted = replace(reference[-1], target_position=(0., 0., .05))
    complete = progress_reward([start, lifted], reference)
    assert complete["reward_percent_of_max"] == pytest.approx(100.)
    assert reward_percentage(.5, 1.)["reward_percent_of_max"] == 50.
    assert reward_percentage(-.25, 1.)["reward_percent_of_max"] == -25.
    with pytest.raises(ValueError, match="below the ceiling"):
        reward_percentage(0., 2.)
    with pytest.raises(ValueError, match="exceeds"):
        reward_percentage(3., 0.)
