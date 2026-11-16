"""Short-horizon GRASP progress against measured expert states, not one future pose."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ProgressState:
    relative_position: tuple[float, ...]
    orientation: tuple[float, ...]
    opening: float
    target_position: tuple[float, ...]
    bilateral_grip: bool
    target_speed: float

    def __post_init__(self):
        if (len(self.relative_position) != 3 or len(self.orientation) != 9
                or len(self.target_position) != 3 or self.target_speed < 0
                or not all(math.isfinite(value) for value in (
                    *self.relative_position, *self.orientation, self.opening,
                    *self.target_position, self.target_speed))):
            raise ValueError("Progress requires finite physical pose/contact state")


def state_distance(actual: ProgressState, expected: ProgressState) -> float:
    position = math.dist(actual.relative_position, expected.relative_position) / .02
    orientation = math.dist(actual.orientation, expected.orientation) / .5
    opening = abs(actual.opening - expected.opening) / .25
    target = math.dist(actual.target_position, expected.target_position) / .02
    return math.sqrt((position ** 2 + orientation ** 2 + opening ** 2 + target ** 2) / 4)


def progress_potential(state: ProgressState, reference: tuple[ProgressState, ...]) -> float:
    if len(reference) < 2 or not reference[-1].bilateral_grip:
        raise ValueError("GRASP progress requires a measured reference ending in bilateral grip")
    candidates = [
        index / (len(reference) - 1) - state_distance(state, expected)
        for index, expected in enumerate(reference)
        if not expected.bilateral_grip or state.bilateral_grip
    ]
    if not candidates:
        raise ValueError("Reference must include an ungrasped approach state")
    value = max(candidates)
    if state.bilateral_grip and state.target_speed < .03:
        lift = max(0., state.target_position[2] - reference[0].target_position[2])
        # A secure lift beyond the recorded closure is advancement, not pose error.
        value = max(value, 1. + min(lift / .05, 1.))
    return value


def reward_percentage(reward: float, initial_potential: float) -> dict[str, float]:
    maximum = 2. - initial_potential
    if not math.isfinite(reward) or not math.isfinite(maximum) or maximum <= 0:
        raise ValueError("Reward percentage requires a finite reward and a start below the ceiling")
    if reward > maximum + 1e-9:
        raise ValueError("Reward exceeds the defined physical-progress ceiling")
    return {"max_possible_reward": maximum, "reward_percent_of_max": 100. * reward / maximum}


def progress_reward(states: list[ProgressState], reference: tuple[ProgressState, ...],
                    *, invalid: bool = False, wrong_object: bool = False) -> dict:
    if len(states) < 2:
        raise ValueError("Progress reward needs the start and at least one following state")
    values = [progress_potential(state, reference) for state in states]
    drop = any(state.bilateral_grip for state in states) and not states[-1].bilateral_grip
    penalty = .25 * (int(invalid) + int(wrong_object) + int(drop))
    # Including the reset state clips every wholly regressing rollout to zero.
    best = max(values[1:])
    reward = best - values[0] - penalty
    return {"reward": reward, **reward_percentage(reward, values[0]),
            "initial_potential": values[0], "best_potential": best,
            "final_potential": values[-1], "penalty": penalty, "lost_grip": drop}


def short_horizon(reference_steps: int, band_count: int = 4) -> int:
    if reference_steps < 2 or band_count < 1:
        raise ValueError("Positive expert segment length and band count required")
    return max(1, min(8, math.floor(reference_steps / band_count / 2)))
