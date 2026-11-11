"""Checkpointed task-type assistance calibration for four-rollout groups."""
from __future__ import annotations

import math


def select_curriculum(training, specification):
    """Resolve an explicit starting pool against the frozen training split."""
    if set(specification) != set(training):
        raise ValueError("Initial curriculum must specify every training task")
    selected, assistance = {}, {}
    for task, rows in specification.items():
        available = {str(episode.source): episode for episode in training[task]}
        if not rows:
            raise ValueError(f"Initial curriculum has no problems for {task}")
        selected[task] = []
        for row in rows:
            source, value = row["source"], row["assistance"]
            if source not in available:
                raise ValueError(f"Curriculum problem is outside frozen TRAIN: {source}")
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= .95:
                raise ValueError(f"Invalid initial curriculum assistance: {source}")
            if source in assistance:
                raise ValueError(f"Duplicate initial curriculum problem: {source}")
            selected[task].append(available[source])
            assistance[source] = value
    return selected, assistance


class AssistanceCalibration:
    def __init__(self, state: dict, *, alpha: float = .2, increment: float = .05):
        if not 0 < alpha <= 1 or not 0 < increment <= 1:
            raise ValueError("Calibration EMA alpha and assistance increment must be in (0, 1]")
        if state and (state.get("alpha") != alpha or state.get("increment") != increment):
            raise ValueError("Resume assistance calibration settings differ")
        self.state = state
        state.setdefault("alpha", alpha)
        state.setdefault("increment", increment)
        state.setdefault("tasks", {})

    def start(self, task: str, initial: float, maximum: float) -> float:
        if not all(math.isfinite(value) for value in (initial, maximum)) or not 0 <= initial <= maximum <= 1:
            raise ValueError("Assistance must be finite and within the supported range")
        record = self.state["tasks"].setdefault(task, {
            "ema": initial, "groups": 0, "mixed": 0, "all_passed": 0, "all_failed": 0,
        })
        value = record["ema"]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid checkpointed task assistance EMA")
        return min(value, maximum)

    def record(self, task: str, assistance: float, successes: int, total: int,
               maximum: float) -> tuple[float, str]:
        if total != 4 or type(successes) is not int or not 0 <= successes <= total:
            raise ValueError("Difficulty calibration requires exactly four valid rollout outcomes")
        if not math.isfinite(assistance) or not 0 <= assistance <= maximum <= 1:
            raise ValueError("Observed assistance is outside the supported range")
        outcome = "mixed" if 0 < successes < total else "all_passed" if successes else "all_failed"
        direction = -1 if successes == total else 1 if not successes else 0
        following = min(maximum, max(0., assistance + direction * self.state["increment"]))
        record = self.state["tasks"][task]
        record["ema"] += self.state["alpha"] * (following - record["ema"])
        record["groups"] += 1
        record[outcome] += 1
        return following, outcome
