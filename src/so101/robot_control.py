from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from typing import Literal, Protocol, TypedDict

Joint = Literal[
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"
]
JOINTS: tuple[Joint, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


class Arm(Protocol):
    def positions(self) -> dict[str, float]: ...
    def enable(self) -> None: ...
    def command(self, positions: dict[str, float]) -> None: ...
    def close(self) -> None: ...


class MotionResult(TypedDict):
    outcome: Literal["target_reached", "partial"]
    start_positions: dict[str, float]
    target_positions: dict[str, float]
    actual_positions: dict[str, float]
    measured_deltas: dict[str, float]
    target_errors: dict[str, float]


class MotionController:
    """Serialize bounded trajectories. Limits are not collision avoidance or an emergency stop."""

    def __init__(
        self,
        arm: Arm,
        *,
        motion: bool,
        sleep: Callable[[float], None] = time.sleep,
        bounds: dict[str, tuple[float, float]] | None = None,
        allow_partial: bool = False,
        speed: float = 5.0,
        step_hz: int = 20,
        on_step: Callable[[dict[str, float], dict[str, float]], None] | None = None,
    ) -> None:
        if not math.isfinite(speed) or not 0 < speed <= 20 or not 10 <= step_hz <= 50:
            raise ValueError("Speed must be within (0, 20] and step_hz within [10, 50]")
        self.arm = arm
        self.motion = motion
        self.sleep = sleep
        self.completion_tolerance = 1.0
        self.allow_partial = allow_partial
        self.speed = speed
        self.step_hz = step_hz
        self.on_step = on_step
        self._holding_targets: dict[str, float] | None = None
        self._last_positions: dict[str, float] | None = None
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.origin = self.checked_positions()
        self.bounds = (
            {name: (value - 15, value + 15) for name, value in self.origin.items()}
            if bounds is None
            else dict(bounds)
        )
        if set(self.bounds) != set(JOINTS) or any(
            not math.isfinite(low)
            or not math.isfinite(high)
            or not low < high
            or not low <= self.origin[name] <= high
            for name, (low, high) in self.bounds.items()
        ):
            raise ValueError("Joint bounds must be finite and contain the startup pose")
        self.moves_left = 3

    def checked_positions(self) -> dict[str, float]:
        positions = self.arm.positions()
        if set(positions) != set(JOINTS) or not all(
            math.isfinite(value) for value in positions.values()
        ):
            raise RuntimeError("Invalid position feedback; motion cannot continue")
        return positions

    def begin_turn(self) -> None:
        self.moves_left = 3

    def status(self) -> dict[str, object]:
        with self.lock:
            try:
                positions = self.checked_positions()
            except BaseException:
                self.stopped.set()
                raise
            return {
                "positions": positions,
                "motion_enabled": self.motion,
                "stopped": self.stopped.is_set(),
                "moves_left_this_turn": self.moves_left,
                "joint_bounds": self.bounds,
                "completion_tolerance": self.completion_tolerance,
                "partial_motion_feedback": self.allow_partial,
                "speed_units_per_second": self.speed,
                "step_hz": self.step_hz,
                "units": "degrees for arm joints; 0-100 percent for gripper",
            }

    def stop(self) -> str:
        self.stopped.set()
        return "Stop latched; no further targets will be sent. Torque may remain on."

    def nudge(self, joint: Joint, delta: float) -> dict[str, object]:
        with self.lock:
            self._require_motion()
            if (
                joint not in JOINTS
                or not math.isfinite(delta)
                or not 0 < abs(delta) <= 5
            ):
                raise ValueError(
                    "Choose a known joint and a nonzero finite delta within +/-5"
                )
            result = self._move({joint: delta}, relative=True)
            return {
                "outcome": result["outcome"],
                "joint": joint,
                "requested_delta": delta,
                "start_positions": result["start_positions"],
                "target": result["target_positions"][joint],
                "actual_positions": result["actual_positions"],
                "measured_delta": result["measured_deltas"][joint],
                "target_error": result["target_errors"][joint],
            }

    def move_to(self, targets: dict[str, float]) -> MotionResult:
        """Move specified joints together to absolute targets in one trajectory."""
        with self.lock:
            self._require_motion()
            if not targets or any(
                name not in JOINTS or not math.isfinite(value)
                for name, value in targets.items()
            ):
                raise ValueError("Choose known joints and finite target positions")
            return self._move(targets, relative=False)

    def _require_motion(self) -> None:
        if not self.motion or self.stopped.is_set():
            raise RuntimeError(
                "Motion is disabled or stop/fault is latched; restart to re-arm"
            )

    def _move(self, requested: dict[str, float], *, relative: bool) -> MotionResult:
        if self.moves_left <= 0:
            raise ValueError(
                "Three moves already used; wait for the next user instruction"
            )
        reported = False
        try:
            start = self.checked_positions()
            if any(
                not self.bounds[name][0] <= start[name] <= self.bounds[name][1]
                for name in JOINTS
            ):
                raise ValueError("Current pose exceeds the startup envelope")
            targets = dict(self._holding_targets or start)
            deltas = {}
            for name, value in requested.items():
                if relative or abs(value - start[name]) > self.completion_tolerance:
                    targets[name] = start[name] + value if relative else value
                    deltas[name] = targets[name] - start[name]
            if not 0 <= targets["gripper"] <= 100:
                raise ValueError("Gripper target must stay within 0-100")
            if any(
                not self.bounds[name][0] <= targets[name] <= self.bounds[name][1]
                for name in JOINTS
            ):
                raise ValueError("Target exceeds configured startup bounds")
            if self._last_positions is not None and any(
                abs(start[name] - self._last_positions[name]) > 2
                for name in JOINTS
                if name not in deltas
            ):
                raise RuntimeError("An uncommanded joint moved between requests; stop latched")
            if not deltas:
                reported = True
                return {
                    "outcome": "target_reached",
                    "start_positions": start,
                    "target_positions": targets,
                    "actual_positions": start,
                    "measured_deltas": {},
                    "target_errors": {},
                }
            self.moves_left -= 1
            self.arm.enable()
            period = 1 / self.step_hz
            trajectory_start = self._holding_targets or start
            steps = max(1, math.ceil(
                max(abs(targets[name] - trajectory_start[name]) for name in deltas)
                / (self.speed * period)
            ))
            command = dict(trajectory_start)

            def checked_motion() -> dict[str, float]:
                actual = self.checked_positions()
                if any(
                    not self.bounds[name][0] <= actual[name] <= self.bounds[name][1]
                    for name in JOINTS
                ):
                    raise RuntimeError("Measured pose exceeded joint bounds; stop latched")
                if any(
                    abs(actual[name] - start[name]) > 2
                    for name in JOINTS
                    if name not in deltas
                ):
                    raise RuntimeError("An uncommanded joint moved; stop latched")
                if any(
                    abs(actual[name] - command[name]) > 8
                    or not min(start[name], targets[name]) - 2
                    <= actual[name]
                    <= max(start[name], targets[name]) + 2
                    for name in deltas
                ):
                    raise RuntimeError("Unexpected joint tracking error; stop latched")
                return actual

            # Speed is per joint, not the sum across simultaneously moving joints.
            for step in range(1, steps + 1):
                tick = time.perf_counter() if self.on_step else None
                if self.stopped.is_set():
                    raise RuntimeError("Stopped during motion; torque may remain on")
                measured = checked_motion()
                command = {
                    name: (
                        trajectory_start[name]
                        + (targets[name] - trajectory_start[name]) * step / steps
                        if name in deltas else targets[name]
                    )
                    for name in JOINTS
                }
                if self.on_step:
                    self.on_step(measured, command)
                self.arm.command(command)
                self.sleep(
                    period if tick is None else max(0, period - (time.perf_counter() - tick))
                )
            deadline = time.monotonic() + 2
            target_reached = False
            while True:
                tick = time.perf_counter() if self.on_step else None
                if self.stopped.is_set():
                    raise RuntimeError("Stopped while checking completion")
                actual = checked_motion()
                if self.on_step:
                    self.on_step(actual, command)
                progress = {
                    name: (actual[name] - start[name]) * math.copysign(1, delta)
                    for name, delta in deltas.items()
                }
                minimum_progress = {
                    name: min(abs(delta) / 2, 0.25) for name, delta in deltas.items()
                }
                reached = {
                    name: abs(actual[name] - targets[name]) <= self.completion_tolerance
                    and progress[name] >= minimum_progress[name]
                    for name in deltas
                }
                if all(reached.values()):
                    target_reached = True
                    break
                if time.monotonic() >= deadline:
                    if self.allow_partial and all(
                        reached[name]
                        or minimum_progress[name] <= progress[name] < abs(deltas[name])
                        for name in deltas
                    ):
                        break
                    raise TimeoutError(
                        f"Joint did not reach target: targets={targets}, actual={actual}, "
                        f"tolerance={self.completion_tolerance:g}, "
                        f"minimum_progress={minimum_progress}; stop latched"
                    )
                self.sleep(
                    period if tick is None else max(0, period - (time.perf_counter() - tick))
                )
            # Measured positions can lag loaded servos; replacing their holding
            # goals with feedback would release grip force and accumulate sag.
            self._holding_targets = dict(command)
            self._last_positions = dict(actual)
            reported = True
            return {
                "outcome": "target_reached" if target_reached else "partial",
                "start_positions": start,
                "target_positions": targets,
                "actual_positions": actual,
                "measured_deltas": {name: actual[name] - start[name] for name in deltas},
                "target_errors": {name: actual[name] - targets[name] for name in deltas},
            }
        finally:
            if not reported:
                self.stopped.set()

    def close(self) -> None:
        self.stop()
        with self.lock:
            self.arm.close()


class SimulatedArm:
    def __init__(self) -> None:
        self.values = dict.fromkeys(JOINTS, 0.0)
        self.values["gripper"] = 50.0

    def positions(self) -> dict[str, float]:
        return dict(self.values)

    def enable(self) -> None:
        pass

    def command(self, positions: dict[str, float]) -> None:
        self.values = dict(positions)

    def close(self) -> None:
        pass
