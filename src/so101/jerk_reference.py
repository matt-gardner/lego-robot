"""Measure each problem's own reference command jerk, without pooling problems."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from so101.cloud_checkpoint import atomic_json


def reference_window(actions: np.ndarray, horizon: int, assistance: float,
                     hold_final: bool = True) -> np.ndarray:
    if horizon < 4 or not np.isfinite(assistance) or not 0 <= assistance <= 1:
        raise ValueError("Reference horizon must be >=4 and assistance in [0,1]")
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or not len(actions) or actions.shape[1] < 5 or not np.isfinite(actions).all():
        raise ValueError("Reference requires finite recorded arm commands")
    prefix = min(int(len(actions) * assistance), len(actions) - 1)
    window = actions[prefix:prefix + horizon, :5]
    return (np.pad(window, ((0, horizon - len(window)), (0, 0)), mode="edge")
            if hold_final else window)


class ReferenceJerkMargins:
    def __init__(self, sources_by_task: dict, horizon: int, fps: float, hold_final: bool = True,
                 workspace: Path = Path(".")):
        if horizon < 4 or not np.isfinite(fps) or fps <= 0:
            raise ValueError("Reference horizon must be >=4 and FPS finite and positive")
        self.horizon, self.fps, self.hold_final = horizon, fps, hold_final
        self.workspace = workspace
        self.sources, self.records, self.profiles = {}, {}, {}
        for task, sources in sources_by_task.items():
            for source in sources:
                self.sources[str(Path(source))] = task

    def summary(self, source: str | Path, assistance: float) -> dict:
        if not np.isfinite(assistance) or not 0 <= assistance <= 1:
            raise ValueError("Reference assistance must be in [0,1]")
        source = str(Path(source))
        key = (source, assistance)
        if key not in self.profiles:
            if source not in self.records:
                path = self.workspace / source
                metadata_path, frames_path = path / "episode.json", path / "frames.jsonl"
                array, reason = None, None
                if source not in self.sources:
                    reason = "no_reference_for_problem"
                elif not metadata_path.exists() or not frames_path.exists():
                    reason = "missing_recording"
                else:
                    metadata = json.loads(metadata_path.read_text())
                    if metadata["task"] != self.sources[source]:
                        raise ValueError(f"Reference command does not match: {source}")
                    if metadata.get("outcome") != "success":
                        reason = "not_successful"
                    elif metadata.get("fps") != self.fps:
                        reason = "missing_or_mismatched_fps"
                    else:
                        with frames_path.open() as stream:
                            actions = [json.loads(line)["action"] for line in stream if line.strip()]
                        if not actions:
                            reason = "empty_recording"
                        else:
                            array = np.asarray(actions, dtype=np.float32)
                            reference_window(array, self.horizon, 0, self.hold_final)
                self.records[source] = (array, reason)
            actions, reason = self.records[source]
            margin = None
            if actions is not None:
                window = reference_window(actions, self.horizon, assistance, self.hold_final)
                if len(window) < 4:
                    reason = "reference_window_too_short"
                else:
                    margin = float(np.sqrt(np.mean(np.diff(window, n=3, axis=0) ** 2)))
            self.profiles[key] = {
                "source": source, "command": self.sources.get(source), "assistance": assistance,
                "margin": margin, "reference_count": int(margin is not None),
                "disabled_reason": reason,
            }
        return self.profiles[key]

    def protocol(self) -> dict:
        return {
            "version": 4, "metric": "RMS third finite difference of commanded five-joint positions",
            "units": "degrees/frame^3", "horizon": self.horizon, "fps": self.fps,
            "reference_scope": "problem", "aggregation": "none",
            "minimum_reference_commands": 4,
            "fixed_horizon": self.hold_final,
            "window": ("expert suffix after the same assistance prefix; hold final command to fill the fixed horizon"
                       if self.hold_final else "expert suffix after the same assistance prefix; stop at reference end or horizon"),
        }

    def calibrate_fallback(self) -> dict:
        profiles = [self.summary(source, 0.) for source in sorted(self.sources)]
        valid = [item for item in profiles if item["margin"] is not None]
        if not valid:
            raise ValueError("No valid successful references for pooled jerk calibration")
        provenance = []
        for item in valid:
            path = self.workspace / item["source"]
            provenance.append({
                "source": item["source"], "task": item["command"],
                "margin": item["margin"],
                "sha256": {
                    name: hashlib.sha256((path / name).read_bytes()).hexdigest()
                    for name in ("episode.json", "frames.jsonl")
                },
            })
        return {
            "schema": 1, "scope": "frozen_successful_training_references",
            "percentile": 75, "margin": float(np.percentile(
                [item["margin"] for item in valid], 75,
            )), "reference_count": len(valid),
            "excluded_count": len(profiles) - len(valid),
            "protocol": self.protocol(), "assistance": 0.,
            "references": provenance,
        }

    def summary_with_fallback(self, source: str | Path, assistance: float,
                              calibration: dict) -> dict:
        profile = self.summary(source, assistance)
        if profile["margin"] is not None:
            return profile
        if (calibration["protocol"] != self.protocol()
                or calibration["scope"] != "frozen_successful_training_references"
                or calibration["reference_count"] < 1
                or not np.isfinite(calibration["margin"]) or calibration["margin"] < 0):
            raise ValueError("Invalid frozen reference jerk calibration")
        return {
            **profile, "margin": calibration["margin"],
            "reference_count": calibration["reference_count"],
            "reference_scope": "frozen_reference_pool",
            "problem_reference_disabled_reason": profile["disabled_reason"],
            "disabled_reason": None,
        }

    def snapshot(self) -> dict:
        return {**self.protocol(), "profiles": list(self.profiles.values()),
                "sources": self.sources}

    def report(self, assistance: float) -> dict:
        return {**self.protocol(), "assistance": assistance,
                "problems": {source: self.summary(source, assistance) for source in self.sources}}


def calibrate_margins(sources_by_task: dict, horizon: int, assistance: float, fps: float,
                      hold_final: bool = True) -> dict:
    return ReferenceJerkMargins(sources_by_task, horizon, fps, hold_final).report(assistance)


def problem_margin(report: dict, source: str | Path) -> float | None:
    entry = report["problems"].get(str(Path(source)))
    if not entry or entry["margin"] is None:
        return None
    margin = float(entry["margin"])
    if not np.isfinite(margin) or margin < 0 or entry["reference_count"] < 1:
        raise ValueError(f"Invalid reference jerk margin for {source}")
    return margin


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--assistance", type=float, default=.8)
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument("--hold-final", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    split = json.loads(args.split.read_text())
    report = calibrate_margins(split["training"], args.horizon, args.assistance, args.fps, args.hold_final)
    atomic_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "references"}, indent=2))
