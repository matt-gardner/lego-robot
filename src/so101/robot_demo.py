"""Continuous wrist-only episodes and offline LeRobot export."""
from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from so101.robot_control import JOINTS


class WristStream:
    """Acquire latest wrist frames with host delivery times, not sensor timestamps."""

    def __init__(self, index: int = 0):
        from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

        self.camera = OpenCVCamera(
            OpenCVCameraConfig(index_or_path=index, width=640, height=480, fps=30)
        )
        self.camera.connect()

    def latest(self) -> tuple[np.ndarray, float]:
        with self.camera.frame_lock:
            frame = self.camera.latest_frame
            timestamp = self.camera.latest_timestamp
        if frame is None or timestamp is None:
            raise RuntimeError("Wrist camera has no frame")
        if time.perf_counter() - timestamp > 0.2:
            raise RuntimeError("Wrist camera frame is stale; stop recording and motion")
        return frame.copy(), timestamp

    def close(self):
        self.camera.disconnect()


class EpisodeRecorder:
    def __init__(self, root: Path, camera, *, fps: int = 10, task: str):
        if fps not in range(10, 31):
            raise ValueError("Recording fps must be within [10, 30]")
        root.mkdir(parents=True, exist_ok=False)
        self.root = root
        self.camera = camera
        self.fps = fps
        self.task = task
        self.start = time.perf_counter()
        self.rows: list[dict] = []
        self.events: list[dict] = []
        self.pending = queue.Queue(maxsize=120)
        self.error: Exception | None = None
        self.closed = False
        self.thread = threading.Thread(target=self._write, daemon=True)
        (root / "wrist").mkdir()
        self.thread.start()

    def _write(self):
        try:
            with (self.root / "frames.jsonl").open("w") as log:
                while True:
                    item = self.pending.get()
                    if item is None:
                        return
                    row, rgb = item
                    path = self.root / row["image"]
                    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                        raise RuntimeError(f"Could not write {path}")
                    log.write(json.dumps(row) + "\n")
                    log.flush()
        except Exception as error:
            self.error = error

    def sample(self, state: dict[str, float], action: dict[str, float]):
        if self.closed:
            raise RuntimeError("Episode has already ended")
        if self.error is not None:
            raise RuntimeError("Episode writer failed") from self.error
        rgb, camera_time = self.camera.latest()
        now = time.perf_counter()
        row = {
            "index": len(self.rows),
            "elapsed": now - self.start,
            "camera_elapsed": camera_time - self.start,
            "camera_age": now - camera_time,
            "state": [state[name] for name in JOINTS],
            "action": [action[name] for name in JOINTS],
            "image": f"wrist/{len(self.rows):06}.jpg",
        }
        if not all(np.isfinite(row[key]).all() for key in ("state", "action")):
            raise ValueError("Cannot record invalid state/action")
        try:
            self.pending.put_nowait((row, rgb))
        except queue.Full as error:
            raise RuntimeError("Recorder cannot keep up; stop instead of dropping frames") from error
        self.rows.append(row)

    def event(self, value: dict):
        self.events.append({"elapsed": time.perf_counter() - self.start, **value})
        (self.root / "events.json").write_text(json.dumps(self.events, indent=2))

    def finish(self, outcome: str, evidence: str):
        if outcome not in ("unverified", "success", "failed", "aborted"):
            raise ValueError("Explicit outcome required")
        if self.closed:
            raise RuntimeError("Episode has already ended")
        self.closed = True
        while self.thread.is_alive():
            try:
                self.pending.put(None, timeout=0.2)
                break
            except queue.Full:
                continue
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise RuntimeError("Episode writer did not stop")
        if self.error is not None:
            raise RuntimeError("Episode writer failed") from self.error
        elapsed = np.array([row["elapsed"] for row in self.rows])
        intervals = np.diff(elapsed)
        drift = elapsed - elapsed[0] - np.arange(len(elapsed)) / self.fps if len(elapsed) else []
        report = {
            "fps": self.fps,
            "task": self.task,
            "joint_names": list(JOINTS),
            "units": "degrees; gripper 0-100 percent",
            "frames": len(self.rows),
            "duration_s": float(elapsed[-1] - elapsed[0]) if len(elapsed) else 0,
            "max_interval_s": float(max(intervals)) if len(intervals) else None,
            "max_grid_drift_s": float(max(abs(drift))) if len(elapsed) else None,
            "max_camera_age_s": max((row["camera_age"] for row in self.rows), default=None),
            "timing_acceptable": bool(
                len(intervals) > 0
                and max(intervals) <= 1.75 / self.fps
                and min(intervals) > 0
                and (len(elapsed) - 1) / (elapsed[-1] - elapsed[0]) >= self.fps * 0.85
                and all(0 <= row["camera_age"] <= 0.2 for row in self.rows)
            ),
            "outcome": outcome,
            "evidence": evidence,
            "camera_features": ["observation.images.wrist"],
            "camera_timestamp_kind": "host frame delivery; not hardware exposure time",
            "continuous": True,
            "events": self.events,
        }
        (self.root / "episode.json").write_text(json.dumps(report, indent=2) + "\n")
        return report


def export_episodes(sources: list[Path], destination: Path, repo_id: str):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    episodes = []
    for source in sources:
        report = json.loads((source / "episode.json").read_text())
        if report["outcome"] != "success" or not report["timing_acceptable"]:
            raise ValueError(f"{source}: only verified, timing-qualified successes can be exported")
        rows = [json.loads(line) for line in (source / "frames.jsonl").read_text().splitlines()]
        if not rows or len(rows) != report["frames"]:
            raise ValueError(f"{source}: frame count mismatch")
        episodes.append((source, report, rows))
    if not episodes or len({report["fps"] for _, report, _ in episodes}) != 1:
        raise ValueError("Provide episodes with one common recording rate")
    simulated = {bool(report.get("simulation", False)) for _, report, _ in episodes}
    if len(simulated) != 1:
        raise ValueError("Do not mix simulated and real episodes before calibrating their joint conventions")
    target_box_presence = {
        "target_box" in row
        for _, _, rows in episodes
        for row in rows
    }
    if len(target_box_presence) != 1:
        raise ValueError("Target-box labels must be present on every frame or no frames")
    has_target_boxes = True in target_box_presence
    target_depth_presence = {
        "target_depth_m" in row
        for _, _, rows in episodes
        for row in rows
    }
    if len(target_depth_presence) != 1:
        raise ValueError("Target-depth labels must be present on every frame or no frames")
    has_target_depths = True in target_depth_presence
    if has_target_depths != has_target_boxes:
        raise ValueError("Target boxes and depths must be exported together")
    first_source, _, first_rows = episodes[0]
    first_image = cv2.imread(str(first_source / first_rows[0]["image"]))
    if first_image is None:
        raise RuntimeError(f"Missing image {first_source / first_rows[0]['image']}")
    image_shape = first_image.shape
    features = {
        "observation.state": {
            "dtype": "float32", "shape": (len(JOINTS),), "names": list(JOINTS)
        },
        "action": {"dtype": "float32", "shape": (6,), "names": list(JOINTS)},
        "observation.images.wrist": {
            "dtype": "image", "shape": image_shape, "names": ["height", "width", "channels"]
        },
        "recording.elapsed": {"dtype": "float64", "shape": (1,), "names": None},
        "recording.camera_elapsed": {"dtype": "float64", "shape": (1,), "names": None},
    }
    if has_target_boxes:
        features["observation.target_box"] = {
            "dtype": "float32",
            "shape": (5,),
            "names": ["center_x", "center_y", "width", "height", "visible"],
        }
        features["observation.target_depth_m"] = {
            "dtype": "float32",
            "shape": (1,),
            "names": ["camera_depth_m"],
        }
    dataset = LeRobotDataset.create(
        repo_id=repo_id, root=destination, fps=episodes[0][1]["fps"],
        features=features, robot_type="so101_sim" if True in simulated else "so101_follower", use_videos=False,
        image_writer_threads=4,
    )
    try:
        for source, report, rows in episodes:
            # Resample the actual clock; never compress deliberation or other gaps.
            times = np.array([row["elapsed"] for row in rows])
            count = int(np.floor((times[-1] - times[0]) * report["fps"] + 1e-8)) + 1
            grid = times[0] + np.arange(count) / report["fps"]
            indices = np.searchsorted(times, grid).clip(0, len(times) - 1)
            previous = np.maximum(indices - 1, 0)
            indices = np.where(abs(times[previous] - grid) < abs(times[indices] - grid), previous, indices)
            if np.max(abs(times[indices] - grid)) > 0.09:
                raise ValueError(f"{source}: frame timestamps cannot support the requested grid")
            for index in indices:
                row = rows[index]
                bgr = cv2.imread(str(source / row["image"]))
                if bgr is None:
                    raise RuntimeError(f"Missing image {source / row['image']}")
                if bgr.shape != image_shape:
                    raise ValueError(f"Inconsistent wrist image shape: {source / row['image']}")
                state = np.asarray(row["state"], dtype=np.float32)
                frame = {
                    "observation.state": state,
                    "action": np.array(row["action"], dtype=np.float32),
                    "observation.images.wrist": cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                    "recording.elapsed": np.array([row["elapsed"]], dtype=np.float64),
                    "recording.camera_elapsed": np.array([row["camera_elapsed"]], dtype=np.float64),
                    "task": report["task"],
                }
                if has_target_boxes:
                    target_box = np.asarray(row["target_box"], dtype=np.float32)
                    if (
                        target_box.shape != (5,)
                        or not np.isfinite(target_box).all()
                        or np.any(target_box[:4] < 0)
                        or np.any(target_box[:4] > 1)
                        or target_box[4] not in (0, 1)
                    ):
                        raise ValueError(f"{source}: invalid target box at frame {row['index']}")
                    frame["observation.target_box"] = target_box
                    target_depth = float(row["target_depth_m"])
                    if not np.isfinite(target_depth) or target_depth <= 0:
                        raise ValueError(f"{source}: invalid target depth at frame {row['index']}")
                    frame["observation.target_depth_m"] = np.array(
                        [target_depth], dtype=np.float32
                    )
                dataset.add_frame(frame)
            dataset.save_episode()
    finally:
        dataset.finalize()
    (destination / "recording_sources.json").write_text(json.dumps([
        {"source": str(source), "report": report} for source, report, _ in episodes
    ], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", type=Path, nargs="+")
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/so101-gray-grasp")
    args = parser.parse_args()
    export_episodes(args.sources, args.destination, args.repo_id)
