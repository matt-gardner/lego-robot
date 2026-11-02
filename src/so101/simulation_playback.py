"""Render bounded, on-demand dual-camera movies without importing a policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import cv2
import mujoco
import numpy as np

from so101.robot_sim import BeamSimulation, restore_episode


def render(source: Path, output: Path, model_root: Path, evidence: Path | None = None):
    source = source.resolve()
    metadata = json.loads((source / "episode.json").read_text())
    rows = [json.loads(line) for line in (source / "frames.jsonl").read_text().splitlines()]
    recording = json.loads(evidence.read_text()) if evidence else None
    sim = BeamSimulation(model_root)
    encoder = None
    try:
        restore_episode(sim, metadata)
        sim.fps = float(recording["fps"] if recording else metadata.get("fps", 10))
        if not np.isfinite(sim.fps) or sim.fps <= 0:
            raise ValueError("Recording FPS must be finite and positive")
        if recording:
            prefix = min(int(len(rows) * recording["assistance"]), max(0, len(rows) - 1))
            if not recording.get("qpos"):
                for row in rows[:prefix]:
                    goal, _ = sim.bounded_policy_action(row["action"])
                    sim.step(goal, "playback_prefix")
            actions = recording["actions"]
            snapshots = recording.get("qpos")
            if snapshots and len(snapshots) != len(actions) + 1:
                raise ValueError("Recorded physics states must include both endpoints")
            sequence = range(len(actions) + 1)
            labels = ("Overview (recorded physics)", "Wrist (recorded physics)") if snapshots else (
                "Overview (action reconstruction)", "Wrist (action reconstruction)")
        else:
            actions = [row["action"] for row in rows]
            snapshots = [row["qpos"] for row in rows] if rows and all("qpos" in row for row in rows) else None
            sequence = range(len(rows))
            labels = ("Overview (recorded physics)" if snapshots else "Overview (action reconstruction)", "Wrist (original recording)")
        if not actions:
            raise ValueError("No recorded actions available for playback")
        encoder = subprocess.Popen([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "640x264",
            "-framerate", str(sim.fps), "-i", "-", "-an", "-c:v", "libx264",
            "-preset", "ultrafast", "-threads", "2", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(output),
        ], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        for index in sequence:
            if snapshots:
                pose = np.asarray(snapshots[index], dtype=np.float64)
                if pose.shape != sim.data.qpos.shape or not np.isfinite(pose).all():
                    raise ValueError("Invalid recorded physics state")
                sim.data.qpos[:] = pose
                mujoco.mj_forward(sim.model, sim.data)
            overview = sim.overview_image()
            if recording:
                wrist = sim.image()
            else:
                path = (source / rows[index]["image"]).resolve()
                if not path.is_relative_to(source) or path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    raise ValueError("Invalid recorded wrist image reference")
                wrist = cv2.imread(str(path))
                if wrist is None:
                    raise OSError(f"Cannot read recorded wrist frame: {path}")
                wrist = cv2.cvtColor(wrist, cv2.COLOR_BGR2RGB)
                if wrist.shape != overview.shape:
                    raise ValueError("Recorded wrist and overview frame dimensions differ")
            header = np.zeros((24, 640, 3), dtype=np.uint8)
            for x, label in zip((6, 326), labels, strict=True):
                cv2.putText(header, label, (x, 16), cv2.FONT_HERSHEY_SIMPLEX,
                            .38, (225, 235, 245), 1, cv2.LINE_AA)
            encoder.stdin.write(np.vstack((header, np.hstack((overview, wrist)))).tobytes())
            if not snapshots and index < len(actions):
                goal, _ = sim.bounded_policy_action(actions[index])
                sim.step(goal, "playback")
        encoder.stdin.close()
        message = encoder.stderr.read().decode(errors="replace")
        if encoder.wait() != 0:
            raise OSError(f"Video encoding failed: {message}")
    finally:
        if encoder is not None:
            if encoder.stdin and not encoder.stdin.closed:
                encoder.stdin.close()
            if encoder.poll() is None:
                encoder.terminate()
                encoder.wait()
            encoder.stderr.close()
        sim.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args()
    render(args.source, args.output, args.model_root, args.evidence)
