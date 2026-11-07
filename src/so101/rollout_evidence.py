"""Small, bounded policy-rollout records; never loads a policy or replays physics."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

from so101.cloud_checkpoint import atomic_json


class RolloutRecorder:
    def __init__(self, path: Path, sim, seed: int, source: str, assistance: float):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.data = {
            "version": 2, "kind": "actual-policy-rollout",
            "seed": seed, "source": source, "assistance": assistance,
            "fps": sim.fps, "scope": "policy suffix after expert assistance",
            "actions": [], "states": [], "frames": [],
            "qpos": [sim.data.qpos.tolist()],
            "scene_version": sim.scene_version,
        }
        self.capture(sim)

    def capture(self, sim) -> None:
        import cv2

        step = len(self.data["actions"])
        if self.data["frames"] and self.data["frames"][-1]["step"] == step:
            return
        filename = f"{self.path.stem}_{step:04}.jpg"
        if not cv2.imwrite(str(self.path.parent / filename),
                           cv2.cvtColor(sim.image(), cv2.COLOR_RGB2BGR)):
            raise OSError(f"Cannot record rollout image: {filename}")
        self.data["frames"].append({"step": step, "image": filename})

    def step(self, sim, action) -> None:
        self.data["actions"].append(action.tolist())
        self.data["states"].append(sim.state().tolist())
        self.data["qpos"].append(sim.data.qpos.tolist())
        if len(self.data["actions"]) % 16 == 0:
            self.capture(sim)

    def finish(self, sim, outcome: dict) -> str:
        self.capture(sim)
        self.data["outcome"] = outcome
        atomic_json(self.path, self.data)
        return str(Path("rollouts") / self.path.parent.name / self.path.name)


def prune_evidence(output: Path, keep_steps: int) -> None:
    if keep_steps < 1:
        raise ValueError("Rollout retention must be positive")
    root = output / "rollouts"
    if root.is_symlink():
        raise ValueError("Rollout retention root must not be a symlink")
    if not root.is_dir():
        return
    steps = sorted((path for path in root.iterdir()
                    if re.fullmatch(r"step_\d{6,}", path.name)
                    and path.is_dir() and not path.is_symlink()),
                   key=lambda path: int(path.name.removeprefix("step_")))
    for path in steps[:-keep_steps]:
        shutil.rmtree(path)
