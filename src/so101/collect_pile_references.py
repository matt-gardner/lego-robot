"""Collect only replay-verified scripted references for immutable pile tasks."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from itertools import zip_longest
from pathlib import Path

import mujoco
import numpy as np

from so101.cloud_checkpoint import atomic_json
from so101.pile_task_runtime import task_success
from so101.rl_pile_adapter import NativePileSimulation, load_native_pool


class ReferenceSimulation(NativePileSimulation):
    def reset_native(self, episode):
        result = super().reset_native(episode)
        self.yaws = {}
        self.initial["objects"] = {}
        for name in self.object_names:
            matrix = self.data.body(name).xmat.reshape(3, 3)
            self.yaws[name] = math.atan2(matrix[1, 0], matrix[0, 0])
            self.initial["objects"][name] = self.position(name).tolist()
        self.reference_rows = []
        self.reference_stable = 0
        self.reference_success = False
        return result

    def step(self, target, phase):
        action = self.encode(np.asarray(target)).astype(np.float64)
        actual, _ = self.bounded_policy_action(action)
        row = {"action": action.tolist(), "phase": phase, "qpos": self.data.qpos.tolist()}
        super().step(actual, phase)
        result = task_success(self.native, self.problem, len(self.reference_rows) + 1,
                              self.reference_stable)
        self.reference_stable = result["stable"]
        self.reference_success = result["success"]
        self.reference_rows.append(row)


def collect(args):
    if args.max_per_task < 1:
        raise ValueError("Reference collection requires a positive per-task attempt limit")
    if args.output.exists():
        raise FileExistsError("Reference output exists; reconcile instead of overwriting")
    args.output.mkdir(parents=True)
    if args.admission is not None:
        entry = json.loads(args.admission.read_text())
        args.pool = args.output / "pilot-pool.json"
        atomic_json(args.pool, {
            "schema_version": 1, "kind": "lego-pile-task-pool", "status": "complete",
            "expert_references": False, "scenes": [entry],
        })
    episodes = load_native_pool(args.pool, set())
    report = {"schema_version": 1, "kind": "pile-expert-references",
              "success_condition": "terminal-native-dwell-v1",
              "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "pool_sha256": hashlib.sha256(args.pool.read_bytes()).hexdigest(),
              "references": [], "attempts": [], "complete": False}
    for task, rows in episodes.items():
        families = {}
        for episode in rows:
            families.setdefault(episode.metadata["scene_family_id"], []).append(episode)
        selected = [episode for group in zip_longest(*families.values())
                    for episode in group if episode is not None][:args.max_per_task]
        for episode in selected:
            sim = ReferenceSimulation(episode, args.model_root, backend="mujoco")
            try:
                target, destination, _ = sim.reset_native(episode)
                sim.record_images = False
                initial = np.empty(mujoco.mj_stateSize(sim.model, mujoco.mjtState.mjSTATE_INTEGRATION))
                mujoco.mj_getState(sim.model, sim.data, initial, mujoco.mjtState.mjSTATE_INTEGRATION)
                try:
                    scripted = sim.demonstrate_command(task, target, destination)
                except RuntimeError as error:
                    if not str(error).startswith("No vertical IK solution for "):
                        raise
                    scripted = {"success": False, "reason": "scripted_ik_failed",
                                "error": str(error)}
                hold = sim.data.ctrl[sim.actuators].copy()
                for _ in range(20):
                    sim.step(hold, "native_reference_verify")
                actions = sim.reference_rows
                sim.reset_native(episode)
                restored = np.empty_like(initial)
                mujoco.mj_getState(sim.model, sim.data, restored, mujoco.mjtState.mjSTATE_INTEGRATION)
                if not np.array_equal(initial, restored):
                    raise ValueError("Reference replay did not restore the exact native integration state")
                maximum_error = 0.
                for row in actions:
                    maximum_error = max(maximum_error, float(np.max(
                        np.abs(sim.data.qpos - np.asarray(row["qpos"])),
                    )))
                    actual, _ = sim.bounded_policy_action(np.asarray(row["action"]))
                    sim.step(actual, row["phase"])
                accepted = sim.reference_success and maximum_error <= 1e-7
                attempt = {"source": str(episode.source), "task": task,
                           "scripted_result": scripted, "replay_success": sim.reference_success,
                           "maximum_qpos_error": maximum_error, "accepted": accepted}
                report["attempts"].append(attempt)
                if accepted:
                    identity = hashlib.sha256(str(episode.source).encode()).hexdigest()
                    frames = args.output / f"{identity}.json"
                    atomic_json(frames, actions)
                    report["references"].append({
                        **attempt, "frames": str(frames.resolve()),
                        "frames_sha256": hashlib.sha256(frames.read_bytes()).hexdigest(),
                        "scene_files_sha256": episode.metadata["native_entry"]["files_sha256"],
                    })
                atomic_json(args.output / "manifest.json", report)
                print(json.dumps({"event": "pile_reference_attempt", **attempt}), flush=True)
            finally:
                sim.close()
    report["complete"] = True
    atomic_json(args.output / "manifest.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--pool", type=Path)
    inputs.add_argument("--admission", type=Path)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-per-task", type=int, default=20)
    collect(parser.parse_args())


if __name__ == "__main__":
    main()
