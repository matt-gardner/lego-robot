from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from so101.robot_sim import (
    BeamSimulation,
    DESTINATION_NAMES,
    MODEL_COMMIT,
    MODEL_ROOT,
    OBJECT_NAMES,
)


def command_specs(
    seed: int, tasks: tuple[str, ...] = ("grasp", "view", "push", "place")
) -> list[tuple[str, str, str | None, str]]:
    specs = []
    for phrase in ("grasp", "view"):
        if phrase not in tasks:
            continue
        for target in OBJECT_NAMES:
            specs.append((phrase, target, None, f"{seed}:{phrase}"))
    for phrase in ("push", "place"):
        if phrase not in tasks:
            continue
        for target in OBJECT_NAMES:
            for destination in DESTINATION_NAMES:
                specs.append((
                    phrase,
                    target,
                    destination,
                    f"{seed}:{phrase}:{target}",
                ))
    return specs


def collect(
    root: Path,
    scenes: int,
    seed: int,
    model_root: Path,
    tasks: tuple[str, ...],
    recovery_checkpoint: Path | None,
    device: str,
    benchmark: Path | None = None,
    repeats: int = 1,
) -> None:
    if scenes <= 0 or repeats <= 0:
        raise ValueError("scenes and repeats must be positive")
    root.mkdir(parents=True, exist_ok=False)
    sim = BeamSimulation(model_root)
    if recovery_checkpoint is not None:
        from so101.evaluate_entity_sim import EntityPolicy

        controller = EntityPolicy(recovery_checkpoint, device, True)
    else:
        controller = None
    successful = []
    attempts = []
    started = time.monotonic()
    episode_index = 0
    if benchmark is None:
        jobs = [
            (seed + scene_index, phrase, target, destination, pair_id)
            for scene_index in range(scenes)
            for phrase, target, destination, pair_id in command_specs(
                seed + scene_index, tasks
            )
        ]
    else:
        examples = json.loads(benchmark.read_text())["examples"]
        jobs = [
            (
                example["seed"],
                example["task"],
                example["target"],
                example["destination"],
                (
                    f"{example['seed']}:{example['task']}:"
                    f"{example['target']}:{example['destination']}"
                ),
            )
            for example in examples
            if example["task"] in tasks
            for _ in range(repeats)
        ]
    try:
        for scene_seed, phrase, target, destination, pair_id in jobs:
                sim.episode_root = None
                sim.reset(
                    scene_seed,
                    varied_layout=True,
                    real_scene=True,
                    color_targets=True,
                )
                recovery = None
                if controller is not None:
                    entities = [(target, "object")]
                    if destination is not None:
                        entities.append((destination, "destination"))
                    sim.configure_command(phrase, entities)
                    controller.reset(sim, scene_seed + episode_index)
                    requested_prefix = int(
                        np.random.default_rng(
                            scene_seed * 100 + episode_index
                        ).integers(20, 81)
                    )
                    executed_prefix = 0
                    recovery_error = None
                    while executed_prefix < requested_prefix:
                        try:
                            actions, _, _, _ = controller.predict_command(
                                sim, phrase, entities
                            )
                        except (RuntimeError, ValueError) as exception:
                            recovery_error = str(exception)
                            break
                        for action in actions[
                            :min(len(actions), requested_prefix - executed_prefix)
                        ]:
                            try:
                                goal, _ = sim.bounded_policy_action(action)
                            except ValueError as exception:
                                recovery_error = str(exception)
                                break
                            sim.step(goal, "policy_prefix")
                            controller.observe(sim, action)
                            executed_prefix += 1
                        if recovery_error is not None:
                            break
                    sim.yaws = {
                        name: float(np.arctan2(
                            sim.data.body(name).xmat.reshape(3, 3)[1, 0],
                            sim.data.body(name).xmat.reshape(3, 3)[0, 0],
                        ))
                        for name in OBJECT_NAMES
                    }
                    sim.initial = {
                        **sim.initial,
                        "arm_state": sim.state().tolist(),
                        "objects": {
                            name: sim.position(name).tolist()
                            for name in OBJECT_NAMES
                        },
                        "yaws": sim.yaws,
                        "recovery_prefix": {
                            "requested_steps": requested_prefix,
                            "executed_steps": executed_prefix,
                            "error": recovery_error,
                        },
                    }
                    sim.snapshot_initial_physics()
                    sim.rows = []
                    sim.frame_start = sim.data.time
                    recovery = sim.initial["recovery_prefix"]
                episode = root / f"episode_{episode_index:05}"
                episode_index += 1
                (episode / "wrist").mkdir(parents=True)
                sim.episode_root = episode
                try:
                    result = sim.demonstrate_command(phrase, target, destination)
                except (RuntimeError, ValueError) as exception:
                    result = {
                        "success": False,
                        "reason": "Expert trajectory failed",
                        "error": str(exception),
                    }
                sim.overview(episode / "diagnostic_overview.jpg")
                report = {
                    "simulation": True,
                    "model_commit": MODEL_COMMIT,
                    "outcome": "success" if result["success"] else "failed",
                    "fps": sim.fps,
                    "frames": len(sim.rows),
                    "task": phrase,
                    "pair_id": pair_id,
                    "scene_seed": scene_seed,
                    "target": target,
                    "target_label": sim.target_labels[target],
                    "destination": destination,
                    "initial": sim.initial,
                    "verification": result,
                    "recovery_prefix": recovery,
                    "duration": len(sim.rows) / sim.fps,
                    "timing_acceptable": len(sim.rows) > 1 and bool(np.allclose(
                        np.diff([row["elapsed"] for row in sim.rows]),
                        1 / sim.fps,
                        atol=1e-8,
                        rtol=0,
                    )),
                }
                (episode / "frames.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in sim.rows)
                )
                (episode / "episode.json").write_text(json.dumps(report, indent=2))
                attempts.append(report)
                if result["success"]:
                    successful.append(str(episode))
                print(json.dumps({
                    "episode": episode.name,
                    "seed": scene_seed,
                    "task": phrase,
                    "target": target,
                    "destination": destination,
                    **result,
                }), flush=True)
                (root / "collection.json").write_text(json.dumps({
                    "simulation": True,
                    "requested_scenes": scenes,
                    "benchmark": str(benchmark) if benchmark is not None else None,
                    "repeats": repeats,
                    "successful_episodes": successful,
                    "attempts": attempts,
                    "wall_seconds": time.monotonic() - started,
                }, indent=2))
    finally:
        sim.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scenes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=("grasp", "view", "push", "place"),
        default=("grasp", "view", "push", "place"),
    )
    parser.add_argument("--recovery-checkpoint", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    collect(
        args.root,
        args.scenes,
        args.seed,
        args.model_root,
        tuple(args.tasks),
        args.recovery_checkpoint,
        args.device,
        args.benchmark,
        args.repeats,
    )


if __name__ == "__main__":
    main()
