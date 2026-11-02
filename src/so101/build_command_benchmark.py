from __future__ import annotations

import argparse
import json
from pathlib import Path

from so101.robot_sim import BeamSimulation, DESTINATION_NAMES, MODEL_ROOT, OBJECT_NAMES


def build(output: Path, examples_per_task: int, seed: int, model_root: Path) -> None:
    if examples_per_task <= 0:
        raise ValueError("examples_per_task must be positive")
    if output.exists():
        raise FileExistsError(output)
    sim = BeamSimulation(model_root)
    examples = []
    try:
        for task_index, task in enumerate(("grasp", "view", "push", "place")):
            accepted = 0
            candidate_seed = seed + task_index * 1_000
            while accepted < examples_per_task:
                target = OBJECT_NAMES[accepted % len(OBJECT_NAMES)]
                destination = (
                    DESTINATION_NAMES[accepted % len(DESTINATION_NAMES)]
                    if task in ("push", "place")
                    else None
                )
                sim.reset(
                    candidate_seed,
                    varied_layout=True,
                    real_scene=True,
                    color_targets=True,
                )
                try:
                    result = sim.demonstrate_command(task, target, destination)
                except (RuntimeError, ValueError) as exception:
                    result = {"success": False, "error": str(exception)}
                if result["success"]:
                    examples.append({
                        "task": task,
                        "seed": candidate_seed,
                        "target": target,
                        "destination": destination,
                        "expert_verification": result,
                    })
                    accepted += 1
                candidate_seed += 1
    finally:
        sim.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "examples_per_task": examples_per_task,
        "selection": "First expert-verified scene for each balanced task/target slot",
        "examples": examples,
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples-per-task", type=int, default=10)
    parser.add_argument("--seed", type=int, default=30_000)
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    args = parser.parse_args()
    build(args.output, args.examples_per_task, args.seed, args.model_root)


if __name__ == "__main__":
    main()
