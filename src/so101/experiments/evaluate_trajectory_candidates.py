from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from so101.evaluate_entity_sim import EntityPolicy
from so101.robot_sim import BeamSimulation, MODEL_ROOT
from so101.train_trajectory_rl import (
    CorpusEpisode,
    load_corpus,
    rollout,
)


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if row["success"]]

    def task_metrics(task: str) -> dict[str, float]:
        task_rows = [row for row in rows if row["task"] == task]
        task_successful = [row for row in task_rows if row["success"]]
        return {
            "success_rate": sum(
                row["success"] for row in task_rows
            ) / len(task_rows),
            "completion_efficiency": (
                float(np.mean([
                    row["completion_efficiency"]
                    for row in task_successful
                ]))
                if task_successful else 0.
            ),
            "rms_jerk": float(np.mean([
                row["rms_jerk"] for row in task_rows
            ])),
            "wrong_object_rate": sum(
                row["wrong_object"] for row in task_rows
            ) / len(task_rows),
        }

    return {
        "episodes": len(rows),
        "success_rate": sum(row["success"] for row in rows) / len(rows),
        "completion_efficiency": (
            float(np.mean([
                row["completion_efficiency"] for row in successful
            ]))
            if successful else 0.
        ),
        "rms_jerk": float(np.mean([
            row["rms_jerk"] for row in rows
        ])),
        "wrong_object_rate": sum(
            row["wrong_object"] for row in rows
        ) / len(rows),
        "by_task": {
            task: task_metrics(task)
            for task in ("grasp", "view", "push", "place")
        },
    }


def evaluate(
    run: Path,
    checkpoint: Path,
    collection: Path,
    output: Path,
    candidates: list[str],
    repeats: int,
    seed: int,
    horizon: int,
    action_steps: int,
    device: str,
    model_root: Path,
) -> None:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    split = json.loads((run / "data_split.json").read_text())
    episodes = load_corpus(collection)
    by_source = {
        str(episode.source): episode
        for task_episodes in episodes.values()
        for episode in task_episodes
    }
    validation: list[CorpusEpisode] = [
        by_source[source] for source in split["validation"]
    ]
    bounds = split["expert_bounds"]
    controller = EntityPolicy(checkpoint, device, True)
    sim = BeamSimulation(model_root)
    delta_paths = {
        path.stem.removeprefix("delta_step_"): path
        for path in run.glob("delta_step_*.pt")
    }
    delta_names = set()
    delta_states = {}
    for candidate in candidates:
        if candidate == "baseline":
            continue
        key = f"{int(candidate):03}"
        if key not in delta_paths:
            raise ValueError(f"No delta checkpoint for step {candidate}")
        state = torch.load(
            delta_paths[key],
            map_location="cpu",
            weights_only=False,
        )["model"]
        delta_states[candidate] = state
        delta_names.update(state)
    parameters = dict(controller.policy.named_parameters())
    baseline = {
        name: parameters[name].detach().cpu().clone()
        for name in delta_names
    }
    report = {
        "run": str(run),
        "checkpoint": str(checkpoint),
        "validation_cases": len(validation),
        "repeats": repeats,
        "seed": seed,
        "candidates": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for candidate in candidates:
            for name, value in baseline.items():
                parameters[name].data.copy_(
                    value.to(parameters[name].device)
                )
            if candidate != "baseline":
                for name, value in delta_states[candidate].items():
                    parameters[name].data.copy_(
                        value.to(parameters[name].device)
                    )
            rows = []
            started = time.monotonic()
            for repeat in range(repeats):
                for case_index, episode in enumerate(validation):
                    trajectory = rollout(
                        sim,
                        controller,
                        episode,
                        seed + repeat * 10_000 + case_index,
                        horizon,
                        action_steps,
                        bounds[episode.task],
                        False,
                    )
                    rows.append({
                        "repeat": repeat,
                        "case_index": case_index,
                        "source": str(episode.source),
                        "task": episode.task,
                        "success": trajectory.success,
                        "wrong_object": trajectory.wrong_object,
                        "steps": trajectory.steps,
                        "completion_efficiency": (
                            trajectory.completion_efficiency
                        ),
                        "rms_jerk": trajectory.rms_jerk,
                    })
                print(json.dumps({
                    "candidate": candidate,
                    "completed_repeats": repeat + 1,
                    "total_repeats": repeats,
                    "elapsed_seconds": time.monotonic() - started,
                }), flush=True)
            candidate_report = {
                **metrics(rows),
                "seconds": time.monotonic() - started,
                "rows": rows,
            }
            report["candidates"][candidate] = candidate_report
            output.write_text(json.dumps(report, indent=2))
            print(json.dumps({
                "candidate": candidate,
                **{
                    key: value
                    for key, value in candidate_report.items()
                    if key not in ("rows", "by_task")
                },
            }), flush=True)
    finally:
        sim.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=("baseline", "45", "70", "100"),
    )
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--seed", type=int, default=951_000)
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--action-steps", type=int, default=16)
    parser.add_argument(
        "--device",
        choices=("cpu", "mps", "cuda"),
        default="mps",
    )
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    args = parser.parse_args()
    evaluate(
        args.run,
        args.checkpoint,
        args.collection,
        args.output,
        list(args.candidates),
        args.repeats,
        args.seed,
        args.horizon,
        args.action_steps,
        args.device,
        args.model_root,
    )


if __name__ == "__main__":
    main()
