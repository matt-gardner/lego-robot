"""Measure useful four-rollout groups without constructing an optimizer."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch

from so101.cloud_checkpoint import atomic_json
from so101.evaluate_entity_sim import EntityPolicy
from so101.rl_pile_adapter import load_native_pool
from so101.rl_renderer import create_simulation
from so101.simulation_corpus import TASKS, load_corpus
from so101.train_trajectory_rl import rollout_group, serialized_trajectory, trainable_parameters


def select_problems(episodes, count, rng):
    """Prefer separate scene families over extra targets from the same scene."""
    families = defaultdict(list)
    for episode in episodes:
        family = episode.metadata.get(
            "scene_family_id", json.dumps(episode.metadata.get("initial", {}).get("seed")),
        )
        families[family].append(episode)
    keys = sorted(families)
    rng.shuffle(keys)
    selected = [families[key][int(rng.integers(len(families[key])))] for key in keys[:count]]
    remaining = [episode for episode in episodes if episode not in selected]
    rng.shuffle(remaining)
    return selected + remaining[:max(0, count - len(selected))]


def summarize(rows):
    cells = {}
    for row in rows:
        key = (row["task"], row["bucket"], row["assistance"])
        cell = cells.setdefault(key, {
            "task": key[0], "bucket": key[1], "assistance": key[2],
            "groups": 0, "rollouts": 0, "successes": 0, "mixed_groups": 0, "invalid_actions": 0,
            "sources": [], "scene_families": [],
        })
        cell["groups"] += 1
        cell["rollouts"] += 4
        cell["successes"] += row["successes"]
        cell["mixed_groups"] += int(0 < row["successes"] < 4)
        cell["invalid_actions"] += row["invalid_actions"]
        cell["sources"].append(row["source"])
        cell["scene_families"].append(row["scene_family"])
    for cell in cells.values():
        cell["success_rate"] = cell["successes"] / cell["rollouts"]
        cell["mixed_group_fraction"] = cell["mixed_groups"] / cell["groups"]
        cell["distinct_scene_families"] = len(set(cell.pop("scene_families")))
    targets = {}
    for task in TASKS:
        candidates = sorted(
            (cell for cell in cells.values() if cell["task"] == task
             and cell["mixed_groups"] >= max(2, (cell["groups"] + 1) // 2)),
            key=lambda cell: (-cell["mixed_group_fraction"], abs(.5 - cell["success_rate"]),
                              cell["invalid_actions"], cell["assistance"], cell["bucket"]),
        )
        targets[task] = {
            "candidates": candidates,
            "status": "pilot-targets-found" if candidates else "no-repeated-mixed-groups-yet",
        }
    return {"cells": list(cells.values()), "targets": targets}


def finished_cell(cells, task, bucket, expected):
    complete = [cell for cell in cells if cell["task"] == task and cell["bucket"] == bucket
                and cell["groups"] == expected]
    return any(cell["mixed_groups"] >= max(2, (expected + 1) // 2)
               or cell["assistance"] == 0 and cell["successes"] == cell["rollouts"]
               for cell in complete)


def save_progress(output, result):
    atomic_json(output, result)
    cells = summarize(result["groups"])["cells"]
    populations = defaultdict(set)
    for row in result["plan"]:
        populations[(row["task"], row["bucket"])].add(row["source"])
    lines = [
        "# SFT7296 inference-only assistance calibration", "",
        "Four rollouts/problem; five distinct TRAIN problems per bucket/task where available.",
        "Levels are tested adaptively: 0, 0.5, 0.95, then 0.25/0.75 if needed.",
        "A target needs mixed outcomes on a majority of the sampled problems (at least two).",
        "No optimizer or training updates. Native piles support assistance 0 only.",
        "task_difficulty_v1 has only four TRAIN problems/task; no duplicate is counted as a fifth.", "",
        "| Bucket | Grasp | View | Push | Place |",
        "|---|---|---|---|---|",
    ]
    for bucket in sorted({key[1] for key in populations}):
        labels = []
        for task in TASKS:
            expected = len(populations.get((task, bucket), []))
            matching = [cell for cell in cells if cell["task"] == task and cell["bucket"] == bucket]
            complete = [cell for cell in matching if cell["groups"] == expected]
            useful = [cell for cell in complete if cell["mixed_groups"] >= max(2, (expected + 1) // 2)]
            zero = next((cell for cell in complete if cell["assistance"] == 0), None)
            if useful:
                chosen = max(useful, key=lambda cell: (cell["mixed_group_fraction"], -cell["assistance"]))
                label = f'**{chosen["assistance"]:g}** ({chosen["mixed_groups"]}/{expected} mixed)'
            elif zero is not None and zero["successes"] == zero["rollouts"]:
                label = f"Too easy at 0 ({expected}/{expected} all-pass)"
            elif matching:
                if result["complete"] or bucket.startswith("native:") and complete:
                    chosen = max(matching, key=lambda cell: cell["mixed_group_fraction"])
                    label = f'No mixed target (best {chosen["assistance"]:g}: {chosen["mixed_groups"]}/{expected})'
                else:
                    chosen = matching[-1]
                    label = (f'Testing {chosen["assistance"]:g}: {chosen["mixed_groups"]}/{chosen["groups"]} mixed'
                             f' ({chosen["groups"]}/{expected} problems)')
            else:
                label = "Pending" if expected else "-"
            labels.append(label)
        lines.append("| " + bucket + " | " + " | ".join(labels) + " |")
    lines.extend(["", f'Completed inference groups: {len(result["groups"])}.',
                  "Status: " + ("complete" if result["complete"] else "running; levels are preliminary") + "."])
    output.with_suffix(".md").write_text("\n".join(lines) + "\n")


def policy_digest(policy):
    digest = hashlib.sha256()
    for name, parameter in policy.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def calibrate(args):
    if (args.groups_per_cell < 2 or 0. not in args.assistance_levels
            or len(set(args.assistance_levels)) != len(args.assistance_levels) or any(
        not np.isfinite(value) or not 0 <= value <= .95 for value in args.assistance_levels
    )):
        raise ValueError("Calibration requires at least two problems/cell and assistance within [0, .95]")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    split_bytes, native_bytes = args.split.read_bytes(), args.native_pool.read_bytes()
    split = json.loads(split_bytes)
    corpus = load_corpus(args.collection)
    by_source = {str(item.source): item for episodes in corpus.values() for item in episodes}
    validation = [by_source[source] for source in split["validation"]]
    excluded = {
        item.metadata.get("scene_family_id", item.metadata.get("initial", {}).get(
            "scene_family_id", hashlib.sha256(json.dumps([
                item.metadata.get("initial", {}).get("seed"),
                item.metadata.get("initial", {}).get("scene_version", 1),
            ]).encode()).hexdigest(),
        )) for item in validation
    }
    buckets = defaultdict(list)
    for task, sources in split["training"].items():
        for source in sources:
            if source in split["validation"]:
                raise ValueError("Calibration training population crosses frozen validation")
            buckets[(task, "corpus:" + Path(source).parent.name)].append(by_source[source])
    native = load_native_pool(args.native_pool, excluded)
    for task, episodes in native.items():
        for episode in episodes:
            count = episode.metadata["native_entry"]["tags"]["piece_count"]
            buckets[(task, f"native:{count}-pieces")].append(episode)
            by_source[str(episode.source)] = episode
    plan = []
    for (task, bucket), episodes in sorted(buckets.items()):
        selected = select_problems(episodes, args.groups_per_cell, rng)
        for problem_index, episode in enumerate(selected):
            seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(4)]
            for assistance in ([0.] if "native_entry" in episode.metadata else args.assistance_levels):
                plan.append({
                    "task": task, "bucket": bucket, "source": str(episode.source),
                    "assistance": assistance, "problem_index": problem_index, "seeds": seeds,
                    "scene_family": episode.metadata.get(
                        "scene_family_id", json.dumps(episode.metadata.get("initial", {}).get("seed")),
                    ),
                })
    level_order = {level: index for index, level in enumerate(args.assistance_levels)}
    plan.sort(key=lambda row: (row["bucket"], TASKS.index(row["task"]),
                              level_order[row["assistance"]], row["problem_index"]))
    controller = EntityPolicy(args.checkpoint, "cuda", True)
    parameters = trainable_parameters(controller, "all")
    if sum(value.numel() for value in parameters) != 453656985:
        raise ValueError("Calibration must reconstruct the production FP32 full policy")
    controller.policy.eval()
    before = policy_digest(controller.policy)
    protocol = {
        "checkpoint": str(args.checkpoint), "policy_sha256": before, "source": args.source_commit,
        "split_sha256": hashlib.sha256(split_bytes).hexdigest(),
        "native_pool_sha256": hashlib.sha256(native_bytes).hexdigest(),
        "groups_per_cell": args.groups_per_cell, "assistance_levels": args.assistance_levels,
        "seed": args.seed, "renderer": "gpu-pathtrace", "spp": 16, "noise": .1,
        "horizon": 300, "action_steps": 16, "rollouts_per_group": 4,
        "optimizer": None, "native_assistance_supported": False,
    }
    result = {"protocol": protocol, "plan": plan, "groups": [], "complete": False}
    if args.output.exists():
        result = json.loads(args.output.read_text())
        if result["protocol"] != protocol or result["plan"] != plan or result["complete"]:
            raise ValueError("Existing calibration differs or already completed; reconcile it")
    save_progress(args.output, result)
    print(json.dumps({"event": "calibration_plan", "groups": len(plan), "protocol": protocol}), flush=True)
    sims = []
    try:
        for _ in range(4):
            sims.append(create_simulation(args.model_root, backend="gpu-pathtrace", samples=16))
        completed = {(row["task"], row["bucket"], row["source"], row["assistance"])
                     for row in result["groups"]}
        populations = defaultdict(set)
        for row in plan:
            populations[(row["task"], row["bucket"])].add(row["source"])
        for row in plan:
            identity = (row["task"], row["bucket"], row["source"], row["assistance"])
            if identity in completed or finished_cell(
                summarize(result["groups"])["cells"], row["task"], row["bucket"],
                len(populations[(row["task"], row["bucket"])]),
            ):
                continue
            trajectories = rollout_group(
                sims, controller, by_source[row["source"]], row["seeds"], 300, 16,
                split["expert_bounds"][row["task"]], row["assistance"], .1, False,
            )
            metrics = [serialized_trajectory(item) for item in trajectories]
            observed = {
                **row, "successes": sum(bool(item.success) for item in trajectories),
                "invalid_actions": sum(item.get("invalid_action_error") is not None for item in metrics),
                "trajectories": metrics,
            }
            result["groups"].append(observed)
            result["summary"] = summarize(result["groups"])
            save_progress(args.output, result)
            print(json.dumps({"event": "calibration_group", "completed_groups": len(result["groups"]),
                              **{key: value for key, value in observed.items() if key != "trajectories"}}),
                  flush=True)
    finally:
        for sim in sims:
            native_sim = getattr(sim, "_native_simulation", None)
            if native_sim is not None:
                native_sim.close()
            sim.close()
    after = policy_digest(controller.policy)
    if after != before:
        raise ValueError("Calibration changed policy parameters despite having no optimizer")
    result.update(complete=True, final_policy_sha256=after)
    save_progress(args.output, result)
    print(json.dumps({"event": "calibration_complete", "output": str(args.output),
                      "summary": result["summary"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--native-pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--groups-per-cell", type=int, default=5)
    parser.add_argument("--assistance-levels", type=float, nargs="+", default=[0., .5, .95, .25, .75])
    parser.add_argument("--seed", type=int, default=1551000)
    calibrate(parser.parse_args())


if __name__ == "__main__":
    main()
