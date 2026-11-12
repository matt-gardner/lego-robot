"""Measure paired saved-state GRASP variants without expert prefixes or updates."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

from so101.calibrate_trajectory_rl import policy_digest
from so101.cloud_checkpoint import atomic_json
from so101.evaluate_entity_sim import EntityPolicy
from so101.pile_task_runtime import load_problem
from so101.probe_trajectory_rl import success_interval
from so101.rl_renderer import create_simulation
from so101.simulation_corpus import CorpusEpisode
from so101.train_trajectory_rl import rollout, serialized_trajectory, trainable_parameters


def summarize(groups):
    cells = defaultdict(lambda: {"successes": 0, "total": 0, "parents": set()})
    for group in groups:
        for key in ((group["level"], None), (group["level"], group["piece_count"])):
            cell = cells[key]
            cell["successes"] += sum(bool(row["success"]) for row in group["rollouts"])
            cell["total"] += len(group["rollouts"])
            cell["parents"].add(group["parent_task_id"])
    return [
        {"assistance": level, "piece_count": size,
         "successes": row["successes"], "total": row["total"],
         "distinct_parents": len(row["parents"]),
         "success_rate": row["successes"] / row["total"],
         "success_interval_95": success_interval(row["successes"], row["total"])}
        for (level, size), row in cells.items() if row["total"]
    ]


def measurement_levels(available, requested):
    levels = available if requested is None else requested
    if not levels or len(set(levels)) != len(levels) or not set(levels) <= set(available):
        raise ValueError("Measured assistance levels must be a nonempty distinct subset of frozen levels")
    return sorted(levels)


def recovery_partition(scene_family, training_families):
    if not training_families:
        return "unpartitioned"
    return "recovery-training" if scene_family in training_families else "recovery-heldout"


def select_parent_families(episodes, requested):
    indexed = list(enumerate(episodes))
    if requested is None:
        return indexed
    available = {next(iter(family.values())).metadata["parent_task_id"]
                 for family in episodes}
    if not requested or len(set(requested)) != len(requested) or not set(requested) <= available:
        raise ValueError("Selected parents must be a nonempty distinct subset of frozen parents")
    return [(index, family) for index, family in indexed
            if next(iter(family.values())).metadata["parent_task_id"] in requested]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "checkpoint", "model-root", "bounds", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--action-steps", type=int, default=16)
    parser.add_argument("--full-state-checkpoint", action="store_true")
    parser.add_argument("--seed", type=int, default=17551000)
    parser.add_argument("--deadline", type=int, required=True)
    parser.add_argument("--levels", type=float, nargs="+")
    parser.add_argument("--expected-policy-sha256",
                        default="52cabd5b2d5df41b3825f323ffba3f56601725b0a25b24dc16cd923310b5f831")
    parser.add_argument("--training-scene-family", action="append", default=[])
    parser.add_argument("--parent-task-id", action="append")
    args = parser.parse_args()
    if args.horizon <= 0 or args.action_steps <= 0 or args.deadline <= time.time():
        raise ValueError("Positive horizon, action steps and future deadline required")
    if args.output.exists():
        raise FileExistsError("Reconcile existing rollout results before retrying")
    plan = json.loads(args.plan.read_bytes())
    if plan["kind"] != "paired-original-pose-grasp-rollouts-v1":
        raise ValueError("Unknown GRASP rollout protocol")
    levels = measurement_levels(plan["levels"], args.levels)
    for name, expected in plan["files_sha256"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Changed frozen input: {name}")
    episodes = []
    for entry in plan["entries"]:
        if entry["split"] != "train":
            raise ValueError("Rollout population must be TRAIN")
        scene = json.loads((Path(entry["root"]) / "problem.json").read_bytes())
        family = {}
        for task in scene["tasks"]:
            load_problem(entry, task["task_id"])
            assistance = task["assistance"]
            if assistance["kind"] != "original-pose-grasp-v1" or task["task_type"] != "grasp":
                raise ValueError("Wrong task or assistance semantics")
            level = assistance["value"]
            if level in family:
                raise ValueError("Duplicate assistance level")
            family[level] = CorpusEpisode(
                Path(entry["root"]) / "tasks" / task["task_id"],
                {"task": "grasp", "target": task["target_instance_id"],
                 "native_entry": entry, "native_task_id": task["task_id"],
                 "scene_family_id": entry["scene_family_id"],
                 "parent_task_id": task["parent_task_id"], "task_assistance": assistance,
                 "original_scene_family": scene["provenance"]["parent_entry"]["scene_family_id"],
                 "reference_available": False},
            )
        if sorted(family) != plan["levels"]:
            raise ValueError("Missing paired assistance levels")
        episodes.append(family)
    selected = select_parent_families(episodes, args.parent_task_id)
    controller = EntityPolicy(
        args.checkpoint, "cuda", True, full_state_checkpoint=args.full_state_checkpoint,
    )
    if args.action_steps > controller.policy.config.chunk_size:
        raise ValueError("Action steps exceed the policy's predicted chunk size")
    trainable_parameters(controller, "all")
    controller.policy.eval()
    before = policy_digest(controller.policy)
    if before != args.expected_policy_sha256:
        raise ValueError("Restored policy differs from its pinned expected policy digest")
    bounds = json.loads(args.bounds.read_bytes())["expert_bounds"]["grasp"]
    result = {
        "protocol": plan, "optimizer": None, "policy_sha256_before": before,
        "complete": False, "horizon": args.horizon, "seed": args.seed,
        "assistance_profile": "original-pose-grasp-v1",
        "expected_rollouts": len(selected) * len(levels),
        "selected_parent_task_ids": args.parent_task_id,
        "measured_assistance_levels": levels,
        "recovery_training_scene_families": args.training_scene_family,
        "partition_scope": "Recovery-heldout excludes recovery-SFT scenes only, not the original SFT population",
        "action_steps": args.action_steps, "denoising_noise_scale": .1, "renderer_samples": 16,
        "renderer": "gpu-pathtrace", "label_renderer": "native-id-segmentation-no-msaa-v1",
        "expert_prefix_assistance": 0, "groups": [], "cells": [],
        "cells_by_recovery_partition": {},
        "scope": "Paired TRAIN screening; Wilson intervals are descriptive, parents are correlated",
    }
    atomic_json(args.output, result)
    sim = create_simulation(args.model_root, backend="gpu-pathtrace", samples=16)
    try:
        for level in reversed(levels):
            for index, family in selected:
                if time.time() >= args.deadline:
                    result["stop_reason"] = "deadline"
                    return
                episode = family[level]
                trajectory = rollout(sim, controller, episode, args.seed + index,
                                     args.horizon, args.action_steps, bounds, False, 0., .1)
                result["groups"].append({
                    "level": level, "source": str(episode.source),
                    "parent_task_id": episode.metadata["parent_task_id"],
                    "original_scene_family": episode.metadata["original_scene_family"],
                    "recovery_partition": recovery_partition(
                        episode.metadata["original_scene_family"], args.training_scene_family,
                    ),
                    "piece_count": episode.metadata["native_entry"]["tags"]["piece_count"],
                    "rollouts": [{"seed": args.seed + index,
                                  **serialized_trajectory(trajectory)}],
                })
                result["cells"] = summarize(result["groups"])
                result["cells_by_recovery_partition"] = {
                    partition: summarize([
                        group for group in result["groups"]
                        if group["recovery_partition"] == partition
                    ])
                    for partition in {group["recovery_partition"] for group in result["groups"]}
                }
                atomic_json(args.output, result)
                print(json.dumps({"event": "grasp_rollout", "level": level,
                                  "completed": len(result["groups"]),
                                  "success": trajectory.success}), flush=True)
        result["complete"] = True
    finally:
        native = getattr(sim, "_native_simulation", None)
        if native is not None:
            native.close()
        sim.close()
        result["policy_sha256_after"] = policy_digest(controller.policy)
        atomic_json(args.output, result)
        if result["policy_sha256_after"] != before:
            raise ValueError("Inference changed policy parameters")


if __name__ == "__main__":
    main()
