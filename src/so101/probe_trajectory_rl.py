"""Serial difficulty probes against immutable snapshots of the training policy."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path

import torch

from so101.cloud_checkpoint import atomic_json
from so101.calibrate_trajectory_rl import policy_digest
from so101.evaluate_entity_sim import EntityPolicy
from so101.rl_pile_adapter import attach_native_references, load_native_pool, load_reference_rows
from so101.rl_renderer import create_simulation
from so101.simulation_corpus import load_corpus
from so101.train_trajectory_rl import rollout, rollout_group, serialized_trajectory, trainable_parameters


def success_interval(successes, total):
    """Wilson 95% interval, including all-pass and all-fail samples."""
    z = 1.959963984540054
    rate = successes / total
    denominator = 1 + z * z / total
    centre = (rate + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return [max(0., centre - radius), min(1., centre + radius)]


def binary_assistance_step(state, assistance, successes, total, max_rounds):
    state["rounds"] += 1
    rate = successes / total
    if .25 <= rate <= .75:
        return None, "measured-mixed-outcomes"
    if rate < .25:
        state["lower"] = max(state["lower"], assistance)
        if assistance == .95:
            return None, "no-mixed-level-at-maximum-assistance"
    else:
        state["upper"] = min(state["upper"], assistance)
    if state["rounds"] >= max_rounds or state["upper"] - state["lower"] < .01:
        return None, "unresolved-within-search-budget"
    return (state["lower"] + state["upper"]) / 2, None


def probe(args):
    if args.native_references is not None and args.native_pool is None:
        raise ValueError("Native reference qualification requires its exact native pool")
    if args.calibrate_assistance and not (args.initial_only and args.native_only):
        raise ValueError("Assistance calibration requires a fixed initial policy and native-only tasks")
    plan = json.loads(args.plan.read_text())
    if not isinstance(plan, list) or not plan:
        raise ValueError("Probe plan must contain problems to measure")
    identities = [(row["task"], row["source"], row["assistance"]) for row in plan]
    if len(identities) != len(set(identities)):
        raise ValueError("Probe plan duplicates a problem/assistance measurement")
    if args.calibrate_assistance:
        seen = set()
        for row in plan:
            if row["source"] not in seen and row["assistance"] != 0:
                raise ValueError("Each calibrated problem must first be measured without assistance")
            seen.add(row["source"])
    split = json.loads(args.split.read_text())
    corpus = load_corpus(args.collection)
    available = {str(item.source): item for rows in corpus.values() for item in rows}
    allowed = {source for rows in split["training"].values() for source in rows}
    if args.native_pool is not None:
        validation = [available[source] for source in split["validation"]]
        excluded = {
            item.metadata.get("scene_family_id", item.metadata.get("initial", {}).get(
                "scene_family_id", hashlib.sha256(json.dumps([
                    item.metadata.get("initial", {}).get("seed"),
                    item.metadata.get("initial", {}).get("scene_version", 1),
                ]).encode()).hexdigest(),
            ))
            for item in validation
        }
        native = load_native_pool(args.native_pool, excluded)
        if args.native_references is not None:
            attach_native_references(
                native, args.native_references,
                pool_sha256=hashlib.sha256(args.native_pool.read_bytes()).hexdigest(),
            )
        for rows in native.values():
            for item in rows:
                available[str(item.source)] = item
                allowed.add(str(item.source))
    for row in plan:
        if row["source"] not in allowed or row["source"] in split["validation"]:
            raise ValueError(f"Probe source outside TRAIN: {row['source']}")
        if not math.isfinite(row["assistance"]) or not 0 <= row["assistance"] <= .95:
            raise ValueError("Invalid probe assistance")
        episode = available[row["source"]]
        if episode.task != row["task"]:
            raise ValueError("Probe task does not match its bound source")
        if args.native_only and "native_entry" not in episode.metadata:
            raise ValueError("Native-only calibration cannot consume legacy problem data")
        if ("native_entry" in episode.metadata and row["assistance"] != 0
                and "native_reference" not in episode.metadata):
            raise ValueError("Native assistance requires a replay-qualified reference")
    if args.repeats < 16:
        raise ValueError("Difficulty probes require at least 16 repeated rollouts")
    if args.batch_rollouts < 1:
        raise ValueError("Inference batch must contain at least one rollout")
    if args.horizon <= 0:
        raise ValueError("Probe horizon must be positive")
    if args.binary_search and (not args.initial_only or not args.native_only or args.calibrate_assistance):
        raise ValueError("Binary search requires fixed initial/native policy and no explicit-plan calibration")
    if args.binary_search and args.search_rounds < 1:
        raise ValueError("Binary search requires at least one measured round")
    if args.output.exists() and not args.resume:
        raise FileExistsError("Probe output already exists; reconcile before restarting")
    if args.resume and (not args.initial_only or not args.output.exists()):
        raise ValueError("Resume requires an existing initial-policy measurement file")
    controller = EntityPolicy(args.checkpoint, "cuda", True)
    trainable_parameters(controller, "all")
    controller.policy.eval()
    before_policy = policy_digest(controller.policy) if args.initial_only else None
    result = {"plan": plan, "repeats": args.repeats, "groups": [], "complete": False,
              "input_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                               for name, path in (("plan", args.plan), ("split", args.split),
                                                  ("native_pool", args.native_pool),
                                                  ("native_references", args.native_references))
                               if path is not None},
              "optimizer": None, "parallel_rollouts": args.batch_rollouts,
              "horizon": args.horizon, "seed": args.seed,
              "measurement_scope": "short-horizon-screening" if args.horizon < 300 else "full-horizon-probe",
              "renderer": "gpu-pathtrace", "samples_per_pixel": 16, "action_steps": 16}
    result["assistance_decisions"] = {}
    result["unmeasured_levels"] = []
    result["policy_sha256_before"] = before_policy
    if args.resume:
        previous = json.loads(args.output.read_text())
        if (previous["complete"] or previous["optimizer"] is not None
                or previous["policy_sha256_before"] != before_policy
                or any(previous[key] != result[key] for key in
                       ("input_sha256", "repeats", "parallel_rollouts", "horizon", "seed"))):
            raise ValueError("Resume changes the measured population, policy, horizon or sampling protocol")
        result = previous
        result["assistance_decisions"] = {}
        result["unmeasured_levels"] = []
    result["search_strategy"] = "binary-search" if args.binary_search else "explicit-plan"
    atomic_json(args.output, result)
    sources = list(dict.fromkeys(row["source"] for row in plan))
    if args.binary_search:
        queue = [{**next(row for row in plan if row["source"] == source), "assistance": .95}
                 for source in sources]
        result["binary_search_bounds"] = {
            source: {"lower": 0., "upper": .95, "rounds": 0} for source in sources
        }
    else:
        queue = list(plan)
    sims = []
    try:
        for _ in range(args.batch_rollouts):
            sims.append(create_simulation(args.model_root, backend="gpu-pathtrace", samples=16))
        loaded_step = None
        for row in queue:
            if args.calibrate_assistance and row["source"] in result["assistance_decisions"]:
                result["unmeasured_levels"].append(row)
                continue
            state_path = args.training_output / "latest_state.pt"
            policy_step = 0
            if not args.initial_only and state_path.exists():
                state = torch.load(state_path, map_location="cpu", weights_only=False, mmap=True)
                expected = dict(controller.policy.named_parameters())
                if set(state["model"]) != set(expected):
                    raise ValueError("Probe checkpoint is not a full-model parameter snapshot")
                controller.policy.load_state_dict(state["model"], strict=False)
                policy_step = state["step"]
                loaded_step = policy_step
                del state
                gc.collect()
            elif not args.initial_only and loaded_step is not None:
                raise FileNotFoundError("Training checkpoint disappeared after a probe loaded it")
            matches = [group for group in result["groups"]
                       if all(group[key] == row[key] for key in ("task", "source", "assistance"))]
            if len(matches) > 1:
                raise ValueError("Resumed measurements duplicate a problem/assistance group")
            group = matches[0] if matches else {
                **row, "policy_step": policy_step, "initial_checkpoint": str(args.checkpoint), "rollouts": [],
            }
            observations = group["rollouts"]
            if (group["policy_step"] != policy_step or len(observations) > args.repeats
                    or [item["seed"] for item in observations] != [
                        args.seed + sources.index(row["source"]) * 10000 + repeat
                        for repeat in range(len(observations))
                    ]):
                raise ValueError("Resumed group changes policy iteration or paired rollout seeds")
            if not matches:
                result["groups"].append(group)
            episode = available[row["source"]]
            bounds = dict(split["expert_bounds"][row["task"]])
            if "native_reference" in episode.metadata:
                bounds["optimal_steps"] = len(load_reference_rows(episode))
            for start in range(len(observations), args.repeats, args.batch_rollouts):
                seeds = [args.seed + sources.index(row["source"]) * 10000 + repeat
                         for repeat in range(start, min(start + args.batch_rollouts, args.repeats))]
                trajectories = (
                    [rollout(sims[0], controller, episode, seeds[0], args.horizon, 16,
                             bounds, False, row["assistance"], .1)]
                    if args.batch_rollouts == 1 else
                    rollout_group(sims[:len(seeds)], controller, episode, seeds, args.horizon, 16,
                                  bounds, row["assistance"], .1, False, store_decisions=False)
                )
                observations.extend({"seed": seed, **serialized_trajectory(trajectory)}
                                    for seed, trajectory in zip(seeds, trajectories, strict=True))
                successes = sum(bool(item["success"]) for item in observations)
                group.update(successes=successes, total=len(observations),
                             success_interval_95=success_interval(successes, len(observations)))
                atomic_json(args.output, result)
                print(json.dumps({"event": "difficulty_probe", "source": row["source"],
                                  "policy_step": policy_step, "assistance": row["assistance"],
                                  "successes": successes, "total": len(observations),
                                  "success_interval_95": group["success_interval_95"]}), flush=True)
            if args.binary_search:
                level, status = binary_assistance_step(
                    result["binary_search_bounds"][row["source"]], row["assistance"],
                    group["successes"], group["total"], args.search_rounds,
                )
                if level is not None:
                    queue.append({**row, "assistance": level})
                else:
                    result["assistance_decisions"][row["source"]] = {
                        "status": status, "assistance": row["assistance"],
                        "successes": group["successes"], "total": group["total"],
                        "success_interval_95": group["success_interval_95"],
                    }
                atomic_json(args.output, result)
            if args.calibrate_assistance:
                status = (
                    "too-easy-without-assistance" if row["assistance"] == 0
                    and group["successes"] == args.repeats else
                    "measured-mixed-outcomes" if .25 <= group["successes"] / args.repeats <= .75
                    else None
                )
                if status is not None:
                    result["assistance_decisions"][row["source"]] = {
                        "status": status, "assistance": row["assistance"],
                        "successes": group["successes"], "total": group["total"],
                        "success_interval_95": group["success_interval_95"],
                    }
    finally:
        for sim in sims:
            native = getattr(sim, "_native_simulation", None)
            if native is not None:
                native.close()
            sim.close()
    result["complete"] = True
    if args.binary_search:
        measured = {(row["source"], row["assistance"]) for row in result["groups"]}
        result["unmeasured_levels"] = [row for row in plan if (row["source"], row["assistance"]) not in measured]
    if args.calibrate_assistance:
        for source in sources:
            result["assistance_decisions"].setdefault(
                source, {"status": "no-mixed-level-among-measured-levels"},
            )
    if args.initial_only:
        result["policy_sha256_after"] = policy_digest(controller.policy)
        if result["policy_sha256_after"] != before_policy:
            raise RuntimeError("Initial-policy calibration unexpectedly changed policy parameters")
    atomic_json(args.output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "collection", "split", "plan", "training-output", "output", "model-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--native-pool", type=Path)
    parser.add_argument("--native-references", type=Path)
    parser.add_argument("--native-only", action="store_true")
    parser.add_argument("--initial-only", action="store_true",
                        help="Always use the initial policy; ignore rolling training checkpoints")
    parser.add_argument("--calibrate-assistance", action="store_true",
                        help="Stop further levels for each problem after measured mixed outcomes")
    parser.add_argument("--repeats", type=int, default=16)
    parser.add_argument("--batch-rollouts", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--seed", type=int, default=2551000)
    parser.add_argument("--horizon", type=int, default=300,
                        help="Maximum policy steps; shorter runs are screening, not full-horizon qualification")
    parser.add_argument("--binary-search", action="store_true",
                        help="Measure the maximum assistance first, then bisect toward 25--75%% success")
    parser.add_argument("--search-rounds", type=int, default=6)
    parser.add_argument("--resume", action="store_true",
                        help="Continue matching initial-policy measurements without repeating completed samples")
    probe(parser.parse_args())


if __name__ == "__main__":
    main()
