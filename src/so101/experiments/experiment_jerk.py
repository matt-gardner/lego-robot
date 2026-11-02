"""Isolated, memory-bounded real-policy jerk qualification; never resumes the main run."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from so101.cloud_checkpoint import atomic_json
from so101.control_model.jerk_regularizer import action_jerk_gradient, gradient_vector, pathwise_backward
from so101.evaluate_entity_sim import EntityPolicy
from so101.jerk_reference import calibrate_margins, problem_margin
from so101.robot_sim import BeamSimulation
from so101.train_trajectory_rl import (
    TASKS, group_advantages, load_corpus,
    normalized_component_advantage, policy_optimizer_step, rollout,
    save_resume_state, serialized_trajectory, trainable_parameters, trajectory_backward,
)


def jerk_advantages(trajectories, representation: str, combined: bool, horizon: int):
    jerks = np.array([item.rms_jerk for item in trajectories], dtype=np.float32)
    if representation == "negative-rms":
        rewards = -jerks
    elif representation == "negative-square":
        rewards = -jerks ** 2
    elif representation == "negative-log":
        rewards = -np.log1p(jerks)
    else:
        raise ValueError(f"Unknown jerk representation: {representation}")
    incomplete = np.array([item.steps < horizon for item in trajectories])
    rewards[incomplete] = float(rewards.min()) - 10.
    advantages = normalized_component_advantage(rewards)
    components = {"jerk": advantages.copy()}
    if combined:
        _, task_components = group_advantages(trajectories, 1., .5, 0.)
        components.update(task_components)
        advantages = advantages + task_components["success"] + .5 * task_components["completion"]
    return np.clip(advantages, -3, 3), components


def summarize(rows, horizon):
    complete = [item["rms_jerk"] for item in rows if item["steps"] == horizon]
    result = {
        "rms_jerk": float(np.mean([item["rms_jerk"] for item in rows])),
        "full_horizon_rms_jerk": float(np.mean(complete)) if complete else None,
        "success_rate": float(np.mean([item["success"] for item in rows])),
        "full_horizon_rate": float(np.mean([item["steps"] == horizon for item in rows])),
        "mean_steps": float(np.mean([item["steps"] for item in rows])),
    }
    result["by_task"] = {
        task: {
            "rms_jerk": float(np.mean([item["rms_jerk"] for item in rows if item["task"] == task])),
            "success_rate": float(np.mean([item["success"] for item in rows if item["task"] == task])),
        } for task in TASKS if any(item["task"] == task for item in rows)
    }
    return result


def problem_batch(pool, rng):
    return [pool[task][int(index)] for task in TASKS
            for index in rng.choice(len(pool[task]), size=2, replace=False)]


def run(args):
    if args.steps < 1 or args.group_size < 2 or args.horizon < 16:
        raise ValueError("Positive steps, group size >=2 and horizon >=16 required")
    if not np.isfinite([args.jerk_scale, args.jerk_weight, args.task_weight]).all() or args.jerk_scale <= 0 or min(args.jerk_weight, args.task_weight) < 0:
        raise ValueError("Objective weights must be finite and nonnegative; jerk scale must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    reference = json.loads((args.reference / "trajectory_rl.json").read_text())
    split = json.loads((args.reference / "data_split.json").read_text())
    bounds = split["expert_bounds"]
    corpus = load_corpus(Path(reference["run_config"]["collection"]))
    source_map = {str(item.source): item for values in corpus.values() for item in values}
    training = {task: [source_map[source] for source in split["training"][task]
                       if not source_map[source].metadata.get("obstacles")] for task in TASKS}
    margins = None
    if args.method == "pathwise":
        margins = (json.loads(args.jerk_reference_margins.read_text()) if args.jerk_reference_margins
                   else calibrate_margins({task: [str(item.source) for item in values]
                                           for task, values in training.items()},
                                          args.horizon, args.assistance, 10))
        if (margins["horizon"] != args.horizon or margins["assistance"] != args.assistance
                or margins["fps"] != 10 or margins["units"] != "degrees/frame^3"
                or not margins.get("fixed_horizon", True)
                or margins.get("reference_scope") != "problem"):
            raise ValueError("Reference jerk margins do not match this trial's measurement protocol")
        if not args.combined and all(problem_margin(margins, source) is None
                                     for source in margins["problems"]):
            raise ValueError("No jerk objective: none of the problems have usable reference trajectories")
        atomic_json(args.output / "jerk_reference_margins.json", margins)
    rng = np.random.default_rng(args.seed)
    fixed = problem_batch(training, np.random.default_rng(args.problem_seed))
    atomic_json(args.output / "data_split.json", split)
    canonical_digest = hashlib.sha256((args.reference / "latest_state.pt").read_bytes()).hexdigest()
    initial = torch.load(args.init_state or args.reference / "latest_state.pt", map_location="cpu", weights_only=False)
    torch.set_num_threads(2)
    if args.device == "mps":
        torch.mps.set_per_process_memory_fraction(.65)
    controller = EntityPolicy(Path(reference["run_config"]["checkpoint"]), args.device, True)
    controller.policy.model.vlm_with_expert.lm_expert.float()
    parameters = trainable_parameters(controller)
    named = dict(controller.policy.named_parameters())
    with torch.no_grad():
        for name, value in initial["model"].items():
            named[name].copy_(value.to(named[name].device))
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=1e-6)
    training_noise_step_offset = 0
    if args.continue_optimizer:
        optimizer.load_state_dict(initial["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
        if args.varied and initial["log"]["run_config"].get("varied"):
            rng.bit_generator.state = initial["rng"]["sampling"]
            training_noise_step_offset = initial["log"].get("training_noise_step_offset", 0) + initial["step"]
    del initial
    sim = BeamSimulation()
    log = {
        "kind": "isolated-jerk-qualification", "run_config": {
            key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
        },
        "reference_checkpoint_sha256": canonical_digest,
        "training_noise_step_offset": training_noise_step_offset,
        "fixed_problems": [str(item.source) for item in fixed],
        "metric": "RMS third finite difference of executed bounded five-joint commands",
        "scope": "64-step pilot" if args.horizon == 64 else f"{args.horizon}-step pilot",
        "probe_scope": "same eight training-source problems, independent fixed noise seeds; NOT held-out scenes",
        "optimizer": "full AdamW, fresh momentum unless explicitly continued; no KL, SFT or backtracking",
        "jerk_gradient": ("score function with normalized jerk advantages" if args.method == "score"
                          else "squared excess RMS command jerk above the problem's own reference / fixed scale squared; problems without references disabled; observation-feedback gradients omitted"),
        "jerk_reference_problems": margins["problems"] if margins else None,
        "steps": [], "probes": [],
    }
    def sample(episode, seed, store):
        return rollout(sim, controller, episode, seed, args.horizon, 16, bounds[episode.task],
                       store, args.assistance, .1, fixed_horizon=True,
                       record_path_noise=store and args.method == "pathwise")
    def probe(step, repeats):
        rows = []
        for index, episode in enumerate(fixed):
            for repeat in range(repeats):
                noise_seed = args.seed + 20_000_000 + index * 100 + repeat
                item = sample(episode, noise_seed, False)
                rows.append({"task": episode.task, "source": str(episode.source),
                             "noise_seed": noise_seed, **serialized_trajectory(item)})
        row = {"step": step, "repeats_per_problem": repeats, **summarize(rows, args.horizon), "trajectories": rows}
        log["probes"].append(row)
        atomic_json(args.output / "experiment.json", log)
        print(json.dumps({"event": "probe", **{k: v for k, v in row.items() if k != "trajectories"}}), flush=True)
        return row
    try:
        probe(0, args.probe_repeats)
        for step in range(1, args.steps + 1):
            started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            task_gradient = (torch.zeros_like(gradient_vector(parameters))
                             if args.method == "pathwise" and args.combined else None)
            problems = problem_batch(training, rng) if args.varied else fixed
            groups = []
            for index, episode in enumerate(problems):
                margin = problem_margin(margins, episode.source) if margins is not None else None
                trajectories = [
                    sample(episode, args.seed + index * 100 + repeat +
                           ((step + training_noise_step_offset) * 100_000
                            if args.fresh_training_noise else 0), True)
                    for repeat in range(args.group_size)
                ]
                advantages, components = jerk_advantages(trajectories, args.representation, args.combined, args.horizon)
                if args.method == "pathwise":
                    if args.combined:
                        _, components = group_advantages(trajectories, 1., .5, 0.)
                        advantages = np.clip(components["success"] + .5 * components["completion"], -3, 3)
                    else:
                        advantages = np.zeros(len(trajectories), dtype=np.float32)
                        components = {}
                denominator = float(np.abs(advantages).sum() + 1e-6)
                jerk_costs = []
                for item, advantage in zip(trajectories, advantages, strict=True):
                    if task_gradient is not None:
                        previous_gradient = gradient_vector(parameters)
                    trajectory_backward(controller, item, args.task_weight * float(advantage) / denominator)
                    if task_gradient is not None:
                        task_gradient += gradient_vector(parameters) - previous_gradient
                    cost = None
                    if args.method == "pathwise" and item.steps == args.horizon:
                        cost = pathwise_backward(controller, item,
                                          torch.as_tensor(sim.encode(sim.limits[:, 0]), device=args.device),
                                          torch.as_tensor(sim.encode(sim.limits[:, 1]), device=args.device),
                                          args.jerk_weight / args.group_size, args.jerk_scale, margin)
                    jerk_costs.append(cost)
                    item.decisions.clear()
                groups.append({
                    "task": episode.task, "source": str(episode.source),
                    "advantages": advantages.tolist(),
                    "advantage_components": {k: v.tolist() for k, v in components.items()},
                    "jerk_margin": margin, "jerk_losses": jerk_costs,
                    "jerk_reference_scope": "problem",
                    "jerk_reference_source": str(episode.source),
                    "jerk_regularizer_enabled": args.method == "pathwise" and margin is not None,
                    "trajectories": [serialized_trajectory(item) for item in trajectories],
                })
                del trajectories, item
                print(json.dumps({"event": "group", "step": step, "problem": index + 1,
                                  "task": episode.task, "jerk": np.mean([t["rms_jerk"] for t in groups[-1]["trajectories"]])}), flush=True)
            component_metrics = {}
            if task_gradient is not None:
                jerk_gradient = gradient_vector(parameters) - task_gradient
                task_norm, jerk_norm = float(task_gradient.norm()), float(jerk_gradient.norm())
                component_metrics = {
                    "weighted_task_gradient_norm": task_norm / 8,
                    "weighted_jerk_gradient_norm": jerk_norm / 8,
                    "task_jerk_gradient_cosine": (
                        float(torch.dot(task_gradient, jerk_gradient)) / (task_norm * jerk_norm)
                        if task_norm and jerk_norm else None
                    ),
                    "gradient_component_scope": "weighted, group-averaged, before shared clipping and Adam preconditioning",
                }
            metrics = policy_optimizer_step(optimizer, parameters, 8, 1.)
            rows = [{"task": g["task"], **item} for g in groups for item in g["trajectories"]]
            row = {"step": step, "groups": groups, "seconds": time.monotonic() - started,
                   **summarize(rows, args.horizon), **metrics, **component_metrics}
            log["steps"].append(row)
            atomic_json(args.output / "experiment.json", log)
            save_resume_state(args.output, controller, optimizer, rng, log)
            print(json.dumps({"event": "step", **{k: v for k, v in row.items() if k != "groups"}}), flush=True)
            probe(step, args.probe_repeats if step == args.steps else 1)
            if (args.output / "STOP").exists():
                print(json.dumps({"event": "requested_stop", "step": step}), flush=True)
                break
        assert hashlib.sha256((args.reference / "latest_state.pt").read_bytes()).hexdigest() == canonical_digest
        before, after = log["probes"][0], log["probes"][-1]
        originals = {(item["source"], item["noise_seed"]): item for item in before["trajectories"]}
        pairs = [(old["rms_jerk"], new["rms_jerk"])
                 for new in after["trajectories"]
                 for old in [originals[(new["source"], new["noise_seed"])]]
                 if old["steps"] == new["steps"] == args.horizon]
        paired = np.array(pairs, dtype=np.float64)
        result = {
            "completed_steps": log["steps"][-1]["step"], "before": before["rms_jerk"],
            "after": after["rms_jerk"], "fractional_change": after["rms_jerk"] / before["rms_jerk"] - 1,
            "before_success": before["success_rate"], "after_success": after["success_rate"],
            "before_full_horizon": before["full_horizon_rate"], "after_full_horizon": after["full_horizon_rate"],
            "paired_full_horizon_count": len(pairs),
            "paired_full_horizon_fractional_change": float(paired[:, 1].mean() / paired[:, 0].mean() - 1) if pairs else None,
        }
        atomic_json(args.output / "result.json", result)
        print(json.dumps({"event": "result", **result}), flush=True)
    finally:
        sim.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("outputs/entity_smolvla_trajectory_rl_all_assisted_lr1e4"))
    parser.add_argument("--init-state", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--assistance", type=float, default=.8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--representation", choices=("negative-rms", "negative-square", "negative-log"), default="negative-rms")
    parser.add_argument("--method", choices=("score", "pathwise"), default="score")
    parser.add_argument("--jerk-scale", type=float, default=6.)
    parser.add_argument("--jerk-reference-margins", type=Path)
    parser.add_argument("--jerk-weight", type=float, default=1.)
    parser.add_argument("--task-weight", type=float, default=1.)
    parser.add_argument("--combined", action="store_true")
    parser.add_argument("--varied", action="store_true")
    parser.add_argument("--continue-optimizer", action="store_true")
    parser.add_argument("--fresh-training-noise", action="store_true")
    parser.add_argument("--probe-repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=73000)
    parser.add_argument("--problem-seed", type=int, default=73100)
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    run(parser.parse_args())
