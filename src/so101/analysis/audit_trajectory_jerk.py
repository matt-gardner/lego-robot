"""Audit saved jerk metrics and actual advantage contributions without loading a model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def audit(log: dict, first_step: int = 215) -> dict:
    sampled = [group for step in log.get("steps", []) if step["step"] >= first_step
               for group in step.get("groups", [])]
    groups = [group for group in sampled if not group.get("discarded")]
    failed = [item for group in sampled for item in group["trajectories"] if not item["success"]]
    successful = [item for group in sampled for item in group["trajectories"] if item["success"]]
    solved = [group for group in groups if group.get("sampling_bucket") == "solved"]
    rough_positive = 0
    no_signal = 0
    for group in groups:
        component = group.get("advantage_components", {}).get("smoothness", [])
        no_signal += not any(abs(value) > 1e-6 for value in component)
        rough_positive += sum(
            item["success"] and smoothness < 0 and advantage > 0
            for item, smoothness, advantage in zip(
                group["trajectories"], component, group.get("advantages", []), strict=True)
        )
    validation = [{key: value for key, value in item.items()
                   if key in ("step", "success_rate", "rms_jerk", "by_task", "sampler", "denoising_noise_scale")}
                  for item in log.get("validation", []) if item["step"] >= 180]
    return {
        "first_step": first_step, "last_step": log["steps"][-1]["step"],
        "accepted_groups": len(groups), "solved_groups": len(solved),
        "discarded_groups": len(sampled) - len(groups),
        "failed_rollouts_without_jerk_reward": len(failed),
        "mean_failed_jerk": mean(item["rms_jerk"] for item in failed) if failed else None,
        "mean_successful_jerk": mean(item["rms_jerk"] for item in successful) if successful else None,
        "solved_fraction": len(solved) / len(groups) if groups else None,
        "mean_solved_assistance": mean(group["start_proximity"] for group in solved) if solved else None,
        "groups_without_smoothness_signal": no_signal,
        "successful_rollouts_with_below_group_mean_smoothness_but_positive_combined_advantage": rough_positive,
        "validation": validation,
        "limitations": [
            "Commanded five-joint third finite differences, degrees/frame^3, policy suffix only.",
            "Not realized mechanical jerk; prefix-to-policy boundary is excluded.",
            "Changing training assistance confounds pooled metrics; frozen unassisted evaluation also rises.",
            "Absolute inverse-jerk scores are still independently standardized, combined and L1-normalized per group.",
            "Failures receive no jerk score; plotted jerk includes failures and discarded groups.",
            "This observational audit cannot isolate optimizer versus stochastic-sampler effects.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--first-step", type=int, default=215)
    args = parser.parse_args()
    print(json.dumps(audit(json.loads(args.log.read_text()), args.first_step), indent=2))
