"""Paired, problem-clustered analysis of isolated jerk qualification experiments."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def keyed(rows):
    seen = defaultdict(int)
    result = {}
    for row in rows:
        source = row["source"]
        repeat = seen[source]
        seen[source] += 1
        key = source, row.get("noise_seed", repeat)
        if key in result:
            raise ValueError(f"Duplicate probe source/noise pair: {key}")
        result[key] = row
    return result


def analyze(data):
    first, last = data["probes"][0], data["probes"][-1]
    original = keyed(first["trajectories"])
    current = keyed(last["trajectories"])
    horizon = data["run_config"]["horizon"]
    pairs = [(old, current[key]) for key, old in original.items() if key in current]
    complete = [(old, new) for old, new in pairs if old["steps"] == new["steps"] == horizon]
    by_problem = defaultdict(list)
    for old, new in complete:
        by_problem[old["source"]].append((old["rms_jerk"], new["rms_jerk"]))
    problems = []
    for source, values in by_problem.items():
        array = np.array(values)
        problems.append({
            "source": source, "pairs": len(values),
            "before": float(array[:, 0].mean()), "after": float(array[:, 1].mean()),
            "fractional_change": float(array[:, 1].mean() / array[:, 0].mean() - 1),
        })
    interval = None
    balanced_change = None
    if problems:
        before = np.array([row["before"] for row in problems])
        after = np.array([row["after"] for row in problems])
        balanced_change = float(after.mean() / before.mean() - 1)
        indices = np.random.default_rng(42).integers(len(problems), size=(10000, len(problems)))
        changes = after[indices].mean(axis=1) / before[indices].mean(axis=1) - 1
        interval = np.quantile(changes, [.025, .975]).tolist()
    common = np.array([(old["rms_jerk"], new["rms_jerk"]) for old, new in complete])
    return {
        "steps": len(data["steps"]), "fixed_problem_count": len(data["fixed_problems"]),
        "paired_complete_rollouts": len(complete),
        "pairing": "source+noise_seed" if all("noise_seed" in row for row in first["trajectories"] + last["trajectories"]) else "source+repeat order (legacy logs)",
        "pooled_paired_complete_fractional_change": float(common[:, 1].mean() / common[:, 0].mean() - 1) if complete else None,
        "problem_balanced_paired_complete_fractional_change": balanced_change,
        "problem_cluster_bootstrap_95pct_fractional_change": interval,
        "problems_improved": sum(row["fractional_change"] < 0 for row in problems),
        "problems_with_complete_pairs": len(problems),
        "before_success": first["success_rate"], "after_success": last["success_rate"],
        "before_full_horizon_rate": first["full_horizon_rate"], "after_full_horizon_rate": last["full_horizon_rate"],
        "by_problem": problems,
        "limitations": [
            "Probes share the eight training-source problems, but use independent fixed noise seeds.",
            "Confidence interval resamples problems, not individual frames; eight problems remain a small sample.",
            "Common-complete comparisons exclude invalid/truncated pairs; full-horizon rates must also be compared.",
            f"Fixed {horizon}-action window at assistance {data['run_config']['assistance']}; not a full unassisted-task qualification.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(json.loads(args.experiment.read_text())), indent=2))
