from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TASKS = ("grasp", "view", "push", "place")
COLORS = {
    "all": "#202020",
    "grasp": "#4472C4",
    "view": "#70AD47",
    "push": "#ED7D31",
    "place": "#A64D79",
}


def plot(training: Path, output: Path) -> None:
    data = json.loads(training.read_text())
    output.mkdir(parents=True, exist_ok=True)
    validation = data["validation"]
    if validation:
        figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        metrics = (
            ("success_rate", "Success rate", 100),
            (
                "completion_efficiency",
                "Completion efficiency (% of optimum)",
                100,
            ),
            ("rms_jerk", "RMS arm jerk (degrees/frame³)", 1),
        )
        for axis, (key, title, scale) in zip(
            axes, metrics, strict=True
        ):
            steps = [row["step"] for row in validation]
            axis.plot(
                steps,
                [scale * row[key] for row in validation],
                "-o",
                color=COLORS["all"],
                linewidth=2.5,
                label="all",
            )
            for task in TASKS:
                axis.plot(
                    steps,
                    [
                        scale * row["by_task"][task][key]
                        for row in validation
                    ],
                    "-o",
                    color=COLORS[task],
                    linewidth=1.5,
                    label=task,
                )
            axis.set_title(title)
            axis.set_xlabel("Optimizer step")
            axis.grid(alpha=.25)
            axis.set_xticks(steps)
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center", ncol=5)
        figure.suptitle(
            "Full-trajectory oracle-box RL validation",
            fontsize=15,
            y=.99,
        )
        figure.tight_layout(rect=(0, 0, 1, .9))
        figure.savefig(output / "top_metrics.png", dpi=180)
        plt.close(figure)

    steps = data["steps"]
    if not steps:
        return
    step_numbers = [row["step"] for row in steps]
    zero_rates = []
    success_means = []
    completion_means = []
    smoothness_means = []
    total_groups = zero_groups = 0
    for row in steps:
        groups = row["groups"]
        total_groups += len(groups)
        zero_groups += sum(group["discarded"] for group in groups)
        zero_rates.append(zero_groups / total_groups)
        trajectories = [
            trajectory
            for group in groups
            for trajectory in group.get("trajectories", [])
        ]
        success_means.append(float(np.mean([
            trajectory["success"] for trajectory in trajectories
        ])) if trajectories else 0.)
        successful = [
            trajectory for trajectory in trajectories
            if trajectory["success"]
        ]
        completion_means.append(float(np.mean([
            trajectory["completion_efficiency"]
            for trajectory in successful
        ])) if successful else 0.)
        smoothness_means.append(float(np.mean([
            trajectory["smoothness"] for trajectory in successful
        ])) if successful else 0.)
    figure, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    for axis, values, title in (
        (axes[0], zero_rates, "Cumulative zero-success group rate"),
        (axes[1], success_means, "Success reward component"),
        (axes[2], completion_means, "Successful completion component"),
        (axes[3], smoothness_means, "Successful smoothness component"),
    ):
        axis.plot(
            step_numbers,
            values,
            "-o",
            color="#5B9BD5",
            linewidth=2,
        )
        axis.set_title(title)
        axis.set_xlabel("Optimizer step")
        axis.grid(alpha=.25)
        axis.set_xticks(step_numbers)
    figure.tight_layout()
    figure.savefig(output / "reward_dynamics.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(args.training, args.output)


if __name__ == "__main__":
    main()
