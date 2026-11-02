from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def task_step_values(
    rows: list[dict],
    task: str,
    key: str,
) -> tuple[list[int], list[float]]:
    task_rows = [row for row in rows if row["task"] == task]
    steps = sorted(set(row["training_step"] for row in task_rows))
    values = []
    retained_steps = []
    for step in steps:
        step_rows = [
            row for row in task_rows
            if row["training_step"] == step
        ]
        if key in {"success", "wrong_object"}:
            episodes = sorted(set(row["episode"] for row in step_rows))
            samples = [
                float(any(
                    bool(row[key]) for row in step_rows
                    if row["episode"] == episode
                ))
                for episode in episodes
            ]
        else:
            samples = [
                float(row[key]) for row in step_rows
                if row.get(key) is not None
            ]
        if samples:
            retained_steps.append(step)
            values.append(float(np.mean(samples)))
    return retained_steps, values


def plot_task_metrics(
    rows: list[dict],
    output: Path,
    filename: str,
    title: str,
    metrics: tuple[tuple[str, str], ...],
) -> None:
    colors = {"push": "#4472C4", "place": "#ED7D31"}
    figure, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True)
    for axis, (key, metric_title) in zip(
        axes.flat, metrics, strict=True
    ):
        for task in ("push", "place"):
            steps, values = task_step_values(rows, task, key)
            axis.plot(
                steps,
                values,
                "-o",
                color=colors[task],
                linewidth=2,
                label=task,
            )
        axis.set_title(metric_title)
        axis.grid(alpha=.25)
        axis.set_xticks(sorted(set(
            row["training_step"] for row in rows
        )))
    for axis in axes[-1]:
        axis.set_xlabel("Optimizer step")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2)
    figure.suptitle(title, fontsize=15, y=.98)
    figure.tight_layout(rect=(0, 0, 1, .95))
    figure.savefig(output / filename, dpi=180)
    plt.close(figure)


def plot(training: Path, output: Path) -> None:
    data = json.loads(training.read_text())
    rows = data["rollouts"]
    if not rows or "training_step" not in rows[0]:
        raise ValueError(
            "Training log predates episode-batched optimizer-step logging"
        )
    output.mkdir(parents=True, exist_ok=True)
    episode_rows = data.get("episode_metrics", [])
    if not episode_rows:
        raise ValueError("Training log has no episode-level metrics")
    top_metrics = (
        ("success", "Episode success rate"),
        (
            "completion_efficiency_percent",
            "Completion efficiency (% of task optimum)",
        ),
        (
            "rms_arm_jerk_degrees_per_frame3",
            "Rollout RMS arm jerk (degrees/frame³)",
        ),
    )
    optimization_metrics = (
        ("best_reward", "Mean selected reward"),
        ("advantage", "Mean group-relative advantage"),
        ("loss", "Mean weighted flow loss"),
        ("reward_task_progress", "Task-progress reward"),
        ("reward_success_bonus", "Success bonus"),
        ("reward_wrong_object_penalty", "Wrong-object penalty"),
        ("reward_jerk_penalty", "Jerk penalty"),
        ("reward_time_penalty", "Time penalty"),
        ("rms_arm_jerk", "Selected-chunk RMS jerk"),
    )
    colors = {"push": "#4472C4", "place": "#ED7D31"}
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for axis, (key, metric_title) in zip(
        axes, top_metrics, strict=True
    ):
        for task in ("push", "place"):
            steps, values = task_step_values(
                episode_rows, task, key
            )
            axis.plot(
                steps,
                values,
                "-o",
                color=colors[task],
                linewidth=2,
                label=task,
            )
        axis.set_title(metric_title)
        axis.set_xlabel("Optimizer step")
        axis.grid(alpha=.25)
        axis.set_xticks(sorted(set(
            row["training_step"] for row in episode_rows
        )))
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2)
    figure.suptitle(
        "Oracle-box RL top metrics by optimizer step",
        fontsize=15,
        y=.99,
    )
    figure.tight_layout(rect=(0, 0, 1, .92))
    figure.savefig(output / "top_metrics.png", dpi=180)
    plt.close(figure)
    plot_task_metrics(
        rows,
        output,
        "optimization_metrics.png",
        "Oracle-box RL optimization by optimizer step",
        optimization_metrics,
    )

    steps = data.get("steps", [])
    if steps:
        figure, axes = plt.subplots(1, 2, figsize=(11, 4))
        optimizer_steps = [row["training_step"] for row in steps]
        for axis, key, title in (
            (axes[0], "gradient_norm", "Pre-clip gradient norm"),
            (axes[1], "parameter_rms_drift", "Parameter RMS drift"),
        ):
            axis.plot(
                optimizer_steps,
                [row[key] for row in steps],
                "-o",
                color="#70AD47",
                linewidth=2,
            )
            axis.set_title(title)
            axis.set_xlabel("Optimizer step")
            axis.grid(alpha=.25)
            axis.set_xticks(optimizer_steps)
        figure.tight_layout()
        figure.savefig(output / "step_diagnostics.png", dpi=180)
        plt.close(figure)

    fields = tuple(rows[0])
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    fields = tuple(episode_rows[0])
    with (output / "episode_metrics.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(episode_rows)
    if steps:
        fields = tuple(steps[0])
        with (output / "optimizer_steps.csv").open(
            "w", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(steps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(args.training, args.output)


if __name__ == "__main__":
    main()
