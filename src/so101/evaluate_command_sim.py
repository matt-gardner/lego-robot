from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from so101.evaluate_entity_sim import EntityPolicy, _draw_box, _oriented_iou
from so101.robot_sim import (
    BeamSimulation,
    DESTINATION_NAMES,
    IMAGE_SHAPE,
    MODEL_ROOT,
    OBJECT_NAMES,
)


def evaluate(
    checkpoint: Path,
    output: Path,
    episodes: int,
    seed: int,
    horizon: int,
    action_steps: int,
    device: str,
    depth_input: bool,
    model_root: Path,
    benchmark: Path | None,
    tasks_filter: tuple[str, ...] | None,
    consistent_noise: bool,
    action_samples: int,
    sample_aggregation: str,
    action_smoothing: float,
    phase_prompts: bool,
    zero_noise: bool,
    zero_noise_tasks: tuple[str, ...] | None,
    task_local_seeds: bool,
    adaptive_action_samples: int | None,
    place_action_samples: int | None,
    edge_grasp_fresh: bool,
    edge_place_fresh: bool,
    edge_place_seed_offset: int,
    retry_invalid_chunks: bool,
    adaptive_place_fresh: bool,
    ensemble_checkpoints: tuple[Path, ...],
    ensemble_action_samples: int | None,
    ensemble_adaptive_action_samples: int | None,
    ensemble_tasks: tuple[str, ...] | None,
    ensemble_score_penalty: float,
) -> None:
    if not 0 < action_smoothing <= 1:
        raise ValueError("action_smoothing must be within (0, 1]")
    if adaptive_action_samples is not None and adaptive_action_samples <= 0:
        raise ValueError("adaptive_action_samples must be positive")
    if place_action_samples is not None and place_action_samples <= 0:
        raise ValueError("place_action_samples must be positive")
    if ensemble_action_samples is not None and ensemble_action_samples <= 0:
        raise ValueError("ensemble_action_samples must be positive")
    if (
        ensemble_adaptive_action_samples is not None
        and ensemble_adaptive_action_samples <= 0
    ):
        raise ValueError(
            "ensemble_adaptive_action_samples must be positive"
        )
    output.mkdir(parents=True, exist_ok=False)
    controller = EntityPolicy(
        checkpoint,
        device,
        depth_input,
        consistent_noise,
        action_samples,
        sample_aggregation,
        phase_prompts,
        zero_noise,
        ensemble_checkpoints,
        ensemble_action_samples,
        ensemble_tasks,
        ensemble_score_penalty,
    )
    sim = BeamSimulation(model_root)
    results = []
    tasks = ("grasp", "view", "push", "place")
    task_episode_counts = {task: 0 for task in tasks}
    benchmark_examples = (
        json.loads(benchmark.read_text())["examples"]
        if benchmark is not None
        else None
    )
    if benchmark_examples is not None:
        if tasks_filter is not None:
            benchmark_examples = [
                example
                for example in benchmark_examples
                if example["task"] in tasks_filter
            ]
        episodes = len(benchmark_examples)
    try:
        for episode in range(episodes):
            if benchmark_examples is None:
                phrase = tasks[episode % len(tasks)]
                target = OBJECT_NAMES[(episode // len(tasks)) % len(OBJECT_NAMES)]
                destination = (
                    DESTINATION_NAMES[(episode // len(tasks)) % 2]
                    if phrase in ("push", "place")
                    else None
                )
                scene_seed = seed + episode // len(tasks)
            else:
                example = benchmark_examples[episode]
                phrase = example["task"]
                target = example["target"]
                destination = example["destination"]
                scene_seed = example["seed"]
            entities = [(target, "object")]
            if destination is not None:
                entities.append((destination, "destination"))
            sim.reset(
                scene_seed,
                varied_layout=True,
                real_scene=True,
                color_targets=True,
            )
            sim.configure_command(phrase, entities)
            task_episode_index = task_episode_counts[phrase]
            task_episode_counts[phrase] += 1
            task_zero_noise = (
                phrase in zero_noise_tasks
                if zero_noise_tasks is not None
                else None
            )
            controller_seed = scene_seed + (
                task_episode_index if task_local_seeds else episode
            )
            if edge_grasp_fresh and phrase == "grasp":
                projected_center, center_valid = sim.projected_entity_center(
                    target
                )
                if center_valid and projected_center[1] >= .98:
                    task_zero_noise = False
            if edge_place_fresh and phrase == "place":
                projected_center, center_valid = sim.projected_entity_center(
                    target
                )
                if center_valid and projected_center[1] >= .75:
                    task_zero_noise = False
                    controller_seed += edge_place_seed_offset
            controller.reset(
                sim,
                controller_seed,
                task_zero_noise,
            )
            controller.action_samples = (
                place_action_samples
                if phrase == "place" and place_action_samples is not None
                else action_samples
            )
            controller.ensemble_action_samples = (
                ensemble_action_samples
                if ensemble_action_samples is not None
                else controller.action_samples
            )
            initial_positions = {
                name: sim.position(name).copy() for name in OBJECT_NAMES
            }
            episode_root = output / f"episode_{episode:03}"
            episode_root.mkdir()
            writer = cv2.VideoWriter(
                str(episode_root / "rollout.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                sim.fps,
                (640, 240),
            )
            if not writer.isOpened():
                raise RuntimeError("Failed to open rollout video writer")
            success = False
            wrong_object = False
            error = None
            executed = 0
            stable = 0
            lifted_once = False
            maximum_target_lift = 0.
            executed_actions: list[np.ndarray] = []
            invalid_replans = 0
            center_errors: list[list[float]] = [[] for _ in entities]
            projected_center_errors: list[list[float]] = [
                [] for _ in entities
            ]
            offscreen_center_errors: list[list[float]] = [
                [] for _ in entities
            ]
            shape_errors: list[list[float]] = [[] for _ in entities]
            ious: list[list[float]] = [[] for _ in entities]
            depth_errors: list[list[float]] = [[] for _ in entities]
            try:
                while executed < horizon and not success and not wrong_object:
                    if (
                        phrase == "place"
                        and adaptive_place_fresh
                        and controller.action_noise is not None
                        and (
                            (not lifted_once and executed >= 60)
                            or (lifted_once and executed >= 180)
                        )
                    ):
                        controller.action_noise = None
                    if (
                        phrase == "push"
                        and adaptive_action_samples is not None
                        and executed >= 120
                    ):
                        controller.action_samples = adaptive_action_samples
                        if ensemble_adaptive_action_samples is not None:
                            controller.ensemble_action_samples = (
                                ensemble_adaptive_action_samples
                            )
                    target_lift = float(
                        sim.position(target)[2] - initial_positions[target][2]
                    )
                    contacts = sim.contacts(target)
                    if phrase == "grasp":
                        phase = (
                            "lift"
                            if contacts == {"fixed", "moving"}
                            else "acquire"
                        )
                    elif phrase == "view":
                        phase = "view"
                    elif phrase == "push":
                        gripper_xy = sim.data.site_xpos[sim.site, :2]
                        target_xy = sim.position(target)[:2]
                        phase = (
                            "push"
                            if contacts
                            else "lower"
                            if np.linalg.norm(gripper_xy - target_xy) < .06
                            else "approach"
                        )
                    else:
                        destination_error = float(np.linalg.norm(
                            sim.position(target)[:2]
                            - sim.position(destination)[:2]
                        ))
                        if contacts == {"fixed", "moving"}:
                            if target_lift < .025 and destination_error > .06:
                                phase = "lift"
                            elif destination_error > .055:
                                phase = "carry"
                            elif target_lift > .025:
                                phase = "lower"
                            else:
                                phase = "release"
                        elif lifted_once:
                            phase = "release"
                        else:
                            phase = "acquire"
                    try:
                        actions, boxes, depths, visibilities = (
                            controller.predict_command(
                                sim, phrase, entities, phase
                            )
                        )
                    except (RuntimeError, ValueError) as exception:
                        error = str(exception)
                        break
                    invalid_chunk = False
                    for step in range(min(action_steps, len(actions), horizon - executed)):
                        try:
                            goal, _ = sim.bounded_policy_action(actions[step])
                        except ValueError as exception:
                            invalid_replans += 1
                            if retry_invalid_chunks and invalid_replans <= 8:
                                invalid_chunk = True
                            else:
                                error = str(exception)
                            break
                        if action_smoothing < 1:
                            previous_goal = sim.data.ctrl[
                                sim.actuators
                            ].copy()
                            goal = previous_goal + action_smoothing * (
                                goal - previous_goal
                            )
                        sim.step(goal, "entity_policy")
                        executed_action = sim.encode(goal)
                        executed_actions.append(executed_action)
                        controller.observe(sim, executed_action)
                        executed += 1
                        wrist = cv2.cvtColor(sim.image(), cv2.COLOR_RGB2BGR)
                        for entity_index, (entity, role) in enumerate(entities):
                            actual = np.asarray(
                                sim.target_oriented_box(entity), dtype=np.float32
                            )
                            predicted = boxes[step, entity_index]
                            projected_center, center_valid = (
                                sim.projected_entity_center(entity)
                            )
                            if center_valid:
                                projected_error = float(np.linalg.norm(
                                    predicted[:2]
                                    - np.asarray(projected_center)
                                ))
                                projected_center_errors[entity_index].append(
                                    projected_error
                                )
                                if actual[5] <= .5:
                                    offscreen_center_errors[
                                        entity_index
                                    ].append(projected_error)
                            if actual[5] > .5:
                                center_errors[entity_index].append(float(
                                    np.linalg.norm(predicted[:2] - actual[:2])
                                ))
                                shape_errors[entity_index].append(float(
                                    np.linalg.norm(predicted[2:5] - actual[2:5])
                                ))
                                if visibilities[step, entity_index] >= .5:
                                    ious[entity_index].append(_oriented_iou(
                                        predicted,
                                        actual[:5],
                                        IMAGE_SHAPE[1],
                                        IMAGE_SHAPE[0],
                                    ))
                            actual_depth, depth_valid = sim.entity_camera_depth(entity)
                            if depth_valid:
                                depth_errors[entity_index].append(abs(
                                    float(depths[step, entity_index]) - actual_depth
                                ))
                            _draw_box(
                                wrist,
                                actual[:5],
                                (0, 255, 0) if role == "object" else (0, 180, 255),
                                role,
                            )
                            if visibilities[step, entity_index] >= .5:
                                _draw_box(
                                    wrist,
                                    predicted,
                                    (255, 180, 0),
                                    f"pred {role}",
                                )
                        overview = cv2.cvtColor(
                            sim.overview_image(), cv2.COLOR_RGB2BGR
                        )
                        cv2.putText(
                            overview,
                            f"{phrase} {target} step={executed}",
                            (8, 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            .5,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )
                        writer.write(np.concatenate((overview, wrist), axis=1))
                        target_lift = (
                            sim.position(target)[2] - initial_positions[target][2]
                        )
                        maximum_target_lift = max(
                            maximum_target_lift, float(target_lift)
                        )
                        lifted_once |= target_lift >= .03
                        for other in OBJECT_NAMES:
                            if other == target:
                                continue
                            other_lift = (
                                sim.position(other)[2] - initial_positions[other][2]
                            )
                            if (
                                other_lift >= .04
                                and sim.contacts(other) == {"fixed", "moving"}
                            ):
                                wrong_object = True
                                break
                        if wrong_object:
                            break
                        condition = False
                        if phrase == "grasp":
                            condition = (
                                target_lift >= .05
                                and sim.contacts(target) == {"fixed", "moving"}
                            )
                        elif phrase == "view":
                            actual = sim.target_oriented_box(target)
                            condition = (
                                executed >= 30
                                and bool(actual[5])
                                and np.linalg.norm(np.asarray(actual[:2]) - .5) < .3
                            )
                        else:
                            destination_error = float(np.linalg.norm(
                                sim.position(target)[:2]
                                - sim.position(destination)[:2]
                            ))
                            if phrase == "push":
                                displacement = float(np.linalg.norm(
                                    sim.position(target)[:2]
                                    - initial_positions[target][:2]
                                ))
                                condition = (
                                    executed >= 30
                                    and destination_error <= .07
                                    and displacement >= .02
                                )
                            else:
                                condition = (
                                    lifted_once
                                    and destination_error <= .04
                                    and sim.contacts(target) != {"fixed", "moving"}
                                )
                        stable = stable + 1 if condition else 0
                        if stable >= sim.fps:
                            success = True
                            break
                    if error is not None:
                        break
                    if invalid_chunk:
                        continue
            finally:
                writer.release()
            entity_metrics = []
            for entity_index, (entity, role) in enumerate(entities):
                entity_metrics.append({
                    "entity": entity,
                    "role": role,
                    "mean_center_error": (
                        float(np.mean(center_errors[entity_index]))
                        if center_errors[entity_index] else None
                    ),
                    "mean_projected_center_error": (
                        float(np.mean(
                            projected_center_errors[entity_index]
                        ))
                        if projected_center_errors[entity_index]
                        else None
                    ),
                    "mean_offscreen_center_error": (
                        float(np.mean(
                            offscreen_center_errors[entity_index]
                        ))
                        if offscreen_center_errors[entity_index]
                        else None
                    ),
                    "mean_shape_error": (
                        float(np.mean(shape_errors[entity_index]))
                        if shape_errors[entity_index] else None
                    ),
                    "mean_oriented_iou": (
                        float(np.mean(ious[entity_index]))
                        if ious[entity_index] else None
                    ),
                    "mean_depth_error_m": (
                        float(np.mean(depth_errors[entity_index]))
                        if depth_errors[entity_index] else None
                    ),
                })
            action_array = np.asarray(executed_actions, dtype=np.float32)
            final_target_lift = float(
                sim.position(target)[2] - initial_positions[target][2]
            )
            target_displacement = float(np.linalg.norm(
                sim.position(target)[:2] - initial_positions[target][:2]
            ))
            final_destination_error = (
                float(np.linalg.norm(
                    sim.position(target)[:2]
                    - sim.position(destination)[:2]
                ))
                if destination is not None
                else None
            )
            arm_delta = (
                np.diff(action_array[:, :5], axis=0)
                if len(action_array) > 1
                else np.zeros((0, 5), dtype=np.float32)
            )
            arm_acceleration = (
                np.diff(arm_delta, axis=0)
                if len(arm_delta) > 1
                else np.zeros((0, 5), dtype=np.float32)
            )
            arm_jerk = (
                np.diff(arm_acceleration, axis=0)
                if len(arm_acceleration) > 1
                else np.zeros((0, 5), dtype=np.float32)
            )
            result = {
                "seed": scene_seed,
                "task": phrase,
                "target": target,
                "destination": destination,
                "success": success,
                "wrong_object": wrong_object,
                "steps": executed,
                "completion_seconds": executed / sim.fps,
                "maximum_target_lift_m": maximum_target_lift,
                "final_target_lift_m": final_target_lift,
                "target_displacement_m": target_displacement,
                "final_destination_error_m": final_destination_error,
                "final_target_contacts": sorted(sim.contacts(target)),
                "p95_arm_delta_degrees_per_frame": (
                    float(np.quantile(np.abs(arm_delta), .95))
                    if arm_delta.size else 0.
                ),
                "rms_arm_acceleration_degrees_per_frame2": (
                    float(np.sqrt(np.mean(arm_acceleration ** 2)))
                    if arm_acceleration.size else 0.
                ),
                "rms_arm_jerk_degrees_per_frame3": (
                    float(np.sqrt(np.mean(arm_jerk ** 2)))
                    if arm_jerk.size else 0.
                ),
                "error": error,
                "invalid_replans": invalid_replans,
                "entity_metrics": entity_metrics,
                "video": str(episode_root / "rollout.mp4"),
            }
            results.append(result)
            print(json.dumps(result), flush=True)
            (episode_root / "result.json").write_text(json.dumps(result, indent=2))
            (output / "evaluation.json").write_text(json.dumps({
                "checkpoint": str(checkpoint),
                "depth_input": depth_input,
                "success_rate": sum(row["success"] for row in results) / len(results),
                "wrong_object_rate": (
                    sum(row["wrong_object"] for row in results) / len(results)
                ),
                "by_task": {
                    task: {
                        "episodes": sum(row["task"] == task for row in results),
                        "successes": sum(
                            row["task"] == task and row["success"]
                            for row in results
                        ),
                    }
                    for task in tasks
                },
                "episodes": results,
            }, indent=2))
    finally:
        sim.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=30_000)
    parser.add_argument("--horizon", type=int, default=260)
    parser.add_argument("--action-steps", type=int, default=16)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument(
        "--depth-input", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--consistent-noise",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--action-samples", type=int, default=1)
    parser.add_argument("--adaptive-action-samples", type=int)
    parser.add_argument("--place-action-samples", type=int)
    parser.add_argument(
        "--edge-grasp-fresh",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--edge-place-fresh",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--edge-place-seed-offset", type=int, default=0)
    parser.add_argument(
        "--retry-invalid-chunks",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--adaptive-place-fresh",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--sample-aggregation",
        choices=("mean", "smoothest", "goal", "oracle"),
        default="mean",
    )
    parser.add_argument("--action-smoothing", type=float, default=1)
    parser.add_argument(
        "--phase-prompts",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--zero-noise",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--zero-noise-tasks",
        nargs="+",
        choices=("grasp", "view", "push", "place"),
    )
    parser.add_argument(
        "--task-local-seeds",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument(
        "--ensemble-checkpoint",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--ensemble-action-samples", type=int)
    parser.add_argument("--ensemble-adaptive-action-samples", type=int)
    parser.add_argument("--ensemble-score-penalty", type=float, default=0.)
    parser.add_argument(
        "--ensemble-tasks",
        nargs="+",
        choices=("grasp", "view", "push", "place"),
    )
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=("grasp", "view", "push", "place"),
    )
    args = parser.parse_args()
    evaluate(
        args.checkpoint,
        args.output,
        args.episodes,
        args.seed,
        args.horizon,
        args.action_steps,
        args.device,
        args.depth_input,
        args.model_root,
        args.benchmark,
        tuple(args.tasks) if args.tasks else None,
        args.consistent_noise,
        args.action_samples,
        args.sample_aggregation,
        args.action_smoothing,
        args.phase_prompts,
        args.zero_noise,
        tuple(args.zero_noise_tasks) if args.zero_noise_tasks else None,
        args.task_local_seeds,
        args.adaptive_action_samples,
        args.place_action_samples,
        args.edge_grasp_fresh,
        args.edge_place_fresh,
        args.edge_place_seed_offset,
        args.retry_invalid_chunks,
        args.adaptive_place_fresh,
        tuple(args.ensemble_checkpoint),
        args.ensemble_action_samples,
        args.ensemble_adaptive_action_samples,
        tuple(args.ensemble_tasks) if args.ensemble_tasks else None,
        args.ensemble_score_penalty,
    )


if __name__ == "__main__":
    main()
