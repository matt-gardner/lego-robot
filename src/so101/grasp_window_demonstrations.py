"""Physical approach/closure demonstrations from independently admitted anchors."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.optimize import least_squares

from so101.cloud_checkpoint import atomic_json
from so101.pile_task_runtime import (
    PileSimulation, body_surface_speed_bound, load_problem, reset_problem,
    target_contact_state,
)
from so101.robot_sim import BeamSimulation
from so101.robot_table_clearance import RobotTableIntersection, require_robot_above_table
from so101.rl_pile_adapter import NativePileSimulation
from so101.simulation_corpus import CorpusEpisode


WINDOW_LEVELS = {"grasp": (.7, .75, .8, .85, .9), "place": (.35, .4, .45, .5)}
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class WindowRejected(ValueError):
    """An admitted static family did not yield an acceptable dynamic window."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diverse_pilot(families: list[dict], limit: int) -> list[dict]:
    selected = []
    for kind in WINDOW_LEVELS:
        groups = {}
        for family in families:
            if family["task"] == kind:
                groups.setdefault(family["entry"]["scene_family_id"], []).append(family)
        ranked = [groups[key] for key in sorted(groups)]
        interleaved = [group[index] for index in range(max(map(len, ranked), default=0))
                       for group in ranked if index < len(group)]
        selected.extend(interleaved[:limit])
    return selected


def gravity_compensated_command(sim: BeamSimulation, desired: np.ndarray) -> np.ndarray:
    actuators = sim.actuators[:5]
    gear = sim.model.actuator_gear[actuators]
    gain = sim.model.actuator_gainprm[actuators, 0]
    if (not np.all(gear[:, 0] == 1) or np.any(gear[:, 1:] != 0)
            or np.any(gain <= 0)
            or not np.all(sim.model.actuator_biasprm[actuators, 1] == -gain)):
        raise ValueError("Expert feedforward requires unit-gear position actuators")
    sim.scratch.qpos[:] = sim.data.qpos
    sim.scratch.qpos[sim.qadr] = desired
    sim.scratch.qvel[:] = 0
    mujoco.mj_forward(sim.model, sim.scratch)
    command = desired.copy()
    command[:5] += sim.scratch.qfrc_bias[sim.dofs[:5]] / gain
    if np.any(command < sim.limits[:, 0]) or np.any(command > sim.limits[:, 1]):
        raise WindowRejected("Gravity-compensated command exceeds existing joint limits")
    return command


def clearance_pose(
    sim: BeamSimulation, desired: np.ndarray, lift_m: float = .0015,
    offset_xy: np.ndarray | None = None,
) -> np.ndarray:
    sim.scratch.qpos[:] = sim.data.qpos
    sim.scratch.qpos[sim.qadr] = desired
    mujoco.mj_forward(sim.model, sim.scratch)
    position = sim.scratch.site_xpos[sim.site].copy() + [0, 0, lift_m]
    if offset_xy is not None:
        position[:2] += offset_xy
    rotation = sim.scratch.xmat[sim.gripper].reshape(3, 3).copy()

    def residual(arm):
        sim.scratch.qpos[sim.qadr] = np.r_[arm, desired[-1]]
        mujoco.mj_forward(sim.model, sim.scratch)
        return np.r_[
            (sim.scratch.site_xpos[sim.site] - position) * 100,
            (sim.scratch.xmat[sim.gripper].reshape(3, 3) - rotation).ravel(),
        ]

    fit = least_squares(residual, desired[:5],
                        bounds=(sim.limits[:5, 0], sim.limits[:5, 1]), max_nfev=100)
    error = residual(fit.x)
    if not fit.success or np.linalg.norm(error[:3]) > .01:
        raise WindowRejected(f"Cannot obtain a {lift_m}m raised approach pose within 0.1mm")
    require_robot_above_table(sim.model, sim.scratch)
    return np.r_[fit.x, desired[-1]]


def window_tasks(scene: dict) -> list[dict]:
    tasks = scene["tasks"]
    kinds = {task["task_type"] for task in tasks}
    parents = {task["parent_task_id"] for task in tasks}
    if len(kinds) != 1 or len(parents) != 1:
        raise ValueError("A window family must have one task type and original parent")
    kind = next(iter(kinds))
    if kind not in WINDOW_LEVELS:
        raise ValueError("Only GRASP and PLACE supply close-grasp windows")
    by_level = {}
    for task in tasks:
        level = task["assistance"]["value"]
        if level in by_level:
            raise ValueError("Duplicate assistance anchor")
        by_level[level] = task
    if not set(WINDOW_LEVELS[kind]) <= by_level.keys():
        raise ValueError("Missing approach/closure anchors")
    return [by_level[level] for level in WINDOW_LEVELS[kind]]


def freeze_intake(prior_grasp: Path, inventory: Path, receipts: Path) -> dict:
    sources = [{"path": str(path), "sha256": sha256(path)} for path in (prior_grasp, inventory)]
    candidates = [(Path(path), None) for path in json.loads(prior_grasp.read_text())["admissions"]]
    jobs = json.loads(inventory.read_text())["jobs"]
    for job in jobs:
        retained = job.get("retained_result", {})
        if job["task_type"] in WINDOW_LEVELS and retained.get("status") == "complete":
            candidates.append((Path(retained["admission_path"]), retained["admission_sha256"]))
    for path in sorted(receipts.glob("*.json")):
        data = json.loads(path.read_text())
        if data.get("task_type") in WINDOW_LEVELS and data["status"] == "complete":
            sources.append({"path": str(path), "sha256": sha256(path)})
            candidates.append((Path(data["admission_path"]), data["admission_sha256"]))
    families = {}
    for path, expected in candidates:
        actual = sha256(path)
        if expected is not None and actual != expected:
            raise ValueError(f"Changed admitted family: {path}")
        entry = json.loads(path.read_text())
        problem_path = Path(entry["root"]) / "problem.json"
        if sha256(problem_path) != entry["files_sha256"]["problem.json"]:
            raise ValueError(f"Changed admitted problem: {problem_path}")
        scene = json.loads(problem_path.read_text())
        selected = window_tasks(scene)
        parent = selected[0]["parent_task_id"]
        row = {"admission": str(path), "admission_sha256": actual, "entry": entry,
               "parent_task_id": parent, "task": selected[0]["task_type"]}
        if parent in families and families[parent] != row:
            raise ValueError(f"Conflicting family versions for {parent}")
        families[parent] = row
    if not families:
        raise ValueError("No published grasp/place families")
    return {
        "kind": "grasp-window-intake-v1", "sources": sources,
        "families": sorted(families.values(), key=lambda row: (row["task"], row["parent_task_id"])),
        "windows": WINDOW_LEVELS,
        "scope": "Published families at this snapshot; live constructor additions need explicit reconciliation",
        "split_scope": "Original family splits retained; no fabricated heldout population",
    }


def generate_window(entry: dict) -> dict:
    scene = json.loads((Path(entry["root"]) / "problem.json").read_text())
    tasks = window_tasks(scene)
    problem = load_problem(entry, tasks[0]["task_id"])
    model = mujoco.MjModel.from_binary_path(str(problem["root"] / "model.mjb"))
    force_ranges = model.actuator_forcerange.copy()
    names = tuple(item["instance_id"] for item in scene["instances"])
    sim = BeamSimulation(model=model, render=False, record_images=False,
                         object_names=names, distractor_names=())
    model.actuator_forcerange[:] = force_ranges
    native = PileSimulation(model, sim.data, None, {
        item["instance_id"]: model.body(item["body_name"]).id for item in scene["instances"]})
    try:
        reset_problem(native, problem)
        initial = tasks[0]["initial_state"]
        nonrobot = np.ones(model.nq, dtype=bool)
        nonrobot[sim.qadr] = False
        goals = []
        for task in tasks:
            state = np.asarray(task["initial_state"]["qpos"])
            if not np.array_equal(state[nonrobot], np.asarray(initial["qpos"])[nonrobot]):
                raise WindowRejected("Anchors change the non-robot scene; cannot splice them into a demonstration")
            goals.append(state[sim.qadr])
        opened_index = 2 if tasks[0]["task_type"] == "grasp" else 1
        anchor_goals = goals
        goals = [
            goals[0], clearance_pose(sim, goals[0], .01),
            clearance_pose(sim, goals[opened_index], .01),
            *(clearance_pose(sim, goal) for goal in goals[opened_index:]),
        ]
        target = tasks[0]["target_instance_id"]
        original_position = native.position(target)
        rows = []
        max_displacement = 0.
        stable = 0
        clearance_substeps = 0
        original_obstacle_contacts = sim.obstacle_robot_contacts

        def checked_obstacle_contacts():
            nonlocal clearance_substeps
            require_robot_above_table(model, sim.data)
            clearance_substeps += 1
            return original_obstacle_contacts()

        sim.obstacle_robot_contacts = checked_obstacle_contacts

        def execute(desired, phase):
            nonlocal max_displacement, stable
            goal = sim.decode(sim.encode(gravity_compensated_command(sim, desired)))
            row = {
                "step": len(rows), "sim_time": float(sim.data.time), "phase": phase,
                "state": sim.state().tolist(), "action": sim.encode(goal).tolist(),
                "goal_radians": goal.tolist(), "pre_qpos": sim.data.qpos.tolist(),
                "desired_pose_radians": desired.tolist(),
            }
            sim.step(goal, phase)
            require_robot_above_table(model, sim.data)
            displacement = float(np.linalg.norm(native.position(target) - original_position))
            max_displacement = max(max_displacement, displacement)
            contact = target_contact_state(native, target)
            speed = float(body_surface_speed_bound(model, sim.data, native.registry[target]))
            stable = stable + 1 if contact["bilateral_grip"] and speed < .03 else 0
            row.update(post_qpos=sim.data.qpos.tolist(), contacts=contact,
                       target_position=native.position(target).tolist(), target_speed_m_s=speed)
            rows.append(row)

        for index, goal in enumerate(goals[1:], 1):
            if index >= 3:
                # Follow observed target motion without changing the physical scene.
                offset_xy = (native.position(target) - original_position)[:2]
                goal = clearance_pose(sim, anchor_goals[opened_index + index - 3],
                                      offset_xy=offset_xy)
                goals[index] = goal
            start = goals[index - 1]
            duration = max(.3, float(np.max(np.abs(goal - start))) / np.deg2rad(20))
            steps = max(1, math.ceil(duration * sim.fps))
            phase = "approach" if index <= 3 else "close"
            for tick in range(1, steps + 1):
                execute(start + (goal - start) * tick / steps, phase)
            for _ in range(3):
                execute(goal, phase)
        closed = goals[-1].copy()
        for _ in range(30):
            if stable >= 10:
                break
            if stable == 0:
                closed[-1] = max(sim.limits[-1, 0], closed[-1] - np.deg2rad(.25))
            execute(closed, "close" if stable == 0 else "hold")
        accepted = stable >= 10
        return {
            "kind": "physical-grasp-window-demonstration-v1", "accepted": accepted,
            "task": tasks[0]["task_type"], "parent_task_id": tasks[0]["parent_task_id"],
            "scene_family_id": entry["scene_family_id"], "split": entry["split"],
            "entry": entry, "initial_task_id": tasks[0]["task_id"],
            "levels": [task["assistance"]["value"] for task in tasks],
            "fps": sim.fps, "physics": mujoco.__version__,
            "generator_source_sha256": SOURCE_SHA256,
            "controller": "20deg-per-second lift-align-descend path with 10mm hover/1.5mm raised closure and measured-target XY feedback; arm gravity feedforward; float32 action roundtrip; bounded 0.25deg contact-feedback closure; original gains/forces unchanged",
            "acceptance": {
                "consecutive_bilateral_low_speed_frames": stable,
                "required_frames": 10, "maximum_target_displacement_m": max_displacement,
                "maximum_hold_speed_m_s": .03,
                "table_clearance_checked_substeps": clearance_substeps,
                "scope": "Approach/closure window only, not full-task lift/place success",
            },
            "full_task_success_claimed": False, "rows": rows,
        }
    finally:
        sim.close()


def render_window(trace_path: Path, output: Path, model_root: Path) -> dict:
    trace = json.loads(trace_path.read_text())
    if not trace["accepted"]:
        raise ValueError("Rejected physical windows cannot supply SFT observations")
    output.mkdir(parents=True, exist_ok=False)
    (output / "wrist").mkdir()
    episode = CorpusEpisode(
        Path(trace["entry"]["root"]) / "tasks" / trace["initial_task_id"],
        {"native_entry": trace["entry"], "native_task_id": trace["initial_task_id"],
         "task": trace["task"]},
    )
    sim = NativePileSimulation(episode, model_root, backend="cpu-pathtrace", samples=16, threads=1)
    # Keep path-traced RGB separate from the maintained exact zero-MSAA ID context.
    sim._segmentation_image = sim.native.segmentation_image
    try:
        target, _, entities = sim.reset_native(episode)
        initial_history_action = sim.state().tolist()
        start_time = float(sim.data.time)
        rows = []
        maximum_error = 0.
        for recorded in trace["rows"]:
            error = float(np.max(np.abs(sim.data.qpos - recorded["pre_qpos"])))
            maximum_error = max(maximum_error, error)
            if error > 1e-8:
                raise ValueError("Rendered demonstration reset/state differs from physical trace")
            image = sim.image()
            image_name = f"wrist/{len(rows):06}.png"
            if not cv2.imwrite(str(output / image_name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
                raise RuntimeError(f"Could not write {output / image_name}")
            boxes = [sim.target_oriented_box(entity) for entity, _ in entities]
            depths = [sim.entity_camera_depth(entity) for entity, _ in entities]
            row = {
                "index": len(rows), "elapsed": float(sim.data.time) - start_time,
                "image": image_name, "state": sim.state().tolist(),
                "action": recorded["action"], "phase": recorded["phase"],
                "command_boxes": [box[:5] for box in boxes],
                "command_visibility": [float(box[5]) for box in boxes],
                "command_depth_m": [float(depth) for depth, _ in depths],
                "command_depth_valid": [float(valid) for _, valid in depths],
                "command_center_valid": [float(sim.projected_entity_center(entity)[1])
                                         for entity, _ in entities],
                "command_roles": [role for _, role in entities],
                "command_entity_ids": [entity for entity, _ in entities],
            }
            rows.append(row)
            sim.step(np.asarray(recorded["goal_radians"]), recorded["phase"])
            error = float(np.max(np.abs(sim.data.qpos - recorded["post_qpos"])))
            maximum_error = max(maximum_error, error)
            if error > 1e-8:
                raise ValueError("Rendered demonstration replay diverged from verified physics")
        # Publish consumable episode metadata only after every replayed step matched.
        with (output / "frames.jsonl").open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        atomic_json(output / "physical-trace.json", trace)
        atomic_json(output / "episode.json", {
            "task": trace["task"], "target": target, "outcome": "success",
            "success_scope": "grasp-window-only-not-full-task",
            "full_task_success_claimed": False, "initial": sim.initial,
            "initial_history_action": initial_history_action,
            "scene_family_id": trace["scene_family_id"], "split": trace["split"],
            "parent_task_id": trace["parent_task_id"], "levels": trace["levels"],
            "pair_id": f"window:{trace['parent_task_id']}",
            "trace": str(output / "physical-trace.json"),
            "trace_sha256": sha256(output / "physical-trace.json"),
            "source_trace_sha256": sha256(trace_path),
            "render_source_sha256": SOURCE_SHA256,
            "frames_sha256": sha256(output / "frames.jsonl"),
            "maximum_replay_qpos_error": maximum_error,
            "rgb_renderer": "cpu-pathtrace", "rgb_samples": 16,
            "label_renderer": "native-id-segmentation-no-msaa-v1",
            "acceptance": trace["acceptance"],
        })
        return {"source": str(output), "frames": len(rows), "maximum_replay_qpos_error": maximum_error,
                "task": trace["task"], "split": trace["split"],
                "scene_family_id": trace["scene_family_id"]}
    finally:
        sim.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    for name in ("prior-grasp", "inventory", "receipts", "output"):
        freeze.add_argument("--" + name, type=Path, required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--intake", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--per-task-limit", type=int)
    generate.add_argument("--resume", action="store_true")
    render = subparsers.add_parser("render")
    render.add_argument("--traces", type=Path, required=True)
    render.add_argument("--intake", type=Path, required=True)
    render.add_argument("--output", type=Path, required=True)
    render.add_argument("--model-root", type=Path, required=True)
    args = parser.parse_args()
    resuming = args.command == "generate" and args.resume
    if args.output.exists() and not resuming:
        raise FileExistsError("Reconcile existing output before restarting")
    if resuming and not args.output.is_dir():
        raise ValueError("No existing physical generation to resume")
    if args.command == "freeze":
        atomic_json(args.output, freeze_intake(args.prior_grasp, args.inventory, args.receipts))
        return
    if args.command == "render":
        status = json.loads(args.traces.read_text())
        if not status["complete"]:
            raise ValueError("Require completed physics generation before rendering")
        if sha256(args.intake) != status["intake_sha256"]:
            raise ValueError("Rendered intake differs from the physical generation snapshot")
        accepted = [report for report in status["reports"] if report["accepted"]]
        if not accepted:
            raise ValueError("No verified windows to render; no SFT dataset produced")
        args.output.mkdir(parents=True)
        atomic_json(args.output / "intake.json", json.loads(args.intake.read_text()))
        atomic_json(args.output / "physical-generation.json", status)
        reports = []
        for report in accepted:
            path = Path(report["trace"])
            if sha256(path) != report["trace_sha256"]:
                raise ValueError("Verified physical trace changed")
            reports.append(render_window(path, args.output / path.stem, args.model_root))
            atomic_json(args.output / "status.json", {"complete": False, "episodes": reports})
        atomic_json(args.output / "collection.json", {
            "kind": "grasp-window-sft-v1", "status": "complete",
            "successful_episodes": [report["source"] for report in reports],
            "episodes": reports,
            "physical_generation": str(args.output / "physical-generation.json"),
            "physical_generation_sha256": sha256(args.output / "physical-generation.json"),
            "intake_sha256": sha256(args.output / "intake.json"),
            "success_scope": "Verified approach/closure windows, not complete native tasks",
            "split_scope": "Original family splits retained; no new heldout claims",
        })
        atomic_json(args.output / "status.json", {"complete": True, "episodes": reports})
        return
    if args.per_task_limit is not None and args.per_task_limit < 1:
        raise ValueError("Positive per-task limit required")
    if not resuming:
        args.output.mkdir(parents=True)
    intake = json.loads(args.intake.read_text())
    families = intake["families"]
    if args.per_task_limit is not None:
        families = diverse_pilot(families, args.per_task_limit)
    planned = [family["parent_task_id"] for family in families]
    status = {
        "complete": False, "intake_sha256": sha256(args.intake), "planned_parents": planned,
        "reports": [],
        "scope": "Physics traces only; RGB/labels and replay verification still required for SFT",
    }
    if resuming:
        previous = json.loads((args.output / "status.json").read_text())
        if (previous["intake_sha256"] != status["intake_sha256"]
                or previous["planned_parents"] != planned):
            raise ValueError("Resume requires the same frozen intake and selected families")
        status["reports"] = previous["reports"]
    reports = status["reports"]
    completed = set()
    for report in reports:
        parent = report["parent_task_id"]
        if parent in completed or parent not in planned:
            raise ValueError("Invalid completed family in generation checkpoint")
        if "trace" in report and sha256(Path(report["trace"])) != report["trace_sha256"]:
            raise ValueError("Completed physical trace changed before resume")
        completed.add(parent)
    atomic_json(args.output / "status.json", status)
    for family in families:
        if family["parent_task_id"] in completed:
            continue
        kind = family["task"]
        path = Path(family["admission"])
        if sha256(path) != family["admission_sha256"]:
            raise ValueError(f"Changed frozen admission: {path}")
        try:
            trace = generate_window(family["entry"])
        except (WindowRejected, RobotTableIntersection) as error:
            report = {"task": kind, "parent_task_id": family["parent_task_id"],
                      "accepted": False, "reason": str(error)}
        else:
            trace_path = args.output / f"{hashlib.sha256(family['parent_task_id'].encode()).hexdigest()}.json"
            atomic_json(trace_path, trace)
            report = {key: value for key, value in trace.items() if key != "rows"}
            report.update(trace=str(trace_path), trace_sha256=sha256(trace_path), frames=len(trace["rows"]))
        reports.append(report)
        atomic_json(args.output / "status.json", status)
        print(json.dumps({key: report[key] for key in ("task", "parent_task_id", "accepted")}), flush=True)
    status["complete"] = True
    atomic_json(args.output / "status.json", status)


if __name__ == "__main__":
    main()
