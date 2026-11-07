"""Generate bounded, seeded task difficulty examples using only the physics expert."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import mujoco
import numpy as np

from so101.cloud_checkpoint import atomic_json
from so101.robot_sim import (
    BeamSimulation, DISTRACTOR_SIZES, MODEL_ROOT, MODEL_COMMIT, OBJECT_NAMES,
    SCENE_VERSION, TABLE_Z,
)


TASKS = ("grasp", "view", "push", "place")
LEVELS = ("easy", "standard", "hard", "challenge")
FACTORS = {
    "grasp": ("clear central pickup", "outer reach and rotation",
              "distant camera start and clutter", "obstacle detour and retention"),
    "view": ("visible central target", "off-center target",
             "initially out of frame; wrist-camera search", "initially occluded; parallax"),
    "push": ("short straight push", "diagonal push",
             "cross-workspace push", "long diagonal push; tighter destination"),
    "place": ("short carry", "diagonal carry",
              "cross-workspace carry", "long carry; precise release"),
}


def snapshot(sim: BeamSimulation) -> None:
    sim.initial.update(
        arm_state=sim.state().tolist(),
        objects={name: sim.position(name).tolist() for name in OBJECT_NAMES},
        yaws=dict(sim.yaws),
    )
    appearance = sim.initial["appearance"]
    for key, names in (
        ("distractors", sim.distractors),
        ("destinations", sim.destination_geoms),
        ("obstacles", sim.obstacle_geoms),
    ):
        appearance[key] = {name: sim.position(name).tolist()
                           for name in names}
    sim.snapshot_initial_physics()


def arm_at(sim: BeamSimulation, point: np.ndarray, yaw: float) -> None:
    arm = sim.ik(point, yaw)
    arm[-1] = sim.limits[-1, 0] + .55 * np.ptp(sim.limits[-1])
    sim.data.qpos[sim.qadr] = arm
    sim.data.qvel[:] = 0
    sim.data.ctrl[sim.actuators] = arm
    mujoco.mj_forward(sim.model, sim.data)


def configure_scene(sim: BeamSimulation, task: str, level: int, seed: int) -> dict:
    if task not in TASKS or level not in range(4):
        raise ValueError("Expected a supported task and difficulty level 0..3")
    sim.episode_root = None
    sim.reset(seed, varied_layout=True, real_scene=True, color_targets=True)
    rng = np.random.default_rng(seed)
    target = "short"
    destination = "goal_right" if task in ("push", "place") else None
    for index, name in enumerate(OBJECT_NAMES):
        address = sim.model.joint(f"{name}_free").qposadr[0]
        position = (
            [.38, -.30 + index * .10, TABLE_Z + .0058]
            if sim.scene_version == SCENE_VERSION else
            [.30, -.21 + index * .035, TABLE_Z + .0058]
        )
        if name == target:
            position[:2] = [.22, .025 if level == 0 else .12]
        yaw = float(rng.uniform(-.12, .12) if level == 0 else rng.uniform(-.5, .5))
        sim.yaws[name] = yaw
        sim.data.qpos[address:address + 7] = [
            *position, math.cos(yaw / 2), 0., 0., math.sin(yaw / 2),
        ]
    for name in sim.distractors:
        sim.set_body_pose(name, [0., 0., -1.])
    if level >= 2:
        for index, name in enumerate(sim.distractors):
            sim.place_prop(
                name, [.16 + index * .025, -.10, TABLE_Z + DISTRACTOR_SIZES[name][2] + .001],
                [1., 0., 0., 0.], rng,
            )
    if destination:
        goals = ([.22, -.06], [.20, .015], [.15, -.08], [.15, -.12]) if task == "push" else (
            [.17, -.025], [.15, .02], [.15, -.08], [.15, -.12])
        goal = goals[level]
        sim.model.body_pos[sim.model.body(destination).id] = [
            *goal, TABLE_Z + (.00002 if sim.scene_version == SCENE_VERSION else .001)
        ]
    mujoco.mj_forward(sim.model, sim.data)
    position = sim.position(target).copy()
    yaw = sim.yaws[target] + math.pi / 2
    arm_at(sim, np.r_[position[:2] + ([0., 0.] if level == 0 else [0., -.10]), .075], yaw)
    sim.settle_initial_scene()
    position = sim.position(target).copy()
    obstacles = []
    if task == "grasp" and level == 3:
        sim.place_grasp_obstacle(target)
        obstacles = ["obstacle_box"]
    sim.configure_command(task, [(target, "object")] +
                          ([(destination, "destination")] if destination else []) +
                          [(name, "obstacle") for name in obstacles])
    visible_before_occluder = None
    pose_rejections = []
    if task == "view":
        if level in (0, 1, 2):
            desired = (lambda box: bool(box[5]) and np.linalg.norm(np.array(box[:2]) - .5) < .20,
                       lambda box: bool(box[5]) and np.linalg.norm(np.array(box[:2]) - .5) >= .20,
                       lambda box: not bool(box[5]) and any(value < 0 or value > 1 for value in box[:2]))[level]
            found = False
            for dx, dy in ((0., 0.), (.03, 0.), (-.03, .03), (0., -.07),
                           (0., -.12), (-.05, -.15), (.02, -.20)):
                try:
                    arm_at(sim, np.r_[position[:2] + [dx, dy], .075], yaw)
                except RuntimeError as error:
                    pose_rejections.append(str(error))
                    continue
                if desired(sim.target_oriented_box(target)):
                    found = True
                    break
            if not found:
                raise ValueError(f"No camera start meeting view level {level}: {pose_rejections}")
        else:
            obstacle_id = sim.model.body("obstacle_box").id
            found = False
            camera_starts = [(-.06, .02), (-.08, .02), (-.05, -.02), (0., -.03)]
            if sim.scene_version == SCENE_VERSION:
                camera_starts = [(0., -.20), (-.05, -.25), (0., -.25), (.05, -.20),
                                 (-.10, 0.), (0., -.10)] + camera_starts
            for dx, dy in camera_starts:
                sim.model.body_pos[obstacle_id] = [0., 0., -1.]
                try:
                    arm_at(sim, np.r_[position[:2] + [dx, dy], .075], yaw)
                except RuntimeError as error:
                    pose_rejections.append(str(error))
                    continue
                unoccluded = int(sim.entity_mask(target).sum())
                if unoccluded < 12:
                    continue
                camera = sim.data.cam_xpos[sim.model.camera("wrist_cam").id].copy()
                fractions = (.25, .35, .45, .55, .65, .75) if sim.scene_version == SCENE_VERSION else (
                    .45, .55, .65, .75
                )
                for fraction in fractions:
                    blocker = position + fraction * (camera - position)
                    blocker[2] = TABLE_Z + .045
                    sim.model.body_pos[obstacle_id] = blocker
                    mujoco.mj_forward(sim.model, sim.data)
                    obstacle_geoms = set(sim.obstacle_geoms["obstacle_box"])
                    target_geoms = set(sim.beam_geoms[target])
                    interpenetrates = any(
                        contact.dist < -1e-5 and (
                            contact.geom1 in obstacle_geoms and contact.geom2 in target_geoms or
                            contact.geom2 in obstacle_geoms and contact.geom1 in target_geoms
                        ) for contact in sim.data.contact
                    )
                    if sim.scene_version == SCENE_VERSION:
                        interpenetrates = bool(sim.initial_penetrations())
                    if interpenetrates:
                        continue
                    visible = int(sim.entity_mask(target).sum())
                    if visible == 0:
                        found = True
                        visible_before_occluder = unoccluded
                        break
                if found:
                    break
            if not found:
                raise ValueError("No verifiably occluded camera start")
            obstacles = ["obstacle_box"]
            sim.configure_command(task, [(target, "object"), ("obstacle_box", "obstacle")])
        sim.hide_unseen_centers = level >= 2
    sim.settle_initial_scene()
    snapshot(sim)
    sim.frame_start = sim.data.time
    box = sim.target_oriented_box(target)
    return {
        "target": target, "destination": destination, "obstacles": obstacles,
        "perception": {"hide_unseen_centers": sim.hide_unseen_centers},
        "constraints": {
            "destination_tolerance_m": (
                (.07, .06, .05, .04)[level] if task == "push"
                else (.04, .035, .03, .025)[level]
            ),
            "avoid_obstacles": task == "grasp" and level == 3,
        },
        "difficulty": {
            "level": level, "name": LEVELS[level], "provenance": "task-difficulty-v1",
            "factors": [FACTORS[task][level]],
            "rejected_camera_poses": pose_rejections,
            "measurements": {
                "initial_target_visible": bool(box[5]),
                "initial_visible_pixels": int(sim.entity_mask(target).sum()),
                "visible_pixels_without_occluder": visible_before_occluder,
                "target_radius_m": float(np.linalg.norm(position[:2])),
                "target_destination_distance_m": float(np.linalg.norm(
                    position[:2] - sim.position(destination)[:2]
                )) if destination else None,
            },
        },
    }


def collect(root: Path, count: int, seed: int, attempts_per_case: int = 3, resume: bool = False) -> dict:
    if not 1 <= count <= 100 or not 1 <= attempts_per_case <= 10:
        raise ValueError("Expected count 1..100 and attempts per case 1..10 (non-overlapping seed ranges)")
    root.mkdir(parents=True, exist_ok=resume)
    report_path = root / "collection.json"
    report = (json.loads(report_path.read_text()) if resume and report_path.exists() else
              {"generator": "task-difficulty-v1", "successful_episodes": [], "attempts": []})
    if report.get("generator") != "task-difficulty-v1":
        raise ValueError("Not a compatible difficulty collection")
    recorded_seed = report.get("seed", report["attempts"][0]["seed"] if report["attempts"] else seed)
    if recorded_seed != seed:
        raise ValueError("Resuming a collection requires its original seed")
    report["seed"] = seed
    completed = {row["seed"] for row in report["attempts"]}
    sim = BeamSimulation(MODEL_ROOT)
    try:
        for task_index, task in enumerate(TASKS):
            for level in range(4):
                for example in range(count):
                    base_seed = seed + task_index * 10000 + level * 1000 + example * 10
                    if any(row["outcome"] == "success" and base_seed <= row["seed"] < base_seed + 10
                           for row in report["attempts"]):
                        continue
                    for attempt in range(attempts_per_case):
                        scene_seed = seed + task_index * 10000 + level * 1000 + example * 10 + attempt
                        if scene_seed in completed:
                            continue
                        source = root / f"{task}_{level}_{example:03}_{attempt}"
                        try:
                            config = configure_scene(sim, task, level, scene_seed)
                        except (ValueError, RuntimeError) as error:
                            report["attempts"].append({
                                "task": task, "difficulty_level": level,
                                "seed": scene_seed, "outcome": "invalid_scene", "error": str(error),
                            })
                            print(json.dumps(report["attempts"][-1]), flush=True)
                            atomic_json(root / "collection.json", report)
                            continue
                        (source / "wrist").mkdir(parents=True, exist_ok=resume)
                        sim.overview(source / "diagnostic_overview.jpg")
                        if not cv2.imwrite(str(source / "initial_wrist.jpg"),
                                           cv2.cvtColor(sim.image(), cv2.COLOR_RGB2BGR)):
                            raise OSError("Could not save initial wrist image")
                        sim.episode_root = source
                        try:
                            result = (sim.demonstrate_obstacle_grasp(config["target"])
                                      if task == "grasp" and level == 3 else
                                      sim.demonstrate_command(task, config["target"], config["destination"]))
                        except (RuntimeError, ValueError) as error:
                            result = {"success": False, "reason": str(error)}
                        if task in ("push", "place"):
                            distance = float(np.linalg.norm(
                                sim.position(config["target"])[:2] -
                                sim.position(config["destination"])[:2]
                            ))
                            result["destination_error_m"] = distance
                            result["success"] = bool(result["success"]) and distance <= config["constraints"]["destination_tolerance_m"]
                            if task == "push":
                                result["success"] &= result.get("displacement_m", 0.) >= .02
                        metadata = {
                            "simulation": True, "model_commit": MODEL_COMMIT,
                            "task": task, "scene_seed": scene_seed, "fps": sim.fps,
                            "frames": len(sim.rows), "initial": sim.initial,
                            "target_label": sim.target_labels[config["target"]],
                            "color_targets": True, **config,
                            "outcome": "success" if result["success"] else "failed",
                            "verification": result,
                        }
                        (source / "frames.jsonl").write_text(
                            "".join(json.dumps(row) + "\n" for row in sim.rows))
                        atomic_json(source / "episode.json", metadata)
                        row = {"source": str(source), "task": task, "difficulty_level": level,
                               "seed": scene_seed, "outcome": metadata["outcome"], "verification": result}
                        report["attempts"].append(row)
                        if result["success"]:
                            report["successful_episodes"].append(str(source))
                        atomic_json(root / "collection.json", report)
                        print(json.dumps(row), flush=True)
                        if result["success"]:
                            break
    finally:
        sim.close()
    report["missing_cases"] = [
        {"task": task, "difficulty_level": level, "example": example}
        for task_index, task in enumerate(TASKS)
        for level in range(4) for example in range(count)
        if not any(row["outcome"] == "success" and
                   seed + task_index * 10000 + level * 1000 + example * 10 <= row["seed"] <
                   seed + task_index * 10000 + level * 1000 + example * 10 + 10
                   for row in report["attempts"])
    ]
    report["requested_per_cell"] = count
    atomic_json(report_path, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/simulation/task_difficulty_v1"))
    parser.add_argument("--count", type=int, default=1, help="Successful examples per task/level (1..100)")
    parser.add_argument("--seed", type=int, default=90000)
    parser.add_argument("--attempts-per-case", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--base-collection", type=Path)
    parser.add_argument("--merged-root", type=Path,
                        help="Write a separate training collection including the base and successful new examples.")
    args = parser.parse_args()
    if bool(args.base_collection) != bool(args.merged_root):
        parser.error("--base-collection and --merged-root must be supplied together")
    if args.merged_root and args.merged_root.resolve() in (args.root.resolve(), args.base_collection.resolve()):
        parser.error("--merged-root must be separate from both source collections")
    result = collect(args.root, args.count, args.seed, args.attempts_per_case, args.resume)
    if result["missing_cases"]:
        print(json.dumps({"incomplete": True, "missing_cases": result["missing_cases"]}))
        raise SystemExit(2)
    if args.merged_root:
        base = json.loads((args.base_collection / "collection.json").read_text())
        args.merged_root.mkdir(parents=True, exist_ok=args.resume)
        atomic_json(args.merged_root / "collection.json", {
            "base_collection": str(args.base_collection), "difficulty_collection": str(args.root),
            "successful_episodes": list(dict.fromkeys(base["successful_episodes"] + result["successful_episodes"])),
        })
    print(json.dumps({"successful": len(result["successful_episodes"]),
                      "attempts": len(result["attempts"])}))
