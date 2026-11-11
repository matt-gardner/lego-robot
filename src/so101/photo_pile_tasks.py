"""Attach robot actions and persistent task goals to immutable photo-style piles."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from so101.cloud_checkpoint import atomic_json
from so101.lego_scene import capture_scene_state, settle_data
from so101.photo_pile_generator import CATALOG_PROFILE, PROFILE, verify_scene
from so101.pile_pool import publish_pool
from so101.pile_task_runtime import GRADER_VERSION, create_simulation, load_problem
from so101.robot_sim import (
    BeamSimulation, DISTRACTOR_SIZES, MODEL_ROOT, OBJECT_NAMES, SCENE_VERSION,
    TABLE_Z, scene_xml,
)

START_PROFILE = "independent-robot-starts-v1"


def select_current_pools(root: Path, paths: list[Path]) -> dict:
    pools, families = [], set()
    if not paths:
        raise ValueError("Current task selection cannot be empty")
    for path in paths:
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Current pool must be contained in the curriculum root")
        content = path.read_bytes()
        manifest = json.loads(content)
        if (manifest.get("kind") != "lego-pile-task-pool"
                or manifest.get("status") != "complete" or not manifest["scenes"]):
            raise ValueError("Current selection requires completed task pools")
        task_count, piece_counts = 0, set()
        for entry in manifest["scenes"]:
            scene = json.loads((Path(entry["root"]) / "problem.json").read_text())
            load_problem(entry, scene["tasks"][0]["task_id"])
            if entry["scene_family_id"] in families or any(
                task.get("start", {}).get("profile") != START_PROFILE
                or "initial_state" not in task for task in scene["tasks"]
            ):
                raise ValueError("Current selection duplicates families or lacks per-task starts")
            families.add(entry["scene_family_id"])
            task_count += len(scene["tasks"])
            piece_counts.add(len(scene["instances"]))
        pools.append({"manifest_path": str(path.resolve()),
                      "sha256": hashlib.sha256(content).hexdigest(),
                      "scenes": len(manifest["scenes"]), "task_count": task_count,
                      "piece_counts": sorted(piece_counts)})
    selection = {"schema_version": 1, "kind": "lego-current-task-pools",
                 "task_start_profile": START_PROFILE, "pools": pools}
    atomic_json(root / "current-task-pools.json", selection)
    return selection


def assign_task_starts(sim: BeamSimulation, scene: dict, snapshot: dict, xml: str) -> None:
    """Sample collision-free arm configurations without moving the settled pile."""
    model = sim.model
    specification = mujoco.mjtState.mjSTATE_INTEGRATION
    baseline = np.asarray(snapshot["integration_state"])
    home = np.asarray(snapshot["qpos"])[sim.qadr]
    robot_bodies = set()
    for joint in sim.joints:
        body = int(model.jnt_bodyid[joint])
        robot_bodies.add(body)
        robot_bodies.update(
            index for index in range(model.nbody)
            if model.body_rootid[index] == model.body_rootid[body]
        )
    mujoco.mj_setState(model, sim.data, baseline, specification)
    mujoco.mj_forward(model, sim.data)
    baseline_contacts = {
        tuple(sorted((int(c.geom1), int(c.geom2)))) for c in sim.data.contact
        if c.dist < .002
    }
    positions = []
    scene["provenance"]["task_start_profile"] = START_PROFILE
    scene["scene_id"] = hashlib.sha256(
        json.dumps(scene["provenance"], sort_keys=True).encode()).hexdigest()
    plan = {"scene_id": scene["scene_id"], "catalog_sha256": scene["catalog_sha256"]}
    for index, task in enumerate(scene["tasks"]):
        seed = [scene["seed"], index, 731, int(scene["scene_id"][:16], 16)]
        rng = np.random.default_rng(np.random.SeedSequence(seed))
        for attempt in range(500):
            mujoco.mj_setState(model, sim.data, baseline, specification)
            q = home + rng.uniform(-1., 1., len(home)) * np.array([.65, .4, .45, .45, 1., .15])
            if np.any(q < sim.limits[:, 0]) or np.any(q > sim.limits[:, 1]):
                continue
            sim.data.qpos[sim.qadr] = q
            sim.data.qvel[sim.dofs] = 0.
            sim.data.qacc_warmstart[sim.dofs] = 0.
            sim.data.ctrl[sim.actuators] = q
            mujoco.mj_forward(model, sim.data)
            position = sim.data.site_xpos[sim.site].copy()
            if (not np.isfinite(position).all() or position[2] < .06
                    or not .07 < position[0] < .34 or abs(position[1]) > .23
                    or any(np.linalg.norm(position - other) < .005 for other in positions)):
                continue
            if any(
                c.dist < .002
                and tuple(sorted((int(c.geom1), int(c.geom2)))) not in baseline_contacts
                and any(int(model.geom_bodyid[g]) in robot_bodies for g in (c.geom1, c.geom2))
                for c in sim.data.contact
            ):
                continue
            state = capture_scene_state(model, sim.data, xml, plan, status="task-start")
            task["initial_state"] = state
            task["start"] = {
                "profile": START_PROFILE, "seed": seed,
                "proposal": attempt, "robot_joint_positions": q.tolist(),
                "tool_position_m": position.tolist(),
                "construction": "independent bounded joint sampling; collision and workspace checked",
                "obstacle_challenge_constructed": False,
            }
            task["task_id"] = f'{scene["scene_id"]}:{task["target_instance_id"]}:{task["task_type"]}'
            positions.append(position)
            break
        else:
            raise ValueError(f"No valid distinct robot start for task {index} after 500 proposals")
    scene["tags"]["task_start_profile"] = START_PROFILE
    mujoco.mj_setState(model, sim.data, baseline, specification)
    mujoco.mj_forward(model, sim.data)


def retask_scene(entry: dict, output: Path, model_root: Path = MODEL_ROOT) -> dict:
    """Publish a new task revision, retaining the original pile/model and family."""
    if output.exists():
        existing = json.loads((output / "admission.json").read_text())
        scene = json.loads((output / "problem.json").read_text())
        if (scene["provenance"].get("parent_problem_sha256")
                != entry["files_sha256"]["problem.json"]
                or scene["tags"].get("task_start_profile") != START_PROFILE
                or scene["generator_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
                or any(task.get("start", {}).get("profile") != START_PROFILE
                       for task in scene["tasks"])):
            raise ValueError("Existing task revision has different provenance")
        load_problem(existing, scene["tasks"][0]["task_id"])
        return existing
    source = Path(entry["root"])
    scene = json.loads((source / "problem.json").read_text())
    load_problem(entry, scene["tasks"][0]["task_id"])
    model = mujoco.MjModel.from_binary_path(str(source / "model.mjb"))
    force_ranges = model.actuator_forcerange.copy()
    sim = BeamSimulation(model_root, model=model, render=False,
                         object_names=tuple(item["instance_id"] for item in scene["instances"]),
                         distractor_names=(), record_images=False)
    model.actuator_forcerange[:] = force_ranges
    try:
        snapshot = json.loads((source / "integration_state.json").read_text())
        xml = (source / "model.xml").read_text()
        if "provenance" not in scene:
            scene["provenance"] = {"parent_scene_id": scene["scene_id"]}
        scene["provenance"]["parent_problem_sha256"] = entry["files_sha256"]["problem.json"]
        assign_task_starts(sim, scene, snapshot, xml)
        scene["generator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        scene["grader_sha256"] = hashlib.sha256(
            Path(__file__).with_name("pile_task_runtime.py").read_bytes()).hexdigest()
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".retask-", dir=output.parent) as directory:
            staging = Path(directory) / "scene"
            shutil.copytree(source, staging)
            atomic_json(staging / "problem.json", scene)
            result = {**entry, "scene_id": scene["scene_id"], "root": str(staging.resolve()),
                      "tags": scene["tags"], "files_sha256": {
                          name: hashlib.sha256((staging / name).read_bytes()).hexdigest()
                          for name in entry["files_sha256"]}}
            checked = create_simulation(load_problem(result, scene["tasks"][0]["task_id"]))
            checked.close()
            result["root"] = str(output.resolve())
            atomic_json(staging / "admission.json", result)
            staging.rename(output)
        return result
    finally:
        sim.close()


def build_problem(source: Path, output: Path, model_root: Path = MODEL_ROOT) -> dict:
    if output.exists():
        entry = json.loads((output / "admission.json").read_text())
        scene = json.loads((output / "problem.json").read_text())
        expected = hashlib.sha256((source / "receipt.json").read_bytes()).hexdigest()
        if scene["provenance"]["receipt_sha256"] != expected:
            raise ValueError("Existing task scene has different source provenance")
        for task in scene["tasks"]:
            load_problem(entry, task["task_id"])
        return entry
    receipt = verify_scene(source)
    if receipt.get("profile") not in (PROFILE, CATALOG_PROFILE) or receipt["restored_original_integration_state"]:
        raise ValueError("Tasks require a verified fresh photo-profile pile")
    photo = ET.parse(source / "model.xml").getroot()
    robot = ET.fromstring(scene_xml(model_root, SCENE_VERSION))
    world = robot.find("worldbody")
    assets = robot.find("asset")
    if world is None or assets is None:
        raise ValueError("Robot scene lacks worldbody/assets")
    for body in list(world.findall("body")):
        if body.get("name") in (*OBJECT_NAMES, *DISTRACTOR_SIZES):
            world.remove(body)
    photo_assets = photo.find("asset")
    if photo_assets is not None:
        names = {node.get("name") for node in assets}
        for mesh in photo_assets.findall("mesh"):
            if mesh.get("name") in names:
                raise ValueError("Photo/robot mesh identity conflict")
            assets.append(copy.deepcopy(mesh))
    table = world.find("geom[@name='table']")
    photo_world = photo.find("worldbody")
    if photo_world is None or table is None:
        raise ValueError("Missing photo world or robot table")
    photo_table = photo_world.find("geom[@type='plane']")
    if photo_table is None or float(photo_table.get("pos", "0 0 0").split()[2]) != 0:
        raise ValueError("Photo pile must use the original zero-height plane")
    for attribute in ("solref", "solimp", "priority", "friction"):
        if attribute in photo_table.attrib:
            table.set(attribute, photo_table.attrib[attribute])
        elif attribute == "friction":
            table.set("friction", "1 .005 .0001")
    robot_option = robot.find("option")
    photo_option = photo.find("option")
    if robot_option is None or photo_option is None:
        raise ValueError("Missing source physics options")
    robot_option.attrib.clear()
    robot_option.attrib.update(photo_option.attrib)
    origin = np.array([.22, 0., TABLE_Z])
    for name, y in (("goal_left", -.14), ("goal_right", .14)):
        marker = world.find(f"body[@name='{name}']")
        if marker is None:
            raise ValueError("Robot scene lacks destination marker")
        marker.set("pos", f".22 {y} {TABLE_Z + .00002}")
    for body in photo_world.findall("body"):
        world.append(copy.deepcopy(body))
    size = robot.find("size")
    if size is None:
        size = ET.SubElement(robot, "size")
    size.attrib.clear()
    size.set("memory", "512M" if receipt["profile"] == CATALOG_PROFILE else "256M")
    xml = ET.tostring(robot, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    names = tuple(obj["instance_id"] for obj in receipt["objects"])
    sim = BeamSimulation(model_root, model=model, render=False,
                         object_names=names, distractor_names=(), record_images=False)
    try:
        home = sim.ik(np.array([.14, -.12, .09]), 0.)
        home[-1] = sim.limits[-1, 0] + .55 * np.ptp(sim.limits[-1])
        sim.data.qpos[sim.qadr] = home
        sim.data.ctrl[sim.actuators] = home
        for obj in receipt["objects"]:
            body = model.body(obj["instance_id"]).id
            joint = int(model.body_jntadr[body])
            q, v = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
            sim.data.qpos[q:q + 7] = obj["qpos"]
            sim.data.qpos[q:q + 3] += origin
            sim.data.qvel[v:v + 6] = obj["qvel"]
        provenance = {
            "source": str(source.resolve()),
            "receipt_sha256": hashlib.sha256((source / "receipt.json").read_bytes()).hexdigest(),
            "source_model_sha256": receipt["model_xml_sha256"],
            "translation_m": origin.tolist(), "profile": receipt["profile"],
            "robot_integration": "source poses translated onto robot table, then full dynamic settling",
        }
        scene_id = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
        family = hashlib.sha256(f"photo-pile:{receipt['seed']}:{len(names)}".encode()).hexdigest()
        plan = {"scene_id": scene_id, "catalog_sha256": receipt["catalog_sha256"],
                "seed": receipt["seed"], "objects": [
                    {"instance_id": obj["instance_id"], "part_id": obj["part_id"]}
                    for obj in receipt["objects"]]}
        _, _, snapshot = settle_data(
            model, sim.data, xml, plan, velocity_tolerance=.0005, contact_limit_m=.005,
            stationarity_metric="surface-speed-v2",
        )
        mujoco.mj_forward(model, sim.data)
        bounds = {}
        for name in names:
            body_id = model.body(name).id
            geoms = np.flatnonzero(model.geom_bodyid == body_id)
            aabb = model.geom_aabb[geoms]
            rotation = sim.data.geom_xmat[geoms].reshape(-1, 3, 3)
            centers = sim.data.geom_xpos[geoms] + np.einsum("nij,nj->ni", rotation, aabb[:, :3])
            half = np.einsum("nij,nj->ni", np.abs(rotation), aabb[:, 3:])
            bounds[name] = ((centers - half).min(axis=0), (centers + half).max(axis=0))
        instances, tasks = [], []
        for name in names:
            body = model.body(name).id
            instances.append({
                "instance_id": name, "body_name": name,
                "part_id": next(obj["part_id"] for obj in receipt["objects"]
                                if obj["instance_id"] == name),
                "geom_names": [model.geom(int(g)).name
                               for g in np.flatnonzero(model.geom_bodyid == body)],
            })
            position = sim.data.xipos[body].tolist()
            lower, upper = bounds[name]
            overlying = [
                other for other in names if other != name
                and sim.data.xipos[model.body(other).id, 2] > position[2] + .001
                and np.all(np.minimum(upper[:2], bounds[other][1][:2])
                           > np.maximum(lower[:2], bounds[other][0][:2]))
            ]
            target_tags = {
                "vertical_center_m": position[2],
                "height_rank_from_top": 1 + sum(
                    int(sim.data.xipos[model.body(other).id, 2] > position[2]) for other in names),
                "overlying_aabb_candidates": overlying,
                "depth_stratum_proxy": "buried_candidate" if overlying else "top_candidate",
                "depth_method": "higher center plus overlapping world AABBs; not measured visibility",
            }
            for kind in ("view", "grasp", "push", "place"):
                goal = {"dwell_steps": 10}
                if kind == "view":
                    goal.update(minimum_pixels=64, center_radius=.20)
                elif kind == "grasp":
                    goal.update(lift_m=.05)
                else:
                    destination = "goal_left" if position[1] >= 0 else "goal_right"
                    goal.update(destination_entity=destination,
                                destination_xy=sim.data.body(destination).xpos[:2].tolist(),
                                tolerance_m=.025, maximum_surface_speed_m_s=.01)
                tasks.append({
                    "task_id": f"{scene_id}:{name}:{kind}", "task_type": kind,
                    "target_instance_id": name, "initial_target_position": position,
                    "goal": goal, "grader_version": GRADER_VERSION,
                    "instruction": f"{kind.upper()} the piece identified by instance {name}.",
                    "solvability": "unmeasured",
                    "target_tags": target_tags,
                })
        tags = {"piece_count": len(names), "collision_profile": receipt["profile"],
                "unique_piece_count": len({obj["part_id"] for obj in receipt["objects"]}),
                "color_mode": receipt.get("color_mode", "original"),
                "unique_color_count": len({tuple(model.geom_rgba[model.body_geomadr[
                    model.body(name).id]]) for name in names}),
                "policy_difficulty_measured": False, "initial_visibility_measured": False}
        scene = {
            "schema_version": 1, "scene_id": scene_id, "scene_family_id": family,
            "seed": receipt["seed"], "catalog_sha256": receipt["catalog_sha256"],
            "instances": instances, "tasks": tasks, "tags": tags, "provenance": provenance,
            "capabilities": {"reset": "mujoco-integration-v1",
                             "target_labels": "instance-segmentation-v1",
                             "expert_references": False, "sft_qualified": False},
            "qualification": receipt["qualification"],
            "physics_version": mujoco.__version__,
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "grader_sha256": hashlib.sha256(
                Path(__file__).with_name("pile_task_runtime.py").read_bytes()).hexdigest(),
        }
        assign_task_starts(sim, scene, snapshot, xml)
        scene_id = scene["scene_id"]
        with TemporaryDirectory(prefix=".photo-tasks-", dir=output.parent) as directory:
            staging = Path(directory) / "scene"
            staging.mkdir()
            (staging / "model.xml").write_text(xml)
            mujoco.mj_saveModel(model, str(staging / "model.mjb"), None)
            for filename, value in (("integration_state.json", snapshot),
                                    ("scene_plan.json", plan), ("problem.json", scene)):
                atomic_json(staging / filename, value)
            entry = {
                "scene_id": scene_id, "scene_family_id": family, "split": "train",
                "root": str(staging.resolve()), "tags": tags,
                "files_sha256": {name: hashlib.sha256((staging / name).read_bytes()).hexdigest()
                                 for name in ("model.xml", "model.mjb", "integration_state.json",
                                              "scene_plan.json", "problem.json")},
            }
            for task in tasks[:1]:
                problem = load_problem(entry, task["task_id"])
                restored = create_simulation(problem)
                try:
                    if not np.array_equal(restored.data.qpos, task["initial_state"]["qpos"]):
                        raise ValueError("Robot task reset changed task-specific poses")
                    if not np.array_equal(restored.data.qvel, task["initial_state"]["qvel"]):
                        raise ValueError("Robot task reset changed velocities")
                    for target in tasks:
                        if not np.allclose(
                            restored.position(target["target_instance_id"]),
                            target["initial_target_position"], atol=1e-10, rtol=0,
                        ):
                            raise ValueError("Task target differs from reset center of mass")
                finally:
                    restored.close()
            entry["root"] = str(output.resolve())
            atomic_json(staging / "admission.json", entry)
            staging.rename(output)
        return entry
    finally:
        sim.close()


def select_scenes(root: Path, count: int, pieces: int) -> list[Path]:
    selected, identities = [], set()
    for path in sorted(root.glob("scene_*/receipt.json")):
        receipt = json.loads(path.read_text())
        identity = receipt["model_xml_sha256"]
        if receipt["piece_count"] != pieces or identity in identities:
            continue
        verify_scene(path.parent)
        selected.append(path.parent)
        identities.add(identity)
        if len(selected) == count:
            return selected
    raise ValueError(f"Need {count} distinct {pieces}-piece scenes; found {len(selected)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scenes", type=int, default=10)
    parser.add_argument("--pieces", type=int, default=20)
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--retask-pool", type=Path)
    parser.add_argument("--select-pools", type=Path, nargs="+")
    args = parser.parse_args()
    if args.scenes <= 0 or args.pieces < 20:
        parser.error("Require a positive scene quota and at least 20 pieces")
    if args.select_pools is not None:
        print(json.dumps(select_current_pools(args.root, args.select_pools)), flush=True)
        return
    if args.retask_pool is not None:
        manifest = json.loads(args.retask_pool.read_text())
        args.root.mkdir(parents=True, exist_ok=True)
        for index, entry in enumerate(manifest["scenes"]):
            retask_scene(entry, args.root / f"scene_{index:09d}", args.model_root)
        print(json.dumps(publish_pool(args.root)), flush=True)
        return
    if args.source_root is None:
        parser.error("Require --source-root or --retask-pool")
    candidates = sorted(args.source_root.glob("scene_*/receipt.json"))
    sources, identities = [], set()
    for candidate in candidates:
        receipt = json.loads(candidate.read_text())
        if receipt["piece_count"] == args.pieces and receipt["model_xml_sha256"] not in identities:
            sources.append(candidate.parent)
            identities.add(receipt["model_xml_sha256"])
    args.root.mkdir(parents=True, exist_ok=True)
    accepted = []
    for source in sources:
        if (args.root / f"rejected-{source.name}.json").exists():
            continue
        index = len(accepted)
        try:
            entry = build_problem(source, args.root / f"scene_{index:09d}", args.model_root)
        except ValueError as error:
            if not str(error).startswith((
                "Scene contact penetration exceeded limit:", "Catalog scene did not settle",
                "Non-finite catalog scene physics", "MuJoCo emitted a warning",
            )):
                raise
            print(json.dumps({"source": str(source), "status": "rejected",
                              "reason": str(error)}), flush=True)
            atomic_json(args.root / f"rejected-{source.name}.json",
                        {"source": str(source), "reason": str(error)})
            continue
        accepted.append(entry)
        print(json.dumps({"scene": index, "source": str(source),
                          "task_count": entry["tags"]["piece_count"] * 4}), flush=True)
        if len(accepted) == args.scenes:
            break
    if len(accepted) != args.scenes:
        raise ValueError(f"Only {len(accepted)} robot-integrated scenes passed; need {args.scenes}")
    print(json.dumps(publish_pool(args.root)), flush=True)


if __name__ == "__main__":
    main()
