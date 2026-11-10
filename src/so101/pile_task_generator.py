"""Generate compact native pile setups and many tasks, never policy difficulty scores."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import signal
import threading
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from so101.catalog_sim import CatalogSimulation
from so101.cloud_checkpoint import atomic_json
from so101.curriculum_workers import memory_state
from so101.lego_curriculum import load_catalog, prepare_scene
from so101.lego_parts_catalog import catalog_read_lock
from so101.lego_scene import RGBA, capture_scene_state, settle_data
from so101.pile_task_runtime import GRADER_VERSION, create_simulation, load_problem
from so101.robot_sim import MODEL_ROOT, TABLE_Z
from so101.settled_piles import support_graph


LOGGER = logging.getLogger(__name__)
MINIMUM_FREE_BYTES = 10 * 1024**3
MAXIMUM_POOL_BYTES = 4 * 1024**3


def resource_check(root: Path, maximum_pool_bytes: int = MAXIMUM_POOL_BYTES) -> None:
    stats = os.statvfs(root)
    if stats.f_bavail * stats.f_frsize < MINIMUM_FREE_BYTES:
        raise BlockingIOError("Pile generation requires 10GiB available")
    memory = memory_state()
    if memory["available_bytes"] < 2 * 1024**3:
        raise BlockingIOError(f"Pile generation memory admission paused: {memory}")
    used = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    if used + 512 * 1024**2 > maximum_pool_bytes:
        raise BlockingIOError("Pile pool budget reached; preserving existing problems")


def propose_pile(catalog, seed: int, sequence: int, count: int, color_mode: str) -> dict:
    if not 2 <= count <= 128 or color_mode not in ("monochrome", "few", "many"):
        raise ValueError("Unsupported piece count or color mode")
    rng = np.random.default_rng(np.random.SeedSequence([seed, sequence]))
    selected = rng.choice(len(catalog.parts), size=min(count, len(catalog.parts)), replace=False)
    subset = replace(catalog, parts=tuple(catalog.parts[int(index)] for index in selected))
    plan = prepare_scene(subset, seed=seed, sequence=sequence,
                         category="pile", task="view", object_count=count)
    palette = list(RGBA)
    rng.shuffle(palette)
    palette = palette[:1 if color_mode == "monochrome" else 2 if color_mode == "few" else len(palette)]
    for index, obj in enumerate(plan["objects"]):
        obj["color"] = palette[int(rng.integers(len(palette)))]
        obj["pile_drop_xy_relative_m"] = (
            np.asarray(obj["position_relative_to_scene_m"][:2]) * .35
        ).tolist()
        obj["position_relative_to_scene_m"] = [10. + index, 0., 1.]
    plan["generator"] = "native-pile-tasks-v1"
    plan["drop_strategy"] = "sequential-low-clearance-v1"
    plan["color_mode"] = color_mode
    plan["destinations"] = [
        {"destination_id": "destination_0", "position_relative_to_scene_m": [0., -.14, .00002]},
        {"destination_id": "destination_1", "position_relative_to_scene_m": [0., .14, .00002]},
    ]
    plan["scene_id"] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    plan["pair_id"] = plan["scene_id"]
    plan["commands"] = []
    return plan


def drop_pieces_sequentially(sim) -> None:
    objects = sim.plan["objects"]
    slots = []
    for index, obj in enumerate(objects):
        body = sim.model.body(obj["instance_id"]).id
        joint = int(sim.model.body_jntadr[body])
        qadr = int(sim.model.jnt_qposadr[joint])
        dadr = int(sim.model.jnt_dofadr[joint])
        slots.append((body, qadr, dadr, float(sim.model.body_gravcomp[body])))
        sim.model.body_gravcomp[body] = 1.
        sim.data.qpos[qadr:qadr + 3] = [10. + index, 0., 1.]
        sim.data.qvel[dadr:dadr + 6] = 0.
    active = []
    try:
        for obj, (body, qadr, dadr, gravity) in zip(objects, slots):
            mujoco.mj_forward(sim.model, sim.data)
            active_bodies = [sim.model.body(item["instance_id"]).id for item in active]
            geoms = np.flatnonzero(np.isin(sim.model.geom_bodyid, active_bodies))
            top = TABLE_Z
            if len(geoms):
                bounds = sim.model.geom_aabb[geoms]
                vertical = sim.data.geom_xmat[geoms].reshape(-1, 3, 3)[:, 2, :]
                top = max(top, float(np.max(
                    sim.data.geom_xpos[geoms, 2]
                    + np.sum(vertical * bounds[:, :3], axis=1)
                    + np.sum(np.abs(vertical) * bounds[:, 3:], axis=1)
                )))
            part = sim.part_metadata[obj["part_id"]]
            lower = (part.vertical_bounds_m or (-part.dimensions_m[2] / 2, 0.))[0]
            relative = obj.get("pile_drop_xy_relative_m", obj["position_relative_to_scene_m"])
            sim.data.qpos[qadr:qadr + 3] = [
                sim.scene_origin[0] + relative[0],
                sim.scene_origin[1] + relative[1], top - lower + .003,
            ]
            sim.data.qvel[dadr:dadr + 6] = 0.
            sim.model.body_gravcomp[body] = gravity
            active.append(obj)
            settle_data(
                sim.model, sim.data, sim.xml, {**sim.plan, "objects": list(active)},
                contact_solver="pyramidal-v1", stationarity_metric="surface-speed-v2",
            )
    finally:
        for body, _, _, gravity in slots:
            sim.model.body_gravcomp[body] = gravity


def describe_scene(sim, plan: dict) -> tuple[list[dict], list[dict], dict]:
    # mj_step leaves derived poses at the pre-integration state.
    mujoco.mj_forward(sim.model, sim.data)
    parts = {part.part_id: part for part in sim.catalog.parts}
    sizes = [parts[obj["part_id"]].dimensions_m for obj in plan["objects"]]
    instances, tasks = [], []
    for obj in plan["objects"]:
        identity = obj["instance_id"]
        body = sim.model.body(identity).id
        instances.append({
            "instance_id": identity, "part_id": obj["part_id"], "color": obj["color"],
            "body_name": identity, "geom_names": [
                sim.model.geom(int(index)).name
                for index in np.flatnonzero(sim.model.geom_bodyid == body)
            ],
        })
        position = sim.position(identity).tolist()
        overlying = [
            other["instance_id"] for other in plan["objects"]
            if other["instance_id"] != identity
            and sim.position(other["instance_id"])[2] > position[2]
            and np.linalg.norm(sim.position(other["instance_id"])[:2] - position[:2])
            < .5 * (np.linalg.norm(parts[obj["part_id"]].dimensions_m[:2])
                    + np.linalg.norm(parts[other["part_id"]].dimensions_m[:2]))
        ]
        for task in ("view", "grasp", "push", "place"):
            goal = {"dwell_steps": 10}
            if task == "view":
                goal.update(minimum_pixels=64, center_radius=.20)
            elif task == "grasp":
                goal["lift_m"] = .05
            else:
                destination = "goal_left" if position[1] >= 0 else "goal_right"
                goal.update(destination_entity=destination,
                            destination_xy=sim.position(destination)[:2].tolist(),
                            tolerance_m=.025, maximum_surface_speed_m_s=.01)
            tasks.append({
                "task_id": f"{plan['scene_id']}:{identity}:{task}",
                "task_type": task, "target_instance_id": identity,
                "initial_target_position": position, "goal": goal,
                "grader_version": GRADER_VERSION,
                "instruction": f"{task.upper()} the piece identified by instance {identity}.",
                "solvability": "unmeasured",
                "target_tags": {
                    "part_dimensions_m": list(parts[obj["part_id"]].dimensions_m),
                    "overlying_body_center_candidates": overlying,
                    "burial_proxy": "higher-body-center within summed bounding radii, not measured visibility",
                },
                "potential_occluder_instance_ids": [
                    other["instance_id"] for other in plan["objects"]
                    if other["instance_id"] != identity
                ],
            })
    dimensions = np.asarray(sizes)
    positions = np.asarray([sim.position(obj["instance_id"]) for obj in plan["objects"]])
    return instances, tasks, {
        "piece_count": len(instances),
        "unique_piece_count": len({obj["part_id"] for obj in plan["objects"]}),
        "unique_color_count": len({obj["color"] for obj in plan["objects"]}),
        "color_mode": plan["color_mode"],
        "color_counts": dict(Counter(obj["color"] for obj in plan["objects"])),
        "mean_piece_dimensions_m": dimensions.mean(axis=0).tolist(),
        "piece_dimensions_variance_m2": dimensions.var(axis=0).tolist(),
        "mean_piece_volume_proxy_m3": float(np.prod(dimensions, axis=1).mean()),
        "piece_volume_proxy_variance_m6": float(np.prod(dimensions, axis=1).var()),
        "body_center_extent_m": np.ptp(positions, axis=0).tolist(),
        "policy_difficulty_measured": False, "initial_visibility_measured": False,
    }


def add_settled_view_walls(sim, plan: dict) -> None:
    mujoco.mj_forward(sim.model, sim.data)
    bodies = {sim.model.body(obj["instance_id"]).id for obj in plan["objects"]}
    geoms = np.flatnonzero(np.isin(sim.model.geom_bodyid, list(bodies)))
    centers = sim.data.geom_xpos[geoms]
    radii = sim.model.geom_rbound[geoms]
    lower = np.min(centers - radii[:, None], axis=0)
    upper = np.max(centers + radii[:, None], axis=0)
    tree = ET.fromstring(sim.xml)
    for index, lateral in enumerate((lower[1] - .011, upper[1] + .011)):
        name = f"pile_view_obstruction_{index}"
        position = [(lower[0] + upper[0]) / 2, lateral, TABLE_Z + .045]
        sim.model.geom_pos[sim.model.geom(name).id] = position
        node = tree.find(f".//geom[@name='{name}']")
        node.set("pos", " ".join(map(str, position)))
    sim.xml = ET.tostring(tree, encoding="unicode")
    mujoco.mj_forward(sim.model, sim.data)
    panels = {sim.model.geom(f"pile_view_obstruction_{i}").id for i in range(2)}
    for contact in sim.data.contact:
        if contact.dist < -.000001 and panels.intersection((contact.geom1, contact.geom2)):
            raise ValueError("Settled VIEW wall intersects existing scene")


def generate_scene(catalog_path: Path, output: Path, *, seed: int, sequence: int,
                   count: int, color_mode: str, model_root: Path = MODEL_ROOT) -> dict:
    sim = None
    # Binary models embed meshes: no catalog-path lifetime dependency after compilation.
    with catalog_read_lock(catalog_path) as manifest:
        catalog = load_catalog(manifest)
        plan = propose_pile(catalog, seed, sequence, count, color_mode)
        with TemporaryDirectory(prefix="pile-task-model-") as directory:
            robot_root = Path(directory)
            robot = ET.parse(model_root / "so101.xml")
            size = robot.getroot().find("size")
            if size is None:
                size = ET.SubElement(robot.getroot(), "size")
            size.attrib.clear()
            size.set("memory", "256M")
            world = robot.getroot().find("worldbody")
            if world is None:
                raise ValueError("Robot model lacks worldbody")
            for index in range(2):
                ET.SubElement(world, "geom", {
                    "name": f"pile_view_obstruction_{index}", "type": "box",
                    "pos": f".22 0 {-10 - index}",
                    "size": ".035 .006 .045", "rgba": ".25 .25 .25 1",
                })
            robot.write(robot_root / "so101.xml")
            for name in ("revision.json", "assets"):
                (robot_root / name).symlink_to((model_root / name).resolve())
            try:
                sim = CatalogSimulation(
                    catalog, plan, model_root=robot_root, render=False, record_images=False,
                    contact_solver="pyramidal-v1", stationarity_metric="surface-speed-v2",
                    initial_arm_position=(.14, -.12, .09),
                    initialize_objects=drop_pieces_sequentially,
                )
            except mujoco.FatalError as error:
                if "mj_stackAlloc: out of memory" not in str(error):
                    raise
                raise ValueError(f"Pile constraint arena exhausted: {error}") from error
    try:
        add_settled_view_walls(sim, plan)
        instances, tasks, tags = describe_scene(sim, plan)
        support = support_graph(sim.model, sim.data, list(sim.object_metadata))
        if support["unsupported_objects"] or not support["object_object_edges"]:
            raise ValueError("Settled proposal is not a table-connected contacting pile")
        tags["contact_graph"] = support
        tags["view_obstructions"] = {
            "kind": "two-post-settling-opaque-panels", "count": 2,
            "initial_target_occlusion_measured": False,
        }
        scene = {
            "schema_version": 1, "scene_id": plan["scene_id"],
            "scene_family_id": hashlib.sha256(f"pile-v1:{seed}:{sequence}".encode()).hexdigest(),
            "catalog_sha256": catalog.manifest_sha256, "seed": seed, "sequence": sequence,
            "instances": instances, "tasks": tasks, "tags": tags,
            "capabilities": {"reset": "mujoco-integration-v1",
                             "target_labels": "instance-segmentation-v1",
                             "expert_references": False, "sft_qualified": False},
            "physics_version": mujoco.__version__,
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "grader_sha256": hashlib.sha256(
                Path(__file__).with_name("pile_task_runtime.py").read_bytes()
            ).hexdigest(),
            "source_assets": {part.part_id: dict(part.assets) for part in catalog.parts
                              if part.part_id in {obj["part_id"] for obj in plan["objects"]}},
        }
        if mujoco.mj_sizeModel(sim.model) > 512 * 1024**2:
            raise BlockingIOError("Compiled pile exceeds 512MiB admitted model reserve")
        output.mkdir(exist_ok=False)
        (output / "model.xml").write_text(sim.xml)
        mujoco.mj_saveModel(sim.model, str(output / "model.mjb"), None)
        atomic_json(output / "integration_state.json",
                    capture_scene_state(sim.model, sim.data, sim.xml, plan, status="settled"))
        atomic_json(output / "scene_plan.json", plan)
        atomic_json(output / "problem.json", scene)
        entry = {
            "scene_id": scene["scene_id"], "scene_family_id": scene["scene_family_id"],
            "split": "train", "root": str(output.resolve()), "tags": tags,
            "files_sha256": {
                name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                for name in ("model.xml", "model.mjb", "integration_state.json",
                             "scene_plan.json", "problem.json")
            },
        }
        restored = create_simulation(load_problem(entry, tasks[0]["task_id"]))
        try:
            for name in ("qpos", "qvel", "ctrl", "act"):
                if not np.array_equal(getattr(restored.data, name), getattr(sim.data, name)):
                    raise ValueError(f"Native pile reset changed {name}")
            for task in tasks:
                if not np.allclose(restored.position(task["target_instance_id"]),
                                   task["initial_target_position"], atol=1e-10, rtol=0):
                    raise ValueError("Native reset changed target center of mass")
                if task["task_type"] in ("push", "place"):
                    goal = task["goal"]
                    marker = restored.data.body(goal["destination_entity"]).xpos[:2]
                    if not np.allclose(marker, goal["destination_xy"], atol=1e-10, rtol=0):
                        raise ValueError("Native goal does not match its visible destination")
        finally:
            restored.close()
        atomic_json(output / "admission.json", entry)
        return entry
    finally:
        if sim is not None:
            sim.close()


def candidate_rejection(error: ValueError) -> bool:
    return str(error).startswith((
        "Scene contact penetration exceeded limit:",
        "Non-finite catalog scene physics",
        "MuJoCo emitted a warning while settling catalog scene",
        "Catalog scene did not settle before physics timeout",
        "Settled proposal is not a table-connected contacting pile",
        "Pile constraint arena exhausted:",
        "Settled VIEW wall intersects existing scene",
    ))


def recover_entry(output: Path) -> dict | None:
    if not output.exists():
        return None
    admission = output / "admission.json"
    if not admission.is_file():
        raise BlockingIOError(f"Interrupted scene retained without admission: {output}")
    entry = json.loads(admission.read_text())
    scene = json.loads((output / "problem.json").read_text())
    load_problem(entry, scene["tasks"][0]["task_id"])
    return entry


def _run(root: Path, catalog: Path, *, seed: int, attempts: int,
         stopping: threading.Event) -> None:
    root.mkdir(parents=True, exist_ok=True)
    stopping = stopping or threading.Event()
    sequence = 0
    completed = 0
    while not stopping.is_set() and (not attempts or completed < attempts):
        while (root / f"attempt_{sequence:09d}.json").exists():
            sequence += 1
        if (root / "STOP").exists():
            return
        try:
            resource_check(root)
        except BlockingIOError as error:
            atomic_json(root / "status.json", {"status": "paused", "reason": str(error)})
            stopping.wait(5)
            continue
        counts = (3, 5, 8, 12, 16, 24, 32, 48, 64)
        count = counts[(sequence // 16) % len(counts)]
        mode = ("monochrome", "few", "many")[sequence % 3]
        receipt = {"sequence": sequence, "piece_count": count, "color_mode": mode}
        try:
            output = root / f"scene_{sequence:09d}"
            entry = recover_entry(output)
            if entry is None:
                entry = generate_scene(catalog, output, seed=seed, sequence=sequence,
                                       count=count, color_mode=mode)
            receipt.update(status="complete", scene=entry)
        except BlockingIOError as error:
            atomic_json(root / "status.json", {"status": "blocked", "reason": str(error)})
            raise
        except ValueError as error:
            if not candidate_rejection(error):
                raise
            LOGGER.warning("Pile proposal %s rejected: %s", sequence, error)
            receipt.update(status="rejected", reason=str(error))
        atomic_json(root / f"attempt_{sequence:09d}.json", receipt)
        completed += 1
        sequence += 1


def run(root: Path, catalog: Path, *, seed: int, attempts: int = 0,
        stopping: threading.Event | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".generator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _run(root, catalog, seed=seed, attempts=attempts,
             stopping=stopping or threading.Event())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--attempts", type=int, default=0, help="0 continuously replenishes the pool")
    args = parser.parse_args()
    if args.attempts < 0:
        parser.error("--attempts must be nonnegative")
    os.nice(19)
    stopping = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopping.set())
    signal.signal(signal.SIGINT, lambda *_: stopping.set())
    run(args.root, args.catalog, seed=args.seed, attempts=args.attempts, stopping=stopping)


if __name__ == "__main__":
    main()
