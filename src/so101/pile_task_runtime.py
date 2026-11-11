"""Native pile resets, persistent instance labels, and reference-free task graders."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from so101.lego_scene import body_surface_speed_bound, restore_scene
from so101.robot_table_clearance import require_robot_above_table


GRADER_VERSION = "native-pile-goals-v1"
TASKS = ("view", "grasp", "push", "place")
MAXIMUM_ASSISTANCE_PRELOAD_N = 1.
LABEL_RENDERER_PROFILE = "native-id-segmentation-no-msaa-v1"


def load_problem(entry: dict, task_id: str) -> dict:
    root = Path(entry["root"])
    if not root.is_absolute() or root.is_symlink():
        raise ValueError("Problem root must be an absolute non-symlink path")
    required = {"model.mjb", "model.xml", "integration_state.json", "problem.json", "scene_plan.json"}
    if set(entry["files_sha256"]) != required:
        raise ValueError("Problem manifest must bind all reset, identity, and task files")
    for name, expected in entry["files_sha256"].items():
        path = root / name
        if (Path(name).is_absolute() or ".." in Path(name).parts or path.is_symlink()
                or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected):
            raise ValueError(f"Missing or changed problem file: {name}")
    scene = json.loads((root / "problem.json").read_text())
    if scene["scene_id"] != entry["scene_id"] or scene["scene_family_id"] != entry["scene_family_id"]:
        raise ValueError("Problem scene/family differs from admission entry")
    tasks = [task for task in scene["tasks"] if task["task_id"] == task_id]
    if len(tasks) != 1 or tasks[0]["grader_version"] != GRADER_VERSION:
        raise ValueError("Unknown task or unsupported grader")
    if tasks[0]["task_type"] not in TASKS:
        raise ValueError("Unsupported native pile task")
    identities = [item["instance_id"] for item in scene["instances"]]
    if len(set(identities)) != len(identities) or tasks[0]["target_instance_id"] not in identities:
        raise ValueError("Invalid task instance identity")
    if len({item["body_name"] for item in scene["instances"]}) != len(identities):
        raise ValueError("Multiple instances refer to the same body")
    return {"root": root, "scene": scene, "task": tasks[0]}


@dataclass
class PileSimulation:
    model: mujoco.MjModel
    data: mujoco.MjData
    renderer: mujoco.Renderer | None
    registry: dict[str, int]
    lifted_once: bool = False
    label_renderer: mujoco.Renderer | None = None

    def close_labels(self) -> None:
        if self.label_renderer is not None:
            if self.label_renderer._gl_context is not None:
                self.label_renderer._gl_context.make_current()
            self.label_renderer.close()
            self.label_renderer = None

    def close(self) -> None:
        self.close_labels()
        if self.renderer is not None:
            if self.renderer._gl_context is not None:
                self.renderer._gl_context.make_current()
            self.renderer.close()

    def position(self, instance: str) -> np.ndarray:
        return self.data.xipos[self.registry[instance]].copy()

    def segmentation_image(self) -> np.ndarray:
        if self.renderer is None:
            raise ValueError("Instance labels require a rendering backend")
        if self.label_renderer is None:
            samples = self.model.vis.quality.offsamples
            try:
                # Resolving multisampled ID colors can invent other valid object IDs.
                self.model.vis.quality.offsamples = 0
                self.label_renderer = mujoco.Renderer(
                    self.model, height=self.renderer.height, width=self.renderer.width)
            finally:
                self.model.vis.quality.offsamples = samples
        renderer = self.label_renderer
        if renderer._gl_context is None or renderer._mjr_context is None:
            raise ValueError("Renderer context is unavailable")
        renderer._gl_context.make_current()
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, renderer._mjr_context)
        renderer.update_scene(self.data, camera="wrist_cam")
        renderer.enable_segmentation_rendering()
        try:
            segmentation = renderer.render()
        finally:
            renderer.disable_segmentation_rendering()
        return segmentation

    def instance_box(self, instance: str) -> dict:
        segmentation = self.segmentation_image()
        geom_ids = np.flatnonzero(self.model.geom_bodyid == self.registry[instance])
        mask = ((segmentation[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM)
                & np.isin(segmentation[:, :, 0], geom_ids))
        return box_from_mask(mask, instance)


def box_from_mask(mask: np.ndarray, instance: str) -> dict:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return {"instance_id": instance, "visible": False, "box_valid": False,
                "box_xyxy": None, "visible_pixels": 0}
    height, width = mask.shape
    return {
        "instance_id": instance, "visible": True, "box_valid": True,
        "box_xyxy": [float(xs.min()) / width, float(ys.min()) / height,
                     float(xs.max() + 1) / width, float(ys.max() + 1) / height],
        "visible_pixels": int(len(xs)),
    }


def create_simulation(problem: dict, renderbackend: str = "none") -> PileSimulation:
    if renderbackend not in ("none", "mujoco"):
        raise ValueError("Unsupported native rendering backend; photorealistic bridge not qualified")
    model = mujoco.MjModel.from_binary_path(str(problem["root"] / "model.mjb"))
    registry = {}
    for instance in problem["scene"]["instances"]:
        body = model.body(instance["body_name"]).id
        names = {model.geom(int(geom)).name
                 for geom in np.flatnonzero(model.geom_bodyid == body)}
        if names != set(instance["geom_names"]):
            raise ValueError("Instance geometry registry differs from compiled model")
        registry[instance["instance_id"]] = body
    sim = PileSimulation(model, mujoco.MjData(model), None, registry)
    reset_problem(sim, problem)
    if problem["task"]["task_type"] in ("push", "place"):
        goal = problem["task"]["goal"]
        marker = sim.data.body(goal["destination_entity"]).xpos[:2]
        if not np.allclose(marker, goal["destination_xy"], atol=1e-10, rtol=0):
            raise ValueError("Grader destination does not match the visible goal entity")
    if renderbackend == "mujoco":
        sim.renderer = mujoco.Renderer(model, height=240, width=320)
    return sim


def reset_problem(sim: PileSimulation, problem: dict) -> None:
    root = problem["root"]
    snapshot = json.loads((root / "integration_state.json").read_text())
    if "initial_state" in problem["task"]:
        snapshot = problem["task"]["initial_state"]
        if (not isinstance(snapshot, dict)
                or snapshot.get("scene_id") != problem["scene"]["scene_id"]):
            raise ValueError("Task-specific snapshot belongs to a different scene")
    restore_scene(sim.model, sim.data, snapshot, (root / "model.xml").read_text())
    sim.lifted_once = False
    progress = problem["task"].get("initial_task_state")
    if progress is not None:
        task = problem["task"]
        assistance = task.get("assistance", {})
        if (not isinstance(progress, dict) or set(progress) != {"lifted_once"}
                or type(progress["lifted_once"]) is not bool
                or not isinstance(assistance, dict) or not task.get("parent_task_id")
                or assistance.get("kind") not in (
                    "direct-task-distance-v1", "direct-task-distance-v2", "original-pose-grasp-v1")
                or assistance.get("parent_task_id") != task.get("parent_task_id")):
            raise ValueError("Invalid assistance initial task-progress state")
        measured = target_contact_state(sim, task["target_instance_id"])
        if (assistance["kind"] in ("direct-task-distance-v2", "original-pose-grasp-v1")
                and max(measured["jaw_normal_forces_n"].values(), default=0.) > MAXIMUM_ASSISTANCE_PRELOAD_N):
            raise ValueError("Assistance reset exceeds the per-jaw construction preload cap")
        if assistance["kind"] == "original-pose-grasp-v1":
            value = assistance.get("value")
            if (task["task_type"] != "grasp" or type(value) not in (int, float)
                    or not np.isfinite(value) or value not in tuple(i / 20 for i in range(20))):
                raise ValueError("Invalid original-pose GRASP assistance value")
            require_robot_above_table(sim.model, sim.data)
            if value >= .9 and not measured["bilateral_grip"]:
                raise ValueError("In-place GRASP reset lacks bilateral loaded contact")
            if value in (.8, .85) and measured["bilateral_grip"]:
                raise ValueError("Open/partial in-place GRASP reset is already grasped")
        if progress["lifted_once"]:
            if (task["task_type"] != "place" or not measured["bilateral_grip"]
                    or measured["table_supported"]):
                raise ValueError("PLACE progress initialization requires a physically held piece")
            sim.lifted_once = True


def target_contact_state(sim: PileSimulation, target: str) -> dict:
    body = sim.registry[target]
    jaws, supported = {}, False
    table = sim.model.geom("table").id
    labels = {sim.model.body("gripper").id: "fixed",
              sim.model.body("moving_jaw_so101_v1").id: "moving"}
    for index, contact in enumerate(sim.data.contact):
        if contact.dist > .0001:
            continue
        geoms = (int(contact.geom1), int(contact.geom2))
        bodies = {int(sim.model.geom_bodyid[g]) for g in geoms}
        if body not in bodies:
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(sim.model, sim.data, index, force)
        for endpoint, label in labels.items():
            if endpoint in bodies and force[0] > 1e-5:
                jaws[label] = jaws.get(label, 0.) + float(force[0])
        supported |= table in geoms and force[0] > 1e-5
    return {"jaw_normal_forces_n": jaws, "table_supported": bool(supported),
            "bilateral_grip": set(jaws) == {"fixed", "moving"}}


def task_predicate(sim: PileSimulation, problem: dict) -> bool:
    task = problem["task"]
    if task["grader_version"] != GRADER_VERSION:
        raise ValueError("Unsupported grader version")
    target = task["target_instance_id"]
    goal = task["goal"]
    position = sim.position(target)
    body = sim.registry[target]
    if task["task_type"] == "view":
        label = sim.instance_box(target)
        if not label["visible"] or label["visible_pixels"] < goal["minimum_pixels"]:
            return False
        x1, y1, x2, y2 = label["box_xyxy"]
        return (np.linalg.norm(np.array([(x1 + x2) / 2, (y1 + y2) / 2]) - .5)
                <= goal["center_radius"])
    measured = target_contact_state(sim, target)
    contacts = set(measured["jaw_normal_forces_n"])
    supported = measured["table_supported"]
    lift = float(position[2] - task["initial_target_position"][2])
    sim.lifted_once |= lift >= .03
    speed = body_surface_speed_bound(sim.model, sim.data, body)
    if task["task_type"] == "grasp":
        return lift >= goal["lift_m"] and contacts == {"fixed", "moving"} and speed < .03
    if task["task_type"] not in ("push", "place"):
        raise ValueError("Unsupported native task")
    error = float(np.linalg.norm(position[:2] - np.asarray(goal["destination_xy"])))
    released = not contacts
    stable = speed < goal["maximum_surface_speed_m_s"]
    if task["task_type"] == "push":
        return supported and released and stable and error <= goal["tolerance_m"]
    return sim.lifted_once and supported and released and stable and error <= goal["tolerance_m"]


def task_success(sim: PileSimulation, problem: dict, executed: int, stable: int) -> dict:
    if executed < 0 or stable < 0:
        raise ValueError("Negative task counters")
    stable = stable + 1 if task_predicate(sim, problem) else 0
    return {"success": stable >= problem["task"]["goal"]["dwell_steps"], "stable": stable}
