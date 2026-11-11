"""Construct parent-linked, nearer task resets directly, without demonstrations."""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import logging
import math
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import mujoco
import numpy as np

from so101.cloud_checkpoint import atomic_json
from so101.lego_scene import capture_scene_state
from so101.pile_task_generator import resource_check
from so101.pile_task_runtime import (
    MAXIMUM_ASSISTANCE_PRELOAD_N, PileSimulation, create_simulation, load_problem,
    reset_problem, target_contact_state, task_predicate,
)
from so101.robot_sim import BeamSimulation, MODEL_ROOT, TABLE_Z


LOGGER = logging.getLogger(__name__)
PROFILE = "direct-task-distance-v2"
STAGES = {
    "view": ("orient", "center", "dwell"),
    "grasp": ("approach", "grip", "lift", "dwell"),
    "push": ("approach", "contact", "push", "withdraw", "dwell"),
    "place": ("approach", "grip", "lift", "transport", "lower", "release", "dwell"),
}


def distance_label(kind: str, stage: str, value: float, metric: str, units: str) -> dict:
    order = STAGES[kind]
    if stage not in order or not math.isfinite(value) or value < 0:
        raise ValueError("Invalid task-stage distance")
    return {
        "schema_version": 1, "definition": "remaining-ordered-task-stages-v1",
        "stage_order": list(order), "active_stage": stage,
        "remaining_stages": list(order[order.index(stage):]),
        "remaining_stage_count": len(order) - order.index(stage),
        "within_stage": {"metric": metric, "value": value, "units": units},
        "ordering": "remaining_stage_count, then within-stage distance for the same metric",
        "qualification": "Task-stage/geometric distance, not measured policy difficulty or expert-prefix percentage",
    }


def original_distance(sim, task):
    target = sim.position(task["target_instance_id"])
    if task["task_type"] == "view":
        camera = sim.model.camera("wrist_cam").id
        axis = -sim.data.cam_xmat[camera].reshape(3, 3)[:, 2]
        ray = target - sim.data.cam_xpos[camera]
        value = float(np.arccos(np.clip(axis @ ray / np.linalg.norm(ray), -1., 1.)))
        return distance_label("view", "orient", value, "camera_target_optical_axis_error", "rad")
    return distance_label(task["task_type"], "approach",
                          float(np.linalg.norm(sim.data.site_xpos[sim.site] - target)),
                          "tool_to_target", "m")


class PlacementSimulation(BeamSimulation):
    def position(self, name):
        if name in self.object_names:
            return self.data.xipos[self.model.body(name).id].copy()
        return super().position(name)


def _put_robot(sim, point, yaw, *, closed=False, push=False, position_tolerance=None):
    precision = {} if position_tolerance is None else {"position_tolerance": position_tolerance}
    q = (sim.push_ik(point, float(sim.data.qpos[sim.qadr[4]]), **precision)
         if push else sim.ik(point, yaw, **precision))
    q[-1] = sim.limits[-1, 0] + (0. if closed else .7) * np.ptp(sim.limits[-1])
    _put_robot_q(sim, q)


def _put_robot_q(sim, q):
    sim.data.qpos[sim.qadr] = q
    sim.data.qvel[sim.dofs] = 0.
    sim.data.qacc_warmstart[sim.dofs] = 0.
    sim.data.ctrl[sim.actuators] = q
    mujoco.mj_forward(sim.model, sim.data)


def _put_target(sim, target, center, quaternion=None):
    body = sim.model.body(target).id
    joint = int(sim.model.body_jntadr[body])
    q, v = int(sim.model.jnt_qposadr[joint]), int(sim.model.jnt_dofadr[joint])
    if quaternion is not None:
        sim.data.qpos[q + 3:q + 7] = quaternion
    sim.data.qvel[v:v + 6] = 0.
    sim.data.qacc_warmstart[v:v + 6] = 0.
    mujoco.mj_forward(sim.model, sim.data)
    sim.data.qpos[q:q + 3] += np.asarray(center) - sim.position(target)
    mujoco.mj_forward(sim.model, sim.data)
    return body, q, v


def _finite_contacts(sim):
    return (
        all(np.isfinite(value).all() for value in (sim.data.qpos, sim.data.qvel, sim.data.qacc))
        and not any(warning.number for warning in sim.data.warning)
        and max((max(0., -float(c.dist)) for c in sim.data.contact), default=0.) <= .005
    )


def _pushing_contact_gap(sim, target):
    bodies = {sim.model.body(name).id for name in ("gripper", "moving_jaw_so101_v1")}
    collidable = (sim.model.geom_contype != 0) | (sim.model.geom_conaffinity != 0)
    jaws = np.flatnonzero(np.isin(sim.model.geom_bodyid, list(bodies)) & collidable)
    pieces = np.flatnonzero((sim.model.geom_bodyid == sim.model.body(target).id) & collidable)
    return min(mujoco.mj_geomDistance(sim.model, sim.data, int(jaw), int(piece), .05, None)
               for jaw in jaws for piece in pieces)


def _put_on_table(sim, target, center):
    body, q, _ = _put_target(sim, target, center)
    geoms = np.flatnonzero((sim.model.geom_bodyid == body)
                          & ((sim.model.geom_contype != 0) | (sim.model.geom_conaffinity != 0)))
    rotations = sim.data.geom_xmat[geoms].reshape(-1, 3, 3)
    lower = (sim.data.geom_xpos[geoms, 2]
             + np.einsum("ni,ni->n", rotations[:, 2], sim.model.geom_aabb[geoms, :3])
             - np.einsum("ni,ni->n", np.abs(rotations[:, 2]), sim.model.geom_aabb[geoms, 3:]))
    sim.data.qpos[q + 2] += TABLE_Z - float(lower.min())
    mujoco.mj_forward(sim.model, sim.data)
    table = sim.model.geom("table").id
    gap = min(mujoco.mj_geomDistance(sim.model, sim.data, table, int(geom), .05, None)
              for geom in geoms)
    sim.data.qpos[q + 2] -= gap + .000005
    mujoco.mj_forward(sim.model, sim.data)


def _view_outside_goal(sim, target, goal):
    camera = sim.model.camera("wrist_cam").id
    points = []
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    for geom in sim.beam_geoms[target]:
        corners = sim.model.geom_aabb[geom, :3] + signs * sim.model.geom_aabb[geom, 3:]
        points.extend(corners @ sim.data.geom_xmat[geom].reshape(3, 3).T
                      + sim.data.geom_xpos[geom])
    local = (np.asarray(points) - sim.data.cam_xpos[camera]) @ sim.data.cam_xmat[camera].reshape(3, 3)
    depths = -local[:, 2]
    if np.all(depths <= 0):
        return True
    if np.any(depths <= 0):
        return False
    half_fov = math.tan(math.radians(float(sim.model.cam_fovy[camera])) / 2)
    normalized = local[:, :2] / (2 * depths[:, None] * half_fov)
    normalized[:, 0] /= 320 / 240
    radius = goal["center_radius"]
    return bool(np.any(normalized.min(axis=0) > radius)
                or np.any(normalized.max(axis=0) < -radius))


def _holding(sim, target, tool_point, yaw, *, position_tolerance=None):
    _put_robot(sim, tool_point, yaw, position_tolerance=position_tolerance)
    opened = sim.data.qpos[sim.qadr[-1]]
    for rotation in range(4):
        sim.data.qpos[sim.qadr[-1]] = opened
        sim.data.ctrl[sim.actuators[-1]] = opened
        mujoco.mj_forward(sim.model, sim.data)
        angle = rotation * math.pi / 2
        quaternion = np.array([math.cos(angle / 2), 0., math.sin(angle / 2), 0.])
        body, _, _ = _put_target(sim, target, sim.data.site_xpos[sim.site], quaternion)
        axis = sim.data.xmat[sim.gripper].reshape(3, 3)[:, 0]
        geoms = np.flatnonzero((sim.model.geom_bodyid == body)
                              & ((sim.model.geom_contype != 0) | (sim.model.geom_conaffinity != 0)))
        rotations = sim.data.geom_xmat[geoms].reshape(-1, 3, 3)
        projected = np.einsum("i,nij->nj", axis, rotations)
        lower = (sim.data.geom_xpos[geoms] @ axis
                 + np.einsum("ni,ni->n", projected, sim.model.geom_aabb[geoms, :3])
                 - np.einsum("ni,ni->n", np.abs(projected), sim.model.geom_aabb[geoms, 3:])).min()
        jaw = sim.model.geom("fixed_jaw_box4").id
        projection = axis @ sim.data.geom_xmat[jaw].reshape(3, 3)
        inner = (sim.data.geom_xpos[jaw] @ axis
                 + projection @ sim.model.geom_aabb[jaw, :3]
                 + np.abs(projection) @ sim.model.geom_aabb[jaw, 3:])
        _put_target(sim, target, sim.position(target) + axis * (inner - .00015 - lower))
        fixed_geoms = np.flatnonzero(
            (sim.model.geom_bodyid == sim.gripper)
            & ((sim.model.geom_contype != 0) | (sim.model.geom_conaffinity != 0)))
        for _ in range(12):
            separation = min(
                mujoco.mj_geomDistance(sim.model, sim.data, int(fixed), int(geom), .03, None)
                for fixed in fixed_geoms for geom in geoms)
            if abs(separation + .000001) <= .0000001:
                break
            _put_target(sim, target, sim.position(target) - axis * (separation + .000001))
        else:
            continue
        previous = opened
        for grip in np.linspace(opened, sim.limits[-1, 0], 32):
            sim.data.qpos[sim.qadr[-1]] = grip
            sim.data.ctrl[sim.actuators[-1]] = grip
            mujoco.mj_forward(sim.model, sim.data)
            measured = target_contact_state(sim.native, target)
            if measured["bilateral_grip"] and not measured["table_supported"] and _finite_contacts(sim):
                closed, open_limit = grip, previous
                for _ in range(16):
                    middle = (closed + open_limit) / 2
                    sim.data.qpos[sim.qadr[-1]] = middle
                    sim.data.ctrl[sim.actuators[-1]] = middle
                    mujoco.mj_forward(sim.model, sim.data)
                    if target_contact_state(sim.native, target)["bilateral_grip"]:
                        closed = middle
                    else:
                        open_limit = middle
                sim.data.qpos[sim.qadr[-1]] = closed
                sim.data.ctrl[sim.actuators[-1]] = closed
                mujoco.mj_forward(sim.model, sim.data)
                measured = target_contact_state(sim.native, target)
                if max(measured["jaw_normal_forces_n"].values(), default=0.) <= MAXIMUM_ASSISTANCE_PRELOAD_N:
                    return measured
                break
            previous = grip
    raise ValueError("No valid directly constructed bilateral holding state")


def construct_state(sim, problem: dict, stage: str, fraction: float) -> tuple[dict, dict]:
    reset_problem(sim.native, problem)
    task = problem["task"]
    baseline = original_distance(sim, task)
    target, kind = task["target_instance_id"], task["task_type"]
    origin = sim.position(target)
    yaw = math.atan2(sim.data.body(target).xmat[3], sim.data.body(target).xmat[0]) + math.pi / 2
    progress = {"lifted_once": False}
    if stage in ("orient", "approach"):
        original_q = sim.data.qpos[sim.qadr].copy()
        start = sim.data.site_xpos[sim.site].copy()
        point = origin + (start - origin) * fraction
        point[2] = max(point[2], origin[2] + .025)
        _put_robot(sim, point, yaw)
        value = float(np.linalg.norm(sim.data.site_xpos[sim.site] - origin))
        metric, units = "tool_to_target", "m"
        if kind == "view":
            nearer = sim.data.qpos[sim.qadr].copy()
            sim.data.qpos[sim.qadr] = original_q + (nearer - original_q) * (1 - fraction) * .5
            sim.data.ctrl[sim.actuators] = sim.data.qpos[sim.qadr]
            mujoco.mj_forward(sim.model, sim.data)
            camera = sim.model.camera("wrist_cam").id
            axis = -sim.data.cam_xmat[camera].reshape(3, 3)[:, 2]
            ray = origin - sim.data.cam_xpos[camera]
            value = float(np.arccos(np.clip(axis @ ray / np.linalg.norm(ray), -1., 1.)))
            metric, units = "camera_target_optical_axis_error", "rad"
    elif kind in ("grasp", "place"):
        destination = np.asarray(task["goal"].get("destination_xy", origin[:2]))
        point = np.r_[origin[:2], max(.07, origin[2] + .035)]
        if stage == "transport":
            point[:2] = destination + (origin[:2] - destination) * fraction
        elif stage == "lower":
            point[:2] = destination
            point[2] = TABLE_Z + .025 + .04 * fraction
        elif kind == "grasp":
            point[2] = task["initial_target_position"][2] + .005 + .02 * fraction
        measured = _holding(sim, target, point, yaw)
        progress["lifted_once"] = kind == "place"
        if stage == "transport":
            value = float(np.linalg.norm(sim.position(target)[:2] - destination))
            metric = "held_target_to_destination_xy"
        elif stage == "lower":
            value = max(0., float(sim.position(target)[2] - TABLE_Z))
            metric = "held_target_height_above_table"
        else:
            value = max(0., float(task["initial_target_position"][2]
                                  + task["goal"].get("lift_m", .03) - sim.position(target)[2]))
            metric = "remaining_target_lift"
        units = "m"
    elif kind == "push":
        destination = np.asarray(task["goal"]["destination_xy"])
        direction = destination - origin[:2]
        original_length = float(np.linalg.norm(direction))
        if original_length <= task["goal"]["tolerance_m"]:
            raise ValueError("Parent target is already inside destination tolerance")
        direction /= original_length
        center = destination - direction * max(original_length * fraction, .04)
        for height, separation in itertools.product((.014, .008, .020), np.linspace(.04, -.015, 24)):
            reset_problem(sim.native, problem)
            _put_robot(sim, np.r_[center - direction * separation, TABLE_Z + height],
                       yaw, closed=True, push=True)
            _put_on_table(sim, target, np.r_[center, origin[2]])
            measured = target_contact_state(sim.native, target)
            if measured["jaw_normal_forces_n"] and measured["table_supported"] and _finite_contacts(sim):
                contact_position = sim.position(target)
                away = np.r_[direction, 0.]
                for clearance in (.001, .003, .006, .012):
                    _put_target(sim, target, contact_position + away * clearance)
                    if not target_contact_state(sim.native, target)["jaw_normal_forces_n"]:
                        break
                else:
                    continue
                touching, clear = 0., clearance
                for _ in range(16):
                    middle = (touching + clear) / 2
                    _put_target(sim, target, contact_position + away * middle)
                    forces = target_contact_state(sim.native, target)["jaw_normal_forces_n"]
                    if max(forces.values(), default=0.) >= .01:
                        touching = middle
                    else:
                        clear = middle
                _put_target(sim, target, contact_position + away * clear)
                measured = target_contact_state(sim.native, target)
                gap = _pushing_contact_gap(sim, target)
                if (measured["table_supported"] and _finite_contacts(sim)
                        and -.0001 <= gap <= .0001
                        and max(measured["jaw_normal_forces_n"].values(), default=0.) <= MAXIMUM_ASSISTANCE_PRELOAD_N):
                    break
        else:
            raise ValueError("No valid directly constructed pushing-contact state")
        value = float(np.linalg.norm(sim.position(target)[:2] - destination))
        metric, units = "target_to_destination_xy", "m"
    else:
        raise ValueError("Unsupported assistance stage")
    if not _finite_contacts(sim):
        raise ValueError("Constructed state is nonfinite, warned, or penetrates beyond 5mm")
    if stage == baseline["active_stage"] and value >= baseline["within_stage"]["value"]:
        raise ValueError("Constructed state is not closer than its original task")
    if kind == "view":
        if not _view_outside_goal(sim, target, task["goal"]):
            raise ValueError("Constructed state lacks a geometric proof that VIEW is not already centered")
    else:
        sim.native.lifted_once = progress["lifted_once"]
        if task_predicate(sim.native, problem):
            raise ValueError("Constructed state already satisfies the parent goal")
    measured = target_contact_state(sim.native, target)
    if max(measured["jaw_normal_forces_n"].values(), default=0.) > MAXIMUM_ASSISTANCE_PRELOAD_N:
        raise ValueError("Constructed state exceeds the 1N per-jaw construction preload cap")
    if kind == "push" and stage == "push":
        measured["pushing_contact_gap_m"] = _pushing_contact_gap(sim, target)
        measured["pushing_contact_tolerance_m"] = .0001
    return distance_label(kind, stage, value, metric, units), {
        "initial_task_state": progress, "physical_predicates": measured,
        "target_position_m": sim.position(target).tolist(),
        "tool_position_m": sim.data.site_xpos[sim.site].tolist(),
    }


def _candidate_failure(error):
    return str(error).startswith((
        "No vertical IK solution", "No fixed-wrist push IK solution",
        "No valid directly", "Constructed state", "Parent target"))


def _near_view(sim, problem, gap):
    target, goal = problem["task"]["target_instance_id"], problem["task"]["goal"]
    camera = sim.model.camera("wrist_cam").id
    tangent = math.tan(math.radians(float(sim.model.cam_fovy[camera])) / 2)

    def table_ray(center, z):
        direction = sim.data.cam_xmat[camera].reshape(3, 3) @ np.array(
            [(center[0] - .5) * 2 * tangent * 320 / 240,
             -(center[1] - .5) * 2 * tangent, -1.])
        distance = (z - sim.data.cam_xpos[camera, 2]) / direction[2]
        if distance <= 0 or not math.isfinite(distance):
            raise ValueError("Constructed state camera does not face the table")
        return (sim.data.cam_xpos[camera] + direction * distance)[:2]

    desired = np.array([.5 + goal["center_radius"] + gap, .5])
    for height in (.07, .05, .09):
        reset_problem(sim.native, problem)
        try:
            _put_robot(sim, np.array([.22, 0., TABLE_Z + height]), 0.)
            _put_on_table(sim, target, np.array([.3, .15, TABLE_Z]))
            center = table_ray(desired, sim.position(target)[2])
            _put_on_table(sim, target, np.r_[center, sim.position(target)[2]])
        except (ValueError, RuntimeError) as error:
            if not _candidate_failure(error):
                raise
            continue
        for _ in range(10):
            label = sim.native.instance_box(target)
            if not label["visible"] or label["visible_pixels"] < goal["minimum_pixels"]:
                break
            box = label["box_xyxy"]
            actual = (np.array(box[:2]) + box[2:]) / 2
            excess = float(np.linalg.norm(actual - .5) - goal["center_radius"])
            if (excess > 0 and abs(excess - gap) <= .0025
                    and _finite_contacts(sim)
                    and target_contact_state(sim.native, target)["table_supported"]):
                return excess, label
            z = sim.position(target)[2]
            shift = table_ray(desired, z) - table_ray(actual, z)
            _put_target(sim, target, sim.position(target) + np.r_[shift, 0.])
    raise ValueError("No valid directly constructed native-pixel VIEW boundary")


def construct_near_state(sim, problem, stage, gap):
    reset_problem(sim.native, problem)
    task = problem["task"]
    target, kind, goal = task["target_instance_id"], task["task_type"], task["goal"]
    origin = sim.position(target)
    extra = {}
    if kind == "grasp":
        desired_z = task["initial_target_position"][2] + goal["lift_m"] - gap
        failure = None
        for xy, yaw in itertools.product((origin[:2], (.22, 0.), (.18, 0.), (.25, 0.)),
                                         (0., math.pi / 2)):
            reset_problem(sim.native, problem)
            point = np.r_[xy, desired_z]
            try:
                for _ in range(4):
                    _holding(sim, target, point, yaw, position_tolerance=.000001)
                    error_z = desired_z - sim.position(target)[2]
                    if abs(error_z) <= .000005:
                        break
                    point[2] += error_z
                else:
                    raise ValueError("Constructed state lift boundary did not converge")
            except (ValueError, RuntimeError) as error:
                if not _candidate_failure(error):
                    raise
                failure = error
                continue
            break
        else:
            raise ValueError(f"No valid directly constructed near-lift grip: {failure}")
        value = float(task["initial_target_position"][2] + goal["lift_m"] - sim.position(target)[2])
        metric, units = "remaining_target_lift", "m"
    elif kind == "push":
        destination = np.asarray(goal["destination_xy"])
        direction = destination - origin[:2]
        if np.linalg.norm(direction) <= goal["tolerance_m"]:
            raise ValueError("Parent target is already inside destination tolerance")
        direction /= np.linalg.norm(direction)
        center = destination - direction * (goal["tolerance_m"] + gap)
        found = False
        for height in (.014, .008, .020):
            reset_problem(sim.native, problem)
            _put_on_table(sim, target, np.r_[center, origin[2]])
            clear_q = None
            for separation in np.linspace(.05, -.025, 32):
                try:
                    _put_robot(sim, np.r_[center - direction * separation, TABLE_Z + height],
                               0., closed=True, push=True)
                except RuntimeError as error:
                    if not _candidate_failure(error):
                        raise
                    continue
                distance = _pushing_contact_gap(sim, target)
                if distance > .00001:
                    clear_q = sim.data.qpos[sim.qadr].copy()
                    continue
                if clear_q is None:
                    continue
                touching_q = sim.data.qpos[sim.qadr].copy()
                for _ in range(18):
                    middle = (touching_q + clear_q) / 2
                    _put_robot_q(sim, middle)
                    if _pushing_contact_gap(sim, target) < .00001:
                        touching_q = middle
                    else:
                        clear_q = middle
                _put_robot_q(sim, clear_q)
                if _finite_contacts(sim):
                    found = True
                    break
            if found:
                break
        if not found:
            raise ValueError("No valid directly constructed near-goal PUSH contact")
        value = float(np.linalg.norm(sim.position(target)[:2] - destination) - goal["tolerance_m"])
        metric, units = "distance_outside_destination_tolerance", "m"
        extra = {"pushing_contact_gap_m": _pushing_contact_gap(sim, target),
                 "pushing_contact_tolerance_m": .0001}
    elif kind == "view":
        value, label = _near_view(sim, problem, gap)
        metric, units = "native_bbox_center_radius_excess", "normalized_image_radius"
        extra = {"native_view_label": label, "native_view_resolution": [320, 240]}
    else:
        raise ValueError("Unsupported near-terminal task")
    measured = target_contact_state(sim.native, target)
    if (not _finite_contacts(sim) or value <= 0
            or abs(value - gap) > (.0025 if kind == "view" else .00001)
            or max(measured["jaw_normal_forces_n"].values(), default=0.) > MAXIMUM_ASSISTANCE_PRELOAD_N
            or (kind == "grasp" and (not measured["bilateral_grip"] or measured["table_supported"]))
            or (kind == "push" and (not measured["table_supported"]
                                   or abs(extra["pushing_contact_gap_m"]) > .0001))
            or task_predicate(sim.native, problem)):
        raise ValueError("Constructed state failed near-terminal physical/native boundary checks")
    return distance_label(kind, stage, value, metric, units), {
        "initial_task_state": {"lifted_once": False},
        "physical_predicates": {**measured, **extra},
        "requested_goal_gap": {"value": gap, "units": units},
        "target_position_m": sim.position(target).tolist(),
        "tool_position_m": sim.data.site_xpos[sim.site].tolist(),
    }


def generate(entry: dict, output: Path, *, task_ids: list[str] | None = None,
             model_root: Path = MODEL_ROOT, near_terminal: bool = False,
             parents_per_type: int | None = None) -> dict:
    if parents_per_type is not None and (not near_terminal or parents_per_type < 1):
        raise ValueError("Parent quota requires the near-terminal recipe and a positive count")
    if output.exists():
        raise FileExistsError("Assistance output exists; reconcile instead of overwriting")
    output.parent.mkdir(parents=True, exist_ok=True)
    source = Path(entry["root"])
    parent = json.loads((source / "problem.json").read_bytes())
    parents = [task for task in parent["tasks"] if task_ids is None or task["task_id"] in task_ids]
    if not parents or (task_ids is not None and {task["task_id"] for task in parents} != set(task_ids)):
        raise ValueError("Unknown or empty parent task selection")
    first = load_problem(entry, parents[0]["task_id"])
    model = mujoco.MjModel.from_binary_path(str(source / "model.mjb"))
    force_ranges = model.actuator_forcerange.copy()
    names = tuple(item["instance_id"] for item in parent["instances"])
    sim = PlacementSimulation(model_root, model=model, render=False, record_images=False,
                              object_names=names, distractor_names=())
    sim.object_names = names
    model.actuator_forcerange[:] = force_ranges
    sim.native = PileSimulation(model, sim.data, None, {name: model.body(name).id for name in names})
    implementation = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    provenance = {"profile": PROFILE, "parent_entry": entry, "generator_sha256": implementation,
                  "maximum_initial_jaw_force_n": MAXIMUM_ASSISTANCE_PRELOAD_N,
                  "runtime_sha256": hashlib.sha256(
                      Path(__file__).with_name("pile_task_runtime.py").read_bytes()).hexdigest(),
                  "parent_task_ids": [task["task_id"] for task in parents]}
    if near_terminal:
        provenance.update(sampling_recipe="near-terminal-v1", parents_per_type=parents_per_type,
                          minimum_variants_per_parent=2)
        if any(task["task_type"] == "view" for task in parents):
            sim.native.renderer = mujoco.Renderer(model, height=240, width=320)
    scene_id = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    plan = {"scene_id": scene_id, "catalog_sha256": parent["catalog_sha256"]}
    tasks, groups = [], []
    covered = {kind: 0 for kind in STAGES}
    try:
        for original in parents:
            if parents_per_type is not None and covered[original["task_type"]] >= parents_per_type:
                continue
            problem = {**first, "task": original}
            reset_problem(sim.native, problem)
            group = {"parent_task_id": original["task_id"], "original_entry": entry,
                     "original_distance": original_distance(sim, original),
                     "assistance_task_ids": [], "unavailable": []}
            kind = original["task_type"]
            stages = {"view": ("orient",), "grasp": ("approach", "lift"),
                      "push": ("approach", "push"), "place": ("approach", "transport", "lower")}[kind]
            if near_terminal:
                stages = {"view": ("center",), "grasp": ("lift",),
                          "push": ("push",), "place": ("lower",)}[kind]
            for stage in stages:
                parameters = ((.005, .01, .02, .04) if kind == "view" else (.0005, .001, .002, .005))
                for fraction in parameters if near_terminal and kind != "place" else (.65, .3):
                    request = (
                        {"requested_goal_gap": {"value": fraction, "units": (
                            "normalized_image_radius" if kind == "view" else "m")}}
                        if near_terminal and kind != "place" else {"fraction": fraction})
                    resource_check(output.parent)
                    if any((root / "STOP").exists() for root in (output.parent, output.parent.parent)):
                        raise InterruptedError("Assistance construction stopped")
                    try:
                        builder = construct_near_state if near_terminal and kind != "place" else construct_state
                        distance, construction = builder(sim, problem, stage, fraction)
                    except (ValueError, RuntimeError) as error:
                        if not _candidate_failure(error):
                            raise
                        group["unavailable"].append({"stage": stage, **request, "reason": str(error)})
                        LOGGER.warning("Assistance construction rejected: %s", error)
                        continue
                    child = copy.deepcopy(original)
                    identity = f"{scene_id}:{original['task_id']}:{stage}:{fraction}"
                    child["task_id"] = hashlib.sha256(identity.encode()).hexdigest()
                    child["parent_task_id"] = original["task_id"]
                    child["initial_task_state"] = construction["initial_task_state"]
                    child["initial_state"] = capture_scene_state(model, sim.data, (source / "model.xml").read_text(),
                                                                 plan, status="task-start")
                    child["assistance"] = {
                        "schema_version": 2, "kind": PROFILE, "parent_task_id": original["task_id"],
                        "construction": "direct state placement; no demonstration, setup trajectory, or expert prefix",
                        **({"construction_fraction": fraction} if not near_terminal or kind == "place" else {}),
                        "distance": distance, **construction,
                    }
                    if near_terminal:
                        child["assistance"]["sampling_recipe"] = (
                            "place-lower-control-v2" if kind == "place" else "near-terminal-v1")
                    child["start"] = {"profile": PROFILE, **construction}
                    if any(np.array_equal(child["initial_state"]["integration_state"],
                                          previous["initial_state"]["integration_state"])
                           for previous in tasks if previous["parent_task_id"] == original["task_id"]):
                        group["unavailable"].append({"stage": stage, **request,
                                                     "reason": "Duplicate physical reset"})
                        continue
                    tasks.append(child)
                    group["assistance_task_ids"].append(child["task_id"])
            groups.append(group)
            if len(group["assistance_task_ids"]) >= 2:
                covered[kind] += 1
    finally:
        sim.native.close()
        sim.close()
    if not tasks:
        raise ValueError("No physically valid assistance variants constructed")
    tags = {**parent["tags"], "task_start_profile": PROFILE, "task_assistance_profile": PROFILE}
    if near_terminal:
        tags["task_assistance_recipe"] = "near-terminal-v1"
    scene = {**parent, "scene_id": scene_id, "tasks": tasks, "task_families": groups, "tags": tags,
             "provenance": provenance, "generator_sha256": implementation,
             "grader_sha256": provenance["runtime_sha256"]}
    if near_terminal:
        scene["sampling_status"] = {
            "candidate_parent_count": len(parents), "attempted_parent_count": len(groups),
            "parents_with_multiple_variants": covered, "requested_parents_per_type": parents_per_type,
            "quota_met": parents_per_type is None or all(
                covered[kind] >= parents_per_type for kind in {task["task_type"] for task in parents}),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".assistance-", dir=output.parent) as directory:
        staging = Path(directory) / "scene"
        staging.mkdir()
        for name in ("model.xml", "model.mjb"):
            shutil.copyfile(source / name, staging / name)
        for name in ("scene_plan.json", "integration_state.json"):
            content = json.loads((source / name).read_bytes())
            atomic_json(staging / name, {**content, "scene_id": scene_id})
        atomic_json(staging / "problem.json", scene)
        result = {**entry, "scene_id": scene_id, "root": str(output.resolve()), "tags": tags,
                  "files_sha256": {name: hashlib.sha256((staging / name).read_bytes()).hexdigest()
                                   for name in entry["files_sha256"]}}
        staged = {**result, "root": str(staging.resolve())}
        verified = create_simulation(load_problem(staged, tasks[0]["task_id"]),
                                     renderbackend="mujoco" if near_terminal and any(
                                         task["task_type"] == "view" for task in tasks) else "none")
        try:
            for task in tasks:
                problem = {"root": staging, "scene": scene, "task": task}
                reset_problem(verified, problem)
                state = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
                mujoco.mj_getState(verified.model, verified.data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
                if not np.array_equal(state, task["initial_state"]["integration_state"]):
                    raise ValueError("Assistance full-state reset differs from construction")
                if not np.array_equal(verified.position(task["target_instance_id"]),
                                      task["start"]["target_position_m"]):
                    raise ValueError("Assistance target changed during reset")
                if near_terminal:
                    if task_predicate(verified, problem):
                        raise ValueError("Near-terminal reset already satisfies the native goal")
                    if task["task_type"] == "view":
                        label = verified.instance_box(task["target_instance_id"])
                        if label != task["assistance"]["physical_predicates"]["native_view_label"]:
                            raise ValueError("Near-terminal native VIEW label differs after reset")
        finally:
            verified.close()
        atomic_json(staging / "admission.json", result)
        staging.rename(output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--near-terminal", action="store_true")
    parser.add_argument("--parents-per-type", type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    os.nice(19)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(generate(json.loads(args.admission.read_bytes()), args.output,
                              task_ids=args.task_id, near_terminal=args.near_terminal,
                              parents_per_type=args.parents_per_type)), flush=True)


if __name__ == "__main__":
    main()
