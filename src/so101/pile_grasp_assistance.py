"""Independently construct approved in-place GRASP assistance, without execution."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import mujoco
import numpy as np
from scipy.optimize import least_squares

from so101.cloud_checkpoint import atomic_json
from so101.lego_scene import capture_scene_state
from so101.pile_task_runtime import (
    MAXIMUM_ASSISTANCE_PRELOAD_N, PileSimulation, create_simulation, load_problem,
    reset_problem, target_contact_state, task_predicate,
)
from so101.robot_sim import BeamSimulation
from so101.robot_table_clearance import (
    require_robot_above_table, robot_body_ids, robot_table_clearance,
)


PROFILE = "original-pose-grasp-v1"
LEVELS = tuple(tick / 20 for tick in range(20))
LOGGER = logging.getLogger(__name__)


class UnavailableGrasp(ValueError):
    """A parent or reset cannot satisfy the approved construction constraints."""


def robot_pose(original: np.ndarray, closed: np.ndarray, open_limit: float,
               assistance: float) -> np.ndarray:
    if not np.isfinite(assistance) or not 0 <= assistance <= .9:
        raise ValueError("Robot-pose assistance must be between 0 and 0.9")
    opened = closed.copy()
    opened[-1] = open_limit
    if assistance == .9:
        return closed.copy()
    if assistance == .8:
        return opened
    if assistance < .8:
        return original + (assistance / .8) * (opened - original)
    return opened + ((assistance - .8) / .1) * (closed - opened)


class GraspConstructor:
    def __init__(self, problem: dict):
        self.problem = problem
        model = mujoco.MjModel.from_binary_path(str(problem["root"] / "model.mjb"))
        names = tuple(item["instance_id"] for item in problem["scene"]["instances"])
        force_ranges = model.actuator_forcerange.copy()
        self.sim = BeamSimulation(model=model, render=False, record_images=False,
                                  object_names=names, distractor_names=())
        model.actuator_forcerange[:] = force_ranges
        self.native = PileSimulation(model, self.sim.data, None, {
            item["instance_id"]: model.body(item["body_name"]).id
            for item in problem["scene"]["instances"]})
        self.bodies = robot_body_ids(model)
        collision = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
        self.jaws = [
            np.flatnonzero(collision & (model.geom_bodyid == model.body(name).id))
            for name in ("gripper", "moving_jaw_so101_v1")
        ]
        self.original = None

    def put(self, q: np.ndarray) -> None:
        sim = self.sim
        sim.data.qpos[sim.qadr] = q
        sim.data.ctrl[sim.actuators] = q
        sim.data.qvel[sim.dofs] = 0
        sim.data.qacc_warmstart[:] = 0
        mujoco.mj_forward(sim.model, sim.data)

    def robot_obstacle_depth(self, target: int | None = None) -> float:
        model, data = self.sim.model, self.sim.data
        depths = []
        for contact in data.contact:
            bodies = {int(model.geom_bodyid[contact.geom1]),
                      int(model.geom_bodyid[contact.geom2])}
            if self.bodies.intersection(bodies) and target not in bodies:
                depths.append(max(0., -float(contact.dist)))
        return max(depths, default=0.)

    def validate(self, task: dict, level: float) -> dict:
        sim, native = self.sim, self.native
        if (not all(np.isfinite(value).all() for value in (
                sim.data.qpos, sim.data.qvel, sim.data.qacc))
                or any(warning.number for warning in sim.data.warning)):
            raise UnavailableGrasp("Nonfinite state or MuJoCo warning")
        try:
            clearance = require_robot_above_table(sim.model, sim.data)
        except ValueError as error:
            raise UnavailableGrasp(str(error)) from error
        depth = self.robot_obstacle_depth()
        if depth > .001:
            raise UnavailableGrasp(f"Robot scene penetration {depth:.9f}m exceeds 1mm")
        measured = target_contact_state(native, task["target_instance_id"])
        if max(measured["jaw_normal_forces_n"].values(), default=0.) > MAXIMUM_ASSISTANCE_PRELOAD_N:
            raise UnavailableGrasp("Per-jaw preload exceeds 1N")
        if level >= .9 and not measured["bilateral_grip"]:
            raise UnavailableGrasp("Grasp does not have bilateral loaded jaw contact")
        if level in (.8, .85) and measured["bilateral_grip"]:
            raise UnavailableGrasp("Open/partially closed reset is already grasped")
        target = native.registry[task["target_instance_id"]]
        ignored = set(int(address) for address in sim.qadr)
        if level > .9:
            joint = int(sim.model.body_jntadr[target])
            address = int(sim.model.jnt_qposadr[joint])
            ignored.update(range(address, address + 7))
        unchanged = [i for i in range(sim.model.nq) if i not in ignored]
        if not np.array_equal(sim.data.qpos[unchanged], self.original[unchanged]):
            raise UnavailableGrasp("Construction changed an original pile pose")
        problem = {**self.problem, "task": task}
        if task_predicate(native, problem):
            raise UnavailableGrasp("Constructed reset already satisfies the original goal")
        return {
            "physical_predicates": measured, "robot_table_clearance": clearance,
            "maximum_robot_contact_depth_m": depth,
            "target_position_m": native.position(task["target_instance_id"]).tolist(),
            "tool_position_m": sim.data.site_xpos[sim.site].tolist(),
            "initial_task_state": {"lifted_once": False},
        }

    def find_anchor(self, task: dict) -> np.ndarray:
        sim, native = self.sim, self.native
        problem = {**self.problem, "task": task}
        reset_problem(native, problem)
        self.original = sim.data.qpos.copy()
        original_q = sim.data.qpos[sim.qadr].copy()
        center = native.position(task["target_instance_id"])
        target = native.registry[task["target_instance_id"]]
        collision = (sim.model.geom_contype != 0) | (sim.model.geom_conaffinity != 0)
        pieces = np.flatnonzero(collision & (sim.model.geom_bodyid == target))
        if not len(pieces) or not all(len(jaw) for jaw in self.jaws):
            raise UnavailableGrasp("Missing target/jaw collision geometry")

        def gaps():
            return np.array([
                min(mujoco.mj_geomDistance(sim.model, sim.data, int(a), int(b), .2, None)
                    for a in jaw for b in pieces)
                for jaw in self.jaws])

        reasons = []
        bounds = (sim.limits[:, 0] + 1e-6, sim.limits[:, 1] - 1e-6)
        for yaw in (0., np.pi / 2, np.pi / 4, -np.pi / 4):
            reset_problem(native, problem)
            try:
                q = sim.ik(center + [0, 0, .008], yaw, position_tolerance=.001)
            except RuntimeError as error:
                if not str(error).startswith("No vertical IK solution"):
                    raise
                reasons.append(str(error))
                continue
            q[-1] = sim.limits[-1, 0] + .5 * np.ptp(sim.limits[-1])

            def coarse_residual(q):
                self.put(q)
                blocked = max(0., self.robot_obstacle_depth(target) - .0003)
                return np.r_[gaps() + .000002,
                             .05 * (sim.data.site_xpos[sim.site] - center), blocked * 2]

            fit = least_squares(coarse_residual, q, bounds=bounds, diff_step=1e-4,
                                max_nfev=100, ftol=1e-9, xtol=1e-9, gtol=1e-9)
            reference = fit.x.copy()

            def fine_residual(q):
                self.put(q)
                forces = target_contact_state(native, task["target_instance_id"])["jaw_normal_forces_n"]
                blocked = max(0., self.robot_obstacle_depth(target) - .000001)
                table = max(0., -robot_table_clearance(sim.model, sim.data)["minimum_clearance_m"])
                return np.r_[gaps() + .0000002, blocked * 10, table * 10,
                             [max(0., forces.get(name, 0.) - .5) * .00001
                              for name in ("fixed", "moving")],
                             .000001 * (q - reference)]

            fit = least_squares(fine_residual, reference, bounds=bounds, diff_step=1e-5,
                                max_nfev=100, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            if not fit.success:
                reasons.append(f"Static contact fit did not converge: {fit.message}")
                continue
            closed = fit.x.copy()
            try:
                for level in LEVELS[:-1]:
                    reset_problem(native, problem)
                    self.put(robot_pose(original_q, closed, sim.limits[-1, 1], level))
                    self.validate(task, level)
            except UnavailableGrasp as error:
                reasons.append(f"At {level:.2f}: {error}")
                continue
            return closed
        raise UnavailableGrasp("; ".join(reasons))

    def family(self, task: dict, scene_id: str) -> list[dict]:
        if task["task_type"] != "grasp" or task.get("parent_task_id"):
            raise ValueError("In-place GRASP construction requires an original GRASP parent")
        sim, native = self.sim, self.native
        problem = {**self.problem, "task": task}
        closed = self.find_anchor(task)
        original_q = self.original[sim.qadr].copy()
        self.put(closed)
        target = native.registry[task["target_instance_id"]]
        rotation = sim.data.xmat[sim.gripper].reshape(3, 3).copy()
        position = sim.data.xpos[sim.gripper].copy()
        target_position = sim.data.xpos[target].copy()
        target_rotation = sim.data.xmat[target].reshape(3, 3).copy()
        xml = (self.problem["root"] / "model.xml").read_text()
        children = []
        for level in LEVELS:
            reset_problem(native, problem)
            if level <= .9:
                self.put(robot_pose(original_q, closed, sim.limits[-1, 1], level))
            else:
                height = task["goal"]["lift_m"] * .5

                def residual(arm):
                    self.put(np.r_[arm, closed[-1]])
                    return np.r_[sim.data.xpos[sim.gripper] - position - [0, 0, height],
                                 .06 * (sim.data.xmat[sim.gripper].reshape(3, 3) - rotation).ravel()]

                fit = least_squares(residual, closed[:5], bounds=(sim.limits[:5, 0], sim.limits[:5, 1]),
                                    max_nfev=150, xtol=1e-12, ftol=1e-12, gtol=1e-12)
                if not fit.success or np.linalg.norm(residual(fit.x)) > .0001:
                    raise UnavailableGrasp("Cannot construct rigidly held lifted robot pose")
                self.put(np.r_[fit.x, closed[-1]])
                transform = sim.data.xmat[sim.gripper].reshape(3, 3) @ rotation.T
                center = sim.data.xpos[sim.gripper] + transform @ (target_position - position)
                quaternion = np.empty(4)
                mujoco.mju_mat2Quat(quaternion, (transform @ target_rotation).ravel())
                address = int(sim.model.jnt_qposadr[int(sim.model.body_jntadr[target])])
                sim.data.qpos[address:address + 7] = np.r_[center, quaternion]
                self.put(sim.data.qpos[sim.qadr].copy())
            metadata = self.validate(task, level)
            child = copy.deepcopy(task)
            child.update(
                task_id=hashlib.sha256(f"{scene_id}:{task['task_id']}:{level:.2f}".encode()).hexdigest(),
                parent_task_id=task["task_id"],
                initial_task_state={"lifted_once": False},
                initial_state=capture_scene_state(
                    sim.model, sim.data, xml, {"scene_id": scene_id,
                                             "catalog_sha256": problem["scene"]["catalog_sha256"]},
                    status="task-start"),
                assistance={"schema_version": 3, "kind": PROFILE, "value": level,
                            "parent_task_id": task["task_id"],
                            "construction": "Independent static reset; no execution or reference trajectory",
                            **metadata},
                start={"profile": PROFILE, **metadata},
            )
            children.append(child)
        return children


def generate(entry: dict, output: Path, *, parents: int = 1,
             task_ids: list[str] | None = None) -> dict:
    if output.exists():
        raise FileExistsError("Output exists; reconcile instead of overwriting")
    if parents < 1:
        raise ValueError("Parent quota must be positive")
    source = Path(entry["root"])
    scene = json.loads((source / "problem.json").read_text())
    tasks = [task for task in scene["tasks"]
             if task["task_type"] == "grasp" and not task.get("parent_task_id")
             and (task_ids is None or task["task_id"] in task_ids)]
    if not tasks or (task_ids is not None and {t["task_id"] for t in tasks} != set(task_ids)):
        raise ValueError("Unknown or empty original GRASP parent selection")
    tasks.sort(key=lambda t: -t["initial_target_position"][2])
    first = load_problem(entry, tasks[0]["task_id"])
    constructor = GraspConstructor(first)
    implementation = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    provenance = {
        "profile": PROFILE, "parent_entry": entry, "generator_sha256": implementation,
        "requested_parent_count": parents, "candidate_parent_ids": [t["task_id"] for t in tasks],
        "levels": list(LEVELS), "grasp_preserves_original_piece_pose": True,
        "definition": {"0_to_0.8": "linear robot joint-pose interpolation",
                       "0.8": "fully open in-place", "0.85": "half jaw closure",
                       "0.9": "bilateral in-place grasp", "0.95": "half original required lift"},
        "robot_table_clearance_sha256": hashlib.sha256(
            Path(__file__).with_name("robot_table_clearance.py").read_bytes()).hexdigest(),
    }
    scene_id = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    children, families = [], []
    accepted = 0
    try:
        for task in tasks:
            if accepted >= parents:
                break
            if any((root / "STOP").exists() for root in (output.parent, output.parent.parent)):
                raise InterruptedError("In-place GRASP construction stopped")
            family = {"parent_task_id": task["task_id"], "original_entry": entry,
                      "assistance_task_ids": [], "unavailable": []}
            try:
                variants = constructor.family(task, scene_id)
            except UnavailableGrasp as error:
                LOGGER.warning("Unavailable %s: %s", task["task_id"], error)
                family["unavailable"].append({"reason": str(error)})
            else:
                children.extend(variants)
                family["assistance_task_ids"] = [t["task_id"] for t in variants]
                accepted += 1
            families.append(family)
    finally:
        constructor.sim.close()
    if not children:
        raise UnavailableGrasp("No original-pose GRASP families constructed")
    tags = {**scene["tags"], "task_start_profile": PROFILE, "task_assistance_profile": PROFILE}
    published = {
        **scene, "scene_id": scene_id, "tasks": children, "task_families": families,
        "tags": tags, "provenance": provenance, "generator_sha256": implementation,
        "sampling_status": {"requested_parents": parents, "accepted_parents": accepted,
                            "attempted_parents": len(families), "quota_met": accepted >= parents},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".in-place-grasp-", dir=output.parent) as directory:
        staging = Path(directory) / "scene"
        staging.mkdir()
        for name in ("model.xml", "model.mjb"):
            shutil.copyfile(source / name, staging / name)
        for name in ("scene_plan.json", "integration_state.json"):
            atomic_json(staging / name, {**json.loads((source / name).read_text()), "scene_id": scene_id})
        atomic_json(staging / "problem.json", published)
        result = {**entry, "scene_id": scene_id, "root": str(output.resolve()), "tags": tags,
                  "files_sha256": {name: hashlib.sha256((staging / name).read_bytes()).hexdigest()
                                   for name in entry["files_sha256"]}}
        staged = {**result, "root": str(staging.resolve())}
        verified = create_simulation(load_problem(staged, children[0]["task_id"]))
        try:
            for child in children:
                reset_problem(verified, {"root": staging, "scene": published, "task": child})
                specification = mujoco.mjtState.mjSTATE_INTEGRATION
                state = np.empty(mujoco.mj_stateSize(verified.model, specification))
                mujoco.mj_getState(verified.model, verified.data, state, specification)
                if not np.array_equal(state, child["initial_state"]["integration_state"]):
                    raise ValueError("Full reset differs from construction")
                require_robot_above_table(verified.model, verified.data)
        finally:
            verified.close()
        atomic_json(staging / "admission.json", result)
        staging.rename(output)
    return {**result, "sampling_status": published["sampling_status"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parents", type=int, default=1)
    parser.add_argument("--task-id", action="append")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    os.nice(19)
    print(json.dumps(generate(json.loads(args.admission.read_text()), args.output,
                              parents=args.parents, task_ids=args.task_id)), flush=True)


if __name__ == "__main__":
    main()
