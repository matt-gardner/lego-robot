"""SO101 expert integration for dynamic catalog objects, not legacy beam resets."""
from __future__ import annotations

import copy
from collections.abc import Callable
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np

from so101.lego_curriculum import Catalog
from so101.lego_scene import assemble_scene, capture_scene_state, restore_scene, settle_data
from so101.renderer_appearance import RendererAppearance
from so101.scene_metrics import configuration_metrics
from so101.robot_sim import (
    BeamSimulation, DISTRACTOR_SIZES, IMAGE_SHAPE, MODEL_ROOT, OBJECT_NAMES,
    SCENE_VERSION, TABLE_Z, scene_xml,
)


class CatalogSimulation(BeamSimulation):
    """Reuse robot kinematics and labels with catalog-owned object geometry."""

    def __init__(
        self, catalog: Catalog, plan: dict, *, model_root: Path = MODEL_ROOT,
        render: bool = True, appearance: RendererAppearance | None = None,
        contact_solver: str = "model-default",
        stationarity_metric: str = "legacy-component-v1",
        velocity_tolerance: float = .005,
        before_recording_write: Callable[[], None] | None = None,
        record_images: bool = True,
    ):
        self.before_recording_write = before_recording_write
        if not record_images and render:
            raise ValueError("Trajectory-only recording must disable rendering")
        root = ET.fromstring(scene_xml(model_root, SCENE_VERSION))
        if contact_solver == "pyramidal-v1":
            option = root.find("option")
            if option is None:
                option = ET.SubElement(root, "option")
            option.set("cone", "pyramidal")
        world = root.find("worldbody")
        if world is None:
            raise ValueError("Robot base has no worldbody")
        for body in list(world.findall("body")):
            if body.get("name") in (*OBJECT_NAMES, *DISTRACTOR_SIZES):
                world.remove(body)
        self.scene_origin = np.array([.22, 0., TABLE_Z])
        self.destination_aliases = {"destination_0": "goal_left", "destination_1": "goal_right"}
        for destination in plan["destinations"]:
            body = world.find(f"body[@name='{self.destination_aliases[destination['destination_id']]}']")
            if body is None:
                raise ValueError("Robot base lacks the requested destination")
            position = self.scene_origin + destination["position_relative_to_scene_m"]
            body.set("pos", " ".join(map(str, position)))
        table = world.find("geom[@name='table']")
        if table is None:
            raise ValueError("Robot base has no table")
        table.set("solref", ".001 1")
        table.set("solimp", ".99 .999 .0001")
        table.set("priority", "1")
        self.xml = assemble_scene(ET.tostring(root, encoding="unicode"), catalog, plan,
                                  scene_origin_m=tuple(self.scene_origin))
        if appearance is not None:
            self.xml = appearance.apply_xml(self.xml)
        model = mujoco.MjModel.from_xml_string(self.xml)
        names = tuple(obj["instance_id"] for obj in plan["objects"])
        super().__init__(model_root, model=model, render=False,
                         object_names=names, distractor_names=(), record_images=record_images)
        self.catalog = catalog
        self.plan = plan
        self.appearance = appearance
        self.part_metadata = {part.part_id: part for part in catalog.parts}
        self.object_metadata = {obj["instance_id"]: obj for obj in plan["objects"]}
        self.yaws = {obj["instance_id"]: obj["yaw_rad"] for obj in plan["objects"]}
        self.target_labels = {
            obj["instance_id"]: f"{obj['color']} LEGO {obj['part_id']}"
            for obj in plan["objects"]
        }
        if appearance is not None:
            appearance.apply_model(self.model)
        home = self.ik(np.array([.20, 0., .09]), 0.)
        home[-1] = self.limits[-1, 0] + .55 * np.ptp(self.limits[-1])
        self.data.qpos[self.qadr] = home
        self.data.ctrl[self.actuators] = home
        _, _, self.settled_snapshot = settle_data(
            self.model, self.data, self.xml, plan, contact_solver=contact_solver,
            stationarity_metric=stationarity_metric, velocity_tolerance=velocity_tolerance,
        )
        self.yaws = {name: self.object_yaw(name) for name in names}
        self.initial = {
            "seed": plan["seed"], "scene_version": 4,
            "simulation_backend": "catalog-v1", "expert_acceptance_version": 1,
            "arm_state": self.state().tolist(),
            "objects": {name: self.position(name).tolist() for name in names},
            "yaws": dict(self.yaws), "qpos": self.data.qpos.tolist(),
            "qvel": self.data.qvel.tolist(), "ctrl": self.data.ctrl.tolist(),
            "appearance": {"target_labels": dict(self.target_labels)},
            "catalog_id": catalog.catalog_id, "catalog_sha256": catalog.manifest_sha256,
            "scene_id": plan["scene_id"],
            "contact_solver": self.settled_snapshot["contact_solver"],
            "stationarity": self.settled_snapshot["stationarity"],
        }
        self.initial["settled_configuration"] = self.configuration_metrics()
        if appearance is not None:
            self.initial["renderer"] = appearance.identity()
        self.contact_penetration_limit_m = .001
        self.table_penetration_limit_m = .0003
        if render:
            self.renderer = mujoco.Renderer(self.model, height=IMAGE_SHAPE[0], width=IMAGE_SHAPE[1])
        self.frame_start = self.data.time
        self.recovery_parent_example_id: str | None = None

    def reset(self, *args, **kwargs) -> None:
        raise ValueError("Catalog scenes reset through their exact settled snapshot")

    def configuration_metrics(self) -> dict:
        return configuration_metrics([
            {"instance_id": name, "position_m": self.position(name).tolist(),
             "rotation": self.data.body(name).xmat.reshape(3, 3).tolist(),
             "dimensions_m": self.part_metadata[obj["part_id"]].dimensions_m}
            for name, obj in self.object_metadata.items()
        ])

    def object_yaw(self, target: str) -> float:
        dimensions = self.part_metadata[self.object_metadata[target]["part_id"]].dimensions_m
        axis = int(np.argmax(dimensions[:2]))
        direction = self.data.body(target).xmat.reshape(3, 3)[:, axis]
        return math.atan2(direction[1], direction[0])

    def restore_initial(self) -> None:
        restore_scene(self.model, self.data, self.settled_snapshot, self.xml)
        self.rows = []
        self.episode_root = None
        self.frame_start = self.data.time
        self.active_target = None
        self.command_entities = []
        self.obstacle_contact_events = []
        self.maximum_contact_penetration_m = 0.
        self.maximum_table_penetration_m = 0.

    def use_recovery_start(self, snapshot: dict, *, parent_example_id: str) -> None:
        """Condition on a valid failed state without importing its action prefix."""
        if not parent_example_id:
            raise ValueError("Recovery requires a parent example identity")
        restore_scene(self.model, self.data, snapshot, self.xml)
        self._checked_contact_depths("recovery_initial", 0)
        hold = np.clip(self.data.qpos[self.qadr], self.limits[:, 0], self.limits[:, 1])
        self.data.ctrl[self.actuators] = hold
        identity = hashlib.sha256(json.dumps(
            [parent_example_id, snapshot["integration_state"]],
            separators=(",", ":"), allow_nan=False,
        ).encode()).hexdigest()
        self.plan = copy.deepcopy(self.plan)
        self.plan["scene_family_id"] = self.plan.get("scene_family_id", self.plan["scene_id"])
        self.plan["scene_id"] = identity
        self.plan["pair_id"] = f"{identity}:{self.plan['task']}"
        self.plan["tags"]["example_kind"] = "post-failure-recovery"
        for index, command in enumerate(self.plan["commands"]):
            command["command_id"] = f"{identity}:{index}"
        self.settled_snapshot = capture_scene_state(
            self.model, self.data, self.xml, self.plan, status="recovery-start",
        )
        self.initial = copy.deepcopy(self.initial)
        self.initial["settled_configuration"] = self.configuration_metrics()
        self.yaws = {name: self.object_yaw(name) for name in self.object_metadata}
        self.initial.update(
            scene_id=identity, scene_family_id=self.plan["scene_family_id"],
            recovery_parent_example_id=parent_example_id,
            arm_state=self.state().tolist(),
            objects={name: self.position(name).tolist() for name in self.object_metadata},
            yaws=dict(self.yaws),
            qpos=self.data.qpos.tolist(), qvel=self.data.qvel.tolist(),
            ctrl=self.data.ctrl.tolist(),
        )
        self.recovery_parent_example_id = parent_example_id

    def observe_recovery_start(self, target: str, task: str, destination: str | None) -> None:
        if self.recovery_parent_example_id is None:
            return
        entities = [(target, "object")]
        if destination is not None:
            entities.append((self.destination_aliases[destination], "destination"))
        self.configure_command(task, entities)
        hold = np.clip(self.data.qpos[self.qadr], self.limits[:, 0], self.limits[:, 1])
        for _ in range(4):
            self.step(hold, "recovery_observe")

    def position(self, name: str) -> np.ndarray:
        if hasattr(self, "object_metadata") and name in self.object_metadata:
            return self.data.body(name).xipos.copy()
        return super().position(name)

    def ik(self, position: np.ndarray, yaw: float, *, rng=None) -> np.ndarray:
        return super().ik(position, yaw, rng=rng,
                          position_tolerance=.0002, rotation_tolerance=.01)

    def push_ik(self, position: np.ndarray, wrist_roll: float) -> np.ndarray:
        return super().push_ik(position, wrist_roll,
                               position_tolerance=.0002, rotation_tolerance=.01)

    def step(self, target: np.ndarray, phase: str):
        if self.episode_root is None or not self.record_images:
            if self.episode_root is not None and self.before_recording_write is not None:
                self.before_recording_write()
            return super().step(target, phase)
        if self.before_recording_write is not None:
            self.before_recording_write()
        index = len(self.rows)
        relative = f"external/{index:06}.jpg"
        if not cv2.imwrite(str(self.episode_root / relative),
                           cv2.cvtColor(self.overview_image(), cv2.COLOR_RGB2BGR)):
            raise OSError("Cannot record external catalog-camera frame")
        try:
            return super().step(target, phase)
        finally:
            if len(self.rows) > index:
                self.rows[index]["external_image"] = relative

    def _open_grasp_pose(self, target: str, center: np.ndarray, height: float, yaw: float):
        xy = center[:2].copy()
        target_body = self.model.body(target).id
        for _ in range(8):
            grasp = self.ik(np.r_[xy, height], yaw)
            grasp[-1] = self.limits[-1, 0] + .55 * np.ptp(self.limits[-1])
            self.scratch.qpos[:] = self.data.qpos
            self.scratch.qpos[self.qadr] = grasp
            mujoco.mj_forward(self.model, self.scratch)
            collisions = []
            for contact in self.scratch.contact:
                bodies = [int(self.model.geom_bodyid[g])
                          for g in (contact.geom1, contact.geom2)]
                if (target_body in bodies and self.gripper in bodies
                        and contact.dist < .0002):
                    normal = contact.frame[:2].copy()
                    if bodies[0] == target_body:
                        normal *= -1
                    collisions.append((float(contact.dist), normal))
            if not collisions:
                hover = self.ik(np.r_[xy, max(.065, height + .05)], yaw)
                hover[-1] = grasp[-1]
                return xy, hover, grasp
            depth, normal = min(collisions, key=lambda item: item[0])
            squared_length = float(normal @ normal)
            if squared_length < .25:
                raise RuntimeError("Fixed-jaw collision cannot be cleared by lateral alignment")
            xy -= normal * ((.0002 - depth) / squared_length)
            if np.linalg.norm(xy - center[:2]) > .02:
                raise RuntimeError("Fixed-jaw clearance exceeds 20 mm grasp alignment budget")
        raise RuntimeError("Cannot align an open fixed jaw clear of the catalog object")

    def _grasp(self, target: str, *, lift_height: float = .07, approach_waypoints=()):
        if approach_waypoints:
            raise ValueError("Catalog obstacle routing is not implemented")
        body = self.data.body(target)
        rotation = body.xmat.reshape(3, 3)
        dimensions = np.array(self.part_metadata[self.object_metadata[target]["part_id"]].dimensions_m)
        yaw = self.object_yaw(target) + math.pi / 2
        center = self.position(target)
        half_height = float(np.sum(np.abs(rotation[2]) * dimensions) / 2)
        contact_height = center[2] + half_height + .0024
        destination_xy = None
        if self.command_phrase == "place":
            destinations = [name for name, role in self.command_entities
                            if role == "destination"]
            if len(destinations) != 1:
                raise ValueError("Catalog place requires exactly one destination")
            destination_xy = self.position(destinations[0])[:2]
        failures = []
        for heading in (yaw, yaw + math.pi):
            try:
                grasp_xy, hover, grasp = self._open_grasp_pose(target, center, contact_height, heading)
                self.ik(np.r_[grasp_xy, max(lift_height, contact_height + .06)], heading)
                if destination_xy is not None:
                    for height in (.070, .018, .080):
                        self.ik(np.r_[destination_xy, height], heading)
            except RuntimeError as error:
                failures.append(str(error))
                continue
            yaw = heading
            break
        else:
            raise RuntimeError("No reachable parallel-jaw grasp heading: " + "; ".join(failures))
        self.yaws[target] = yaw - math.pi / 2
        hover[-1] = self.limits[-1, 0] + .55 * np.ptp(self.limits[-1])
        self.move(hover, "approach")
        grasp[-1] = hover[-1]
        self.move(grasp, "descend")
        closed = grasp.copy()
        closed[-1] = self.limits[-1, 0]
        self.move(closed, "close")
        pinch = sorted(self.contacts(target))
        if pinch != ["fixed", "moving"]:
            return closed, {"success": False, "reason": "No two-jaw catalog-object contact",
                            "pinch": pinch}
        lift = self.ik(np.r_[self.data.site_xpos[self.site, :2],
                            max(lift_height, contact_height + .06)], yaw)
        lift[-1] = closed[-1]
        self.move(lift, "lift")
        return lift, {"success": True, "pinch": pinch}

    def demonstrate_view(self, target: str) -> dict:
        position = self.position(target)
        yaw = self.object_yaw(target) + math.pi / 2
        failures = []
        for heading in (yaw, yaw + math.pi):
            try:
                self.ik(np.r_[position[:2], .070], heading)
                self.ik(np.r_[position[:2] + [.010, -.010], .065], heading + .12)
            except RuntimeError as error:
                failures.append(str(error))
                continue
            self.yaws[target] = heading - math.pi / 2
            return super().demonstrate_view(target)
        raise RuntimeError("No reachable two-pose view heading: " + "; ".join(failures))

    def demonstrate_push(self, target: str, destination: str) -> dict:
        current = self.position(target)
        finish_xy = self.position(destination)[:2]
        direction = finish_xy - current[:2]
        distance = float(np.linalg.norm(direction))
        if distance > .055:
            behind = current[:2] - direction / distance * .025
            wrist_roll = float(self.data.qpos[self.qadr[4]])
            for position in (np.r_[behind, .055], np.r_[behind, TABLE_Z + .014],
                             np.r_[finish_xy, TABLE_Z + .014], np.r_[finish_xy, .065]):
                self.push_ik(position, wrist_roll)
        return super().demonstrate_push(target, destination)

    def demonstrate_scene_command(self, command: dict) -> dict:
        if command["task"] != self.plan["task"]:
            raise ValueError("Command task does not match scene plan")
        target = command["object_instance_id"]
        if target not in self.object_metadata:
            raise ValueError("Command references an unknown catalog object")
        destination = command["destination_id"]
        if destination is not None:
            destination = self.destination_aliases[destination]
        return self.demonstrate_command(command["task"], target, destination)
