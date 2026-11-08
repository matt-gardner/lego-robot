"""SO101 expert integration for dynamic catalog objects, not legacy beam resets."""
from __future__ import annotations

import copy
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
from so101.robot_sim import (
    BeamSimulation, DISTRACTOR_SIZES, IMAGE_SHAPE, MODEL_ROOT, OBJECT_NAMES,
    SCENE_VERSION, TABLE_Z, scene_xml,
)


class CatalogSimulation(BeamSimulation):
    """Reuse robot kinematics and labels with catalog-owned object geometry."""

    def __init__(
        self, catalog: Catalog, plan: dict, *, model_root: Path = MODEL_ROOT,
        render: bool = True, appearance: RendererAppearance | None = None,
    ):
        root = ET.fromstring(scene_xml(model_root, SCENE_VERSION))
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
                         object_names=names, distractor_names=())
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
        _, _, self.settled_snapshot = settle_data(self.model, self.data, self.xml, plan)
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
        }
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
        self.yaws = {}
        for name, obj in self.object_metadata.items():
            dimensions = self.part_metadata[obj["part_id"]].dimensions_m
            axis = int(np.argmax(dimensions[:2]))
            direction = self.data.body(name).xmat.reshape(3, 3)[:, axis]
            self.yaws[name] = math.atan2(direction[1], direction[0])
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

    def step(self, target: np.ndarray, phase: str):
        if self.episode_root is None:
            return super().step(target, phase)
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

    def _grasp(self, target: str, *, lift_height: float = .07, approach_waypoints=()):
        if approach_waypoints:
            raise ValueError("Catalog obstacle routing is not implemented")
        body = self.data.body(target)
        rotation = body.xmat.reshape(3, 3)
        dimensions = np.array(self.part_metadata[self.object_metadata[target]["part_id"]].dimensions_m)
        axis = int(np.argmax(dimensions[:2]))
        direction = rotation[:, axis]
        yaw = math.atan2(direction[1], direction[0]) + math.pi / 2
        center = self.position(target)
        half_height = float(np.sum(np.abs(rotation[2]) * dimensions) / 2)
        contact_height = center[2] + half_height + .0024
        failures = []
        for heading in (yaw, yaw + math.pi):
            try:
                hover = self.ik(np.r_[center[:2], max(.065, contact_height + .05)], heading)
                grasp = self.ik(np.r_[center[:2], contact_height], heading)
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
