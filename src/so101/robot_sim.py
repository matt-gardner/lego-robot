"""Hardware-free SO-101 gray-beam demonstrations using MuJoCo contact physics."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
from urllib.request import urlopen
import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np

from so101.control_model.dataset import oriented_box_from_mask
from so101.robot_control import JOINTS

MODEL_COMMIT = "f3b6013629c7b94f8ee93fd977b161fa8350f9bc"
MODEL_URL = (
    "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/"
    f"{MODEL_COMMIT}/robotstudio_so101/"
)
MODEL_ROOT = Path("data/simulation/model")
TABLE_Z = -0.003
BEAM_HALF_HEIGHT = 0.0048
IMAGE_SHAPE = (240, 320, 3)
POLICY_ARM_CORRECTION_LIMIT = 5.
PROJECTED_CENTER_RANGE = (-4., 5.)
TASK = "Pick up the {target} gray beam."
OBJECT_NAMES = ("studded", "hollow", "short", "wide")
DESTINATION_NAMES = ("goal_left", "goal_right")
OBSTACLE_NAMES = ("obstacle_box",)
TARGET_COLORS = {
    "teal": np.array([.02, .55, .58]),
    "orange": np.array([.93, .34, .04]),
    "navy": np.array([.03, .10, .32]),
    "green": np.array([.04, .48, .12]),
    "brown": np.array([.36, .13, .05]),
    "gray": np.array([.38, .39, .40]),
}


def restore_episode(sim: BeamSimulation, metadata: dict) -> tuple[str, str | None, list[tuple[str, str]]]:
    initial = metadata["initial"]
    appearance = initial.get("appearance", {})
    labels = appearance.get("target_labels", {})
    color_targets = metadata.get("color_targets")
    if color_targets is None:
        color_targets = any(labels.get(name, name) != name for name in OBJECT_NAMES)
    sim.reset(
        int(initial["seed"]),
        varied_layout=bool(initial.get("varied_layout", True)),
        real_scene=bool(appearance.get("real_scene", True)),
        color_targets=bool(color_targets),
        visual_cue_target=bool(metadata.get(
            "visual_cue_target", appearance.get("visual_cue_target", False),
        )),
    )
    sim.yaws = {name: float(yaw) for name, yaw in initial["yaws"].items()}
    for name, position in initial["objects"].items():
        address = sim.model.joint(f"{name}_free").qposadr[0]
        yaw = sim.yaws[name]
        sim.data.qpos[address:address + 7] = [
            *position, math.cos(yaw / 2), 0, 0, math.sin(yaw / 2),
        ]
    arm = sim.decode(np.asarray(initial["arm_state"], dtype=np.float64))
    sim.data.qpos[sim.qadr] = arm
    sim.data.ctrl[sim.actuators] = arm
    if "light_diffuse" in appearance:
        sim.model.light_diffuse[:] = appearance["light_diffuse"]
    if "light_ambient" in appearance:
        sim.model.light_ambient[:] = appearance["light_ambient"]
    if "light_castshadow" in appearance:
        sim.model.light_castshadow[:] = appearance["light_castshadow"]
    if "table_rgba" in appearance:
        sim.model.geom_rgba[sim.model.geom("table").id] = appearance["table_rgba"]
    for name, position in appearance.get("distractors", {}).items():
        sim.model.body_pos[sim.distractors[name]] = position
    for name, position in appearance.get("destinations", {}).items():
        sim.model.body_pos[sim.model.body(name).id] = position
    for name, position in appearance.get("obstacles", {}).items():
        sim.model.body_pos[sim.model.body(name).id] = position
    sim.target_labels = dict(labels)
    for name, label in labels.items():
        color = label.split()[0]
        if color in TARGET_COLORS:
            sim.model.geom_rgba[sim.beam_geoms[name], :3] = TARGET_COLORS[color]
    mujoco.mj_forward(sim.model, sim.data)
    sim.initial = initial
    sim.hide_unseen_centers = bool(metadata.get("perception", {}).get("hide_unseen_centers", False))
    sim.episode_constraints = dict(metadata.get("constraints", {}))
    target = str(metadata["target"])
    destination = metadata.get("destination")
    entities = [(target, "object")]
    if destination is not None:
        entities.append((str(destination), "destination"))
    entities.extend((str(obstacle), "obstacle") for obstacle in metadata.get("obstacles", []))
    sim.configure_command(str(metadata["task"]), entities)
    return target, destination, entities


def fetch_model(root: Path) -> None:
    """Fetch only the pinned Apache-2.0 model and its referenced assets."""
    root.mkdir(parents=True, exist_ok=True)

    def download(name: str) -> bytes:
        with urlopen(MODEL_URL + name, timeout=60) as response:
            content = response.read()
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".download")
        temporary.write_bytes(content)
        temporary.replace(path)
        return content

    xml = download("so101.xml")
    meshes = [mesh.attrib["file"] for mesh in ET.fromstring(xml).findall("asset/mesh")]
    if any(Path(name).name != name for name in meshes):
        raise ValueError("Unexpected upstream asset path")
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download, ["LICENSE", "README.md", *[f"assets/{name}" for name in meshes]]))
    (root / "revision.json").write_text(json.dumps({"commit": MODEL_COMMIT, "url": MODEL_URL}))


def scene_xml(root: Path) -> str:
    revision = json.loads((root / "revision.json").read_text())
    if revision["commit"] != MODEL_COMMIT:
        raise ValueError("Model revision mismatch; rerun robot_sim.py setup")
    tree = ET.parse(root / "so101.xml")
    xml = tree.getroot()
    xml.find("compiler").set("meshdir", str((root / "assets").resolve()))
    xml.find("option").set("timestep", ".002")
    xml.find("option").set("iterations", "50")
    for name in ("collision_gripper", "collision_gripper_mesh"):
        geom = xml.find(f"default/default/default[@class='{name}']/geom")
        geom.set("solref", ".004 1")
        geom.set("solimp", ".95 .99 .001")
    for material in xml.findall("asset/material"):
        if material.get("rgba") == "1 0.82 0.12 1":
            material.set("rgba", ".9 .9 .88 1")
    world = xml.find("worldbody")
    ET.SubElement(
        world, "light", pos="0.1 0 0.8", dir="0 0 -1",
        ambient=".55 .55 .55", diffuse=".65 .65 .65", castshadow="false",
    )
    ET.SubElement(world, "geom", name="table", type="plane", size=".6 .6 .01",
                  pos=f"0 0 {TABLE_Z}", rgba=".82 .82 .79 1", friction=".7 .005 .0001")
    for name in OBJECT_NAMES:
        body = ET.SubElement(world, "body", name=name)
        ET.SubElement(body, "freejoint", name=f"{name}_free")
        common = {"rgba": ".38 .39 .40 1", "friction": ".8 .005 .0001", "density": "1040"}
        if name == "studded":
            ET.SubElement(body, "geom", name="studded_shell", type="box",
                          size=".064 .008 .0048", **common)
            for i in range(16):
                for j in range(2):
                    ET.SubElement(body, "geom", type="cylinder", size=".0024 .0009",
                                  pos=f"{(i - 7.5) * .008} {(j - .5) * .008} .0057",
                                  contype="0", conaffinity="0", mass="0", **common)
        elif name == "hollow":
            for part, size, pos in (
                ("base", ".064 .008 .0012", "0 0 -.0036"),
                ("side_a", ".064 .0012 .0036", "0 .0068 .0012"),
                ("side_b", ".064 .0012 .0036", "0 -.0068 .0012"),
                ("end_a", ".0012 .0056 .0036", ".0628 0 .0012"),
                ("end_b", ".0012 .0056 .0036", "-.0628 0 .0012"),
            ):
                ET.SubElement(body, "geom", name=f"hollow_{part}", type="box",
                              size=size, pos=pos, **common)
        elif name == "short":
            ET.SubElement(body, "geom", name="short_shell", type="box",
                          size=".045 .008 .0048", **common)
        else:
            ET.SubElement(body, "geom", name="wide_shell", type="box",
                          size=".050 .011 .0048", **common)
    for name, rgba in (
        ("goal_left", ".72 .18 .65 1"),
        ("goal_right", ".15 .42 .88 1"),
    ):
        body = ET.SubElement(world, "body", name=name, pos="0 0 -1")
        ET.SubElement(
            body, "geom", name=f"{name}_geom", type="cylinder",
            size=".027 .001", rgba=rgba, contype="0", conaffinity="0",
        )
    for name, size, rgba in (
        ("green_plate_a", ".11 .045 .002", ".05 .38 .12 1"),
        ("green_plate_b", ".075 .04 .002", ".04 .32 .10 1"),
        ("green_piece", ".025 .012 .006", ".03 .34 .09 1"),
        ("gray_clutter", ".055 .007 .0048", ".38 .39 .40 1"),
    ):
        body = ET.SubElement(world, "body", name=name, pos="0 0 -1")
        ET.SubElement(
            body, "geom", name=f"{name}_geom", type="box", size=size, rgba=rgba,
            contype="0", conaffinity="0",
        )
    obstacle = ET.SubElement(
        world, "body", name="obstacle_box", pos="0 0 -1"
    )
    ET.SubElement(
        obstacle,
        "geom",
        name="obstacle_box_geom",
        type="box",
        size=".025 .028 .045",
        rgba=".72 .12 .08 1",
        friction=".9 .01 .001",
    )
    gripper = world.find(".//body[@name='gripper']")
    ET.SubElement(gripper, "site", name="grasp", pos="0 0 -.096", size=".002", group="3")
    return ET.tostring(xml, encoding="unicode")


class BeamSimulation:
    """The expert sees object poses; exported policy observations never do."""

    def __init__(self, model_root: Path = MODEL_ROOT, *, render: bool = True):
        self.model = mujoco.MjModel.from_xml_string(scene_xml(model_root))
        self.data = mujoco.MjData(self.model)
        self.scratch = mujoco.MjData(self.model)
        self.joints = np.array([self.model.joint(name).id for name in JOINTS])
        self.qadr = self.model.jnt_qposadr[self.joints]
        self.dofs = self.model.jnt_dofadr[self.joints]
        self.actuators = np.array([self.model.actuator(name).id for name in JOINTS])
        self.limits = self.model.jnt_range[self.joints].copy()
        self.model.actuator_forcerange[self.actuators[-1]] *= .5
        self.site = self.model.site("grasp").id
        self.gripper = self.model.body("gripper").id
        self.moving_jaw = self.model.body("moving_jaw_so101_v1").id
        self.renderer = mujoco.Renderer(self.model, height=IMAGE_SHAPE[0], width=IMAGE_SHAPE[1]) if render else None
        self.rows: list[dict] = []
        self.episode_root: Path | None = None
        self.fps = 10
        self.frame_start = 0.
        self.initial = {}
        self.visual_cue_target = False
        self.hide_unseen_centers = False
        self.episode_constraints: dict[str, Any] = {}
        self.active_target: str | None = None
        self.command_phrase = "grasp"
        self.command_entities: list[tuple[str, str]] = []
        self.base_material_colors = self.model.mat_rgba.copy()
        self.base_geom_colors = self.model.geom_rgba.copy()
        self.beam_geoms = {
            name: np.flatnonzero(self.model.geom_bodyid == self.model.body(name).id)
            for name in OBJECT_NAMES
        }
        shell = self.model.geom("studded_shell").id
        self.studded_studs = self.beam_geoms["studded"][
            self.beam_geoms["studded"] != shell
        ]
        self.distractors = {
            name: self.model.body(name).id
            for name in ("green_plate_a", "green_plate_b", "green_piece", "gray_clutter")
        }
        self.destination_geoms = {
            name: np.array([self.model.geom(f"{name}_geom").id])
            for name in DESTINATION_NAMES
        }
        self.obstacle_geoms = {
            name: np.array([self.model.geom(f"{name}_geom").id])
            for name in OBSTACLE_NAMES
        }
        self.active_obstacles: list[str] = []
        self.obstacle_contact_events: list[dict[str, Any]] = []

    def close(self):
        if self.renderer is not None:
            self.renderer.close()

    def state(self) -> np.ndarray:
        return self.encode(self.data.qpos[self.qadr])

    def encode(self, radians: np.ndarray) -> np.ndarray:
        result = np.rad2deg(radians).astype(np.float32)
        low, high = self.limits[-1]
        result[-1] = 100 * (radians[-1] - low) / (high - low)
        return result

    def decode(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action)
        if action.shape != (6,) or not np.isfinite(action).all() or not 0 <= action[-1] <= 100:
            raise ValueError("Expected six finite joint targets, with gripper in [0, 100]")
        radians = np.deg2rad(action).astype(np.float64)
        low, high = self.limits[-1]
        radians[-1] = low + action[-1] / 100 * (high - low)
        if np.any(radians < self.limits[:, 0] - 1e-6) or np.any(radians > self.limits[:, 1] + 1e-6):
            raise ValueError("Action exceeds simulated joint limits")
        return radians

    def bounded_policy_action(self, action: np.ndarray) -> tuple[np.ndarray, float]:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (6,) or not np.isfinite(action).all():
            raise ValueError("Expected six finite policy joint targets")
        bounded = np.clip(action, self.encode(self.limits[:, 0]), self.encode(self.limits[:, 1]))
        correction = float(np.max(np.abs(action - bounded)))
        arm_correction = float(np.max(np.abs(action[:5] - bounded[:5])))
        if arm_correction > POLICY_ARM_CORRECTION_LIMIT:
            raise ValueError(f"Policy target exceeds a joint boundary by {arm_correction:.3f} units")
        return self.decode(bounded), correction

    def image(self) -> np.ndarray:
        if self.renderer is None:
            raise RuntimeError("Rendering was not enabled")
        self.renderer.update_scene(self.data, camera="wrist_cam")
        image = self.renderer.render().copy()
        if self.visual_cue_target:
            if self.active_target is None:
                raise RuntimeError("Visual target cue requested without an active target")
            mask = self.target_mask(self.active_target)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = max(contours, key=cv2.contourArea)
                x, y, width, height = cv2.boundingRect(contour)
                cv2.rectangle(image, (x, y), (x + width, y + height), (255, 255, 255), 2)
                moments = cv2.moments(contour)
                if moments["m00"]:
                    center = (
                        round(moments["m10"] / moments["m00"]),
                        round(moments["m01"] / moments["m00"]),
                    )
                    cv2.circle(image, center, 4, (255, 255, 255), -1)
            else:
                cv2.putText(
                    image, "TARGET OCCLUDED", (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    .45, (255, 255, 255), 1, cv2.LINE_AA,
                )
        return image

    def entity_mask(self, entity: str) -> np.ndarray:
        if self.renderer is None:
            raise RuntimeError("Rendering was not enabled")
        geoms = self.beam_geoms.get(entity)
        if geoms is None:
            geoms = self.destination_geoms.get(entity)
        if geoms is None:
            geoms = self.obstacle_geoms.get(entity)
        if geoms is None:
            raise ValueError(f"Unknown entity: {entity}")
        self.renderer.update_scene(self.data, camera="wrist_cam")
        self.renderer.enable_segmentation_rendering()
        try:
            segmentation = self.renderer.render().copy()
        finally:
            self.renderer.disable_segmentation_rendering()
        return np.isin(segmentation[:, :, 0], geoms).astype(np.uint8)

    def target_mask(self, target: str) -> np.ndarray:
        return self.entity_mask(target)

    def target_box(self, target: str | None = None) -> list[float]:
        target = self.active_target if target is None else target
        if target is None:
            raise RuntimeError("Target box requested without an active target")
        mask = self.target_mask(target)
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return [0., 0., 0., 0., 0.]
        height, width = mask.shape
        x1, x2 = float(xs.min()) / width, float(xs.max() + 1) / width
        y1, y2 = float(ys.min()) / height, float(ys.max() + 1) / height
        return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, 1.]

    def target_oriented_box(self, target: str | None = None) -> list[float]:
        target = self.active_target if target is None else target
        if target is None:
            raise RuntimeError("Target box requested without an active target")
        box = list(oriented_box_from_mask(self.entity_mask(target)))
        center, valid = self.projected_entity_center(target)
        if valid:
            box[:2] = center
        return box

    def projected_entity_center(
        self, target: str
    ) -> tuple[list[float], bool]:
        if (
            target not in self.beam_geoms
            and target not in self.destination_geoms
            and target not in self.obstacle_geoms
        ):
            raise ValueError(f"Unknown target: {target}")
        if self.hide_unseen_centers and not self.entity_mask(target).any():
            return [0., 0.], False
        camera_id = self.model.camera("wrist_cam").id
        camera_position = self.data.cam_xpos[camera_id]
        camera_rotation = self.data.cam_xmat[camera_id].reshape(3, 3)
        camera_point = camera_rotation.T @ (
            self.position(target) - camera_position
        )
        depth = float(-camera_point[2])
        if not np.isfinite(camera_point).all() or depth <= 0:
            return [0., 0.], False
        half_height = depth * math.tan(
            math.radians(float(self.model.cam_fovy[camera_id])) / 2
        )
        half_width = half_height * IMAGE_SHAPE[1] / IMAGE_SHAPE[0]
        center = [
            .5 + float(camera_point[0]) / (2 * half_width),
            .5 - float(camera_point[1]) / (2 * half_height),
        ]
        return np.clip(
            center,
            PROJECTED_CENTER_RANGE[0],
            PROJECTED_CENTER_RANGE[1],
        ).tolist(), True

    def entity_camera_depth(self, target: str) -> tuple[float, bool]:
        if (
            target not in self.beam_geoms
            and target not in self.destination_geoms
            and target not in self.obstacle_geoms
        ):
            raise ValueError(f"Unknown target: {target}")
        if self.hide_unseen_centers and not self.entity_mask(target).any():
            return 0., False
        camera_id = self.model.camera("wrist_cam").id
        camera_position = self.data.cam_xpos[camera_id]
        camera_rotation = self.data.cam_xmat[camera_id].reshape(3, 3)
        camera_point = camera_rotation.T @ (self.position(target) - camera_position)
        depth = float(-camera_point[2])
        return max(depth, 1e-4), bool(np.isfinite(depth) and depth > 0)

    def target_depth(self, target: str | None = None) -> float:
        target = self.active_target if target is None else target
        if target is None:
            raise RuntimeError("Target depth requested without an active target")
        depth, valid = self.entity_camera_depth(target)
        if not valid:
            raise RuntimeError(f"Invalid camera depth for {target}: {depth}")
        return depth

    def configure_command(
        self, phrase: str, entities: list[tuple[str, str]]
    ) -> None:
        if phrase not in ("grasp", "view", "push", "place"):
            raise ValueError(f"Unsupported command phrase: {phrase}")
        if not entities or len(entities) > 4:
            raise ValueError("Commands require one to four entities")
        for entity, role in entities:
            if (
                entity not in self.beam_geoms
                and entity not in self.destination_geoms
                and entity not in self.obstacle_geoms
            ):
                raise ValueError(f"Unknown command entity: {entity}")
            if role not in ("object", "destination", "reference", "obstacle"):
                raise ValueError(f"Unsupported entity role: {role}")
        self.command_phrase = phrase
        self.command_entities = list(entities)
        objects = [entity for entity, role in entities if role == "object"]
        self.active_target = objects[0] if objects else entities[0][0]
        self.active_obstacles = [
            entity for entity, role in entities if role == "obstacle"
        ]

    def overview_image(self) -> np.ndarray:
        if self.renderer is None:
            raise RuntimeError("Rendering was not enabled")
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [.17, 0, .07]
        camera.distance, camera.azimuth, camera.elevation = .65, 135, -35
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()

    def overview(self, path: Path):
        if not cv2.imwrite(str(path), cv2.cvtColor(self.overview_image(), cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"Failed to save {path}")

    def position(self, name: str) -> np.ndarray:
        return self.data.body(name).xpos.copy()

    def reset(
        self, seed: int, *, varied_layout: bool = False, real_scene: bool = False,
        color_targets: bool = False, visual_cue_target: bool = False,
    ):
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.yaws = {}
        self.visual_cue_target = visual_cue_target
        self.hide_unseen_centers = False
        self.episode_constraints = {}
        self.active_target = None
        self.command_entities = []
        self.active_obstacles = []
        self.obstacle_contact_events = []
        side = int(rng.choice([-1, 1])) if varied_layout else 1
        lateral_jitter = .012 if varied_layout else .008
        slots = [
            (.220, side * -.045),
            (.220, side * .045),
            (.205, side * -.135),
            (.205, side * .135),
        ]
        for index, name in enumerate(OBJECT_NAMES):
            address = self.model.joint(f"{name}_free").qposadr[0]
            yaw = rng.uniform(-.45, .45)
            self.yaws[name] = yaw
            x, y = slots[index]
            self.data.qpos[address:address + 7] = [
                x + rng.uniform(-lateral_jitter, lateral_jitter),
                y + rng.uniform(-lateral_jitter, lateral_jitter),
                TABLE_Z + BEAM_HALF_HEIGHT + .001, math.cos(yaw / 2), 0, 0, math.sin(yaw / 2),
            ]
        mujoco.mj_forward(self.model, self.data)
        home = np.array([rng.uniform(.21, .24), rng.uniform(-.035, .035), rng.uniform(.06, .075)])
        q = self.ik(home, 0, rng=rng)
        q[-1] = self.limits[-1, 0] + .55 * np.ptp(self.limits[-1])
        self.data.qpos[self.qadr] = q
        self.data.ctrl[self.actuators] = q
        mujoco.mj_forward(self.model, self.data)
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)
        self.rows = []
        self.frame_start = self.data.time
        self.initial = {
            "seed": seed, "varied_layout": varied_layout, "arm_state": self.state().tolist(),
            "objects": {name: self.position(name).tolist() for name in self.yaws},
            "yaws": self.yaws,
        }
        self.model.mat_rgba[:] = self.base_material_colors
        self.model.geom_rgba[:] = self.base_geom_colors
        self.model.mat_rgba[:, :3] *= rng.uniform(.9, 1.03)
        self.model.light_diffuse[:] = rng.uniform(.5, .95)
        self.model.light_ambient[:] = rng.uniform(.1, .7)
        self.model.light_castshadow[:] = rng.random() < .5
        table_range = (.90, .99) if real_scene else (.74, .88)
        self.model.geom_rgba[self.model.geom("table").id, :3] = rng.uniform(*table_range)
        if color_targets:
            color_names = rng.choice(list(TARGET_COLORS), size=len(OBJECT_NAMES), replace=False)
            self.target_labels = {
                name: f"{color} {name} beam"
                for name, color in zip(OBJECT_NAMES, color_names, strict=True)
            }
            for name, color in zip(OBJECT_NAMES, color_names, strict=True):
                self.model.geom_rgba[self.beam_geoms[name], :3] = TARGET_COLORS[color]
            if rng.random() < .5:
                self.model.geom_rgba[self.studded_studs, 3] = 0
        else:
            self.target_labels = {name: name for name in OBJECT_NAMES}
            all_beam_geoms = np.concatenate(list(self.beam_geoms.values()))
            self.model.geom_rgba[all_beam_geoms, :3] = rng.uniform(.32, .43)
        distractor_layout = {
            "green_plate_a": (.16, -.13, TABLE_Z + .002),
            "green_plate_b": (.28, .105, TABLE_Z + .002),
            "green_piece": (.22, -.09, TABLE_Z + .006),
            "gray_clutter": (.16, .12, TABLE_Z + BEAM_HALF_HEIGHT),
        }
        for name, body_id in self.distractors.items():
            if real_scene:
                x, y, z = distractor_layout[name]
                self.model.body_pos[body_id] = [
                    x + rng.uniform(-.015, .015),
                    y + rng.uniform(-.015, .015),
                    z,
                ]
                yaw = rng.uniform(-.5, .5)
                self.model.body_quat[body_id] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
                if color_targets:
                    color = rng.choice([
                        name for name in TARGET_COLORS if name not in color_names
                    ])
                    self.model.geom_rgba[self.model.geom(f"{name}_geom").id, :3] = TARGET_COLORS[color]
            else:
                self.model.body_pos[body_id] = [0, 0, -1]
        goal_positions = {
            "goal_left": (.155 + rng.uniform(-.012, .012), -.100 + rng.uniform(-.008, .008)),
            "goal_right": (.155 + rng.uniform(-.012, .012), .100 + rng.uniform(-.008, .008)),
        }
        for name, (x, y) in goal_positions.items():
            self.model.body_pos[self.model.body(name).id] = [x, y, TABLE_Z + .001]
        for name in OBSTACLE_NAMES:
            self.model.body_pos[self.model.body(name).id] = [0, 0, -1]
        mujoco.mj_forward(self.model, self.data)
        self.initial["appearance"] = {
            "light_diffuse": self.model.light_diffuse.tolist(),
            "light_ambient": self.model.light_ambient.tolist(),
            "light_castshadow": self.model.light_castshadow.astype(bool).tolist(),
            "table_rgba": self.model.geom_rgba[self.model.geom("table").id].tolist(),
            "real_scene": real_scene,
            "distractors": {
                name: self.model.body_pos[body_id].tolist()
                for name, body_id in self.distractors.items()
            },
            "target_labels": self.target_labels,
            "visual_cue_target": visual_cue_target,
            "destinations": {
                name: self.position(name).tolist() for name in DESTINATION_NAMES
            },
            "obstacles": {
                name: self.position(name).tolist() for name in OBSTACLE_NAMES
            },
        }

    def place_grasp_obstacle(
        self,
        target: str,
        obstacle: str = "obstacle_box",
    ) -> None:
        if target not in self.beam_geoms:
            raise ValueError(f"Unknown grasp target: {target}")
        if obstacle not in self.obstacle_geoms:
            raise ValueError(f"Unknown obstacle: {obstacle}")
        target_position = self.position(target)
        if abs(target_position[1]) < .08:
            raise ValueError("Obstacle grasps require an outer target")
        body = self.model.body(obstacle).id
        self.model.body_pos[body] = [
            target_position[0],
            target_position[1] * .40,
            TABLE_Z + .045,
        ]
        self.model.body_quat[body] = [1, 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)
        self.active_obstacles = [obstacle]
        self.initial["appearance"]["obstacles"][obstacle] = (
            self.position(obstacle).tolist()
        )

    def obstacle_robot_contacts(self) -> list[str]:
        obstacle_geom_ids = {
            int(geom)
            for name in self.active_obstacles
            for geom in self.obstacle_geoms[name]
        }
        if not obstacle_geom_ids:
            return []
        excluded_bodies = {
            0,
            *(self.model.body(name).id for name in OBJECT_NAMES),
            *(self.model.body(name).id for name in DESTINATION_NAMES),
            *(self.model.body(name).id for name in OBSTACLE_NAMES),
        }
        contacts = set()
        for contact in self.data.contact:
            geoms = (int(contact.geom1), int(contact.geom2))
            if not obstacle_geom_ids.intersection(geoms):
                continue
            other_geom = (
                geoms[1] if geoms[0] in obstacle_geom_ids else geoms[0]
            )
            other_body = int(self.model.geom_bodyid[other_geom])
            if other_body not in excluded_bodies:
                contacts.add(self.model.body(other_body).name)
        return sorted(contacts)

    def ik(self, position: np.ndarray, yaw: float, *, rng=None) -> np.ndarray:
        rng = np.random.default_rng(0) if rng is None else rng
        desired = np.array([
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw), math.cos(yaw), 0],
            [0, 0, 1],
        ])
        best = math.inf
        for attempt in range(12):
            self.scratch.qpos[:] = self.data.qpos
            if attempt:
                self.scratch.qpos[self.qadr[:5]] = rng.uniform(
                    self.limits[:5, 0] * .8, self.limits[:5, 1] * .8
                )
            for _ in range(180):
                mujoco.mj_forward(self.model, self.scratch)
                error_pos = position - self.scratch.site_xpos[self.site]
                current = self.scratch.xmat[self.gripper].reshape(3, 3)
                quaternion = np.empty(4)
                mujoco.mju_mat2Quat(quaternion, (desired @ current.T).ravel())
                if quaternion[0] < 0:
                    quaternion *= -1
                magnitude = np.linalg.norm(quaternion[1:])
                error_rot = quaternion[1:] * (2 * math.atan2(magnitude, quaternion[0]) / max(magnitude, 1e-9))
                best = min(best, float(np.linalg.norm(error_pos)))
                if np.linalg.norm(error_pos) < .001 and np.linalg.norm(error_rot) < .10:
                    return self.scratch.qpos[self.qadr].copy()
                jac_pos, jac_rot = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
                mujoco.mj_jacSite(self.model, self.scratch, jac_pos, jac_rot, self.site)
                jac = np.vstack((jac_pos[:, self.dofs[:5]], .06 * jac_rot[:, self.dofs[:5]]))
                error = np.r_[error_pos, .06 * error_rot]
                delta = jac.T @ np.linalg.solve(jac @ jac.T + np.eye(6) * 1e-5, error)
                delta *= min(1, .15 / max(np.linalg.norm(delta), 1e-9))
                self.scratch.qpos[self.qadr[:5]] = np.clip(
                    self.scratch.qpos[self.qadr[:5]] + delta,
                    self.limits[:5, 0] + .01, self.limits[:5, 1] - .01,
                )
        raise RuntimeError(f"No vertical IK solution for {position.tolist()}, best position error {best:.4f} m")

    def push_ik(self, position: np.ndarray, wrist_roll: float) -> np.ndarray:
        """Reach a push point while keeping wrist roll fixed and tool axis vertical."""
        best = math.inf
        desired_axis = np.array([0., 0., 1.])
        for attempt in range(8):
            self.scratch.qpos[:] = self.data.qpos
            if attempt:
                rng = np.random.default_rng(attempt)
                self.scratch.qpos[self.qadr[:4]] = rng.uniform(
                    self.limits[:4, 0] * .8, self.limits[:4, 1] * .8
                )
            self.scratch.qpos[self.qadr[4]] = wrist_roll
            for _ in range(220):
                mujoco.mj_forward(self.model, self.scratch)
                error_pos = position - self.scratch.site_xpos[self.site]
                current = self.scratch.xmat[self.gripper].reshape(3, 3)
                error_rot = np.cross(current[:, 2], desired_axis)
                best = min(best, float(np.linalg.norm(error_pos)))
                if (
                    np.linalg.norm(error_pos) < .0015
                    and np.linalg.norm(error_rot) < .12
                ):
                    return self.scratch.qpos[self.qadr].copy()
                jac_pos = np.zeros((3, self.model.nv))
                jac_rot = np.zeros((3, self.model.nv))
                mujoco.mj_jacSite(
                    self.model, self.scratch, jac_pos, jac_rot, self.site
                )
                columns = self.dofs[:4]
                jac = np.vstack((
                    jac_pos[:, columns],
                    .04 * jac_rot[:, columns],
                ))
                error = np.r_[error_pos, .04 * error_rot]
                delta = jac.T @ np.linalg.solve(
                    jac @ jac.T + np.eye(6) * 1e-5, error
                )
                delta *= min(1, .12 / max(np.linalg.norm(delta), 1e-9))
                self.scratch.qpos[self.qadr[:4]] = np.clip(
                    self.scratch.qpos[self.qadr[:4]] + delta,
                    self.limits[:4, 0] + .01,
                    self.limits[:4, 1] - .01,
                )
                self.scratch.qpos[self.qadr[4]] = wrist_roll
        raise RuntimeError(
            f"No fixed-wrist push IK solution for {position.tolist()}, "
            f"best position error {best:.4f} m"
        )

    def step(self, target: np.ndarray, phase: str):
        target = np.asarray(target, dtype=np.float64)
        if target.shape != (6,) or not np.isfinite(target).all():
            raise ValueError("Expected six finite simulated joint targets")
        if np.any(target < self.limits[:, 0] - 1e-6) or np.any(target > self.limits[:, 1] + 1e-6):
            raise ValueError("Target exceeds simulated joint limits")
        target = np.clip(target, self.limits[:, 0], self.limits[:, 1])
        if self.episode_root is not None:
            row = {
                "index": len(self.rows), "elapsed": self.data.time - self.frame_start,
                "camera_elapsed": self.data.time - self.frame_start, "camera_age": 0.,
                "state": self.state().tolist(), "action": self.encode(target).tolist(),
                "qpos": self.data.qpos.tolist(),
                "image": f"wrist/{len(self.rows):06}.jpg", "phase": phase,
            }
            if self.active_target is not None and not self.command_entities:
                self.command_entities = [(self.active_target, "object")]
            if self.command_entities and self.renderer is not None:
                command_boxes = [
                    self.target_oriented_box(entity)
                    for entity, _ in self.command_entities
                ]
                command_depth_and_valid = [
                    self.entity_camera_depth(entity)
                    for entity, _ in self.command_entities
                ]
                command_depth = [value for value, _ in command_depth_and_valid]
                command_depth_valid = [
                    float(valid) for _, valid in command_depth_and_valid
                ]
                command_center_valid = [
                    float(self.projected_entity_center(entity)[1])
                    for entity, _ in self.command_entities
                ]
                target_box = self.target_box()
                target_oriented_box = self.target_oriented_box()
                target_depth, _ = self.entity_camera_depth(self.active_target)
                row.update({
                    "target_box": target_box,
                    "target_oriented_box": target_oriented_box,
                    "target_depth_m": target_depth,
                    "command_boxes": [box[:5] for box in command_boxes],
                    "command_visibility": [box[5] for box in command_boxes],
                    "command_depth_m": command_depth,
                    "command_depth_valid": command_depth_valid,
                    "command_center_valid": command_center_valid,
                    "command_roles": [role for _, role in self.command_entities],
                    "command_entity_ids": [
                        entity for entity, _ in self.command_entities
                    ],
                })
            path = self.episode_root / row["image"]
            if not cv2.imwrite(str(path), cv2.cvtColor(self.image(), cv2.COLOR_RGB2BGR)):
                raise RuntimeError(f"Failed to save {path}")
            self.rows.append(row)
        self.data.ctrl[self.actuators] = target
        for substep in range(
            round(1 / self.fps / self.model.opt.timestep)
        ):
            mujoco.mj_step(self.model, self.data)
            contacts = self.obstacle_robot_contacts()
            if contacts:
                self.obstacle_contact_events.append({
                    "frame": len(self.rows),
                    "substep": substep,
                    "phase": phase,
                    "bodies": contacts,
                })
        if not np.isfinite(self.data.qpos).all() or any(w.number for w in self.data.warning):
            raise RuntimeError("Simulation became unstable")

    def move(self, target: np.ndarray, phase: str, *, seconds: float | None = None):
        start = self.data.ctrl[self.actuators].copy()
        duration = max(.3, np.max(np.abs(target - start)) / np.deg2rad(20)) if seconds is None else seconds
        steps = max(1, math.ceil(duration * self.fps))
        for index in range(1, steps + 1):
            self.step(start + (target - start) * index / steps, phase)
        for _ in range(3):
            self.step(target, phase)

    def contacts(self, target: str) -> set[str]:
        body = self.model.body(target).id
        result = set()
        for contact in self.data.contact:
            if contact.dist > .001:
                continue
            bodies = {int(self.model.geom_bodyid[contact.geom1]), int(self.model.geom_bodyid[contact.geom2])}
            if body in bodies:
                if self.gripper in bodies:
                    result.add("fixed")
                if self.moving_jaw in bodies:
                    result.add("moving")
        return result

    def _grasp(
        self,
        target: str,
        *,
        lift_height: float = .07,
        approach_waypoints: tuple[np.ndarray, ...] = (),
    ) -> tuple[np.ndarray, dict]:
        pinch = []
        closed = self.data.ctrl[self.actuators].copy()
        yaw = self.yaws[target] + math.pi / 2
        for attempt, (height_offset, yaw_offset) in enumerate(
            ((0., 0.), (-.0015, .025), (.001, -.025))
        ):
            position = self.position(target)
            attempt_yaw = yaw + yaw_offset
            hover = self.ik(np.r_[position[:2], .065], attempt_yaw)
            hover[-1] = self.limits[-1, 0] + .50 * np.ptp(self.limits[-1])
            if attempt == 0:
                for waypoint_index, waypoint in enumerate(
                    approach_waypoints
                ):
                    waypoint_goal = self.ik(waypoint, attempt_yaw)
                    waypoint_goal[-1] = hover[-1]
                    self.move(
                        waypoint_goal,
                        f"avoid_obstacle_{waypoint_index + 1}",
                    )
            self.move(hover, "approach" if attempt == 0 else "grasp_retry")
            grasp = self.ik(
                np.r_[self.position(target)[:2], TABLE_Z + .012 + height_offset],
                attempt_yaw,
            )
            grasp[-1] = hover[-1]
            self.move(grasp, "descend")
            closed = grasp.copy()
            closed[-1] = self.limits[-1, 0]
            self.move(closed, "close")
            pinch = sorted(self.contacts(target))
            if pinch == ["fixed", "moving"]:
                yaw = attempt_yaw
                break
            reopened = closed.copy()
            reopened[-1] = hover[-1]
            self.move(reopened, "retry_open", seconds=.3)
            self.move(hover, "retry_retreat")
        if pinch != ["fixed", "moving"]:
            return closed, {
                "success": False,
                "reason": "No verified two-jaw contact before lift",
                "pinch": pinch,
            }
        lift = self.ik(np.r_[self.data.site_xpos[self.site, :2], lift_height], yaw)
        lift[-1] = closed[-1]
        self.move(lift, "lift")
        return lift, {"success": True, "pinch": pinch}

    def demonstrate(self, target: str) -> dict:
        self.configure_command("grasp", [(target, "object")])
        lift, grasp = self._grasp(target)
        if not grasp["success"]:
            return grasp
        heights, retained = [], []
        for _ in range(10):
            self.step(lift, "verify")
            heights.append(float(self.position(target)[2] - self.initial["objects"][target][2]))
            retained.append(self.contacts(target) == {"fixed", "moving"})
        success = min(heights) >= .05 and all(retained)
        return {
            "success": success, "reason": "Lift and one-second two-jaw retention check",
            "minimum_lift_m": min(heights), "retained_samples": sum(retained),
            "retention_samples": len(retained), "pinch": grasp["pinch"],
        }

    def demonstrate_obstacle_grasp(
        self,
        target: str,
        obstacle: str = "obstacle_box",
    ) -> dict:
        self.configure_command(
            "grasp", [(target, "object"), (obstacle, "obstacle")]
        )
        self.place_grasp_obstacle(target, obstacle)
        position = self.position(target)
        route_x = position[0] - .050
        route_y = position[1]
        waypoints = (
            np.array([route_x, route_y * .20, .080]),
            np.array([route_x, route_y, .080]),
        )
        lift, grasp = self._grasp(
            target,
            approach_waypoints=waypoints,
        )
        if not grasp["success"]:
            return {
                **grasp,
                "obstacle_contacts": len(self.obstacle_contact_events),
            }
        heights, retained = [], []
        for _ in range(10):
            self.step(lift, "verify")
            heights.append(
                float(
                    self.position(target)[2]
                    - self.initial["objects"][target][2]
                )
            )
            retained.append(
                self.contacts(target) == {"fixed", "moving"}
            )
        collision_free = not self.obstacle_contact_events
        return {
            "success": (
                min(heights) >= .05 and all(retained) and collision_free
            ),
            "reason": (
                "Collision-free detour, lift, and one-second retention check"
            ),
            "minimum_lift_m": min(heights),
            "retained_samples": sum(retained),
            "retention_samples": len(retained),
            "pinch": grasp["pinch"],
            "obstacle_contacts": len(self.obstacle_contact_events),
            "waypoints": [waypoint.tolist() for waypoint in waypoints],
        }

    def demonstrate_view(self, target: str) -> dict:
        self.configure_command("view", [(target, "object")])
        position = self.position(target)
        yaw = self.yaws[target] + math.pi / 2
        first = self.ik(np.r_[position[:2], .070], yaw)
        first[-1] = self.limits[-1, 0] + .55 * np.ptp(self.limits[-1])
        self.move(first, "view_approach")
        second = self.ik(np.r_[position[:2] + [.010, -.010], .065], yaw + .12)
        second[-1] = first[-1]
        self.move(second, "view_parallax")
        for _ in range(10):
            self.step(second, "view_verify")
        box = self.target_oriented_box(target)
        center_error = float(np.linalg.norm(np.asarray(box[:2]) - .5))
        return {
            "success": bool(box[5]) and center_error < .3,
            "reason": "Target remains visible near the wrist-image center",
            "center_error": center_error,
        }

    def demonstrate_place(self, target: str, destination: str) -> dict:
        self.configure_command(
            "place", [(target, "object"), (destination, "destination")]
        )
        lift, grasp = self._grasp(target, lift_height=.070)
        if not grasp["success"]:
            return grasp
        destination_position = self.position(destination)
        yaw = self.yaws[target] + math.pi / 2
        carry = self.ik(np.r_[destination_position[:2], .070], yaw)
        carry[-1] = lift[-1]
        self.move(carry, "place_carry")
        lower = self.ik(np.r_[destination_position[:2], .018], yaw)
        lower[-1] = lift[-1]
        self.move(lower, "place_lower")
        opened = lower.copy()
        opened[-1] = self.limits[-1, 0] + .60 * np.ptp(self.limits[-1])
        self.move(opened, "place_release", seconds=.5)
        retreat = opened.copy()
        retreat[:5] = self.ik(np.r_[destination_position[:2], .08], yaw)[:5]
        self.move(retreat, "place_retreat")
        for _ in range(5):
            self.step(retreat, "place_verify")
        distance = float(np.linalg.norm(
            self.position(target)[:2] - destination_position[:2]
        ))
        released = self.contacts(target) != {"fixed", "moving"}
        return {
            "success": distance <= .04 and released,
            "reason": "Object released over the requested destination",
            "destination_error_m": distance,
            "released": released,
            "pinch": grasp["pinch"],
        }

    def demonstrate_push(self, target: str, destination: str) -> dict:
        self.configure_command(
            "push", [(target, "object"), (destination, "destination")]
        )
        start_position = self.position(target).copy()
        destination_position = self.position(destination)
        closed = self.limits[-1, 0]
        wrist_roll = float(self.data.qpos[self.qadr[4]])
        retreat = None
        for cycle in range(4):
            current_position = self.position(target).copy()
            direction = destination_position[:2] - current_position[:2]
            distance = float(np.linalg.norm(direction))
            if distance <= .055:
                break
            direction /= max(distance, 1e-9)

            behind = current_position[:2] - direction * .025
            approach = self.push_ik(np.r_[behind, .055], wrist_roll)
            approach[-1] = closed
            self.move(
                approach,
                "push_approach" if cycle == 0 else "push_realign",
            )
            contact = self.push_ik(
                np.r_[behind, TABLE_Z + .014], wrist_roll
            )
            contact[-1] = closed
            self.move(contact, "push_lower")
            finish_xy = destination_position[:2]
            finish = self.push_ik(
                np.r_[finish_xy, TABLE_Z + .014], wrist_roll
            )
            finish[-1] = closed
            self.move(finish, "push", seconds=max(.8, distance / .07))
            retreat = self.push_ik(np.r_[finish_xy, .065], wrist_roll)
            retreat[-1] = closed
            self.move(retreat, "push_retreat")
        if retreat is None:
            retreat = self.data.ctrl[self.actuators].copy()
        for _ in range(5):
            self.step(retreat, "push_verify")
        destination_error = float(np.linalg.norm(
            self.position(target)[:2] - destination_position[:2]
        ))
        displacement = float(np.linalg.norm(
            self.position(target)[:2] - start_position[:2]
        ))
        initial_destination_error = float(np.linalg.norm(
            start_position[:2] - destination_position[:2]
        ))
        return {
            "success": destination_error <= .070 and (
                displacement >= .02 or initial_destination_error <= .070
            ),
            "reason": "Object pushed into the requested destination region",
            "destination_error_m": destination_error,
            "displacement_m": displacement,
        }

    def demonstrate_command(
        self, phrase: str, target: str, destination: str | None = None
    ) -> dict:
        if phrase == "grasp":
            return self.demonstrate(target)
        if phrase == "view":
            return self.demonstrate_view(target)
        if destination is None:
            raise ValueError(f"{phrase} requires a destination")
        if phrase == "push":
            return self.demonstrate_push(target, destination)
        if phrase == "place":
            return self.demonstrate_place(target, destination)
        raise ValueError(f"Unsupported command phrase: {phrase}")


class _WristPolicy:
    def __init__(self, checkpoint: Path, device: str, action_steps: int | None = None):
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self.model = SmolVLAPolicy.from_pretrained(checkpoint).to(device).eval()
        if set(self.model.config.image_features) != {"observation.images.wrist"}:
            raise ValueError("Evaluation requires a wrist-only policy")
        if action_steps is not None:
            if not 0 < action_steps <= self.model.config.chunk_size:
                raise ValueError("Action steps must be positive and cannot exceed the trained chunk size")
            self.model.config.n_action_steps = action_steps
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.model.config, pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": device}},
        )

    def reset(self, seed: int):
        import torch

        torch.manual_seed(seed)
        self.model.reset()

    def action(
        self, sim: BeamSimulation, target: str, task: str | None = None
    ) -> np.ndarray:
        import torch

        state = sim.state()
        expected_state_dim = self.model.config.input_features["observation.state"].shape[0]
        target_box = np.asarray(sim.target_box(target), dtype=np.float32)
        if expected_state_dim == len(JOINTS) + 5:
            state = np.concatenate((state, target_box))
        elif expected_state_dim != len(JOINTS):
            raise ValueError(
                f"Unsupported policy state dimension {expected_state_dim}; expected 6 or 11"
            )
        observation = {
            "observation.state": torch.from_numpy(state).unsqueeze(0),
            "observation.images.wrist": torch.from_numpy(sim.image()).permute(2, 0, 1).float().unsqueeze(0) / 255,
            "task": [task or TASK.format(target=target)],
        }
        if "observation.target_box" in self.model.config.input_features:
            observation["observation.target_box"] = torch.from_numpy(target_box).unsqueeze(0)
        with torch.inference_mode():
            action = self.postprocessor(self.model.select_action(self.preprocessor(observation)))
        return action.squeeze(0).cpu().numpy()


def _prepare_recovery_start(sim: BeamSimulation, controller: _WristPolicy, target: str, seed: int,
task: str | None = None,):
    sim.episode_root = None
    controller.reset(seed)
    executed, error = 0, None
    for _ in range(int(np.random.default_rng(seed).integers(20, 86))):
        try:
            goal, _ = sim.bounded_policy_action(controller.action(sim, target, task=task))
        except ValueError as invalid:
            error = str(invalid)
            break
        sim.step(goal, "policy_prefix")
        executed += 1
    sim.yaws = {
        name: math.atan2(sim.data.body(name).xmat.reshape(3, 3)[1, 0],
                         sim.data.body(name).xmat.reshape(3, 3)[0, 0])
        for name in OBJECT_NAMES
    }
    sim.initial = {
        **sim.initial, "arm_state": sim.state().tolist(),
        "objects": {name: sim.position(name).tolist() for name in sim.yaws},
        "yaws": sim.yaws, "recovery_prefix": {"steps": executed, "error": error},
    }
    sim.rows = []
    sim.frame_start = sim.data.time


def collect(root: Path, model_root: Path, episodes: int, seed: int, attempts: int,
            varied_layout: bool = False, recovery_checkpoint: Path | None = None, device: str = "mps",
            real_scene: bool = False, color_targets: bool = False,
            paired_targets: bool = False, visual_cue_target: bool = False,
            target_color: str | None = None, base_phrase: str | None = None):
    if episodes <= 0 or attempts < episodes:
        raise ValueError("Require positive episode count and attempts >= episodes")
    if target_color is not None and (not color_targets or paired_targets):
        raise ValueError("Target-color collection requires color targets without paired targets")
    root.mkdir(parents=True, exist_ok=False)
    controller = _WristPolicy(recovery_checkpoint, device, 16) if recovery_checkpoint is not None else None
    sim = BeamSimulation(model_root)
    successes, outcomes = [], []
    started = time.monotonic()
    next_seed = seed
    try:
        for attempt in range(attempts):
            target = ("studded", "hollow")[
                attempt % 2 if paired_targets else len(successes) % 2
            ]
            while True:
                scene_seed = (
                    next_seed if target_color is not None
                    else seed + (attempt // 2 if paired_targets else attempt)
                )
                if target_color is not None:
                    next_seed += 1
                sim.reset(
                    scene_seed, varied_layout=varied_layout, real_scene=real_scene,
                    color_targets=color_targets, visual_cue_target=visual_cue_target,
                )
                if target_color is None or sim.target_labels[target].startswith(
                    f"{target_color} "
                ):
                    break
            sim.active_target = target
            task = base_phrase or (
                f"Pick up the {sim.target_labels[target]}."
                if color_targets else TASK.format(target=target)
            )
            if controller is not None:
                _prepare_recovery_start(sim, controller, target, scene_seed, task)
            episode = root / f"attempt_{attempt:04}"
            (episode / "wrist").mkdir(parents=True)
            sim.episode_root = episode
            result = sim.demonstrate(target)
            sim.overview(episode / "diagnostic_overview.jpg")
            report = {
                "simulation": True, "model_commit": MODEL_COMMIT,
                "outcome": "success" if result["success"] else "failed",
                "timing_acceptable": len(sim.rows) > 1 and bool(np.allclose(
                    np.diff([row["elapsed"] for row in sim.rows]), 1 / sim.fps, atol=1e-8, rtol=0
                )),
                "clock": "MuJoCo simulated time, not wall time",
                "fps": sim.fps, "frames": len(sim.rows),
                "task": task,
                "target": target, "target_label": sim.target_labels[target],
                "initial": sim.initial, "verification": result,
                "real_scene": real_scene,
                "color_targets": color_targets,
                "paired_targets": paired_targets,
                "visual_cue_target": visual_cue_target,
                "target_color": target_color,
                "base_phrase": base_phrase,
                "recovery_checkpoint": str(recovery_checkpoint) if recovery_checkpoint is not None else None,
                "duration": len(sim.rows) / sim.fps,
            }
            (episode / "frames.jsonl").write_text("".join(json.dumps(row) + "\n" for row in sim.rows))
            (episode / "episode.json").write_text(json.dumps(report, indent=2))
            outcomes.append(report)
            print(json.dumps({"attempt": attempt, "target": target, **result}), flush=True)
            if result["success"]:
                successes.append(str(episode))
            (root / "collection.json").write_text(json.dumps({
                "simulation": True, "requested": episodes, "successful_episodes": successes,
                "attempts": outcomes, "wall_seconds": time.monotonic() - started,
            }, indent=2))
            if len(successes) == episodes:
                return
        raise RuntimeError(f"Only {len(successes)}/{episodes} verified episodes; inspect {root}")
    finally:
        sim.close()


def export_collection(roots: list[Path], destination: Path, repo_id: str):
    from so101.robot_demo import export_episodes

    sources = []
    for root in roots:
        report = json.loads((root / "collection.json").read_text())
        if not report["simulation"]:
            raise ValueError("Expected a simulated collection")
        sources.extend(Path(path) for path in report["successful_episodes"])
    export_episodes(sources, destination, repo_id)


def training_command(dataset: Path, output: Path, steps: int, batch_size: int, device: str,
                     pretrained: str = "lerobot/smolvla_base", learning_rate: float | None = None) -> list[str]:
    if steps <= 0 or batch_size <= 0:
        raise ValueError("Training steps and batch size must be positive")
    if learning_rate is not None and (not math.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("Learning rate must be finite and positive")
    info = json.loads((dataset / "meta/info.json").read_text())
    cameras = [name for name in info["features"] if name.startswith("observation.images.")]
    if info["robot_type"] != "so101_sim" or cameras != ["observation.images.wrist"]:
        raise ValueError("Expected a wrist-only SO-101 simulation dataset")
    if output.exists():
        raise FileExistsError(f"Use a fresh training output directory: {output}")
    command = [
        sys.executable, "-m", "lerobot.scripts.lerobot_train",
        f"--policy.path={pretrained}", "--policy.input_features=null",
        "--policy.chunk_size=16", "--policy.n_action_steps=16",
        f"--dataset.repo_id=local/{dataset.name}", f"--dataset.root={dataset.resolve()}",
        f"--output_dir={output.resolve()}", f"--steps={steps}", f"--batch_size={batch_size}",
        f"--policy.device={device}", "--policy.push_to_hub=false",
        "--wandb.enable=false", "--save_checkpoint_to_hub=false",
        "--num_workers=4", "--persistent_workers=true", "--env_eval_freq=0",
        "--save_freq=1000", "--log_freq=25",
    ]
    if learning_rate is not None:
        command.append(f"--policy.optimizer_lr={learning_rate}")
    return command


def evaluate(checkpoint: Path, root: Path, model_root: Path, episodes: int,
             seed: int, horizon: int, device: str, action_steps: int | None = None,
             varied_layout: bool = False, real_scene: bool = False,
             color_targets: bool = False, paired_targets: bool = False,
             visual_cue_target: bool = False, target_color: str | None = None,
             base_phrase: str | None = None):
    if episodes <= 0 or horizon <= 0 or (action_steps is not None and action_steps <= 0):
        raise ValueError("Evaluation episode count and horizon must be positive")
    if target_color is not None and (not color_targets or paired_targets):
        raise ValueError("Target-color evaluation requires color targets without paired targets")
    root.mkdir(parents=True, exist_ok=False)
    controller = _WristPolicy(checkpoint, device, action_steps)
    sim = BeamSimulation(model_root)
    results = []
    next_seed = seed
    try:
        for episode in range(episodes):
            target = ("studded", "hollow")[episode % 2]
            other = "hollow" if target == "studded" else "studded"
            while True:
                scene_seed = (
                    next_seed if target_color is not None
                    else seed + (episode // 2 if paired_targets else episode)
                )
                if target_color is not None:
                    next_seed += 1
                sim.reset(
                    scene_seed, varied_layout=varied_layout, real_scene=real_scene,
                    color_targets=color_targets, visual_cue_target=visual_cue_target,
                )
                if target_color is None or sim.target_labels[target].startswith(f"{target_color} "):
                    break
            sim.active_target = target
            task = base_phrase or (
                f"Pick up the {sim.target_labels[target]}."
                if color_targets else TASK.format(target=target)
            )
            controller.reset(scene_seed)
            episode_root = root / f"episode_{episode:03}"
            (episode_root / "wrist").mkdir(parents=True)
            sim.episode_root = episode_root
            retained, success, error = 0, False, None
            steps, executed = 0, 0
            projections, max_correction = 0, 0.
            invalid_action = None
            for steps in range(1, horizon + 1):
                raw_action = controller.action(sim, target, task=task)
                try:
                    goal, correction = sim.bounded_policy_action(raw_action)
                except ValueError as invalid:
                    error = str(invalid)
                    invalid_action = raw_action.tolist()
                    break
                sim.step(goal, "policy")
                projections += correction > 0
                max_correction = max(max_correction, correction)
                sim.rows[-1]["raw_action"] = raw_action.tolist()
                sim.rows[-1]["target_contacts"] = sorted(sim.contacts(target))
                sim.rows[-1]["target_elevation"] = float(
                    sim.position(target)[2] - sim.initial["objects"][target][2]
                )
                sim.rows[-1]["other_contacts"] = sorted(sim.contacts(other))
                sim.rows[-1]["other_elevation"] = float(
                    sim.position(other)[2] - sim.initial["objects"][other][2]
                )
                executed += 1
                if sim.rows[-1]["other_elevation"] >= .05 and sim.contacts(other) == {"fixed", "moving"}:
                    error = f"Policy lifted the unrequested {other} beam"
                    break
                lifted = sim.position(target)[2] - sim.initial["objects"][target][2] >= .05
                retained = retained + 1 if lifted and sim.contacts(target) == {"fixed", "moving"} else 0
                if retained >= sim.fps:
                    success = True
                    break
            sim.overview(root / f"overview_{episode:03}.jpg")
            if not cv2.imwrite(str(root / f"wrist_{episode:03}.jpg"), cv2.cvtColor(sim.image(), cv2.COLOR_RGB2BGR)):
                raise RuntimeError("Failed to save policy evaluation image")
            result = {
                "seed": scene_seed, "target": target,
                "target_label": sim.target_labels[target], "task": task, "success": success,
                "steps": executed, "action_attempts": steps,
                "error": error, "initial": sim.initial,
                "boundary_projections": projections, "max_boundary_correction": max_correction,
                "invalid_action": invalid_action,
            }
            (episode_root / "frames.jsonl").write_text("".join(json.dumps(row) + "\n" for row in sim.rows))
            (episode_root / "result.json").write_text(json.dumps(result, indent=2))
            results.append(result)
            print(json.dumps(result), flush=True)
            (root / "evaluation.json").write_text(json.dumps({
                "simulation": True, "checkpoint": str(checkpoint),
                "horizon": horizon, "fps": sim.fps, "first_seed": seed,
                "action_steps": controller.model.config.n_action_steps,
                "varied_layout": varied_layout,
                "real_scene": real_scene,
                "color_targets": color_targets,
                "paired_targets": paired_targets,
                "visual_cue_target": visual_cue_target,
                "target_color": target_color,
                "base_phrase": base_phrase,
                "success_rate": sum(row["success"] for row in results) / len(results),
                "episodes": results,
            }, indent=2))
    finally:
        sim.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("setup")
    collector = commands.add_parser("collect")
    collector.add_argument("--root", type=Path, required=True)
    collector.add_argument("--episodes", type=int, default=10)
    collector.add_argument("--seed", type=int, default=0)
    collector.add_argument("--attempts", type=int, default=30)
    collector.add_argument("--varied-layout", action="store_true")
    collector.add_argument("--real-scene", action="store_true")
    collector.add_argument("--color-targets", action="store_true")
    collector.add_argument(
        "--paired-targets", action="store_true",
        help="Record both target requests from each identical randomized scene.",
    )
    collector.add_argument("--visual-cue-target", action="store_true")
    collector.add_argument("--target-color", choices=tuple(TARGET_COLORS))
    collector.add_argument("--base-phrase", choices=("grasp", "push", "view", "place"))
    collector.add_argument("--recovery-checkpoint", type=Path)
    collector.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    exporter = commands.add_parser("export")
    exporter.add_argument("--collection", type=Path, action="append", required=True)
    exporter.add_argument("--destination", type=Path, required=True)
    exporter.add_argument("--repo-id", default="local/so101-sim-gray")
    trainer = commands.add_parser("train")
    trainer.add_argument("--dataset", type=Path, required=True)
    trainer.add_argument("--output", type=Path, required=True)
    trainer.add_argument("--steps", type=int, default=20_000)
    trainer.add_argument("--batch-size", type=int, default=1)
    trainer.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    trainer.add_argument("--pretrained", default="lerobot/smolvla_base")
    trainer.add_argument("--learning-rate", type=float)
    evaluator = commands.add_parser("evaluate")
    evaluator.add_argument("--checkpoint", type=Path, required=True)
    evaluator.add_argument("--root", type=Path, required=True)
    evaluator.add_argument("--episodes", type=int, default=20)
    evaluator.add_argument("--seed", type=int, default=10_000)
    evaluator.add_argument("--horizon", type=int, default=300)
    evaluator.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    evaluator.add_argument("--action-steps", type=int)
    evaluator.add_argument("--varied-layout", action="store_true")
    evaluator.add_argument("--real-scene", action="store_true")
    evaluator.add_argument("--color-targets", action="store_true")
    evaluator.add_argument(
        "--paired-targets", action="store_true",
        help="Evaluate both target requests from each identical randomized scene.",
    )
    evaluator.add_argument("--visual-cue-target", action="store_true")
    evaluator.add_argument("--target-color", choices=tuple(TARGET_COLORS))
    evaluator.add_argument("--base-phrase", choices=("grasp", "push", "view", "place"))
    args = parser.parse_args()
    if args.command == "setup":
        fetch_model(args.model_root)
    elif args.command == "collect":
        collect(args.root, args.model_root, args.episodes, args.seed, args.attempts,
                args.varied_layout, args.recovery_checkpoint, args.device, args.real_scene,
                args.color_targets, args.paired_targets, args.visual_cue_target,
                args.target_color, args.base_phrase)
    elif args.command == "export":
        export_collection(args.collection, args.destination, args.repo_id)
    elif args.command == "train":
        subprocess.run(training_command(
            args.dataset, args.output, args.steps, args.batch_size, args.device,
            args.pretrained, args.learning_rate
        ), check=True)
    else:
        evaluate(args.checkpoint, args.root, args.model_root, args.episodes,
                 args.seed, args.horizon, args.device, args.action_steps, args.varied_layout,
                 args.real_scene, args.color_targets, args.paired_targets,
                 args.visual_cue_target, args.target_color, args.base_phrase)


if __name__ == "__main__":
    main()
