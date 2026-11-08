"""Reproducible render-only composition fixtures, not manipulation demonstrations."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import mujoco
import numpy as np
from OpenGL import GL

from so101.renderer_appearance import RendererAppearance
from so101.renderer_appearance_v2 import WristAppearanceV2
from so101.renderer_appearance_v3 import WristAppearanceV3
from so101.renderer_appearance_v4 import WristAppearanceV4
from so101.renderer_appearance_v5 import WristAppearanceV5
from so101.robot_sim import BEAM_HALF_HEIGHT, MODEL_COMMIT, OBJECT_NAMES, TABLE_Z, BeamSimulation


def table_point(sim: BeamSimulation, center: tuple[float, float], height: float) -> np.ndarray:
    camera = sim.model.camera("wrist_cam").id
    rotation = sim.data.cam_xmat[camera].reshape(3, 3)
    position = sim.data.cam_xpos[camera]
    half_height = math.tan(math.radians(float(sim.model.cam_fovy[camera])) / 2)
    ray = rotation @ np.array([(center[0] - .5) * 2 * half_height * 4 / 3,
                               (.5 - center[1]) * 2 * half_height, -1])
    distance = (height - position[2]) / ray[2]
    if distance <= 0:
        raise ValueError("Composition ray does not intersect tabletop in front of camera")
    return position + distance * ray


def place_in_image(sim: BeamSimulation, name: str, center: tuple[float, float], color) -> None:
    xyz = table_point(sim, center, TABLE_Z + BEAM_HALF_HEIGHT)
    camera = sim.model.camera("wrist_cam").id
    rotation = sim.data.cam_xmat[camera].reshape(3, 3)
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    address = sim.model.joint(f"{name}_free").qposadr[0]
    sim.data.qpos[address:address + 7] = [*xyz, math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    sim.model.geom_rgba[sim.beam_geoms[name], :3] = color
    mujoco.mj_forward(sim.model, sim.data)


def qualify(references: Path, output: Path, *, appearance: RendererAppearance | None = None) -> None:
    output.mkdir(parents=True, exist_ok=False)
    profile = appearance if appearance is not None else RendererAppearance()
    comparisons = []
    frames = []
    sim = BeamSimulation(appearance=profile)
    try:
        backend = GL.glGetString(GL.GL_RENDERER).decode()
        if "llvmpipe" not in backend.lower() and "softpipe" not in backend.lower():
            raise RuntimeError(f"Qualification requires software CPU renderer, got {backend}")
        for index, filename in enumerate(("wrist_current.jpg", "autopilot_scene_wrist.jpg",
                                          "target_teal_current_wrist.jpg")):
            raw = (references / filename).read_bytes()
            reference_hash = hashlib.sha256(raw).hexdigest()
            if reference_hash != profile.config["reference_sha256"][filename]:
                raise ValueError(f"Reference hash mismatch: {filename}")
            reference = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if reference is None:
                raise ValueError(f"Invalid JPEG: {filename}")
            sim.reset(10)
            grasp = sim.data.site_xpos[sim.site].copy()
            grasp[2] = .010
            arm = sim.ik(grasp, 0, rng=np.random.default_rng(10))
            arm[-1] = sim.data.qpos[sim.qadr[-1]]
            if isinstance(profile, WristAppearanceV2):
                arm[-1] = sim.limits[-1, 0] + .3 * np.ptp(sim.limits[-1])
            sim.data.qpos[sim.qadr] = arm
            sim.data.ctrl[sim.actuators] = arm
            for object_index, name in enumerate(OBJECT_NAMES):
                address = sim.model.joint(f"{name}_free").qposadr[0]
                sim.data.qpos[address:address + 3] = [.8, .5 + object_index * .2,
                                                     TABLE_Z + BEAM_HALF_HEIGHT]
            for name in sim.destination_geoms:
                sim.model.body_pos[sim.model.body(name).id] = [0, 0, -1]
            mujoco.mj_forward(sim.model, sim.data)
            if index == 1:
                place_in_image(sim, "short", (.99, .38), (.93, .34, .04))
            elif index == 2:
                place_in_image(sim, "studded", (.50, .76), (.02, .55, .58))
                place_in_image(sim, "short", (.30, -.03), (.93, .34, .04))
            image = sim.image()
            frame_path = output / f"{Path(filename).stem}_simulation.png"
            if not cv2.imwrite(str(frame_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
                raise OSError(f"Failed to write {frame_path}")
            side_by_side = np.concatenate((cv2.resize(reference, (320, 240)),
                                          cv2.cvtColor(image, cv2.COLOR_RGB2BGR)), axis=1)
            comparisons.append(side_by_side)
            frames.append({"reference": filename, "reference_sha256": reference_hash,
                           "render": frame_path.name, "qpos": sim.data.qpos.tolist(),
                           "entities": {name: {"visible_pixels": int(sim.entity_mask(name).sum()),
                                               "center_depth_m": sim.entity_camera_depth(name)[0]}
                                        for name in OBJECT_NAMES}})
        comparison_path = output / "reference_vs_simulation.png"
        if not cv2.imwrite(str(comparison_path), np.concatenate(comparisons)):
            raise OSError(f"Failed to write {comparison_path}")
        report = {"renderer": profile.identity(), "model_commit": MODEL_COMMIT,
                  "mujoco_version": mujoco.__version__, "gl_renderer": backend,
                  "qualification": "render-only hand-composed legacy beam proxies; not settled "
                  "catalog scenes, calibrated cameras, accepted training data or transfer evidence",
                  "frames": frames}
        (output / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        sim.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--appearance", choices=("v1", "v2", "v3", "v4", "v5"), default="v1")
    args = parser.parse_args()
    appearance = (WristAppearanceV5() if args.appearance == "v5" else
                  WristAppearanceV4() if args.appearance == "v4" else
                  WristAppearanceV3() if args.appearance == "v3" else
                  WristAppearanceV2() if args.appearance == "v2" else None)
    qualify(args.references, args.output, appearance=appearance)


if __name__ == "__main__":
    main()
