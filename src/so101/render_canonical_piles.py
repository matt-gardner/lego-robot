"""CPU path-traced images and same-camera piece labels for canonical saved piles."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from types import SimpleNamespace

import mujoco
import numpy as np
from PIL import Image

from so101.capture_pile_task_images import integration_state
from so101.cloud_checkpoint import atomic_json
from so101.lego_scene import restore_scene, runtime_model_sha256
from so101.pathtrace_pile import DEFAULT_PROFILE, file_hash, render_pathtrace
from so101.photoreal_task_images import admit_render, export_scene, mono_camera
from so101.pile_directory import read_json
from so101.pile_piece_coordinates import piece_coordinates


SOURCE_FILES = ("model.mjb", "model.xml", "integration_state.json",
                "problem.json", "scene_plan.json")


@contextmanager
def render_workspace(scratch, name):
    work = Path(tempfile.mkdtemp(prefix=name + "-", dir=scratch))
    try:
        yield work
    except Exception:
        print(f"Failed render diagnostics retained: {work}", flush=True)
        raise
    else:
        shutil.rmtree(work)


def source_hashes(directory):
    return {name: file_hash(directory / name) for name in SOURCE_FILES}


def render_one(directory, profile_path, scratch, expected):
    if source_hashes(directory) != expected:
        raise ValueError(f"Frozen source changed: {directory}")
    pile = read_json(directory / "pile.json")
    profile = read_json(profile_path)
    contract = {
        "scene_files_sha256": expected,
        "profile_sha256": file_hash(profile_path),
        "implementation_sha256": file_hash(Path(__file__)),
        "export_implementation_sha256": file_hash(
            Path(__file__).with_name("photoreal_task_images.py")),
        "pathtrace_implementation_sha256": file_hash(
            Path(__file__).with_name("pathtrace_pile.py")),
        "mujoco_version": mujoco.__version__,
    }
    receipt_path = directory / "render_receipt.json"
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if receipt["contract"] != contract:
            raise ValueError(f"Existing photoreal render contract differs: {directory}")
        if (file_hash(directory / "rendered.png") != receipt["image_sha256"]
                or pile["rendered"]["sha256"] != receipt["image_sha256"]
                or pile["piece_coordinates"]["pile_image_sha256"] != receipt["image_sha256"]):
            raise ValueError(f"Published image/metadata differs: {directory}")
        return "reused"
    admit_render(scratch)
    started = time.monotonic()
    model = mujoco.MjModel.from_binary_path(str(directory / "model.mjb"))
    data = mujoco.MjData(model)
    snapshot = read_json(directory / "integration_state.json")
    restore_scene(model, data, snapshot, (directory / "model.xml").read_text())
    sim = SimpleNamespace(model=model, data=data)
    before = integration_state(sim).copy()
    physical = runtime_model_sha256(model)
    instances = read_json(directory / "problem.json")["instances"]
    model.vis.quality.offsamples = 0
    model.vis.map.znear = .001 / model.stat.extent
    model.vis.map.zfar = 5. / model.stat.extent
    width, height = profile["width"], profile["height"]
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    centers = np.array([data.xpos[model.body(item["body_name"]).id] for item in instances])
    camera.lookat[:] = np.mean(centers, axis=0)
    camera.distance = max(.40, float(np.max(np.linalg.norm(centers - camera.lookat, axis=1))) * 3)
    camera.azimuth, camera.elevation = 135., -65.
    with render_workspace(scratch, directory.name) as work:
        with mujoco.Renderer(model, height=height, width=width) as renderer:
            renderer.update_scene(data, camera=camera)
            geometry_camera = mono_camera(renderer.scene)
            coordinates = piece_coordinates(sim, renderer, instances)
            renderer.enable_segmentation_rendering()
            try:
                segmentation = renderer.render().copy()
            finally:
                renderer.disable_segmentation_rendering()
        np.savez_compressed(work / "segmentation.npz", segmentation=segmentation)
        exported = export_scene(sim, directory, work / "export", geometry_camera)
        description = read_json(exported)
        description["source_state_file"] = "integration_state.json"
        table = model.geom("table").id
        description["table_center_m"] = data.geom_xpos[table].tolist()
        description["table_half_size_m"] = model.geom_size[table, :2].tolist()
        rotation = data.geom_xmat[table].reshape(3, 3)
        description["table_yaw_deg"] = float(np.rad2deg(np.arctan2(rotation[1, 0], rotation[0, 0])))
        atomic_json(exported, description)
        rendered = render_pathtrace(exported, work / "pathtrace", profile_path)
        if (not np.array_equal(before, integration_state(sim))
                or physical != runtime_model_sha256(model)):
            raise ValueError("Appearance rendering changed saved physics")
        if source_hashes(directory) != expected:
            raise ValueError("Frozen sources changed while rendering")
        image = work / "pathtrace" / "pile.png"
        with Image.open(image) as decoded:
            decoded.verify()
        if image.stat().st_size > 8 * 1024**2:
            raise ValueError("Image exceeds canonical consumer bound")
        image_sha = file_hash(image)
        coordinates["pile_image_sha256"] = image_sha
        coordinates["scene_files_sha256"] = expected
        coordinates["piece_coordinate_provenance"] = {
            "method": coordinates["piece_coordinate_method"],
            "segmentation_sha256": coordinates["piece_segmentation_sha256"],
            "alignment": rendered["segmentation_alignment"],
        }
        receipt = {
            "contract": contract, "image_sha256": image_sha,
            "previous_image": pile.get("rendered"),
            "camera": geometry_camera, "physics_unchanged": True,
            "piece_count": len(instances), "elapsed_seconds": time.monotonic() - started,
            "appearance": rendered,
        }
        # Merge into the latest metadata; task writers do not own these fields.
        current = read_json(directory / "pile.json")
        if current != pile:
            raise ValueError("Pile metadata changed during rendering")
        current["rendered"] = {
            "sha256": image_sha, "camera": "external",
            "renderer": rendered["backend"], "width": width, "height": height,
            "image_kind": "admitted-pile-camera", "receipt": "render_receipt.json",
            "profile_sha256": contract["profile_sha256"],
        }
        current["piece_coordinates"] = coordinates
        temporary_image = directory / ".rendered.png.tmp"
        shutil.copyfile(image, temporary_image)
        temporary_image.replace(directory / "rendered.png")
        atomic_json(directory / "pile.json", current, compact=True)
        atomic_json(receipt_path, receipt, compact=True)
    return "rendered"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if os.environ.get("MUJOCO_GL") != "osmesa" or mujoco.__version__ != "3.17.0":
        raise ValueError("CPU OSMesa MuJoCo 3.17 runtime required")
    os.nice(10)
    args.scratch.mkdir(parents=True, exist_ok=True)
    directories = sorted(args.root.glob("piles/pile_[0-9][0-9][0-9][0-9]"))
    selection = args.scratch / "selection.json"
    if not selection.exists():
        atomic_json(selection, {path.name: source_hashes(path) for path in directories})
    frozen = read_json(selection)
    if set(frozen) != {path.name for path in directories}:
        raise ValueError("Pile population differs from frozen selection")
    directories.sort(key=lambda path: ((path / "rendered.png").exists(), path.name))
    for index, directory in enumerate(directories):
        if args.limit is not None and index >= args.limit:
            break
        result = render_one(directory, args.profile, args.scratch, frozen[directory.name])
        print(json.dumps({"pile": directory.name, "status": result}), flush=True)


if __name__ == "__main__":
    main()
