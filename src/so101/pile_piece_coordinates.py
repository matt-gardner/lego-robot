"""Visible piece coordinates measured in the exact admitted-pile camera."""
import hashlib
import json
import argparse
import io
import os
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from so101.capture_pile_task_images import integration_state, target_annotation
from so101.cloud_checkpoint import atomic_json
from so101.lego_scene import restore_scene
from so101.photoreal_task_images import mono_camera
from so101.pile_task_runtime import create_simulation, load_problems


def piece_coordinates(sim, renderer, instances):
    renderer.enable_segmentation_rendering()
    try:
        segmentation = renderer.render().copy()
    finally:
        renderer.disable_segmentation_rendering()
    pieces = []
    for instance in instances:
        entity = instance["instance_id"]
        body = sim.model.body(instance["body_name"]).id
        geoms = np.flatnonzero(sim.model.geom_bodyid == body)
        mask = ((segmentation[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM)
                & np.isin(segmentation[:, :, 0], geoms))
        pieces.append({
            "id": entity, "entity_id": entity,
            "position_m": sim.data.xpos[body].tolist(),
            "instance": instance,
            **target_annotation(mask),
        })
    if len({piece["id"] for piece in pieces}) != len(pieces):
        raise ValueError("Duplicate piece identity")
    return {
        "pieces": pieces,
        "piece_camera": mono_camera(renderer.scene),
        "piece_coordinate_method": "pile-camera-visible-geom-segmentation",
        "piece_segmentation_sha256": hashlib.sha256(segmentation.tobytes()).hexdigest(),
        "piece_image_width": segmentation.shape[1],
        "piece_image_height": segmentation.shape[0],
    }


def generate(entry, receipt, image_path):
    problems = load_problems(entry)
    sim = create_simulation(problems[0])
    renderer = None
    try:
        sim.model.vis.quality.offsamples = 0
        sim.model.vis.map.znear = .001 / sim.model.stat.extent
        sim.model.vis.map.zfar = 5. / sim.model.stat.extent
        restore_scene(sim.model, sim.data,
                      json.loads((problems[0]["root"] / "integration_state.json").read_bytes()),
                      (problems[0]["root"] / "model.xml").read_text())
        before = integration_state(sim).copy()
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        centers = np.array([sim.data.xpos[body] for body in sim.registry.values()])
        camera.lookat[:] = np.mean(centers, axis=0)
        camera.distance = max(.40, float(np.max(np.linalg.norm(centers - camera.lookat, axis=1))) * 3)
        camera.azimuth, camera.elevation = 135., -65.
        renderer = mujoco.Renderer(sim.model, height=240, width=320)
        renderer.update_scene(sim.data, camera=camera)
        body = image_path.read_bytes()
        if hashlib.sha256(body).hexdigest() != receipt["pile"]["sha256"]:
            raise ValueError("Retained pile image hash mismatch")
        original = np.asarray(Image.open(io.BytesIO(body)).convert("RGB"))
        if not np.array_equal(original, renderer.render()):
            raise ValueError("Reconstructed pile camera pixels differ from retained image")
        if mono_camera(renderer.scene) != receipt["pile"]["camera_provenance"]:
            raise ValueError("Reconstructed pile camera differs from retained camera")
        result = piece_coordinates(sim, renderer, problems[0]["scene"]["instances"])
        if not np.array_equal(before, integration_state(sim)):
            raise ValueError("Piece coordinate capture changed physics state")
        return {**result, "pile_image_sha256": receipt["pile"]["sha256"],
                "scene_files_sha256": entry["files_sha256"]}
    finally:
        if renderer is not None:
            renderer.close()
        sim.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("MUJOCO_GL") != "osmesa" or mujoco.__version__ != "3.17.0":
        raise ValueError("Original CPU OSMesa MuJoCo 3.17 runtime required")
    if hashlib.sha256(args.snapshot.read_bytes()).hexdigest() != args.sha256:
        raise ValueError("Input snapshot hash mismatch")
    os.nice(10)
    args.output.mkdir(parents=True, exist_ok=True)
    for scene in json.loads(args.snapshot.read_bytes())["scenes"]:
        disk = os.statvfs(args.output)
        limit = int(Path("/sys/fs/cgroup/memory.max").read_text())
        used = int(Path("/sys/fs/cgroup/memory.current").read_text())
        if disk.f_bavail * disk.f_frsize < 11 * 1024**3 or limit - used < 2 * 1024**3:
            raise RuntimeError("Insufficient bounded render headroom")
        admission = Path(scene["admission"]).read_bytes()
        if hashlib.sha256(admission).hexdigest() != scene["admission_sha256"]:
            raise ValueError("Admission hash mismatch")
        entry = json.loads(admission)
        directory = args.images / scene["scene_id"]
        receipt = json.loads((directory / "scene.json").read_bytes())
        output = args.output / f'{scene["scene_id"]}.json'
        if output.exists():
            saved = json.loads(output.read_bytes())
            if (saved["pile_image_sha256"] != receipt["pile"]["sha256"]
                    or saved["scene_files_sha256"] != entry["files_sha256"]):
                raise ValueError("Existing piece coordinate bindings differ")
            continue
        result = generate(entry, receipt, directory / receipt["pile"]["image_file"])
        atomic_json(output, {"scene_id": scene["scene_id"], **result}, compact=True)
        print(json.dumps({"scene": scene["scene_id"], "pieces": len(result["pieces"])}), flush=True)


if __name__ == "__main__":
    main()
