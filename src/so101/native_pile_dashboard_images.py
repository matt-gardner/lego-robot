"""Serial software-camera renders of hash-bound admitted piles and task starts."""
import argparse
from collections import Counter
import hashlib
import inspect
import json
import os
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from so101.capture_pile_task_images import integration_state
from so101.cloud_checkpoint import atomic_json
from so101.lego_scene import restore_scene, runtime_model_sha256
from so101.photoreal_task_images import mono_camera
from so101.pile_task_runtime import create_simulation, load_problems, reset_problem
from so101.task_image_roles import role_annotations


CHANNEL = "native-pile-camera-images-v1"
MAX_MANIFEST_BYTES = 32 * 1024 * 1024
LEGACY_IMPLEMENTATION_SHA = "83f7316e60b0c51afc291613c3f263fb5a0922491f20089b753d74b6eb3d65f9"
LEGACY_RENDER_SOURCE_SHA = "a22658107064c5966935eb1beacead8a7a6ea1cfb8329826a69295eb3a95abb2"


def digest(body):
    return hashlib.sha256(body).hexdigest()


def render_source_sha():
    return digest("\n".join(inspect.getsource(function) for function in (
        boxed_image, save_image, render_scene)).encode())


def publish_manifest(manifest, directory, root, output):
    body = json.dumps(manifest, allow_nan=False, separators=(",", ":")).encode()
    if len(body) > MAX_MANIFEST_BYTES:
        raise ValueError(f"Camera manifest exceeds consumer read limit: "
                         f"{len(body)} > {MAX_MANIFEST_BYTES} bytes; current publication unchanged")
    sha = digest(body)
    path = directory / f"{sha}.json"
    if path.exists():
        if path.read_bytes() != body:
            raise ValueError("Immutable camera manifest changed")
    else:
        atomic_json(path, manifest, compact=True)
    atomic_json(output, {"path": str(path.relative_to(root)), "sha256": sha})
    return path


def image_index(row, relative):
    keys = ("sha256", "camera", "renderer", "width", "height", "image_kind", "is_policy_camera",
            "target_bbox_xyxy", "target_visible_pixels", "target_bbox_status")
    result = {key: row[key] for key in keys if key in row}
    result["path"] = relative + "/" + row["image_file"]
    result["receipt_path"] = relative + "/scene.json"
    if "target_annotations" in row:
        result["target_annotations"] = [
            {key: annotation[key] for key in (
                "role", "entity_id", "target_bbox_xyxy", "target_visible_pixels", "target_bbox_status")}
            for annotation in row["target_annotations"]
        ]
    return result


def boxed_image(rgb, annotations):
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    for row in annotations:
        box = row["target_bbox_xyxy"]
        if box is not None:
            color = (255, 221, 0) if row["role"] == "source_object" else (40, 220, 255)
            x0, y0, x1, y1 = box
            draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=color, width=2)
    return image


def save_image(image, directory):
    temporary = directory / "image.tmp.png"
    image.save(temporary)
    sha = digest(temporary.read_bytes())
    destination = directory / f"{sha}.png"
    temporary.replace(destination)
    return {"image_file": destination.name, "path": destination.name, "sha256": sha}


def render_scene(entry, directory):
    problems = load_problems(entry)
    if not problems or entry["split"] != "train":
        raise ValueError("Rendering requires a nonempty admitted TRAIN scene")
    directory.mkdir(parents=True, exist_ok=True)
    implementation = render_source_sha()
    contract = {"admission_sha256": digest(json.dumps(entry, sort_keys=True).encode()),
                "implementation_sha256": implementation, "mujoco_version": mujoco.__version__}
    completed = directory / "scene.json"
    if completed.exists():
        receipt = json.loads(completed.read_bytes())
        if receipt["contract"] != contract:
            # Accept the original file fingerprint only for this exact rendering implementation.
            legacy = {**contract, "implementation_sha256": LEGACY_IMPLEMENTATION_SHA}
            if implementation != LEGACY_RENDER_SOURCE_SHA or receipt["contract"] != legacy:
                raise ValueError("Existing scene render contract changed")
        for image in [receipt["pile"], *receipt["tasks"]]:
            if digest((directory / image["image_file"]).read_bytes()) != image["sha256"]:
                raise ValueError("Existing scene image changed")
        return receipt
    sim = create_simulation(problems[0])
    renderer = None
    try:
        # Disable multisampling for RGB and labels so both use identical pixel coverage.
        sim.model.vis.quality.offsamples = 0
        # Compiled scene statistics can include distant drop geometry: use metric clipping.
        sim.model.vis.map.znear = .001 / sim.model.stat.extent
        sim.model.vis.map.zfar = 5. / sim.model.stat.extent
        renderer = mujoco.Renderer(sim.model, height=240, width=320)
        restore_scene(sim.model, sim.data,
                      json.loads((problems[0]["root"] / "integration_state.json").read_bytes()),
                      (problems[0]["root"] / "model.xml").read_text())
        before = integration_state(sim)
        physical = runtime_model_sha256(sim.model)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        centers = np.array([sim.data.xpos[body] for body in sim.registry.values()])
        camera.lookat[:] = np.mean(centers, axis=0)
        camera.distance = max(.40, float(np.max(np.linalg.norm(
            centers - camera.lookat, axis=1))) * 3)
        camera.azimuth, camera.elevation = 135., -65.
        renderer.update_scene(sim.data, camera=camera)
        pile = {**save_image(Image.fromarray(renderer.render().copy()), directory),
                "camera": "external", "camera_provenance": mono_camera(renderer.scene),
                "image_kind": "admitted-pile-camera",
                "integration_state_sha256": digest(before.tobytes())}
        if not np.array_equal(before, integration_state(sim)):
            raise ValueError("Pile capture changed the admitted integration state")
        task_camera = mujoco.MjvCamera()
        task_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        task_camera.lookat[:] = [.22, 0, sim.data.geom("table").xpos[2]]
        task_camera.distance, task_camera.azimuth, task_camera.elevation = .45, 135., -65.
        tasks, cached_state, rgb, segmentation, camera_provenance = [], None, None, None, None
        for problem in problems:
            reset_problem(sim, problem)
            before = integration_state(sim)
            state_sha = digest(before.tobytes())
            if state_sha != cached_state:
                if physical != runtime_model_sha256(sim.model):
                    raise ValueError("Task reset changed the admitted physical model")
                renderer.update_scene(sim.data, camera=task_camera)
                camera_provenance = mono_camera(renderer.scene)
                rgb = renderer.render().copy()
                renderer.enable_segmentation_rendering()
                try:
                    segmentation = renderer.render().copy()
                finally:
                    renderer.disable_segmentation_rendering()
                if physical != runtime_model_sha256(sim.model):
                    raise ValueError("Camera capture changed the physical model")
                cached_state = state_sha
            annotations = role_annotations(sim.model, sim.data, problem["task"], segmentation)
            task = problem["task"]
            source = str(problem["root"] / "tasks" / task["task_id"])
            tasks.append({
                **save_image(boxed_image(rgb, annotations), directory),
                "problem_id": digest(source.encode()), "episode_source": source,
                "task_id": task["task_id"], "task_type": task["task_type"],
                "target_instance_id": task["target_instance_id"],
                "camera": "task_scene", "camera_provenance": camera_provenance,
                "is_policy_camera": False,
                "integration_state_sha256": state_sha, "target_annotations": annotations,
                **{key: annotations[0][key] for key in (
                    "target_bbox_xyxy", "target_visible_pixels", "target_bbox_status")},
                "image_kind": "boxed-task-camera",
            })
            if not np.array_equal(before, integration_state(sim)):
                raise ValueError("Task capture changed integration state")
        if physical != runtime_model_sha256(sim.model):
            raise ValueError("Task resets changed the admitted physical model")
        common = {"backend": "mujoco-osmesa-cpu", "renderer": "mujoco-osmesa-cpu",
                  "width": 320, "height": 240, "physics_unchanged": True,
                  "is_gpu_policy_frame": False, "assistance": 0,
                  "scene_files_sha256": entry["files_sha256"]}
        receipt = {"contract": contract, "pile": {**common, **pile},
                   "tasks": [{**common, **task} for task in tasks]}
        atomic_json(completed, receipt)
        return receipt
    finally:
        if renderer is not None:
            renderer.close()
        sim.close()


def publish(snapshot, snapshot_sha, root, output, limit_scenes=None):
    if digest(snapshot.read_bytes()) != snapshot_sha:
        raise ValueError("Selected immutable snapshot hash differs")
    population = json.loads(snapshot.read_bytes())
    directory = root / CHANNEL / snapshot_sha
    directory.mkdir(parents=True, exist_ok=True)
    frozen = directory / "input-snapshot.json"
    if frozen.exists() and frozen.read_bytes() != snapshot.read_bytes():
        raise ValueError("Frozen input snapshot changed")
    if not frozen.exists():
        frozen.write_bytes(snapshot.read_bytes())
    problems, piles, coverage = [], [], Counter()
    seen = set()
    for index, scene in enumerate(population["scenes"]):
        if scene["scene_id"] in seen:
            raise ValueError("Duplicate selected pile")
        seen.add(scene["scene_id"])
        admission = Path(scene["admission"])
        body = admission.read_bytes()
        if digest(body) != scene["admission_sha256"]:
            raise ValueError("Selected admission hash differs")
        entry = json.loads(body)
        if entry["scene_id"] != scene["scene_id"]:
            raise ValueError("Selected admission identity differs")
        selected = load_problems(entry)
        scene_tags = {key: selected[0]["scene"]["tags"][key] for key in (
            "piece_count", "obstacles") if key in selected[0]["scene"]["tags"]}
        scene_directory = directory / scene["scene_id"]
        receipt = None
        if limit_scenes is None or index < limit_scenes:
            receipt = render_scene(entry, scene_directory)
        elif (scene_directory / "scene.json").exists():
            receipt = render_scene(entry, scene_directory)
        rendered = {row["task_id"]: row for row in receipt["tasks"]} if receipt else {}
        relative = str(scene_directory.relative_to(directory))
        if receipt:
            piles.append({
                "id": "admission:" + scene["admission_sha256"], "scene_id": scene["scene_id"],
                "admission_sha256": scene["admission_sha256"], "tags": scene_tags,
                "status": "admitted TRAIN pile camera render; not expert/SFT qualified",
                "image_kind": "admitted-pile-camera", "tasks": [],
                "initial_images": [image_index(receipt["pile"], relative)],
            })
        for problem in selected:
            task = problem["task"]
            row = rendered.get(task["task_id"])
            source = str(problem["root"] / "tasks" / task["task_id"])
            coverage["selected_tasks"] += 1
            if row:
                coverage["rendered_tasks"] += 1
                coverage["target_" + row["target_bbox_status"]] += 1
                for annotation in row["target_annotations"][1:]:
                    coverage["destination_" + annotation["target_bbox_status"]] += 1
            problems.append({
                "id": digest(source.encode()), "task": task["task_type"],
                "bucket_id": f'current:{scene["piece_count"]}-pieces',
                "source": {"episode_source": source, "scene_id": scene["scene_id"],
                           "task_id": task["task_id"], "sha256": entry["files_sha256"]["problem.json"],
                           "split": "train", "kind": "admitted"},
                "tags": {"scene": scene_tags,
                         "target_instance_id": task["target_instance_id"], "goal": task["goal"]},
                "feasibility": {"status": "unmeasured"}, "measurement_status": "unmeasured",
                "initial_images": [image_index(row, relative)] if row else [],
            })
    counts = {"selected_piles": len(seen), "rendered_piles": len(piles),
              "missing_piles": len(seen) - len(piles), "selected_tasks": coverage["selected_tasks"],
              "rendered_tasks": coverage["rendered_tasks"],
              "missing_tasks": coverage["selected_tasks"] - coverage["rendered_tasks"],
              "errors": 0, **{key: coverage[key] for key in (
                  "target_visible", "target_not_visible", "destination_visible", "destination_not_visible")}}
    manifest = {
        "kind": "lego-data-buckets", "schema_version": 1,
        "snapshot_id": digest(json.dumps([snapshot_sha, problems, piles], sort_keys=True).encode()),
        "measurement_status": "unmeasured", "complete": False,
        "qualification": "TRAIN starts, not expert references or SFT qualification",
        "provenance": {"published_pile_snapshot_sha256": snapshot_sha,
                       "input_snapshot": str(snapshot), "implementation_sha256": digest(Path(__file__).read_bytes())},
        "image_coverage": counts, "problems": problems, "admitted_piles": piles, "groups": [], "buckets": [],
    }
    manifest_path = publish_manifest(manifest, directory, root, output)
    print(json.dumps({**counts, "manifest": str(manifest_path), "input_sha256": snapshot_sha}), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--limit-scenes", type=int)
    args = parser.parse_args()
    if os.environ.get("MUJOCO_GL") != "osmesa":
        raise ValueError("This bounded renderer requires MUJOCO_GL=osmesa; no GPU backend authorized")
    os.nice(10)
    output = args.root / "publications" / CHANNEL / "current.json"
    publish(args.snapshot, args.sha256, args.root, output, args.limit_scenes)


if __name__ == "__main__":
    main()
