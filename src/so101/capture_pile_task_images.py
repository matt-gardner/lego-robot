"""Capture exact-reset wrist images and visible target-geometry boxes for pile tasks."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from so101.cloud_checkpoint import atomic_json
from so101.rl_pile_adapter import NativePileSimulation, load_native_pool


def target_annotation(mask):
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("Target annotation requires a two-dimensional boolean segmentation mask")
    y, x = np.nonzero(mask)
    return {
        "target_bbox_xyxy": [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1]
        if len(x) else None,
        "target_visible_pixels": len(x),
        "target_bbox_status": "visible" if len(x) else "not_visible",
        "target_bbox_method": "mujoco-target-geom-segmentation",
    }


def capture(args):
    pool_bytes = args.pool.read_bytes()
    pool_sha = hashlib.sha256(pool_bytes).hexdigest()
    implementation = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    scenes = defaultdict(list)
    for rows in load_native_pool(args.pool, set()).values():
        for episode in rows:
            scenes[episode.metadata["native_entry"]["scene_id"]].append(episode)
    args.output.mkdir(parents=True, exist_ok=True)
    completed = 0
    for scene_index, episodes in enumerate(scenes.values()):
        if args.limit_scenes is not None and scene_index >= args.limit_scenes:
            break
        identities = {str(episode.source): hashlib.sha256(str(episode.source).encode()).hexdigest()
                      for episode in episodes}
        complete = True
        for identity in identities.values():
            path = args.output / f"{identity}.json"
            if not path.exists():
                complete = False
                break
            receipt = json.loads(path.read_text())
            image = args.output / receipt["image_file"]
            if (receipt["pool_sha256"] != pool_sha
                    or receipt["capture_implementation_sha256"] != implementation
                    or hashlib.sha256(image.read_bytes()).hexdigest() != receipt["sha256"]):
                raise ValueError("Existing task image differs from the immutable capture contract")
        if complete:
            completed += len(episodes)
            continue
        backend = "mujoco" if args.camera == "external" else "cpu-pathtrace"
        sim = NativePileSimulation(episodes[0], args.model_root, backend=backend, samples=16)
        try:
            state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
            before = np.empty(mujoco.mj_stateSize(sim.model, state_spec))
            after = np.empty_like(before)
            mujoco.mj_getState(sim.model, sim.data, before, state_spec)
            camera_provenance = {}
            if args.camera == "external":
                camera = mujoco.MjvCamera()
                camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                camera.lookat[:] = np.mean([sim.position(name) for name in sim.object_names], axis=0)
                camera.distance, camera.azimuth, camera.elevation = .40, 135., -65.
                camera_provenance = {
                    "type": "mujoco-free-camera", "lookat": camera.lookat.tolist(),
                    "distance": camera.distance, "azimuth": camera.azimuth,
                    "elevation": camera.elevation,
                }
                sim.renderer.update_scene(sim.data, camera=camera)
                rgb = sim.renderer.render().copy()
                sim.renderer.enable_segmentation_rendering()
                try:
                    segmentation = sim.renderer.render().copy()
                finally:
                    sim.renderer.disable_segmentation_rendering()
            else:
                rgb = sim.image()
                segmentation = sim._segmentation_image()
            mujoco.mj_getState(sim.model, sim.data, after, state_spec)
            if not np.array_equal(before, after):
                raise ValueError("Initial RGB/segmentation capture changed the integration state")
            height, width = rgb.shape[:2]
            if segmentation.shape[:2] != (height, width):
                raise ValueError("Target segmentation dimensions differ from the wrist RGB")
            temporary = args.output / "initial-image.png.tmp"
            Image.fromarray(rgb).save(temporary, format="PNG")
            image_sha = hashlib.sha256(temporary.read_bytes()).hexdigest()
            image_path = args.output / f"{image_sha}.png"
            temporary.replace(image_path)
            for episode in episodes:
                target, _, _ = sim.reset_native(episode)
                mujoco.mj_getState(sim.model, sim.data, after, state_spec)
                if not np.array_equal(before, after):
                    raise ValueError("Task-specific reset differs from the shared scene image state")
                mask = ((segmentation[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM)
                        & np.isin(segmentation[:, :, 0], sim.beam_geoms[target]))
                receipt = {
                    "problem_id": identities[str(episode.source)], "episode_source": str(episode.source),
                    "pool_sha256": pool_sha, "capture_implementation_sha256": implementation,
                    "target_instance_id": target, "image_file": image_path.name,
                    "sha256": image_sha, "width": width, "height": height,
                    "camera": args.camera, "camera_provenance": camera_provenance,
                    "backend": backend, "samples": 16 if backend == "cpu-pathtrace" else None,
                    "is_gpu_policy_frame": False, "physics_unchanged": True, "assistance": 0.,
                    **target_annotation(mask),
                }
                atomic_json(args.output / f'{receipt["problem_id"]}.json', receipt)
            completed += len(episodes)
            print(json.dumps({"event": "task_initial_images", "completed_tasks": completed,
                              "scene_id": episode.metadata["native_entry"]["scene_id"]}), flush=True)
        finally:
            sim.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pool", "model-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--limit-scenes", type=int)
    parser.add_argument("--camera", choices=("wrist_cam", "external"), default="wrist_cam")
    capture(parser.parse_args())


if __name__ == "__main__":
    main()
