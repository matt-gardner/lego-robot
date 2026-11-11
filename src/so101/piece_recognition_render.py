"""Opt-in RGB/box and canonical-view export for the pile recognition experiment."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image

from so101.capture_pile_task_images import target_annotation
from so101.lego_scene import restore_scene
from so101.piece_recognition_data import digest, read_bound, write_new_json
from so101.rl_renderer import compile_visual_model


def save_image(root: Path, image: np.ndarray) -> dict:
    content = image.tobytes()
    name = hashlib.sha256(content + str(image.shape).encode()).hexdigest() + ".png"
    path = root / name
    if not path.exists():
        Image.fromarray(image).save(path)
    return {"path": str(path.resolve()), "sha256": digest(path)}


def camera_for(model, data, bodies, azimuth: float, elevation: float):
    indices = np.flatnonzero(np.isin(model.geom_bodyid, bodies))
    if not len(indices):
        raise ValueError("No geometry for the selected piece bodies")
    centers = data.geom_xpos[indices]
    lower = np.min(centers - model.geom_rbound[indices, None], axis=0)
    upper = np.max(centers + model.geom_rbound[indices, None], axis=0)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (lower + upper) / 2
    camera.distance = max(.03, float(np.linalg.norm(upper - lower)) * 1.4)
    camera.azimuth, camera.elevation = azimuth, elevation
    return camera


def canonical_model(part: dict, root: Path):
    for asset in part["assets"]:
        read_bound(root / asset["path"], asset["sha256"])
    tree = ET.fromstring((root / part["mjcf"]).read_bytes())
    # An appearance-only copy, not a change to catalog physics or renderer owners.
    for parent in tree.iter():
        for node in list(parent):
            if node.tag in ("joint", "freejoint", "inertial"):
                parent.remove(node)
            elif node.tag == "geom" and node.get("group") != "2":
                parent.remove(node)
    for geom in tree.findall(".//geom"):
        geom.set("rgba", ".6 .6 .6 1")
    return compile_visual_model(ET.tostring(tree, encoding="unicode"))


def render(plan_path: Path, output: Path, views: int = 8, minimum_pixels: int = 64) -> dict:
    if views < 4 or minimum_pixels < 1:
        raise ValueError("Need at least four views and a positive visible-pixel threshold")
    if output.exists():
        raise FileExistsError(output)
    plan = json.loads(plan_path.read_bytes())
    if plan.get("kind") != "pile-recognition-plan":
        raise ValueError("Expected a recognition preparation plan")
    catalog_path = Path(plan["catalog_subset"])
    catalog = json.loads(read_bound(catalog_path, plan["catalog_subset_sha256"]))
    output.mkdir(parents=True)
    images = output / "images"
    images.mkdir()
    rows, visibility = [], Counter()
    for scene in plan["scenes"]:
        for name, expected in scene["files_sha256"].items():
            read_bound(Path(name), expected)
        xml = Path(scene["model"]).read_text()
        model = (mujoco.MjModel.from_binary_path(scene["binary"]) if scene["binary"]
                 else mujoco.MjModel.from_xml_string(xml))
        data = mujoco.MjData(model)
        snapshot = json.loads(Path(scene["state"]).read_bytes())
        restore_scene(model, data, snapshot, xml)
        initial = data.qpos.copy()
        model.vis.global_.offwidth, model.vis.global_.offheight = 320, 240
        bodies = [model.body(label["body_name"]).id for label in scene["labels"]]
        captures = []
        with mujoco.Renderer(model, height=240, width=320) as renderer:
            for view in range(views):
                camera = camera_for(model, data, bodies, 360 * view / views, -45)
                renderer.update_scene(data, camera=camera)
                rgb = renderer.render().copy()
                renderer.enable_segmentation_rendering()
                try:
                    segmentation = renderer.render().copy()
                finally:
                    renderer.disable_segmentation_rendering()
                targets = []
                for label, body in zip(scene["labels"], bodies, strict=True):
                    geoms = np.flatnonzero(model.geom_bodyid == body)
                    mask = ((segmentation[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM)
                            & np.isin(segmentation[:, :, 0], geoms))
                    annotation = target_annotation(mask)
                    status = ("eligible" if annotation["target_visible_pixels"] >= minimum_pixels
                              else "invisible" if not annotation["target_visible_pixels"] else "too_small")
                    visibility[status] += 1
                    if status == "eligible":
                        targets.append({**label, **annotation})
                captures.append({**save_image(images, rgb), "targets": targets})
                if not np.array_equal(initial, data.qpos):
                    raise ValueError("Recognition rendering changed saved pile poses")
        for index, capture in enumerate(captures):
            if capture["targets"]:
                rows.append({
                    "family": scene["family"], "split": scene["split"],
                    "source": scene["source"], "view": index,
                    "history": [{k: v for k, v in captures[(index - offset) % views].items()
                                 if k != "targets"} for offset in (3, 2, 1, 0)],
                    "targets": capture["targets"],
                })
    canonical = {}
    for part in catalog["parts"]:
        model = canonical_model(part, catalog_path.parent)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        model.vis.global_.offwidth = model.vis.global_.offheight = 224
        canonical[part["part_id"]] = []
        with mujoco.Renderer(model, height=224, width=224) as renderer:
            for elevation in (-60, 0, 60):
                for azimuth in (0, 90, 180, 270):
                    camera = camera_for(model, data, list(range(1, model.nbody)), azimuth, elevation)
                    renderer.update_scene(data, camera=camera)
                    canonical[part["part_id"]].append(save_image(images, renderer.render().copy()))
    if any(not any(row["split"] == split for row in rows)
           for split in ("train", "development", "test")):
        raise ValueError("Visibility filtering left an empty experiment split")
    result = {
        "schema_version": 1, "kind": "pile-recognition-images",
        "plan": str(plan_path.resolve()), "plan_sha256": digest(plan_path),
        "parts": plan["parts"], "canonical": canonical, "rows": rows,
        "visibility": dict(visibility), "views_per_pile": views,
        "minimum_pixels": minimum_pixels, "physics_version": mujoco.__version__,
        "implementation_sha256": {
            name: digest(Path(__file__).with_name(name))
            for name in ("piece_recognition_render.py", "piece_recognition_data.py",
                         "capture_pile_task_images.py", "lego_scene.py", "rl_renderer.py")
        },
        "renderer": "unchanged MuJoCo raster, 320x240, static camera orbit; no dynamics",
        "history_semantics": "four static-pile orbit views; boxes refer only to final view",
    }
    write_new_json(output / "images.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--minimum-pixels", type=int, default=64)
    args = parser.parse_args()
    result = render(args.plan, args.output, args.views, args.minimum_pixels)
    print(json.dumps({"frames": len(result["rows"]), "visibility": result["visibility"]}))


if __name__ == "__main__":
    main()
