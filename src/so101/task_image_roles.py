"""Derive source/destination roles from saved same-camera INITIAL segmentation."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import mujoco
import numpy as np

from so101.capture_pile_task_images import target_annotation
from so101.cloud_checkpoint import atomic_json
from so101.photoreal_task_images import integration_state
from so101.pile_task_runtime import create_simulation, load_problem, reset_problem


def annotate_roles(images, output):
    simulations = {}
    completed = 0
    try:
        for path in sorted(images.glob("*.json")):
            receipt_bytes = path.read_bytes()
            receipt = json.loads(receipt_bytes)
            source = Path(receipt["episode_source"])
            root, task_id = source.parent.parent, source.name
            scene = json.loads((root / "problem.json").read_bytes())
            entry = {
                "root": str(root), "scene_id": scene["scene_id"], "scene_family_id": scene["scene_family_id"],
                "files_sha256": receipt["scene_files_sha256"],
            }
            problem = load_problem(entry, task_id)
            if root not in simulations:
                simulations[root] = create_simulation(problem)
            sim = simulations[root]
            reset_problem(sim, problem)
            if (hashlib.sha256(integration_state(sim).tobytes()).hexdigest()
                    != receipt["integration_state_sha256"]):
                raise ValueError("Role annotations differ from rendered INITIAL state")
            segmentation_path = images / receipt.get(
                "segmentation_file", f'_scenes/{scene["scene_id"]}/segmentation.npz',
            )
            if not segmentation_path.resolve().is_relative_to(images.resolve()):
                raise ValueError("Role segmentation escapes its captured population")
            if hashlib.sha256(segmentation_path.read_bytes()).hexdigest() != receipt["segmentation_sha256"]:
                raise ValueError("Saved same-camera segmentation changed")
            with np.load(segmentation_path) as archive:
                segmentation = archive["segmentation"]
            if segmentation.shape != (receipt["height"], receipt["width"], 2):
                raise ValueError("Role segmentation differs from image dimensions")
            task = problem["task"]
            roles = [("source_object", task["target_instance_id"], "lego_object")]
            if task["task_type"] in ("push", "place"):
                entity = task["goal"]["destination_entity"]
                if not np.allclose(sim.data.body(entity).xpos[:2], task["goal"]["destination_xy"],
                                   atol=1e-10, rtol=0):
                    raise ValueError("Destination marker differs from task goal")
                roles.append(("destination", entity, "goal_marker"))
            annotations = []
            for role, entity, kind in roles:
                body = sim.model.body(entity).id
                geoms = np.flatnonzero(sim.model.geom_bodyid == body)
                mask = ((segmentation[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM)
                        & np.isin(segmentation[:, :, 0], geoms))
                annotation = target_annotation(mask)
                if role == "source_object" and any(annotation[key] != receipt[key] for key in annotation):
                    raise ValueError("Source role differs from original target annotation")
                annotations.append({
                    "role": role, "entity_id": entity, "entity_kind": kind,
                    "bbox_semantics": "visible goal-marker geometry, not tolerance-region/amodal"
                    if role == "destination" else "visible source-object geometry",
                    **({"goal": task["goal"]} if role == "destination" else {}),
                    **annotation,
                })
            updated = {
                **receipt, "target_annotations": annotations,
                "role_implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "source_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
            }
            output.mkdir(parents=True, exist_ok=True)
            asset = output / receipt["image_file"]
            if not asset.exists():
                shutil.copyfile(images / receipt["image_file"], asset)
            if hashlib.sha256(asset.read_bytes()).hexdigest() != receipt["sha256"]:
                raise ValueError("Role derivative image changed")
            destination = output / path.name
            if destination.exists() and json.loads(destination.read_bytes()) != updated:
                raise ValueError("Existing role derivative differs from its pinned source")
            atomic_json(destination, updated)
            completed += 1
    finally:
        for sim in simulations.values():
            sim.close()
    return {"annotated_tasks": completed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(annotate_roles(args.images, args.output)))


if __name__ == "__main__":
    main()
