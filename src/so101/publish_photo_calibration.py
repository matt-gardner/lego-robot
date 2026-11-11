"""Publish new photo-pile progress without substituting historical policy measurements."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import struct

from so101.cloud_checkpoint import atomic_json


def publish(pool_path: Path, references_path: Path, output: Path,
            measurements_path: Path | None = None, images_paths: list[Path] | None = None):
    pool_bytes, reference_bytes = pool_path.read_bytes(), references_path.read_bytes()
    pool, references = json.loads(pool_bytes), json.loads(reference_bytes)
    pool_sha = hashlib.sha256(pool_bytes).hexdigest()
    if (pool.get("kind") != "lego-pile-task-pool" or pool.get("status") != "complete"
            or references.get("kind") != "pile-expert-references"
            or references.get("pool_sha256") != pool_sha
            or references.get("success_condition") != "terminal-native-dwell-v1"):
        raise ValueError("Photo progress is not bound to the supplied immutable task pool")
    problems, by_source, buckets = [], {}, defaultdict(lambda: defaultdict(list))
    for entry in pool["scenes"]:
        root = Path(entry["root"])
        if entry["split"] != "train":
            raise ValueError("Photo dashboard population must be the admitted TRAIN pool")
        for name, expected in entry["files_sha256"].items():
            if Path(name).name != name or hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
                raise ValueError("Photo scene dependency differs from its admission")
        scene = json.loads((root / "problem.json").read_text())
        bucket = f'photo:{scene["tags"]["piece_count"]}-pieces'
        for task in scene["tasks"]:
            source = str(root / "tasks" / task["task_id"])
            if source in by_source:
                raise ValueError("Duplicate photo task source")
            identity = hashlib.sha256(source.encode()).hexdigest()
            problem = {
                "id": identity, "bucket_id": bucket, "task": task["task_type"],
                "source": {"kind": "photo", "path": str(root / "problem.json"),
                           "sha256": entry["files_sha256"]["problem.json"],
                           "scene_id": scene["scene_id"], "task_id": task["task_id"],
                           "split": "train", "episode_source": source},
                "tags": {"scene": scene["tags"], "scene_family_id": scene["scene_family_id"],
                         "target_instance_id": task["target_instance_id"],
                         "initial_target_position": task["initial_target_position"],
                         "goal": task["goal"], "solvability": task["solvability"],
                         "grader_version": task["grader_version"]},
                "measurement_status": "unmeasured",
                "feasibility": {"status": "unknown", "evidence": None},
                "initial_images": [], "initial_images_status": "not-yet-rendered",
            }
            problems.append(problem)
            by_source[source] = problem
            buckets[bucket][task["task_type"]].append(identity)
    if not problems:
        raise ValueError("Photo publication has no admitted problems")
    images, image_digest = [], hashlib.sha256()
    seen_images = set()
    for images_path in images_paths or []:
        for path in sorted(images_path.glob("*.json")):
            receipt_bytes = path.read_bytes()
            receipt = json.loads(receipt_bytes)
            problem = by_source[receipt["episode_source"]]
            image_identity = (problem["id"], receipt["camera"])
            if image_identity in seen_images:
                raise ValueError("Duplicate task/camera image receipt")
            seen_images.add(image_identity)
            name = receipt["image_file"]
            if Path(name).name != name:
                raise ValueError("Task image escapes the captured asset directory")
            image = images_path / name
            body = image.read_bytes()
            if (receipt["pool_sha256"] != pool_sha or receipt["problem_id"] != problem["id"]
                    or receipt["target_instance_id"] != problem["tags"]["target_instance_id"]
                    or receipt["physics_unchanged"] is not True or receipt["assistance"] != 0
                    or hashlib.sha256(body).hexdigest() != receipt["sha256"]
                    or not body.startswith(b"\x89PNG\r\n\x1a\n")
                    or struct.unpack(">II", body[16:24]) != (receipt["width"], receipt["height"])):
                raise ValueError("Task image/annotation differs from its unchanged reset receipt")
            images.append((problem, receipt, image))
            image_digest.update(receipt_bytes)
    attempts = references["attempts"]
    if (len({row["source"] for row in attempts}) != len(attempts)
            or any(row["source"] not in by_source
                   or row["task"] != by_source[row["source"]]["task"] for row in attempts)):
        raise ValueError("Reference attempts duplicate or cross the photo population")
    measurement_bytes = measurements_path.read_bytes() if measurements_path is not None else b""
    measurements = json.loads(measurement_bytes) if measurement_bytes else None
    groups = []
    if measurements is not None:
        if (measurements["input_sha256"]["native_pool"] != pool_sha
                or measurements["optimizer"] is not None
                or not measurements.get("policy_sha256_before")):
            raise ValueError("Policy measurements do not belong to this inference-only photo calibration")
        for row in measurements["groups"]:
            problem = by_source[row["source"]]
            observations = row["rollouts"]
            successes = sum(item["success"] for item in observations)
            if not observations or row["task"] != problem["task"] or row["policy_step"] != 0:
                raise ValueError("Policy group is empty or changes the bound task")
            problem["measurement_status"] = "calibrating"
            groups.append({
                "problem_id": problem["id"], "assistance": row["assistance"],
                "attempts": len(observations), "successes": successes,
                "planned_attempts": measurements["repeats"],
                "success_interval_95": row["success_interval_95"],
                "invalid": sum(bool(item["invalid_action_error"]) for item in observations),
                "policy_failures": len(observations) - successes, "feasibility_failures": None,
                "status": "calibrating", "seeds": [item["seed"] for item in observations],
                "policy_step": row["policy_step"],
                "mixed_group_yield": {"eligible": int(0 < successes < len(observations)), "total": 1},
            })
    progress = {
        "phase": "policy_assistance_calibration" if measurements is not None else "expert_reference_qualification",
        "reference_attempts": len(attempts), "reference_accepted": len(references["references"]),
        "reference_complete": references["complete"],
        "policy_calibration_status": ("complete" if measurements["complete"] else "running")
        if measurements is not None else "pending",
        "policy_rollouts": sum(group["attempts"] for group in groups), "pool_sha256": pool_sha,
    }
    implementation = hashlib.sha256(Path(__file__).read_bytes()).digest()
    snapshot_id = hashlib.sha256(
        implementation + pool_bytes + reference_bytes + measurement_bytes + image_digest.digest(),
    ).hexdigest()
    directory = output / "snapshots" / snapshot_id
    for problem, receipt, image in images:
        asset = directory / "assets" / image.name
        asset.parent.mkdir(parents=True, exist_ok=True)
        if not asset.exists():
            shutil.copyfile(image, asset)
        if hashlib.sha256(asset.read_bytes()).hexdigest() != receipt["sha256"]:
            raise ValueError("Published task image asset changed")
        problem["initial_images"].append({
            **receipt, "path": str(asset.relative_to(directory)), "renderer": receipt["backend"],
            "provenance": receipt,
        })
        problem["initial_images_status"] = "published"
    manifest = {
        "kind": "lego-data-buckets", "schema_version": 1, "snapshot_id": snapshot_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "complete": bool(measurements and measurements["complete"]),
        "measurement_status": "calibrating" if groups else "unmeasured",
        "progress": progress, "problems": problems, "groups": groups,
        "buckets": [{"id": bucket, "label": bucket,
                     "population_total": sum(map(len, tasks.values())), "population_status": "enumerated",
                     "tasks": [{"task": task, "population_total": len(ids), "problem_ids": ids}
                               for task, ids in tasks.items()]}
                    for bucket, tasks in sorted(buckets.items())],
        "protocol": {"population": "new-photo-piles-only", "native_pool_sha256": pool_sha,
                     "reference_collector_sha256": references["collector_sha256"],
                     "policy_sha256_before": measurements.get("policy_sha256_before")
                     if measurements is not None else None,
                     "policy_sha256_after": measurements.get("policy_sha256_after")
                     if measurements is not None else None,
                     "reference_progress_is_policy_success_rate": False},
        "provenance": {"native_pool_sha256": pool_sha,
                       "publisher_implementation_sha256": implementation.hex(),
                       "population_status": "immutable-photo-TRAIN-pool"},
        "notes": ["Reference acceptance is not policy success or proof of general task feasibility.",
                  "Historical screening measurements and images are not reused for these tasks."],
    }
    destination = directory / "manifest.json"
    if not destination.exists():
        atomic_json(destination, manifest)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    relative = str(destination.relative_to(output))
    pointer = {"schema_version": 1, "snapshot_id": snapshot_id, "path": relative,
               "sha256": digest, "manifest_path": relative, "manifest_sha256": digest}
    atomic_json(output / "current.json", pointer)
    return {**pointer, "problems": len(problems), "initial_images": len(images),
            "problems_with_images": len({problem["id"] for problem, _, _ in images}),
            "cameras": dict(Counter(receipt["camera"] for _, receipt, _ in images)),
            "tasks": dict(Counter(problem["task"] for problem in problems)), "progress": progress}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pool", "references", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--measurements", type=Path)
    parser.add_argument("--images", type=Path, action="append")
    args = parser.parse_args()
    print(json.dumps(publish(args.pool, args.references, args.output, args.measurements, args.images)))


if __name__ == "__main__":
    main()
