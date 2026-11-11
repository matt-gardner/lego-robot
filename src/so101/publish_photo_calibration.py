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


def copy_image(image: Path, directory: Path, expected: str) -> str:
    asset = directory / "assets" / image.name
    asset.parent.mkdir(parents=True, exist_ok=True)
    if not asset.exists():
        shutil.copyfile(image, asset)
    if hashlib.sha256(asset.read_bytes()).hexdigest() != expected:
        raise ValueError("Published task image asset changed")
    return str(asset.relative_to(directory))


def retain_native_inventory(manifest: dict, original: dict, source: Path,
                            source_sha: str, directory: Path) -> None:
    if original.get("kind") != "lego-data-buckets" or original.get("schema_version") != 1:
        raise ValueError("Native inventory requires a supported immutable bucket snapshot")
    retained = [row for row in original["problems"] if row["source"]["kind"] == "native"]
    identities = {row["id"] for row in retained}
    if (not retained or len(identities) != len(retained)
            or identities.intersection(row["id"] for row in manifest["problems"])):
        raise ValueError("Native inventory is empty or duplicates problem identities")
    scope = "native-historical-screening"
    provenance = {
        "snapshot_id": original["snapshot_id"], "manifest_sha256": source_sha,
        "native_pool_sha256": original["provenance"]["native_pool_sha256"],
    }
    for problem in retained:
        if (problem["source"]["split"] != "train"
                or hashlib.sha256(Path(problem["source"]["path"]).read_bytes()).hexdigest()
                != problem["source"]["sha256"]):
            raise ValueError("Native inventory scene differs from its pinned provenance")
        problem["measurement_scope"] = scope
        problem["population_provenance"] = provenance
        for image in problem.get("initial_images", []):
            relative = Path(image["path"])
            path = (source.parent / relative).resolve()
            if relative.is_absolute() or not path.is_relative_to(source.parent.resolve()):
                raise ValueError("Native image escapes its pinned snapshot")
            image["path"] = copy_image(path, directory, image["sha256"])
    historical = [
        {**row, "measurement_scope": scope, "provenance": provenance}
        for row in original["groups"] if row["problem_id"] in identities
    ]
    manifest["measurement_scopes"][scope] = {
        "label": "Native historical four-rollout screening", "status": "screening",
        "provenance": provenance, "protocol": original["protocol"],
        "problem_count": len(retained), "policy_rollouts": sum(row["attempts"] for row in historical),
    }
    manifest["problems"].extend(retained)
    manifest["groups"].extend(historical)
    for bucket in original["buckets"]:
        tasks = []
        for task in bucket["tasks"]:
            ids = [identity for identity in task["problem_ids"] if identity in identities]
            if ids:
                tasks.append({**task, "problem_ids": ids, "population_total": len(ids)})
        if tasks:
            manifest["buckets"].append({
                **bucket, "tasks": tasks, "population_total": sum(row["population_total"] for row in tasks),
                "measurement_scope": scope,
            })
    manifest["progress"]["native_inventory_problems"] = len(retained)
    manifest["progress"]["historical_native_policy_rollouts"] = sum(row["attempts"] for row in historical)
    manifest["provenance"]["native_historical_snapshot"] = provenance
    manifest["protocol"]["population"] = "native-piles"


def publication_pointer(output: Path, publication_id: str | None) -> Path:
    if publication_id is not None and (
            not publication_id or publication_id in (".", "..") or Path(publication_id).name != publication_id):
        raise ValueError("Publication identity must be a single directory name")
    return output / "current.json" if publication_id is None else (
        output / "publications" / publication_id / "current.json"
    )


def publish_manifest(manifest: dict, directory: Path, output: Path, publication_id: str | None) -> dict:
    current = publication_pointer(output, publication_id)
    destination = directory / "manifest.json"
    if not destination.exists():
        atomic_json(destination, manifest)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    relative = str(destination.relative_to(output))
    pointer = {"schema_version": 1, "snapshot_id": manifest["snapshot_id"], "path": relative,
               "sha256": digest, "manifest_path": relative, "manifest_sha256": digest}
    atomic_json(current, pointer)
    return pointer


def publish_native_channel(original: dict, source: Path, source_bytes: bytes,
                           implementation: bytes, output: Path) -> dict:
    identity = "native-" + original["provenance"]["native_pool_sha256"]
    snapshot_id = hashlib.sha256(implementation + b"native-only" + source_bytes).hexdigest()
    directory = output / "snapshots" / snapshot_id
    manifest = {
        "kind": "lego-data-buckets", "schema_version": 1, "snapshot_id": snapshot_id,
        "dataset_id": identity, "generated_at": datetime.now(timezone.utc).isoformat(),
        "complete": False, "measurement_status": "screening",
        "problems": [], "groups": [], "buckets": [], "measurement_scopes": {},
        "progress": {"phase": "historical_native_screening", "policy_rollouts": 0},
        "protocol": dict(original["protocol"]), "provenance": dict(original["provenance"]),
        "notes": ["Historical native screening, not current photo calibration or training qualification."],
    }
    retain_native_inventory(manifest, original, source, hashlib.sha256(source_bytes).hexdigest(), directory)
    return publish_manifest(manifest, directory, output, identity)


def publish(pool_path: Path, references_path: Path, output: Path,
            measurements_path: Path | None = None, images_paths: list[Path] | None = None,
            native_snapshot_path: Path | None = None, native_snapshot_sha256: str | None = None,
            publication_id: str | None = None):
    publication_pointer(output, publication_id)
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
                "measurement_scope": "photo-assistance-calibration",
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
                "measurement_scope": "photo-assistance-calibration",
                "provenance": {"native_pool_sha256": pool_sha},
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
    if (native_snapshot_path is None) != (native_snapshot_sha256 is None):
        raise ValueError("Native snapshot path and pinned SHA256 must be supplied together")
    native_bytes = native_snapshot_path.read_bytes() if native_snapshot_path is not None else b""
    if native_snapshot_path is not None and hashlib.sha256(native_bytes).hexdigest() != native_snapshot_sha256:
        raise ValueError("Native snapshot differs from its pinned SHA256")
    dataset_id = publication_id or "photo-" + pool_sha
    snapshot_id = hashlib.sha256(
        implementation + dataset_id.encode() + pool_bytes + reference_bytes + measurement_bytes + image_digest.digest(),
    ).hexdigest()
    directory = output / "snapshots" / snapshot_id
    for problem, receipt, image in images:
        problem["initial_images"].append({
            **receipt, "path": copy_image(image, directory, receipt["sha256"]), "renderer": receipt["backend"],
            "provenance": receipt,
        })
        problem["initial_images_status"] = "published"
    manifest = {
        "kind": "lego-data-buckets", "schema_version": 1, "snapshot_id": snapshot_id,
        "dataset_id": dataset_id,
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
                     "horizon": measurements.get("horizon") if measurements is not None else None,
                     "measurement_scope": measurements.get("measurement_scope")
                     if measurements is not None else None,
                     "measurement_results_sha256": hashlib.sha256(measurement_bytes).hexdigest()
                     if measurements is not None else None,
                     "measurement_input_sha256": measurements["input_sha256"]
                     if measurements is not None else None,
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
                  "Native historical screening belongs only to its original native problems, not photo outcomes."],
    }
    manifest["measurement_scopes"] = {
        "photo-assistance-calibration": {
            "label": "Current photo-pile assistance calibration",
            "status": progress["policy_calibration_status"],
            "provenance": {"native_pool_sha256": pool_sha},
            "protocol": dict(manifest["protocol"]), "problem_count": len(problems),
            "policy_rollouts": progress["policy_rollouts"],
        },
    }
    if native_snapshot_path is not None:
        if publication_id is None:
            raise ValueError("Native inventory bootstrap requires an independent photo publication identity")
        publish_native_channel(json.loads(native_bytes), native_snapshot_path, native_bytes, implementation, output)
    pointer = publish_manifest(manifest, directory, output, publication_id)
    all_images = [image for problem in problems for image in problem["initial_images"]]
    return {**pointer, "problems": len(problems), "initial_images": len(all_images),
            "problems_with_images": sum(bool(problem["initial_images"]) for problem in problems),
            "cameras": dict(Counter(image["camera"] for image in all_images)),
            "tasks": dict(Counter(problem["task"] for problem in problems)), "progress": progress}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pool", "references", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--measurements", type=Path)
    parser.add_argument("--images", type=Path, action="append")
    parser.add_argument("--native-snapshot", type=Path)
    parser.add_argument("--native-snapshot-sha256")
    parser.add_argument("--publication-id")
    args = parser.parse_args()
    print(json.dumps(publish(args.pool, args.references, args.output, args.measurements, args.images,
                             args.native_snapshot, args.native_snapshot_sha256, args.publication_id)))


if __name__ == "__main__":
    main()
