"""Consume one bounded learner request without learning or policy evaluation."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path

from so101.cloud_checkpoint import atomic_json
from so101.pile_task_generator import (
    candidate_rejection, generate_scene, recover_entry, resource_check,
)
from so101.robot_sim import MODEL_ROOT


def validate_request(request: dict) -> None:
    if request.get("schema_version") != 1 or request.get("kind") != "pile-task-generation":
        raise ValueError("Unsupported pile request schema")
    identity = request.get("request_id")
    if (not isinstance(identity, str) or not identity
            or any(not (char.isalnum() or char in "-_") for char in identity)):
        raise ValueError("Invalid request identity")
    if request.get("split") != "train":
        raise ValueError("Learner requests must be TRAIN-only")
    for key in ("output_root", "catalog"):
        if not isinstance(request.get(key), str) or not Path(request[key]).is_absolute():
            raise ValueError(f"{key} must be absolute")
    if "model_root" in request and (
        not isinstance(request["model_root"], str) or not Path(request["model_root"]).is_absolute()
    ):
        raise ValueError("model_root must be absolute")
    if type(request.get("seed")) is not int or request["seed"] < 0:
        raise ValueError("Request requires a nonnegative integer seed")
    if type(request.get("max_scenes")) is not int or not 1 <= request["max_scenes"] <= 24:
        raise ValueError("Request must bound max_scenes to 1..24")
    tags = request.get("tag_constraints", {})
    if set(tags) - {"piece_count", "color_mode"}:
        raise ValueError("Unsupported tag constraints; cannot silently ignore requested populations")
    if type(tags.get("piece_count")) is not int or not 2 <= tags["piece_count"] <= 128:
        raise ValueError("Require piece_count in 2..128")
    if tags.get("color_mode") not in ("monochrome", "few", "many"):
        raise ValueError("Unsupported color distribution")
    if request.get("task_types") != ["view", "grasp", "push", "place"]:
        raise ValueError("Current generator emits all four tasks per target; other task sets unsupported")
    exclusions = request.get("excluded_scene_families")
    if not isinstance(exclusions, list) or not all(isinstance(item, str) for item in exclusions):
        raise ValueError("Explicit excluded_scene_families list required")


def process_request(path: Path) -> dict:
    request = json.loads(path.read_text())
    identity = request.get("request_id", "")
    if not isinstance(identity, str) or not identity or any(
        not (char.isalnum() or char in "-_") for char in identity
    ):
        raise ValueError("Cannot write result for an invalid request identity")
    result_path = path.parent / f"{identity}.result.json"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with (path.parent / f"{identity}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if result.get("request_sha256") != digest:
                raise ValueError("Request identity reused with different contents")
            return result
        result = {"schema_version": 1, "request_id": identity, "request_sha256": digest}
        try:
            validate_request(request)
            root = Path(request["output_root"]) / identity
            root.parent.mkdir(parents=True, exist_ok=True)
            resource_check(root.parent)
            root.mkdir(exist_ok=True)
            binding = root / "request_sha256.json"
            if binding.exists() and json.loads(binding.read_text()) != digest:
                raise ValueError("Output directory belongs to a different request")
            if not binding.exists():
                if any(root.iterdir()):
                    raise ValueError("Cannot adopt an unbound nonempty output directory")
                atomic_json(binding, digest)
            entries, rejected = [], []
            tags = request["tag_constraints"]
            for sequence in range(request["max_scenes"]):
                family = hashlib.sha256(f"pile-v1:{request['seed']}:{sequence}".encode()).hexdigest()
                if family in request["excluded_scene_families"]:
                    rejected.append({"sequence": sequence, "reason": "excluded_family"})
                    continue
                resource_check(root.parent)
                try:
                    output = root / f"scene_{sequence:06d}"
                    entry = recover_entry(output)
                    if entry is None:
                        entry = generate_scene(
                            Path(request["catalog"]), output,
                            seed=request["seed"], sequence=sequence,
                            count=tags["piece_count"], color_mode=tags["color_mode"],
                            model_root=Path(request.get("model_root", str(MODEL_ROOT))),
                        )
                    entries.append(entry)
                except ValueError as error:
                    if not candidate_rejection(error):
                        raise
                    rejected.append({"sequence": sequence, "reason": str(error)})
            manifest = {
                "schema_version": 1, "kind": "lego-pile-task-pool",
                "status": "complete", "request_id": identity, "scenes": entries,
                "rejections": rejected, "expert_references": False,
            }
            manifest_path = root / "manifest.json"
            atomic_json(manifest_path, manifest)
            result.update(
                status="complete" if entries else "blocked",
                manifest_path=str(manifest_path.resolve()),
                sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                generated_scenes=len(entries), rejections=rejected,
            )
        except BlockingIOError as error:
            result.update(status="blocked", error=str(error))
        except (ValueError, OSError, RuntimeError) as error:
            result.update(status="rejected", error=f"{type(error).__name__}: {error}")
        atomic_json(result_path, result)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    os.nice(19)
    result = process_request(args.request)
    print(json.dumps(result), flush=True)
    if result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
