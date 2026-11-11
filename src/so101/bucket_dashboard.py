"""Read-only, hash-pinned learning bucket publications."""
import hashlib
import json
from pathlib import Path

def validate_annotation(row, width, height):
    status, box, pixels = (row[key] for key in
                           ("target_bbox_status", "target_bbox_xyxy", "target_visible_pixels"))
    if type(pixels) is not int or pixels < 0:
        raise ValueError("Invalid visible pixel count")
    if status == "visible":
        if (not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box)
                or not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)
                or not 0 < pixels <= (box[2] - box[0]) * (box[3] - box[1])):
            raise ValueError("Visible annotation has invalid exclusive pixel bounds/count")
    elif status == "not_visible":
        if box is not None or pixels != 0:
            raise ValueError("Invisible annotation cannot have a fabricated box or positive pixel count")
    else:
        raise ValueError("Captured segmentation must measure visible or not_visible")


class BucketDashboard:
    def __init__(self, root: Path, curriculum_root: Path | None = None):
        self.root = root.resolve()
        self.curriculum_root = curriculum_root.resolve() if curriculum_root else None

    def read(self, relative: str, expected: str | None = None, limit: int = 32 * 1024 * 1024) -> bytes:
        path = (self.root / relative).resolve()
        if Path(relative).is_absolute() or not path.is_relative_to(self.root):
            raise ValueError("Bucket artifact path escapes publication root")
        with path.open("rb") as stream:
            body = stream.read(limit + 1)
        if len(body) > limit:
            raise ValueError("Bucket artifact exceeds read limit")
        if expected is not None and hashlib.sha256(body).hexdigest() != expected:
            raise ValueError("Bucket artifact hash mismatch")
        return body

    def snapshot(self) -> dict:
        selection = self.curriculum_root / "current-task-pools.json" if self.curriculum_root else None
        pointers = sorted(self.root.glob("publications/*/current.json"))
        if selection and selection.exists():
            descriptor = json.loads(self.curriculum_read(selection))
            if descriptor.get("kind") != "lego-current-task-pools" or descriptor.get("schema_version") != 1:
                raise ValueError("Unsupported current task pool selection")
            pools = descriptor["pools"]
            if not isinstance(pools, list) or not 1 <= len(pools) <= 100:
                raise ValueError("Invalid current task pool population")
            publications = [self.live_pool(pool) for pool in pools]
        elif not pointers:
            return self.publication("current.json")
        else:
            if len(pointers) > 100:
                raise ValueError("Too many bucket publications")
            publications = [self.publication(str(path.relative_to(self.root))) for path in pointers]
        problems, groups, buckets, datasets, identities = [], [], [], [], set()
        for publication in publications:
            manifest = publication["manifest"]
            dataset = manifest["snapshot_id"]
            datasets.append(manifest)
            for problem in publication["problems"]["problems"]:
                if problem["id"] in identities:
                    raise ValueError("Duplicate problem across bucket publications")
                identities.add(problem["id"])
                problems.append({**problem, "dataset_snapshot_id": dataset,
                                 "snapshot_directory": publication["snapshot_directory"]})
            groups.extend({**group, "dataset_snapshot_id": dataset}
                          for group in publication["calibration"]["groups"])
            buckets.extend(publication["problems"]["buckets"])
        revision = hashlib.sha256(json.dumps([p["snapshot_id"] for p in datasets]).encode()).hexdigest()
        return {"manifest": {"snapshot_id": revision, "datasets": datasets},
                "problems": {"problems": problems, "buckets": buckets},
                "calibration": {"groups": groups, "protocol": {},
                                "qualification_status": "multiple datasets",
                                "complete": all(p["calibration"]["complete"] for p in publications)}}

    def curriculum_path(self, source: str) -> Path:
        path = Path(source)
        producer = Path("/mnt/dockyard/lego-robot-data/data/lego_curriculum")
        if path.is_relative_to(producer):
            path = self.curriculum_root / path.relative_to(producer)
        elif not path.is_absolute():
            path = self.curriculum_root / path
        path = path.resolve()
        if not path.is_relative_to(self.curriculum_root):
            raise ValueError("Task pool path escapes curriculum root")
        return path

    def curriculum_read(self, path: Path, expected: str | None = None) -> bytes:
        path = self.curriculum_path(str(path))
        with path.open("rb") as stream:
            body = stream.read(32 * 1024 * 1024 + 1)
        if len(body) > 32 * 1024 * 1024:
            raise ValueError("Task pool file exceeds read limit")
        if expected is not None and hashlib.sha256(body).hexdigest() != expected:
            raise ValueError("Task pool file hash mismatch")
        return body

    def live_pool(self, selected: dict) -> dict:
        body = self.curriculum_read(Path(selected["manifest_path"]), selected["sha256"])
        pool = json.loads(body)
        if pool["kind"] != "lego-pile-task-pool" or pool["status"] != "complete":
            raise ValueError("Current task pool is not complete")
        problems, buckets = [], {}
        for entry in pool["scenes"]:
            if entry["split"] != "train":
                raise ValueError("Current task pool crosses admitted TRAIN population")
            for name in entry["files_sha256"]:
                if Path(name).name != name:
                    raise ValueError("Invalid task scene dependency path")
            scene = json.loads(self.curriculum_read(
                Path(entry["root"]) / "problem.json", entry["files_sha256"]["problem.json"]))
            bucket = f'current:{scene["tags"]["piece_count"]}-pieces'
            tasks = buckets.setdefault(bucket, {})
            for task in scene["tasks"]:
                source = str(Path(entry["root"]) / "tasks" / task["task_id"])
                identity = hashlib.sha256(source.encode()).hexdigest()
                tasks.setdefault(task["task_type"], []).append(identity)
                problems.append({
                    "id": identity, "bucket": bucket, "task": task["task_type"],
                    "source": {"kind": "current", "episode_source": source,
                               "path": str(Path(entry["root"]) / "problem.json"),
                               "sha256": entry["files_sha256"]["problem.json"],
                               "scene_id": scene["scene_id"], "task_id": task["task_id"], "split": "train"},
                    "tags": {"scene": scene["tags"], "target_instance_id": task["target_instance_id"],
                             "goal": task["goal"], "task_start": task.get("task_start")},
                    "initial_images": [], "initial_images_status": "not-yet-rendered",
                    "feasibility": {"status": "unmeasured"},
                })
        if len(problems) != selected["task_count"] or len(pool["scenes"]) != selected["scenes"]:
            raise ValueError("Current task pool population differs from selection")
        manifest = {"snapshot_id": selected["sha256"], "dataset_id": selected["sha256"],
                    "measurement_status": "unmeasured", "protocol": {},
                    "provenance": {"native_pool_sha256": selected["sha256"],
                                   "pool_path": selected["manifest_path"]}}
        publication = {
            "manifest": manifest, "snapshot_directory": "",
            "problems": {"problems": problems, "buckets": [
                {"id": bucket, "label": bucket, "population_total": sum(map(len, tasks.values())),
                 "tasks": [{"task": task, "population_total": len(ids), "problem_ids": ids}
                           for task, ids in sorted(tasks.items())]}
                for bucket, tasks in sorted(buckets.items())]},
            "calibration": {"groups": [], "protocol": {}, "qualification_status": "unmeasured", "complete": False},
        }
        self.live_metrics(publication)
        self.live_images(publication)
        return publication

    def publication(self, pointer_path: str) -> dict:
        pointer = json.loads(self.read(pointer_path, limit=1024 * 1024))
        relative = pointer["path"]
        manifest = json.loads(self.read(relative, pointer["sha256"]))
        if manifest.get("schema_version") != 1 or manifest.get("kind") != "lego-data-buckets":
            raise ValueError("Unsupported bucket manifest")
        if not isinstance(manifest.get("problems"), list):
            raise ValueError("Bucket problems population is missing")
        if not isinstance(manifest.get("groups"), list):
            raise ValueError("Bucket measurement groups are missing")
        problems = []
        identities = set()
        for problem in manifest["problems"]:
            identity = problem["id"]
            if not isinstance(identity, str) or identity in identities:
                raise ValueError("Bucket problem identity missing or duplicated")
            identities.add(identity)
            problems.append({**problem, "bucket": problem.get("bucket_id", problem.get("bucket"))})
        publication = {
            "manifest": {key: value for key, value in manifest.items()
                         if key not in ("problems", "groups", "buckets", "plan")},
            "problems": {"problems": problems, "buckets": manifest.get("buckets", [])},
            "calibration": {"groups": manifest["groups"], "protocol": manifest.get("protocol", {}),
                            "complete": manifest.get("complete", False),
                            "qualification_status": manifest.get("measurement_status", "unknown")},
            "snapshot_directory": str(Path(relative).parent),
        }
        self.live_metrics(publication)
        self.live_images(publication)
        return publication

    def live_metrics(self, publication: dict) -> None:
        pool = publication["manifest"].get("provenance", {}).get("native_pool_sha256")
        paths = sorted(self.root.glob("live-metrics/*/results.json"))
        if len(paths) > 100:
            raise ValueError("Too many live metric files")
        selected = []
        for path in paths:
            body = self.read(str(path.relative_to(self.root)))
            row = json.loads(body)
            if row["input_sha256"]["native_pool"] == pool:
                selected.append((body, row))
        if not selected:
            return
        if len(selected) != 1:
            raise ValueError("Multiple live metric protocols for one pool; select a unique current metrics file")
        body, row = selected[0]
        if row["optimizer"] is not None or not row["policy_sha256_before"] or not isinstance(row["horizon"], int):
            raise ValueError("Live bucket metrics must describe an explicit inference-only protocol")
        by_source = {p["source"]["episode_source"]: p for p in publication["problems"]["problems"]}
        groups = []
        for group in row["groups"]:
            problem = by_source[group["source"]]
            rollouts = group["rollouts"]
            if group["task"] != problem["task"] or group["policy_step"] != 0 or not rollouts:
                raise ValueError("Live metric group differs from selected inference task")
            groups.append({
                "problem_id": problem["id"], "assistance": group["assistance"],
                "attempts": len(rollouts), "successes": sum(bool(r["success"]) for r in rollouts),
                "invalid": sum(bool(r["invalid_action_error"]) for r in rollouts),
                "measurement_scope": row["measurement_scope"], "horizon": row["horizon"],
                "status": "screening",
            })
        manifest = publication["manifest"]
        manifest["snapshot_id"] = hashlib.sha256(
            manifest.get("snapshot_id", "").encode() + body).hexdigest()
        protocol = {"horizon": row["horizon"], "measurement_scope": row["measurement_scope"],
                    "measurement_results_sha256": hashlib.sha256(body).hexdigest(),
                    "measurement_input_sha256": row["input_sha256"]}
        manifest["protocol"] = protocol
        manifest["measurement_status"] = "screening"
        count = sum(g["attempts"] for g in groups)
        manifest["measurement_scopes"] = {row["measurement_scope"]: {
            "label": f'{row["horizon"]}-step {row["measurement_scope"]}',
            "protocol": protocol, "policy_rollouts": count, "status": "screening",
        }}
        manifest["progress"] = {**manifest.get("progress", {}),
                               "policy_rollouts": count, "phase": row["measurement_scope"],
                               "policy_calibration_status": "complete" if row["complete"] else "running"}
        publication["calibration"] = {"groups": groups, "protocol": protocol,
                                      "qualification_status": "screening", "complete": row["complete"]}

    def live_images(self, publication: dict) -> None:
        """Compose completed renderer receipts directly, without a publishing step."""
        pool = publication["manifest"].get("provenance", {}).get("native_pool_sha256")
        if not isinstance(pool, str) or len(pool) != 64 or any(c not in "0123456789abcdef" for c in pool):
            return
        versions = []
        for directory in self.root.glob("initial-photo-external-v*"):
            version = directory.name.removeprefix("initial-photo-external-v")
            if version.isascii() and version.isdecimal() and int(version) > 0:
                versions.append((int(version), directory))
        if len(versions) > 100:
            raise ValueError("Too many live renderer versions")
        paths = {}
        for _, directory in sorted(versions):
            paths.update({path.name: path for path in (directory / pool).glob("*.json")})
            paths.update({path.name: path for path in (directory / pool / "_roles-v1").glob("*.json")})
        if len(paths) > 20000:
            raise ValueError("Too many live renderer receipts")
        problems = {p["id"]: p for p in publication["problems"]["problems"]}
        digest = hashlib.sha256()
        for path in sorted(paths.values()):
            relative = str(path.relative_to(self.root))
            body = self.read(relative, limit=1024 * 1024)
            row = json.loads(body)
            problem = problems.get(row["problem_id"])
            if problem is None:
                raise ValueError("Live renderer receipt crosses selected population")
            source = problem["source"]
            if (row["pool_sha256"] != pool or row["episode_source"] != source["episode_source"]
                    or row["problem_id"] != hashlib.sha256(row["episode_source"].encode()).hexdigest()
                    or row["scene_files_sha256"]["problem.json"] != source["sha256"]
                    or row["target_instance_id"] != problem["tags"]["target_instance_id"]
                    or row["physics_unchanged"] is not True or row["assistance"] != 0
                    or row["is_gpu_policy_frame"] is not False or row["camera"] != "external"
                    or row["backend"] != "mitsuba-llvm_ad_rgb"):
                raise ValueError("Live renderer receipt differs from selected initial task")
            name = row["image_file"]
            if Path(name).name != name:
                raise ValueError("Live renderer asset must have a local basename")
            validate_annotation(row, row["width"], row["height"])
            for annotation in row.get("target_annotations", []):
                validate_annotation(annotation, row["width"], row["height"])
            image = {**row, "path": str(path.parent.relative_to(self.root) / name),
                     "root_relative": True, "renderer": row["backend"], "provenance": row}
            old = [i for i in problem.get("initial_images", []) if i.get("camera") != "external"]
            problem["initial_images"] = [image, *old]
            problem["initial_images_status"] = "live-renderer-receipt"
            digest.update(body)
        if paths:
            manifest = publication["manifest"]
            original = manifest.get("snapshot_id", "")
            manifest["snapshot_id"] = hashlib.sha256(original.encode() + digest.digest()).hexdigest()
            manifest["live_images"] = {"corrected_tasks": len(paths), "receipts_sha256": digest.hexdigest()}

    def image(self, problem_id: str, index: int, snapshot_id: str | None = None) -> tuple[bytes, str]:
        snapshot = self.snapshot()
        problems = [row for row in snapshot["problems"]["problems"] if row["id"] == problem_id]
        if len(problems) != 1 or index < 0:
            raise ValueError("Unknown bucket problem/image")
        revision = problems[0].get("dataset_snapshot_id", snapshot["manifest"].get("snapshot_id"))
        if snapshot_id is not None and revision != snapshot_id:
            raise ValueError("Bucket publication changed; refresh the page")
        images = problems[0].get("initial_images", [])
        if index >= len(images):
            raise ValueError("Unknown bucket problem/image")
        image = images[index]
        relative = Path(image["path"]) if image.get("root_relative") else (
            Path(problems[0].get("snapshot_directory", snapshot.get("snapshot_directory", ""))) / image["path"])
        body = self.read(str(relative), image["sha256"], limit=8 * 1024 * 1024)
        if body.startswith(b"\x89PNG\r\n\x1a\n"):
            return body, "image/png"
        if body.startswith(b"\xff\xd8\xff"):
            return body, "image/jpeg"
        raise ValueError("Bucket image must be PNG or JPEG")
