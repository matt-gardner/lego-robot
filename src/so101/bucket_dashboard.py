"""Read-only, hash-pinned learning bucket publications."""
import hashlib
import json
import logging
from pathlib import Path
import threading

LOGGER = logging.getLogger(__name__)

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
        self._cache = None
        self._cache_error = None
        self._cache_lock = threading.Lock()
        self._cache_ready = threading.Event()
        self._cache_stop = threading.Event()
        self._cache_thread = None

    def start_cache(self, interval: float = 20) -> None:
        def refresh():
            while not self._cache_stop.is_set():
                self.refresh_cache()
                self._cache_stop.wait(interval)
        self._cache_thread = threading.Thread(target=refresh, name="bucket-dashboard-cache", daemon=True)
        self._cache_thread.start()

    def stop_cache(self) -> None:
        self._cache_stop.set()
        if self._cache_thread:
            self._cache_thread.join()

    def refresh_cache(self) -> None:
        try:
            snapshot = self.snapshot()
            body = json.dumps(snapshot).encode()
            index = {p["id"]: p for p in [
                *snapshot["problems"]["problems"], *snapshot.get("admitted_piles", [])]}
            with self._cache_lock:
                self._cache = (snapshot, body, index)
                self._cache_error = None
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            LOGGER.exception("Cannot refresh bucket dashboard cache")
            with self._cache_lock:
                self._cache_error = error
        finally:
            self._cache_ready.set()

    def cached(self):
        self._cache_ready.wait()
        with self._cache_lock:
            if self._cache_error is not None:
                raise self._cache_error
            return self._cache

    def response(self) -> bytes:
        return self.cached()[1] if self._cache_thread else json.dumps(self.snapshot()).encode()

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
        admissions = self.admitted_piles()
        for pile in admissions:
            for problem in pile.pop("tasks"):
                if problem["id"] not in identities:
                    identities.add(problem["id"])
                    problems.append(problem)
        task_scenes = {p.get("source", {}).get("scene_id") for p in problems}
        admissions = [p for p in admissions if p["scene_id"] not in task_scenes]
        pilots = self.distance_pilots(pointers)
        revision = hashlib.sha256(json.dumps(
            [[p["snapshot_id"] for p in datasets], admissions,
             [(p["id"], p["dataset_snapshot_id"]) for p in problems],
             pilots], sort_keys=True).encode()).hexdigest()
        return {"manifest": {"snapshot_id": revision, "datasets": datasets},
                "admitted_piles": admissions,
                "distance_pilots": pilots,
                "problems": {"problems": problems, "buckets": buckets},
                "calibration": {"groups": groups, "protocol": {},
                                "qualification_status": "multiple datasets",
                                "complete": all(p["calibration"]["complete"] for p in publications)}}

    def distance_pilots(self, pointers: list[Path]) -> list:
        pilots = []
        for pointer in pointers:
            publication = self.publication(str(pointer.relative_to(self.root)), compose=False)
            problems = publication["problems"]["problems"]
            assisted = [p for p in problems if p.get("tags", {}).get(
                "task_assistance", {}).get("kind") == "direct-task-distance-v2"]
            if not assisted:
                continue
            if len(assisted) != len(problems):
                raise ValueError("Distance pilot mixes assisted and original task starts")
            identities = {p["id"] for p in assisted}
            parents = set()
            for problem in assisted:
                tags = problem["tags"]
                parent = tags["parent_task_id"]
                assistance = tags["task_assistance"]
                if parent != assistance["parent_task_id"] or not parent:
                    raise ValueError("Distance pilot parent identity differs")
                distance = assistance["distance"]
                if distance["active_stage"] not in distance["stage_order"]:
                    raise ValueError("Distance pilot stage is missing from task order")
                parents.add(parent)
            manifest = publication["manifest"]
            series = manifest.get("measurement_series", [])
            groups = ([{**g, "measurement_series_id": s["series_id"], "protocol": s["protocol"]}
                       for s in series for g in s["groups"]] if series else
                      publication["calibration"]["groups"])
            if any(g["problem_id"] not in identities for g in groups):
                raise ValueError("Distance pilot measurement crosses child population")
            pilots.append({"manifest": manifest, "problems": assisted, "groups": groups,
                           "parent_count": len(parents), "child_count": len(assisted)})
        superseded = {p["manifest"].get("provenance", {}).get("supersedes_native_pool_sha256")
                      for p in pilots}
        return [p for p in pilots
                if p["manifest"].get("provenance", {}).get("native_pool_sha256") not in
                (superseded - {None})]

    def admitted_piles(self) -> list:
        pins = sorted(self.root.glob("pile-previews-v1/*/admission.json"))
        sources = {pin.parent.name: (pin, False) for pin in pins}
        if self.curriculum_root:
            for workstream in ("photo-task-progression-collective-v1", "catalog-diverse-20-v1",
                               "catalog-diverse-larger-v1"):
                for pin in sorted((self.curriculum_root / workstream).glob(
                        "pieces-*/tasks/scene_*/admission.json")):
                    body = self.curriculum_read(pin)
                    sources[hashlib.sha256(body).hexdigest()] = (pin, True)
        pins = sorted(sources.items())
        if len(pins) > 20000:
            raise ValueError("Too many admitted pile previews")
        piles = []
        for sha, (pin, generator_source) in pins:
            admission = json.loads(self.curriculum_read(pin, sha) if generator_source else
                                   self.read(str(pin.relative_to(self.root)), sha))
            if admission["split"] != "train":
                raise ValueError("Pile preview crosses TRAIN admission")
            scene = json.loads(self.curriculum_read(
                Path(admission["root"]) / "problem.json", admission["files_sha256"]["problem.json"]))
            images = []
            preview_directory = self.root / "pile-previews-v1" / sha
            for path in sorted(preview_directory.glob("*.json")):
                if path.name == "admission.json":
                    continue
                row = json.loads(self.read(str(path.relative_to(self.root)), limit=1024 * 1024))
                task = next((t for t in scene["tasks"]
                             if str(Path(admission["root"]) / "tasks" / t["task_id"]) == row["episode_source"]), None)
                if (task is None or row["pool_sha256"] != sha
                        or row["scene_files_sha256"]["problem.json"] != admission["files_sha256"]["problem.json"]
                        or row["target_instance_id"] != task["target_instance_id"]
                        or row["problem_id"] != hashlib.sha256(row["episode_source"].encode()).hexdigest()
                        or row["physics_unchanged"] is not True or row["assistance"] != 0
                        or row["is_gpu_policy_frame"] is not False):
                    raise ValueError("Pile preview differs from pinned admission")
                if Path(row["image_file"]).name != row["image_file"]:
                    raise ValueError("Pile preview asset escapes receipt directory")
                validate_annotation(row, row["width"], row["height"])
                for annotation in row.get("target_annotations", []):
                    validate_annotation(annotation, row["width"], row["height"])
                images.append({**row, "renderer": row["backend"], "root_relative": True,
                               "path": str(path.parent.relative_to(self.root) / row["image_file"])})
            revision = hashlib.sha256(json.dumps([sha, images], sort_keys=True).encode()).hexdigest()
            tasks = []
            for task in scene["tasks"]:
                source = str(Path(admission["root"]) / "tasks" / task["task_id"])
                identity = hashlib.sha256(source.encode()).hexdigest()
                tasks.append({
                    "id": identity, "bucket": f'current:{scene["tags"]["piece_count"]}-pieces',
                    "task": task["task_type"],
                    "source": {"kind": "admitted", "episode_source": source,
                               "path": str(Path(admission["root"]) / "problem.json"),
                               "sha256": admission["files_sha256"]["problem.json"],
                               "scene_id": scene["scene_id"], "task_id": task["task_id"], "split": "train"},
                    "tags": {"scene": scene["tags"], "target_instance_id": task["target_instance_id"],
                             "goal": task.get("goal"), "task_start": task.get("task_start")},
                    "measurement_status": "unmeasured", "feasibility": {"status": "unmeasured"},
                    "dataset_snapshot_id": revision, "snapshot_directory": "",
                    "initial_images": [image for image in images if image["problem_id"] == identity],
                })
            piles.append({"id": "admission:" + sha, "admission_sha256": sha, "tasks": tasks,
                          "scene_id": scene["scene_id"], "tags": scene["tags"],
                          "status": "admitted pile preview — not a complete task pool",
                          "initial_images": images, "dataset_snapshot_id": revision})
        return piles

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
        self.shared_metrics(publication)
        self.live_metrics(publication)
        self.live_images(publication)
        return publication

    def shared_metrics(self, publication: dict) -> None:
        pool = publication["manifest"]["provenance"]["native_pool_sha256"]
        pointers = sorted(self.root.glob("publications/*/current.json"))
        if len(pointers) > 100:
            raise ValueError("Too many bucket publications")
        matches = []
        for pointer in pointers:
            candidate = self.publication(str(pointer.relative_to(self.root)), compose=False)
            if candidate["manifest"].get("provenance", {}).get("native_pool_sha256") == pool:
                matches.append(candidate)
        if len(matches) > 1:
            raise ValueError("Multiple shared measurement publications for selected pool")
        if not matches:
            return
        candidate = matches[0]
        selected = {p["id"]: p for p in publication["problems"]["problems"]}
        measured = {p["id"]: p for p in candidate["problems"]["problems"]}
        series = candidate["manifest"].get("measurement_series", [])
        if not isinstance(series, list) or len(series) > 100:
            raise ValueError("Invalid retained measurement series")
        series_ids = [s["series_id"] for s in series]
        if len(set(series_ids)) != len(series_ids):
            raise ValueError("Duplicate retained measurement series identity")
        all_groups = [*candidate["calibration"]["groups"],
                      *(g for s in series for g in s["groups"])]
        for group in all_groups:
            identity = group["problem_id"]
            if identity not in selected or identity not in measured:
                raise ValueError("Shared measurement crosses selected task population")
            current, original = selected[identity], measured[identity]
            if (current["task"] != original["task"]
                    or current["source"]["episode_source"] != original["source"]["episode_source"]
                    or current["source"]["sha256"] != original["source"]["sha256"]):
                raise ValueError("Shared measurement source differs from selected task")
        publication["calibration"] = candidate["calibration"]
        manifest = publication["manifest"]
        for key in ("protocol", "progress", "measurement_status", "measurement_scopes", "complete",
                    "measurement_series", "current_measurement_series_id", "latest_measurement_series_id"):
            if key in candidate["manifest"]:
                manifest[key] = candidate["manifest"][key]
        if series:
            combined = [{**group, "measurement_series_id": s["series_id"],
                         "protocol": s["protocol"]}
                        for s in series for group in s["groups"]]
            publication["calibration"] = {
                **candidate["calibration"], "groups": combined,
                "complete": all(s["complete"] for s in series),
            }
            manifest["progress"] = {**manifest.get("progress", {}),
                                    "policy_rollouts": sum(g["attempts"] for g in combined)}
            manifest["complete"] = publication["calibration"]["complete"]
        manifest["snapshot_id"] = hashlib.sha256(
            (manifest["snapshot_id"] + candidate["manifest"]["snapshot_id"]).encode()).hexdigest()

    def publication(self, pointer_path: str, compose: bool = True) -> dict:
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
        if compose:
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
                    or row["backend"] not in ("mitsuba-llvm_ad_rgb", "mitsuba-cuda_ad_rgb")):
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
        if self._cache_thread:
            snapshot, _, identities = self.cached()
            problems = [identities[problem_id]] if problem_id in identities else []
        else:
            snapshot = self.snapshot()
            problems = [row for row in [*snapshot["problems"]["problems"], *snapshot.get("admitted_piles", [])]
                        if row["id"] == problem_id]
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
