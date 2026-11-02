"""Read-only joins between source episodes, logged policy outcomes and solved entries."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import threading
import time
from pathlib import Path


def identifier(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:20]


class TrainingDataBrowser:
    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()
        self.data_root = self.workspace / "data" / "simulation"
        self.lock = threading.RLock()
        self.movie_lock = threading.Lock()
        self.cache = {}

    def source_path(self, source: str) -> Path:
        path = (self.workspace / source).resolve()
        if not path.is_relative_to(self.data_root.resolve()):
            raise ValueError("Source is outside simulation data")
        return path

    def index(self, log_path: Path) -> dict:
        with self.lock:
            log = json.loads(log_path.read_text())
            collections = ([] if log.get("kind") == "sft" else
                           sorted(self.data_root.glob("task_difficulty*/collection.json")))
            collection = log.get("run_config", {}).get("collection")
            collection_path = self.source_path(collection) / "collection.json" if collection else None
            split_path = log_path.parent / "data_split.json"
            files = [log_path, *collections, *([collection_path] if collection_path else []),
                     *([split_path] if split_path.exists() else [])]
            revision = tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in files)
            cached = self.cache.get(log_path)
            if cached and cached[0] == revision and time.monotonic() - cached[1] < 30:
                return cached[2]
            split = json.loads(split_path.read_text()) if split_path.exists() else {}
            train = {source for values in split.get("training", {}).values() for source in values}
            validation = set(split.get("validation", []))
            sources = train | validation
            if collection_path:
                report = json.loads(collection_path.read_text())
                sources.update(report["successful_episodes"])
            for path in collections:
                report = json.loads(path.read_text())
                sources.update(row["source"] for row in report.get("attempts", []) if row.get("source"))
            groups = [(step["step"], index, group)
                      for step in log.get("steps", [])
                      for index, group in enumerate(step.get("groups", []))
                      if group.get("source")]
            sources.update(group["source"] for _, _, group in groups)
            solved = list(log.get("solved_problems", {}).values())
            sources.update(item["source"] for item in solved)
            examples = {}
            by_source = {}
            for source in sorted(sources):
                metadata = json.loads((self.source_path(source) / "episode.json").read_text())
                difficulty = metadata.get("difficulty")
                row = {
                    "id": identifier(source), "source": source, "task": metadata["task"],
                    "target": metadata.get("target_label", metadata.get("target")),
                    "destination": metadata.get("destination"), "seed": metadata["initial"]["seed"],
                    "difficulty": difficulty, "difficulty_level": difficulty["level"] if difficulty else "unrated",
                    "expert_outcome": metadata.get("outcome", "unknown"),
                    "split": "validation" if source in validation else "training" if source in train else "not-in-run",
                    "used": False, "rollouts": 0, "passed": 0, "failed": 0,
                    "last_step": None, "solved_entries": [], "history": [],
                }
                examples[row["id"]] = row
                by_source[source] = row
            rollouts = {}
            source_rollouts = {item["id"]: [] for item in examples.values()}
            for step, group_index, group in groups:
                source = by_source[group["source"]]
                source["used"] = True
                source["last_step"] = step
                trajectories = group.get("trajectories", [])
                source["rollouts"] += len(trajectories)
                source["passed"] += sum(bool(item.get("success")) for item in trajectories)
                source["failed"] += sum(not item.get("success") for item in trajectories)
                source["history"].append({
                    "step": step, "assistance": group.get("start_proximity", 0.),
                    "bucket": group.get("sampling_bucket", "main"), "discarded": bool(group.get("discarded")),
                    "passed": sum(bool(item.get("success")) for item in trajectories), "total": len(trajectories),
                })
                for trajectory_index, item in enumerate(trajectories):
                    key = identifier(f"{step}:{group_index}:{trajectory_index}")
                    rollouts[key] = {
                        "id": key, "example_id": source["id"], "source": source["source"],
                        "task": source["task"], "target": source["target"],
                        "difficulty": source["difficulty"], "difficulty_level": source["difficulty_level"],
                        "split": source["split"], "used": True,
                        "step": step, "last_step": step, "assistance": group.get("start_proximity", 0.),
                        "bucket": group.get("sampling_bucket", "main"),
                        "discarded": bool(group.get("discarded")), **item,
                        "jerk_regularizer_applied": bool(group.get("jerk_regularizer_applied")),
                        "group_index": group_index, "trajectory_index": trajectory_index,
                    }
                    source_rollouts[source["id"]].append(key)
            easy = {}
            for item in solved:
                source = by_source[item["source"]]
                source["solved_entries"].append(item)
                key = identifier(json.dumps([item["source"], item["start_proximity"]]))
                history = [row for row in source["history"]
                           if abs(row["assistance"] - item["start_proximity"]) < 1e-6]
                easy[key] = {**source, **item, "id": key, "example_id": source["id"],
                             "assistance": item["start_proximity"],
                             "used": bool(history), "rollouts": sum(row["total"] for row in history),
                             "passed": sum(row["passed"] for row in history),
                             "failed": sum(row["total"] - row["passed"] for row in history),
                             "last_step": history[-1]["step"] if history else None}
            result = {"examples": examples, "rollouts": rollouts, "solved": easy,
                      "source_rollouts": source_rollouts,
                      "last_step": log.get("steps", [{}])[-1].get("step") if log.get("steps") else None}
            self.cache[log_path] = (revision, time.monotonic(), result)
            if len(self.cache) > 2:
                oldest = next(key for key in self.cache if key != log_path)
                del self.cache[oldest]
            return result

    def evidence_path(self, log_path: Path, row: dict) -> Path | None:
        reference = row.get("evidence")
        if not reference:
            return None
        path = (log_path.parent / reference).resolve()
        root = (log_path.parent / "rollouts").resolve()
        if (not root.is_relative_to(log_path.parent.resolve())
                or not path.is_relative_to(root) or path.suffix != ".json"):
            raise ValueError("Invalid rollout evidence reference")
        return path

    def list(self, log_path: Path, query: dict) -> dict:
        index = self.index(log_path)
        view = query.get("view", "examples")
        if view not in ("examples", "rollouts", "solved"):
            raise ValueError("Invalid data view")
        allowed = {
            "task": ("all", "grasp", "view", "push", "place"),
            "difficulty": ("all", "unrated", "0", "1", "2", "3"),
            "usage": ("all", "used", "unused", "validation"),
            "bucket": ("all", "solved"), "outcome": ("all", "passed", "failed"),
        }
        for key, values in allowed.items():
            if query.get(key, "all") not in values:
                raise ValueError(f"Invalid {key} filter")
        offset, limit = int(query.get("offset", 0)), int(query.get("limit", 24))
        if offset < 0 or not 1 <= limit <= 100:
            raise ValueError("Expected offset >= 0 and limit 1..100")
        rows = []
        for row in index[view].values():
            example = index["examples"][row.get("example_id", row["id"])]
            if query.get("task", "all") not in ("all", row["task"]):
                continue
            if query.get("difficulty", "all") not in ("all", str(row["difficulty_level"])):
                continue
            usage = query.get("usage", "all")
            if (usage == "used" and not row["used"] or usage == "unused" and row["used"]
                    or usage == "validation" and row["split"] != "validation"):
                continue
            bucket = query.get("bucket", "all")
            if bucket == "solved" and not example["solved_entries"]:
                continue
            outcome = query.get("outcome", "all")
            passed = int(bool(row.get("success"))) if view == "rollouts" else row["passed"]
            failed = int(not row.get("success")) if view == "rollouts" else row["failed"]
            if outcome == "passed" and not passed or outcome == "failed" and not failed:
                continue
            search = query.get("search", "").lower()
            if search and search not in f'{row["source"]} {row["target"]} {row.get("seed", "")}'.lower():
                continue
            rows.append(row)
        rows.sort(key=lambda row: (-(row.get("last_step") or -1), row["task"], row["source"], row["id"]))
        page = []
        for row in rows[offset:offset + limit]:
            value = {key: item for key, item in row.items() if key != "history"}
            evidence = self.evidence_path(log_path, row) if view == "rollouts" else None
            value["evidence_status"] = ("available" if evidence.exists() else "expired") if evidence else "not_recorded"
            page.append(value)
        return {
            "items": page, "total": len(rows), "offset": offset, "limit": limit,
            "last_step": index["last_step"],
            "counts": {name: len(index[name]) for name in ("examples", "rollouts", "solved")},
        }

    def detail(self, log_path: Path, view: str, item_id: str) -> dict:
        index = self.index(log_path)
        if view not in ("examples", "rollouts", "solved") or item_id not in index[view]:
            raise FileNotFoundError("Example or rollout not found")
        row = index[view][item_id]
        example = index["examples"][row.get("example_id", row["id"])]
        metadata = json.loads((self.source_path(row["source"]) / "episode.json").read_text())
        result = {**row, "metadata": metadata, "source_history": example["history"][-100:],
                  "source_history_total": len(example["history"]), "example_id": example["id"]}
        recorded = []
        for key in index["source_rollouts"][example["id"]]:
            candidate = index["rollouts"][key]
            evidence = self.evidence_path(log_path, candidate)
            if evidence and evidence.exists():
                recorded.append({field: candidate[field] for field in (
                    "id", "task", "target", "step", "assistance", "success",
                    "group_index", "trajectory_index",
                )})
        result["recorded_trajectories"] = sorted(
            recorded, key=lambda item: (item["step"], item["group_index"], item["trajectory_index"]),
        )
        result["source_rollout_total"] = len(index["source_rollouts"][example["id"]])
        if view == "rollouts":
            path = self.evidence_path(log_path, row)
            if path is None or not path.exists():
                return {**result, "frames": [], "actions": [], "movie_available": False,
                        "evidence_status": "expired" if path else "not_recorded"}
            evidence = json.loads(path.read_text())
            description = ("Both views rendered from recorded physics states." if evidence.get("qpos")
                           else "Both views reconstructed from recorded commands; not original camera footage.")
            return {**result, **evidence, "evidence_status": "available",
                    "movie_available": bool(evidence["actions"]), "playback_description": description}
        path = self.source_path(row["source"]) / "frames.jsonl"
        frames = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        return {**result, "frames": [{"step": i, "image": frame["image"]} for i, frame in enumerate(frames)],
                "actions": [frame["action"] for frame in frames], "evidence_status": "expert_only",
                "movie_available": bool(frames), "playback_description":
                    "Original recorded wrist images alongside an overview "
                    + ("rendered from recorded physics states." if frames and all("qpos" in frame for frame in frames)
                       else "reconstructed from recorded commands (not original overview footage).")}

    def movie(self, log_path: Path, view: str, item_id: str) -> Path:
        index = self.index(log_path)
        if view not in ("examples", "rollouts", "solved") or item_id not in index[view]:
            raise FileNotFoundError("Example or rollout not found")
        row = index[view][item_id]
        source = self.source_path(row["source"])
        evidence = self.evidence_path(log_path, row) if view == "rollouts" else None
        if view == "rollouts" and (evidence is None or not evidence.exists()):
            raise FileNotFoundError("This rollout has no retained recording to play")
        renderer = Path(__file__).with_name("simulation_playback.py")
        inputs = [source / "episode.json", source / "frames.jsonl", renderer,
                  Path(__file__).with_name("robot_sim.py")]
        if evidence:
            inputs.append(evidence)
        revision = [(str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in inputs]
        cache = self.workspace / "outputs" / ".data-browser-playback"
        if cache.is_symlink():
            raise ValueError("Playback cache must not be a symlink")
        with self.movie_lock:
            cache.mkdir(parents=True, exist_ok=True)
            target = cache / f"{identifier(json.dumps(revision))}.mp4"
            if target.exists():
                target.touch()
                return target
            temporary = target.with_suffix(".partial.mp4")
            command = [sys.executable, "-m", "so101.simulation_playback", "--source", str(source),
                       "--model-root", str(self.data_root / "model"), "--output", str(temporary)]
            if evidence:
                command.extend(("--evidence", str(evidence)))
            try:
                completed = subprocess.run(command, capture_output=True, text=True, timeout=180)
                if completed.returncode:
                    raise OSError(f"Cannot prepare playback: {completed.stderr[-3000:]}")
                temporary.replace(target)
            except subprocess.TimeoutExpired as error:
                raise OSError("Playback rendering exceeded its 180-second limit") from error
            finally:
                temporary.unlink(missing_ok=True)
            movies = sorted(cache.glob("*.mp4"), key=lambda path: path.stat().st_mtime_ns)
            for expired in movies[:-20]:
                expired.unlink()
            return target

    def media(self, log_path: Path, view: str, item_id: str, image: str) -> Path:
        index = self.index(log_path)
        if view not in ("examples", "rollouts", "solved") or item_id not in index[view]:
            raise FileNotFoundError("Example or rollout not found")
        row = index[view][item_id]
        if view == "rollouts":
            evidence = self.evidence_path(log_path, row)
            if evidence is None:
                raise FileNotFoundError("Historical policy images were not recorded")
            root = evidence.parent.resolve()
            frames = json.loads(evidence.read_text())["frames"]
            frame_index = int(image)
            if frame_index < 0:
                raise ValueError("Negative frame index")
            image_name = frames[frame_index]["image"]
        else:
            root = self.source_path(row["source"])
            if image == "overview":
                image_name = "diagnostic_overview.jpg"
            elif image == "initial" and (root / "initial_wrist.jpg").exists():
                image_name = "initial_wrist.jpg"
            else:
                frame_index = 0 if image == "initial" else int(image)
                if frame_index < 0:
                    raise ValueError("Negative frame index")
                with (root / "frames.jsonl").open() as stream:
                    for number, line in enumerate(stream):
                        if number == frame_index:
                            image_name = json.loads(line)["image"]
                            break
                    else:
                        raise FileNotFoundError("Frame not found")
        path = (root / image_name).resolve()
        if not path.is_relative_to(root) or path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            raise ValueError("Invalid image reference")
        return path
