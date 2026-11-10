"""Training-only assistance and problem-difficulty state, with durable data requests."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

from so101.cloud_checkpoint import atomic_json


class DifficultyController:
    def __init__(self, state: dict | None = None):
        self.state = state if state is not None else {
            "schema": 1, "slices": {}, "events": [], "admissions": {},
        }
        if self.state.get("schema") != 1:
            raise ValueError("Unsupported difficulty curriculum state")

    def slice(self, key: str, maximum: float) -> dict:
        return self.state["slices"].setdefault(key, {
            "level": 0, "assistance": maximum, "window": [], "pending": None,
            "request_attempts": {},
        })

    def admit(self, key: str, level: int, sha256: str) -> None:
        if level <= 0 or len(sha256) != 64:
            raise ValueError("Difficulty admission requires a positive tier and immutable hash")
        identity = f"{key}:{level}"
        previous = self.state["admissions"].get(identity)
        if previous is not None and previous != sha256:
            raise ValueError(f"An admitted difficulty pool changed: {identity}")
        self.state["admissions"][identity] = sha256

    def record(self, key: str, successes: int, total: int, maximum: float,
               decrement: float, step: int) -> dict | None:
        if (total <= 0 or not 0 <= successes <= total
                or not math.isfinite(maximum) or not 0 <= maximum <= 1
                or not math.isfinite(decrement) or not 0 < decrement <= 1):
            raise ValueError("Invalid training difficulty observation")
        item = self.slice(key, maximum)
        window = item["window"]
        window.append({"successes": successes, "total": total})
        del window[:-4]
        if len(window) < 4:
            return None
        rate = sum(row["successes"] for row in window) / sum(row["total"] for row in window)
        previous = {"level": item["level"], "assistance": item["assistance"]}
        action = None
        next_level = item["level"] + 1
        if rate > .6:
            if item["assistance"] > 0:
                item["assistance"] = round(max(0., item["assistance"] - decrement), 6)
                action = "reduce-assistance"
            elif f"{key}:{next_level}" in self.state["admissions"]:
                item.update(level=next_level, pending=None)
                action = "advance"
            elif next_level > 62 and item["pending"] is None:
                item["pending"] = -1
                action = "generator-bounds-exhausted"
            elif item["pending"] is None:
                item["pending"] = next_level
                attempts = item.setdefault("request_attempts", {})
                attempts[str(next_level)] = attempts.get(str(next_level), 0) + 1
                action = "request"
        elif rate < .4:
            if item["assistance"] < maximum:
                item["assistance"] = round(min(maximum, item["assistance"] + decrement), 6)
                action = "increase-assistance"
            elif item["level"] > 0:
                item.update(level=item["level"] - 1, assistance=0., pending=None)
                action = "backoff"
        if action is None:
            return None
        event = {
            "step": step, "slice": key, "action": action, "success_rate": rate,
            "successes": sum(row["successes"] for row in window),
            "total": sum(row["total"] for row in window),
            "previous": previous, "level": item["level"], "assistance": item["assistance"],
        }
        item["window"] = []
        self.state["events"].append(event)
        return event

    def failed_request(self, result: dict) -> None:
        item = self.state["slices"][result["slice"]]
        if (item.get("pending_request_id") != result["request_id"]
                or item["pending"] != result["level"]):
            return
        attempts = item["request_attempts"][str(result["level"])]
        item["pending"] = None if attempts < 3 else -1
        item["pending_request_id"] = None
        self.state["events"].append({
            "slice": result["slice"], "action": "generation-failed",
            "request_id": result["request_id"], "attempt": attempts,
            "retry": attempts < 3, "error": result.get("error", result["status"]),
        })


class HarderDataRequests:
    def __init__(self, root: Path, catalog: Path, command: list[str] | None = None,
                 timeout: float = 3600.):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Harder-data request timeout must be positive")
        self.root, self.command, self.timeout = root, command, timeout
        self.catalog = catalog.resolve()
        self.children: dict[str, subprocess.Popen] = {}
        self.root.mkdir(parents=True, exist_ok=True)

    def request(self, key: str, level: int, task: str, seed: int,
                excluded_families: list[str], tags: dict) -> Path:
        payload = {
            "schema_version": 1, "kind": "pile-task-generation", "split": "train",
            "task_types": ["view", "grasp", "push", "place"], "seed": seed, "max_scenes": 8,
            "catalog": str(self.catalog),
            "tag_constraints": tags,
            "excluded_scene_families": sorted(set(excluded_families)),
        }
        identity = hashlib.sha256(json.dumps(
            {"slice": key, "level": level, **payload}, sort_keys=True,
        ).encode()).hexdigest()[:24]
        directory = self.root / identity
        path = directory / "request.json"
        if path.exists():
            return path
        directory.mkdir(exist_ok=True)
        payload.update(request_id=identity, output_root=str((directory / "pool").resolve()))
        atomic_json(path, payload)
        atomic_json(directory / "dispatch.json", {
            "created": time.time(), "slice": key, "level": level,
            "status": "queued" if self.command else "blocked",
            "error": None if self.command else "No harder-data worker command configured",
        })
        return path

    def results(self) -> list[dict]:
        results = []
        for path in sorted(self.root.glob("*/request.json")):
            request = json.loads(path.read_text())
            dispatch = json.loads(path.with_name("dispatch.json").read_text())
            result_path = path.with_name(f"{request['request_id']}.result.json")
            if result_path.exists():
                result = json.loads(result_path.read_text())
                if (result.get("request_id") != request["request_id"]
                        or result.get("request_sha256") != hashlib.sha256(path.read_bytes()).hexdigest()
                        or result.get("status") not in ("complete", "blocked", "rejected")):
                    raise ValueError(f"Invalid harder-data result: {result_path}")
                if result["status"] == "complete":
                    manifest = Path(result["manifest_path"])
                    if not manifest.resolve().is_relative_to(Path(request["output_root"]).resolve()):
                        raise ValueError(f"Harder-data manifest escaped its requested output root: {manifest}")
                    if hashlib.sha256(manifest.read_bytes()).hexdigest() != result["sha256"]:
                        raise ValueError(f"Harder-data manifest checksum mismatch: {manifest}")
                results.append({**dispatch, **result, "request_path": str(path)})
            else:
                child = self.children.get(request["request_id"])
                if child is not None and child.poll() is not None and not result_path.exists():
                    raise RuntimeError(f"Harder-data worker exited {child.returncode} without result: {path}")
                if dispatch["status"] not in ("blocked", "queued") and time.time() - dispatch["created"] > self.timeout:
                    if child is not None and child.poll() is None:
                        child.terminate()
                    raise TimeoutError(f"Harder-data request exceeded its bound: {path}")
                results.append({**dispatch, "request_id": request["request_id"], "request_path": str(path)})
        if self.command and not any(row["status"] in ("running", "launching") for row in results):
            queued = next((row for row in results if row["status"] == "queued"), None)
            if queued is not None:
                path = Path(queued["request_path"])
                queued.update(status="launching", created=time.time())
                atomic_json(path.with_name("dispatch.json"), queued)
                with path.with_name("worker.log").open("ab") as log:
                    child = subprocess.Popen(
                        [*self.command, "--request", str(path.resolve())],
                        stdout=log, stderr=subprocess.STDOUT,
                    )
                self.children[queued["request_id"]] = child
                queued.update(status="running", pid=child.pid)
                atomic_json(path.with_name("dispatch.json"), queued)
        return results

    def close(self):
        for identity, child in self.children.items():
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                path = self.root / identity / "request.json"
                result_path = path.with_name(f"{identity}.result.json")
                if not result_path.exists():
                    atomic_json(result_path, {
                        "request_id": identity, "status": "blocked",
                        "request_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "error": "Owning RL process stopped before generation completed",
                    })
