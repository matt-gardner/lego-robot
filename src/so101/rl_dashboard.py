"""Read-only, live dashboard for RL and supervised training logs."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import logging
import math
import re
import shlex
import sqlite3
import subprocess
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from statistics import mean
from so101.jerk_reference import ReferenceJerkMargins
from typing import Any
from urllib.parse import urlsplit, parse_qs

from so101.training_data_browser import TrainingDataBrowser
from so101.lego_data_browser import LegoDataBrowser


ASSETS = Path(__file__).with_name("dashboard")
LOG_NAMES = ("trajectory_rl.json", "rl_training.json", "sft_training.json")
LOGGER = logging.getLogger(__name__)


def active_run_outputs() -> set[Path]:
    processes = subprocess.run(
        ["ps", "-ax", "-o", "args="],
        capture_output=True, text=True, check=True, timeout=5,
    )
    outputs = set()
    modules = {"so101.train_trajectory_rl", "so101.train_entity_rl", "so101.train_entity_sim"}
    scripts = {"train_trajectory_rl.py", "train_entity_rl.py", "train_entity_sim.py"}
    trainers = modules | scripts
    for line in processes.stdout.splitlines():
        if not any(name in line for name in trainers):
            continue
        try:
            arguments = shlex.split(line)
        except ValueError:
            LOGGER.warning("Cannot parse trainer process command")
            continue
        if not arguments or not Path(arguments[0]).name.lower().startswith("python"):
            continue
        # Only Python's actual execution target counts, not a wrapper's arguments.
        script = 1
        while script < len(arguments) and arguments[script].startswith("-"):
            option = arguments[script]
            if option in {"-m", "-c", "--"}:
                break
            script += 2 if option in {"-W", "-X"} else 1
        module = script < len(arguments) and arguments[script] == "-m"
        if module:
            script += 1
        elif script < len(arguments) and arguments[script] == "--":
            script += 1
        if script >= len(arguments) or (
            arguments[script] not in modules if module else Path(arguments[script]).name not in scripts
        ):
            continue
        options = arguments[script + 1:]
        for i, option in enumerate(options):
            if option == "--output" and i + 1 < len(options):
                outputs.add(Path(options[i + 1]).resolve())
                break
            if option.startswith("--output="):
                outputs.add(Path(option.partition("=")[2]).resolve())
                break
    return outputs


def average(values: list[Any]) -> float | None:
    finite = [
        float(value) for value in values
        if isinstance(value, (int, float)) and math.isfinite(value)
    ]
    return mean(finite) if finite else None


def trajectory_metrics(rows: list[dict]) -> dict:
    successful = [row for row in rows if row.get("success")]
    return {
        "samples": len(rows),
        "success_rate": average([row.get("success") for row in rows]),
        "wrong_object_rate": average([row.get("wrong_object") for row in rows]),
        "completion_efficiency": average([
            row.get("completion_efficiency") for row in successful
        ]),
        "rms_jerk": average([row.get("rms_jerk") for row in rows]),
        "reward": average([row.get("reward") for row in rows]),
        "smoothness": average([row.get("smoothness") for row in successful]),
    }


def normalize_log(data: dict, jerk_references: ReferenceJerkMargins | None = None) -> dict:
    """Keep scalar plot data only; never ship per-frame or parameter data."""
    series: dict[tuple[str, str], list[dict]] = defaultdict(list)
    steps = data.get("steps", [])
    sft = data.get("kind") == "sft"
    trajectory = not sft and "validation" in data
    axis = "Optimizer step"
    if sft:
        for row in steps:
            series[("train", "all")].append({
                "step": row["step"],
                **{key: row.get(key) for key in (
                    "loss", "action_loss", "box_loss",
                    "learning_rate", "gradient_norm",
                    "step_seconds", "completed_at", "eta_seconds",
                )},
            })
    elif trajectory:
        for row in steps:
            groups = row.get("groups", [])
            by_task: dict[str, list[dict]] = defaultdict(list)
            for group in groups:
                by_task[group["task"]].append(group)
            for task, task_groups in [("all", groups), *by_task.items()]:
                trajectories = [
                    item for group in task_groups
                    for item in group.get("trajectories", [])
                ]
                point = {
                    "step": row["step"],
                    **trajectory_metrics(trajectories),
                    "discard_rate": average([
                        group.get("discarded") for group in task_groups
                    ]),
                    "assistance": average([
                        group.get("start_proximity") for group in task_groups
                    ]),
                }
                jerk_gaps = []
                for group in task_groups:
                    margin = (group.get("jerk_margin")
                              if group.get("jerk_reference_scope") == "problem" else None)
                    if jerk_references is not None and group.get("source") is not None:
                        assistance = group.get("start_proximity")
                        if isinstance(assistance, (int, float)) and math.isfinite(assistance):
                            margin = jerk_references.summary(group["source"], assistance)["margin"]
                    if not isinstance(margin, (int, float)) or not math.isfinite(margin) or margin < 0:
                        continue
                    jerk_gaps.extend(
                        item["rms_jerk"] - margin
                        for item in group.get("trajectories", [])
                        if item.get("steps", 0) >= 4
                        and isinstance(item.get("rms_jerk"), (int, float))
                        and math.isfinite(item["rms_jerk"])
                    )
                point["jerk_margin_gap"] = average(jerk_gaps)
                point["jerk_margin_samples"] = len(jerk_gaps)
                if task == "all":
                    for key in (
                        "learning_rate", "gradient_norm", "sft_loss",
                        "step_seconds", "completed_at", "eta_seconds", "sft_coefficient",
                    ):
                        point[key] = row.get(key)
                series[("train", task)].append(point)
    else:
        episode_rows = data.get("episode_metrics", [])
        if not episode_rows:
            # Old logs have decision-level outcomes, not full-episode outcomes.
            axis = "Episode index (decision-level metrics)"
            episode_rows = data.get("rollouts", [])
        grouped: dict[tuple[int, str], list[dict]] = defaultdict(list)
        for row in episode_rows:
            step = row["training_step"] if "training_step" in row else row["episode"]
            grouped[(step, row["task"])].append(row)
            grouped[(step, "all")].append(row)
        for (step, task), rows in sorted(grouped.items()):
            point = {
                "step": step,
                "samples": len(rows),
                "success_rate": average([row.get("success") for row in rows]),
                "wrong_object_rate": average([
                    row.get("wrong_object") for row in rows
                ]),
                "completion_efficiency": average([
                    row["completion_efficiency_percent"] / 100
                    for row in rows
                    if row.get("success")
                    and row.get("completion_efficiency_percent") is not None
                ]),
                "rms_jerk": average([
                    row.get("rms_arm_jerk_degrees_per_frame3")
                    for row in rows
                ]),
            }
            if task == "all":
                point["learning_rate"] = data.get("learning_rate")
            series[("train", task)].append(point)
        for row in steps:
            for point in series.get(("train", "all"), []):
                if point["step"] == row["training_step"]:
                    point.update({
                        key: row.get(key)
                        for key in (
                            "gradient_norm", "parameter_rms_drift",
                            "step_seconds", "completed_at",
                        )
                    })
    if trajectory or sft:
        for row in data.get("validation", []):
            for task, metrics in [("all", row), *row.get("by_task", {}).items()]:
                point = {
                    "step": row["step"],
                    **{key: metrics.get(key) for key in (
                        "success_rate", "completion_efficiency", "rms_jerk", "wrong_object_rate",
                    )},
                    "samples": metrics.get("episodes"),
                }
                if metrics.get("success_rate") == 0:
                    point["completion_efficiency"] = None
                series[("eval", task)].append(point)
    normalized_series = [
        {"phase": phase, "task": task, "points": sorted(points, key=lambda p: p["step"])}
        for (phase, task), points in series.items()
    ]
    for item in normalized_series:
        for point in item["points"]:
            for key, value in point.items():
                if isinstance(value, float) and not math.isfinite(value):
                    point[key] = None
    training = series.get(("train", "all"), [])
    validation = series.get(("eval", "all"), [])
    return {
        "kind": "sft" if sft else "rl",
        "axis": axis,
        "series": normalized_series,
        "latest_train": training[-1] if training else None,
        "latest_eval": validation[-1] if validation else None,
        "best_eval": max(
            (row for row in validation if row.get("success_rate") is not None),
            key=lambda row: row["success_rate"],
            default=None,
        ),
        "config": {
            key: data[key] for key in (
                "algorithm", "group_size", "groups_per_task", "sft_coefficient",
                "advantage_weights", "branched_from", "learning_rate",
                "run_config", "policy_optimization", "policy_optimization_changes",
            ) if key in data
        },
        "curriculum": data.get("assistance_curriculum"),
    }


class RunStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.cache: dict[str, dict] = {}
        self.reference_cache: dict[str, tuple[str, ReferenceJerkMargins]] = {}
        self.paths: dict[str, Path] = {}
        self.lock = threading.RLock()

    def discover(self) -> dict[str, Path]:
        paths = {
            path for name in (*LOG_NAMES, "attempt*.json")
            for path in self.root.rglob(name)
            if path.resolve().is_relative_to(self.root)
        }
        self.paths = {
            hashlib.sha256(str(path.relative_to(self.root)).encode()).hexdigest()[:16]: path
            for path in paths
        }
        return self.paths

    def path_for(self, run_id: str) -> Path | None:
        with self.lock:
            path = self.paths.get(run_id)
            if path is None:
                path = self.discover().get(run_id)
            if path is not None and path.is_file() and path.resolve().is_relative_to(self.root):
                return path
            return None

    def load(self, run_id: str, path: Path) -> dict:
        cached = self.cache.get(run_id)
        try:
            stat = path.stat()
            revision = f"{stat.st_mtime_ns}:{stat.st_size}"
            if cached and cached["revision"] == revision and not cached.get("error"):
                return cached
            data = json.loads(path.read_text())
            if not isinstance(data, dict) or not (
                data.get("kind") == "sft" or any(
                    key in data for key in ("validation", "rollouts", "episode_metrics")
                )
            ):
                raise ValueError("File does not contain a supported training log")
            jerk_references = None
            split_path = path.with_name("data_split.json")
            if data.get("kind") != "sft" and "validation" in data and split_path.exists():
                split_revision = f"{split_path.stat().st_mtime_ns}:{data.get('run_config', {}).get('horizon', 300)}"
                entry = self.reference_cache.get(run_id)
                if entry is None or entry[0] != split_revision:
                    split = json.loads(split_path.read_text())
                    jerk_references = ReferenceJerkMargins(
                        split["training"], data.get("run_config", {}).get("horizon", 300),
                        10, hold_final=False, workspace=self.root.parent,
                    )
                    self.reference_cache[run_id] = (split_revision, jerk_references)
                else:
                    jerk_references = entry[1]
            result = {
                "id": run_id,
                "name": str(path.parent.relative_to(self.root)),
                "path": str(path.relative_to(self.root)),
                "archived": path.name not in LOG_NAMES,
                "kind": "sft" if path.name == "sft_training.json" else "rl",
                "updated_at": datetime.fromtimestamp(
                    stat.st_mtime, timezone.utc
                ).isoformat(),
                "revision": revision,
                "error": None,
                **normalize_log(data, jerk_references),
            }
            latest = result["latest_train"]
            completed_at = latest.get("completed_at") if latest else None
            has_timestamp = (
                isinstance(completed_at, (int, float))
                and math.isfinite(completed_at) and completed_at > 0
            )
            previous = (cached or {}).get("latest_train")
            if has_timestamp:
                last_step_at = completed_at
            elif latest and previous and latest["step"] == previous["step"]:
                last_step_at = cached["last_step_at"]
            else:
                last_step_at = stat.st_mtime if latest else None
            result["last_step_at"] = last_step_at
            result["last_step_time_estimated"] = bool(latest) and not has_timestamp
            if result["archived"]:
                result["name"] += f" / {path.stem}"
            self.cache[run_id] = result
            return result
        except (OSError, ValueError, KeyError, TypeError) as error:
            LOGGER.warning("Cannot read %s: %s", path, error)
            # Writers currently replace JSON in-place. Preserve the last complete
            # snapshot, but explicitly mark it stale until the write finishes.
            return {
                **(cached or {
                    "id": run_id,
                    "name": str(path.relative_to(self.root)),
                    "path": str(path.relative_to(self.root)),
                    "archived": path.name not in LOG_NAMES,
                    "revision": None,
                    "series": [],
                }),
                "error": f"Cannot read latest log: {error}",
            }

    def update_activity(self, runs: list[dict]) -> None:
        activity_error = None
        try:
            outputs = active_run_outputs()
        except (OSError, subprocess.SubprocessError) as error:
            LOGGER.warning("Cannot check trainer activity: %s", error)
            outputs = set()
            activity_error = f"Cannot check trainer activity: {error}"
        for run in runs:
            status_path = (self.root / run["path"]).parent / "remote_status.json"
            if not run["archived"] and status_path.exists():
                try:
                    remote = json.loads(status_path.read_text())
                    now = time.time()
                    observed = remote.get("observed_at")
                    heartbeat = remote.get("heartbeat")
                    def recent(value):
                        return (
                            isinstance(value, (int, float))
                            and math.isfinite(value) and -30 <= now - value <= 90
                        )
                    fresh = recent(observed) and recent(heartbeat)
                    known = recent(observed) and (
                        recent(heartbeat) or remote["state"] not in ("running", "starting", "checkpointing")
                    )
                    run["remote"] = {
                        **remote,
                        "state": remote["state"] if known else "stale",
                        "contact_age_seconds": max(0, now - observed)
                        if isinstance(observed, (int, float)) and math.isfinite(observed) else None,
                    }
                    run["active"] = fresh and remote["state"] == "running"
                    run["activity_error"] = remote.get("error")
                    if remote.get("backup_error"):
                        run["activity_error"] = (
                            (run["activity_error"] + "; " if run["activity_error"] else "")
                            + "Checkpoint backup: " + remote["backup_error"]["error"]
                        )
                except (OSError, ValueError, KeyError, TypeError) as error:
                    run["active"] = False
                    run["remote"] = {"state": "error", "contact_age_seconds": None}
                    run["activity_error"] = f"Cannot read remote status: {error}"
                continue
            run.pop("remote", None)
            run["active"] = (
                None if activity_error else not run["archived"]
                and (self.root / run["path"]).parent.resolve() in outputs
            )
            run["activity_error"] = activity_error

    def catalog(self) -> list[dict]:
        with self.lock:
            runs = [
                self.load(run_id, path)
                for run_id, path in self.discover().items()
            ]
            self.update_activity(runs)
            return sorted(
                [
                    {key: value for key, value in run.items() if key != "series"}
                    for run in runs
                ],
                key=lambda run: run.get("updated_at", ""),
                reverse=True,
            )

    def get(self, run_id: str) -> dict | None:
        with self.lock:
            path = self.discover().get(run_id)
            if path is None:
                return None
            run = self.load(run_id, path)
            self.update_activity([run])
            return run


def make_server(root: Path, host: str, port: int, base_path: str = "",
                notice: str = "", lego_catalog_root: Path | None = None,
                lego_episode_root: Path | None = None) -> ThreadingHTTPServer:
    base_path = base_path.rstrip("/")
    if not re.fullmatch(r"(?:/[A-Za-z0-9_-]+)*", base_path):
        raise ValueError("Base path must contain only slash-separated letters, digits, _ or -")
    store = RunStore(root)
    browser = TrainingDataBrowser(root.resolve().parent)
    lego_browser = LegoDataBrowser(root.resolve().parent, lego_catalog_root, lego_episode_root)
    assets = {
        "/": ("index.html", "text/html; charset=utf-8"),
        "/dashboard.js": ("dashboard.js", "text/javascript; charset=utf-8"),
        "/dashboard.css": ("dashboard.css", "text/css; charset=utf-8"),
        "/data": ("data.html", "text/html; charset=utf-8"),
        "/data.js": ("data.js", "text/javascript; charset=utf-8"),
        "/data.css": ("data.css", "text/css; charset=utf-8"),
        "/lego": ("lego.html", "text/html; charset=utf-8"),
        "/examples": ("lego.html", "text/html; charset=utf-8"),
        "/lego.js": ("lego.js", "text/javascript; charset=utf-8"),
        "/lego.css": ("lego.css", "text/css; charset=utf-8"),
        "/part-viewer.js": ("part-viewer.js", "text/javascript; charset=utf-8"),
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            if base_path:
                if path == base_path:
                    self.send_body(308, b"", "text/plain", {"Location": base_path + "/"})
                    return
                if not path.startswith(base_path + "/"):
                    self.respond(404, {"error": "Not found"})
                    return
                path = path[len(base_path):]
            if path == "/api/lego":
                query = {key: values[0] for key, values in parse_qs(urlsplit(self.path).query).items()}
                try:
                    with lego_browser.lock:
                        if "frame" in query:
                            body, content_type = lego_browser.media(query)
                            self.send_body(200, body, content_type)
                        elif query.get("preview") == "image":
                            self.send_body(200, lego_browser.preview(query), "image/svg+xml")
                        elif query.get("preview") == "geometry":
                            self.respond(200, lego_browser.geometry(query))
                        else:
                            self.respond(200, lego_browser.list(query))
                except FileNotFoundError as error:
                    self.respond(404, {"error": str(error)})
                except (ValueError, KeyError, TypeError) as error:
                    LOGGER.warning("Invalid LEGO catalog request or metadata: %s", error)
                    self.respond(400, {"error": str(error)})
                except (OSError, sqlite3.Error) as error:
                    LOGGER.exception("Cannot read LEGO catalog")
                    self.respond(500, {"error": str(error)})
            elif path.startswith("/api/data/"):
                query = {key: values[0] for key, values in parse_qs(urlsplit(self.path).query).items()}
                log_path = store.path_for(path.removeprefix("/api/data/"))
                if log_path is None:
                    self.respond(404, {"error": "Run not found"})
                    return
                try:
                    view = query.get("view", "examples")
                    if "movie" in query:
                        if query["movie"] == "prepare":
                            status = browser.prepare_movie(log_path, view, query.get("item", ""))
                            self.respond(200 if status["ready"] else 202, status)
                        else:
                            movie = browser.movie(log_path, view, query.get("item", ""))
                            self.send_movie(movie)
                    elif "image" in query:
                        image = browser.media(log_path, view, query.get("item", ""), query["image"])
                        content_type = "image/png" if image.suffix.lower() == ".png" else "image/jpeg"
                        self.send_body(200, image.read_bytes(), content_type)
                    elif "item" in query:
                        self.respond(200, browser.detail(log_path, view, query["item"]))
                    else:
                        self.respond(200, browser.list(log_path, query))
                except (FileNotFoundError, IndexError):
                    self.respond(404, {"error": "Data or recorded frame not found"})
                except (ValueError, KeyError, TypeError) as error:
                    LOGGER.warning("Invalid training data request or metadata: %s", error)
                    self.respond(400, {"error": f"Invalid training data request or metadata: {error}"})
                except OSError as error:
                    LOGGER.exception("Cannot read training data")
                    self.respond(500, {"error": f"Cannot read training data: {error}"})
            elif path == "/api/runs":
                self.respond(200, {
                    "runs": store.catalog(),
                    "server_time": datetime.now(timezone.utc).isoformat(),
                })
            elif path.startswith("/api/runs/"):
                run = store.get(path.removeprefix("/api/runs/"))
                self.respond(200 if run is not None else 404, run or {"error": "Run not found"})
            elif path in assets:
                filename, content_type = assets[path]
                body = (ASSETS / filename).read_bytes()
                if filename.endswith(".html"):
                    page = body.decode().replace(
                        'data-base-path=""', f'data-base-path="{base_path}"',
                    )
                    if path == "/examples":
                        page = page.replace('data-browser-mode="parts"', 'data-browser-mode="examples"')
                        page = page.replace("<h1>LEGO parts</h1>", "<h1>Recorded examples</h1>")
                    # Family authentication lives at the origin root, not the dashboard mount.
                    page = re.sub(r'href="/(?!family(?:/|"))', f'href="{base_path}/', page)
                    page = page.replace('src="/', f'src="{base_path}/')
                    if notice:
                        page = page.replace(
                            "<body>",
                            f'<body><p class="context" role="status">{html.escape(notice)}</p>',
                        )
                    body = page.encode()
                self.send_body(200, body, content_type)
            else:
                self.respond(404, {"error": "Not found"})

        def respond(self, status: int, value: Any) -> None:
            self.send_body(
                status,
                json.dumps(value, allow_nan=False).encode(),
                "application/json",
            )

        def send_movie(self, path: Path) -> None:
            size = path.stat().st_size
            start, end = 0, size - 1
            requested = self.headers.get("Range")
            if requested:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested)
                if not match or not any(match.groups()):
                    self.send_body(416, b"", "video/mp4", {"Content-Range": f"bytes */{size}"})
                    return
                first, last = match.groups()
                if first:
                    start = int(first)
                    end = min(int(last), size - 1) if last else size - 1
                else:
                    start = max(0, size - int(last))
                if not 0 <= start <= end < size:
                    self.send_body(416, b"", "video/mp4", {"Content-Range": f"bytes */{size}"})
                    return
            with path.open("rb") as stream:
                stream.seek(start)
                body = stream.read(end - start + 1)
            headers = {"Accept-Ranges": "bytes"}
            if requested:
                headers["Content-Range"] = f"bytes {start}-{end}/{size}"
            self.send_body(206 if requested else 200, body, "video/mp4", headers)

        def send_body(self, status: int, body: bytes, content_type: str,
                      headers: dict[str, str] | None = None) -> None:
            try:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                for name, value in (headers or {}).items():
                    self.send_header(name, value)
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'self'; script-src 'self'; style-src 'self'; "
                    "connect-src 'self'; img-src 'self'; media-src 'self'; frame-ancestors 'none'",
                )
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                LOGGER.debug("Client disconnected while receiving %s", self.path)

        def log_message(self, format: str, *args: Any) -> None:
            LOGGER.debug(format, *args)

    return ThreadingHTTPServer((host, port), Handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--base-path", default="", help="Reverse-proxy prefix, e.g. /lego-robot")
    parser.add_argument("--notice", default="", help="Deployment status shown on both pages")
    parser.add_argument("--lego-catalog-root", type=Path,
                        help="Read-only directory containing versioned LEGO catalog releases")
    parser.add_argument("--lego-episode-root", type=Path,
                        help="Read-only directory containing immutable episode SQLite releases")
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error(f"Run directory does not exist: {args.root}")
    logging.basicConfig(level=logging.INFO)
    try:
        server = make_server(args.root, args.host, args.port, args.base_path, args.notice,
                             args.lego_catalog_root, args.lego_episode_root)
    except ValueError as error:
        parser.error(str(error))
    print(f"RL dashboard: http://{args.host}:{server.server_port}{args.base_path.rstrip('/')}/",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
