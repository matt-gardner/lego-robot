"""Memory-aware idle-priority worker supervision, not CPU quotas or idle windows."""
from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading

from so101.cloud_checkpoint import atomic_json
from so101.produce_catalog import recipe


LOGGER = logging.getLogger(__name__)


def worker_slots(*, cpus: int, available_bytes: int, peak_bytes: int,
                 resident_bytes: list[int], pressure: bool) -> int:
    if cpus < 1 or peak_bytes < 1 or available_bytes < 0 or any(rss < 0 for rss in resident_bytes):
        raise ValueError("Invalid measured worker resource inputs")
    if pressure:
        return 0
    reserved_growth = sum(max(0, peak_bytes - rss) for rss in resident_bytes)
    return max(0, min(cpus - len(resident_bytes),
                      (available_bytes - reserved_growth) // peak_bytes))


def memory_state() -> dict:
    cgroup = Path("/sys/fs/cgroup")
    current = int((cgroup / "memory.current").read_text())
    limit_text = (cgroup / "memory.max").read_text().strip()
    statistics = dict(line.split() for line in (cgroup / "memory.stat").read_text().splitlines())
    host = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    host_available = int(host["MemAvailable"].split()[0]) * 1024
    inactive_file = int(statistics["inactive_file"])
    # Inactive file pages are reclaimable cache, not an archive-size memory estimate.
    remaining = host_available if limit_text == "max" else max(
        0, int(limit_text) - current + inactive_file
    )
    pressure = (cgroup / "memory.pressure").read_text() + Path("/proc/pressure/memory").read_text()
    pressured = any(float(field.split("=")[1]) > 0
                    for line in pressure.splitlines()
                    for field in line.split() if field.startswith("avg10="))
    return {"available_bytes": min(host_available, remaining),
            "memory_current": current, "memory_max": limit_text,
            "inactive_file_bytes": inactive_file, "pressure": pressured}


def process_state(pid: int) -> tuple[str, int, int] | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        status = Path(f"/proc/{pid}/status").read_text()
    except FileNotFoundError:
        return None
    resident = next((int(line.split()[1]) * 1024 for line in status.splitlines()
                     if line.startswith("VmRSS:")), 0)
    return fields[0], int(fields[19]), resident


def supervise(root: Path, catalog: Path, appearance: Path, part_ids: list[str],
              seed: int, initial_sequence: int, worker_peak_kib: int,
              stopping: threading.Event) -> None:
    if worker_peak_kib <= 0:
        raise ValueError("A measured matching-worker peak is required")
    ionice = shutil.which("ionice")
    if ionice is None:
        raise RuntimeError("Idle I/O scheduling requires ionice")
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = recipe(catalog, appearance, part_ids, seed)
        state_path = root / "supervisor.json"
        state = json.loads(state_path.read_bytes()) if state_path.exists() else {
            "recipe": identity, "workers": {}, "measured_worker_peak_kib": worker_peak_kib,
            "initial_sequence": initial_sequence,
        }
        if state["recipe"] != identity or state["initial_sequence"] != initial_sequence:
            raise ValueError("Supervisor recipe changed; use a new root")
        children = {}
        try:
            while not stopping.is_set():
                if recipe(catalog, appearance, part_ids, seed) != identity:
                    raise ValueError("Pinned supervisor inputs changed during execution")
                active = []
                for index, worker in state["workers"].items():
                    process = process_state(worker["pid"])
                    if process is None or process[0] == "Z" or process[1] != worker["start"]:
                        child = children.pop(index, None)
                        code = child.poll() if child is not None else None
                        expected_termination = (
                            worker["status"] == "stopping" and code == -signal.SIGTERM
                        )
                        if code not in (None, 0) and not expected_termination:
                            raise RuntimeError(f"Worker {index} exited {code}; inspect {worker['log']}")
                        worker["status"] = "exited"
                        worker["returncode"] = code
                        continue
                    active.append((index, worker, process[2]))
                    report_path = Path(worker["root"]) / "producer.json"
                    if report_path.exists():
                        report = json.loads(report_path.read_bytes())
                        for attempt in report["attempts"]:
                            state["measured_worker_peak_kib"] = max(
                                state["measured_worker_peak_kib"], attempt["peak_rss_kib"]
                            )
                            if (attempt.get("worker_source_sha256") == identity["source_sha256"]
                                    and attempt["status"] == "qualified"
                                    and attempt.get("object_count", 2) >= 8):
                                state["current_recipe_measured"] = True
                memory = memory_state()
                peak = state["measured_worker_peak_kib"] * 1024
                residents = [rss for _, _, rss in active]
                slots = worker_slots(cpus=len(os.sched_getaffinity(0)),
                                     available_bytes=memory["available_bytes"], peak_bytes=peak,
                                     resident_bytes=residents,
                                     pressure=memory["pressure"] and bool(active))
                reserved_growth = sum(max(0, peak - rss) for rss in residents)
                running = [row for row in active if row[1]["status"] != "stopping"]
                if ((memory["pressure"] and len(running) > 1)
                        or reserved_growth > memory["available_bytes"]):
                    candidates = running
                    if candidates:
                        _, worker, _ = candidates[-1]
                        os.kill(worker["pid"], signal.SIGTERM)
                        worker["status"] = "stopping"
                elif (slots and not any(worker["status"] == "stopping" for _, worker, _ in active)
                      and (not active or state.get("current_recipe_measured", False))):
                    active_indices = {int(index) for index, _, _ in active}
                    index = str(next(i for i in range(len(os.sched_getaffinity(0)))
                                     if i not in active_indices))
                    worker_root = root / "workers" / index
                    log_path = root / f"worker_{index}.log"
                    command = [
                        ionice, "-c", "3", sys.executable, "-c",
                        "import os;os.sched_setscheduler(0,os.SCHED_IDLE,os.sched_param(0));"
                        "os.nice(19);from so101.produce_catalog import main;main()",
                        "--root", str(worker_root), "--catalog", str(catalog),
                        "--appearance", str(appearance), "--seed", str(seed),
                        "--ledger", str(root / "ledger.sqlite"),
                        "--initial-sequence", str(initial_sequence),
                    ]
                    for part_id in part_ids:
                        command.extend(["--part-id", part_id])
                    with log_path.open("ab") as log:
                        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                    children[index] = child
                    process = process_state(child.pid)
                    if process is None:
                        raise RuntimeError(f"Worker startup failed; inspect {log_path}")
                    state["workers"][index] = {
                        "pid": child.pid, "start": process[1], "root": str(worker_root),
                        "log": str(log_path), "status": "running",
                    }
                state["memory"] = memory
                state["active_workers"] = len(active)
                atomic_json(state_path, state)
                # This is the resource controller's refresh interval, not an idle admission window.
                stopping.wait(5)
        finally:
            for worker in state["workers"].values():
                process = process_state(worker["pid"])
                if process is not None and process[0] != "Z" and process[1] == worker["start"]:
                    os.kill(worker["pid"], signal.SIGTERM)
            for child in children.values():
                child.wait()
            for worker in state["workers"].values():
                process = process_state(worker["pid"])
                worker["status"] = "exited" if process is None or process[0] == "Z" else "stopping"
            state["active_workers"] = sum(worker["status"] != "exited"
                                          for worker in state["workers"].values())
            state["shutdown_requested"] = True
            atomic_json(state_path, state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--appearance", type=Path, required=True)
    parser.add_argument("--part-id", action="append", default=[])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--initial-sequence", type=int, required=True)
    parser.add_argument("--worker-peak-kib", type=int, required=True)
    args = parser.parse_args()
    stopped = threading.Event()
    signal.signal(signal.SIGTERM, lambda *args: stopped.set())
    signal.signal(signal.SIGINT, lambda *args: stopped.set())
    logging.basicConfig(level=logging.INFO)
    supervise(args.root, args.catalog, args.appearance, args.part_id, args.seed,
              args.initial_sequence, args.worker_peak_kib, stopped)


if __name__ == "__main__":
    main()
