"""Serial CPU queue: cover admitted piles before expanding task image coverage."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def receipt_path(output, entry, task):
    source = str(Path(entry["root"]) / "tasks" / task["task_id"])
    return output / (hashlib.sha256(source.encode()).hexdigest() + ".json")


def discover(data, output):
    pools = read(data / "current-task-pools.json")["pools"]
    stages = data / "photo-task-progression-collective-v1"
    entries = []
    seen = set()
    for row in pools:
        manifest = Path(row["manifest_path"])
        if digest(manifest) != row["sha256"]:
            raise ValueError(f"Pool pin changed: {manifest}")
        pool = read(manifest)
        if pool["status"] != "complete":
            raise ValueError(f"Incomplete selected pool: {manifest}")
        for entry in pool["scenes"]:
            entries.append((entry, output / row["sha256"], row["sha256"], "selected"))
            seen.add(entry["scene_id"])
    for admission in sorted(stages.glob("pieces-*/tasks/scene_*/admission.json")):
        entry = read(admission)
        if entry["scene_id"] in seen:
            continue
        if entry["split"] != "train":
            raise ValueError(f"Non-TRAIN admission: {admission}")
        sha = digest(admission)
        destination = output.parent / "pile-previews-v1" / sha
        destination.mkdir(parents=True, exist_ok=True)
        pin = destination / "admission.json"
        if pin.exists() and digest(pin) != sha:
            raise ValueError(f"Admission pin changed: {pin}")
        if not pin.exists():
            pin.write_bytes(admission.read_bytes())
        entries.append((entry, destination, sha, "admitted-pile"))
        seen.add(entry["scene_id"])
    return entries


def next_job(entries):
    pending_tasks = []
    for entry, output, sha, scope in entries:
        problem = Path(entry["root"]) / "problem.json"
        if digest(problem) != entry["files_sha256"]["problem.json"]:
            raise ValueError(f"Problem pin changed: {problem}")
        tasks = read(problem)["tasks"]
        missing = []
        measured = False
        for task in tasks:
            path = receipt_path(output, entry, task)
            if path.exists():
                receipt = read(path)
                if receipt["pool_sha256"] != sha or digest(output / receipt["image_file"]) != receipt["sha256"]:
                    raise ValueError(f"Invalid existing receipt: {path}")
                measured = True
            else:
                missing.append(task)
        if not measured:
            return {"entry": entry, "output": str(output), "sha": sha,
                    "task": tasks[0], "phase": "piles", "scope": scope}
        if missing and scope == "selected":
            pending_tasks.append({"entry": entry, "output": str(output), "sha": sha,
                                  "task": missing[0], "phase": "tasks", "scope": scope})
    return pending_tasks[0] if pending_tasks else None


def render_job(job):
    import mujoco
    from so101.cloud_checkpoint import atomic_json
    from so101.photoreal_task_images import (
        DEFAULT_PROFILE, admit_render, create_simulation, integration_state,
        load_problem, render_reset_group, reset_problem,
    )
    import so101.photoreal_task_images as producer

    entry, task, output = job["entry"], job["task"], Path(job["output"])
    admit_render(output)
    meta = output / "_meta"
    meta.mkdir(parents=True, exist_ok=True)
    contract_path = meta / "contract.json"
    implementation = {
        name: digest(Path(producer.__file__).with_name(name))
        for name in ("photoreal_task_images.py", "task_image_roles.py", "pathtrace_pile.py",
                     "pile_task_runtime.py", "lego_scene.py", "capture_pile_task_images.py")
    }
    contract = {"pool_sha256": job["sha"], "profile_sha256": digest(DEFAULT_PROFILE),
                "implementation_sha256": implementation, "schema_version": 1}
    if contract_path.exists() and read(contract_path) != contract:
        raise ValueError("Existing render contract changed")
    atomic_json(contract_path, contract)
    sim = create_simulation(load_problem(entry, task["task_id"]))
    try:
        reset_problem(sim, load_problem(entry, task["task_id"]))
        state_sha = hashlib.sha256(integration_state(sim).tobytes()).hexdigest()
        state_dir = output / "_scenes" / entry["scene_id"] / state_sha
        if state_dir.exists():
            preserved = state_dir.with_name(state_sha + f".interrupted-{time.time_ns()}")
            state_dir.rename(preserved)
            print(f"Preserved unpublished partial state: {preserved}", flush=True)
        profile = read(DEFAULT_PROFILE)
        sim.model.vis.global_.offwidth = max(sim.model.vis.global_.offwidth, profile["width"])
        sim.model.vis.global_.offheight = max(sim.model.vis.global_.offheight, profile["height"])
        sim.renderer = mujoco.Renderer(sim.model, height=profile["height"], width=profile["width"])
        render_reset_group(sim, entry, [task], state_sha, output, DEFAULT_PROFILE, contract)
    finally:
        sim.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--job", type=Path)
    args = parser.parse_args()
    if args.job:
        render_job(read(args.job))
        return
    from so101.cloud_checkpoint import atomic_json

    queue = args.root / "_queue"
    lock = (queue / "worker.lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.nice(10)
    while True:
        entries = discover(args.data, args.root)
        job = next_job(entries)
        summary = {
            "pid": os.getpid(), "phase": job["phase"] if job else "awaiting_new_piles",
            "priority": "all-piles-before-tasks", "discovered_piles": len(entries),
            "covered_piles": sum(any(receipt_path(output, entry, task).exists()
                                    for task in read(Path(entry["root"]) / "problem.json")["tasks"])
                                 for entry, output, _, _ in entries),
        }
        atomic_json(queue / "status.json", summary)
        if job is None:
            return
        atomic_json(queue / "active-job.json", job)
        try:
            subprocess.run([sys.executable, __file__, "--job", str(queue / "active-job.json")],
                           check=True)
        except subprocess.CalledProcessError as error:
            atomic_json(queue / "status.json", {**summary, "phase": "failed", "error": str(error)})
            raise


if __name__ == "__main__":
    main()
