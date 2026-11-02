"""Single Azure GPU worker, eviction watcher, backups and independent lease guard."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
from email.utils import formatdate
import json
import logging
import os
from pathlib import Path
import subprocess
import threading
import time
from urllib.parse import quote, urlencode
from urllib.request import ProxyHandler, Request, build_opener, urlopen

from so101.cloud_checkpoint import atomic_json, recover, sha256, validate_state, publish


LOGGER = logging.getLogger(__name__)
IMDS = "http://169.254.169.254/metadata/"


def metadata(path: str) -> dict:
    request = Request(IMDS + path, headers={"Metadata": "true"})
    with build_opener(ProxyHandler({})).open(request, timeout=5) as response:
        return json.load(response)


def access_token(resource: str) -> str:
    query = urlencode({"api-version": "2018-02-01", "resource": resource})
    return metadata("identity/oauth2/token?" + query)["access_token"]


def lease() -> tuple[float, dict]:
    compute = metadata("instance?api-version=2021-02-01")["compute"]
    tags = {row["name"]: row["value"] for row in compute.get("tagsList", [])}
    deadline = float(tags["so101_deadline"])
    if deadline <= 0:
        raise ValueError("Invalid compute lease deadline")
    return deadline, compute


def deallocate(compute: dict) -> None:
    resource = (
        f"/subscriptions/{compute['subscriptionId']}/resourceGroups/"
        f"{compute['resourceGroupName']}/providers/Microsoft.Compute/"
        f"virtualMachines/{compute['name']}"
    )
    request = Request(
        "https://management.azure.com" + resource + "/deallocate?api-version=2024-07-01",
        data=b"", method="POST",
        headers={"Authorization": "Bearer " + access_token("https://management.azure.com/")},
    )
    with urlopen(request, timeout=30) as response:
        if response.status not in (200, 202):
            raise RuntimeError(f"VM deallocate returned HTTP {response.status}")


def guard(control: Path) -> None:
    deadline, compute = lease()
    if time.time() >= deadline:
        config = json.loads((control / "worker.json").read_text())
        request_stop(control, Path(config["root"]) / config["output"], "budget")
        deallocate(compute)


@contextmanager
def intent_lock(control: Path):
    with (control / "intent.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def request_stop(control: Path, output: Path, reason: str) -> None:
    with intent_lock(control):
        intent = json.loads((control / "intent.json").read_text())
        stop = output / "STOP"
        if not stop.exists():
            marker = f"cloud:{reason}\n"
            stop.write_text(marker)
            intent["stop_marker"] = marker
        # An eviction notice must never undo a user pause or expired budget.
        if reason == "eviction" and intent["desired"] == "paused":
            atomic_json(control / "intent.json", intent)
            return
        intent.update(desired="paused" if reason in ("pause", "budget") else "running",
                      stop_reason=reason)
        atomic_json(control / "intent.json", intent)


def acknowledge_stop(control: Path, output: Path, explicit: bool = False) -> None:
    with intent_lock(control):
        intent = json.loads((control / "intent.json").read_text())
        if not explicit and intent["desired"] != "running":
            raise ValueError("Pause requested during startup; refusing to start trainer")
        stop = output / "STOP"
        if stop.exists():
            if stop.read_text() != intent.get("stop_marker"):
                raise ValueError("Unowned STOP marker: inspect/acknowledge it before resuming")
            stop.unlink()
        intent.pop("stop_reason", None)
        intent.pop("stop_marker", None)
        intent["desired"] = "running"
        atomic_json(control / "intent.json", intent)


def verify_assets(root: Path, manifest: dict) -> None:
    for name, digest in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or sha256(path) != digest:
            raise ValueError(f"Staged asset checksum mismatch: {name}")


def prepare_resume(output: Path, manifest: dict) -> dict:
    import torch

    if list((output / "generations").glob("step_*.json")):
        return recover(output)
    if sha256(output / "latest_state.pt") != manifest["initial_state_sha256"]:
        raise ValueError("Initial state checksum mismatch")
    state = torch.load(output / "latest_state.pt", map_location="cpu", weights_only=False)
    validate_state(state)
    publish(output, state, retain=3)
    return state


def trainer_command(config: dict) -> list[str]:
    recipe = config["recipe"].copy()
    recipe.update(
        device="cuda", resume=True, refresh_corpus=False,
        checkpoint_generations=3, allow_device_rng_migration=True,
        device_rng_seed=recipe["seed"], save_final_checkpoint=False,
    )
    command = [config["python"], "-u", "-m", "so101.train_trajectory_rl"]
    for key, value in recipe.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if key in ("resume", "save_final_checkpoint", "record_rollouts"):
                command.append(flag if value else "--no-" + key.replace("_", "-"))
            elif value:
                command.append(flag)
        elif value is not None:
            command.extend([flag, str(value)])
    return command


def put_blob(account: str, container: str, name: str, path: Path) -> None:
    url = f"https://{account}.blob.core.windows.net/{container}/{quote(name, safe='/')}"
    headers = {
        "Authorization": "Bearer " + access_token("https://storage.azure.com/"),
        "x-ms-version": "2023-11-03", "x-ms-blob-type": "BlockBlob",
        "Content-Length": str(path.stat().st_size),
        "Content-Type": "application/json" if path.suffix == ".json" else "application/octet-stream",
        "x-ms-date": formatdate(usegmt=True),
    }
    with path.open("rb") as stream:
        request = Request(url, data=stream, headers=headers, method="PUT")
        with urlopen(request, timeout=120) as response:
            if response.status != 201:
                raise RuntimeError(f"Blob upload returned HTTP {response.status}")


def backup_latest(control: Path, config: dict) -> None:
    output = Path(config["root"]) / config["output"]
    index = output / "checkpoint_manifest.json"
    if not index.exists():
        return
    manifest = json.loads(index.read_text())
    previous_path = control / "backup.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    if previous.get("step") == manifest["step"]:
        return
    checkpoint = output / "generations" / manifest["file"]
    # Open a stable snapshot so retention pruning cannot invalidate the upload.
    snapshot = control / "backup.pt"
    snapshot.unlink(missing_ok=True)
    os.link(checkpoint, snapshot)
    try:
        if sha256(snapshot) != manifest["sha256"]:
            raise ValueError("Backup checkpoint hash mismatch")
        prefix = f"{config['run_name']}/checkpoints/{checkpoint.stem}"
        upload = lambda name, path: put_blob(
            config["storage_account"], "training", name, path
        )
        upload(prefix + ".pt", snapshot)
        upload(prefix + ".split.json", output / "data_split.json")
        upload(prefix + ".identity.json", output / "cloud_identity.json")
        committed = control / "backup-manifest.json"
        atomic_json(committed, manifest)
        upload(prefix + ".json", committed)  # Commit object is always last.
        atomic_json(previous_path, {"step": manifest["step"], "at": time.time()})
    finally:
        snapshot.unlink(missing_ok=True)


def backup_loop(control: Path, config: dict, done: threading.Event) -> None:
    while not done.is_set():
        try:
            backup_latest(control, config)
            (control / "backup-error.json").unlink(missing_ok=True)
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            LOGGER.error("Checkpoint backup failed: %s", error)
            atomic_json(control / "backup-error.json", {"error": str(error)})
        done.wait(15)


def status_payload(control: Path, state: str, pid: int | None, error=None) -> dict:
    config = json.loads((control / "worker.json").read_text())
    output = Path(config["root"]) / config["output"]
    manifest = output / "checkpoint_manifest.json"
    backup = control / "backup.json"
    backup_error = control / "backup-error.json"
    return {
        "state": state, "pid": pid, "heartbeat": time.time(), "error": error,
        "step": json.loads(manifest.read_text())["step"] if manifest.exists() else None,
        "backup": json.loads(backup.read_text()) if backup.exists() else None,
        "backup_error": json.loads(backup_error.read_text()) if backup_error.exists() else None,
    }


def run_worker(control: Path) -> None:
    config = json.loads((control / "worker.json").read_text())
    root = Path(config["root"])
    output = root / config["output"]
    intent = json.loads((control / "intent.json").read_text())
    if intent["desired"] != "running":
        atomic_json(control / "status.json", status_payload(control, "paused", None))
        return
    with (control / "trainer.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(control / "status.json", status_payload(control, "starting", None))
        deadline, compute = lease()
        if time.time() >= deadline:
            raise ValueError("Compute lease expired; start from the controller")
        manifest = json.loads((root / "cloud_manifest.json").read_text())
        identity = json.loads((output / "cloud_identity.json").read_text())
        if sha256(root / "cloud_manifest.json") != identity["manifest_sha256"]:
            raise ValueError("Staged source/data manifest identity mismatch")
        verify_assets(root, manifest)
        resume_state = prepare_resume(output, manifest)
        last_step = resume_state["step"]
        del resume_state  # Never retain a CPU snapshot beside the policy worker.
        acknowledge_stop(control, output)
        environment = os.environ | {
            "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl",
            "HF_HOME": str(root / "cloud_hf"),
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "OMP_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false",
        }
        done = threading.Event()
        backup = threading.Thread(target=backup_loop, args=(control, config, done), daemon=True)
        backup.start()
        events_error = None
        try:
            with (output / "worker.stdout.log").open("ab", buffering=0) as stdout:
                child = subprocess.Popen(
                    trainer_command(config), cwd=root, env=environment,
                    stdout=stdout, stderr=subprocess.STDOUT,
                )
                try:
                    while child.poll() is None:
                        intent = json.loads((control / "intent.json").read_text())
                        stopping = "stop_reason" in intent
                        try:
                            events = metadata("scheduledevents?api-version=2020-07-01")
                            for event in events.get("Events", []):
                                if compute["name"] in event.get("Resources", []) and event.get("EventType") in (
                                    "Preempt", "Terminate", "Redeploy", "Reboot", "Freeze",
                                ):
                                    request_stop(control, output, "eviction")
                                    stopping = True
                            events_error = None
                        except (OSError, ValueError, KeyError) as error:
                            events_error = f"Eviction watcher unavailable: {error}"
                            LOGGER.error("%s", events_error)
                        atomic_json(control / "status.json", status_payload(
                            control, "checkpointing" if stopping else "running",
                            child.pid, events_error,
                        ))
                        time.sleep(5)
                except BaseException:
                    child.terminate()
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                    raise
            if child.returncode != 0:
                raise RuntimeError(
                    f"Trainer exited {child.returncode}; inspect worker.stdout.log. "
                    "Fatal errors are not automatically restarted."
                )
            intent = json.loads((control / "intent.json").read_text())
            finished = status_payload(control, "paused", None)
            if finished["step"] >= config["recipe"]["steps"]:
                finished["state"] = "completed"
            elif intent.get("stop_reason") == "eviction":
                finished["state"] = "interrupted"
            elif not (output / "STOP").exists():
                raise RuntimeError(f"Trainer exited early after step {last_step}")
            else:
                intent["desired"] = "paused"
                atomic_json(control / "intent.json", intent)
            atomic_json(control / "status.json", finished)
        finally:
            done.set()
            backup.join()
        # Also commit the last checkpoint when a short run finishes between polls.
        backup_latest(control, config)
        (control / "backup-error.json").unlink(missing_ok=True)
        atomic_json(control / "status.json", status_payload(control, finished["state"], None))


def install(control: Path) -> None:
    config = json.loads((control / "worker.json").read_text())
    user = os.environ["USER"]
    root = config["root"]
    if any(character in (root + str(control) + user) for character in "\n\r%\"\\"):
        raise ValueError("Unsupported systemd path/account characters")
    service = (
        "[Unit]\nDescription=SO101 single Spot trainer\nAfter=network-online.target\n"
        f"RequiresMountsFor={control}\n[Service]\nType=simple\nUser={user}\n"
        f'WorkingDirectory={root}\nExecStart="{config["python"]}" -m so101.cloud_worker '
        f'--control "{control}" run\nRestart=no\nKillMode=control-group\n'
        "TimeoutStopSec=30\n[Install]\nWantedBy=multi-user.target\n"
    )
    guard_service = (
        "[Unit]\nDescription=SO101 independent compute lease guard\n"
        f"RequiresMountsFor={control}\n[Service]\nType=oneshot\nUser={user}\n"
        f'Environment="PYTHONPATH={root}/src"\n'
        f'ExecStart=/usr/bin/python3 -m so101.cloud_worker --control "{control}" guard\n'
    )
    timer = (
        "[Unit]\nDescription=SO101 lease checks\n[Timer]\nOnBootSec=20\n"
        "OnUnitActiveSec=20\n[Install]\nWantedBy=timers.target\n"
    )
    for name, body in (
        ("so101-trainer.service", service),
        ("so101-guard.service", guard_service),
        ("so101-guard.timer", timer),
    ):
        path = control / name
        path.write_text(body)
        subprocess.run(["sudo", "-n", "install", "-m", "644", str(path),
                        "/etc/systemd/system/" + name], check=True)
    subprocess.run(["sudo", "-n", "systemctl", "daemon-reload"], check=True)
    subprocess.run(["sudo", "-n", "systemctl", "enable", "so101-trainer.service"], check=True)
    subprocess.run(["sudo", "-n", "systemctl", "enable", "--now", "so101-guard.timer"], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, default=Path("/mnt/training/control"))
    parser.add_argument("action", choices=("run", "guard", "install", "pause", "resume", "status"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    control = args.control
    config = json.loads((control / "worker.json").read_text())
    output = Path(config["root"]) / config["output"]
    try:
        if args.action == "run":
            run_worker(control)
        elif args.action == "guard":
            guard(control)
        elif args.action == "install":
            install(control)
        elif args.action == "pause":
            request_stop(control, output, "pause")
        elif args.action == "resume":
            acknowledge_stop(control, output, explicit=True)
            subprocess.run(
                ["sudo", "-n", "systemctl", "start", "so101-trainer.service"], check=True,
            )
        elif args.action == "status":
            status = json.loads((control / "status.json").read_text())
            log = output / "trajectory_rl.json"
            status["log_revision"] = (
                f"{log.stat().st_mtime_ns}:{log.stat().st_size}" if log.exists() else None
            )
            status["evidence_available"] = (output / "rollouts").is_dir()
            status["split_available"] = (output / "data_split.json").is_file()
            status["rollout_retention_steps"] = config["recipe"].get("rollout_retention_steps", 20)
            print(json.dumps(status))
    except (OSError, ValueError, KeyError, RuntimeError, ImportError, subprocess.SubprocessError) as error:
        if args.action == "run":
            with intent_lock(control):
                intent = json.loads((control / "intent.json").read_text())
                intent["desired"] = "paused"
                atomic_json(control / "intent.json", intent)
            atomic_json(control / "status.json", status_payload(control, "error", None, str(error)))
        raise


if __name__ == "__main__":
    main()
