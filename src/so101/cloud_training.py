"""Local Azure Spot controller. All paid mutations require --apply."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import ipaddress
import json
import logging
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import time
import uuid

from so101.cloud_checkpoint import atomic_json, sha256, validate_state


ROOT = Path(__file__).resolve().parents[2]
REMOTE = "/mnt/training/workspace"
CONTROL = "/mnt/training/control"
LOGGER = logging.getLogger(__name__)


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    uuid.UUID(config["subscription"])
    for key in ("resource_group", "vm_name", "admin"):
        if not re.fullmatch(r"[a-z][a-z0-9-]{1,39}", config[key]):
            raise ValueError(f"Invalid {key}")
    if not re.fullmatch(r"[a-z0-9]{3,24}", config["storage_account"]):
        raise ValueError("storage_account must be 3–24 lowercase letters/digits")
    if not re.fullmatch(r"[a-z0-9]+", config["location"]) or config["zone"] not in ("1", "2", "3"):
        raise ValueError("Invalid region/zone")
    if config["vm_size"] not in ("Standard_NC24ads_A100_v4", "Standard_NC40ads_H100_v5"):
        raise ValueError("Select one supported single-GPU Azure SKU")
    image = config["image"].split(":")
    if len(image) != 4 or image[-1].lower() == "latest" or not re.fullmatch(r"\d[\d.]+", image[-1]):
        raise ValueError("Select a pinned publisher:offer:sku:version Ubuntu 24.04 image")
    network = ipaddress.ip_network(config["ssh_source_cidr"], strict=True)
    if network.version != 4 or network.prefixlen != 32:
        raise ValueError("SSH must be restricted to your current public IPv4 /32")
    for key in ("spot_max_usd_per_hour", "max_session_hours", "total_gpu_hours"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if config["max_session_hours"] > 12 or config["max_session_hours"] > config["total_gpu_hours"]:
        raise ValueError("Session must be <=12 hours and within the total GPU-hour budget")
    if not isinstance(config["disk_gib"], int) or config["disk_gib"] < 128:
        raise ValueError("Use a persistent disk >=128 GiB")
    relative_path(config["run"])
    if not config["run"].startswith("outputs/") or config["run"].startswith("outputs/cloud/"):
        raise ValueError("run must name a local trainer output, not a cloud mirror")
    return config


def relative_path(name: str) -> Path:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Only workspace-relative paths are allowed: {name}")
    if not (ROOT / path).resolve().is_relative_to(ROOT):
        raise ValueError(f"Path leaves the workspace: {name}")
    return path


def state_directory(config: dict) -> Path:
    return ROOT / "outputs/cloud/control" / config["vm_name"]


def mirror_directory(config: dict) -> Path:
    return ROOT / "outputs/cloud" / config["vm_name"]


def az(config: dict, *arguments: str) -> object:
    try:
        process = subprocess.run(
            ["az", *arguments, "--subscription", config["subscription"],
             "--only-show-errors", "--output", "json"],
            text=True, capture_output=True, check=True, timeout=1800,
        )
    except subprocess.CalledProcessError as error:
        LOGGER.error("Azure CLI failed: %s", (error.stderr or "").strip())
        raise
    return json.loads(process.stdout) if process.stdout.strip() else None


def reserve_session(directory: Path, config: dict, now: float) -> float:
    path = directory / "controller.json"
    state = json.loads(path.read_text()) if path.exists() else {"reserved_hours": 0}
    hours = config["max_session_hours"]
    if state["reserved_hours"] + hours > config["total_gpu_hours"] + 1e-9:
        state["desired"] = "paused"
        atomic_json(path, state)
        raise ValueError("Total reserved GPU-hour budget exhausted; no allocation attempted")
    deadline = now + hours * 3600
    state.update(
        reserved_hours=state["reserved_hours"] + hours,
        desired="running", deadline=deadline,
        max_compute_usd=(state["reserved_hours"] + hours) * config["spot_max_usd_per_hour"],
    )
    state.setdefault("allocations", []).append({"at": now, "hours": hours, "deadline": deadline})
    atomic_json(path, state)
    return deadline


def parameters(config: dict, deadline: float) -> dict:
    publisher, offer, sku, version = config["image"].split(":")
    public = Path(config["ssh_public_key"]).expanduser().read_text().strip()
    if not public.startswith("ssh-ed25519 "):
        raise ValueError("Use a dedicated Ed25519 SSH public key")
    values = {
        "location": config["location"], "zone": config["zone"],
        "name": config["vm_name"], "vmSize": config["vm_size"],
        "image": dict(publisher=publisher, offer=offer, sku=sku, version=version),
        "admin": config["admin"], "sshKey": public, "sshCidr": config["ssh_source_cidr"],
        "storageAccount": config["storage_account"],
        "spotMax": str(config["spot_max_usd_per_hour"]),
        "deadline": str(deadline),
        "shutdownTime": datetime.fromtimestamp(deadline, timezone.utc).strftime("%H%M"),
        "diskGiB": config["disk_gib"],
    }
    return {key: {"value": value} for key, value in values.items()}


def deploy(config: dict, directory: Path) -> None:
    # Validate local inputs/auth/scope before consuming a reservation or allocating.
    parameters(config, time.time() + config["max_session_hours"] * 3600)
    state_path = directory / "controller.json"
    previous = json.loads(state_path.read_text()) if state_path.exists() else None
    target = {key: config[key] for key in (
        "subscription", "resource_group", "vm_name", "storage_account", "location", "zone",
    )}
    if previous:
        if previous.get("deployment_target") != target:
            raise ValueError("Existing reservation belongs to a different deployment target")
        if previous.get("deployment_complete") or (directory / "staged.json").exists():
            raise ValueError("Deployment already completed/staged; use start for the existing VM")
    if az(config, "group", "exists", "--name", config["resource_group"]):
        resources = az(config, "resource", "list", "--resource-group", config["resource_group"])
        if resources and not previous:
            raise ValueError("Deploy requires an empty dedicated resource group; inspect existing resources")
    deadline = reserve_session(directory, config, time.time())
    state = json.loads(state_path.read_text())
    state["desired"] = "paused"  # Deployment is not authorization to start a policy.
    state.update(deployment_target=target, deployment_complete=False)
    atomic_json(state_path, state)
    params = directory / "parameters.json"
    atomic_json(params, {"parameters": parameters(config, deadline)})
    az(config, "group", "create", "--name", config["resource_group"],
       "--location", config["location"], "--tags", "so101=training")
    try:
        az(config, "deployment", "group", "create",
           "--resource-group", config["resource_group"], "--name", "so101-spot",
           "--template-file", str(ROOT / "cloud/azure-resources.json"),
           "--parameters", "@" + str(params))
    except subprocess.SubprocessError as deployment_error:
        LOGGER.error("Deployment failed; attempting compute deallocation, retaining all storage")
        try:
            az(config, "vm", "deallocate", "--resource-group", config["resource_group"],
               "--name", config["vm_name"])
        except subprocess.SubprocessError as shutdown_error:
            LOGGER.error("Could not confirm deallocation: %s. Check the Azure portal NOW.", shutdown_error)
        raise deployment_error
    state = json.loads(state_path.read_text())
    state["deployment_complete"] = True
    atomic_json(state_path, state)
    print("Spot VM deployed; session clock is running. Mount/bootstrap/stage before starting training.")


def connection(config: dict, directory: Path) -> tuple[str, list[str]]:
    address = az(config, "network", "public-ip", "show",
                 "--resource-group", config["resource_group"],
                 "--name", config["vm_name"] + "-ip", "--query", "ipAddress")
    ipaddress.IPv4Address(str(address))
    return f"{config['admin']}@{address}", [
        "-i", str(Path(config["ssh_private_key"]).expanduser()),
        "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=yes",
        "-o", "UserKnownHostsFile=" + str(directory / "known_hosts"),
        "-o", "ConnectTimeout=10",
    ]


def pin_host(config: dict, directory: Path) -> None:
    destination, _ = connection(config, directory)
    result = az(config, "vm", "run-command", "invoke", "--resource-group", config["resource_group"],
                "--name", config["vm_name"], "--command-id", "RunShellScript",
                "--scripts", "cat /etc/ssh/ssh_host_ed25519_key.pub")
    messages = "\n".join(row["message"] for row in result["value"])
    keys = re.findall(r"^ssh-ed25519 [A-Za-z0-9+/=]+(?: .*)?$", messages, re.MULTILINE)
    if len(keys) != 1:
        raise ValueError("Could not verify a unique host key through Azure Run Command")
    address = destination.split("@")[1]
    key = " ".join(keys[0].split()[:2])
    path = directory / "known_hosts"
    if path.exists() and path.read_text().strip() != f"{address} {key}":
        raise ValueError(
            "Host key/address changed. Inspect Azure Run Command output, then explicitly "
            "archive the old controller known_hosts file and run pin-host again."
        )
    path.write_text(f"{address} {key}\n")
    print("SSH host key verified using the authenticated Azure control plane.")


def ssh(config: dict, directory: Path, arguments: list[str], timeout=120) -> str:
    destination, options = connection(config, directory)
    try:
        result = subprocess.run(
            ["ssh", *options, destination, shlex.join(arguments)],
            check=True, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.CalledProcessError as error:
        LOGGER.error("Worker SSH command failed: %s", (error.stderr or "").strip())
        raise
    return result.stdout


def worker(config: dict, directory: Path, action: str) -> str:
    return ssh(config, directory, [
        "env", "PYTHONPATH=" + REMOTE + "/src", "python3", "-m",
        "so101.cloud_worker", "--control", CONTROL, action,
    ])


def mount_disk(config: dict, directory: Path, initialize: bool) -> None:
    transfer(config, directory, str(ROOT / "scripts/mount_cloud_disk.sh"),
             "@WORKER@:/tmp/so101-mount-disk.sh")
    command = ["sudo", "-n", "bash", "/tmp/so101-mount-disk.sh"]
    if initialize:
        command.append("--initialize-empty-disk")
    print(ssh(config, directory, command))


def bootstrap(config: dict, directory: Path) -> None:
    ssh(config, directory, ["test", "!", "-e", CONTROL + "/worker.json"])
    with tempfile.TemporaryDirectory(prefix="so101-bootstrap-") as temporary:
        archive_path = Path(temporary) / "bootstrap.tar.gz"
        files = [ROOT / "pyproject.toml", ROOT / "uv.lock"]
        for folder in ("scripts", "src"):
            files.extend(path for path in (ROOT / folder).rglob("*")
                         if path.is_file() and "__pycache__" not in path.parts)
        with tarfile.open(archive_path, "w:gz") as archive:
            for path in files:
                relative_path(str(path.relative_to(ROOT)))
                if path.is_symlink():
                    raise ValueError(f"Do not stage symlinks: {path}")
                archive.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
        transfer(config, directory, str(archive_path), "@WORKER@:/mnt/training/bootstrap.tar.gz")
    ssh(config, directory, ["mkdir", "-p", REMOTE])
    ssh(config, directory, ["tar", "-xzf", "/mnt/training/bootstrap.tar.gz", "-C", REMOTE])
    print(ssh(config, directory, [
        "bash", REMOTE + "/scripts/setup_cloud_gpu.sh", "--storage", "/mnt/training",
    ], timeout=1800))


def local_trainer_active(config: dict) -> bool:
    from so101.rl_dashboard import active_run_outputs
    return (ROOT / config["run"]).resolve() in active_run_outputs()


def stage_files(recipe: dict, split: dict) -> set[Path]:
    files = {Path(name) for name in ("pyproject.toml", "uv.lock")}
    for directory in ("src", "scripts", "data/simulation/model"):
        files.update(
            path.relative_to(ROOT) for path in (ROOT / directory).rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        )
    checkpoint = relative_path(recipe["checkpoint"])
    files.update(path.relative_to(ROOT) for path in (ROOT / checkpoint).rglob("*") if path.is_file())
    collection = relative_path(recipe["collection"]) / "collection.json"
    files.add(collection)
    report = json.loads((ROOT / collection).read_text())
    sources = set(report["successful_episodes"]) | set(split["validation"])
    for task_sources in split["training"].values():
        sources.update(task_sources)
    for source in sources:
        for name in ("episode.json", "frames.jsonl"):
            files.add(relative_path(source) / name)
    return files


def cached_model_assets(recipe: dict, temporary: Path) -> tuple[dict[Path, Path], dict[str, str]]:
    """Flatten only the two required public Hub snapshots, never tokens/other caches."""
    from huggingface_hub import try_to_load_from_cache

    metadata = json.loads((ROOT / relative_path(recipe["checkpoint"]) / "metadata.json").read_text())
    policy_repo = metadata["source_checkpoint"]
    repos = [policy_repo]
    assets = {}
    revisions = {}
    for repo in repos:
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
            raise ValueError(f"Expected a public Hub model identifier, got {repo!r}")
        cached = try_to_load_from_cache(repo, "config.json")
        if not isinstance(cached, str):
            raise ValueError(f"Required cached Hub configuration missing: {repo}")
        snapshot = Path(cached).parent
        revision = snapshot.name
        if not re.fullmatch(r"[a-f0-9]{40}", revision):
            raise ValueError(f"Cannot identify pinned Hub snapshot for {repo}")
        revisions[repo] = revision
        prefix = Path("cloud_hf/hub") / ("models--" + repo.replace("/", "--"))
        for file in snapshot.rglob("*"):
            if file.is_file():
                assets[prefix / "snapshots" / revision / file.relative_to(snapshot)] = file.resolve()
        reference = temporary / (repo.replace("/", "--") + ".ref")
        reference.write_text(revision)
        assets[prefix / "refs/main"] = reference
        if repo == policy_repo:
            vlm = json.loads(Path(cached).read_text())["vlm_model_name"]
            if vlm != repo:
                repos.append(vlm)
    return assets, revisions


def prepare(config: dict, destination: Path) -> None:
    import torch

    output = ROOT / config["run"]
    if destination.exists():
        raise ValueError("Choose a new bundle filename; existing snapshots are immutable")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="so101-stage-") as temporary:
        snapshot = Path(temporary) / "latest_state.pt"
        # Opening once holds a complete old inode even when the trainer renames its next state.
        with (output / "latest_state.pt").open("rb") as source, snapshot.open("wb") as target:
            shutil.copyfileobj(source, target, 1024 * 1024)
        state = torch.load(snapshot, map_location="cpu", weights_only=False)
        validate_state(state)
        recipe = state["log"]["run_config"].copy()
        recipe.pop("allow_device_rng_migration", None)
        recipe.pop("device_rng_seed", None)
        recipe["output"] = config["run"]
        if recipe.get("rl_estimator") != "denoising-grpo" or state["log"].get("training_objective") != "reward-only":
            raise ValueError("Expected the existing reward-only denoising run")
        split_path = output / "data_split.json"
        split = json.loads(split_path.read_text())
        files = stage_files(recipe, split)
        files.add(relative_path(config["run"]) / "data_split.json")
        for path in files:
            relative_path(str(path))
            if (ROOT / path).is_symlink():
                raise ValueError(f"Do not stage symlinks: {path}")
        hub_assets, revisions = cached_model_assets(recipe, Path(temporary))
        hashes = {str(path): sha256(ROOT / path) for path in sorted(files)}
        hashes.update({str(name): sha256(source) for name, source in hub_assets.items()})
        manifest = {
            "step": state["step"], "recipe": recipe, "run": config["run"],
            "initial_state_sha256": sha256(snapshot),
            "files": hashes, "hub_revisions": revisions,
        }
        manifest_path = Path(temporary) / "cloud_manifest.json"
        atomic_json(manifest_path, manifest)
        with tarfile.open(destination, "w:gz", compresslevel=1) as archive:
            for path in sorted(files):
                archive.add(ROOT / path, arcname=str(path), recursive=False)
            for name, source in sorted(hub_assets.items()):
                archive.add(source, arcname=str(name), recursive=False)
            archive.add(snapshot, arcname=config["run"] + "/latest_state.pt")
            archive.add(manifest_path, arcname="cloud_manifest.json")
            log_path = Path(temporary) / "trajectory_rl.json"
            atomic_json(log_path, state["log"])
            archive.add(log_path, arcname=config["run"] + "/trajectory_rl.json")
        atomic_json(destination.with_suffix(destination.suffix + ".json"), {
            "sha256": sha256(destination), "step": state["step"],
            "live_local_trainer": local_trainer_active(config),
        })
    print(f"Prepared step {manifest['step']}: {destination}. This does not pause or upload anything.")


def validate_archive(path: Path) -> None:
    with tarfile.open(path) as archive:
        for member in archive:
            relative_path(member.name)
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"Archive contains a link/device: {member.name}")


def transfer(config: dict, directory: Path, source: str, destination: str) -> None:
    target, options = connection(config, directory)
    subprocess.run(
        ["rsync", "-az", "--partial", "--inplace", "-e", shlex.join(["ssh", *options]),
         source.replace("@WORKER@", target), destination.replace("@WORKER@", target)],
        check=True, timeout=1800,
    )


def stage(config: dict, directory: Path, bundle: Path) -> None:
    validate_archive(bundle)
    descriptor = json.loads(bundle.with_suffix(bundle.suffix + ".json").read_text())
    if sha256(bundle) != descriptor["sha256"]:
        raise ValueError("Bundle checksum mismatch")
    # A stale pre-handoff snapshot can be replaced, but never a committed cloud run.
    ssh(config, directory, ["python3", "-c",
        "import json; from pathlib import Path; "
        f"assert not Path({REMOTE + '/' + config['run'] + '/checkpoint_manifest.json'!r}).exists(), "
        "'Cloud run already committed: staging would overwrite it'; "
        f"p=Path({CONTROL + '/intent.json'!r}); "
        "assert not p.exists() or json.loads(p.read_text())['desired']=='paused', "
        "'Worker is not paused'",
    ])
    transfer(config, directory, str(bundle), "@WORKER@:/mnt/training/stage.tar.gz")
    remote_hash = ssh(config, directory, ["sha256sum", "/mnt/training/stage.tar.gz"]).split()[0]
    if remote_hash != descriptor["sha256"]:
        raise ValueError("Transferred archive checksum mismatch")
    ssh(config, directory, ["mkdir", "-p", REMOTE, CONTROL])
    ssh(config, directory, ["tar", "-xzf", "/mnt/training/stage.tar.gz", "-C", REMOTE], timeout=300)
    manifest = json.loads(ssh(config, directory, ["cat", REMOTE + "/cloud_manifest.json"]))
    if manifest["run"] != config["run"]:
        raise ValueError("Bundle run does not match configuration")
    worker_config = {
        "root": REMOTE, "python": "/mnt/training/so101-cloud/venv/bin/python",
        "output": config["run"], "recipe": manifest["recipe"],
        "run_name": config["vm_name"], "storage_account": config["storage_account"],
    }
    identity = {"manifest_sha256": sha256_archive_manifest(bundle)}
    ssh(config, directory, ["test", "-x", worker_config["python"]])
    # Upload backups before exposing a runnable worker.
    az(config, "storage", "blob", "upload", "--account-name", config["storage_account"],
       "--container-name", "training", "--auth-mode", "login",
       "--name", config["vm_name"] + "/assets/" + descriptor["sha256"] + ".tar.gz",
       "--file", str(bundle), "--overwrite", "true")
    for name, value in (
        ("worker.json", worker_config), ("intent.json", {"desired": "paused"}),
        ("cloud_identity.json", identity),
    ):
        local = directory / name
        atomic_json(local, value)
        remote = REMOTE + "/" + config["run"] if name == "cloud_identity.json" else CONTROL
        transfer(config, directory, str(local), "@WORKER@:" + remote + "/" + name)
    worker(config, directory, "install")
    # Status is valid even before the first (slow) CUDA policy construction.
    initial_status = directory / "initial-status.json"
    atomic_json(initial_status, {
        "state": "paused", "heartbeat": time.time(), "pid": None,
        "step": None, "error": None,
    })
    transfer(config, directory, str(initial_status), "@WORKER@:" + CONTROL + "/status.json")
    ready = directory / "staged.json"
    atomic_json(ready, {"step": manifest["step"], "initial_state_sha256": manifest["initial_state_sha256"]})
    transfer(config, directory, str(ready), "@WORKER@:" + CONTROL + "/staged.json")
    print("Assets staged and backed up; trainer remains paused. Start only after local handoff.")


def sha256_archive_manifest(bundle: Path) -> str:
    import hashlib
    with tarfile.open(bundle) as archive:
        stream = archive.extractfile("cloud_manifest.json")
        if stream is None:
            raise ValueError("Missing staging manifest")
        return hashlib.sha256(stream.read()).hexdigest()


def start(config: dict, directory: Path, automatic: bool = False) -> None:
    if local_trainer_active(config):
        raise ValueError("Local trainer is still running. Checkpoint/pause it before cloud start.")
    observed = az(config, "vm", "get-instance-view", "--resource-group", config["resource_group"],
                  "--name", config["vm_name"], "--query", "instanceView.statuses")
    power = next((row["code"] for row in observed if row["code"].startswith("PowerState/")), None)
    controller_path = directory / "controller.json"
    controller = json.loads(controller_path.read_text())
    if automatic and controller["desired"] != "running":
        return
    if power != "PowerState/running" or controller["deadline"] <= time.time():
        deadline = reserve_session(directory, config, time.time())
        az(config, "vm", "update", "--resource-group", config["resource_group"],
           "--name", config["vm_name"], "--set", f"tags.so101_deadline={deadline}")
        az(config, "vm", "auto-shutdown", "--resource-group", config["resource_group"],
           "--name", config["vm_name"], "--time",
           datetime.fromtimestamp(deadline, timezone.utc).strftime("%H%M"))
        if power != "PowerState/running":
            az(config, "vm", "start", "--resource-group", config["resource_group"], "--name", config["vm_name"])
    else:
        if not automatic:
            controller["desired"] = "running"
            atomic_json(controller_path, controller)
    ready = json.loads(ssh(config, directory, ["cat", CONTROL + "/staged.json"]))
    # Automatic retries must enforce this too, including a failed first handoff.
    has_generation = ssh(config, directory, [
        "python3", "-c",
        "from pathlib import Path; print(Path("
        + repr(REMOTE + "/" + config["run"] + "/checkpoint_manifest.json")
        + ").exists())",
    ]).strip() == "True"
    if not has_generation and sha256(ROOT / config["run"] / "latest_state.pt") != ready["initial_state_sha256"]:
        raise ValueError("Stale initial snapshot. Restage the final paused local checkpoint before first start.")
    worker(config, directory, "resume")
    sync(config, directory)
    print("Cloud start requested. Inspect status; starting is not yet a completed training step.")


def sync(config: dict, directory: Path) -> dict:
    with (directory / "mirror.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return sync_locked(config, directory)


def sync_locked(config: dict, directory: Path) -> dict:
    mirror = mirror_directory(config)
    mirror.mkdir(parents=True, exist_ok=True)
    status_path = mirror / "remote_status.json"
    previous = json.loads(status_path.read_text()) if status_path.exists() else {}
    try:
        observed = json.loads(worker(config, directory, "status"))
        if observed.get("log_revision") != previous.get("log_revision"):
            incoming = mirror / "trajectory_rl.json.incoming"
            transfer(config, directory, "@WORKER@:" + REMOTE + "/" + config["run"] + "/trajectory_rl.json",
                     str(incoming))
            log = json.loads(incoming.read_text())
            remote_run = "@WORKER@:" + REMOTE + "/" + config["run"]
            if observed.get("split_available"):
                split_incoming = mirror / "data_split.json.incoming"
                transfer(config, directory, remote_run + "/data_split.json", str(split_incoming))
                split = json.loads(split_incoming.read_text())
                if not isinstance(split.get("training"), dict) or not isinstance(split.get("validation"), list):
                    raise ValueError("Invalid mirrored data split")
                atomic_json(mirror / "data_split.json", split)
            if observed.get("evidence_available"):
                from so101.rollout_evidence import prune_evidence
                evidence = mirror / "rollouts"
                evidence.mkdir(exist_ok=True)
                transfer(config, directory, remote_run + "/rollouts/", str(evidence) + "/")
                prune_evidence(mirror, observed.get("rollout_retention_steps", 20))
            atomic_json(mirror / "trajectory_rl.json", log)
        result = {**observed, "observed_at": time.time()}
        atomic_json(status_path, result)
        return result
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        atomic_json(status_path, {
            **previous, "state": "unreachable", "error": str(error),
            "observed_at": previous.get("observed_at"),
        })
        raise


def pause(config: dict, directory: Path) -> None:
    path = directory / "controller.json"
    state = json.loads(path.read_text())
    state["desired"] = "paused"
    atomic_json(path, state)  # Pause intent survives an unreachable worker.
    worker(config, directory, "pause")
    print("Pause requested. Trainer finishes its step; use status before deallocating.")


def watch(config: dict, directory: Path, interval: float = 30) -> None:
    backoff = interval
    while True:
        state = json.loads((directory / "controller.json").read_text())
        try:
            observed = az(
                config, "vm", "get-instance-view", "--resource-group", config["resource_group"],
                "--name", config["vm_name"], "--query", "instanceView.statuses",
            )
            codes = {row["code"] for row in observed}
            if "PowerState/deallocated" in codes and state["desired"] == "running":
                # An expired session must never be treated as an eviction to bypass the budget.
                if time.time() >= state["deadline"]:
                    with (directory / "controller.lock").open("a") as lock:
                        fcntl.flock(lock, fcntl.LOCK_EX)
                        state = json.loads((directory / "controller.json").read_text())
                        if time.time() >= state["deadline"]:
                            state["desired"] = "paused"
                            atomic_json(directory / "controller.json", state)
                    print("Session expired; paused. Explicit start is required for another reservation.")
                else:
                    with (directory / "controller.lock").open("a") as lock:
                        fcntl.flock(lock, fcntl.LOCK_EX)
                        start(config, directory, automatic=True)
            elif "PowerState/running" in codes:
                status = sync(config, directory)
                print(json.dumps(status), flush=True)
                if status["state"] in ("error", "completed"):
                    state["desired"] = "paused"
                    atomic_json(directory / "controller.json", state)
                    if status["state"] == "error":
                        raise RuntimeError(status["error"])
                    return
            backoff = interval
        except subprocess.CalledProcessError as error:
            # Only known capacity errors are retried; authentication/configuration errors stop.
            message = error.stderr or ""
            if not any(code in message for code in ("AllocationFailed", "SkuNotAvailable", "OverconstrainedAllocationRequest")):
                # SSH can fail while a previously observed VM is being evicted.
                if error.cmd and Path(error.cmd[0]).name in ("ssh", "rsync"):
                    LOGGER.warning("Worker contact lost; rechecking Azure power state: %s", error)
                    time.sleep(backoff)
                    continue
                raise
            LOGGER.warning("Spot capacity unavailable; retry in %.0fs: %s", backoff, message)
            waiting = mirror_directory(config) / "remote_status.json"
            atomic_json(waiting, {"state": "waiting_for_capacity", "observed_at": time.time()})
            backoff = min(backoff * 2, 600)
        time.sleep(backoff)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="Authorize cloud changes/transfers")
    parser.add_argument("action", choices=(
        "plan", "deploy", "pin-host", "ssh", "prepare", "stage",
        "mount", "bootstrap", "start", "resume", "pause", "status", "sync", "watch", "logs",
        "deallocate", "destroy-compute",
    ))
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--initialize-empty-disk", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    config = load_config(args.config)
    directory = state_directory(config)
    directory.mkdir(parents=True, exist_ok=True)
    # Short control operations are serialized; a sleeping watch never holds this lock.
    mutating = args.action not in ("plan", "prepare", "status", "sync", "logs", "ssh")
    if mutating and not args.apply:
        parser.error("This action requires --apply; plan is offline and free")
    with (directory / "controller.lock").open("w") as lock:
        if args.action != "watch":
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == "plan":
            print(json.dumps({
                "provider": "Azure Spot ONLY; no on-demand fallback",
                "sku": config["vm_size"], "region": config["location"], "zone": config["zone"],
                "session_hours": config["max_session_hours"],
                "compute_upper_estimate_usd": config["spot_max_usd_per_hour"] * config["total_gpu_hours"],
                "excluded": "disks, backups, networking, taxes; NOT a hard billing cap",
                "image": config["image"],
            }, indent=2))
        elif args.action == "deploy":
            deploy(config, directory)
        elif args.action == "prepare":
            if args.bundle is None:
                parser.error("--bundle is required")
            prepare(config, args.bundle)
        elif args.action == "pin-host":
            pin_host(config, directory)
        elif args.action == "ssh":
            target, options = connection(config, directory)
            subprocess.run(["ssh", *options, target], check=True)
        elif args.action == "stage":
            if args.bundle is None:
                parser.error("--bundle is required")
            stage(config, directory, args.bundle)
        elif args.action == "mount":
            mount_disk(config, directory, args.initialize_empty_disk)
        elif args.action == "bootstrap":
            bootstrap(config, directory)
        elif args.action in ("start", "resume"):
            start(config, directory)
        elif args.action == "pause":
            pause(config, directory)
        elif args.action in ("status", "sync"):
            print(json.dumps(sync(config, directory), indent=2))
        elif args.action == "watch":
            with (directory / "watch.lock").open("a") as watch_lock:
                fcntl.flock(watch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                watch(config, directory)
        elif args.action == "logs":
            print(ssh(config, directory, ["tail", "-n", "60", REMOTE + "/" + config["run"] + "/worker.stdout.log"]))
        elif args.action in ("deallocate", "destroy-compute"):
            state_path = directory / "controller.json"
            state = json.loads(state_path.read_text())
            state["desired"] = "paused"
            atomic_json(state_path, state)
            if args.action == "deallocate":
                az(config, "vm", "deallocate", "--resource-group", config["resource_group"], "--name", config["vm_name"])
            else:
                az(config, "vm", "delete", "--resource-group", config["resource_group"],
                   "--name", config["vm_name"], "--yes")
            print("Compute stopped/deleted. Retained disk, backups and IP still incur charges.")


if __name__ == "__main__":
    main()
