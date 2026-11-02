import io
import json
from pathlib import Path
import subprocess
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import so101.cloud_checkpoint as checkpoints
import so101.cloud_training as cloud
import so101.cloud_worker as worker
from so101.train_trajectory_rl import restore_resume_state, save_resume_state


def small_state(tmp_path, step=7):
    model = torch.nn.Linear(2, 1)
    controller = SimpleNamespace(policy=model, device="cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    (tmp_path / "data_split.json").write_text('{"validation":[]}')
    log = {"steps": [{"step": step}], "solved_problems": {"old": {"solved_at_step": 2}}}
    save_resume_state(tmp_path, controller, optimizer, np.random.default_rng(42), log)
    return torch.load(tmp_path / "latest_state.pt", weights_only=False), controller, optimizer


def test_generations_reject_corruption_and_recover_previous_commit(tmp_path):
    state, _, _ = small_state(tmp_path)
    checkpoints.publish(tmp_path, state, retain=3)
    state["step"] = 8
    state["log"]["steps"].append({"step": 8})
    checkpoints.publish(tmp_path, state, retain=3)
    (tmp_path / "generations/step_000000008.pt").write_bytes(b"partial")
    recovered = checkpoints.recover(tmp_path)
    assert recovered["step"] == 7
    assert recovered["log"]["solved_problems"] == {"old": {"solved_at_step": 2}}
    assert torch.load(tmp_path / "latest_state.pt", weights_only=False)["step"] == 7
    recovered["step"] = 8
    recovered["log"]["steps"].append({"step": 8})
    checkpoints.publish(tmp_path, recovered, retain=3)
    assert checkpoints.recover(tmp_path)["step"] == 8
    assert list((tmp_path / "generations/rejected").glob("*.json"))


def test_uncommitted_generation_is_ignored_and_can_be_recomputed(tmp_path):
    state, _, _ = small_state(tmp_path)
    checkpoints.publish(tmp_path, state, retain=2)
    orphan = tmp_path / "generations/step_000000008.pt"
    orphan.write_bytes(b"killed before manifest commit")
    assert checkpoints.recover(tmp_path)["step"] == 7
    state["step"] = 8
    state["log"]["steps"].append({"step": 8})
    checkpoints.publish(tmp_path, state, retain=2)
    assert checkpoints.recover(tmp_path)["step"] == 8


def test_checkpoint_retention_and_split_identity(tmp_path):
    state, _, _ = small_state(tmp_path)
    for step in (7, 8, 9):
        state["step"] = step
        state["log"]["steps"] = [{"step": step}]
        checkpoints.publish(tmp_path, state, retain=2)
    assert len(list((tmp_path / "generations").glob("*.pt"))) == 2
    (tmp_path / "data_split.json").write_text('{"validation":["changed"]}')
    with pytest.raises(ValueError, match="No verified"):
        checkpoints.recover(tmp_path)


def test_backend_migration_is_explicit_and_never_uses_foreign_rng_bytes(tmp_path, monkeypatch):
    state, controller, optimizer = small_state(tmp_path)
    state["rng"]["backend"] = "mps"
    state["rng"]["device"] = torch.tensor([1, 2], dtype=torch.uint8)
    controller.device = "cuda"  # Tiny CPU parameters; no CUDA allocation needed.
    seeded = []
    monkeypatch.setattr(torch.cuda, "manual_seed", seeded.append)
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda *a: pytest.fail("foreign RNG applied"))
    with pytest.raises(ValueError, match="explicitly allow"):
        restore_resume_state(state, controller, optimizer, np.random.default_rng(), 1e-4)
    restored = restore_resume_state(
        state, controller, optimizer, np.random.default_rng(), 1e-4, True, 61000,
    )
    assert seeded == [61000]
    assert restored["device_rng_migrations"][-1] == {
        "after_step": 7, "source": "mps", "target": "cuda", "seed": 61000, "bit_exact": False,
    }
    assert optimizer.state[next(controller.policy.parameters())]["step"] == 1


def test_same_backend_restores_device_rng_without_reseeding(tmp_path, monkeypatch):
    state, controller, optimizer = small_state(tmp_path)
    state["rng"]["backend"] = "cuda"
    state["rng"]["device"] = torch.tensor([3, 4], dtype=torch.uint8)
    controller.device = "cuda"
    seen = []
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda value: seen.append(value.tolist()))
    monkeypatch.setattr(torch.cuda, "manual_seed", lambda *a: pytest.fail("unexpected reseed"))
    restore_resume_state(state, controller, optimizer, np.random.default_rng(), 1e-4)
    assert seen == [[3, 4]]


def test_nonfinite_checkpoint_is_not_committed(tmp_path):
    state, _, _ = small_state(tmp_path)
    next(iter(state["model"].values())).fill_(float("nan"))
    with pytest.raises(ValueError, match="Non-finite"):
        checkpoints.publish(tmp_path, state, retain=3)
    assert not list((tmp_path / "generations").glob("*.json"))


def config():
    return {
        "subscription": "00000000-0000-0000-0000-000000000001",
        "resource_group": "so101-training", "location": "eastus", "zone": "1",
        "vm_name": "so101-gpu", "vm_size": "Standard_NC24ads_A100_v4",
        "image": "canonical:ubuntu-24_04-lts:server:24.04.20261001",
        "admin": "so101", "ssh_public_key": "/unused", "ssh_private_key": "/unused",
        "ssh_source_cidr": "203.0.113.1/32", "storage_account": "so101example123",
        "spot_max_usd_per_hour": 1.2, "max_session_hours": 2,
        "total_gpu_hours": 4, "disk_gib": 128, "run": "outputs/local-run",
    }


@pytest.mark.parametrize("key,value", [
    ("spot_max_usd_per_hour", -1), ("total_gpu_hours", float("nan")),
    ("ssh_source_cidr", "0.0.0.0/0"), ("image", "canonical:offer:sku:latest"),
    ("run", "../elsewhere"), ("vm_size", "eight-gpu-machine"),
])
def test_unsafe_cloud_config_rejected(tmp_path, key, value):
    payload = config() | {key: value}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        cloud.load_config(path)


def test_budget_reservation_survives_restart_and_refuses_excess(tmp_path):
    assert cloud.reserve_session(tmp_path, config(), 100) == 7300
    assert cloud.reserve_session(tmp_path, config(), 200) == 7400
    with pytest.raises(ValueError, match="exhausted"):
        cloud.reserve_session(tmp_path, config(), 300)
    saved = json.loads((tmp_path / "controller.json").read_text())
    assert saved["reserved_hours"] == 4
    assert len(saved["allocations"]) == 2


def test_active_local_run_prevents_cloud_start(tmp_path, monkeypatch):
    monkeypatch.setattr(cloud, "local_trainer_active", lambda _: True)
    monkeypatch.setattr(cloud, "az", lambda *a: pytest.fail("cloud API called before local check"))
    with pytest.raises(ValueError, match="Local trainer"):
        cloud.start(config(), tmp_path)


@pytest.mark.parametrize("automatic", [False, True])
def test_stale_first_handoff_cannot_start_even_on_automatic_retry(tmp_path, monkeypatch, automatic):
    monkeypatch.setattr(cloud, "ROOT", tmp_path)
    monkeypatch.setattr(cloud.time, "time", lambda: 100)
    monkeypatch.setattr(cloud, "local_trainer_active", lambda _: False)
    output = tmp_path / "outputs/local-run"
    output.mkdir(parents=True)
    (output / "latest_state.pt").write_bytes(b"final newer snapshot")
    cloud.reserve_session(tmp_path, config(), 100)
    monkeypatch.setattr(cloud, "az", lambda *a: [{"code": "PowerState/running"}])
    answers = iter([json.dumps({"initial_state_sha256": "stale"}), "False"])
    monkeypatch.setattr(cloud, "ssh", lambda *a, **k: next(answers))
    monkeypatch.setattr(cloud, "worker", lambda *a: pytest.fail("stale worker started"))
    with pytest.raises(ValueError, match="Stale initial snapshot"):
        cloud.start(config(), tmp_path, automatic=automatic)


def test_deploy_refuses_existing_resources_before_reservation(tmp_path, monkeypatch):
    public = tmp_path / "public"
    public.write_text("ssh-ed25519 AAAA test")
    settings = config() | {"ssh_public_key": str(public)}
    calls = []

    def fake_az(config, *args):
        calls.append(args)
        if args[:2] == ("group", "exists"):
            return True
        if args[:2] == ("resource", "list"):
            return [{"type": "Microsoft.Compute/disks"}]
        pytest.fail("Unexpected mutating cloud call")

    monkeypatch.setattr(cloud, "az", fake_az)
    with pytest.raises(ValueError, match="empty dedicated"):
        cloud.deploy(settings, tmp_path)
    assert not (tmp_path / "controller.json").exists()
    assert len(calls) == 2


def test_failed_deploy_can_retry_same_target_without_resetting_budget(tmp_path, monkeypatch):
    public = tmp_path / "public"
    public.write_text("ssh-ed25519 AAAA test")
    settings = config() | {"ssh_public_key": str(public)}
    attempts = []

    def fake_az(config, *args):
        if args[:2] == ("group", "exists"):
            return True
        if args[:2] == ("resource", "list"):
            return []
        if args[:3] == ("deployment", "group", "create"):
            attempts.append(args)
            if len(attempts) == 1:
                raise subprocess.CalledProcessError(1, ["az"], stderr="AllocationFailed")
        return {}

    monkeypatch.setattr(cloud, "az", fake_az)
    with pytest.raises(subprocess.CalledProcessError):
        cloud.deploy(settings, tmp_path)
    cloud.deploy(settings, tmp_path)
    saved = json.loads((tmp_path / "controller.json").read_text())
    assert saved["reserved_hours"] == 4 and len(attempts) == 2
    assert saved["deployment_complete"] and saved["desired"] == "paused"


def test_capacity_retry_reports_waiting_and_backs_off_without_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(cloud, "ROOT", tmp_path)
    monkeypatch.setattr(cloud.time, "time", lambda: 100)
    cloud.reserve_session(tmp_path, config(), 100)
    monkeypatch.setattr(cloud, "az", lambda *a: [{"code": "PowerState/deallocated"}])

    def failed(*args, **kwargs):
        raise subprocess.CalledProcessError(1, ["az", "vm", "start"], stderr="AllocationFailed")

    monkeypatch.setattr(cloud, "start", failed)
    sleeps = []

    def stop_after_one(delay):
        sleeps.append(delay)
        raise StopIteration

    monkeypatch.setattr(cloud.time, "sleep", stop_after_one)
    with pytest.raises(StopIteration):
        cloud.watch(config(), tmp_path, interval=30)
    assert sleeps == [60]
    assert json.loads((cloud.mirror_directory(config()) / "remote_status.json").read_text())["state"] == "waiting_for_capacity"


def test_archive_rejects_traversal_and_symlinks(tmp_path):
    for name, symlink in (("../outside", False), ("safe", True)):
        bundle = tmp_path / "bad.tar.gz"
        with tarfile.open(bundle, "w:gz") as archive:
            member = tarfile.TarInfo(name)
            if symlink:
                member.type = tarfile.SYMTYPE
                member.linkname = "/etc/passwd"
            archive.addfile(member, io.BytesIO())
        with pytest.raises(ValueError):
            cloud.validate_archive(bundle)


def test_asset_closure_excludes_images_and_unrelated_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(cloud, "ROOT", tmp_path)
    code = tmp_path / "src/so101"
    code.mkdir(parents=True)
    (code / "__init__.py").write_text("")
    (code / "cloud_worker.py").write_text("# worker")
    assets = code / "dashboard"
    assets.mkdir()
    (assets / "index.html").write_text("<html></html>")
    source = tmp_path / "data/episode"
    source.mkdir(parents=True)
    for name in ("episode.json", "frames.jsonl", "camera.png"):
        (source / name).write_text("{}")
    collection = tmp_path / "data/collection"
    collection.mkdir()
    (collection / "collection.json").write_text(json.dumps({"successful_episodes": ["data/episode"]}))
    checkpoint = tmp_path / "outputs/base"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.pt").write_bytes(b"base")
    files = cloud.stage_files(
        {"checkpoint": "outputs/base", "collection": "data/collection"},
        {"training": {"view": ["data/episode"]}, "validation": []},
    )
    assert Path("data/episode/frames.jsonl") in files
    assert Path("outputs/base/model.pt") in files
    assert Path("src/so101/cloud_worker.py") in files
    assert Path("src/so101/dashboard/index.html") in files
    assert not any(path.suffix == ".png" for path in files)
    assert not any(path.name == ".env" for path in files)


def test_recipe_preserved_except_explicit_cloud_resume_flags():
    recipe = {"seed": 61000, "learning_rate": 1e-4, "smoothness_advantage_weight": 1.,
              "device": "mps", "steps": 5000, "rl_estimator": "denoising-grpo",
              "resume": True, "refresh_corpus": False, "save_final_checkpoint": False,
              "record_rollouts": False}
    command = worker.trainer_command({"python": "/cuda/python", "recipe": recipe})
    assert command[:4] == ["/cuda/python", "-u", "-m", "so101.train_trajectory_rl"]
    assert command[command.index("--device") + 1] == "cuda"
    assert command[command.index("--learning-rate") + 1] == "0.0001"
    assert command[command.index("--smoothness-advantage-weight") + 1] == "1.0"
    assert "--allow-device-rng-migration" in command
    assert "--refresh-corpus" not in command
    assert "--no-save-final-checkpoint" in command
    assert "--no-record-rollouts" in command


def test_cloud_mirror_copies_split_and_bounded_policy_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(cloud, "ROOT", tmp_path)
    observed = {"state": "paused", "log_revision": "new", "split_available": True,
                "evidence_available": True, "rollout_retention_steps": 1}
    monkeypatch.setattr(cloud, "worker", lambda *args: json.dumps(observed))
    transfers = []
    def transfer(config, directory, source, destination):
        transfers.append(source)
        path = Path(destination)
        if source.endswith("/trajectory_rl.json"):
            path.write_text('{"steps":[{"step":3}]}')
        elif source.endswith("/data_split.json"):
            path.write_text('{"training":{"grasp":["source"]},"validation":["heldout"]}')
        elif source.endswith("/rollouts/"):
            for step in (2, 3):
                folder = path / f"step_{step:06}"
                folder.mkdir(parents=True, exist_ok=True)
                (folder / "group_0_0.json").write_text('{"frames":[]}')
        else:
            pytest.fail(source)
    monkeypatch.setattr(cloud, "transfer", transfer)
    cloud.sync_locked(config(), tmp_path)
    mirror = cloud.mirror_directory(config())
    assert json.loads((mirror / "data_split.json").read_text())["validation"] == ["heldout"]
    assert [folder.name for folder in (mirror / "rollouts").iterdir()] == ["step_000003"]
    cloud.sync_locked(config(), tmp_path)
    assert len(transfers) == 3


def test_stop_intent_does_not_remove_unowned_stop(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    (tmp_path / "intent.json").write_text('{"desired":"running"}')
    (output / "STOP").write_text("user stop")
    worker.request_stop(tmp_path, output, "eviction")
    with pytest.raises(ValueError, match="Unowned"):
        worker.acknowledge_stop(tmp_path, output)
    assert (output / "STOP").read_text() == "user stop"


def test_compute_guard_enforces_deadline_without_controller(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    (tmp_path / "worker.json").write_text(json.dumps({"root": str(tmp_path), "output": "output"}))
    (tmp_path / "intent.json").write_text('{"desired":"running"}')
    monkeypatch.setattr(worker, "lease", lambda: (1, {"name": "one-vm"}))
    seen = []
    monkeypatch.setattr(worker, "deallocate", seen.append)
    worker.guard(tmp_path)
    assert seen == [{"name": "one-vm"}]
    assert json.loads((tmp_path / "intent.json").read_text())["desired"] == "paused"
    assert (output / "STOP").read_text() == "cloud:budget\n"


def test_backup_publishes_manifest_only_after_checkpoint_and_dependencies(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    state, _, _ = small_state(output)
    (output / "cloud_identity.json").write_text("{}")
    checkpoints.publish(output, state, retain=3)
    config = {"root": str(tmp_path), "output": "output",
              "run_name": "run", "storage_account": "example"}
    seen = []
    monkeypatch.setattr(worker, "put_blob", lambda account, container, name, path: seen.append(name))
    worker.backup_latest(tmp_path, config)
    assert [name.rsplit(".", 1)[-1] for name in seen] == ["pt", "json", "json", "json"]
    assert seen[-1].endswith("step_000000007.json")
    assert json.loads((tmp_path / "backup.json").read_text())["step"] == 7
    assert not (tmp_path / "backup.pt").exists()


def test_backup_failure_never_commits_or_advances_backed_up_step(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    state, _, _ = small_state(output)
    checkpoints.publish(output, state, retain=3)

    def failed(*args):
        raise OSError("upload interrupted")

    monkeypatch.setattr(worker, "put_blob", failed)
    with pytest.raises(OSError, match="interrupted"):
        worker.backup_latest(tmp_path, {
            "root": str(tmp_path), "output": "output",
            "run_name": "run", "storage_account": "example",
        })
    assert not (tmp_path / "backup.json").exists()
    assert not (tmp_path / "backup.pt").exists()


def test_failed_pause_still_persists_local_pause_intent(tmp_path, monkeypatch):
    cloud.reserve_session(tmp_path, config(), 100)

    def failed(*args):
        raise subprocess.CalledProcessError(255, ["ssh"])

    monkeypatch.setattr(cloud, "worker", failed)
    with pytest.raises(subprocess.CalledProcessError):
        cloud.pause(config(), tmp_path)
    assert json.loads((tmp_path / "controller.json").read_text())["desired"] == "paused"


def test_eviction_cannot_undo_pause_or_claim_a_different_stop_marker(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    (tmp_path / "intent.json").write_text('{"desired":"running"}')
    worker.request_stop(tmp_path, output, "pause")
    worker.request_stop(tmp_path, output, "eviction")
    assert json.loads((tmp_path / "intent.json").read_text())["desired"] == "paused"
    worker.request_stop(tmp_path, output, "budget")
    worker.acknowledge_stop(tmp_path, output, explicit=True)
    assert not (output / "STOP").exists()
    assert json.loads((tmp_path / "intent.json").read_text())["desired"] == "running"


def test_pause_during_preflight_cannot_be_implicitly_acknowledged(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    (tmp_path / "intent.json").write_text('{"desired":"running"}')
    worker.request_stop(tmp_path, output, "pause")
    with pytest.raises(ValueError, match="Pause requested"):
        worker.acknowledge_stop(tmp_path, output)


def test_prepare_uses_atomic_snapshot_log_and_preserves_asset_closure(tmp_path, monkeypatch):
    monkeypatch.setattr(cloud, "ROOT", tmp_path)
    monkeypatch.setattr(cloud, "local_trainer_active", lambda _: True)
    monkeypatch.setattr(cloud, "cached_model_assets", lambda *a: ({}, {}))
    output = tmp_path / "outputs/local-run"
    output.mkdir(parents=True)
    state, _, _ = small_state(output)
    state["log"].update(
        training_objective="reward-only",
        run_config={"checkpoint": "outputs/base", "collection": "data/collection",
                    "rl_estimator": "denoising-grpo", "seed": 61000, "steps": 5000},
    )
    torch.save(state, output / "latest_state.pt")
    # This later, non-atomic plot file must not override the snapshot's history.
    (output / "trajectory_rl.json").write_text('{"steps":[{"step":999}]}')
    base = tmp_path / "outputs/base"
    base.mkdir()
    (base / "model.pt").write_bytes(b"not loaded")
    collection = tmp_path / "data/collection"
    collection.mkdir(parents=True)
    (collection / "collection.json").write_text('{"successful_episodes":["data/episode"]}')
    episode = tmp_path / "data/episode"
    episode.mkdir()
    (episode / "episode.json").write_text("{}")
    (episode / "frames.jsonl").write_text("{}\n")
    (episode / "image.png").write_bytes(b"excluded")
    (tmp_path / "uv.lock").write_text("")
    (tmp_path / "pyproject.toml").write_text("")
    # This fixture only needs validation sources, not actual corpus reconstruction.
    (output / "data_split.json").write_text('{"training":{},"validation":["data/episode"]}')
    bundle = tmp_path / "bundle.tar.gz"
    cloud.prepare(config(), bundle)
    cloud.validate_archive(bundle)
    with tarfile.open(bundle) as archive:
        log = json.load(archive.extractfile("outputs/local-run/trajectory_rl.json"))
        manifest = json.load(archive.extractfile("cloud_manifest.json"))
        assert log["steps"][-1]["step"] == 7
        assert manifest["step"] == 7
        assert "data/episode/frames.jsonl" in manifest["files"]
        assert not any(name.endswith(".png") for name in archive.getnames())
    assert json.loads(bundle.with_suffix(".gz.json").read_text())["live_local_trainer"]


def test_model_cache_staging_pins_only_required_snapshots_not_tokens(tmp_path, monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(cloud, "ROOT", tmp_path)
    base = tmp_path / "outputs/base"
    base.mkdir(parents=True)
    (base / "metadata.json").write_text('{"source_checkpoint":"owner/policy"}')
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "token").write_text("must not be staged")
    configs = {}
    for repo, revision in (("owner/policy", "a" * 40), ("owner/vlm", "b" * 40)):
        snapshot = cache / repo / "snapshots" / revision
        snapshot.mkdir(parents=True)
        path = snapshot / "config.json"
        path.write_text('{"vlm_model_name":"owner/vlm"}' if repo.endswith("policy") else "{}")
        (snapshot / "weights.safetensors").write_bytes(b"weights")
        configs[repo] = str(path)
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda repo, name: configs[repo])
    assets, revisions = cloud.cached_model_assets({"checkpoint": "outputs/base"}, tmp_path)
    assert revisions == {"owner/policy": "a" * 40, "owner/vlm": "b" * 40}
    assert len(assets) == 6
    assert not any(path.name == "token" for path in assets)
    assert assets[Path("cloud_hf/hub/models--owner--policy/refs/main")].read_text() == "a" * 40


def test_worker_round_trip_supervision_and_final_backup_without_policy(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    control = tmp_path / "control"
    control.mkdir()
    state, _, _ = small_state(output)
    manifest = {"files": {}, "initial_state_sha256": checkpoints.sha256(output / "latest_state.pt")}
    checkpoints.atomic_json(tmp_path / "cloud_manifest.json", manifest)
    checkpoints.atomic_json(output / "cloud_identity.json", {
        "manifest_sha256": checkpoints.sha256(tmp_path / "cloud_manifest.json"),
    })
    checkpoints.atomic_json(control / "worker.json", {
        "root": str(tmp_path), "python": "/cuda/python", "output": "output",
        "recipe": {"seed": 61000, "steps": 8, "learning_rate": 1e-4},
        "run_name": "run", "storage_account": "example",
    })
    checkpoints.atomic_json(control / "intent.json", {"desired": "running"})
    monkeypatch.setattr(worker, "lease", lambda: (float("inf"), {"name": "one-vm"}))
    monkeypatch.setattr(worker, "metadata", lambda _: {"Events": []})
    monkeypatch.setattr(worker, "backup_loop", lambda *a: None)
    monkeypatch.setattr(worker.time, "sleep", lambda _: None)
    uploads = []
    monkeypatch.setattr(worker, "put_blob", lambda account, container, name, path: uploads.append(name))
    commands = []

    class Child:
        pid = 123
        returncode = 0
        first = True

        def poll(self):
            if self.first:
                self.first = False
                state["step"] = 8
                state["log"]["steps"].append({"step": 8})
                checkpoints.publish(output, state, retain=3)
                return None
            return 0

    def popen(command, **kwargs):
        commands.append(command)
        assert kwargs["cwd"] == tmp_path
        return Child()

    monkeypatch.setattr(worker.subprocess, "Popen", popen)
    worker.run_worker(control)
    status = json.loads((control / "status.json").read_text())
    assert status["state"] == "completed" and status["step"] == 8
    assert status["backup"]["step"] == 8
    assert json.loads((control / "backup.json").read_text())["step"] == 8
    assert uploads[-1].endswith("step_000000008.json")
    assert commands[0][commands[0].index("--device") + 1] == "cuda"


def test_arm_contract_uses_spot_retained_storage_and_restricted_ssh():
    template = json.loads((Path(__file__).resolve().parents[1] / "cloud/azure-resources.json").read_text())
    resources = {resource["type"]: resource for resource in template["resources"]}
    vm = resources["Microsoft.Compute/virtualMachines"]["properties"]
    assert vm["priority"] == "Spot"
    assert vm["evictionPolicy"] == "Deallocate"
    assert vm["storageProfile"]["dataDisks"][0]["deleteOption"] == "Detach"
    assert vm["billingProfile"]["maxPrice"] == "[json(parameters('spotMax'))]"
    rule = resources["Microsoft.Network/networkSecurityGroups"]["properties"]["securityRules"][0]["properties"]
    assert rule["destinationPortRange"] == "22"
    assert rule["sourceAddressPrefix"] == "[parameters('sshCidr')]"
