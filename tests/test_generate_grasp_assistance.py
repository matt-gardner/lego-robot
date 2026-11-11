import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location(
    "generate_grasp_assistance", Path(__file__).parents[1] / "scripts/generate_grasp_assistance.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value).encode()
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def test_inventory_includes_unselected_admissions_and_deduplicates_originals(tmp_path):
    source = tmp_path / "source"
    scene = {
        "scene_id": "scene", "physics_version": "3.17.0",
        "tasks": [{"task_id": "original", "task_type": "grasp"},
                  {"task_id": "view", "task_type": "view"},
                  {"task_id": "old-child", "task_type": "grasp", "parent_task_id": "original"}],
    }
    digest = write_json(source / "problem.json", scene)
    entry = {"root": str(source), "scene_id": "scene", "split": "train",
             "files_sha256": {"problem.json": digest}}
    manifest = tmp_path / "pool.json"
    pool_digest = write_json(manifest, {"scenes": [entry]})
    pointer = tmp_path / "pointer.json"
    write_json(pointer, {"pools": [{"manifest_path": str(manifest), "sha256": pool_digest}]})
    extra = tmp_path / "extra"
    write_json(extra / "admission.json", entry)
    plan = runner.inventory(pointer, [extra])
    assert plan["task_count"] == 1
    assert plan["scene_count"] == 1
    assert plan["jobs"][0]["task_id"] == "original"
    assert {source["kind"] for source in plan["sources"]} == {"selected-pool", "additional-admission"}
    write_json(manifest, {"scenes": []})
    with pytest.raises(ValueError, match="Source hash changed"):
        runner.inventory(pointer, [])


def test_unavailable_parent_has_durable_receipt_and_is_not_retried(tmp_path, monkeypatch):
    for name in ("receipts", "logs", "entries", "families"):
        (tmp_path / name).mkdir()
    calls = []

    def unavailable(command, **kwargs):
        calls.append(command)
        kwargs["stdout"].write(
            b"UnavailableGrasp: No original-pose GRASP families constructed\n")
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(runner.subprocess, "run", unavailable)
    job = {"job_id": "job", "task_id": "original", "entry": {}}
    result = runner.run_job(tmp_path, job, 10)
    assert result["status"] == "unavailable"
    assert runner.run_job(tmp_path, job, 10) == result
    assert len(calls) == 1
    summary = runner.summarize(tmp_path, {"jobs": [job], "task_count": 1,
                                        "scene_count": 1, "split_counts": {"train": 1}}, "complete")
    assert summary["finished_parents"] == 1
    assert summary["parent_outcomes"] == {"unavailable": 1}
    assert summary["published_children"] == 0


def test_timeout_is_not_reported_as_geometric_unavailability(tmp_path, monkeypatch):
    for name in ("receipts", "logs", "entries", "families"):
        (tmp_path / name).mkdir()

    def timeout(*args, **kwargs):
        raise runner.subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(runner.subprocess, "run", timeout)
    result = runner.run_job(tmp_path, {"job_id": "job", "task_id": "original", "entry": {}}, 10)
    assert result["status"] == "timeout"
    assert "not a claim of task infeasibility" in result["reason"]
