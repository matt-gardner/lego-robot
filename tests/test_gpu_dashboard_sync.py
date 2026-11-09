import importlib.util
import json
from pathlib import Path
import time

import pytest


spec = importlib.util.spec_from_file_location(
    "gpu_dashboard_sync", Path(__file__).parents[1] / "scripts/sync_gpu_dashboard.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("name,container", [
    ("entity_smolvla_sft_b64_lr1e-04", "b64-lr1e-04"),
    ("entity_smolvla_sft_b64_lr3e-05", "b64-lr3e5"),
    ("entity_smolvla_sft_b64_lr3e-05", "b64-lr3e-05"),
])
def test_dynamic_and_original_container_names_remain_active(tmp_path, name, container):
    module.sync({
        "observed_at": time.time(), "running": [f"lego-sweep-job-{container}"],
        "logs": {name: {"status": "training", "steps": [{"step": 1}]}},
    }, tmp_path, "job")
    status = json.loads((tmp_path / name / "remote_status.json").read_text())
    assert status["state"] == "running"
    assert status["phase"] == "training"
    assert status["error"] is None


@pytest.mark.parametrize("running,state", [(["long-container"], "running"),
                                          (["long-container-other"], "failed")])
def test_long_run_uses_new_instance_and_exact_container_identity(tmp_path, running, state):
    name = "entity_smolvla_sft_long_b64_lr3e-05"
    module.sync({
        "observed_at": time.time(), "running": running,
        "logs": {name: {"status": "training", "steps": [{"step": 385}]}},
    }, tmp_path, "new-job", instance_id="new-instance", container_name="long-container")
    status = json.loads((tmp_path / name / "remote_status.json").read_text())
    assert status["state"] == state
    assert status["instance_id"] == "new-instance"
    assert status["job_id"] == "new-job"
