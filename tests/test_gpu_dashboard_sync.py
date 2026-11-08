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
