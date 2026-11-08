import json
import sqlite3
import threading
import time
from unittest.mock import Mock

import pytest

from so101 import control_workflow


@pytest.fixture
def broker(monkeypatch):
    current = Mock()
    current.lock = threading.RLock()
    current.db = sqlite3.connect(":memory:")
    current.db.row_factory = sqlite3.Row
    current.job.return_value = {
        "id": "a" * 32, "digest": "b" * 64, "instance": "i-abc",
        "status": "idle", "deadline": time.time() + 600,
        "spec": json.dumps({"log_group": "/so101/worker"}),
    }
    current.owned_instance.return_value = {"State": {"Name": "running"}}
    monkeypatch.setattr(control_workflow, "trusted_json", lambda _: {
        "b" * 64: {
            "spec_id": "approved",
            "steps": {"provision": {"document": "so101-worker-provision", "version": "1",
                                    "parameters": {"SpecSHA256": ["b" * 64]}}},
        },
    })
    yield current
    current.db.close()


def test_preparation_retries_observation_not_accepted_dispatch(broker):
    broker.aws.side_effect = [
        {"Command": {"CommandId": "accepted"}},
        {"Status": "InProgress"},
        {"Status": "Success", "StandardOutputContent": "done"},
    ]
    assert control_workflow.prepare(broker, "owner", "a" * 32, "provision")["status"] == "InProgress"
    assert control_workflow.prepare(broker, "owner", "a" * 32, "provision")["status"] == "Success"
    assert sum(call.args[1] == "send-command" for call in broker.aws.call_args_list) == 1


def test_response_loss_does_not_replay_dispatch(broker):
    broker.aws.side_effect = TimeoutError("response lost")
    with pytest.raises(TimeoutError):
        control_workflow.prepare(broker, "owner", "a" * 32, "provision")
    broker.aws.side_effect = None
    broker.aws.return_value = {"Commands": []}
    result = control_workflow.prepare(broker, "owner", "a" * 32, "provision")
    assert result["status"] == "DispatchUncertain"
    assert broker.aws.call_args.args[1] == "list-commands"


def test_other_owners_are_rejected_before_preparation(broker):
    broker.job.side_effect = PermissionError("job belongs to another session")
    with pytest.raises(PermissionError):
        control_workflow.prepare(broker, "other-owner", "a" * 32, "provision")
    broker.aws.assert_not_called()
