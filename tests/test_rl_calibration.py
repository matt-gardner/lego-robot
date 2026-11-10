import pytest
import torch

from so101.cloud_checkpoint import validate_state
from so101.rl_calibration import AssistanceCalibration


def test_task_ema_shares_feedback_across_problems_but_not_tasks_and_resumes():
    state = {}
    calibration = AssistanceCalibration(state, alpha=.5, increment=.1)
    assert calibration.start("grasp", .5, .95) == .5
    assistance, outcome = calibration.record("grasp", .5, 4, 4, .95)
    assert assistance == .4 and outcome == "all_passed"
    assert calibration.start("grasp", .9, .95) == pytest.approx(.45)
    assistance, outcome = calibration.record("grasp", assistance, 0, 4, .95)
    assert assistance == .5 and outcome == "all_failed"
    assert calibration.start("view", 0., 0.) == 0.
    resumed = AssistanceCalibration(state, alpha=.5, increment=.1)
    assert resumed.start("grasp", .9, .95) == pytest.approx(.475)
    assert resumed.record("grasp", .5, 2, 4, .95) == (.5, "mixed")
    assert state["tasks"]["grasp"]["groups"] == 3
    assert state["tasks"]["grasp"]["mixed"] == 1
    assert resumed.start("grasp", 0., 0.) == 0.
    with pytest.raises(ValueError, match="exactly four"):
        resumed.record("grasp", .5, 2, 8, .95)
    with pytest.raises(ValueError, match="Resume"):
        AssistanceCalibration(state, alpha=.3, increment=.1)


@pytest.mark.parametrize("invalid", [None, "updated", "missing_flag", "missing_groups", "missing_state"])
def test_empty_adam_requires_explicitly_skipped_complete_history(invalid):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    state = {
        "step": 2, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": {},
        "log": {"steps": [
            {"step": 1, "optimizer_update_applied": False},
            {"step": 2, "optimizer_update_applied": False},
        ]},
    }
    if invalid == "updated":
        state["log"]["steps"][0]["optimizer_update_applied"] = True
    elif invalid == "missing_flag":
        del state["log"]["steps"][0]["optimizer_update_applied"]
    elif invalid == "missing_groups":
        state["optimizer"]["param_groups"] = []
    elif invalid == "missing_state":
        del state["optimizer"]["state"]
    if invalid is None:
        validate_state(state)
    else:
        with pytest.raises(ValueError, match="Missing trained model or Adam state"):
            validate_state(state)
