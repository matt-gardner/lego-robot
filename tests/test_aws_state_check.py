from copy import deepcopy
import random

import numpy as np
import pytest
import torch

from so101.aws_state_check import check


def states(tmp_path):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(1, 1)).sum().backward()
    optimizer.step()
    original = {
        "step": 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "log": {"steps": [{"step": 1}], "run_config": {
            "device": "mps", "steps": 2, "validation_interval": 1}},
        "rng": {"backend": "mps", "device": torch.zeros(8, dtype=torch.uint8),
                "python": random.getstate(), "numpy": np.random.get_state(),
                "sampling": np.random.default_rng(1).bit_generator.state,
                "torch": torch.get_rng_state()},
    }
    candidate = deepcopy(original)
    candidate["log"]["run_config"]["device"] = "cuda"
    candidate["rng"]["backend"] = "cuda"
    source, target = tmp_path / "original.pt", tmp_path / "candidate.pt"
    torch.save(original, source)
    torch.save(candidate, target)
    return source, target, candidate


def test_cpu_validation_accepts_consistent_migrated_full_state(tmp_path):
    source, target, candidate = states(tmp_path)
    result = check(target, source)
    assert result["step"] == 1 and result["size"] == target.stat().st_size
    assert len(result["sha256"]) == 64


@pytest.mark.parametrize("failure", ["recipe", "history", "nonfinite", "rng", "foreign"])
def test_bad_recipe_state_history_or_rng_is_rejected(tmp_path, failure):
    source, target, candidate = states(tmp_path)
    if failure == "recipe":
        candidate["log"]["run_config"]["steps"] = 3
    elif failure == "history":
        candidate["log"]["steps"][0]["changed"] = True
    elif failure == "nonfinite":
        next(iter(candidate["model"].values())).fill_(float("nan"))
    elif failure == "rng":
        candidate["rng"].pop("sampling")
    else:
        candidate["rng"]["backend"] = "mps"
    torch.save(candidate, target)
    with pytest.raises((ValueError, KeyError)):
        check(target, source)
