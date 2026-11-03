"""CPU validation inside the pinned runtime before worker checkpoint publication."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import random

import numpy as np
import torch

from so101.cloud_checkpoint import sha256, validate_state
from so101.resume_rng import plan_device_rng
from so101.train_entity_sim import resume_training_log


def check(state_path: Path, original_path: Path) -> dict:
    original = torch.load(original_path, map_location="cpu", weights_only=False)
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    validate_state(original, allow_untrained=True)
    validate_state(state, allow_untrained=True)
    expected = state["log"]["run_config"]
    if (expected.get("device") != "cuda"
            or expected.get("validation_interval") != original["log"]["run_config"].get("validation_interval")):
        raise ValueError("Published SFT device or evaluation schedule differs from approved recovery")
    resume_training_log(deepcopy(original["log"]), expected, original["step"],
                        allow_device_migration=True)
    if (type(expected.get("steps")) is not int
            or not original["step"] <= state["step"] <= expected["steps"]):
        raise ValueError("Published checkpoint update is outside the original trial")
    if [row.get("step") for row in state["log"]["steps"]] != list(range(1, state["step"] + 1)):
        raise ValueError("Published checkpoint update history is not continuous")
    if state["log"]["steps"][:original["step"]] != original["log"]["steps"]:
        raise ValueError("Published checkpoint rewrote original update history")
    plan_device_rng(state, "cuda", False, None)
    rng = state["rng"]
    random.Random().setstate(rng["python"])
    np.random.RandomState().set_state(rng["numpy"])
    np.random.default_rng().bit_generator.state = rng["sampling"]
    torch.Generator().set_state(rng["torch"])
    if (not isinstance(rng["device"], torch.Tensor) or rng["device"].dtype != torch.uint8
            or rng["device"].ndim != 1 or rng["device"].numel() == 0):
        raise ValueError("Missing valid CUDA RNG bytes")
    return {"step": state["step"], "sha256": sha256(state_path),
            "size": state_path.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.state, args.original)))


if __name__ == "__main__":
    main()
