"""CPU validation inside the pinned runtime before worker checkpoint publication."""
from __future__ import annotations

import argparse
from copy import deepcopy
from contextlib import redirect_stdout
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

from so101.cloud_checkpoint import sha256, validate_state
from so101.resume_rng import plan_device_rng
from so101.train_entity_sim import resume_training_log
from so101.sweep_entity_sft import verify_log


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


def check_exports(output: Path, original_path: Path) -> dict:
    # Loading the public policy is needed only for standalone export validation.
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from so101.control_model.modeling_entity_smolvla import EntityModelConfig, attach_entity_modules

    original = torch.load(original_path, map_location="cpu", weights_only=False)
    validate_state(original, allow_untrained=True)
    if original["step"] != 0:
        raise ValueError("Fresh export requires its genuinely untrained initial state")
    reference = original["log"]
    del original
    trial = {key: reference["run_config"][key]
             for key in ("output", "steps", "batch_size", "learning_rate", "validation_interval")}
    if (trial["steps"] * trial["batch_size"] != 24576
            or trial["steps"] != 4 * trial["validation_interval"]):
        raise ValueError("Export must preserve the 24,576-sample budget and four quarters")
    check(output / "training_state.pt", original_path)
    state = torch.load(output / "training_state.pt", map_location="cpu", weights_only=False)
    log = state["log"]
    config = log["run_config"]
    if state["step"] != trial["steps"]:
        raise ValueError("Exported optimizer has not completed the declared trial")
    verify_log(log, trial, reference)
    if log.get("supervised_probe") != reference.get("supervised_probe"):
        raise ValueError("Exported held-out frame population differs from initial evaluation")
    if json.loads((output / "sft_training.json").read_text()) != log:
        raise ValueError("Exported log differs from full-state history")
    entity_config = EntityModelConfig(history_steps=config["history_steps"],
                                     depth_input=config["depth_mode"] != "blind")
    if json.loads((output / "entity_config.json").read_text()) != asdict(entity_config):
        raise ValueError("Standalone entity configuration differs from training")
    metadata = json.loads((output / "metadata.json").read_text())
    if (metadata.get("source_checkpoint") != config["pretrained"]
            or metadata.get("normalization") != log["dataset"]["normalization"]):
        raise ValueError("Standalone base or normalization differs from training")
    policy = SmolVLAPolicy.from_pretrained(config["pretrained"])
    attach_entity_modules(policy, entity_config)
    layout = {name: value.shape for name, value in policy.state_dict().items()}
    frozen = {name: value for name, value in policy.state_dict().items()
              if name not in state["model"]}
    del policy
    model = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
    if (model.keys() != layout.keys()
            or any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all()
                   or value.shape != layout[name] for name, value in model.items())
            or any(not torch.equal(value, model[name]) for name, value in frozen.items())
            or any(not torch.equal(value, model[name]) for name, value in state["model"].items())):
        raise ValueError("Standalone model differs from exact trained/frozen parameters")
    expert = [value for name, value in model.items()
              if name.startswith("model.vlm_with_expert.lm_expert.")]
    if (not expert or metadata.get("action_expert_dtype") !=
            str(expert[0].dtype).removeprefix("torch.")):
        raise ValueError("Action expert dtype differs from metadata")
    del model, expert, frozen
    if json.loads((output / "data_split.json").read_text()).get("validation") != config["validation_sources"]:
        raise ValueError("Standalone split differs from frozen validation problems")
    names = ["training_state.pt", "model.pt", "entity_config.json", "metadata.json",
             "sft_training.json", "data_split.json"]
    for step in range(0, trial["steps"] + 1, trial["validation_interval"]):
        name = f"checkpoints/training_state_{step:06d}.pt"
        receipt = check(output / name, original_path)
        retained = torch.load(output / name, map_location="cpu", weights_only=False)
        if (receipt["step"] != step or retained["log"]["steps"] != log["steps"][:step]
                or retained["log"]["validation"] !=
                [row for row in log["validation"] if row["step"] <= step]
                or retained["log"].get("supervised_probe") != log["supervised_probe"]):
            raise ValueError("Retained quarter differs from authentic final history/probe")
        del retained
        names.append(name)
    return {name: {"size": (output / name).stat().st_size, "sha256": sha256(output / name)}
            for name in names}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--state", type=Path)
    target.add_argument("--exports", type=Path)
    parser.add_argument("--original", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    with redirect_stdout(sys.stderr):
        result = (check_exports(args.exports, args.original) if args.exports is not None
                  else check(args.state, args.original))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
