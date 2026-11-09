"""Extend a completed constant-plateau run without resetting optimizer or RNG."""
import argparse
import copy
from pathlib import Path

import torch

from so101.cloud_checkpoint import atomic_json, sha256, validate_state
from so101.train_entity_sim import scheduled_learning_rate


def prepare(source, destination, steps):
    source_digest = sha256(source)
    state = torch.load(source, map_location="cpu", weights_only=False)
    validate_state(state)
    config = state["log"]["run_config"]
    if (config["batch_size"] != 64 or config["learning_rate"] != 3e-5
            or config["learning_rate_schedule"] != "warmup-constant"
            or config["device"] != "cuda"
            or not config.get("gradient_accumulation")
            or config.get("micro_batch_size") != 8
            or steps <= state["step"] or state["step"] != config["steps"]):
        raise ValueError("Continuation requires completed B64/LR3e-5 constant-plateau training")
    previous = copy.deepcopy(config)
    warmup_updates = min(config["steps"] - 1, max(1, round(
        config["steps"] * config["warmup_fraction"])))
    config["steps"] = steps
    config["warmup_fraction"] = warmup_updates / steps
    output = "outputs/entity_smolvla_sft_long_b64_lr3e-05"
    config["output"] = output
    config["training_state"] = f"{output}/training_state.pt"
    for step in (state["step"], state["step"] + 1, steps):
        if scheduled_learning_rate(step, steps, 3e-5, "warmup-constant",
                                   config["warmup_fraction"]) != 3e-5:
            raise ValueError("Extension would rewarm or decay the plateau learning rate")
    state["log"].setdefault("continuations", []).append({
        "from_step": state["step"], "source_sha256": source_digest,
        "previous_run_config": previous, "new_run_config": copy.deepcopy(config),
        "preserved": "Exact model, full Adam state, RNG, training/evaluation history, and frozen data",
        "warmup_updates": warmup_updates,
    })
    state["log"]["status"] = "training"
    if sha256(source) != source_digest:
        raise ValueError("Source checkpoint changed during continuation preparation")
    destination.mkdir(parents=True, exist_ok=False)
    torch.save(state, destination / "training_state.pt")
    receipt = {
        "from_step": state["step"], "steps": steps, "warmup_updates": warmup_updates,
        "warmup_fraction": config["warmup_fraction"], "output": output,
        "batch_size": 64, "learning_rate": 3e-5, "micro_batch_size": 8,
        "validation_interval": config["validation_interval"],
        "source_sha256": source_digest,
    }
    atomic_json(destination / "continuation.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=16384)
    args = parser.parse_args()
    prepare(args.source, args.destination, args.steps)


if __name__ == "__main__":
    main()
