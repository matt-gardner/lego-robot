"""Prepare an LR/batch trial from an evaluated, genuinely untrained checkpoint."""
import argparse
import json
import math
from pathlib import Path

import torch

from so101.cloud_checkpoint import atomic_json, sha256, validate_state


def prepare(reference, directory, batch_size, learning_rate, repair_existing=False):
    if (batch_size <= 0 or batch_size % 2 or 24576 % batch_size
            or (24576 // batch_size) % 4
            or not math.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("Trial requires an even batch, four sample-matched quarters, and a positive finite LR")
    source_digest = sha256(reference)
    state = torch.load(reference, map_location="cpu", weights_only=False)
    validate_state(state, allow_untrained=True)
    log = state["log"]
    if (state["step"] != 0 or state["optimizer"]["state"] or log["steps"]
            or [row["step"] for row in log["validation"]] != [0]
            or log["supervised_probe"]["samples"] != 160
            or len(log["validation"][0]["rollouts"]) != 40):
        raise ValueError("Reference must be evaluated at step zero with no optimizer updates or Adam moments")
    if log["run_config"]["device"] != "cuda":
        raise ValueError("Cached baseline must use the same CUDA evaluation runtime")
    if sha256(reference) != source_digest:
        raise ValueError("Untrained baseline changed during preparation")
    name = f"entity_smolvla_sft_b{batch_size}_lr{learning_rate:.0e}"
    output = f"outputs/entity_smolvla_sft_lr_sweep/{name}"
    steps = 24576 // batch_size
    config = log["run_config"]
    config.update({
        "output": output, "steps": steps, "batch_size": batch_size,
        "learning_rate": learning_rate, "validation_interval": steps // 4,
        "training_state": f"{output}/training_state.pt",
        "gradient_accumulation": True, "micro_batch_size": 8,
    })
    for group in state["optimizer"]["param_groups"]:
        group["lr"] = learning_rate
    log["micro_batch_size"] = min(batch_size, 8)
    log["optimizer_batch_size"] = batch_size
    log["status"] = "initialized"
    baseline_seconds = log["validation"][0]["evaluation_seconds"]
    log["validation"][0]["evaluation_seconds"] = 0
    log["validation"][0]["cached_initial_evaluation"] = {
        "source_sha256": source_digest,
        "source_evaluation_seconds": baseline_seconds,
        "reason": "Exact untrained model/RNG and frozen evaluation retained; only LR/batch/schedule length differ.",
    }
    log["adaptive_initialization"] = {
        "source": str(reference), "source_sha256": source_digest,
        "source_step": 0, "samples": 24576,
    }
    if repair_existing:
        previous = torch.load(directory / "training_state.pt", map_location="cpu", weights_only=False)
        validate_state(previous, allow_untrained=True)
        if (previous["step"] != 0 or previous["optimizer"]["state"]
                or previous["log"]["steps"]
                or previous["log"]["adaptive_initialization"]["source_sha256"] != source_digest):
            raise ValueError("Only a never-trained initialization from this exact baseline may be repaired")
    directory.mkdir(parents=True, exist_ok=repair_existing)
    checkpoints = directory / "checkpoints"
    checkpoints.mkdir(exist_ok=repair_existing)
    torch.save(state, directory / "training_state.pt")
    torch.save(state, checkpoints / "training_state_000000.pt")
    receipt = {
        "batch_size": batch_size, "learning_rate": learning_rate,
        "steps": steps, "samples": 24576, "validation_interval": steps // 4,
        "output": output, "source_step": 0, "empty_adam": True,
        "initial_evaluation_reused": True, "source_sha256": source_digest,
    }
    atomic_json(directory / "initialization.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--repair-existing", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.reference, args.directory, args.batch_size,
                             args.learning_rate, args.repair_existing)))


if __name__ == "__main__":
    main()
