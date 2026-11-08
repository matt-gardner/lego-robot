"""Run an isolated continuation using the installed trainer and a chosen microbatch."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

from so101 import train_entity_sim as trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--microbatch", type=int, required=True)
    parser.add_argument("--until-step", type=int, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--no-activation-checkpointing", action="store_true")
    parser.add_argument("--true-gradient-accumulation", action="store_true")
    args, training_args = parser.parse_known_args()
    if args.microbatch <= 0 or args.microbatch > 128 or args.microbatch % 2:
        raise ValueError("Benchmark microbatch must be positive, even, and at most 128")
    if args.true_gradient_accumulation:
        training_args += ["--gradient-accumulation", "--micro-batch-size", str(args.microbatch)]
    activation_checkpointing = not (args.no_activation_checkpointing or args.true_gradient_accumulation)
    output = Path(training_args[training_args.index("--output") + 1])
    state_path = Path(training_args[training_args.index("--training-state") + 1])
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    start_step = state["step"]
    if not start_step + 2 <= args.until_step <= state["log"]["run_config"]["steps"]:
        raise ValueError("Benchmark interval must follow the saved checkpoint")
    del state
    with state_path.open("rb") as stream:
        base_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    forward = trainer.forward_supervised_batch
    write = trainer.atomic_json
    samples = []
    peaks = []
    started = time.time()

    def chosen_forward(model, inputs, micro_batch_size=16):
        return forward(model, inputs, args.microbatch)

    trainer.forward_supervised_batch = chosen_forward
    if args.no_activation_checkpointing:
        def direct_forward(function, *inputs, use_reentrant=False):
            return function(*inputs)

        trainer.checkpoint = direct_forward
    for name in ("paired_indices", "task_contrast_indices", "obstacle_balanced_indices"):
        original = getattr(trainer.EntityTrajectoryDataset, name)

        def sampled(self, *values, _original=original, **options):
            indices = _original(self, *values, **options)
            samples.append(hashlib.sha256(
                json.dumps([int(index) for index in indices]).encode()).hexdigest())
            return indices

        setattr(trainer.EntityTrajectoryDataset, name, sampled)

    def observed_write(path, value):
        if path.name == "sft_training.json" and value.get("kind") == "sft":
            value["microbatch_benchmark"] = {
                "base_step": start_step, "base_sha256": base_sha,
                "microbatch": args.microbatch, "until_step": args.until_step,
                "activation_checkpointing": activation_checkpointing,
                "gradient_accumulation": args.true_gradient_accumulation,
            }
            for row in value["steps"]:
                if row["step"] > start_step:
                    row["micro_batch_size"] = args.microbatch
                    row["gradient_accumulation"] = args.true_gradient_accumulation
            if value["steps"] and value["steps"][-1]["step"] > start_step:
                torch.cuda.synchronize()
                peaks.append({
                    "allocated": torch.cuda.max_memory_allocated(),
                    "reserved": torch.cuda.max_memory_reserved(),
                })
            if value["steps"] and value["steps"][-1]["step"] >= args.until_step:
                (output / "STOP").write_text("Benchmark interval complete.\n")
        write(path, value)

    trainer.atomic_json = observed_write
    sys.argv = ["so101.train_entity_sim", *training_args]
    try:
        trainer.main()
    except torch.cuda.OutOfMemoryError as error:
        write(args.receipt, {
            "status": "out_of_memory", "microbatch": args.microbatch,
            "base_sha256": base_sha, "error": str(error),
            "activation_checkpointing": activation_checkpointing,
            "gradient_accumulation": args.true_gradient_accumulation,
        })
        raise SystemExit(86) from error
    log = json.loads((output / "sft_training.json").read_text())
    rows = [row for row in log["steps"] if row["step"] > start_step]
    if [row["step"] for row in rows] != list(range(start_step + 1, args.until_step + 1)):
        raise ValueError("Benchmark did not complete its requested optimizer interval")
    if len(samples) != len(rows):
        raise ValueError("Sample fingerprint count differs from optimizer update count")
    write(args.receipt, {
        "status": "completed", "microbatch": args.microbatch,
        "activation_checkpointing": activation_checkpointing,
        "gradient_accumulation": args.true_gradient_accumulation,
        "base_step": start_step, "base_sha256": base_sha,
        "until_step": args.until_step, "updates": len(rows),
        "steady_mean_seconds": sum(row["step_seconds"] for row in rows[1:]) / (len(rows) - 1),
        "max_allocated_bytes": max(peak["allocated"] for peak in peaks),
        "max_reserved_bytes": max(peak["reserved"] for peak in peaks),
        "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "sample_fingerprints": samples, "elapsed_seconds": time.time() - started,
    })


if __name__ == "__main__":
    main()
