"""Validate and durably publish a full long-run recovery checkpoint."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import uuid


def check(snapshot, original):
    # Torch is available in the pinned validation image, not the host publisher.
    import torch
    import numpy as np
    from so101.cloud_checkpoint import sha256, validate_state
    from so101.resume_rng import plan_device_rng

    torch.set_num_threads(4)
    state = torch.load(snapshot, map_location="cpu", weights_only=False)
    reference = torch.load(original, map_location="cpu", weights_only=False)
    validate_state(state)
    validate_state(reference)
    log = state["log"]
    config = log["run_config"]
    continuation = log["continuations"][-1]
    expected = {
        **reference["log"]["run_config"],
        "steps": config["steps"], "warmup_fraction": 38 / config["steps"],
        "output": "outputs/entity_smolvla_sft_long_b64_lr3e-05",
        "training_state": "outputs/entity_smolvla_sft_long_b64_lr3e-05/training_state.pt",
    }
    if (continuation["source_sha256"] != sha256(original)
            or continuation["previous_run_config"] != reference["log"]["run_config"]
            or config != continuation["new_run_config"]
            or config != expected
            or config["batch_size"] != 64 or config["learning_rate"] != 3e-5
            or config["learning_rate_schedule"] != "warmup-constant"
            or round(config["steps"] * config["warmup_fraction"]) != 38
            or config["device"] != "cuda" or not config["gradient_accumulation"]
            or config["micro_batch_size"] != 8
            or not 384 <= state["step"] <= config["steps"]):
        raise ValueError("Checkpoint does not match the authorized constant-rate continuation")
    if (set(state["model"]) != set(reference["model"])
            or set(state["optimizer"]["state"]) != set(reference["optimizer"]["state"])
            or state["optimizer"]["param_groups"] != reference["optimizer"]["param_groups"]
            or set(state["rng"]) != set(reference["rng"])
            or not {"backend", "sampling", "python", "numpy", "torch", "device"} <= set(state["rng"])
            or state["rng"]["backend"] != "cuda"
            or log["supervised_probe"] != reference["log"]["supervised_probe"]
            or log["dataset"] != reference["log"]["dataset"]):
        raise ValueError("Checkpoint lost model/Adam/RNG or changed the frozen data/probe")
    if (log["steps"][:384] != reference["log"]["steps"]
            or [row["step"] for row in log["steps"]] != list(range(1, state["step"] + 1))):
        raise ValueError("Checkpoint changed or omitted optimizer history")
    plan_device_rng(state, "cuda", False, None)
    random.Random().setstate(state["rng"]["python"])
    np.random.RandomState().set_state(state["rng"]["numpy"])
    np.random.default_rng().bit_generator.state = state["rng"]["sampling"]
    torch.Generator().set_state(state["rng"]["torch"])
    for name, tensor in state["model"].items():
        previous = reference["model"][name]
        if tensor.shape != previous.shape or tensor.dtype != previous.dtype:
            raise ValueError(f"Model parameter shape/dtype changed: {name}")
    for parameter, slots in state["optimizer"]["state"].items():
        previous = reference["optimizer"]["state"][parameter]
        if set(slots) != set(previous):
            raise ValueError(f"Adam state fields changed: {parameter}")
        for name, value in slots.items():
            if (not isinstance(value, torch.Tensor) or value.shape != previous[name].shape
                    or value.dtype != previous[name].dtype):
                raise ValueError(f"Adam tensor shape/dtype changed: {parameter}/{name}")
    for name in ("torch", "device"):
        value = state["rng"][name]
        previous = reference["rng"][name]
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.uint8
                or value.shape != previous.shape):
            raise ValueError(f"Invalid retained RNG tensor: {name}")
    identity = lambda row: (row["task"], row["source"], row["seed"])
    problems = [identity(row) for row in reference["log"]["validation"][-1]["rollouts"]]
    for validation in log["validation"]:
        if [identity(row) for row in validation["rollouts"]] != problems:
            raise ValueError("Fixed rollout population changed")
    return {
        "step": state["step"], "sha256": sha256(snapshot),
        "size": snapshot.stat().st_size,
        "spec_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
    }


def publish(root, bundle, image, job_id):
    # The immutable stdlib publisher is restored separately from the GPU image.
    sys.path.insert(0, str(bundle / "publisher-source/src"))
    from so101.aws_checkpoint import CheckpointPublisher
    from so101.cloud_checkpoint import atomic_json

    base = root / "backup-work"
    base.mkdir(mode=0o700, exist_ok=True)
    work = base / uuid.uuid4().hex
    work.mkdir(mode=0o700)
    snapshot = work / "snapshot.pt"
    shutil.copyfile(root / "output/training_state.pt", snapshot)
    command = [
        "docker", "run", "--rm", "--network", "none", "--read-only", "--user", "0:0",
        "--cpus", "4", "--tmpfs", "/tmp:rw,size=1g",
        "--mount", f"type=bind,src={root},dst=/long-run,readonly",
        "--mount", f"type=bind,src={bundle / 'source'},dst=/long-source,readonly",
        image, "python", "/long-source/scripts/publish_long_gpu_checkpoint.py", "check",
        "--original", "/long-run/inputs/baseline_384.pt",
        "--snapshot", "/long-run/" + str(snapshot.relative_to(root)),
    ]
    response = subprocess.run(command, capture_output=True, text=True)
    if response.returncode:
        raise RuntimeError(f"Full checkpoint validation failed: {response.stderr}")
    verified = json.loads(response.stdout)
    step = verified["step"]
    receipt = root / "backup-receipts" / f"step_{step:09d}.json"
    if receipt.exists():
        snapshot.unlink()
        work.rmdir()
        print(receipt.read_text(), flush=True)
        return
    publisher = CheckpointPublisher(
        owner="133961745003", aws_cli="/usr/local/bin/aws",
        output_prefix="s3://so101-training-133961745003-us-west-2/runs/lego-robot/",
    )

    def validate(copied):
        if copied.stat().st_size != verified["size"]:
            raise ValueError("Checkpoint size changed before publication")
        checksum = hashlib.sha256()
        with copied.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(block)
        if checksum.hexdigest() != verified["sha256"]:
            raise ValueError("Checkpoint bytes changed after full-state validation")

    manifest = publisher.publish(
        snapshot, work, job_id=job_id, step=step,
        spec_sha256=verified["spec_sha256"], max_bytes=2_000_000_000,
        validate_snapshot=validate,
    )
    published = json.loads((work / job_id / f"step_{step:09d}/published.json").read_text())
    published.update(step=step, manifest=asdict(manifest), work_directory=str(work))
    atomic_json(receipt, published)
    print(json.dumps(published), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    checker = commands.add_parser("check")
    checker.add_argument("--snapshot", type=Path, required=True)
    checker.add_argument("--original", type=Path, required=True)
    publisher = commands.add_parser("publish")
    publisher.add_argument("--root", type=Path, required=True)
    publisher.add_argument("--bundle", type=Path, required=True)
    publisher.add_argument("--image", required=True)
    publisher.add_argument("--job-id", required=True)
    args = parser.parse_args()
    if args.command == "check":
        print(json.dumps(check(args.snapshot, args.original)), flush=True)
    else:
        publish(args.root, args.bundle, args.image, args.job_id)


if __name__ == "__main__":
    main()
