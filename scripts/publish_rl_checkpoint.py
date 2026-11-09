"""Publish RL delta/Adam/RNG bound to an immutable SFT initialization bundle."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import torch

from so101.aws_checkpoint import CheckpointPublisher
from so101.cloud_checkpoint import atomic_json, sha256, validate_state
from so101.resume_rng import plan_device_rng


def check(path: Path) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    validate_state(state)
    log = state["log"]
    if (log["rl_estimator"] != "denoising-grpo"
            or log["training_objective"] != "task-reward-plus-reference-jerk"
            or log["jerk_regularizer"]["weight"] != .02
            or log["jerk_regularizer"]["scale"] != 6
            or [row["step"] for row in log["steps"]] != list(range(1, state["step"] + 1))
            or not log["validation"] or not log["assistance_curriculum"]):
        raise ValueError("RL snapshot lost its authorized objective, history, or curriculum")
    if not all(name.startswith(("model.action_in_proj.", "model.action_out_proj."))
               for name in state["model"]):
        raise ValueError("Unexpected trainable parameters in projection-only RL state")
    plan_device_rng(state, "cuda", False, None)
    if not {"python", "numpy", "sampling", "torch", "device"} <= state["rng"].keys():
        raise ValueError("Incomplete RL sampling RNG")
    return {"step": state["step"], "run_config": log["run_config"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--dependencies", type=Path, required=True,
                        help="Tar of source, data split, transition receipt, and versioned SFT recovery receipt")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--profile", help="Explicit operator AWS profile; omit on an instance-profile worker")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    # Atomically replaced trainer states remain immutable after opening/copying.
    snapshot = args.root / "downloaded_state.pt"
    if snapshot.exists():
        raise FileExistsError(snapshot)
    shutil.copyfile(args.state, snapshot)
    verified = check(snapshot)
    dependency_sha = sha256(args.dependencies)
    spec = {**verified, "dependencies_sha256": dependency_sha,
            "recovery": "Recreate initial_policy from versioned SFT state and transition source, then restore RL delta/Adam/RNG"}
    spec_sha = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    def request(command, **kwargs):
        if args.profile is not None:
            command = [command[0], "--profile", args.profile, *command[1:]]
            kwargs.pop("env")
        try:
            return subprocess.run(command, **kwargs)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"AWS publication failed: {error.stderr}") from error

    publisher = CheckpointPublisher(
        owner="133961745003", aws_cli=shutil.which("aws"),
        output_prefix="s3://so101-training-133961745003-us-west-2/runs/lego-robot/rl/",
        run=request,
    )
    dependencies = publisher.put(
        args.dependencies, publisher.output_prefix + args.job_id
        + f"/step_{verified['step']:09d}/{dependency_sha}.tar",
    )
    atomic_json(args.root / "dependencies.json", {
        "artifact": asdict(dependencies), "spec_sha256": spec_sha, **spec,
    })
    manifest = publisher.publish(
        snapshot, args.root, job_id=args.job_id, step=verified["step"],
        spec_sha256=spec_sha, max_bytes=128 * 1024 * 1024,
        validate_snapshot=check,
    )
    receipt = {"state_manifest": asdict(manifest), "dependencies": asdict(dependencies),
               "step": verified["step"], "spec_sha256": spec_sha}
    atomic_json(args.root / "receipt.json", receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
