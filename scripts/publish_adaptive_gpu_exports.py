"""Validate and version-publish completed adaptive artifacts without GPU training."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

# Use the already installed complete publisher, not the unrelated broker rollout.
sys.path.insert(0, "/home/ubuntu/lego-sweep/source/src")
from so101.aws_checkpoint import CheckpointPublisher
from so101.cloud_checkpoint import atomic_json


def run(root, image, job_id, expected_arms=5):
    publication = root / "publication"
    publication.mkdir(mode=0o700, exist_ok=True)
    publisher = CheckpointPublisher(
        owner="133961745003", aws_cli="/usr/local/bin/aws",
        output_prefix="s3://so101-training-133961745003-us-west-2/runs/lego-robot/",
    )
    results = []
    for path in sorted((root / "output").glob("*/sft_training.json")):
        directory = path.parent
        log = json.loads(path.read_text())
        config = log["run_config"]
        arm = f"b{config['batch_size']}-lr{config['learning_rate']:.0e}"
        receipt = publication / f"{arm}-export.json"
        if not receipt.exists():
            command = ["docker", "run", "--rm", "--network", "none", "--read-only",
                       "--cpus", "4", "--tmpfs", "/tmp:rw,size=1g"]
            mounts = [
                (root / "output", "/exports"),
                (Path("/home/ubuntu/lego-sweep/inputs/hf_cache"), "/work/cache/huggingface"),
            ]
            for module in ("aws_state_check", "sweep_entity_sft", "train_entity_sim"):
                mounts.append((Path("/home/ubuntu/lego-sweep/microbatch-scripts") / f"{module}.py",
                               f"/opt/so101/src/so101/{module}.py"))
            for source, destination in mounts:
                command += ["--mount", f"type=bind,src={source},dst={destination},readonly"]
            command += [image, "python", "-m", "so101.aws_state_check",
                        "--exports", f"/exports/{directory.name}",
                        "--original", f"/exports/{directory.name}/checkpoints/training_state_000000.pt"]

            def validate(_):
                response = subprocess.run(command, capture_output=True, text=True)
                if response.returncode:
                    raise RuntimeError(f"Validation failed for {arm}: {response.stderr}")
                return json.loads(response.stdout)

            atomic_json(root / "export_progress.json", {"phase": "validating-and-publishing", "arm": arm})
            manifest = publisher.publish_exports(
                directory, publication, job_id=job_id, arm=arm,
                step=config["steps"], validation_interval=config["validation_interval"],
                spec_sha256=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                max_bytes=12_000_000_000, validate_output=validate,
            )
            atomic_json(receipt, {"arm": arm, "job_id": job_id, "manifest": asdict(manifest)})
        final = log["validation"][-1]
        results.append({
            "arm": arm, "samples": log["steps"][-1]["samples_seen"],
            "successes": sum(row["success"] for row in final["rollouts"]),
            "episodes": len(final["rollouts"]),
            "heldout_loss": final["supervised_loss"],
            "heldout_frames": log["supervised_probe"]["samples"],
            "export": json.loads(receipt.read_text()),
        })
    if len(results) != expected_arms:
        raise ValueError(f"Expected all {expected_arms} completed trials before export completion")
    atomic_json(root / "adaptive_comparison.json", results)
    atomic_json(root / "export_progress.json", {"phase": "all-adaptive-exports-complete", "arms": results})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--expected-arms", type=int, default=5)
    args = parser.parse_args()
    try:
        run(args.root, args.image, args.job_id, args.expected_arms)
    except Exception as error:
        atomic_json(args.root / "export_failure.json", {"error": str(error)})
        raise


if __name__ == "__main__":
    main()
