"""Preserve the explicitly deferred full checkpoint before instance termination."""
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, "/home/ubuntu/lego-sweep/source/src")
from so101.aws_checkpoint import CheckpointPublisher
from so101.cloud_checkpoint import atomic_json, sha256


def main():
    source = Path("/home/ubuntu/lego-sweep/output/entity_smolvla_sft_b64_lr3e-06")
    log = json.loads((source / "sft_training.json").read_text())
    if log["status"] != "stopped" or log["steps"][-1]["step"] != 26 or not (source / "STOP").is_file():
        raise ValueError("Deferred arm is not the preserved stopped update-26 trial")
    root = Path("/opt/dlami/nvme/lego-adaptive-82b32533")
    snapshots = root / "deferred-checkpoint-publication"
    snapshots.mkdir(mode=0o700, exist_ok=False)
    publisher = CheckpointPublisher(
        owner="133961745003", aws_cli="/usr/local/bin/aws",
        output_prefix="s3://so101-training-133961745003-us-west-2/runs/lego-robot/",
    )
    names = ["STOP", "training_state.pt", "model.pt", "metadata.json",
             "entity_config.json", "data_split.json", "sft_training.json"]
    names += [str(path.relative_to(source)) for path in sorted(
        (source / "checkpoints").glob("training_state_*.pt"))]
    artifacts = {}
    for name in names:
        target = snapshots / name
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        before = sha256(source / name)
        shutil.copyfile(source / name, target)
        if sha256(target) != before or sha256(source / name) != before:
            raise ValueError("Deferred checkpoint changed during preservation")
        prefix = "82b325334a4f4d24ba37c5a2ecf6e981/deferred-b64-lr3e-06/"
        artifacts[name] = asdict(publisher.put(
            target, publisher.output_prefix + prefix + before + "/" + name))
    atomic_json(root / "deferred-checkpoint-export.json",
                {"step": 26, "status": "deferred-not-completed", "files": artifacts})


if __name__ == "__main__":
    main()
