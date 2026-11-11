#!/usr/bin/env python3
"""Write a detached Docker argv for the managed launcher; never execute it."""
import argparse
import json
from pathlib import Path
import re


def command(image: str, experiment: Path, checkpoint: Path, cache: Path,
            container: str, *, freeze_backbone: bool = False) -> list[str]:
    if not re.fullmatch(r"[^ ]+@sha256:[0-9a-f]{64}", image):
        raise ValueError("Use a qualified immutable GPU image digest containing the new source")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]+", container):
        raise ValueError("Invalid container name")
    paths = [experiment, checkpoint, cache]
    if any(not path.is_absolute() or "," in str(path) for path in paths):
        raise ValueError("Mounts require absolute paths without commas")
    output = (experiment / ("output-frozen" if freeze_backbone else "output")
              / "entity_smolvla_sft_b4_lr1e-05")
    result = ["docker", "run", "--detach", "--name", container, "--gpus", "all",
              "--network", "none", "--read-only", "--tmpfs", "/tmp:rw,nosuid,nodev,size=1g",
              "--mount", f"type=bind,src={experiment},dst={experiment}",
              "--mount", f"type=bind,src={checkpoint},dst={checkpoint},readonly",
              "--mount", f"type=bind,src={cache},dst=/cache,readonly",
              "--env", "HF_HOME=/cache", "--env", "HF_HUB_OFFLINE=1",
              "--env", "TRANSFORMERS_OFFLINE=1", image,
              "python", "-m", "so101.train_piece_recognition",
              "--images", str(experiment / "render" / "images.json"),
              "--checkpoint", str(checkpoint), "--output", str(output),
              "--device", "cuda", "--steps", "500"]
    if freeze_backbone:
        result.append("--freeze-backbone")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--container", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze-backbone", action="store_true")
    args = parser.parse_args()
    prepared = command(args.image, args.experiment_root, args.checkpoint, args.cache,
                       args.container, freeze_backbone=args.freeze_backbone)
    with args.output.open("x") as stream:
        json.dump(prepared, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
