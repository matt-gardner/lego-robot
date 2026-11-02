"""Adapt locked CPU wheel pins for uv's explicit CUDA backend selection."""

import argparse
from pathlib import Path
import re


def cuda_requirements(requirements: str) -> str:
    packages = {"torch", "torchvision", "torchcodec"}
    found = set()
    result = []
    for line in requirements.splitlines():
        match = re.match(r"^(torch|torchvision|torchcodec)==([^ ;]+)", line)
        if match:
            found.add(match[1])
            version = match[2]
            if "+" in version and not version.endswith("+cpu"):
                raise ValueError(f"Unexpected accelerator pin: {line}")
            line = line.replace("==" + version, "==" + version.removesuffix("+cpu"), 1)
        result.append(line)
    if found != packages:
        raise ValueError(f"Missing locked PyTorch packages: {sorted(packages - found)}")
    return "\n".join(result) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    args.destination.write_text(cuda_requirements(args.source.read_text()))
