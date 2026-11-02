"""Durable, checksummed checkpoint generations; no policy construction."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    fsync_directory(path.parent)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_state(state: dict, *, allow_untrained: bool = False) -> None:
    import torch

    required = {"step", "model", "optimizer", "log", "rng"}
    if not required <= state.keys():
        raise ValueError(f"Incomplete resume state: missing {required - state.keys()}")
    if type(state["step"]) is not int or state["step"] < 0:
        raise ValueError("Resume step must be a nonnegative integer")
    if (not isinstance(state["model"], dict) or not state["model"]
            or not isinstance(state["optimizer"], dict)
            or not isinstance(state["log"], dict)
            or not isinstance(state["log"].get("steps"), list)):
        raise ValueError("Invalid resume model, optimizer, or training log")
    history = state["log"]["steps"]
    if allow_untrained and state["step"] == 0:
        if (history or state["optimizer"].get("state") != {}
                or not isinstance(state["optimizer"].get("param_groups"), list)
                or not state["optimizer"]["param_groups"]
                or any(not isinstance(group, dict)
                       or not isinstance(group.get("params"), list)
                       or not group["params"] for group in state["optimizer"]["param_groups"])):
            raise ValueError("Untrained resume state requires empty history/Adam state and valid parameter groups")
    else:
        if (not history or not isinstance(history[-1], dict)
                or state["step"] != history[-1].get("step")):
            raise ValueError("Resume model and log steps do not match")
        if not state["optimizer"].get("state"):
            raise ValueError("Missing trained model or Adam state")

    def finite(value: Any) -> None:
        if isinstance(value, torch.Tensor):
            if not torch.isfinite(value).all():
                raise ValueError("Non-finite model/optimizer checkpoint")
        elif isinstance(value, dict):
            for item in value.values():
                finite(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                finite(item)

    finite(state["model"])
    finite(state["optimizer"])


def publish(output: Path, state: dict, retain: int) -> None:
    import torch

    if retain < 0:
        raise ValueError("Checkpoint retention must be nonnegative")
    temporary = output / "latest_state.pt.tmp"
    with temporary.open("wb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    if retain:
        validate_state(state)
        directory = output / "generations"
        directory.mkdir(exist_ok=True)
        name = f"step_{state['step']:09d}"
        checkpoint = directory / f"{name}.pt"
        if checkpoint.exists():
            if checkpoint.with_suffix(".json").exists():
                raise ValueError(f"Checkpoint generation already exists: {checkpoint}")
            # A kill before commit may leave data for this recomputed step.
            checkpoint.unlink()
        temporary.replace(checkpoint)
        fsync_directory(directory)
        manifest = {
            "step": state["step"], "file": checkpoint.name,
            "sha256": sha256(checkpoint), "size": checkpoint.stat().st_size,
            "split_sha256": sha256(output / "data_split.json"),
        }
        identity = output / "cloud_identity.json"
        if identity.exists():
            manifest["identity_sha256"] = sha256(identity)
        atomic_json(directory / f"{name}.json", manifest)
        atomic_json(output / "checkpoint_manifest.json", manifest)
        os.link(checkpoint, temporary)
    temporary.replace(output / "latest_state.pt")
    fsync_directory(output)
    if retain:
        for manifest_path in sorted(directory.glob("step_*.json"))[:-retain]:
            payload = json.loads(manifest_path.read_text())
            manifest_path.unlink()
            (directory / payload["file"]).unlink()
        fsync_directory(directory)


def verified_generation(output: Path) -> tuple[Path, dict]:
    """Find the newest committed, intact generation, reporting rejected ones."""
    for path in sorted((output / "generations").glob("step_*.json"), reverse=True):
        try:
            manifest = json.loads(path.read_text())
            if manifest["file"] != path.with_suffix(".pt").name:
                raise ValueError("Generation filename mismatch")
            checkpoint = path.with_suffix(".pt")
            if (
                checkpoint.stat().st_size != manifest["size"]
                or sha256(checkpoint) != manifest["sha256"]
                or sha256(output / "data_split.json") != manifest["split_sha256"]
            ):
                raise ValueError("Checkpoint or split checksum mismatch")
            if "identity_sha256" in manifest and (
                sha256(output / "cloud_identity.json") != manifest["identity_sha256"]
            ):
                raise ValueError("Source/data identity mismatch")
            return checkpoint, manifest
        except (OSError, ValueError, KeyError) as error:
            logging.error("Rejected checkpoint generation %s: %s", path, error)
    raise ValueError(f"No verified checkpoint generation in {output}")


def recover(output: Path) -> dict:
    import torch

    checkpoint, manifest = verified_generation(output)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    validate_state(state)
    if state["step"] != manifest["step"]:
        raise ValueError("Checkpoint and manifest steps do not match")
    # Free rejected newer step names so the trainer can recompute them.
    for rejected in (output / "generations").glob("step_*.json"):
        if rejected.stem > checkpoint.stem:
            quarantine = output / "generations/rejected"
            quarantine.mkdir(exist_ok=True)
            suffix = uuid.uuid4().hex
            rejected.replace(quarantine / f"{rejected.stem}.{suffix}.json")
            data = rejected.with_suffix(".pt")
            if data.exists():
                data.replace(quarantine / f"{data.stem}.{suffix}.pt")
            fsync_directory(quarantine)
            fsync_directory(output / "generations")
    temporary = output / "latest_state.pt.recover"
    temporary.unlink(missing_ok=True)
    os.link(checkpoint, temporary)
    temporary.replace(output / "latest_state.pt")
    atomic_json(output / "checkpoint_manifest.json", manifest)
    fsync_directory(output)
    return state
