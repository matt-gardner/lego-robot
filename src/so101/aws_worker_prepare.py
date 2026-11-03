"""Root-plan-bound preparation of staged archive inputs for read-only container mounts."""
from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath

from so101.aws_extract import extract
from so101.aws_worker_bootstrap import load_manifest
from so101.cloud_checkpoint import sha256
from so101.control_broker import canonical, trusted_json, trusted_path


def prepare(manifest_path: Path, manifest_sha256: str, plan_path: Path, plan_sha256: str,
            staged: Path) -> None:
    manifest = load_manifest(manifest_path, manifest_sha256)
    plan = trusted_json(plan_path)
    if (hashlib.sha256(canonical(plan)).hexdigest() != plan_sha256
            or plan.get("spec_sha256") != manifest["spec_sha256"]
            or plan.get("argv") != manifest["spec"]["recipe"].get("argv")):
        raise ValueError("Input preparation must bind the exact approved plan/specification")
    staged = trusted_path(staged)
    binding = trusted_json(staged / ".bootstrap.json")
    if (binding.get("manifest_sha256") != manifest_sha256
            or binding.get("spec_sha256") != manifest["spec_sha256"]
            or sha256(staged / ".staged.json") != binding.get("staging_receipt_sha256")):
        raise ValueError("Input preparation requires a committed exact staging receipt")
    archives = plan.get("archives")
    if not isinstance(archives, dict):
        raise ValueError("Root plan requires an explicit archive preparation map")
    destinations = []
    for name, settings in archives.items():
        artifact = manifest["artifacts"].get(name)
        if (not isinstance(artifact, dict) or not isinstance(settings, dict)
                or set(settings) != {"destination", "max_bytes", "max_members"}):
            raise ValueError("Archive preparation must name a verified staged artifact")
        destination = settings["destination"]
        if not isinstance(destination, str):
            raise ValueError("Prepared input destination must be a relative string path")
        relative = PurePosixPath(destination)
        if (relative.is_absolute()
                or not relative.parts or ".." in relative.parts
                or "\\" in destination or str(relative) != destination
                or destination.startswith(".") or destination in manifest["artifacts"]):
            raise ValueError("Prepared inputs must stay in new named staged directories")
        if any(relative == old or relative in old.parents or old in relative.parents
               for old in destinations):
            raise ValueError("Prepared archive destinations must not overlap")
        destinations.append(relative)
        archive = trusted_path(staged / name)
        if sha256(archive) != artifact["sha256"]:
            raise ValueError("Staged archive changed before input preparation")
        extract(archive, staged / destination, max_bytes=settings["max_bytes"],
                max_members=settings["max_members"])
