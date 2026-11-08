"""Prepare reproducible LEGO scene proposals; no simulation or data collection."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random

from so101.cloud_checkpoint import sha256


CATEGORIES = ("separated", "cluster", "overlap", "pile")
TASKS = ("grasp", "view", "push", "place")
STATUSES = {"imported", "prepared", "verified", "unavailable"}
COLORS = ("gray", "green", "orange", "navy", "teal", "brown")
LDRAW_TO_MUJOCO = [[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]]


class CatalogUnavailable(ValueError):
    """A valid catalog has no qualified parts yet."""


@dataclass(frozen=True)
class Part:
    part_id: str
    family: str
    dimensions_m: tuple[float, float, float]
    mjcf: str
    assets: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class Catalog:
    catalog_id: str
    manifest_sha256: str
    parts: tuple[Part, ...]
    root: Path
    manifest_path: Path | None = None


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _asset(root: Path, name: object) -> tuple[str, Path]:
    name = _text(name, "asset path")
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"Asset escapes catalog: {name}")
    path = root
    for component in relative.parts:
        path = path / component
        if path.is_symlink():
            raise ValueError(f"Symlink in catalog asset: {name}")
    if not path.resolve().is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing or invalid catalog asset: {name}")
    return relative.as_posix(), path


def load_catalog(manifest: Path) -> Catalog:
    """Validate every verified part's declared assets before sampling any part."""
    raw = manifest.read_bytes()
    report = json.loads(raw)
    if (not isinstance(report, dict) or type(report.get("schema_version")) is not int
            or report["schema_version"] != 1):
        raise ValueError("Expected LEGO catalog schema_version 1")
    catalog_id = _text(report.get("catalog_id"), "catalog_id")
    entries = report.get("parts")
    if not isinstance(entries, list):
        raise ValueError("Catalog parts must be a list")
    root = manifest.parent.resolve()
    parts = []
    identities = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Catalog part must be an object")
        part_id = _text(entry.get("part_id"), "part_id")
        if part_id in identities:
            raise ValueError(f"Duplicate catalog part_id: {part_id}")
        identities.add(part_id)
        family = _text(entry.get("family"), f"{part_id} family")
        _text(entry.get("description"), f"{part_id} description")
        if not isinstance(entry.get("status"), str) or entry["status"] not in STATUSES:
            raise ValueError(f"Invalid catalog status: {part_id}")
        if entry["status"] != "verified":
            continue
        geometry = entry.get("geometry")
        if (not isinstance(geometry, dict)
                or geometry.get("frame_ldraw_to_mujoco") != LDRAW_TO_MUJOCO
                or geometry.get("metres_per_ldu") != .0004):
            raise ValueError(f"Unsupported LEGO coordinate frame or units: {part_id}")
        dimensions = entry.get("dimensions_m")
        if (not isinstance(dimensions, list) or len(dimensions) != 3
                or any(type(value) not in (int, float)
                       or not math.isfinite(value) or value <= 0 for value in dimensions)):
            raise ValueError(f"Invalid dimensions_m: {part_id}")
        mjcf, _ = _asset(root, entry.get("mjcf"))
        assets = entry.get("assets")
        if not isinstance(assets, list) or not assets:
            raise ValueError(f"Missing asset checksums: {part_id}")
        names = set()
        hashes = []
        for asset in assets:
            if not isinstance(asset, dict):
                raise ValueError(f"Invalid asset record: {part_id}")
            name, path = _asset(root, asset.get("path"))
            if name in names:
                raise ValueError(f"Duplicate asset path: {part_id}/{name}")
            names.add(name)
            digest = asset.get("sha256")
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)):
                raise ValueError(f"Invalid asset SHA256: {part_id}/{name}")
            if sha256(path) != digest:
                raise ValueError(f"Asset checksum mismatch: {part_id}/{name}")
            hashes.append((name, digest))
        if mjcf not in names:
            raise ValueError(f"MJCF missing from checksummed assets: {part_id}")
        parts.append(Part(part_id, family, tuple(float(v) for v in dimensions),
                          mjcf, tuple(hashes)))
    if not parts:
        raise CatalogUnavailable(f"No verified LEGO parts in catalog {catalog_id}")
    return Catalog(catalog_id, hashlib.sha256(raw).hexdigest(),
                   tuple(sorted(parts, key=lambda part: part.part_id)), root, manifest.resolve())


def visual_audit_due(*, category: str, accepted_count: int, seed: int) -> bool:
    """Select exactly one accepted example per ten within each category."""
    _text(category, "audit category")
    if type(accepted_count) is not int or accepted_count < 1 or type(seed) is not int:
        raise ValueError("accepted_count must be positive and seed an integer")
    block, offset = divmod(accepted_count - 1, 10)
    identity = json.dumps([category, seed, block], separators=(",", ":")).encode()
    rng = random.Random(int(hashlib.sha256(identity).hexdigest(), 16))
    return offset == rng.randrange(10)


def prepare_scene(
    catalog: Catalog,
    *,
    seed: int,
    sequence: int,
    category: str,
    task: str,
    object_count: int,
) -> dict:
    """Sample part families and placement proposals, not accepted trajectories."""
    if type(seed) is not int or type(sequence) is not int or sequence < 0:
        raise ValueError("seed must be an integer and sequence nonnegative")
    if category not in CATEGORIES or task not in TASKS:
        raise ValueError("Unknown curriculum category or task")
    if type(object_count) is not int or object_count < 2:
        raise ValueError("Paired scene proposals need at least two objects")
    if not catalog.parts:
        raise CatalogUnavailable("No verified LEGO parts")
    identity = {
        "generator": "lego-scene-plan-v6", "catalog_sha256": catalog.manifest_sha256,
        "seed": seed, "sequence": sequence, "category": category,
        "task": task, "object_count": object_count,
        "sampleable_part_ids": sorted(part.part_id for part in catalog.parts),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    scene_id = hashlib.sha256(encoded).hexdigest()
    rng = random.Random(int(scene_id, 16))
    families: dict[str, list[Part]] = {}
    for part in catalog.parts:
        families.setdefault(part.family, []).append(part)
    family_names = sorted(families)
    objects = []
    columns = math.ceil(math.sqrt(object_count))
    rows = math.ceil(object_count / columns)
    spacing = max(.04, max(max(part.dimensions_m[:2]) for part in catalog.parts) * 1.5)
    for index in range(object_count):
        part = rng.choice(families[rng.choice(family_names)])
        if category == "separated":
            x = (index % columns - (columns - 1) / 2) * spacing
            y = (index // columns - (rows - 1) / 2) * spacing
        else:
            spread = {"cluster": .06, "overlap": .035, "pile": .02}[category]
            x, y = rng.uniform(-spread, spread), rng.uniform(-spread, spread)
        # Only stacked proposals need staggered drop heights.
        height = .01 + part.dimensions_m[2]
        if category in ("overlap", "pile"):
            height += index * max(part.dimensions_m[2], .01)
        objects.append({
            "instance_id": f"object_{index:05d}", "part_id": part.part_id,
            "family": part.family, "color": rng.choice(COLORS),
            "position_relative_to_scene_m": [x, y, height],
            "yaw_rad": rng.uniform(-math.pi, math.pi),
        })
    selected = rng.sample(objects, 2)
    if task in ("grasp", "view") and selected[0]["color"] == selected[1]["color"]:
        selected[1]["color"] = rng.choice([
            color for color in COLORS if color != selected[0]["color"]
        ])
    commands = []
    for index in range(2):
        commands.append({
            "command_id": f"{scene_id}:{index}",
            "task": task,
            "object_instance_id": selected[index if task in ("grasp", "view") else 0]["instance_id"],
            "destination_id": f"destination_{index}" if task in ("push", "place") else None,
        })
    return {
        "schema_version": 1, **identity, "scene_id": scene_id,
        "catalog_id": catalog.catalog_id, "status": "proposed",
        "requires_physical_settling": True,
        "objects": objects,
        "destinations": [
            {"destination_id": "destination_0", "position_relative_to_scene_m": [0., -.10, 0.]},
            {"destination_id": "destination_1", "position_relative_to_scene_m": [0., .10, 0.]},
        ] if task in ("push", "place") else [],
        "pair_id": f"{scene_id}:{task}", "commands": commands,
        "tags": {
            "category": category, "task": task, "object_count": object_count,
            "category_semantics": "proposal distribution, not measured settled difficulty",
            "part_ids": sorted({obj["part_id"] for obj in objects}),
            "families": sorted({obj["family"] for obj in objects}),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sequence", type=int, required=True)
    parser.add_argument("--category", choices=CATEGORIES, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--object-count", type=int, required=True)
    args = parser.parse_args()
    try:
        catalog = load_catalog(args.catalog)
        plan = prepare_scene(catalog, seed=args.seed, sequence=args.sequence,
                             category=args.category, task=args.task,
                             object_count=args.object_count)
    except (OSError, ValueError) as error:
        parser.exit(2, f"Cannot prepare LEGO scene: {error}\n")
    print(json.dumps(plan, allow_nan=False))


if __name__ == "__main__":
    main()
