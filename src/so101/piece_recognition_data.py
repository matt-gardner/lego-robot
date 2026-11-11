"""Prepare hash-bound recognition inputs from saved piles; never render or train."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import xml.etree.ElementTree as ET

from so101.part_preview import visual_geometry


PHOTO_TEMPLATE_SHA256 = "ae154b5b6c938c8b55867dea647338de1eb52aec4d3ca6ee8ce5ab1117e64543"
PHOTO_PART_IDS = ("3001", "3002", "3003", "3004", "3020", "3022", "3700", "3702")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_new_json(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def read_bound(path: Path, expected: str) -> bytes:
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError(f"Changed experiment input: {path}")
    return content


def visual_signature(tree: ET.Element, body: ET.Element) -> str:
    meshes = {mesh.get("name"): mesh for mesh in tree.findall("./asset/mesh")}
    geometries = []
    for geom in body.iter("geom"):
        if geom.get("group") != "2":
            continue
        mesh = meshes.get(geom.get("mesh"))
        if mesh is None or mesh.get("file"):
            raise ValueError("Photo identity recovery requires inline visual meshes")
        geometries.append({
            "mesh": {k: v for k, v in mesh.attrib.items() if k != "name"},
            "transform": {k: geom.get(k) for k in ("pos", "quat", "euler", "size", "type")},
        })
    if not geometries:
        raise ValueError("Photo identity recovery found no visual geometry")
    return hashlib.sha256(json.dumps(geometries, sort_keys=True).encode()).hexdigest()


def photo_identity_map(template: Path) -> dict[str, str]:
    if digest(template) != PHOTO_TEMPLATE_SHA256:
        raise ValueError("Photo identity mapping requires the exact retained template")
    tree = ET.fromstring(template.read_bytes())
    bodies = tree.findall("./worldbody/body")
    if len(bodies) != 90:
        raise ValueError("Expected the retained 90-body photo template")
    result = {}
    for index, body in enumerate(bodies[:len(PHOTO_PART_IDS)]):
        signature = visual_signature(tree, body)
        part = PHOTO_PART_IDS[index % len(PHOTO_PART_IDS)]
        if signature in result and result[signature] != part:
            raise ValueError("Photo template maps one geometry to multiple IDs")
        result[signature] = part
    return result


def assign_splits(scenes: list[dict], seed: int) -> list[dict]:
    """Union provenance relatives before assigning whole components to splits."""
    parent = list(range(len(scenes)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    owners = {}
    for index, scene in enumerate(scenes):
        for key in scene["group_keys"]:
            if key in owners:
                parent[find(index)] = find(owners[key])
            owners[key] = index
    groups = defaultdict(list)
    for index in range(len(scenes)):
        groups[find(index)].append(index)
    components = sorted(groups.values(), key=lambda ids: min(scenes[i]["source"] for i in ids))
    if len(components) < 3:
        raise ValueError("Need at least three independent pile families for train/dev/test")
    random.Random(seed).shuffle(components)
    test_count = max(1, round(len(components) * .15))
    dev_count = max(1, round(len(components) * .15))
    for position, indices in enumerate(components):
        split = ("test" if position < test_count else
                 "development" if position < test_count + dev_count else "train")
        family = hashlib.sha256(
            "\n".join(sorted({key for i in indices for key in scenes[i]["group_keys"]})).encode()
        ).hexdigest()
        for index in indices:
            scenes[index] = {**scenes[index], "split": split, "family": family}
    return scenes


def category_labels(part: dict) -> dict[str, str]:
    description = part["description"].lower()
    kind = part["family"]
    if "minifig head " in description or description == "minifig head":
        kind = "head"
    elif "gear" in description:
        kind = "gear"
    return {"family": part["family"], "kind": kind}


def discover(roots: list[Path], identities: dict[str, str]) -> tuple[list[dict], list[dict]]:
    scenes, exclusions, seen = [], [], set()
    for search in roots:
        if not search.is_dir():
            raise ValueError(f"Missing pile search root: {search}")
        for directory, subdirs, files in os.walk(search, followlinks=False):
            subdirs[:] = sorted(name for name in subdirs
                                if not name.startswith(".") and name not in
                                ("tasks", "source", "assets", "_scenes", "_meshes"))
            root = Path(directory).resolve()
            if root in seen or "model.xml" not in files:
                continue
            seen.add(root)
            kind = None
            if "problem.json" in files and "integration_state.json" in files:
                metadata_path = root / "problem.json"
                metadata = json.loads(metadata_path.read_bytes())
                if "instances" not in metadata or "scene_family_id" not in metadata:
                    exclusions.append({"source": str(root), "reason": "unsupported_problem_schema"})
                    continue
                objects = metadata["instances"]
                state_path = root / "integration_state.json"
                kind = "problem"
            elif "receipt.json" in files:
                metadata_path = root / "receipt.json"
                metadata = json.loads(metadata_path.read_bytes())
                if (metadata.get("status") == "settled" and metadata.get("objects")
                        and "integration_state" in metadata):
                    objects = metadata["objects"]
                    state_path = metadata_path
                    kind = "photo"
                elif ("scene_plan.json" in files and "integration_state.json" in files
                      and metadata.get("generator") == "native-settled-piles-v1"):
                    objects = json.loads((root / "scene_plan.json").read_bytes())["objects"]
                    state_path = root / "integration_state.json"
                    kind = "native"
            if kind is None:
                exclusions.append({"source": str(root), "reason": "not_a_supported_saved_pile"})
                continue
            xml_path = root / "model.xml"
            xml_sha = digest(xml_path)
            tree = None
            labels, recovered = [], 0
            for obj in objects:
                part_id = obj["part_id"]
                body_name = obj.get("body_name", obj["instance_id"])
                if part_id == "catalog-demo":
                    if tree is None:
                        tree = ET.fromstring(xml_path.read_bytes())
                    body = next((b for b in tree.findall(".//body")
                                 if b.get("name") == body_name), None)
                    if body is None:
                        raise ValueError(f"Missing labeled body {body_name}: {root}")
                    part_id = identities.get(visual_signature(tree, body))
                    recovered += part_id is not None
                labels.append({"instance_id": obj["instance_id"], "body_name": body_name,
                               "part_id": part_id})
            if any(not label["part_id"] for label in labels):
                exclusions.append({"source": str(root), "reason": "unresolved_exact_part_identity"})
                continue
            if len({label["body_name"] for label in labels}) != len(labels):
                raise ValueError(f"Duplicate body labels: {root}")
            state_sha, metadata_sha = digest(state_path), digest(metadata_path)
            provenance = metadata.get("provenance", {})
            keys = [f"root:{root}", f"receipt:{metadata_sha}", f"model:{xml_sha}"]
            if metadata.get("scene_family_id"):
                keys.append(f'family:{metadata["scene_family_id"]}')
            if provenance.get("receipt_sha256"):
                keys.append(f'receipt:{provenance["receipt_sha256"]}')
            if provenance.get("source"):
                keys.append(f'root:{Path(provenance["source"]).resolve()}')
            files_bound = {str(xml_path): xml_sha, str(state_path): state_sha,
                           str(metadata_path): metadata_sha}
            binary = root / "model.mjb"
            if binary.exists():
                files_bound[str(binary)] = digest(binary)
            scenes.append({
                "source": str(root), "kind": kind, "metadata": str(metadata_path),
                "state": str(state_path), "model": str(xml_path),
                "binary": str(binary) if binary.exists() else None,
                "catalog_sha256": metadata.get("catalog_sha256"),
                "labels": labels, "recovered_identities": recovered,
                "identity_basis": "explicit IDs; placeholders require exact hash-pinned template geometry",
                "files_sha256": files_bound, "group_keys": keys,
            })
    return scenes, exclusions


def prepare(roots: list[Path], catalog: Path, template: Path, output: Path, seed: int) -> dict:
    # Catalog publication imports optional simulation libraries; inventory does not need them.
    from so101.lego_parts_catalog import catalog_read_lock

    if output.exists():
        raise FileExistsError(output)
    identities = photo_identity_map(template)
    scenes, exclusions = discover(roots, identities)
    scenes = assign_splits(scenes, seed)
    requested = {label["part_id"] for scene in scenes for label in scene["labels"]}
    with catalog_read_lock(catalog):
        raw = catalog.read_bytes()
        report = json.loads(raw)
        parts = {part["part_id"]: part for part in report["parts"]}
        unsupported = sorted(part for part in requested
                             if part not in parts or parts[part]["status"] != "verified")
        if unsupported:
            raise ValueError(f"Pile IDs lack verified catalog geometry: {unsupported}")
        output.mkdir(parents=True)
        assets = output / "catalog"
        assets.mkdir()
        selected = {}
        for identity in sorted(requested):
            part = parts[identity]
            for asset in part["assets"]:
                relative = Path(asset["path"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Catalog asset escapes its root")
                source = catalog.parent / relative
                if source.is_symlink():
                    raise ValueError("Catalog assets cannot be symlinks")
                target = assets / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(read_bound(source, asset["sha256"]))
            # Validate the visual adapter now, before a later render starts.
            visual_geometry((assets / part["mjcf"]).read_bytes())
            selected[identity] = {**part, "categories": category_labels(part)}
        write_new_json(assets / "catalog.json", {
            "schema_version": 1, "catalog_id": "recognition-subset",
            "parent_manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "parts": list(selected.values()),
        })
    counts = {split: Counter(label["part_id"] for scene in scenes if scene["split"] == split
                             for label in scene["labels"])
              for split in ("train", "development", "test")}
    result = {
        "schema_version": 1, "kind": "pile-recognition-plan", "seed": seed,
        "catalog_sha256": hashlib.sha256(raw).hexdigest(),
        "catalog_subset": str((assets / "catalog.json").resolve()),
        "catalog_subset_sha256": digest(assets / "catalog.json"),
        "photo_template_sha256": PHOTO_TEMPLATE_SHA256,
        "search_roots": [str(path.resolve()) for path in roots],
        "scenes": scenes, "exclusions": exclusions,
        "parts": {key: {"categories": value["categories"],
                        "description": value["description"]} for key, value in selected.items()},
        "split_instance_counts": {key: dict(value) for key, value in counts.items()},
        "test_only_ids": sorted(set(counts["test"]) - set(counts["train"])),
        "scope": "Synthetic saved-pile recognition feasibility; not action SFT, real transfer or RL.",
    }
    write_new_json(output / "plan.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pile-root", type=Path, action="append", required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--photo-template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=61000)
    args = parser.parse_args()
    report = prepare(args.pile_root, args.catalog, args.photo_template, args.output, args.seed)
    print(json.dumps({"scenes": len(report["scenes"]), "parts": len(report["parts"]),
                      "excluded": len(report["exclusions"]),
                      "test_only_ids": report["test_only_ids"]}))


if __name__ == "__main__":
    main()
