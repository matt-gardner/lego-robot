"""Publish verified continuous TRAIN problems for asynchronous learner intake."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from so101.cloud_checkpoint import atomic_json
from so101.pile_task_runtime import load_problem


def publish_pool(root: Path) -> dict:
    entries, assets = [], {}
    seen = set()
    for admission in sorted(root.glob("scene_*/admission.json")):
        entry = json.loads(admission.read_text())
        if entry["split"] != "train" or entry["scene_id"] in seen:
            raise ValueError("Continuous pool contains non-TRAIN or duplicate scenes")
        seen.add(entry["scene_id"])
        scene_root = Path(entry["root"])
        if scene_root.resolve() != admission.parent.resolve():
            raise ValueError("Admission root differs from its stored location")
        scene = json.loads((scene_root / "problem.json").read_text())
        if not scene["tasks"]:
            raise ValueError("Admitted scene has no tasks")
        for task in scene["tasks"]:
            load_problem(entry, task["task_id"])
        tree = ET.parse(scene_root / "model.xml")
        compiler = tree.find("compiler")
        meshdir = Path(compiler.get("meshdir", ".")) if compiler is not None else Path(".")
        for node in tree.iter():
            if "file" not in node.attrib:
                continue
            path = Path(node.attrib["file"])
            if not path.is_absolute():
                path = meshdir / path
            if not path.is_absolute():
                path = scene_root / path
            path = path.resolve(strict=True)
            assets[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append(entry)
    manifest = {
        "schema_version": 1, "kind": "lego-pile-task-pool", "status": "complete",
        "expert_references": False, "scenes": entries, "xml_asset_files_sha256": assets,
    }
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    snapshot = root / f"pool-{digest}.json"
    if snapshot.exists():
        if json.loads(snapshot.read_text()) != manifest:
            raise ValueError("Immutable pool snapshot changed")
    else:
        atomic_json(snapshot, manifest)
    atomic_json(root / "current-pool.json", manifest)
    return {"manifest_path": str(snapshot.resolve()),
            "sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
            "scenes": len(entries)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(publish_pool(args.root)), flush=True)


if __name__ == "__main__":
    main()
