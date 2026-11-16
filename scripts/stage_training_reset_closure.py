"""Stage verified retained training reset sources beneath their canonical piles."""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.audit_legacy_lego_dependencies import FILES, digest
from scripts.import_pile_directories import atomic_json, hardlink, make_directory


def stage(root: Path, audit_path: Path) -> dict:
    report = json.loads(audit_path.read_bytes())
    mapping = {entry["source"]: entry["canonical"] for entry in report["canonical"]}
    pending = dict(report["dependencies"])
    while pending:
        progress = False
        for source_name, binding in list(pending.items()):
            source = Path(source_name)
            for name in FILES:
                if digest(source / name) != binding["files_sha256"][name]:
                    raise ValueError(f"Source changed after audit: {source / name}")
            if source_name not in mapping:
                scene = json.loads((source / "problem.json").read_bytes())
                parent = scene.get("provenance", {}).get("parent_entry")
                if not parent or parent["root"] not in mapping:
                    continue
                parent_target = Path(mapping[parent["root"]])
                relative = parent_target.relative_to(root / "piles")
                pile = root / "piles" / relative.parts[0]
                target = pile / "checkpoints" / (
                    "training_" + hashlib.sha256(source_name.encode()).hexdigest())
                make_directory(target, root)
                for name in FILES:
                    hardlink(source / name, target / name)
                atomic_json(target / "source.json", {
                    "original_root": source_name, **binding,
                    "qualification": "Original retained training input; not newly qualified",
                })
                mapping[source_name] = str(target)
            target = Path(mapping[source_name])
            for name in FILES:
                if digest(target / name) != binding["files_sha256"][name]:
                    raise ValueError(f"Canonical training source differs: {target / name}")
            del pending[source_name]
            progress = True
        if not progress:
            raise ValueError(f"Training sources have no exact canonical parent: {list(pending)}")
    destination = root / "piles" / "source-paths.json"
    atomic_json(destination, {"schema_version": 1, "roots": mapping,
                             "files": report["dependencies"]})
    return {"roots": len(mapping), "manifest": str(destination)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(stage(args.root, args.audit)))


if __name__ == "__main__":
    main()
