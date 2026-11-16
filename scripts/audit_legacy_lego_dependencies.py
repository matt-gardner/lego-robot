"""Verify canonical and accepted-training reset closure before legacy retirement."""
import argparse
import hashlib
import json
from pathlib import Path

FILES = ("model.mjb", "model.xml", "problem.json", "integration_state.json", "scene_plan.json")


def digest(path: Path) -> str:
    stat = path.stat()
    return inode_digest(path, stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


_digests: dict[tuple[int, int, int, int], str] = {}


def inode_digest(path: Path, device: int, inode: int, size: int, mtime: int) -> str:
    key = (device, inode, size, mtime)
    if key in _digests:
        return _digests[key]
    with path.open("rb") as stream:
        value = hashlib.file_digest(stream, "sha256").hexdigest()
    _digests[key] = value
    return value


def audit(root: Path, training_status: Path, assistance_root: Path) -> dict:
    dependencies = {}

    def preserve(source: Path, hashes: dict | None = None):
        source = source.absolute()
        key = str(source)
        if key in dependencies:
            if hashes and any(dependencies[key]["files_sha256"][name] != value
                              for name, value in hashes.items()):
                raise ValueError(f"Conflicting source binding: {source}")
            return
        actual = {name: digest(source / name) for name in FILES}
        if hashes and any(actual[name] != expected for name, expected in hashes.items()):
            raise ValueError(f"Changed source closure: {source}")
        dependencies[key] = {"files_sha256": actual}
        scene = json.loads((source / "problem.json").read_bytes())
        provenance = scene.get("provenance", {})
        parent = provenance.get("parent_entry")
        if parent:
            preserve(Path(parent["root"]), parent["files_sha256"])

    canonical = []
    for path in sorted((root / "piles").glob("pile_*/pile.json")):
        pile = json.loads(path.read_bytes())
        source = Path(pile["source"]["original_root"])
        preserve(source, pile["source"]["files_sha256"])
        for name, sha in pile["source"]["files_sha256"].items():
            if digest(path.parent / name) != sha:
                raise ValueError(f"Canonical source differs: {path.parent / name}")
        canonical.append({"source": str(source), "canonical": str(path.parent)})
        for task_path in sorted((path.parent / "tasks").glob("task_*.json")):
            task = json.loads(task_path.read_bytes())
            for checkpoint in task["assistance_checkpoints"]:
                source = Path(checkpoint["source"]["path"]).parent
                preserve(source, checkpoint["files_sha256"])
                target = (task_path.parent / checkpoint["directory"]).resolve()
                for name, sha in checkpoint["files_sha256"].items():
                    if digest(target / name) != sha:
                        raise ValueError(f"Canonical checkpoint differs: {target / name}")
                canonical.append({"source": str(source), "canonical": str(target)})
    status = json.loads(training_status.read_bytes())
    accepted = [report for report in status["reports"] if report.get("accepted")]
    for report in accepted:
        entry = report["entry"]
        preserve(Path(entry["root"]), entry["files_sha256"])
        trace = Path(report["trace"])
        if str(trace).startswith("/mnt/data/"):
            trace = training_status.parents[2] / trace.relative_to("/mnt/data")
        if digest(trace) != report["trace_sha256"]:
            raise ValueError(f"Accepted training trace changed: {trace}")
        trace_data = json.loads(trace.read_bytes())
        trace_entry = trace_data.get("entry")
        if trace_entry:
            preserve(Path(trace_entry["root"]), trace_entry["files_sha256"])
    inventory = json.loads((assistance_root / "inventory.json").read_bytes())
    for scene in inventory["scenes"].values():
        if digest(Path(scene["admission"])) != scene["admission_sha256"]:
            raise ValueError("Active assistance admission changed")
        entry = scene["entry"]
        preserve(Path(entry["root"]), entry["files_sha256"])
    return {
        "accepted_training_parents": len(accepted),
        "active_assistance_scenes": len(inventory["scenes"]),
        "active_assistance_roots": [scene["entry"]["root"]
                                    for scene in inventory["scenes"].values()],
        "dependencies": dependencies, "canonical": canonical,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--training-status", type=Path, required=True)
    parser.add_argument("--assistance-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root, args.training_status, args.assistance_root)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({key: len(value) if isinstance(value, (dict, list)) else value
                      for key, value in result.items()}))


if __name__ == "__main__":
    main()
