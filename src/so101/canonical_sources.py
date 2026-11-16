"""Resolve immutable historical source bindings without rewriting their hashes."""
import json
from pathlib import Path


def canonical_source_root(source: Path) -> Path:
    if not source.is_absolute():
        raise ValueError("Historical source root must be absolute")
    for ancestor in source.parents:
        manifest = ancestor / "piles" / "source-paths.json"
        if not manifest.is_file():
            continue
        mapping = json.loads(manifest.read_bytes())
        if mapping["schema_version"] != 1:
            raise ValueError("Unsupported canonical source mapping schema")
        destination = mapping["roots"].get(str(source))
        if destination is None:
            return source
        target = Path(destination)
        piles = ancestor / "piles"
        if not target.is_absolute() or not target.resolve().is_relative_to(piles.resolve()):
            raise ValueError("Canonical source mapping escapes the pile directory")
        current = piles
        for component in target.relative_to(piles).parts:
            current = current / component
            if current.is_symlink():
                raise ValueError("Canonical source mapping contains a symlink")
        if not target.is_dir():
            raise ValueError("Canonical source mapping destination is missing")
        return target
    return source
