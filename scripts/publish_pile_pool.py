"""Publish an immutable manifest of admitted native pile scenes across producer roots."""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(roots: list[Path]) -> dict:
    scenes, sources = [], []
    for root in roots:
        progress = json.loads((root/"progress.json").read_bytes())
        sources.append({
            "root": str(root),
            "catalog_sha256": sha256(Path(progress["catalog"])),
            **{key: progress.get(key) for key in (
                "generator", "started_at", "budget_profile", "controller_sha256",
                "qualification", "compact_stage_roots")},
            "attempts": len(progress.get("attempts", [])),
            "rejections": sum(a["status"] != "complete" for a in progress.get("attempts", [])),
        })
        for admission in sorted((root/"piles").glob("scene_*/admission.json")):
            entry = json.loads(admission.read_bytes())
            for name, digest in entry["files_sha256"].items():
                if sha256(admission.parent/name) != digest:
                    raise ValueError(f"{admission.parent/name} does not match its admission hash")
            scenes.append({
                "admission": str(admission), "admission_sha256": sha256(admission),
                "scene_id": entry["scene_id"], "piece_count": entry["tags"]["piece_count"],
                "sampling_profile": json.loads((admission.parent/"scene_plan.json").read_bytes())
                .get("sampling_profile"),
            })
    if len({scene["scene_id"] for scene in scenes}) != len(scenes):
        raise ValueError("Duplicate scene IDs across roots")
    return {
        "published_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "qualification": "TRAIN reset/task start states; no expert references or SFT qualification",
        "sources": sources,
        "counts_by_piece_count": dict(sorted(collections.Counter(s["piece_count"] for s in scenes).items())),
        "counts_by_sampling_profile": dict(collections.Counter(s["sampling_profile"] for s in scenes)),
        "scenes": scenes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True, help="Directory for snapshot manifests")
    args = parser.parse_args()
    manifest = snapshot(args.root)
    args.output.mkdir(parents=True, exist_ok=True)
    stamp = manifest["published_at"].replace(":", "").split(".")[0]
    target = args.output/f"pile-pool-{stamp}.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    os.replace(temporary, target)
    latest = args.output/"latest.json.tmp"
    latest.write_text(json.dumps({"manifest": str(target), "sha256": sha256(target)}))
    os.replace(latest, args.output/"latest.json")
    print(target, manifest["counts_by_piece_count"], manifest["counts_by_sampling_profile"])


if __name__ == "__main__":
    main()
