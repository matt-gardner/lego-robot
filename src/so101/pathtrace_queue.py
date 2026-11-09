"""Run a filtered archive queue without changing its frozen renderer identity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from so101.cloud_checkpoint import atomic_json
from so101.episode_storage import EpisodeStorage
from so101.pathtrace_pile import file_hash
from so101.pathtrace_recordings import rerender_release


def select_tasks(manifest: dict, excluded_tasks: set[str]) -> dict:
    selected, excluded = [], []
    for release in manifest["releases"]:
        tasks = {episode["task"] for episode in release["episodes"]}
        if tasks & excluded_tasks:
            if not tasks <= excluded_tasks:
                raise ValueError("Cannot exclude only one command of a paired release")
            excluded.append(release)
        else:
            selected.append(release)
    return {
        **manifest, "releases": selected,
        "episodes": sum(len(r["episodes"]) for r in selected),
        "frames": sum(e["frames"] for r in selected for e in r["episodes"]),
        "excluded_tasks": sorted(excluded_tasks),
        "excluded_releases": excluded,
    }


def run_queue(output: Path, excluded_tasks: set[str]) -> None:
    original = output / "inventory.json"
    manifest = select_tasks(json.loads(original.read_text()), excluded_tasks)
    profile = json.loads((output / "profile.json").read_text())
    identity = {
        "original_inventory_sha256": file_hash(original),
        "queue_implementation_sha256": file_hash(Path(__file__)),
        "excluded_tasks": sorted(excluded_tasks),
        "renderer_profile_sha256": file_hash(output / "profile.json"),
    }
    identity_path = output / "queue_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Existing queue has a different implementation or selection")
    atomic_json(identity_path, identity)
    atomic_json(output / "selected_inventory.json", manifest)
    for release in manifest["excluded_releases"]:
        if (output / release["release_id"]).exists():
            raise ValueError("Excluded derivative exists; resolve its ownership before continuing")
    for index, entry in enumerate(manifest["releases"]):
        destination = output / entry["release_id"]
        atomic_json(output / "queue_progress.json", {
            "status": "rendering", "release_index": index,
            "release_id": entry["release_id"], "total_releases": len(manifest["releases"]),
            "total_frames": manifest["frames"],
        })
        if (destination / "qualification.json").exists():
            with EpisodeStorage(destination / "episodes.sqlite") as storage:
                storage.verify()
                storage.verify_sources(destination)
        else:
            rerender_release(entry, destination, profile)
    atomic_json(output / "completion.json", {
        "status": "complete", "episodes": manifest["episodes"], "frames": manifest["frames"],
        "releases": len(manifest["releases"]), "excluded_tasks": sorted(excluded_tasks),
        "queue_identity": identity,
    })
    atomic_json(output / "queue_progress.json", {"status": "complete"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-task", action="append", default=[], choices=["view"])
    args = parser.parse_args()
    run_queue(args.output, set(args.exclude_task))


if __name__ == "__main__":
    main()
