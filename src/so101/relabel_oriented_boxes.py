from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np

from so101.robot_sim import BeamSimulation, LEGACY_SCENE_VERSION, MODEL_ROOT


def relabel(
    source: Path,
    destination: Path,
    model_root: Path,
    limit: int | None,
    resume: bool,
) -> None:
    report = json.loads((source / "collection.json").read_text())
    episodes = report["successful_episodes"]
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        episodes = episodes[:limit]
    destination.mkdir(parents=True, exist_ok=resume)
    relabeled = []
    sim = BeamSimulation(model_root)
    try:
        for episode_index, source_name in enumerate(episodes):
            source_episode = Path(source_name)
            metadata = json.loads((source_episode / "episode.json").read_text())
            destination_episode = destination / source_episode.name
            if (
                resume
                and (destination_episode / "frames.jsonl").exists()
                and (destination_episode / "episode.json").exists()
            ):
                relabeled.append(str(destination_episode))
                continue
            if destination_episode.exists():
                shutil.rmtree(destination_episode)
            rows = [
                json.loads(line)
                for line in (source_episode / "frames.jsonl").read_text().splitlines()
            ]
            sim.use_scene_version(int(metadata["initial"].get("scene_version", LEGACY_SCENE_VERSION)))
            sim.reset(
                metadata["initial"]["seed"],
                varied_layout=metadata["initial"]["varied_layout"],
                real_scene=metadata["real_scene"],
                color_targets=metadata["color_targets"],
                visual_cue_target=metadata["visual_cue_target"],
            )
            sim.active_target = metadata["target"]
            destination_episode.mkdir()
            os.symlink(
                (source_episode / "wrist").resolve(),
                destination_episode / "wrist",
                target_is_directory=True,
            )
            overview = source_episode / "diagnostic_overview.jpg"
            if overview.exists():
                os.symlink(
                    overview.resolve(),
                    destination_episode / "diagnostic_overview.jpg",
                )
            output_rows = []
            for row in rows:
                np.testing.assert_allclose(
                    sim.state(),
                    row["state"],
                    atol=.25,
                    err_msg=f"{source_episode} frame {row['index']} diverged during replay",
                )
                oriented_box = sim.target_oriented_box()
                updated = {
                    **row,
                    "target_oriented_box": oriented_box,
                    "command_boxes": [oriented_box[:5]],
                    "command_visibility": [oriented_box[5]],
                }
                output_rows.append(updated)
                sim.step(sim.decode(np.asarray(row["action"], dtype=np.float32)), row["phase"])
            (destination_episode / "frames.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in output_rows)
            )
            metadata["box_representation"] = (
                "cx, cy, diagonal-normalized mean extent, anisotropy*cos(2theta), "
                "anisotropy*sin(2theta), visibility"
            )
            (destination_episode / "episode.json").write_text(
                json.dumps(metadata, indent=2)
            )
            relabeled.append(str(destination_episode))
            print(json.dumps({
                "episode": episode_index,
                "source": str(source_episode),
                "frames": len(rows),
            }), flush=True)
    finally:
        sim.close()
    output_report = {
        **report,
        "requested": len(relabeled),
        "successful_episodes": relabeled,
        "source_collection": str(source),
        "box_representation": (
            "cx, cy, diagonal-normalized mean extent, anisotropy*cos(2theta), "
            "anisotropy*sin(2theta), visibility"
        ),
    }
    (destination / "collection.json").write_text(json.dumps(output_report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    relabel(args.source, args.destination, args.model_root, args.limit, args.resume)


if __name__ == "__main__":
    main()
