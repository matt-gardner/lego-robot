"""Policy-free simulation catalog loading and trajectory reference statistics."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

TASKS = ("grasp", "view", "push", "place")


@dataclass(frozen=True)
class CorpusEpisode:
    source: Path
    metadata: dict[str, Any]

    @property
    def task(self) -> str:
        return str(self.metadata["task"])


def load_catalog(collection: Path) -> dict:
    report = json.loads((collection / "collection.json").read_text())
    if "status" in report and report["status"] != "complete":
        raise ValueError(f"Simulation dataset is not complete: {collection}")
    return report


def load_corpus(collection: Path) -> dict[str, list[CorpusEpisode]]:
    report = load_catalog(collection)
    episodes: dict[str, list[CorpusEpisode]] = defaultdict(list)
    for source_name in report["successful_episodes"]:
        source = Path(source_name)
        metadata = json.loads((source / "episode.json").read_text())
        task = metadata["task"]
        if task in TASKS and metadata.get("outcome", "success") == "success":
            if "scene_version" in report and metadata["initial"]["scene_version"] != report["scene_version"]:
                raise ValueError(f"Mixed physics versions in simulation dataset: {source}")
            episodes[task].append(CorpusEpisode(source, metadata))
    missing = [task for task in TASKS if not episodes[task]]
    if missing:
        raise ValueError(f"Corpus has no episodes for: {missing}")
    return dict(episodes)


def episode_actions(episode: CorpusEpisode) -> np.ndarray:
    with (episode.source / "frames.jsonl").open() as stream:
        actions = [json.loads(line)["action"][:5] for line in stream]
    return np.asarray(actions, dtype=np.float32)


def expert_bounds(episodes: dict[str, list[CorpusEpisode]]) -> dict[str, dict[str, float]]:
    result = {}
    for task, task_episodes in episodes.items():
        lengths, jerks = [], []
        for episode in task_episodes:
            actions = episode_actions(episode)
            lengths.append(len(actions))
            jerk = np.diff(actions, n=3, axis=0)
            jerks.append(float(np.sqrt(np.mean(jerk ** 2))) if jerk.size else 0.)
        result[task] = {
            "optimal_steps": float(np.quantile(lengths, .1)),
            "jerk_p10": float(np.quantile(jerks, .1)),
            "jerk_p90": float(np.quantile(jerks, .9)),
        }
    return result
