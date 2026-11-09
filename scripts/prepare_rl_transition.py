"""Materialize immutable SFT initialization and lightweight RL references."""
import argparse
import json
from pathlib import Path

import torch

from so101.cloud_checkpoint import atomic_json, sha256, validate_state
from so101.episode_storage import EpisodeStorage


def prepare(state_path: Path, database: Path, destination: Path) -> dict:
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    validate_state(state)
    destination.mkdir(parents=True, exist_ok=False)
    checkpoint = destination / "initial_policy"
    checkpoint.mkdir()
    torch.save(state["model"], checkpoint / "model.pt")
    config = state["log"]["run_config"]
    atomic_json(checkpoint / "entity_config.json", {
        "max_entities": 4,
        "history_steps": config["history_steps"],
        "role_token_length": 8,
        "depth_input": config["depth_mode"] != "blind",
    })
    expert = next(value for name, value in state["model"].items()
                  if "lm_expert" in name and value.is_floating_point())
    atomic_json(checkpoint / "metadata.json", {
        "source_checkpoint": "lerobot/smolvla_base",
        "normalization": state["log"]["dataset"]["normalization"],
        "action_expert_dtype": str(expert.dtype).removeprefix("torch."),
    })
    with EpisodeStorage(database) as storage:
        catalog = storage.catalog
        for source in catalog["successful_episodes"]:
            relative = Path(source)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe corpus path: {source}")
            target = destination / relative
            target.mkdir(parents=True)
            for name in ("episode.json", "frames.jsonl"):
                content = storage.read(relative, name)
                if name == "episode.json":
                    json.loads(content)
                (target / name).write_bytes(content)
        collection = destination / "data/simulation/rl_transition"
        collection.mkdir(parents=True)
        atomic_json(collection / "collection.json", catalog)
    receipt = {
        "sft_step": state["step"], "sft_state_sha256": sha256(state_path),
        "database_sha256": sha256(database),
        "episodes": len(catalog["successful_episodes"]),
        "model_sha256": sha256(checkpoint / "model.pt"),
        "source": "SFT model only; RL starts a new optimizer and sampling RNG",
    }
    atomic_json(destination / "transition.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.state, args.database, args.destination)))
