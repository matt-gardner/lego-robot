import importlib.util
import json
from pathlib import Path

import pytest
import torch


spec = importlib.util.spec_from_file_location(
    "prepare_rl_transition",
    Path(__file__).parents[1] / "scripts/prepare_rl_transition.py",
)
transition = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transition)


@pytest.mark.parametrize("source", ["data/simulation/example", "../escape"])
def test_transition_preserves_reference_bytes_and_rejects_unsafe_paths(tmp_path, monkeypatch, source):
    state_path = tmp_path / "state.pt"
    model = {"model.lm_expert.weight": torch.ones(2, dtype=torch.bfloat16)}
    torch.save({
        "step": 7656, "model": model,
        "log": {
            "run_config": {"history_steps": 4, "depth_mode": "known"},
            "dataset": {"normalization": {"action_mean": [0.] * 6}},
        },
    }, state_path)
    monkeypatch.setattr(transition, "validate_state", lambda state: None)
    database = tmp_path / "episodes.sqlite"
    database.write_bytes(b"immutable test database")
    contents = {"episode.json": b'{"task":"grasp"}\n', "frames.jsonl": b'{"action":[1,2,3]}\n'}

    class Storage:
        catalog = {"successful_episodes": [source]}

        def __init__(self, path):
            assert path == database

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, path, name):
            assert str(path) == source
            return contents[name]

    monkeypatch.setattr(transition, "EpisodeStorage", Storage)
    output = tmp_path / "rl"
    original = state_path.read_bytes()
    if source.startswith(".."):
        with pytest.raises(ValueError, match="Unsafe corpus path"):
            transition.prepare(state_path, database, output)
        assert not (tmp_path / "escape").exists()
    else:
        receipt = transition.prepare(state_path, database, output)
        assert receipt["sft_step"] == 7656
        assert (output / source / "frames.jsonl").read_bytes() == contents["frames.jsonl"]
        exported = torch.load(output / "initial_policy/model.pt", weights_only=True)
        assert torch.equal(exported["model.lm_expert.weight"], model["model.lm_expert.weight"])
        metadata = json.loads((output / "initial_policy/metadata.json").read_text())
        assert metadata["action_expert_dtype"] == "bfloat16"
    assert state_path.read_bytes() == original
