import json
from pathlib import Path

import pytest

from so101.cloud_checkpoint import sha256
from so101 import rebuild_physical_dataset as rebuild
from so101.robot_sim import MODEL_ROOT, SCENE_VERSION
from so101.simulation_corpus import load_catalog


@pytest.fixture
def source(tmp_path):
    if not (MODEL_ROOT / "revision.json").exists():
        pytest.skip("Download the pinned simulation model")
    root = tmp_path / "legacy"
    root.mkdir()
    paths = []
    for task in ("grasp", "view", "push", "place"):
        for repetition in range(2):
            episode = root / f"{task}_{repetition}"
            episode.mkdir()
            (episode / "episode.json").write_text(json.dumps({
                "task": task, "target": "short", "pair_id": f"90000:{task}",
                "destination": "goal_right" if task in ("push", "place") else None,
                "difficulty": {"level": 0},
                "initial": {"seed": 90000, "appearance": {
                    "target_labels": {"short": "teal short beam"},
                }},
            }))
            paths.append(str(episode))
    (root / "collection.json").write_text(json.dumps({"successful_episodes": paths}))
    return root


def test_regeneration_publishes_checked_new_data_and_maps_split_without_rewriting_sources(source, tmp_path):
    original = json.loads((source / "collection.json").read_text())
    before = {path: sha256(path) for path in source.rglob("episode.json")}
    split = tmp_path / "split.json"
    split.write_text(json.dumps({
        "training": {
            task: [path for path in original["successful_episodes"][:-1]
                   if json.loads((Path(path) / "episode.json").read_text())["task"] == task]
            for task in ("grasp", "view", "push", "place")
        },
        "validation": original["successful_episodes"][-1:],
        "expert_bounds": {"place": {"optimal_steps": 999}},
    }))
    output = tmp_path / "corrected"
    report = rebuild.regenerate(source, output, validation_split=split, publish_catalog=True)
    assert report["status"] == "complete"
    assert report["task_counts"] == {"grasp": 2, "view": 2, "push": 2, "place": 2}
    assert len(report["successful_episodes"]) == 8
    assert report["consumer_qualification"]["sft_episode_count"] == 8
    assert report["consumer_qualification"]["rl_tasks"] == report["task_counts"]
    assert before == {path: sha256(path) for path in before}
    assert json.loads((output / "historical" / "collection.json").read_text()) == original
    active = json.loads((source / "collection.json").read_text())
    assert active["scene_version"] == SCENE_VERSION
    assert set(active["successful_episodes"]).isdisjoint(original["successful_episodes"])
    mapped = json.loads((output / "data_split.json").read_text())
    assert mapped["validation"] == [report["replacements"][original["successful_episodes"][-1]]]
    assert mapped["expert_bounds"]["place"]["optimal_steps"] != 999
    assert not mapped["provenance"]["historical_metrics_comparable"]
    for path in report["successful_episodes"]:
        metadata, frames = rebuild.validate_episode(Path(path))
        assert frames > 2
        assert metadata["target_label"] == "teal short beam"
        assert metadata["maximum_contact_penetration_m"] <= rebuild.CONTACT_LIMIT_M
        assert metadata["maximum_table_penetration_m"] <= rebuild.TABLE_CONTACT_LIMIT_M
    resumed = rebuild.regenerate(source, output, validation_split=split, publish_catalog=True)
    assert resumed["attempts"] == report["attempts"]


def test_interrupted_recording_resumes_owned_partial_output(source, tmp_path, monkeypatch):
    catalog = json.loads((source / "collection.json").read_text())
    (source / "collection.json").write_text(json.dumps({
        "successful_episodes": [catalog["successful_episodes"][2]],
    }))
    output = tmp_path / "corrected"
    demonstrate = rebuild.demonstrate
    calls = 0

    def interrupted(sim, task, config):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("interrupted recording")
        return demonstrate(sim, task, config)

    monkeypatch.setattr(rebuild, "demonstrate", interrupted)
    with pytest.raises(OSError, match="interrupted recording"):
        rebuild.regenerate(source, output)
    assert (output / "episode_00000" / "attempt_00" / "wrist").is_dir()
    monkeypatch.setattr(rebuild, "demonstrate", demonstrate)
    assert rebuild.regenerate(source, output)["status"] == "complete"


def test_partial_dataset_is_not_published_and_changed_physics_cannot_mix_with_it(source, tmp_path, monkeypatch):
    original_hash = sha256(source / "collection.json")
    output = tmp_path / "corrected"
    with pytest.raises(ValueError, match="incomplete"):
        rebuild.regenerate(source, output, limit=1, publish_catalog=True)
    assert sha256(source / "collection.json") == original_hash
    with pytest.raises(ValueError, match="not complete"):
        load_catalog(output)
    monkeypatch.setattr(rebuild, "generator_fingerprint", lambda _: "different physics")
    with pytest.raises(ValueError, match="physics changed"):
        rebuild.regenerate(source, output)
