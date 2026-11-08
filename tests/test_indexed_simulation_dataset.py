import json

import cv2
import numpy as np
import pytest
import torch

from so101.control_model.simulation_dataset import EntityTrajectoryDataset
from so101.episode_storage import consolidate
from so101.benchmark_indexed_loader import benchmark


def make_collection(tmp_path):
    sources = []
    for index in range(4):
        source = tmp_path / f"episode_{index}"
        source.mkdir()
        sources.append(str(source))
        (source / "episode.json").write_text(json.dumps({
            "task": "grasp", "pair_id": str(index // 2), "initial": {},
        }))
        rows = []
        for frame in range(3):
            image = f"{frame}.jpg"
            assert cv2.imwrite(str(source / image), np.full((8, 8, 3), index * 30 + frame, np.uint8))
            rows.append({
                "image": image, "phase": "approach",
                "state": [float(index + frame)] * 6,
                "action": [float(index - frame)] * 6,
                "command_boxes": [[.5, .5, .1, 0., 0.], [.3, .4, .2, 0., 0.]],
                "command_visibility": [1, 1], "command_depth_m": [.2, .3],
                "command_depth_valid": [1, 1], "command_roles": ["object", "object"],
            })
        (source / "frames.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (tmp_path / "collection.json").write_text(json.dumps({"successful_episodes": sources}))
    destination = tmp_path / "episodes-v1.sqlite"
    consolidate(tmp_path, destination)
    return sources, destination


@pytest.mark.parametrize("held_out", [False, True])
def test_seeded_examples_splits_normalization_and_pairs_exact(tmp_path, held_out):
    sources, destination = make_collection(tmp_path)
    excluded = tuple(sources[:2] if held_out else ())
    original = EntityTrajectoryDataset(tmp_path, 4, 3, excluded_sources=excluded)
    indexed = EntityTrajectoryDataset(
        tmp_path, 4, 3, excluded_sources=excluded, storage_path=destination,
    )
    assert original.episodes == indexed.episodes
    assert original.samples == indexed.samples
    assert original.normalization() == indexed.normalization()
    assert original.pair_groups == indexed.pair_groups
    for method in ("paired_indices", "balanced_indices"):
        assert getattr(original, method)(np.random.default_rng(61000), 64) == getattr(
            indexed, method
        )(np.random.default_rng(61000), 64)
    for index in range(len(original)):
        np.random.seed(61000 + index)
        expected = original[index]
        np.random.seed(61000 + index)
        actual = indexed[index]
        for key in expected:
            if isinstance(expected[key], torch.Tensor):
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
            else:
                assert actual[key] == expected[key]
    for source in sources:
        for file in tmp_path.joinpath(source).iterdir():
            file.unlink()
    assert indexed[0]["images"].shape == (3, 3, 8, 8)


def test_catalog_mismatch_rejected(tmp_path):
    sources, destination = make_collection(tmp_path)
    (tmp_path / "collection.json").write_text(json.dumps({"successful_episodes": sources[::-1]}))
    with pytest.raises(ValueError, match="catalog"):
        EntityTrajectoryDataset(tmp_path, 4, 3, storage_path=destination)


def test_indexed_storage_preserves_clean_expert_rejection(tmp_path):
    sources, _ = make_collection(tmp_path)
    source = tmp_path / sources[0]
    metadata = json.loads((source / "episode.json").read_bytes())
    metadata.update(outcome="success", initial={"expert_acceptance_version": 1})
    (source / "episode.json").write_text(json.dumps(metadata))
    rows = (source / "frames.jsonl").read_text().replace("approach", "grasp_retry")
    (source / "frames.jsonl").write_text(rows)
    destination = tmp_path / "retry-v1.sqlite"
    consolidate(tmp_path, destination)
    for storage_path in (None, destination):
        with pytest.raises(ValueError, match="failed pass/retry"):
            EntityTrajectoryDataset(tmp_path, 4, 3, storage_path=storage_path)


def test_cpu_benchmark_verifies_training_and_frozen_probe(tmp_path):
    sources, destination = make_collection(tmp_path)
    result = benchmark(tmp_path, destination, tuple(sources[:2]), 2)
    for backend in ("original", "container"):
        measured = result["results"][backend]
        assert measured["episodes"] == 2
        assert measured["samples"] == 6
        assert measured["access"]["random"]["examples"] == 2
        assert measured["frozen_probe"]["examples"] == 8
    assert len(result["equivalent_sha256"]) == 64


def test_cpu_benchmark_rejects_changed_example_pixels(tmp_path):
    sources, destination = make_collection(tmp_path)
    assert cv2.imwrite(str(tmp_path / sources[0] / "0.jpg"), np.full((8, 8, 3), 255, np.uint8))
    with pytest.raises(ValueError, match="example content differs"):
        benchmark(tmp_path, destination, (), 1)
