import json
from collections import Counter

from so101.produce_catalog import run, scheduled_scene


def test_schedule_prioritizes_manipulation_and_reserves_views_for_dense_piles():
    plans = [scheduled_scene(sequence) for sequence in range(48)]
    assert Counter(task for _, task, _ in plans) == {
        "grasp": 15, "place": 15, "push": 15, "view": 3,
    }
    assert all(category == "pile" and count == 8
               for category, task, count in plans if task == "view")
    assert {(category, task) for category, task, _ in plans if task != "view"} == {
        (category, task)
        for category in ("separated", "cluster", "overlap", "pile")
        for task in ("grasp", "place", "push")
    }


def test_worker_resumes_completed_pair_without_generating_again(tmp_path, monkeypatch):
    root = tmp_path / "producer"
    catalog = tmp_path / "catalog.json"
    appearance = tmp_path / "appearance.json"
    catalog.write_text("{}")
    appearance.write_text("{}")
    monkeypatch.setattr("so101.produce_catalog.load_catalog", lambda path: object())
    monkeypatch.setattr("so101.produce_catalog.load_appearance", lambda path: object())
    monkeypatch.setattr("so101.produce_catalog.prepare_scene", lambda *args, **kwargs: {})
    from so101.produce_catalog import recipe
    root.mkdir()
    (root / "producer.json").write_text(json.dumps({
        "recipe": recipe(catalog, appearance, [], 7), "next_sequence": 0, "attempts": [],
    }))
    attempt = root / "attempt_000000000"
    attempt.mkdir()
    import hashlib
    family = hashlib.sha256(json.dumps(
        ["new-catalog-development-v1", 7, 0], separators=(",", ":")
    ).encode()).hexdigest()
    report = {
        "status": "complete", "scene_family_id": family, "split": "development",
        "tags": {"category": "separated"}, "release_id": "fixture",
        "successful_episodes": ["/fixture/0", "/fixture/1"],
        "attempts": [
            {"example_id": str(i), "source": f"/fixture/{i}", "outcome": "success"}
            for i in range(2)
        ],
    }
    (attempt / "collection.json").write_text(json.dumps(report))
    monkeypatch.setattr("so101.produce_catalog.qualify_pair",
                        lambda path: {"storage_sha256": "fixture", "samples": 4})
    monkeypatch.setattr("so101.produce_catalog.publish_pair", lambda *args: "/release")
    monkeypatch.setattr("so101.produce_catalog.CatalogSimulation",
                        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("rerun")))
    run(root, catalog, appearance, [], 7,
        lambda: json.loads((root / "producer.json").read_text())["next_sequence"] == 1)
    state = json.loads((root / "producer.json").read_text())
    assert state["attempts"][0]["status"] == "qualified"
    assert state["next_sequence"] == 1
