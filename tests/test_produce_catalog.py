import json
from collections import Counter
from types import SimpleNamespace

from so101.produce_catalog import DISK_FREE_FLOOR, DiskBackpressure, run, scheduled_scene
from so101.catalog_sim import CatalogSimulation


def test_schedule_only_generates_manipulation_across_all_categories():
    plans = [scheduled_scene(sequence) for sequence in range(48)]
    assert Counter(task for _, task, _ in plans) == {
        "grasp": 16, "place": 16, "push": 16,
    }
    assert {(category, task) for category, task, _ in plans} == {
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
    (attempt / "episodes.sqlite").write_bytes(b"mock qualified storage")
    (attempt / "source").mkdir()
    (attempt / "source" / "parts_catalog.json").write_bytes(b"{}")
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


def test_disk_pause_preserves_state_until_exact_write_headroom(tmp_path, monkeypatch):
    required = 1024**2
    free = iter([DISK_FREE_FLOOR + required - 1, DISK_FREE_FLOOR + required])
    monkeypatch.setattr("so101.produce_catalog.os.statvfs", lambda path: SimpleNamespace(
        f_bavail=next(free), f_frsize=1,
    ))
    disk = DiskBackpressure(tmp_path)
    pauses = []
    monkeypatch.setattr(disk.wakeup, "wait", lambda timeout: pauses.append(
        json.loads(disk.status_path.read_text())
    ))
    assert disk.wait(required, "recording")
    assert len(pauses) == 1
    assert pauses[0]["status"] == "resource_paused"
    assert pauses[0]["preserves_inflight_state"]
    status = json.loads(disk.status_path.read_text())
    assert status["status"] == "ready"
    assert status["available_bytes"] == DISK_FREE_FLOOR + required


def test_stopped_admission_does_not_wait_for_disk_or_claim_an_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr("so101.produce_catalog.os.statvfs",
                        lambda path: (_ for _ in ()).throw(AssertionError("disk access")))
    disk = DiskBackpressure(tmp_path)
    assert not disk.wait(1024, "admission", lambda: True)
    assert not disk.status_path.exists()


def test_recording_pause_precedes_camera_writes_and_simulation_steps(tmp_path, monkeypatch):
    sim = object.__new__(CatalogSimulation)
    sim.episode_root = tmp_path
    sim.rows = []
    sim.record_images = True

    class Paused(Exception):
        pass

    def pause():
        raise Paused()

    sim.before_recording_write = pause
    monkeypatch.setattr("so101.catalog_sim.cv2.imwrite",
                        lambda *args: (_ for _ in ()).throw(AssertionError("camera write")))
    monkeypatch.setattr("so101.robot_sim.BeamSimulation.step",
                        lambda *args: (_ for _ in ()).throw(AssertionError("physics step")))
    try:
        sim.step(None, "push")
    except Paused:
        pass
    else:
        raise AssertionError("Recording bypassed disk pause")
    assert sim.rows == []
