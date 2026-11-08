import json
import threading
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from so101.rl_dashboard import make_server, RunStore
from so101.rollout_evidence import RolloutRecorder, prune_evidence
from so101.training_data_browser import TrainingDataBrowser, identifier


@pytest.fixture
def dataset(tmp_path):
    a = "data/simulation/corpus/a"
    b = "data/simulation/corpus/b"
    c = "data/simulation/task_difficulty_test/c"
    for source, task in ((a, "grasp"), (b, "view"), (c, "push")):
        root = tmp_path / source
        (root / "wrist").mkdir(parents=True)
        metadata = {"task": task, "target": "short", "initial": {"seed": 12},
                    "outcome": "failed" if source == c else "success"}
        if source == c:
            metadata["difficulty"] = {"level": 3, "name": "challenge", "factors": ["long push"]}
        (root / "episode.json").write_text(json.dumps(metadata))
        (root / "frames.jsonl").write_text(json.dumps({"image": "wrist/0.jpg", "action": [0] * 6}) + "\n")
        (root / "wrist/0.jpg").write_bytes(b"jpeg")
        (root / "diagnostic_overview.jpg").write_bytes(b"overview")
    (tmp_path / "data/simulation/corpus/collection.json").write_text(json.dumps({"successful_episodes": [a, b]}))
    (tmp_path / "data/simulation/task_difficulty_test/collection.json").write_text(
        json.dumps({"attempts": [{"source": c, "outcome": "failed"}], "successful_episodes": []}))
    output = tmp_path / "outputs/run"
    output.mkdir(parents=True)
    (output / "data_split.json").write_text(json.dumps({"training": {"grasp": [a]}, "validation": [b]}))
    def group(assistance, outcomes, discarded=False):
        return {"source": a, "task": "grasp", "start_proximity": assistance, "discarded": discarded,
                "trajectories": [{"success": outcome, "rms_jerk": 3., "steps": 10} for outcome in outcomes]}
    log = {"validation": [], "run_config": {"collection": "data/simulation/corpus"},
           "steps": [{"step": 1, "groups": [group(.8, [True, False])]},
                     {"step": 2, "groups": [group(.8, [True, True])]},
                     {"step": 3, "groups": [group(0., [False, False], True)]}],
           "solved_problems": {"a": {"source": a, "task": "grasp", "start_proximity": .8, "solved_at_step": 2}}}
    path = output / "trajectory_rl.json"
    path.write_text(json.dumps(log))
    return tmp_path, path, log, a, b, c


def test_source_usage_outcomes_difficulty_and_exact_assistance_bucket(dataset):
    root, path, _, a, b, c = dataset
    browser = TrainingDataBrowser(root)
    result = browser.list(path, {})
    assert result["counts"] == {"examples": 3, "rollouts": 6, "solved": 1}
    row = next(item for item in result["items"] if item["source"] == a)
    assert (row["rollouts"], row["passed"], row["failed"]) == (6, 3, 3)
    assert row["difficulty_level"] == "unrated"
    assert row["expert_acceptance"] == "legacy final-success (may include retries)"
    assert browser.list(path, {"outcome": "failed"})["total"] == 1
    assert browser.list(path, {"difficulty": "3", "usage": "unused"})["items"][0]["source"] == c
    assert browser.list(path, {"usage": "validation"})["items"][0]["source"] == b
    solved = browser.list(path, {"view": "solved"})["items"][0]
    assert (solved["assistance"], solved["rollouts"], solved["passed"], solved["failed"]) == (.8, 4, 3, 1)
    latest = browser.list(path, {"view": "rollouts", "limit": "1"})["items"][0]
    assert latest["step"] == 3 and latest["discarded"]
    detail = browser.detail(path, "rollouts", latest["id"])
    assert detail["evidence_status"] == "not_recorded"
    assert not detail["frames"] and not detail["actions"]
    assert len(detail["source_history"]) == 3


def test_data_first_browser_without_any_training_log(dataset):
    root, path, _, a, b, c = dataset
    path.unlink()
    browser = TrainingDataBrowser(root)
    result = browser.list(None, {})
    assert {row["source"] for row in result["items"]} == {a, b, c}
    assert all(not row["used"] and row["split"] == "not recorded" for row in result["items"])
    assert browser.list(None, {"expert_outcome": "failed"})["items"][0]["source"] == c
    assert browser.list(None, {"task": "view"})["items"][0]["source"] == b
    assert browser.list(None, {"difficulty": "3"})["items"][0]["source"] == c
    assert browser.detail(None, "examples", identifier(a))["metadata"]["task"] == "grasp"
    assert browser.media(None, "examples", identifier(a), "initial").read_bytes() == b"jpeg"
    (root / b / "episode.json").unlink()
    browser.cache.clear()
    assert browser.list(None, {})["unavailable_source_count"] == 1
    assert browser.list(None, {})["total"] == 2
    with pytest.raises(ValueError, match="require a training run"):
        browser.list(None, {"view": "rollouts"})
    server = make_server(root / "outputs", "127.0.0.1", 0, "/lego-robot")
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        with urlopen(f"http://127.0.0.1:{server.server_port}/lego-robot/api/data/all?task=grasp") as response:
            assert json.load(response)["items"][0]["source"] == a
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def test_source_generation_order_is_not_policy_usage_and_preserves_filters(dataset):
    root, path, _, a, b, c = dataset
    for source, timestamp in [
        (a, "2026-11-08T17:00:00Z"),
        (b, "2026-11-08T18:00:00+00:00"),
    ]:
        episode = root / source / "episode.json"
        metadata = json.loads(episode.read_text())
        metadata["generated_at"] = timestamp
        episode.write_text(json.dumps(metadata))
    browser = TrainingDataBrowser(root)
    first = browser.list(path, {"limit": "1"})
    assert first["items"][0]["source"] == b
    assert first["items"][0]["last_step"] is None
    assert browser.list(path, {"offset": "1", "limit": "1"})["items"][0]["source"] == a
    legacy = browser.list(path, {"offset": "2", "limit": "1"})["items"][0]
    assert legacy["source"] == c and legacy["generated_at"] is None
    assert first["total"] == 3
    assert browser.list(path, {"usage": "used"})["items"][0]["source"] == a
    assert browser.list(path, {"usage": "unused", "limit": "1"})["items"][0]["source"] == b
    assert browser.list(path, {"view": "rollouts", "limit": "1"})["items"][0]["step"] == 3
    assert browser.detail(path, "examples", identifier(b))["metadata"]["generated_at"] == (
        "2026-11-08T18:00:00+00:00")


def test_sft_uses_its_collection_not_unrelated_historical_catalogs(dataset):
    root, path, log, a, b, c = dataset
    log.update(kind="sft", steps=[], solved_problems={})
    path.write_text(json.dumps(log))
    (root / c / "episode.json").unlink()
    browser = TrainingDataBrowser(root)
    assert {row["source"] for row in browser.list(path, {})["items"]} == {a, b}
    (root / a / "episode.json").unlink()
    with pytest.raises(FileNotFoundError):
        TrainingDataBrowser(root).list(path, {})


def test_sft_collection_changes_invalidate_browser_cache(dataset):
    root, path, log, a, b, _ = dataset
    log.update(kind="sft", steps=[], solved_problems={})
    path.write_text(json.dumps(log))
    browser = TrainingDataBrowser(root)
    assert browser.list(path, {})["total"] == 2
    collection = root / "data/simulation/corpus/collection.json"
    collection.write_text(json.dumps({"successful_episodes": [a, b, "data/simulation/corpus/missing"]}))
    with pytest.raises(FileNotFoundError):
        browser.list(path, {})


def test_completed_physical_preview_is_visible_without_claiming_training_usage(dataset):
    root, path, _, _, _, _ = dataset
    source = "data/simulation/physical_preview_v3/episode_00000"
    episode = root / source
    episode.mkdir(parents=True)
    (episode / "episode.json").write_text(json.dumps({
        "task": "place", "target": "short", "initial": {"seed": 42}, "outcome": "success",
    }))
    catalog = episode.parent / "collection.json"
    report = {"status": "building", "successful_episodes": [source]}
    catalog.write_text(json.dumps(report))
    browser = TrainingDataBrowser(root)
    assert browser.list(path, {"search": "physical_preview_v3"})["total"] == 0
    report["status"] = "complete"
    catalog.write_text(json.dumps(report))
    row = browser.list(path, {"search": "physical_preview_v3"})["items"][0]
    assert row["source"] == source
    assert row["split"] == "not-in-run" and not row["used"] and row["rollouts"] == 0


def test_bounded_recording_and_browser_actual_frames(dataset):
    root, path, log, a, _, _ = dataset
    class Sim:
        fps = 10
        data = SimpleNamespace(qpos=np.zeros(34))
        scene_version = 1
        def image(self): return np.zeros((24, 32, 3), dtype=np.uint8)
        def state(self): return np.ones(6, dtype=np.float32)
    sim = Sim()
    capture = path.parent / "rollouts/step_000003/group_0_0.json"
    recorder = RolloutRecorder(capture, sim, 123, a, 0.)
    for i in range(20):
        recorder.step(sim, np.full(6, i, dtype=np.float32))
    reference = recorder.finish(sim, {"success": False})
    log["steps"][-1]["groups"][0]["trajectories"][0]["evidence"] = reference
    path.write_text(json.dumps(log))
    browser = TrainingDataBrowser(root)
    row = next(item for item in browser.list(path, {"view": "rollouts"})["items"] if item.get("evidence"))
    detail = browser.detail(path, "rollouts", row["id"])
    assert detail["kind"] == "actual-policy-rollout" and detail["seed"] == 123
    assert [frame["step"] for frame in detail["frames"]] == [0, 16, 20]
    assert len(detail["actions"]) == 20
    assert len(detail["qpos"]) == 21
    assert detail["movie_available"]
    assert "recorded physics" in detail["playback_description"]
    assert browser.media(path, "rollouts", row["id"], "2").is_file()
    (capture.parent.parent / "step_000004").mkdir()
    unrelated = capture.parent.parent / "notes"
    unrelated.mkdir()
    prune_evidence(path.parent, 1)
    assert unrelated.exists() and not capture.exists()
    assert browser.detail(path, "rollouts", row["id"])["evidence_status"] == "expired"
    assert not browser.detail(path, "rollouts", row["id"])["movie_available"]


def test_source_navigation_includes_all_recorded_iterations_and_rechecks_retention(dataset):
    root, path, log, a, _, _ = dataset
    recording = path.parent / "rollouts/step_000001/example.json"
    recording.parent.mkdir(parents=True)
    recording.write_text(json.dumps({"actions": [[0] * 6], "frames": []}))
    group = log["steps"][0]["groups"][0]
    log["steps"] = [
        {"step": step, "groups": [{**group, "trajectories": [
            {"success": True, "steps": 10, "evidence": "rollouts/step_000001/example.json"},
            {"success": False, "steps": 10},
        ]}]}
        for step in range(1, 106)
    ]
    path.write_text(json.dumps(log))
    browser = TrainingDataBrowser(root)
    detail = browser.detail(path, "examples", identifier(a))
    assert len(detail["source_history"]) == 100
    assert detail["source_rollout_total"] == 210
    assert [row["step"] for row in detail["recorded_trajectories"]] == list(range(1, 106))
    trajectory = browser.detail(path, "rollouts", detail["recorded_trajectories"][-1]["id"])
    assert trajectory["example_id"] == identifier(a)
    assert trajectory["recorded_trajectories"] == detail["recorded_trajectories"]
    assert "reconstructed" in trajectory["playback_description"]
    recording.unlink()
    assert not browser.detail(path, "examples", identifier(a))["recorded_trajectories"]


def test_movie_cache_reuses_rendering_and_surfaces_failure(dataset, monkeypatch):
    root, path, _, a, _, _ = dataset
    calls = []
    def render(command, **kwargs):
        calls.append(command)
        Path(command[command.index("--output") + 1]).write_bytes(b"movie")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr('so101.training_data_browser.subprocess.run', render)
    browser = TrainingDataBrowser(root)
    movie = browser.movie(path, "examples", identifier(a))
    assert calls[0][1:3] == ["-m", "so101.simulation_playback"]
    assert movie.read_bytes() == b"movie"
    assert browser.movie(path, "examples", identifier(a)) == movie and len(calls) == 1
    frames = root / a / "frames.jsonl"
    frames.write_text(frames.read_text() * 2)
    monkeypatch.setattr('so101.training_data_browser.subprocess.run',
                        lambda *a, **k: SimpleNamespace(returncode=1, stderr="renderer failed"))
    with pytest.raises(OSError, match="renderer failed"):
        browser.movie(path, "examples", identifier(a))
    assert not list(movie.parent.glob("*.partial.mp4"))


def test_preparation_is_prompt_deduplicated_and_bounded(dataset, monkeypatch):
    root, path, _, a, b, _ = dataset
    browser = TrainingDataBrowser(root)
    started, release = threading.Event(), threading.Event()
    calls = []
    def render(command, **kwargs):
        calls.append(command)
        started.set()
        assert release.wait(5)
        Path(command[command.index("--output") + 1]).write_bytes(b"movie")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr('so101.training_data_browser.subprocess.run', render)
    try:
        assert browser.prepare_movie(path, "examples", identifier(a)) == {
            "ready": False, "status": "preparing",
        }
        assert started.wait(2)
        assert not browser.prepare_movie(path, "examples", identifier(a))["ready"]
        assert browser.prepare_movie(path, "examples", identifier(b))["status"] == "waiting"
        assert len(calls) == 1
    finally:
        release.set()
    browser.preparation[1].result(timeout=5)
    assert browser.prepare_movie(path, "examples", identifier(a))["ready"]
    assert len(calls) == 1


def test_preparation_surfaces_renderer_failure_and_allows_retry(dataset, monkeypatch):
    root, path, _, a, _, _ = dataset
    browser = TrainingDataBrowser(root)
    monkeypatch.setattr('so101.training_data_browser.subprocess.run',
                        lambda *args, **kwargs: SimpleNamespace(returncode=1, stderr="renderer failed"))
    browser.prepare_movie(path, "examples", identifier(a))
    with pytest.raises(OSError, match="renderer failed"):
        browser.preparation[1].result(timeout=5)
    with pytest.raises(OSError, match="renderer failed"):
        browser.prepare_movie(path, "examples", identifier(a))
    assert browser.preparation is None


def test_preparation_http_returns_json_while_rendering(dataset, monkeypatch):
    root, path, _, a, _, _ = dataset
    started, release = threading.Event(), threading.Event()
    def render(command, **kwargs):
        started.set()
        assert release.wait(5)
        Path(command[command.index("--output") + 1]).write_bytes(b"movie")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr('so101.training_data_browser.subprocess.run', render)
    server = make_server(root / "outputs", "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    run_id = next(iter(RunStore(root / "outputs").discover()))
    url = f"http://127.0.0.1:{server.server_port}/api/data/{run_id}?item={identifier(a)}&movie=prepare"
    try:
        response = urlopen(url, timeout=2)
        assert response.status == 202
        assert response.headers["Content-Type"] == "application/json"
        assert json.load(response)["ready"] is False
        assert started.wait(2)
        release.set()
        for _ in range(100):
            response = urlopen(url, timeout=2)
            if json.load(response)["ready"]:
                assert response.status == 200
                break
            threading.Event().wait(.01)
        else:
            pytest.fail("Preparation never became ready")
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join()


def test_media_rejects_source_and_evidence_escapes(dataset):
    root, path, log, a, _, _ = dataset
    browser = TrainingDataBrowser(root)
    with pytest.raises(ValueError, match="outside simulation"):
        browser.source_path("../../private")
    outside = root / "private.jpg"
    outside.write_bytes(b"not public")
    image = root / a / "wrist/0.jpg"
    image.unlink()
    image.symlink_to(outside)
    with pytest.raises(ValueError, match="Invalid image"):
        browser.media(path, "examples", identifier(a), "0")
    log["steps"][-1]["groups"][0]["trajectories"][0]["evidence"] = "../../private.json"
    path.write_text(json.dumps(log))
    with pytest.raises(ValueError, match="Invalid rollout"):
        browser.list(path, {"view": "rollouts"})


def test_data_http_filters_media_and_errors(dataset, monkeypatch):
    root, path, _, a, _, _ = dataset
    server = make_server(root / "outputs", "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    run_id = next(iter(RunStore(root / "outputs").discover()))
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        assert b"Data browser" in urlopen(base + "/data").read()
        response = json.load(urlopen(base + f"/api/data/{run_id}?task=grasp&usage=used"))
        assert response["total"] == 1
        media = urlopen(base + f"/api/data/{run_id}?item={identifier(a)}&image=overview")
        assert media.headers["Content-Type"] == "image/jpeg" and media.read() == b"overview"
        movie = root / "example.mp4"
        movie.write_bytes(b"0123456789")
        monkeypatch.setattr(TrainingDataBrowser, "movie", lambda *args: movie)
        movie_url = base + f"/api/data/{run_id}?item={identifier(a)}&movie=1"
        for requested, content, span in (
            ("bytes=0-3", b"0123", "bytes 0-3/10"),
            ("bytes=-3", b"789", "bytes 7-9/10"),
            ("bytes=6-", b"6789", "bytes 6-9/10"),
        ):
            response = urlopen(Request(movie_url, headers={"Range": requested}))
            assert response.status == 206 and response.read() == content
            assert response.headers["Content-Range"] == span
            assert "media-src 'self'" in response.headers["Content-Security-Policy"]
        for requested in ("bytes=100-", "bytes=-0", "bytes=4-1", "invalid"):
            with pytest.raises(HTTPError) as error:
                urlopen(Request(movie_url, headers={"Range": requested}))
            assert error.value.code == 416
        for suffix, code in ((f"{run_id}?limit=101", 400), (f"{run_id}?task=bad", 400),
                             (f"{run_id}?item=missing", 404), ("missing", 404)):
            with pytest.raises(HTTPError) as error:
                urlopen(base + "/api/data/" + suffix)
            assert error.value.code == code
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
