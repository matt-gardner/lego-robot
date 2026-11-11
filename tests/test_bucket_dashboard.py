import hashlib
import json
import threading
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from so101.bucket_dashboard import BucketDashboard


def publish(root, manifest):
    directory = root / "snapshots" / "one"
    directory.mkdir(parents=True)
    body = json.dumps(manifest).encode()
    (directory / "manifest.json").write_bytes(body)
    (root / "current.json").write_text(json.dumps({
        "path": "snapshots/one/manifest.json",
        "sha256": hashlib.sha256(body).hexdigest(),
    }))
    return directory


def test_snapshot_population_and_hash_pinned_initial_image(tmp_path):
    image = b"\x89PNG\r\n\x1a\nexample"
    directory = publish(tmp_path, {
        "kind": "lego-data-buckets", "schema_version": 1, "measurement_status": "screening",
        "problems": [{"id": "scene:target:push", "bucket_id": "native3", "task": "push",
                      "initial_images": [{"path": "initial.png",
                                          "sha256": hashlib.sha256(image).hexdigest()}]}],
        "groups": [], "buckets": [{"id": "native3", "population_total": 1}],
    })
    (directory / "initial.png").write_bytes(image)
    browser = BucketDashboard(tmp_path)
    assert browser.snapshot()["problems"]["problems"][0]["bucket"] == "native3"
    assert browser.image("scene:target:push", 0) == (image, "image/png")
    with pytest.raises(ValueError, match="publication changed"):
        browser.image("scene:target:push", 0, "old-snapshot")
    (directory / "initial.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        browser.image("scene:target:push", 0)
    (directory / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        browser.snapshot()


def test_reject_escape_missing_population_and_duplicate_identity(tmp_path):
    browser = BucketDashboard(tmp_path)
    for path in ("../outside", "/etc/passwd"):
        with pytest.raises(ValueError, match="escapes"):
            browser.read(path)
    publish(tmp_path, {"kind": "lego-data-buckets", "schema_version": 1, "groups": []})
    with pytest.raises(ValueError, match="population"):
        browser.snapshot()


def test_duplicate_ids_are_not_merged(tmp_path):
    publish(tmp_path, {"kind": "lego-data-buckets", "schema_version": 1,
                      "problems": [{"id": "same"}, {"id": "same"}], "groups": []})
    with pytest.raises(ValueError, match="duplicated"):
        BucketDashboard(tmp_path).snapshot()


def test_auto_discovers_independent_publications_and_preserves_images(tmp_path):
    for name in ("native", "photo"):
        root = tmp_path / name
        directory = publish(root, {
            "kind": "lego-data-buckets", "schema_version": 1, "snapshot_id": name,
            "problems": [{"id": name, "bucket_id": name}], "groups": [],
            "buckets": [{"id": name}],
        })
        pointer = json.loads((root / "current.json").read_text())
        pointer["path"] = f"{name}/" + pointer["path"]
        channel = tmp_path / "publications" / name
        channel.mkdir(parents=True)
        (channel / "current.json").write_text(json.dumps(pointer))
    browser = BucketDashboard(tmp_path)
    assert {p["id"] for p in browser.snapshot()["problems"]["problems"]} == {"native", "photo"}
    assert len(browser.snapshot()["manifest"]["datasets"]) == 2


def test_live_renderer_files_survive_base_publication_replacement(tmp_path):
    source = "/scenes/tasks/push"
    identity = hashlib.sha256(source.encode()).hexdigest()
    pool = "a" * 64
    manifest = {
        "kind": "lego-data-buckets", "schema_version": 1, "snapshot_id": "base",
        "provenance": {"native_pool_sha256": pool},
        "problems": [{"id": identity, "task": "push",
                      "source": {"episode_source": source, "sha256": "scene"},
                      "tags": {"target_instance_id": "brick_0"}}], "groups": [],
    }
    directory = publish(tmp_path, manifest)
    receipts = tmp_path / "initial-photo-external-v1" / pool
    receipts.mkdir(parents=True)
    image = b"\x89PNG\r\n\x1a\nlive"
    (receipts / "image.png").write_bytes(image)
    row = {
        "problem_id": identity, "pool_sha256": pool, "episode_source": source,
        "scene_files_sha256": {"problem.json": "scene"}, "target_instance_id": "brick_0",
        "physics_unchanged": True, "assistance": 0, "is_gpu_policy_frame": False,
        "camera": "external", "backend": "mitsuba-llvm_ad_rgb", "width": 10, "height": 10,
        "image_file": "image.png", "sha256": hashlib.sha256(image).hexdigest(),
        "target_bbox_status": "visible", "target_bbox_xyxy": [1, 1, 3, 3], "target_visible_pixels": 4,
    }
    receipt = receipts / f"{identity}.json"
    receipt.write_text(json.dumps(row))
    browser = BucketDashboard(tmp_path)
    first = browser.snapshot()["manifest"]["snapshot_id"]
    assert browser.image(identity, 0, first) == (image, "image/png")
    # A legacy callback may replace base data, but cannot erase live renderer files.
    manifest["snapshot_id"] = "next"
    body = json.dumps(manifest).encode()
    (directory / "manifest.json").write_bytes(body)
    (tmp_path / "current.json").write_text(json.dumps({
        "path": "snapshots/one/manifest.json", "sha256": hashlib.sha256(body).hexdigest(),
    }))
    second = browser.snapshot()["manifest"]["snapshot_id"]
    assert first != second
    assert browser.image(identity, 0, second) == (image, "image/png")
    newer = tmp_path / "initial-photo-external-v2" / pool
    newer.mkdir(parents=True)
    newer_image = b"\x89PNG\r\n\x1a\nv2"
    (newer / "image.png").write_bytes(newer_image)
    newer_row = {**row, "sha256": hashlib.sha256(newer_image).hexdigest()}
    newer_receipt = newer / receipt.name
    newer_receipt.write_text(json.dumps(newer_row))
    assert browser.image(identity, 0) == (newer_image, "image/png")
    # A higher-numbered empty batch must not hide an older completed task.
    (tmp_path / "initial-photo-external-v10" / pool).mkdir(parents=True)
    assert browser.image(identity, 0) == (newer_image, "image/png")
    latest = tmp_path / "initial-photo-external-v10" / pool
    (latest / "image.png").write_bytes(image)
    latest_receipt = latest / receipt.name
    latest_receipt.write_text(json.dumps(row))
    assert browser.image(identity, 0) == (image, "image/png")
    row["scene_files_sha256"]["problem.json"] = "wrong-state"
    latest_receipt.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="selected initial task"):
        browser.snapshot()


def test_live_metrics_replace_old_protocol_without_publication(tmp_path):
    pool = "a" * 64
    publish(tmp_path, {
        "kind": "lego-data-buckets", "schema_version": 1, "snapshot_id": "base",
        "provenance": {"native_pool_sha256": pool},
        "problems": [{"id": "p", "task": "push", "source": {"episode_source": "task"}}],
        "groups": [{"problem_id": "p", "attempts": 10, "successes": 8}],
    })
    path = tmp_path / "live-metrics" / "short" / "results.json"
    path.parent.mkdir(parents=True)
    row = {
        "input_sha256": {"native_pool": pool}, "optimizer": None, "policy_sha256_before": "policy",
        "horizon": 50, "measurement_scope": "short-horizon-screening", "complete": False,
        "groups": [{"source": "task", "task": "push", "policy_step": 0, "assistance": .95,
                    "rollouts": [{"success": False, "invalid_action_error": None}]}],
    }
    path.write_text(json.dumps(row))
    browser = BucketDashboard(tmp_path)
    snapshot = browser.snapshot()
    assert len(snapshot["calibration"]["groups"]) == 1
    assert snapshot["calibration"]["groups"][0]["successes"] == 0
    assert snapshot["calibration"]["protocol"]["horizon"] == 50
    row["groups"][0]["rollouts"].append({"success": True, "invalid_action_error": None})
    path.write_text(json.dumps(row))
    assert browser.snapshot()["calibration"]["groups"][0]["attempts"] == 2
    assert browser.snapshot()["manifest"]["snapshot_id"] != snapshot["manifest"]["snapshot_id"]


def test_current_generator_selection_is_read_live_without_bucket_publication(tmp_path):
    curriculum = tmp_path / "curriculum"
    scene_root = curriculum / "scene"
    scene_root.mkdir(parents=True)
    scene = {"scene_id": "scene", "tags": {"piece_count": 20}, "tasks": [
        {"task_id": "push", "task_type": "push", "target_instance_id": "brick_0", "goal": {}},
    ]}
    scene_bytes = json.dumps(scene).encode()
    (scene_root / "problem.json").write_bytes(scene_bytes)
    pool = {"kind": "lego-pile-task-pool", "status": "complete", "scenes": [
        {"root": str(scene_root), "split": "train",
         "files_sha256": {"problem.json": hashlib.sha256(scene_bytes).hexdigest()}},
    ]}
    pool_bytes = json.dumps(pool).encode()
    (curriculum / "pool.json").write_bytes(pool_bytes)
    descriptor = {"schema_version": 1, "kind": "lego-current-task-pools", "pools": [
        {"manifest_path": str(curriculum / "pool.json"), "sha256": hashlib.sha256(pool_bytes).hexdigest(),
         "task_count": 1, "scenes": 1},
    ]}
    current = curriculum / "current-task-pools.json"
    current.write_text(json.dumps(descriptor))
    browser = BucketDashboard(tmp_path / "dashboard", curriculum)
    snapshot = browser.snapshot()
    assert len(snapshot["problems"]["problems"]) == 1
    assert snapshot["problems"]["problems"][0]["source"]["task_id"] == "push"
    assert snapshot["calibration"]["groups"] == []
    identity = snapshot["problems"]["problems"][0]["id"]
    shared = {
        "kind": "lego-data-buckets", "schema_version": 1, "snapshot_id": "measurements",
        "provenance": {"native_pool_sha256": descriptor["pools"][0]["sha256"]},
        "problems": snapshot["problems"]["problems"],
        "groups": [{"problem_id": identity, "assistance": .95, "attempts": 16, "successes": 8}],
        "protocol": {"horizon": 50, "measurement_scope": "short-horizon-screening"},
        "complete": True, "measurement_status": "screening",
        "measurement_series": [
            {"series_id": "prior", "label": "Prior", "protocol": {"horizon": 50}, "complete": True,
             "groups": [{"problem_id": identity, "assistance": .95, "attempts": 16, "successes": 8}]},
            {"series_id": "new", "label": "New", "protocol": {"horizon": 50}, "complete": True,
             "groups": [{"problem_id": identity, "assistance": .95, "attempts": 4, "successes": 0}]},
        ],
        "current_measurement_series_id": "prior",
    }
    dashboard = tmp_path / "dashboard"
    publish(dashboard, shared)
    channel = dashboard / "publications" / "current-pool"
    channel.mkdir(parents=True)
    (channel / "current.json").write_bytes((dashboard / "current.json").read_bytes())
    matched = browser.snapshot()
    assert matched["calibration"]["groups"][0]["successes"] == 8
    assert matched["manifest"]["datasets"][0]["protocol"]["horizon"] == 50
    assert matched["calibration"]["complete"] is True
    assert len(matched["calibration"]["groups"]) == 2
    assert sum(g["attempts"] for g in matched["calibration"]["groups"]) == 20
    assert len(matched["manifest"]["datasets"][0]["measurement_series"]) == 2
    # A different pool must not inherit measurements even with matching task IDs.
    shared["provenance"]["native_pool_sha256"] = "old-pool"
    body = json.dumps(shared).encode()
    (dashboard / "snapshots/one/manifest.json").write_bytes(body)
    (channel / "current.json").write_text(json.dumps({
        "path": "snapshots/one/manifest.json", "sha256": hashlib.sha256(body).hexdigest(),
    }))
    assert browser.snapshot()["calibration"]["groups"] == []
    descriptor["pools"][0]["sha256"] = "wrong"
    current.write_text(json.dumps(descriptor))
    with pytest.raises(ValueError, match="hash mismatch"):
        browser.snapshot()
    descriptor["pools"][0]["manifest_path"] = "/etc/passwd"
    current.write_text(json.dumps(descriptor))
    with pytest.raises(ValueError, match="escapes"):
        browser.snapshot()


def test_prefixed_http_page_missing_publication_and_exact_image(tmp_path):
    from so101.rl_dashboard import make_server

    root = tmp_path / "outputs"
    root.mkdir()
    server = make_server(root, "127.0.0.1", 0, "/lego-robot")
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}/lego-robot"
    try:
        page = urlopen(base + "/buckets").read().decode()
        assert 'href="/lego-robot/buckets" aria-current="page"' in page
        assert "buckets.js" in page
        with pytest.raises(HTTPError) as missing:
            urlopen(base + "/api/buckets")
        assert missing.value.code == 404
        publication = tmp_path / "data" / "bucket_dashboard"
        image = b"\x89PNG\r\n\x1a\ninitial"
        directory = publish(publication, {
            "kind": "lego-data-buckets", "schema_version": 1,
            "problems": [{"id": "p", "initial_images": [
                {"path": "initial.png", "sha256": hashlib.sha256(image).hexdigest()}]}],
            "groups": [],
        })
        (directory / "initial.png").write_bytes(image)
        assert json.load(urlopen(base + "/api/buckets"))["problems"]["problems"][0]["id"] == "p"
        assert urlopen(base + "/api/buckets/image?problem=p&index=0").read() == image
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


@pytest.mark.parametrize("workstream", [
    "photo-task-progression-collective-v1", "catalog-diverse-20-v1", "catalog-diverse-larger-v1",
])
def test_admitted_tasks_are_available_before_pool_or_preview_completion(tmp_path, workstream):
    curriculum = tmp_path / "curriculum"
    curriculum.mkdir()
    scene = {"scene_id": "new-scene", "tags": {"piece_count": 40}, "tasks": [
        {"task_id": "view", "task_type": "view", "target_instance_id": "brick_0"},
    ]}
    body = json.dumps(scene).encode()
    (curriculum / "problem.json").write_bytes(body)
    admission = {"root": str(curriculum), "split": "train",
                 "files_sha256": {"problem.json": hashlib.sha256(body).hexdigest()}}
    admission_bytes = json.dumps(admission).encode()
    sha = hashlib.sha256(admission_bytes).hexdigest()
    dashboard = tmp_path / "dashboard"
    directory = dashboard / "pile-previews-v1" / sha
    directory.mkdir(parents=True)
    (directory / "admission.json").write_bytes(admission_bytes)
    publish(dashboard, {"kind": "lego-data-buckets", "schema_version": 1,
                        "snapshot_id": "empty", "problems": [], "groups": []})
    # Use the same multi-dataset response shape as the live service.
    channel = dashboard / "publications" / "base"
    channel.mkdir(parents=True)
    (channel / "current.json").write_bytes((dashboard / "current.json").read_bytes())
    browser = BucketDashboard(dashboard, curriculum)
    first = browser.snapshot()
    assert len(first["problems"]["problems"]) == 1
    assert first["problems"]["problems"][0]["task"] == "view"
    assert first["problems"]["problems"][0]["bucket"] == "current:40-pieces"
    assert first["problems"]["problems"][0]["initial_images"] == []
    assert first["admitted_piles"] == []
    generated = curriculum / workstream / "pieces-40/tasks/scene_000/admission.json"
    generated.parent.mkdir(parents=True)
    generated.write_bytes(admission_bytes)
    (directory / "admission.json").unlink()
    assert len(browser.snapshot()["problems"]["problems"]) == 1
    image = b"\x89PNG\r\n\x1a\nadmitted"
    (directory / "image.png").write_bytes(image)
    source = str(curriculum / "tasks" / "view")
    receipt = {"pool_sha256": sha, "episode_source": source,
               "problem_id": hashlib.sha256(source.encode()).hexdigest(),
               "target_instance_id": "brick_0", "scene_files_sha256": admission["files_sha256"],
               "physics_unchanged": True, "assistance": 0, "is_gpu_policy_frame": False,
               "image_file": "image.png", "sha256": hashlib.sha256(image).hexdigest(),
               "backend": "mitsuba-llvm_ad_rgb", "width": 10, "height": 10,
               "target_bbox_status": "not_visible", "target_bbox_xyxy": None, "target_visible_pixels": 0}
    (directory / "task.json").write_text(json.dumps(receipt))
    snapshot = browser.snapshot()
    pile = snapshot["problems"]["problems"][0]
    assert browser.image(pile["id"], 0, pile["dataset_snapshot_id"]) == (image, "image/png")
    assert snapshot["manifest"]["snapshot_id"] != first["manifest"]["snapshot_id"]
    assert snapshot["calibration"]["groups"] == []


def test_cached_response_and_images_do_not_rebuild_inventory(tmp_path, monkeypatch):
    image = b"\x89PNG\r\n\x1a\ncached"
    directory = publish(tmp_path, {
        "kind": "lego-data-buckets", "schema_version": 1, "snapshot_id": "cached",
        "problems": [{"id": "p", "initial_images": [
            {"path": "image.png", "sha256": hashlib.sha256(image).hexdigest()}]}],
        "groups": [],
    })
    (directory / "image.png").write_bytes(image)
    browser = BucketDashboard(tmp_path)
    browser.start_cache(interval=3600)
    try:
        body = browser.response()
        def unexpected():
            raise AssertionError("request recomputed inventory")
        monkeypatch.setattr(browser, "snapshot", unexpected)
        assert browser.response() is body
        assert browser.image("p", 0, "cached") == (image, "image/png")
        assert browser.image("p", 0, "cached") == (image, "image/png")
    finally:
        browser.stop_cache()


def test_distance_pilot_reports_parents_and_stages_without_baseline_merge(tmp_path):
    root = tmp_path / "pilot"
    manifest = {
        "kind": "lego-data-buckets", "schema_version": 1, "snapshot_id": "pilot",
        "problems": [{"id": "child", "task": "place", "tags": {
            "parent_task_id": "parent", "task_assistance": {
                "kind": "direct-task-distance-v2", "parent_task_id": "parent",
                "distance": {"active_stage": "release", "stage_order": ["carry", "release"],
                             "remaining_stages": ["release"],
                             "within_stage": {"metric": "height", "value": .01, "units": "m"}},
            }}}],
        "groups": [{"problem_id": "child", "assistance": 0, "attempts": 1, "successes": 1}],
    }
    publish(root, manifest)
    browser = BucketDashboard(tmp_path)
    pointer = json.loads((root / "current.json").read_bytes())
    pointer["path"] = "pilot/" + pointer["path"]
    channel = tmp_path / "publications" / "pilot"
    channel.mkdir(parents=True)
    pin = channel / "current.json"
    pin.write_text(json.dumps(pointer))
    pilot = browser.distance_pilots([pin])[0]
    assert pilot["parent_count"] == 1
    assert pilot["child_count"] == 1
    assert pilot["problems"][0]["tags"]["task_assistance"]["distance"]["active_stage"] == "release"
    assert pilot["groups"][0]["assistance"] == 0
    manifest["snapshot_id"] = "successor"
    manifest["provenance"] = {"native_pool_sha256": "new", "supersedes_native_pool_sha256": "old"}
    successor = tmp_path / "successor"
    publish(successor, manifest)
    successor_pointer = json.loads((successor / "current.json").read_bytes())
    successor_pointer["path"] = "successor/" + successor_pointer["path"]
    new_channel = tmp_path / "publications" / "successor"
    new_channel.mkdir()
    new_pin = new_channel / "current.json"
    new_pin.write_text(json.dumps(successor_pointer))
    old = json.loads((root / "snapshots/one/manifest.json").read_bytes())
    old["provenance"] = {"native_pool_sha256": "old"}
    body = json.dumps(old).encode()
    (root / "snapshots/one/manifest.json").write_bytes(body)
    pointer["sha256"] = hashlib.sha256(body).hexdigest()
    pin.write_text(json.dumps(pointer))
    retained = browser.distance_pilots([pin, new_pin])
    assert len(retained) == 1
    assert retained[0]["manifest"]["snapshot_id"] == "successor"
    assert (root / "snapshots/one/manifest.json").exists()
