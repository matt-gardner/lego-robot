import hashlib
import pytest

from so101.bucket_dashboard import BucketDashboard
from so101.bucket_store import connection, current_inventory, import_current, publish_current


def fixture_document(root):
    body = b"\x89PNG\r\n\x1a\noriginal"
    (root / "old").mkdir()
    (root / "old/image.png").write_bytes(body)
    sha = hashlib.sha256(body).hexdigest()
    return {
        "manifest": {"snapshot_id": "original", "datasets": [
            {"snapshot_id": "native", "provenance": {"published_pile_snapshot_sha256": "pool"}},
            {"snapshot_id": "measured", "protocol": {"horizon": 300}},
        ]},
        "problems": {"problems": [
            {"id": "native-task", "dataset_snapshot_id": "native", "snapshot_directory": "old",
             "initial_images": []},
            {"id": "measured-task", "dataset_snapshot_id": "measured", "initial_images": []},
        ], "buckets": []},
        "admitted_piles": [{"id": "pile", "dataset_snapshot_id": "native",
                            "snapshot_directory": "old",
                            "initial_images": [{"path": "image.png", "sha256": sha}]}],
        "distance_pilots": [],
        "calibration": {"groups": [{"problem_id": "measured-task", "attempts": 4,
                                     "dataset_snapshot_id": "measured"}]},
    }, body


def test_import_is_self_contained_and_images_survive_legacy_removal(tmp_path):
    document, image = fixture_document(tmp_path)
    import_current(tmp_path, document)
    (tmp_path / "old/image.png").unlink()
    browser = BucketDashboard(tmp_path)
    assert browser.image("pile", 0) == (image, "image/png")
    current = browser.snapshot()
    row = current["admitted_piles"][0]
    assert "snapshot_directory" not in row
    assert row["initial_images"][0]["path"].startswith("images/")
    assert current["calibration"] == document["calibration"]


def test_failed_import_rolls_back_inventory_and_images(tmp_path):
    document, _ = fixture_document(tmp_path)
    import_current(tmp_path, document)
    original = current_inventory(tmp_path)
    document["admitted_piles"][0]["initial_images"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        import_current(tmp_path, document)
    assert current_inventory(tmp_path) == original
    with connection(tmp_path) as db:
        assert db.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 1


def test_publish_replaces_only_current_channel_and_collects_unreferenced_images(tmp_path):
    document, _ = fixture_document(tmp_path)
    import_current(tmp_path, document)
    image = b"\x89PNG\r\n\x1a\nreplacement"
    (tmp_path / "new.png").write_bytes(image)
    manifest = {
        "kind": "lego-data-buckets", "schema_version": 1, "snapshot_id": "new-native",
        "provenance": {"published_pile_snapshot_sha256": "new-pool"}, "groups": [],
        "problems": [{"id": "new-task", "initial_images": []}],
        "admitted_piles": [{"id": "new-pile", "initial_images": [
            {"path": "new.png", "sha256": hashlib.sha256(image).hexdigest()}]}],
    }
    publish_current(tmp_path, manifest, tmp_path)
    current = current_inventory(tmp_path)
    assert {row["id"] for row in current["problems"]["problems"]} == {"new-task", "measured-task"}
    assert current["calibration"] == document["calibration"]
    assert current["manifest"]["datasets"][0] == document["manifest"]["datasets"][1]
    assert BucketDashboard(tmp_path).image("new-pile", 0) == (image, "image/png")
    with connection(tmp_path) as db:
        assert db.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 1
    assert manifest["admitted_piles"][0]["initial_images"][0]["path"] == "new.png"


def test_import_rejects_image_escape_without_replacing_current(tmp_path):
    document, _ = fixture_document(tmp_path)
    document["admitted_piles"][0]["initial_images"][0]["path"] = "../../outside.png"
    with pytest.raises(ValueError, match="escapes"):
        import_current(tmp_path, document)


def test_task_photos_are_not_imported_but_piece_coordinates_and_pile_image_are(tmp_path):
    document, image = fixture_document(tmp_path)
    task = document["problems"]["problems"][0]
    task["task"] = "grasp"
    task["initial_images"] = list(document["admitted_piles"][0]["initial_images"])
    pieces = [{"instance_id": "brick_0", "body_position_m": [0.1, 0.2, 0.3],
               "target_bbox_xyxy": [1, 2, 3, 4], "target_visible_pixels": 4,
               "target_bbox_status": "visible"}]
    document["admitted_piles"] = [{
        "id": "pile", "scene_id": "scene", "pieces": pieces,
        "snapshot_directory": "old", "initial_images": list(task["initial_images"]),
    }]
    import_current(tmp_path, document)
    current = current_inventory(tmp_path)
    assert current["problems"]["problems"][0]["initial_images"] == []
    assert current["admitted_piles"][0]["pieces"] == pieces
    with pytest.raises(ValueError, match="Unknown"):
        BucketDashboard(tmp_path).image("native-task", 0)
    assert BucketDashboard(tmp_path).image("pile", 0) == (image, "image/png")
