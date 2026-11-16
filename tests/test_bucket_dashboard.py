import hashlib
import json

import pytest

from so101.bucket_dashboard import BucketDashboard
from so101.bucket_store import connection, import_current


def current(root):
    image = b"\x89PNG\r\n\x1a\npile"
    (root / "pile.png").write_bytes(image)
    document = {
        "manifest": {"datasets": [], "snapshot_id": "current"},
        "problems": {"problems": [], "buckets": []},
        "admitted_piles": [{"id": "pile", "pieces": [], "initial_images": [
            {"path": "pile.png", "root_relative": True,
             "sha256": hashlib.sha256(image).hexdigest()}]}],
        "distance_pilots": [], "calibration": {"groups": []},
    }
    import_current(root, document)
    return image


def test_reader_uses_only_current_database_even_when_legacy_files_exist(tmp_path):
    image = current(tmp_path)
    (tmp_path / "current.json").write_text("invalid retired pointer")
    (tmp_path / "snapshots").mkdir()
    (tmp_path / "snapshots/broken.json").write_text("invalid retired snapshot")
    browser = BucketDashboard(tmp_path)
    assert len(json.loads(browser.response())["admitted_piles"]) == 1
    assert browser.image("pile", 0) == (image, "image/png")
    with pytest.raises(ValueError, match="Invalid"):
        browser.image("pile", -1)


def test_cache_surfaces_read_failure_instead_of_serving_stale_success(tmp_path):
    current(tmp_path)
    browser = BucketDashboard(tmp_path)
    browser.refresh_cache()
    assert browser.cached()[0]["manifest"]["snapshot_id"] == "current"
    (tmp_path / "current.sqlite3").unlink()
    browser.refresh_cache()
    with pytest.raises(OSError, match="database operation"):
        browser.cached()


def test_image_digest_is_checked_on_read(tmp_path):
    current(tmp_path)
    with connection(tmp_path, writable=True) as db:
        db.execute("UPDATE images SET body=?", (b"changed",))
    with pytest.raises(ValueError, match="hash mismatch"):
        BucketDashboard(tmp_path).image("pile", 0)
