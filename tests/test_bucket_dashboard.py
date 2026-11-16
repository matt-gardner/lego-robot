import hashlib
import json

import pytest

from so101.bucket_dashboard import BucketDashboard


def current(root):
    image = b"\x89PNG\r\n\x1a\npile"
    directory = root / "piles/pile_0001"
    (directory / "tasks").mkdir(parents=True)
    (directory / "rendered.png").write_bytes(image)
    document = {
        "schema_version": 1, "manifest": {"datasets": []}, "buckets": [],
        "distance_pilots": [], "calibration": {"groups": []},
    }
    (root / "piles/metadata.json").write_text(json.dumps(document))
    (directory / "pile.json").write_text(json.dumps({
        "id": "pile_0001", "scene_id": "scene", "tags": {},
        "qualification": "unmeasured", "pieces": [],
        "rendered": {"sha256": hashlib.sha256(image).hexdigest()},
    }))
    return image


def test_reader_uses_only_canonical_directories_even_when_legacy_files_exist(tmp_path):
    image = current(tmp_path)
    (tmp_path / "current.json").write_text("invalid retired pointer")
    (tmp_path / "snapshots").mkdir()
    (tmp_path / "snapshots/broken.json").write_text("invalid retired snapshot")
    (tmp_path / "current.sqlite3").write_text("invalid retired database")
    browser = BucketDashboard(tmp_path)
    assert len(json.loads(browser.response())["admitted_piles"]) == 1
    assert browser.image("pile_0001", 0) == (image, "image/png")
    with pytest.raises(ValueError, match="Unknown"):
        browser.image("pile_0001", -1)


def test_cache_surfaces_read_failure_instead_of_serving_stale_success(tmp_path):
    current(tmp_path)
    browser = BucketDashboard(tmp_path)
    browser.refresh_cache()
    assert len(json.loads(browser.cached())["admitted_piles"]) == 1
    (tmp_path / "piles/metadata.json").unlink()
    browser.refresh_cache()
    with pytest.raises(OSError):
        browser.cached()


def test_image_digest_is_checked_on_read(tmp_path):
    current(tmp_path)
    (tmp_path / "piles/pile_0001/rendered.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest differs"):
        BucketDashboard(tmp_path).image("pile_0001", 0)


def test_refresh_keeps_only_serialized_inventory_and_does_not_decode_large_documents(tmp_path, monkeypatch):
    current(tmp_path)
    browser = BucketDashboard(tmp_path)

    def reject_decode(*args):
        raise AssertionError("Cache refresh must not retain a second decoded inventory")

    monkeypatch.setattr("so101.bucket_dashboard.current_inventory", reject_decode)
    browser.refresh_cache()
    before = browser.cached()
    browser.refresh_cache()
    assert isinstance(before, bytes)
    assert browser.cached() == before
