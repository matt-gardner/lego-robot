import hashlib
import json
import sqlite3
import threading
from urllib.request import urlopen

import pytest

from so101.lego_data_browser import LegoDataBrowser


def catalog(tmp_path, version="v1"):
    root = tmp_path / "data" / version
    root.mkdir(parents=True)
    path = root / "catalog.json"
    path.write_text(json.dumps({
        "schema_version": 1, "catalog_id": version,
        "parts": [
            {"part_id": "54200", "description": "Slope", "family": "slope", "status": "verified"},
            {"part_id": "missing", "description": "Missing", "family": "tile",
             "status": "unavailable", "failure_reason": "Dependency missing"},
        ],
    }))
    return path


def episode_release(tmp_path):
    root = tmp_path / "data"
    root.mkdir(exist_ok=True)
    path = root / "legacy.sqlite"
    with sqlite3.connect(path) as db:
        db.executescript("""
            PRAGMA user_version=1;
            CREATE TABLE catalog(content BLOB);
            CREATE TABLE files(source TEXT, name TEXT, content BLOB, sha256 TEXT);
        """)
        db.execute("INSERT INTO catalog VALUES (?)", (b'{"successful_episodes":["old/source"]}',))
        for name, content in [
            ("episode.json", b'{"task":"grasp","initial":{"seed":1}}'),
            ("frames.jsonl", b'{"image":"wrist.png","timestamp":0.5}\n'),
            ("wrist.png", b"recorded-image"),
        ]:
            db.execute("INSERT INTO files VALUES (?,?,?,?)",
                       ("old/source", name, content, hashlib.sha256(content).hexdigest()))
    return path


def test_parts_versions_filters_and_exact_identity(tmp_path):
    first = catalog(tmp_path)
    catalog(tmp_path, "v2")
    browser = LegoDataBrowser(tmp_path)
    releases = browser.list({})["releases"]
    assert len(releases) == 2
    release = next(row for row in releases if row["version"] == "v1")
    assert release["catalog_sha256"] == hashlib.sha256(first.read_bytes()).hexdigest()
    result = browser.list({"release": release["id"], "status": "unavailable"})
    assert result["total"] == 1
    assert result["items"][0]["failure_reason"] == "Dependency missing"
    assert browser.list({"release": release["id"], "search": "slope"})["total"] == 1
    assert browser.list({})["feedback"]["available"] is False
    first.write_text(first.read_text() + "\n")
    with pytest.raises(FileNotFoundError):
        browser.list({"release": release["id"]})


def test_sqlite_legacy_labels_readonly_and_original_media(tmp_path):
    path = episode_release(tmp_path)
    original = path.read_bytes()
    browser = LegoDataBrowser(tmp_path)
    release = browser.list({})["releases"][0]
    item = browser.list({"release": release["id"]})["items"][0]
    assert item["expert_acceptance"] == "legacy final-success (may include retries)"
    assert item["split"] == "not recorded"
    query = {"release": release["id"], "item": item["id"]}
    assert browser.list(query)["frames"] == [
        {"frame_index": 0, "timestamp": 0.5, "wrist": True, "external": False}]
    assert browser.media({**query, "frame": "0"}) == (b"recorded-image", "image/png")
    with pytest.raises(FileNotFoundError, match="No recorded external"):
        browser.media({**query, "frame": "0", "camera": "external"})
    with pytest.raises(ValueError):
        browser.media({**query, "frame": "-1"})
    assert path.read_bytes() == original


def test_symlink_catalog_is_rejected(tmp_path):
    path = catalog(tmp_path)
    link = tmp_path / "data" / "linked"
    link.symlink_to(path.parent, target_is_directory=True)
    browser = LegoDataBrowser(tmp_path)
    with pytest.raises(ValueError, match="symlinks"):
        browser.contained(link / "catalog.json")


def test_corrupted_episode_fails_explicitly(tmp_path):
    path = episode_release(tmp_path)
    with sqlite3.connect(path) as db:
        db.execute("UPDATE files SET content=? WHERE name='episode.json'", (b"{}",))
    browser = LegoDataBrowser(tmp_path)
    release = browser.list({})["releases"][0]
    with pytest.raises(ValueError, match="checksum"):
        browser.list({"release": release["id"]})


def test_explicit_catalog_root(tmp_path):
    catalog(tmp_path / "published")
    browser = LegoDataBrowser(tmp_path / "workspace", tmp_path / "published" / "data")
    release = browser.list({})["releases"][0]
    assert browser.list({"release": release["id"]})["total"] == 2


def test_dashboard_prefix_and_lego_api(tmp_path):
    from so101.rl_dashboard import make_server

    catalog(tmp_path)
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    server = make_server(outputs, "127.0.0.1", 0, "/lego-robot")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}/lego-robot"
    try:
        with urlopen(base + "/lego") as response:
            page = response.read().decode()
            assert 'data-base-path="/lego-robot"' in page
            assert 'src="/lego-robot/lego.js"' in page
            assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
        with urlopen(base + "/api/lego") as response:
            result = json.load(response)
            assert len(result["releases"]) == 1
            assert result["feedback"]["available"] is False
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
