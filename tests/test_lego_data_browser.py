import hashlib
import json
import sqlite3
import threading
import tracemalloc
from urllib.request import urlopen

import pytest

from so101.lego_data_browser import LegoDataBrowser
from so101.generation_order import generation_order


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


def test_generation_order_uses_dates_not_ids_copy_times_or_qualification(tmp_path):
    releases = []
    for name, generated_at in [
        ("a-new", "2026-11-08T18:00:00Z"),
        ("z-old", "2026-11-08T18:30:00+01:00"),
        ("legacy", None),
    ]:
        (tmp_path / name).mkdir()
        path = episode_release(tmp_path / name)
        metadata = {"release_id": name, "generated_at": generated_at}
        episodes = [
            {"example_id": "z-new", "task": "grasp", "generated_at": "2026-11-08T18:00:00Z"},
            {"example_id": "a-tie", "task": "grasp", "generated_at": "2026-11-08T18:00:00+00:00"},
            {"example_id": "a-old", "task": "view", "generated_at": "2026-11-08T18:30:00+01:00"},
            {"example_id": "legacy", "task": "grasp"},
        ]
        with sqlite3.connect(path) as db:
            db.execute("UPDATE catalog SET content=?", (json.dumps(metadata).encode(),))
            db.execute("DELETE FROM files")
            for index, episode in enumerate(episodes):
                content = json.dumps(episode).encode()
                db.execute("INSERT INTO files VALUES (?,?,?,?)",
                           (f"source-{index}", "episode.json", content,
                            hashlib.sha256(content).hexdigest()))
        releases.append(path)
    # The older release is copied last, but that is not generation chronology.
    releases[1].touch()
    browser = LegoDataBrowser(tmp_path, episode_root=tmp_path)
    discovered = browser.list({})["releases"]
    assert [row["version"] for row in discovered] == ["a-new", "z-old", "legacy"]
    query = {"release": discovered[0]["id"], "limit": "2"}
    first = browser.list(query)
    assert [row["id"] for row in first["items"]] == ["a-tie", "z-new"]
    second = browser.list({**query, "offset": "2"})
    assert [row["id"] for row in second["items"]] == ["a-old", "legacy"]
    assert first["total"] == second["total"] == 4
    filtered = browser.list({**query, "task": "grasp", "offset": "1"})
    assert [row["id"] for row in filtered["items"]] == ["z-new", "legacy"]
    detail = browser.list({"release": discovered[0]["id"], "item": "z-new"})
    assert detail["item"]["source"] == "source-0"
    assert detail["release"]["artifact_sha256"] == hashlib.sha256(releases[0].read_bytes()).hexdigest()
    assert browser.list({})["releases"] == discovered
    with sqlite3.connect(releases[1]) as db:
        db.execute("UPDATE catalog SET content=?", (json.dumps({
            "release_id": "z-old", "generated_at": "2026-11-09T00:00:00Z",
        }).encode(),))
    refreshed = browser.list({})["releases"]
    assert refreshed[0]["version"] == "z-old"
    assert refreshed[0]["id"] != discovered[1]["id"]


@pytest.mark.parametrize("value", ["2026-11-08T18:00:00", "2026-02-30T00:00:00Z", 123, ""])
def test_invalid_generation_timestamp_is_explicit(value):
    with pytest.raises(ValueError, match="generated_at"):
        generation_order({"id": "example", "generated_at": value})


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


def test_unpublished_hidden_releases_are_not_discovered(tmp_path):
    catalog(tmp_path)
    staging = catalog(tmp_path, ".catalog-staging-incomplete")
    staging.write_text("incomplete JSON")
    nested = tmp_path / "data" / "published" / ".catalog-staging-nested"
    nested.mkdir(parents=True)
    (nested / "catalog.json").write_text("incomplete JSON")
    hidden_sqlite = tmp_path / "data" / ".episode-staging.sqlite"
    hidden_sqlite.write_bytes(b"incomplete SQLite")
    releases = LegoDataBrowser(tmp_path).list({})["releases"]
    assert [(row["kind"], row["version"]) for row in releases] == [("parts", "v1")]
    assert staging.read_text() == "incomplete JSON"
    assert hidden_sqlite.read_bytes() == b"incomplete SQLite"


def test_exact_invalidation_preserves_history_and_identity(tmp_path):
    path = catalog(tmp_path)
    original = path.read_bytes()
    browser = LegoDataBrowser(tmp_path)
    release = browser.list({})["releases"][0]
    notices = tmp_path / "data" / "notices"
    notices.mkdir()
    notice = {
        "schema_version": 1, "notice_id": "frame-defect", "catalog_id": "v1",
        "catalog_sha256": release["catalog_sha256"], "status": "invalidated",
        "reason": "Wrong coordinate frame", "source_work_item": "catalog-work",
        "evidence": {"height_axis": "Y"},
    }
    notice_path = notices / "frame-defect.json"
    notice_path.write_text(json.dumps(notice))
    result = browser.list({"release": release["id"], "status": "verified"})
    assert result["total"] == 1
    assert result["items"][0]["status"] == "verified"
    assert result["items"][0]["stored_status"] == "verified"
    assert result["items"][0]["effective_status"] == "invalidated"
    assert result["items"][0]["qualification_status"] == "invalidated"
    assert browser.list({"release": release["id"], "effective_status": "verified"})["total"] == 0
    assert result["release"]["qualification_notices"][0]["notice_sha256"] == (
        hashlib.sha256(notice_path.read_bytes()).hexdigest())
    assert browser.list({"release": release["id"], "qualification_status": "not_invalidated"})["total"] == 0
    assert path.read_bytes() == original
    assert browser.list({})["releases"][0]["id"] == release["id"]
    notice["catalog_sha256"] = "0" * 64
    notice_path.write_text(json.dumps(notice))
    assert browser.list({"release": release["id"]})["items"][0]["effective_status"] == "verified"
    notice["catalog_sha256"] = release["catalog_sha256"]
    notice["catalog_id"] = "other-release"
    notice_path.write_text(json.dumps(notice))
    assert browser.list({"release": release["id"]})["release"]["qualification_notices"] == []


def test_malformed_notice_fails_instead_of_hiding_invalidation(tmp_path):
    catalog(tmp_path)
    root = tmp_path / "data" / "notices"
    root.mkdir()
    (root / "broken.json").write_text('{"schema_version":1}')
    with pytest.raises(ValueError, match="qualification notice"):
        LegoDataBrowser(tmp_path).list({})


def test_discovery_memory_does_not_retain_historical_payloads(tmp_path):
    payload_size = 512 * 1024
    for index in range(8):
        path = catalog(tmp_path, f"v{index}")
        metadata = json.loads(path.read_text())
        metadata["parts"][0]["evidence"] = "x" * payload_size
        path.write_text(json.dumps(metadata))
    tracemalloc.start()
    try:
        releases = LegoDataBrowser(tmp_path).releases()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(releases) == 8
    assert all("metadata" not in release for release in releases.values())
    assert peak < 5 * payload_size


def test_selected_release_does_not_parse_other_catalogs(tmp_path):
    catalog(tmp_path)
    other = catalog(tmp_path, "v2")
    browser = LegoDataBrowser(tmp_path)
    selected = next(row["id"] for row in browser.list({})["releases"] if row["version"] == "v1")
    other.write_text("malformed unrelated catalog")
    assert browser.list({"release": selected})["total"] == 2
    with pytest.raises(ValueError):
        browser.list({})


def test_warm_discovery_reuses_payload_summary_but_refreshes_notices(tmp_path, monkeypatch):
    path = catalog(tmp_path)
    browser = LegoDataBrowser(tmp_path)
    release = browser.list({})["releases"][0]
    original_read = type(path).read_bytes
    reads = []
    def read(candidate):
        reads.append(candidate)
        return original_read(candidate)
    monkeypatch.setattr(type(path), "read_bytes", read)
    assert browser.list({})["releases"][0]["id"] == release["id"]
    assert path not in reads
    root = path.parent.parent / "notices"
    root.mkdir()
    (root / "invalid.json").write_text(json.dumps({
        "schema_version": 1, "notice_id": "invalid", "catalog_id": "v1",
        "catalog_sha256": release["catalog_sha256"], "status": "invalidated",
        "reason": "Bad frame", "source_work_item": "work", "evidence": {},
    }))
    assert browser.list({})["releases"][0]["qualification_notices"][0]["status"] == "invalidated"
    assert path not in reads
    path.write_text(path.read_text() + "\n")
    assert browser.list({})["releases"][0]["id"] != release["id"]
    assert path in reads


def test_catalog_discovery_does_not_traverse_release_assets(tmp_path):
    path = catalog(tmp_path)
    assets = path.parent / "parts" / "source"
    assets.mkdir(parents=True)
    (assets / "catalog.json").write_text("not a published release")
    releases = LegoDataBrowser(tmp_path).list({})["releases"]
    assert len(releases) == 1


def test_real_geometry_preview_is_hash_pinned_and_invalidated_parts_stay_hidden(tmp_path):
    path = catalog(tmp_path)
    metadata = json.loads(path.read_text())
    geometry = b"""<mujoco><asset><mesh name="visual" vertex="0 0 0 1 0 0 0 1 1" face="0 1 2"/></asset>
    <worldbody><body><geom type="mesh" mesh="visual" group="2" rgba=".8 .25 .1 1"/></body></worldbody></mujoco>"""
    mesh_path = path.parent / "parts" / "54200" / "part.xml"
    mesh_path.parent.mkdir(parents=True)
    mesh_path.write_bytes(geometry)
    metadata["parts"][0].update({
        "mjcf": "parts/54200/part.xml",
        "assets": [{"path": "parts/54200/part.xml", "sha256": hashlib.sha256(geometry).hexdigest()}],
    })
    path.write_text(json.dumps(metadata))
    browser = LegoDataBrowser(tmp_path)
    release = browser.list({})["releases"][0]
    query = {"release": release["id"], "item": "54200"}
    assert release["validated_parts"] == 1
    assert browser.geometry(query)["meshes"][0]["faces"] == [[0, 1, 2]]
    assert b"<polygon" in browser.preview(query)
    assert browser.list({"release": release["id"], "effective_status": "verified"})["total"] == 1
    mesh_path.write_bytes(geometry + b"\n")
    with pytest.raises(ValueError, match="checksum"):
        browser.preview(query)


@pytest.mark.parametrize("mesh", [
    b'<mujoco><asset><mesh name="m" vertex="nan 0 0" face="0 0 0"/></asset><worldbody><geom type="mesh" mesh="m" group="2"/></worldbody></mujoco>',
    b'<mujoco><asset><mesh name="m" vertex="0 0 0" face="0 0 2"/></asset><worldbody><geom type="mesh" mesh="m" group="2"/></worldbody></mujoco>',
    b'<mujoco><worldbody><body pos="1 0 0"/></worldbody></mujoco>',
])
def test_invalid_visual_geometry_is_not_rendered(mesh):
    from so101.part_preview import visual_geometry

    with pytest.raises(ValueError):
        visual_geometry(mesh)


def test_declared_variants_group_with_plain_base_without_qualification_claim(tmp_path):
    path = catalog(tmp_path)
    metadata = json.loads(path.read_text())
    metadata["parts"][0]["decoration"] = "undecorated"
    metadata["parts"].append({
        "part_id": "54200p01", "description": "Printed slope", "family": "slope",
        "status": "imported", "decoration": "print", "base_part_id": "54200",
        "base_part_relationship": {"basis": "ldraw_official_numbering",
                                   "geometry_equivalence": "not_qualified"},
    })
    path.write_text(json.dumps(metadata))
    browser = LegoDataBrowser(tmp_path)
    release = browser.list({})["releases"][0]["id"]
    variants = browser.list({"release": release, "item": "54200"})["variants"]
    assert [variant["part_id"] for variant in variants] == ["54200p01"]
    assert variants[0]["status"] == "imported"
    assert variants[0]["relationship"]["geometry_equivalence"] == "not_qualified"
    assert browser.list({"release": release, "item": "54200p01"})["variants"][0]["part_id"] == "54200"


def test_readiness_type_footprint_decoration_filters(tmp_path):
    path = catalog(tmp_path)
    metadata = json.loads(path.read_text())
    metadata["parts"][0].update(description="Slope Brick 31  2 x  1", decoration="undecorated")
    path.write_text(json.dumps(metadata))
    browser = LegoDataBrowser(tmp_path)
    release = browser.list({})["releases"][0]["id"]
    result = browser.list({"release": release, "readiness": "ready", "family": "slope",
                           "footprint": "1 x 2", "decoration": "undecorated"})
    assert result["total"] == 1
    assert result["facets"]["family"] == ["slope", "tile"]
    assert browser.list({"release": release, "readiness": "not_ready"})["total"] == 1


def test_external_recordings_use_logical_keys_and_hash_bound_receipt(tmp_path):
    (tmp_path / "published").mkdir()
    path = episode_release(tmp_path / "published")
    receipt = path.parent / "qualification.json"
    receipt.write_text(json.dumps({
        "storage_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "scope": "development diagnostic, not training",
    }))
    browser = LegoDataBrowser(tmp_path / "workspace", episode_root=path.parent)
    release = browser.list({})["releases"][0]
    assert "development diagnostic" in release["qualification"]["scope"]
    row = browser.list({"release": release["id"]})["items"][0]
    assert browser.media({"release": release["id"], "item": row["id"], "frame": "0"})[0] == b"recorded-image"
    receipt.write_text('{"storage_sha256":"wrong"}')
    with pytest.raises(ValueError, match="receipt"):
        browser.list({})


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
