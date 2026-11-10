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
