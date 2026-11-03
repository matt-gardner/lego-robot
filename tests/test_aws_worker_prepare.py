import hashlib
import io
import tarfile

import pytest

from so101.aws_worker_prepare import prepare
from so101.control_broker import canonical


def setup(tmp_path, monkeypatch):
    import so101.aws_worker_prepare as module
    archive = tmp_path / "collection.tar"
    with tarfile.open(archive, "w") as bundle:
        info = tarfile.TarInfo("collection.json")
        info.size = 2
        bundle.addfile(info, io.BytesIO(b"{}"))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (tmp_path / ".staged.json").write_text("{}")
    manifest = {"spec_sha256": "a" * 64, "spec": {"recipe": {"argv": ["approved"]}},
                "artifacts": {"collection.tar": {"sha256": digest}}}
    plan = {"spec_sha256": "a" * 64, "argv": ["approved"], "archives": {
        "collection.tar": {"destination": "collection", "max_bytes": 2, "max_members": 1},
    }}
    binding = {"manifest_sha256": "b" * 64, "spec_sha256": "a" * 64,
               "staging_receipt_sha256": hashlib.sha256(b"{}").hexdigest()}
    monkeypatch.setattr(module, "load_manifest", lambda *args: manifest)
    monkeypatch.setattr(module, "trusted_path", lambda path: path.resolve(strict=True))
    monkeypatch.setattr(module, "trusted_json",
                        lambda path: binding if path.name == ".bootstrap.json" else plan)
    return plan, dict(manifest_path=tmp_path / "manifest", manifest_sha256="b" * 64,
                      plan_path=tmp_path / "plan", staged=tmp_path)


def test_preparation_binds_verified_archive_and_creates_readonly_inputs(tmp_path, monkeypatch):
    plan, kwargs = setup(tmp_path, monkeypatch)
    prepare(**kwargs, plan_sha256=hashlib.sha256(canonical(plan)).hexdigest())
    assert (tmp_path / "collection/collection.json").read_bytes() == b"{}"
    assert (tmp_path / "collection/.extracted.json").is_file()


def test_changed_staged_archive_is_rejected_before_extracting(tmp_path, monkeypatch):
    plan, kwargs = setup(tmp_path, monkeypatch)
    (tmp_path / "collection.tar").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        prepare(**kwargs, plan_sha256=hashlib.sha256(canonical(plan)).hexdigest())
    assert not (tmp_path / "collection").exists()


def test_preparation_cannot_escape_staged_root(tmp_path, monkeypatch):
    plan, kwargs = setup(tmp_path, monkeypatch)
    plan["archives"]["collection.tar"]["destination"] = "../outside"
    with pytest.raises(ValueError, match="new named"):
        prepare(**kwargs, plan_sha256=hashlib.sha256(canonical(plan)).hexdigest())
