import base64
import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "verify_adaptive_exports", Path(__file__).parents[1] / "scripts/verify_adaptive_gpu_exports.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("field,value", [
    ("VersionId", "another-version"),
    ("ContentLength", 5),
    ("ChecksumSHA256", "different-content"),
])
def test_independent_confirmation_rejects_version_size_or_checksum_mismatch(monkeypatch, field, value):
    artifact = {"uri": "s3://bucket/artifact", "version": "version", "size": 4, "sha256": "ab" * 32}
    head = {"VersionId": "version", "ContentLength": 4,
            "ChecksumSHA256": base64.b64encode(bytes.fromhex(artifact["sha256"])).decode()}
    head[field] = value
    monkeypatch.setattr(module, "request", lambda *args: head)
    with pytest.raises(ValueError, match="confirmation failed"):
        module.confirm(artifact)
