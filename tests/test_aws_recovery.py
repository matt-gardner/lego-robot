import base64
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest

from so101.aws_artifacts import Artifact
from so101.aws_recovery import recover
from so101.control_broker import canonical


def setup():
    job, spec, prefix = "1" * 32, "a" * 64, "s3://approved/runs/lego-robot/"
    body = b"validated opaque full state"
    checkpoint = Artifact(prefix + job + "/step_000000304/" + hashlib.sha256(body).hexdigest() + ".pt",
                          "checkpoint-v1", hashlib.sha256(body).hexdigest(), len(body))
    manifest = canonical({"schema": 1, "job_id": job, "spec_sha256": spec, "step": 304,
                          "checkpoint": asdict(checkpoint)})
    key = "runs/lego-robot/" + job + "/step_000000304/" + hashlib.sha256(manifest).hexdigest() + ".json"
    objects = {key: (manifest, "manifest-v1"), checkpoint.uri.removeprefix("s3://approved/"): (body, checkpoint.version)}
    calls = []
    def request(*args):
        calls.append(args)
        if args[1] == "list-objects-v2":
            assert args[args.index("--prefix") + 1] == "runs/lego-robot/" + job + "/"
            return {"Contents": [{"Key": item} for item in objects], "IsTruncated": False}
        key = next(item.removeprefix("--key=") for item in args if item.startswith("--key="))
        data, version = objects[key]
        if args[1] == "get-object":
            assert "--version-id=" + version in args
            Path(args[-1]).write_bytes(data)
        return {"VersionId": version, "ContentLength": len(data),
                "ChecksumSHA256": base64.b64encode(hashlib.sha256(data).digest()).decode()}
    kwargs = dict(owner="133961745003", prefix=prefix, job_id=job, spec_sha256=spec)
    return request, kwargs, calls, objects


def test_discovery_pins_manifest_and_checkpoint_versions_without_ssm_stdout(tmp_path):
    request, kwargs, calls, objects = setup()
    receipt = recover(request, tmp_path, **kwargs)
    assert receipt["step"] == 304 and receipt["checkpoint"]["version"] == "checkpoint-v1"
    assert [args[1] for args in calls] == ["list-objects-v2", "head-object", "get-object", "head-object"]
    assert "--version-id=checkpoint-v1" in calls[-1]
    assert json.loads((tmp_path / kwargs["job_id"] / "receipt.json").read_text()) == receipt


def test_partial_higher_checkpoint_does_not_hide_previous_committed_manifest(tmp_path):
    request, kwargs, calls, objects = setup()
    objects["runs/lego-robot/" + kwargs["job_id"] + "/step_000000305/" + "b" * 64 + ".pt"] = (b"partial", "partial")
    assert recover(request, tmp_path, **kwargs)["step"] == 304


@pytest.mark.parametrize("failure", ["truncated", "wrong-version", "wrong-checksum", "wrong-body"])
def test_incomplete_or_mismatched_recovery_never_commits_receipt(tmp_path, failure):
    original, kwargs, calls, objects = setup()
    def request(*args):
        result = original(*args)
        if failure == "truncated" and args[1] == "list-objects-v2":
            result["IsTruncated"] = True
        elif failure == "wrong-version" and args[1] == "get-object":
            result["VersionId"] = "wrong"
        elif failure == "wrong-checksum" and args[1] == "head-object":
            result["ChecksumSHA256"] = "wrong"
        elif failure == "wrong-body" and args[1] == "get-object":
            Path(args[-1]).write_bytes(b"changed")
        return result
    with pytest.raises(ValueError):
        recover(request, tmp_path, **kwargs)
    assert not (tmp_path / kwargs["job_id"] / "receipt.json").exists()
