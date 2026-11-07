"""Freeze spec-independent user data before constructing the job specification."""
from __future__ import annotations

import base64
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import zlib


def userdata(root: Path, identity: dict, *, source_sha256: str, launcher_sha256: str) -> bytes:
    if any(not re.fullmatch(r"[0-9a-f]{64}", value)
           for value in (source_sha256, launcher_sha256)):
        raise ValueError("User data must pin raw source and launcher hashes")
    code = (root / "scripts/so101-worker-early.py").read_bytes()
    guard = (root / "src/so101/aws_worker_guard.py").read_bytes()
    unit = (root / "scripts/lego-robot-worker-guard.service").read_bytes()
    config = dict(identity, source_sha256=source_sha256, launcher_sha256=launcher_sha256,
                  early_sha256=hashlib.sha256(code).hexdigest(),
                  guard_sha256=hashlib.sha256(guard).hexdigest(),
                  guard_unit_sha256=hashlib.sha256(unit).hexdigest())
    if set(identity) != {"account_id", "region", "ami", "instance_type", "broker_id", "aws_cli"}:
        raise ValueError("User data identity must not contain job/spec/configuration hashes")
    payload = json.dumps({"config": config, "code": base64.b64encode(code).decode(),
                          "guard": base64.b64encode(guard).decode(),
                          "unit": base64.b64encode(unit).decode()}, separators=(",", ":")).encode()
    encoded = base64.b64encode(zlib.compress(payload)).decode()
    result = (
        "#!/bin/sh\nset -eu\n/usr/bin/python3 -I - <<'PY'\n"
        "import base64,json,zlib\n"
        f"p=json.loads(zlib.decompress(base64.b64decode('{encoded}')))\n"
        "code=base64.b64decode(p['code']); n={'__name__':'so101_early'}\n"
        "exec(compile(code,'<pinned-early-bootstrap>','exec'),n)\n"
        "n['bootstrap'](p['config'],base64.b64decode(p['guard']),"
        "base64.b64decode(p['unit']),code)\nPY\n"
    ).encode()
    if len(result) > 16384:
        raise ValueError("Spec-independent EC2 user data exceeds 16KiB")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--launcher-sha256", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    body = userdata(args.root, json.loads(args.identity.read_bytes()),
                    source_sha256=args.source_sha256, launcher_sha256=args.launcher_sha256)
    with args.destination.open("xb") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"bootstrap_sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)}))


if __name__ == "__main__":
    main()
