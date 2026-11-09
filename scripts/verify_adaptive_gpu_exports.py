"""Independently confirm immutable S3 versions for a completed comparison."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from urllib.parse import urlsplit


def request(operation, artifact, *options):
    uri = urlsplit(artifact["uri"])
    if (uri.scheme != "s3" or uri.netloc != "so101-training-133961745003-us-west-2"
            or not uri.path.startswith("/runs/lego-robot/") or not artifact["version"]):
        raise ValueError("Artifact is outside the versioned approved output namespace")
    response = subprocess.run([
        "aws", "--profile", "lego-artifacts", "s3api", operation,
        "--bucket", uri.netloc, "--key", uri.path.lstrip("/"),
        "--version-id", artifact["version"], "--checksum-mode", "ENABLED",
        "--expected-bucket-owner", "133961745003", "--region", "us-west-2",
        "--output", "json", "--no-cli-pager", *options,
    ], check=True, capture_output=True, text=True)
    return json.loads(response.stdout)


def confirm(artifact):
    head = request("head-object", artifact)
    expected = base64.b64encode(bytes.fromhex(artifact["sha256"])).decode()
    if (head["VersionId"] != artifact["version"] or head["ContentLength"] != artifact["size"]
            or head.get("ChecksumSHA256") != expected):
        raise ValueError(f"Independent S3 confirmation failed: {artifact['uri']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", type=Path)
    args = parser.parse_args()
    rows = json.loads(args.comparison.read_text())
    if len(rows) != 6 or len({row["arm"] for row in rows}) != 6:
        raise ValueError("Comparison must include all six distinct completed arms")
    files, total = 0, 0
    with tempfile.TemporaryDirectory(prefix="lego-export-verification-") as scratch:
        for row in rows:
            artifact = row["export"]["manifest"]
            confirm(artifact)
            path = Path(scratch) / "manifest.json"
            request("get-object", artifact, str(path))
            if hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]:
                raise ValueError("Downloaded manifest digest differs from export receipt")
            manifest = json.loads(path.read_text())
            steps = 24576 // int(row["arm"].split("-")[0][1:])
            names = {"training_state.pt", "model.pt", "entity_config.json",
                     "metadata.json", "sft_training.json", "data_split.json"}
            names.update(f"checkpoints/training_state_{step:06d}.pt"
                         for step in range(0, steps + 1, steps // 4))
            if (manifest["arm"] != row["arm"] or manifest["step"] != steps
                    or set(manifest["files"]) != names):
                raise ValueError("Manifest arm or complete artifact population differs")
            for entry in manifest["files"].values():
                confirm(entry)
                files += 1
                total += entry["size"]
    print(json.dumps({"verified_arms": len(rows), "verified_manifests": len(rows),
                      "verified_files": files, "verified_bytes": total}))


if __name__ == "__main__":
    main()
