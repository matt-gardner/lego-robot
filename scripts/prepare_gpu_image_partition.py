"""Freeze missing task images and their exact binary-model inputs for GPU staging."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

from render_pile_first import read, receipt_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--completed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--publication-root", type=Path, required=True)
    args = parser.parse_args()
    pool_sha = hashlib.sha256(args.pool.read_bytes()).hexdigest()
    pool = read(args.pool)
    if pool["status"] != "complete":
        raise ValueError("GPU partition requires a completed immutable pool")
    args.output.mkdir(parents=True, exist_ok=False)
    jobs = []
    assets = {args.pool}
    for entry in pool["scenes"]:
        root = Path(entry["root"])
        for name, sha in entry["files_sha256"].items():
            path = root / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
                raise ValueError(f"Changed admitted asset: {path}")
            assets.add(path)
        for task in read(root / "problem.json")["tasks"]:
            existing = receipt_path(args.completed, entry, task)
            if existing.exists():
                receipt = read(existing)
                if (receipt["pool_sha256"] != pool_sha
                        or hashlib.sha256((args.completed / receipt["image_file"]).read_bytes()).hexdigest()
                        != receipt["sha256"]):
                    raise ValueError(f"Changed completed image: {existing}")
                continue
            jobs.append({"entry": entry, "task": task, "sha": pool_sha, "scope": "selected",
                         "phase": "gpu-backlog", "variant": "cuda_ad_rgb",
                         "output": str(args.publication_root / pool_sha)})
    manifest = {"schema_version": 1, "pool_sha256": pool_sha, "task_count": len(jobs),
                "jobs": jobs}
    (args.output / "jobs.json").write_text(json.dumps(manifest) + "\n")
    asset_manifest = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in sorted(assets)}
    (args.output / "assets.json").write_text(json.dumps(asset_manifest, indent=2) + "\n")
    with tarfile.open(args.output / "inputs.tar.gz", "w:gz") as archive:
        for path in sorted(assets):
            archive.add(path, arcname=str(path).lstrip("/"), recursive=False)
    print(json.dumps({"jobs": len(jobs), "pool_sha256": pool_sha,
                      "jobs_sha256": hashlib.sha256((args.output / "jobs.json").read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
