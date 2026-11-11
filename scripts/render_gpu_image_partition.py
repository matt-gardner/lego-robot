"""Run a bounded reserved image partition; each completed task is independently atomic."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--deadline-unix", type=float, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_bytes())
    if args.limit < 1:
        raise ValueError("A positive bounded task limit is required")
    completed = 0
    active = args.manifest.with_name("active-job.json")
    for job in manifest["jobs"]:
        if time.time() >= args.deadline_unix:
            break
        if job["sha"] != manifest["pool_sha256"] or job["variant"] != "cuda_ad_rgb":
            raise ValueError("Job differs from reserved GPU partition")
        source = str(Path(job["entry"]["root"]) / "tasks" / job["task"]["task_id"])
        receipt = Path(job["output"]) / (hashlib.sha256(source.encode()).hexdigest() + ".json")
        if receipt.exists():
            row = json.loads(receipt.read_bytes())
            if hashlib.sha256((receipt.parent / row["image_file"]).read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError(f"Changed completed GPU image: {receipt}")
            continue
        active.write_text(json.dumps(job) + "\n")
        subprocess.run([sys.executable, str(Path(__file__).with_name("render_pile_first.py")),
                        "--job", str(active)], check=True,
                       timeout=max(1, args.deadline_unix - time.time()))
        completed += 1
        if completed >= args.limit:
            break
    print(json.dumps({"new_completed_tasks": completed, "reserved_tasks": manifest["task_count"]}))


if __name__ == "__main__":
    main()
