"""Compatibility validation for retired image-publication callbacks; never publish."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def validate_live_images(images):
    count = 0
    for path in sorted(images.glob("*.json")):
        row = json.loads(path.read_bytes())
        name = row["image_file"]
        if Path(name).name != name or row["physics_unchanged"] is not True:
            raise ValueError("Invalid unchanged INITIAL image receipt")
        if hashlib.sha256((images / name).read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("INITIAL image differs from its durable receipt")
        count += 1
    if not count:
        raise ValueError("No complete durable image receipts")
    return {"mode": "live-file-validation", "validated_receipts": count, "published": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--publication-id", required=True)
    args = parser.parse_args()
    print("Image publication is retired; the dashboard reads durable receipts live. "
          "This compatibility command only validates assets.", file=sys.stderr)
    print(json.dumps(validate_live_images(args.images)))


if __name__ == "__main__":
    main()
