from __future__ import annotations

import argparse
import json
from pathlib import Path

from so101.episode_storage import consolidate


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a new immutable episode container")
    parser.add_argument("collection", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(consolidate(args.collection, args.destination), sort_keys=True))


if __name__ == "__main__":
    main()
