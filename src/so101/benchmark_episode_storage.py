"""Bounded CPU byte-read comparison; does not decode images or run training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

from so101.episode_storage import EpisodeStorage, consolidate


def benchmark(collection: Path, destination: Path, reads: int = 1024) -> dict:
    if reads <= 0:
        raise ValueError("reads must be positive")
    start = time.perf_counter()
    report = json.loads((collection / "collection.json").read_bytes())
    images = []
    frames = 0
    for source in report["successful_episodes"]:
        rows = (Path(source) / "frames.jsonl").read_bytes().splitlines()
        frames += len(rows)
        images.extend((Path(source), json.loads(row)["image"]) for row in rows)
    original_startup = time.perf_counter() - start
    start = time.perf_counter()
    population = consolidate(collection, destination)
    conversion = time.perf_counter() - start
    start = time.perf_counter()
    with EpisodeStorage(destination) as storage:
        for source in storage.catalog["successful_episodes"]:
            storage.read(Path(source), "episode.json")
            storage.read(Path(source), "frames.jsonl")
        container_startup = time.perf_counter() - start
        measurements = {}
        count = min(reads, len(images))
        random_indices = random.Random(61000).sample(range(len(images)), count)
        for mode, indices in (
            ("sequential", range(count)), ("random", random_indices)
        ):
            timings = {}
            digests = []
            for backend in ("original", "container"):
                digest = hashlib.sha256()
                size = 0
                start = time.perf_counter()
                for index in indices:
                    source, name = images[index]
                    content = ((source / name).read_bytes() if backend == "original"
                               else storage.read(source, name))
                    digest.update(content)
                    size += len(content)
                elapsed = time.perf_counter() - start
                timings[backend] = {
                    "seconds": elapsed, "bytes": size, "reads": count,
                    "bytes_per_second": size / elapsed,
                }
                digests.append(digest.hexdigest())
            if digests[0] != digests[1]:
                raise ValueError(f"{mode} byte equivalence failed")
            measurements[mode] = timings
    return {
        "population": {**population, "frames": frames},
        "conversion_seconds": conversion,
        "startup_seconds": {
            "original_catalog_and_rows": original_startup,
            "container_catalog_and_metadata_bytes": container_startup,
        },
        "image_reads": measurements,
        "caveat": "warm-cache byte reads; startup phases differ; no decode or training measurement",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--reads", type=int, default=1024)
    args = parser.parse_args()
    print(json.dumps(benchmark(args.collection, args.destination, args.reads), sort_keys=True))


if __name__ == "__main__":
    main()
