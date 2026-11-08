"""CPU-only loader equivalence and warm-cache timing; never loads a model."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from so101.control_model.simulation_dataset import EntityTrajectoryDataset


def benchmark(collection: Path, storage: Path, excluded: tuple[str, ...], reads: int) -> dict:
    if reads <= 0:
        raise ValueError("reads must be positive")
    results = {}
    signatures = []
    for backend, storage_path in (("original", None), ("container", storage)):
        start = time.perf_counter()
        dataset = EntityTrajectoryDataset(
            collection, 50, 4, excluded_sources=excluded, storage_path=storage_path
        )
        startup = time.perf_counter() - start
        digest = hashlib.sha256()
        for source, metadata, rows in dataset.episodes:
            digest.update(str(source).encode())
            digest.update(json.dumps([metadata, rows], sort_keys=True).encode())
        digest.update(json.dumps({
            "samples": dataset.samples, "pairs": dataset.pair_groups,
            "normalization": dataset.normalization(),
        }, sort_keys=True).encode())
        rng = np.random.default_rng(61000)
        paired = dataset.paired_indices(rng, 64)
        digest.update(json.dumps(paired).encode())
        digest.update(json.dumps(rng.bit_generator.state, sort_keys=True).encode())
        count = min(reads, len(dataset))
        timings = {}
        for mode, indices in (
            ("sequential", list(range(count))),
            ("random", np.random.default_rng(61000).choice(len(dataset), count, replace=False).tolist()),
        ):
            start = time.perf_counter()
            for index in indices:
                np.random.seed(61000 + index)
                example = dataset[index]
                for key, value in sorted(example.items()):
                    digest.update(key.encode())
                    if isinstance(value, torch.Tensor):
                        digest.update(json.dumps([str(value.dtype), list(value.shape)]).encode())
                        digest.update(value.numpy().tobytes())
                    else:
                        digest.update(json.dumps(value).encode())
            elapsed = time.perf_counter() - start
            timings[mode] = {"examples": count, "seconds": elapsed, "examples_per_second": count / elapsed}
        results[backend] = {
            "startup_seconds": startup, "episodes": len(dataset.episodes),
            "samples": len(dataset), "access": timings,
        }
        normalization = dataset.normalization()
        catalog = json.loads((collection / "collection.json").read_bytes())["successful_episodes"]
        if dataset.storage is not None:
            dataset.storage.close()
        del dataset
        gc.collect()
        if excluded:
            start = time.perf_counter()
            held_out = set(excluded)
            probe = EntityTrajectoryDataset(
                collection, 50, 4, normalization=normalization,
                excluded_sources=tuple(source for source in catalog if source not in held_out),
                storage_path=storage_path,
            )
            probe_startup = time.perf_counter() - start
            episodes = {str(source): index for index, (source, _, _) in enumerate(probe.episodes)}
            rng = np.random.default_rng(1061000)
            indices = []
            for source in excluded:
                candidates = probe.episode_sample_indices[episodes[source]]
                indices.extend(int(index) for index in rng.choice(
                    candidates, size=4, replace=len(candidates) < 4
                ))
            for source, metadata, rows in probe.episodes:
                digest.update(str(source).encode())
                digest.update(json.dumps([metadata, rows], sort_keys=True).encode())
            digest.update(json.dumps([indices, probe.normalization()]).encode())
            start = time.perf_counter()
            for index in indices:
                np.random.seed(1061000 + index)
                for key, value in sorted(probe[index].items()):
                    digest.update(key.encode())
                    if isinstance(value, torch.Tensor):
                        digest.update(json.dumps([str(value.dtype), list(value.shape)]).encode())
                    digest.update(value.numpy().tobytes() if isinstance(value, torch.Tensor)
                                  else json.dumps(value).encode())
            results[backend]["frozen_probe"] = {
                "episodes": len(probe.episodes), "examples": len(indices),
                "startup_seconds": probe_startup,
                "read_seconds": time.perf_counter() - start,
                "normalization": "training-only", "seed": 1061000,
            }
            if probe.storage is not None:
                probe.storage.close()
            del probe
            gc.collect()
        signatures.append(digest.hexdigest())
    if signatures[0] != signatures[1]:
        raise ValueError("Loader metadata, normalization, sampling RNG or example content differs")
    return {"results": results, "equivalent_sha256": signatures[0],
            "excluded_sources": len(excluded), "seed": 61000,
            "caveat": "warm-cache CPU loader; 50-action horizon/4-history; no model or training"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection", type=Path)
    parser.add_argument("storage", type=Path)
    parser.add_argument("--validation-split", type=Path)
    parser.add_argument("--reads", type=int, default=32)
    args = parser.parse_args()
    excluded = tuple(json.loads(args.validation_split.read_bytes())["validation"]) if args.validation_split else ()
    torch.set_num_threads(1)
    print(json.dumps(benchmark(args.collection, args.storage, excluded, args.reads), sort_keys=True))


if __name__ == "__main__":
    main()
