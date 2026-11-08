# Indexed episode storage

`so101.consolidate_dataset` creates a **new** SQLite format-version-1 container
from a complete simulation collection. Run from the same working directory as
the original catalog's source paths:

```sh
python -m so101.consolidate_dataset data/simulation/COLLECTION /new/path/COLLECTION-v1.sqlite
```

The catalog and every regular file under its successful episodes are stored as
exact bytes, with indexed `(original source path, relative filename)` keys and
per-file SHA-256 checksums. Catalog order, source identities, images, frame
order, phases and acceptance metadata are not transformed. JPEGs are not
recompressed. This is an episode dataset container, not a model-cache format.
SQLite is used instead of compressed tar because training reads random
history windows, not just sequential episodes: indexed BLOB reads need no
archive scan, decompression seek or extracted image tree. It uses Python's
standard library rather than adding another storage dependency.

Publication uses an exclusive link only after SQLite integrity and all content
checksums pass; existing destinations are never overwritten. Originals are
never deleted or rewritten. Interrupted work is not a published container.
Use a versioned destination on a filesystem supporting hard links. Migration
requires space for the new copy; it does not reclaim original disk space.

`EpisodeStorage` opens the container read-only with SQLite's immutable mode,
and supports indexed byte reads without extraction or per-image file opens.
Only mount genuinely immutable completed copies; never modify a mounted
container. `verify()` explicitly checks all stored bytes. Bind deployments
to the container's whole-file SHA-256 and format version, not its filename.
Each process/thread reuses its own connection with a 16 MiB SQLite page-cache
target (not a process-memory cap). The trainer's main-thread training and
frozen-probe datasets therefore have two connections, with a combined 32 MiB
cache target. Worker copies reopen independently after fork or pickling.
No whole-file memory mapping or decoded-image cache is introduced.

The optional trainer `--episode-storage FILE` retains the existing
`--collection` catalog directory, working directory, frozen validation split
and original source identities. Only the small original `collection.json`
needs to remain present; episode metadata, frames and images (including frozen
validation/probe metadata) are read from the container. The storage selection
is provenance outside `run_config`, not a recipe change. Without the option,
directory loading is unchanged. Expert-acceptance guards apply to both modes.
No training resume, STOP removal or recipe change is authorized by migration.

CPU-only measurement tools:

```sh
python -m so101.benchmark_episode_storage data/simulation/COLLECTION /new/path/COLLECTION-v1.sqlite --reads 1024
OMP_NUM_THREADS=1 python -m so101.benchmark_indexed_loader data/simulation/COLLECTION /new/path/COLLECTION-v1.sqlite --validation-split outputs/RUN/data_split.json --reads 32
```

The first creates a new container, checks every original file against stored
bytes, and reports population, conversion time and equal sequential/random
image-byte reads. Its startup phases differ and are not a loader speedup.
The second loads each backend separately, comparing all episode metadata and
rows, sample/pair order, normalization, seed61000 paired indices and subsequent
sampling RNG state, plus identical seeded output tensors from sequential and
random examples. It measures real loader startup and examples/second at
horizon50/history4, without a policy, optimizer or training. Measurements are
warm-cache and do not establish cold-disk or GPU throughput.

Operational activation requires a newly bound immutable container artifact,
published source and a rebuilt consumer image/bundle with the flag. An older
worker is not activated by publishing source. Qualification and activation
receipts belong to the operator; never relabel an existing image or resume a
stopped job as part of storage migration.
