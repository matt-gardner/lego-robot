# Canonical pile directories

The canonical layout is `lego-robot-data/pieces/` for the catalog and
`lego-robot-data/piles/pile_0001/` for each pile. The directory migration is
complete for retained data. The dashboard reads these directories directly; the
previous SQLite store and obsolete generation datasets have been removed.

Each pile contains the original `model.mjb`, `model.xml`, `problem.json`,
`integration_state.json`, and `scene_plan.json`, plus `pile.json`. If an original
pile image exists, it is retained byte-for-byte as `rendered.png`; missing images
are explicitly recorded, not synthesized. `tasks/task_001.json` retains the
original dashboard record and task definition, piece identity, initial arm
joint positions, reset state, destination, and assistance checkpoint references.

Actual persisted assistance variants include their complete saved initial state,
task definition, original assistance protocol, and hash-bound model files under
`checkpoints/`. Compact measurement receipts live under `assistance/`; they are
not executable checkpoints and are never presented as reconstructed state vectors
or new grasp-hold qualification.

`scripts/import_pile_directories.py` performs the one-time import from the old
dashboard store. It hardlinks immutable source files without moving, modifying,
or changing permissions on active workers' original inputs. Task revisions can
inherit a baseline with its original scene identity only when every task has an
exact revision-specific state and all source files match the admission hashes.

`piles/metadata.json` retains shared measurement metadata. Primary tasks are read
from their individual files by `so101.pile_directory`, not from a SQLite fallback.
The reader streams its API projection without retaining the decoded task population.

There are no data or catalog file locks. JSON installation uses atomic replacement.
An initial import keeps a `.incomplete` marker until the complete tree is published;
the reader rejects an incomplete tree. Do not remove the marker after a failed
import. Resume with the same inputs, or inspect the reported source mismatch.
The initial importer refuses to overwrite an already completed canonical tree.

For ongoing assistance outcomes, run:

```sh
PYTHONPATH=src:. python scripts/import_pile_directories.py \
  --root /path/to/lego-robot-data --refresh-assistance \
  --assistance-root /path/to/frozen-assistance-run
```

Only finalized `receipts/*.json` files are consumed. Batch indexes, status files,
temporary writes and coordinator locks are not read. Existing task state and
qualification remain unchanged; outcomes retain their original inventory labels
and lineage. Repeating the refresh does not rewrite unchanged task records.
Unknown parents, task hash mismatches and corrupted retained copies are errors,
not silently skipped outcomes. Each updated task is installed atomically; an
interrupted refresh can be safely repeated. Source receipts must remain immutable.
The receipt index retains only source paths and hashes, not the decoded population.
Each payload is decoded and its hash rechecked when attached, keeping memory bounded
as the completed receipt population grows.

The host runs this refresh through `lego-canonical-assistance.service` and
`lego-canonical-assistance.timer`, every two minutes after the preceding run ends.
Systemd prevents overlapping runs. The pinned published adapter reads fast-v4
receipts; its protected filesystem allows writes only beneath the canonical
`piles/` directory. CPU is capped at 50%, memory at 128 MiB, and each invocation
has a two-minute timeout. Failed exports remain visible in the service journal.
This is a data export, not an assistance job launcher or a worker-restart watchdog.
The frozen assistance runtime, source inventories and task protocols are unchanged.
Small receipt files are copied byte-for-byte and installed atomically so export
also works across read-only/read-write mount boundaries; model files remain linked.

Original XML retains its hash-bound absolute mesh paths. The 18 referenced robot
meshes live under `piles/assets/robot-model/assets`; a compatibility alias preserves
the original XML paths without duplicating assets. Compiled models embed meshes.

The initial migration preserved 155 piles and 32,336 primary tasks, with 95 original
pile images and 465 saved assistance variants. The other 60 piles explicitly lack
rendered images. All 409 accepted retained training reports (293 GRASP, 116 PLACE)
and their recursive reset dependencies are preserved: 567 unique five-file source
roots, verified by actual compiled-model resets after legacy deletion.

`piles/source-paths.json` maps historical source roots to canonical directories.
`so101.canonical_sources.canonical_source_root` is consumed by the native task
loader, preserving archived report and entry hashes without rewriting their path
strings. Future historical replay must use this loader, not a pre-migration copy.
Additional retained training states live beneath their exact parent pile's
`checkpoints/training_*` directory.

The active assistance run lives under `piles/measurements/fast-v4`. Its frozen
inventory cannot change paths without changing job identities, so a small
`piles/compatibility/` view and the top-level `data` symlink keep its 62 source
directories and historical runtime paths resolvable. The view uses hardlinks to
canonical models, not another allocated copy of the retired generation pools.
Real directories were relocated atomically with `renameat2(RENAME_EXCHANGE)`;
open files, worker staging, locks, frozen manifests and task protocols were
preserved. The assistance owner verified continuing receipts with zero errors.

Obsolete catalog experiments, renderer datasets, unused physics-generation
batches, stopped assistance copies and legacy curriculum trees were retired only
after source closure and privileged process-reference checks. Recursive retained
assistance receipt/inventory lineage is preserved by exact hash under
`piles/measurements/provenance/`, including its historical-path mapping.

The compatibility paths must remain until the active frozen assistance run has
finished and its owner has verified final receipts. They are aliases, not retained
obsolete datasets. Model caches and training outputs are separate from pile data.
