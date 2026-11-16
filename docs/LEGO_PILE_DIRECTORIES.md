# Canonical pile directories

The canonical layout is `lego-robot-data/pieces/` for the catalog and
`lego-robot-data/piles/pile_0001/` for each pile. The directory migration is
in progress. The dashboard reads these directories directly; the previous SQLite
store is retained only as migration evidence, not as a read fallback.

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

This layout does not yet promise standalone XML asset closure: original XML can
retain absolute mesh paths. Original compiled models and active source paths must
remain available until consumer migration and reset validation are complete.

The initial migration preserved 155 piles and 32,336 primary tasks, with 95 original
pile images and 465 saved assistance variants. The other 60 piles explicitly lack
rendered images. Legacy producer and assistance runtimes still require coordinated
publication/consumer transitions before their old directories can be retired.
