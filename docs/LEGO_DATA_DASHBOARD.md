# Interactive catalog and curriculum dashboard

## Current bucket data

The bucket browser has one data source:
`<bucket-dashboard-root>/current.sqlite3`. It contains the current task inventory,
measurement records and protocols, pile records, per-piece coordinates, and the
original retained pile-image bytes. The browser does not discover version
directories, choose snapshots, or overlay older image publications.

`BucketDashboard` reads this database only. Its background worker caches the
already serialized inventory every 20 seconds; it does not retain decoded copies
of the large task population during refresh. Image requests use an indexed database lookup and
verify the byte digest. Read/cache failures are logged and surfaced, not returned
as fresh success. `/api/buckets` retains task and measurement metadata for the
charts. `/api/buckets/image?problem=<pile ID>&index=0` serves the current pile
image without a snapshot selector. Historical source hashes and measurement
series identities remain scientific provenance, not selectable storage versions.

Writers call `so101.bucket_store.publish_current(root, manifest, asset_directory,
channel=...)`. They supply decoded current task records, original protocols and
groups, and root-contained hash-bound assets. One transaction replaces the
channel, preserves unrelated populations/measurements, stores any new pile
images, and removes unreferenced image bytes. Repeated updates do not accumulate
historical inventory files. The database uses full auto-vacuum so removed images
release pages rather than leaving an indefinitely growing file.

`so101.publish_photo_calibration` writes current measurement data directly into
the database. Independent measurement series keep their original protocols and
counts; mirrored compatibility groups are not counted twice. Measures remain
bound to exact task sources and pools. A result from a sampled subset is not
silently treated as a measured rate for an entire enumerated bucket.

`so101.native_pile_dashboard_images` uses a resumable `staging/native-piles`
working directory and writes the current population through the same writer.
Staging is renderer work, not a second browser data source. It must not be
removed while an owning renderer is using it. Source models, admissions,
integration states, task definitions, and physical parameters remain producer
inputs and are not dashboard cleanup targets.

## Pile and piece inspector

Task-specific dashboard photos are retired. Task records keep their source,
goal, feasibility and measurement data, but have no image assets. A pile has one
retained image and one card per physical piece, not four cards for its task roles.
Selecting a piece overlays its coordinates on the shared pile image.

Piece bounds must come from segmentation using the **same pile camera and
admitted initial state as that image**. Task-camera boxes must never be reused:
the cameras differ. Piece records retain identity, world position, visible pixel
count and exclusive pixel bounds. An occluded/offscreen piece has `not_visible`
and no invented box. Coordinate-generation provenance binds the scene files,
camera and retained image digest. These are TRAIN-start visualizations, not
policy observations, expert references, SFT qualification or proof of competence.

## Migration and cleanup

`scripts/import_current_dashboard.py` imports a pinned complete live inventory
and checked per-scene piece-coordinate files into the database. It checks that
the coordinate population and pile-image digests match before committing.
Migration must preserve current task/measurement fields and exact original
pile-image bytes. Verify the served API and representative images after managed
activation before removing obsolete dashboard directories.

Old `snapshots`, `publications`, `initial-photo-external-v*`, task-image shards
and unused dashboard render assets are not runtime inputs after migration.
Cleanup must enumerate exact in-scope paths, preserve any shared/live aliases
and separately required evidence, and never touch producer datasets, negatives,
training/checkpoint files or another worker's active runtime.

## Training observatory

Training logs are independent of bucket publications. Nested `sft_training.json`
logs with kind `piece_recognition_sft` appear in Recognition, with training
minibatch measurements separate from held-out evaluations. A sibling
`test_results.json` supplies final held-out results; absent results stay missing.
Recognition loss is not combined with action SFT or RL.

Nested `grasp_evaluation.json` files with kind `grasp_assistance_evaluation`
appear in Policy evaluation, separate from RL/SFT/recognition. The mirror writes
the atomic adapter and checked process heartbeat after each successful refresh.
Curves use completed rollout denominators and saved-state assistance, not
optimizer steps. Recovery-heldout is not an original-SFT test split.

After a successful mirror and remote health check, call
`so101.publish_grasp_evaluation.publish_evaluation(results_path, published_output,
state="running")`; use `completed` after completion. Refresh every 20 seconds.
Raw results/provenance remain unchanged; model weights and sample tensors are
not dashboard publications.
