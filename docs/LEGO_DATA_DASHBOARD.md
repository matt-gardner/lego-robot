# Workstream 2: Interactive catalog and curriculum dashboard

## Live data buckets (schema 1)

The read-only `/buckets` page reads `--bucket-dashboard-root`, defaulting to
`data/bucket_dashboard` beside the configured outputs root. Shared data uses
`outputs/entity_smolvla_sft_sweep_24576_direct_82b32533/bucket_dashboard`
in the shared data workspace (writable by the producer without permissions changes).
The running dashboard precomputes its inventory, serialized API response and
problem/image lookup in one background cache worker at startup and every
20 seconds. HTTP requests reuse that snapshot; individual image requests do not
rebuild or rescan the inventory. Original image bytes remain hash-checked.
Cache refresh errors are logged and surfaced explicitly, not hidden as fresh
data. The browser builds task modal contents only when a pile is opened.
The dashboard composes ordinary producer files during these background refreshes.
No renderer or generator session needs to run a dashboard publisher.
The renderer writes atomic completed per-task receipts and assets under
`initial-photo-external-v<N>/<pool SHA>/`; higher positive numeric versions take
precedence per completed task, not per entire batch. Within a version,
`_roles-v1/<task>.json` takes precedence over that task's original receipt.
Invalid selected receipts fail explicitly rather than hiding behind older data.
The reader validates pool, canonical source,
scene hash, target, reset invariants, annotations and image hashes, then serves
the original asset without copying it. A metrics callback replacing a legacy
snapshot cannot erase these images.

`live-metrics/<scope>/results.json` contains the live inference result file.
It is joined only to its exact `input_sha256.native_pool`; horizon and scope
are explicit. Its groups replace the legacy current metrics rather than mixing
50-step screening with archived 300-step outcomes. Multiple current result
files for the same pool fail explicitly instead of combining protocols.
Actual transport of remote result bytes is separate from dashboard publishing.
Current selector inventories also join existing shared
`publications/*/current.json` measurements by exact pool SHA and validate each
measured problem's task, canonical source and scene hash. The shared protocol,
progress and groups are preserved; unmeasured pools remain unmeasured. Rates
describe only the sampled problems, not the entire enumerated pool.

Tasks are read immediately from each admitted pile's hash-pinned `problem.json`,
without waiting for a complete pool or rendered preview. Active generator
admissions under `photo-task-progression-collective-v1` and
`catalog-diverse-20-v1` are discovered independently of renderer progress.
Canonical task IDs deduplicate selected-pool tasks. All admitted tasks appear
in their piece-count bucket and carousel; matching images attach only to their
exact task. Missing images and unmeasured policy rates remain separate states.
The reader discovers `pile-previews-v1/<admission SHA>/admission.json`, validates
the pin and scene problem hash, and shows an explicit pending state until an
atomic task receipt and original PNG arrive in that directory. Receipt pool SHA
means admission SHA in this channel, not a complete pool. These previews never
add task populations or policy groups, and are served through the same bounded,
hash-checked image endpoint with an `admission:<SHA>` identity.

`--bucket-curriculum-root` defaults to the existing read-only
`/mnt/data/lego-curriculum` mount. The generator's atomic
`current-task-pools.json` (`kind: lego-current-task-pools`, `schema_version: 1`,
`pools: [{manifest_path,sha256,scenes,task_count,piece_counts}]`) selects current
complete TRAIN pools directly. Paths under the canonical producer prefix
`/mnt/dockyard/lego-robot-data/data/lego_curriculum` are mapped into that mount,
without changing canonical task identities. Missing matching images remain
unrendered, never substituted from a previous task/reset. Pool hashes, scene
problem hashes and population counts are checked. No recursive search or mtime
guessing selects historical/subset pools.

Until that generator selection exists, old inventories remain readable via
the compatibility snapshot format. `current.json` points to
`{"path":"snapshots/<id>/manifest.json","sha256":"<manifest bytes SHA256>"}`.
These legacy snapshots are not required for new generator or renderer updates.
The manifest has `kind: lego-data-buckets`, `schema_version: 1`,
`generated_at`, `measurement_status` (screening/calibrating/qualified),
`provenance` (generator/source/grader/protocol hashes), `protocol`, `buckets`,
`problems`, and `groups`. Unknown extension fields are retained.

Buckets enumerate every bucket/task population independently of measured samples:
`{id,label,population_total,tasks:[{task,population_total}]}`. Problems enumerate
the entire population, with stable `id`, `bucket_id`, `task`, source identity and
hash, split, native scene/task IDs where applicable, tags, provenance,
`feasibility:{status,evidence}`, and `initial_images`. Missing measurements,
physical qualification, visibility/reach and images are explicitly unmeasured.
Image entries contain snapshot-relative `path`, byte `sha256`, `camera`, actual
`width/height` and renderer/protocol provenance; PNG/JPEG only, no upscaling claim.
Images are hash-checked and served only through exact selected problem identity.

Groups join by `problem_id`; legacy raw groups may join the exact
bucket/task/source/problem_index tuple. Preserve assistance, attempts (rollouts),
successes, invalid actions, trajectories, seed, measurement status and
`mixed_group_yield:{eligible,total}`. Feasibility failure attribution is null
unless measured; observed unsuccessful rollouts do not establish infeasibility.
Pilot four-rollout groups are screening only, never calibrated qualification.
The page shows app-derived Wilson 95% descriptive intervals (without accounting
for group/seed correlation), actual within-group mixed yield, per-problem curves
and within-bucket observed spread. Piece count/proxy tags are not difficulty,
and pooled 50% success does not imply per-problem learnability. No GPU queries,
model loads, old deleted renders or temporary paths are used.
The primary overview has one row per bucket, with a small chart in a table cell:
assistance percent on the x axis and success rate on the y axis, both 0–100%.
Task types have distinct colors; points show measured rates with descriptive
Wilson intervals and exact numerator/denominator tooltips, without interpolation.
Per-task sampled-problem coverage and rollout counts sit beside the chart.
Unmeasured buckets retain an explicit missing state instead of zero-valued points.
Clicking a row filters the pile inspector to its bucket.
The dashboard excludes historical `source.kind=legacy` problems and `corpus:`
bucket summaries from the overview, selectors and problem inspector. This is a
display-only retirement; producer snapshots, calibration and underlying data are
unchanged. The remaining published population currently comprises native buckets.
Below the table, the inspector groups bucket → pile → target-specific task.
Pile identity is the source scene ID (or an explicit pile/scene ID); missing
identity stays separate rather than merging unrelated problems. Pagination counts
piles, not tasks, so a pile's matching tasks stay together. Collapsible bucket
sections contain pile cards with one initial pile image and meaningful scene
tags (piece/shape/color counts, colors, obstructions and available tasks).
Click a pile card to reveal target tasks, labeled by action, object number and
dimensions when available. Technical identities remain in provenance, not card
headings. Task details expand to measurements and provenance.

An optional top-level `progress` object distinguishes expert qualification from
policy measurements: `phase`, `reference_attempts`, `reference_accepted`,
`policy_calibration_status`, `policy_rollouts`, `pool_sha256`. The page labels
reference acceptance as **not model pass rate** and zero policy rollouts as no
policy rates measured yet. A new pool publishes its own population/protocol and
empty groups until measured, never recycled old-pool screening outcomes.

Independent datasets publish `publications/<dataset>/current.json` in the same
shared root, each pointing to a root-relative hash-pinned snapshot. The dashboard
automatically discovers these pointers on every API read, retains all datasets
and their protocol/progress, and joins measurements only to exact problem IDs.
This is the legacy inventory fallback, superseded by `current-task-pools.json`.
Duplicate problem identities fail explicitly rather than mixing datasets.
Root `current.json` remains a compatibility fallback only when no dataset pointers
exist. The browser refreshes publications every 20 seconds unless a pile modal is
open, preserving inspection; manual Refresh is also available.
Pile cards open a native dialog with close/Escape/backdrop dismissal instead of
expanding in the grid.
The dialog presents one selected task with a large annotated image at the top
and a horizontally scrolling strip of smaller task thumbnails below. Previous,
Next, thumbnail clicks and arrow keys change the selection; task measurements
and provenance remain attached to the selected task.

Each task image can publish `target_bbox_xyxy` in image pixels
`[xmin,ymin,xmax_exclusive,ymax_exclusive]`, `target_visible_pixels` and
`target_bbox_status` (visible/not_visible/unmeasured), directly or in image
provenance. Boxes must be measured from exact target-geometry segmentation for
the hash-bound RGB image and actual dimensions. The task inspector overlays
these boxes at responsive image scale; absent/zero-visible targets show explicit
unmeasured/not-visible states, not guessed or amodal boxes.

Push/place images support `target_annotations` (image-level or in provenance),
with independent `source_object` and `destination` roles using those same box
and visibility fields. Source boxes are yellow; destination boxes are cyan,
with a textual legend. The destination is visible goal-marker geometry, not
a second LEGO object or an inferred tolerance region. Include `entity_id`,
`entity_kind`, `bbox_semantics` and the original `goal` for provenance.
Legacy singular boxes remain source-only; missing destination measurements
are explicitly labeled.

The buckets page remembers bucket collapse state, provenance disclosure state,
bucket/task/search filters, the loaded pile count and each pile's selected task
in browser-local storage across manual and automatic refreshes. Removed filter
values and task identities fall back to the available publication. Storage
failures are logged without preventing browsing.

Approved by Matt November 8, 2026. A dedicated worker owns dashboard discovery,
browsing, feedback and activation, not collection or part qualification.

## Outcome

Show the available parts catalog and current/historical data generations.
Use `LEGO_PARTS_CATALOG.md` for the catalog contract and
`LEGO_CURRICULUM.md` for scene/example metadata. Show imported, prepared,
verified and unavailable parts separately; loading a mesh is not physical
qualification or demonstrated robot competence.

Each example needs a stable ID, generation/release ID, category, task, part IDs,
scene family, split, renderer/catalog fingerprints, geometry/configuration
tags, expert acceptance and visual-audit status. Filter by those fields and
show current work, rejected attempts and prior immutable generations without
rewriting historical labels. Distinguish legacy final-success, clean expert,
post-failure recovery and not-in-training examples.

Expose wrist and external video, selected-object identity, acceptance evidence,
and the failed parent of a recovery without presenting failed actions as SFT
targets. Support parts previews and explicit preparation failure reasons.

## Feedback

Attach comments and "looks wrong" feedback to exact immutable part/example IDs
and versions, optionally a frame/time and issue category. Persist feedback
with delivery state and idempotency identity. Route through the request monitor
to the appropriate catalog, realism or curriculum owner; confirm delivery or
show the actual failure. Do not silently acknowledge undelivered comments.

Preserve existing access controls. Coordinate missing Dockyard/request-monitor
functionality with the Work Scheduler rather than inventing an unauthenticated
agent-message endpoint or writing under Matt's identity.

## Implementation and activation

Reuse the existing dashboard and data browser, keeping current run views and
labels intact. The browser reads directory episode metadata and immutable EpisodeStorage
SQLite releases through its read-only adapter; hidden staging paths are excluded.

Physical-pair raster derivatives retain source example/family IDs and splits,
but have distinct artifact-bound release and dashboard view IDs. They must carry
`physical_pair_render` provenance and a matching `qualification.json` with
image/sample validation and storage hash. These views are labeled **Derived CPU
raster**, never original camera recordings or additional unique training data.
Physical source acceptance, image/sample qualification, visual audit and training
admission remain separate claims.

Publication verification uses `/api/lego?release=<artifact-bound-release-id>&item=<source-example-id>`
for provenance/frame identity, and `/api/data/all?archive_release=<artifact-bound-release-id>`
for the exact derivative-only population. Its returned dashboard item ID selects
details and native video through the existing `view=examples&item=<dashboard-id>`
endpoints. Native encoding reads archived derivative images without rerunning
physics. Original releases and their IDs/labels remain unchanged. A published
acknowledgment requires immutable publication and live projection/media evidence;
staging or physical success alone does not satisfy it.

Managed catalog status uses `/api/lego?inventory=parts` to verify immutable parts
publications without transferring unrelated example qualification receipts.
This projection retains complete parts release summaries and qualification notices;
the default inventory and selected-release browser responses are unchanged. The
managed verifier retains its response-size and release-count limits and rejects
non-parts rows explicitly. Its release counts describe the parts inventory only;
data/media verification remains separate.

Own frontend/backend dashboard changes, focused tests, managed activation and
live verification. Coordinate file ownership with realism and curriculum
workers. No GPU, training, hardware motion or deletion of historical data.
