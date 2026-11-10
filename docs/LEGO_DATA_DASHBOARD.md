# Workstream 2: Interactive catalog and curriculum dashboard

## Data buckets publication (schema 1)

The read-only `/buckets` page reads `--bucket-dashboard-root`, defaulting to
`data/bucket_dashboard` beside the configured outputs root. The current producer
publication uses
`outputs/entity_smolvla_sft_sweep_24576_direct_82b32533/bucket_dashboard`
in the shared data workspace (writable by the producer without permissions changes).
Trainer owns calibration and INITIAL renders; generator owns native
problem identities/tags. `current.json` atomically publishes
`{"path":"snapshots/<id>/manifest.json","sha256":"<manifest bytes SHA256>"}`.
Write immutable snapshot assets before replacing this pointer.
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
Images are hash-checked and served only through exact published problem identity.

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
The primary overview is a compact bucket/task/assistance table with population,
success/counts/intervals, measured-problem coverage, mixed-group yield, per-problem
success range and initial-image coverage. Unmeasured task populations retain a
visible row; clicking a measured or unmeasured row filters the problem inspector.

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
