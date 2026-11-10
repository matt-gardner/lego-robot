# Workstream 4: Continuous LEGO data curriculum

## Current direction: reference-free pile problems

Matt's current scope is continuous random-catalog pile generation, not more
demonstration trajectories or stored RGB. `so101.pile_task_generator` samples
verified parts from the single current catalog, cycles monochrome/few/many-color
distributions, and increases piece counts across successive proposals. Every
accepted pile supplies VIEW/GRASP/PUSH/PLACE tasks for every piece, including
occluded pieces, without policy screening or assumed solvability. Two opaque
physical panels provide additional VIEW obstructions. Initial visibility is not
measured during image-free generation; geometry-based burial tags are explicitly
proxies, not claims about visible fraction.

Run the new generator separately from the suspended historical producer:
`PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m
so101.pile_task_generator --catalog <catalog root>/current/catalog.json
--root <new pile pool>`. `--attempts N` bounds attempts; the default continuously
generates until STOP, termination, or resource backpressure. One writer holds the
pool lock; admission retains 10GiB disk/2GiB available memory, rejects memory
pressure, and pauses at a 4GiB pool budget with a 512MiB per-scene reserve. It does
not delete consumed data or assert that a byte budget represents multiple days of
RL supply. Throughput and the learner's consumption rate still need measurement.
Both CLIs run at nice 19; this is low scheduling priority, not a measured guarantee
of idle CPU capacity. Launch from an immutable source checkout for truthful source
fingerprints. No replacement service has been activated; the historical producer
remains suspended and its deleted monitoring cron must not be recreated.

Each scene stores a self-contained `model.mjb`, full MuJoCo integration state,
XML/proposal/provenance, persistent instance/body/geometry identities, tasks and
descriptive tags. Catalog locking covers loading and compilation; binary reset
does not depend on retaining superseded catalog directories. Exact binary reset
round-trip is checked before admission. Failed settling/contact proposals produce
explicit rejection receipts; operational/configuration failures are not hidden
as difficult scenes.
`admission.json` is written only after verification. Restart revalidates admitted
scenes; an interrupted scene lacking admission is retained and explicitly blocks
recovery rather than being silently accepted or overwritten.

`so101.pile_task_runtime` exposes `load_problem(entry, task_id)`,
`create_simulation(problem, renderbackend="none"|"mujoco")`,
`reset_problem(sim, problem)` and
`task_success(sim, problem, executed, stable) -> {"success", "stable"}`.
The simulation exposes its native model/data, instance positions and live
instance-specific segmentation boxes. Hidden targets have an invalid/null box;
unburying never changes the target identity. This is not a drop-in
`BeamSimulation` episode adapter. Learner action/observation integration and the
photorealistic dynamic-model backend remain separate qualifications.
Positions and lift thresholds use the piece's center of mass, matching the catalog
simulator. PUSH/PLACE goal coordinates bind the actual visible goal entity.

The versioned `native-pile-goals-v1` graders require sustained useful VIEW pixels
and centering, a stable bilateral lift for GRASP, and released, table-supported,
slow target placement at a destination for PUSH/PLACE (PLACE additionally requires
a prior lift). These are explicit simulated task contracts, not success-reference,
material-calibration, or real-world transfer claims.

Learner overrides use `python -m so101.pile_task_worker --request <absolute JSON>`.
Schema 1, kind `pile-task-generation`, requires `request_id`, `split: "train"`,
absolute `catalog`/`output_root`, `seed`, `max_scenes` (1..24),
`task_types: ["view", "grasp", "push", "place"]`,
`excluded_scene_families`, and `tag_constraints` containing `piece_count` (2..128)
and `color_mode` (`monochrome`, `few`, `many`). Unknown constraints fail explicitly.
The atomic `<request_id>.result.json` binds the request hash and reports
`complete`, `blocked`, or `rejected`; admitted scene entries bind all reset/task
files by hash. No expert reference or saved image is required. Existing results
are idempotent and changed requests cannot reuse an identity.
The output directory is bound to the request hash for recovery after interruption.

## Historical trajectory pipeline (suspended)

Approved by Matt November 8, 2026. This dedicated session owned scene preparation,
physics-only trajectories, clean paired physical validation, stage timings and
the accepted-pair handoff. Rendering, image-dependent qualification and browser
publication are downstream responsibilities. Catalog and host integration have
separate owners.

The photo-realism owner maintains the standalone native pile utility:
`PYTHONPATH=src python -m so101.settled_piles --catalog <pinned catalog> --output <new root> --parts 3020 3021 3022 --counts 3 2`.
Its saved `model.mjb` is the runtime authority for integration-state restoration;
XML retains pinned asset references. This is a separate reset artifact, not an
accepted demonstration pair, successful expert reference, or claim of RL
compatibility or task solvability.

## Physics-first generation and downstream handoff

The standalone `python -m so101.collect_harder_view --collection <existing root>
--model-root <SO101 assets> --training-sources <frozen training paths JSON>
--output <new root> --attempts 16` prepares compact
VIEW references using the existing episode/reset interface. It changes initial
base-joint poses, requires an initially unsatisfied legacy VIEW goal, regenerates
expert actions and independently replays them with pre-action state agreement and
the existing goal dwell. It renders transient segmentation, not image files.
The 10GiB disk admission and 64MiB output limit remain mandatory. Its separate
train/development manifest groups source scene seeds; it never replaces the fixed
evaluation split. These are candidate difficulty strata, not measured policy
difficulty or SFT qualification. No references exist until collection and replay
succeed. Runtime launch belongs to the trainer owner while local storage is blocked.
The required training-source allowlist is a JSON list of absolute episode paths
from the trainer's frozen training membership, excluding heldout scene families.
For a separate spatial candidate experiment, add `--profile spatial-v2
--exclude-manifest <previous manifest.json>`. This excludes every previously
admitted family and predeclares target XY/yaw plus base/shoulder/wrist pose strata.
Target height stays at its original table-supported height; no unsupported piles
or silent goal changes are introduced. Initial collisions, unreachable expert
vantages, and finite replay mismatches are rejected explicitly. These strata are
not proven harder: screen policy success on train families only, select the
recipe without development feedback, and evaluate untouched development families.
The bounded follow-up uses `--recipe-indices 8 20 28 --families-per-split 4
--attempts 24`, with repeated `--exclude-manifest` arguments for the union of
previous releases. The source allowlist still excludes frozen evaluation families.
Before simulation, the collector selects the first four fresh families per split
by sorted family hash (one source per family, sorted path), freezes the full
24-attempt cross-product in the building manifest, and fails if either split
lacks coverage. Selection uses no policy outcomes. These three recipes are
predeclared equally; do not select only the single previously failing example.

`python -m so101.produce_physical_catalog` runs the manipulation-only schedule
without constructing a renderer or writing images. It records pre-action full
qpos, qvel, actuator state/control, encoded robot state/action, phase, elapsed
simulation time and command identities/roles. Native physics and clean
single-pass checks remain active; both commands independently restore the same
settled state. Failed or interrupted attempts never enter the rendering queue.

Accepted pairs have collection status `physically_validated`, not `complete`.
The immutable schema-version-1 `lego-physical-pair` manifest in
`render_queue/ready/<release_id>.json` contains the source pair root, file hashes,
two example IDs, protected family/split membership and wall-clock stage timings.
It explicitly marks `images_pending: true` and `sft_qualified: false`.
The downstream worker restores each saved pre-action qpos and calls
`mj_forward`; it must not replay controls, mutate physical source files or
infer image/SFT qualification from physical success. Recompute image-dependent
labels from the rendering model and retain explicit derivative provenance.

The manifest and source files remain in place. Once downstream processing has
finished, the consumer writes `render_queue/acknowledged/<release_id>.json`
with matching `release_id` and terminal status `published` or `image_rejected`.
The producer uses those acknowledgments only for backpressure, not promotion.
Admission reserves 2 GiB within a 4 GiB outstanding-queue budget and pauses on
memory pressure or insufficient memory/disk capacity. No source data is deleted.

The separate consumer owns `render_physical_pairs.py` and its tests. Its agreed
interface is `--queue <producer>/render_queue --output <derivative-root> [--once] [--publish]`.
Derivatives are staged under `<derivative-root>/.staging/<source_release>/<attempt>`,
outside the immutable producer sources. The consumer verifies source hashes
before and after rendering, claims the queue with a queue-wide lock, and records
retry journals and downstream stage timings. Rendering starts with one idle CPU
thread and idle I/O, with its own memory and disk guards.

Do not publish staged derivatives into the browser or acknowledge `published`
until image qualification and an explicit derivative projection preserve source
identity, original labels, deduplication and protected family/split membership.
Browser integration is a downstream responsibility, not physical qualification.
With `--publish`, qualified derivatives produce
`<derivative-root>/publication_requests/<source_release_id>.json`. The consumer
waits for matching host browser evidence in
`<derivative-root>/browser_receipts/<source_release_id>.json` before acknowledging
the source as `published`. A rendered release alone does not clear producer
backpressure; pending or failed browser verification is not image rejection.
The same locked consumer publishes image-qualified pairs and services browser
acknowledgments at subsequent raster checkpoints and resource waits, rather than
waiting for the whole pair to finish.
Validated interrupted handoffs are rediscovered on restart; incomplete
trajectories are retained and never relabeled as successes.

Measured wall timings include proposal, construction/settling, paired execution
and physical validation, and handoff packaging. Each episode additionally
separates physics/contact-check time, trajectory capture time and numeric
validation time. These are nested measurements, not additive stages.
`timing_<sequence>.json` records the final packaging duration; the manifest
contains the preceding stage timings. Failed-stage elapsed times are explicit.
Existing image-producing collectors remain legacy interfaces, not this process.

`wrist-cad-v6` is an explicit recording profile restoring the original upstream
robot CAD jaw visuals, rather than the unmeasured tapered cosmetic sleeves in
v2/v3. It retains v3 camera, lighting and photometric settings, without cosmetic
sleeve blur. Contact geometry, dynamics and historical profiles remain unchanged.
The upstream CAD is not a hardware calibration; push-path/contact accuracy still
requires separate validation. Existing releases are never rerendered or relabeled
by selecting this profile for future recordings.

Future producer source checks actual `statvfs` available bytes on its output
filesystem. Admission/archive preparation requires an 8 GiB free-space floor
plus a 2 GiB operational pair allowance; these are conservative operating
budgets, not measured guarantees about unrelated writers. Every recorded step
checks the floor plus 1 MiB before either camera write or physics advancement.
Low space waits with the exact in-flight state retained in RAM and records
`disk_status.json` as `resource_paused`, not a physics rejection. Consolidation
requires twice the attempt's logical file bytes plus 64 MiB above the floor;
publication checks the actual copy bytes above it. No automatic data deletion
occurs. SIGTERM prevents new admission; an already paused in-flight pair still
needs storage recovery to finish gracefully. Killing that process loses its RAM,
so retain it rather than launching a replacement. This guard does not reserve
space against unrelated host writers.

## Priorities and progression

Use verified entries from `LEGO_PARTS_CATALOG.md` as they become available.
Prepare code now; do not generate with fake catalog parts or copy the old beams
under new IDs. Generate reproducible physically settled loose LEGO scenes:

1. Separated varied parts: different sizes, shapes, poses and commands.
2. Touching clusters: reduced jaw clearance and neighboring-object contacts.
3. Overlap and stacking: partial visibility, supporting objects and blocked
   approaches.
4. Dense mixed piles: target burial, extraction corridors and cluttered
   destinations.

These are sampling categories, not approval gates. Keep easy anchors while
expanding configurations. Describe complexity by part mix/count, packing,
pile height, visible target fraction, local neighbors, obstruction, approach
clearance and part/gripper size ratio. Count alone is not difficulty.
Separated proposals retain at least 4 cm center spacing for small parts rather
than scaling down to LEGO size alone, and center partially filled grids on the
scene. This distinguishes easy hand-clearance anchors from touching clusters.
Plan version v4 binds the changed proposal recipe; historical v3 plans stay
unchanged. The robot home is centered on the scene rather than 10 cm to its side.

Plan v5 additionally makes paired grasp/view target colors distinct and labels
category tags explicitly as proposal distributions, not measured settled
difficulty. Future episode `initial.settled_configuration` reports actual body
positions and rotated nominal-dimension envelopes, pairwise center distances,
horizontal envelope overlaps and vertical separations. These descriptive
envelopes are not exact concave geometry, collision clearance or occlusion
qualification. They allow audits to distinguish an overlap proposal that
settled apart from genuinely overlapping data without relabeling old episodes.

Plan v6 uses lateral destinations 10 cm either side of the scene center, replacing
the unreachable longitudinal endpoints observed in catalog push/place attempts.
Catalog push IK uses 0.2 mm position and 0.01 axis-error tolerances while legacy
defaults remain unchanged. Push preflights all four waypoints before recorded
motion; grasp/place heading selection also checks lift and placement waypoints.
Reachable IK does not establish contact-valid, successful paired supervision:
the existing physical and single-pass acceptance checks still apply.

Plan v7 samples distinct part IDs before repeating a cohort, retaining
family-balanced selection. The producer proposes 2 separated, 4 clustered,
6 overlapping, and 8 piled objects rather than two objects in every category.
New-source auxiliary admission requires a qualified eight-object pair; a
two-object success does not establish the memory cost of larger piles.
Imported/tabletop-verified parts still need actual robot-expert qualification.
Plan v8 cumulatively separates initial source-geometry vertical bounds for mixed
part heights. LDraw bodies are not necessarily centered within their dimensions:
assuming symmetric half-heights caused an eight-piece proposal to intersect at
time zero. Source bounds are used when present; older bounds-free catalogs keep
the explicitly approximate symmetric-dimension fallback. Actual contact-valid
settling remains required.
Plan v9 resamples clustered horizontal positions until conservative source-origin
bounding circles are disjoint with 0.1 mm clearance. This prevents verified
time-zero intersections in four-piece clusters; it does not establish touching
clusters, approach clearance, or contact-valid manipulation.

Settling exposes an opt-in `stationarity_metric="surface-speed-v2"` in
`settle_scene` and `settle_data`: its conservative surface-speed bound combines
linear speed with rotation and a geometry radius measured from the body COM,
in metres per second. Snapshots record the metric, units, tolerance, continuous
stationary interval and final maximum speed. The default remains
`legacy-component-v1`; CPU curriculum workers select their metric explicitly.
This changes stationarity measurement, not penetration checks or collision
geometry. Quiet convex-visual-hull demos do not qualify native catalog cavities
or resolve the producer's early-impact failures.
The new CPU-worker recipe explicitly selects `contact_solver="pyramidal-v1"`,
`stationarity_metric="surface-speed-v2"` and a 0.005 m/s surface-speed tolerance.
The recipe and settled snapshot bind these choices; archived XML declares the
pyramidal cone too. General simulation defaults remain unchanged. Activation uses
a fresh immutable source/root after actual paired recording/SFT verification;
earlier frozen roots retain their historical settings.

Catalog grasps now preflight the open fixed jaw against the stationary target
geometry. When needed, the TCP shifts laterally along measured contact normals
to clear the jaw, rather than widening the moving jaw and pushing the target
during descent. Alignment is bounded to 20 mm and fails explicitly when no
lateral clearance is possible. Physical descent, two-jaw contact and retained
lift still require independent runtime acceptance.

Continuous generation schedules only grasp, place and push, in equal proportions.
VIEW demonstrations are excluded because their intended behavior is unclear in
the data browser. This is scheduling priority, not a measured successful
data distribution; rejected manipulation attempts remain visible.

SFT is one command at a time: grasp, push, place and view. Generate meaningful
alternate target/destination commands from the same saved scene for paired
examples. Many physical objects do not require many command entities. Use
fixed LEGO physics and available versioned realism configurations; no arbitrary
mass/friction curricula, RL or multi-command training. Orientation-sensitive
placement is recorded in `CONTROL_MODEL_PLAN.md` as a future command extension.

## Scene-plan and artifact interface

`so101.lego_curriculum` prepares deterministic, policy-free scene plans from
verified catalog entries. Plans contain schema/version, sequence, seed, catalog
ID and SHA, category, task, object instance/part IDs, initial placement proposals,
and two command choices. Initial positions are proposals for physical settling,
not claims of collision-free scenes or accepted demonstrations.

The catalog-backed scene/expert adapter saves every dynamic object and velocity,
robot state, labels, camera/renderer identity and physics fingerprints. Namespace
part instances/assets and keep source part identity separate from instance ID.
Legacy four-object BeamSimulation resets are not valid for catalog scenes.

`so101.lego_scene.assemble_scene` now imports catalog MJCF templates into an
explicit-radian caller-provided base XML. It namespaces each instance's
body/joint/geometry/material/mesh identifiers, resolves only declared asset
paths, rechecks hashes, and rejects template global settings.

`settle_scene` compiles and simulates the proposal, rejects non-finite physics,
MuJoCo warnings and excessive interval penetration, and requires stationary
objects. It returns all-object metadata plus a full MuJoCo integration-state
snapshot bound to the XML hash and physics version. `restore_scene` validates
the snapshot before modifying the destination state, including warm-start and
control state. A runtime-model hash additionally binds mesh geometry, contact
settings, actuator dynamics, joint properties and solver options. Settling
timeouts/velocity/contact tolerances are physical
acceptance parameters, not daily generation quotas.

Corrected real catalog entries have also exercised tabletop settling and exact
next-step restoration. These scene diagnostics do not establish robot grasping,
paired SFT qualification, camera calibration or successful transfer.

`so101.catalog_sim.CatalogSimulation` inserts catalog objects into the SO101
model, initializes robot controls and settles the scene. It reuses the robot
kinematics, command labels and strict physical checks without the legacy object
registry/reset. Catalog grasp proposals use actual part dimensions/orientation
and request 0.2 mm / 0.01 rad IK precision, leaving legacy defaults unchanged,
and preflight both equivalent parallel-jaw headings before recording any action;
this is planning, not a failed-action retry. The robot table uses `solref=.001 1`,
`solimp=.99 .999 .0001`, priority 1; catalog part contact settings are preserved.
Acceptance still requires at most 1 mm general and 0.3 mm table penetration,
at least 5 cm target lift and ten consecutive two-jaw retention observations.
Tabletop catalog qualification with a different table/contact profile is not
silently claimed to be identical to robot-scene qualification.

Catalog view commands likewise preflight both original observation poses using
equivalent half-turn wrist headings before recording motion. Actual settled
orientation replaces proposed yaw in new initial metadata and recovery starts.
The formerly failed separated-view scene has passed both independent view
commands and all 119 paired SFT samples with this change; historical failures
remain unchanged. No observation-height, physics or acceptance threshold was
relaxed.

`so101.collect_catalog_sim` records a single scene pair with synchronized wrist
and external pre-action images, actions and entity labels. `complete` requires
both commands to pass independently from the same saved state; one successful
command is not a qualified pair. Source hashes, exact compiled `model.mjb`,
`model.xml`, scene plan and integration state are archived inside each episode,
using contained hardlinks within the new attempt directory. Source archives
also retain the exact hash-checked parts manifest, including LDraw attribution,
licenses, source dependencies and qualification limits. Release IDs bind the
scene, generator, renderer and exact scene archive hashes. Existing
`EpisodeStorage` can therefore retain all scene/source artifacts alongside
original JPEGs without parent-directory media addressing. Compiled replay needs
the recorded MuJoCo version and validated model/state hashes; legacy playback
must not interpret `simulation_backend=catalog-v1` as a legacy reset recipe.

Future accepted episode metadata records `generated_at` as a timezone-aware
UTC RFC3339 wall-clock timestamp, captured after clean recording validation and
frame-file completion, immediately before publishing `episode.json`. A complete
collection records its own `generated_at` after both independent commands pass,
immediately before publishing the completed catalog. Failed episodes and
incomplete collections have no generation-completion timestamp. These dates
are not simulated elapsed time, qualification time, or storage-copy time.
Immutable conversion preserves them byte-for-byte; historical undated episodes
remain undated. Seeds, scene/protocol versions and IDs are not recency clocks.

The one-pair CLI is not a continuous producer, an immutable release publisher,
an audit service or a claim of full SFT reader qualification. Those interfaces
remain separate implementation work. Example CPU qualification invocation:

```sh
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LP_NUM_THREADS=1 PYTHONPATH=src \
  python -m so101.collect_catalog_sim --catalog /catalog/catalog.json \
  --output /new/attempt --seed 20261108 --sequence 0 --category separated \
  --task grasp --object-count 2 --part-id 3005
```

Preparation CLI (prints a proposal only, does not run physics or write data):

```sh
PYTHONPATH=src python -m so101.lego_curriculum --catalog /catalog/catalog.json \
  --seed 1000000 --sequence 0 --category pile --task grasp --object-count 30
```

The loader checks every verified entry's declared asset hashes and refuses an
empty verified subset. Hash checking trusts the catalog worker to enumerate all
transitive assets and actually qualify geometry; it does not replace MuJoCo
qualification. The proposal binds the exact manifest hash, uses family-balanced
part sampling, and records alternate commands. The scene adapter must enforce
robot reachability, settling, actual visibility and feasibility.

Generated examples must retain stable example/scene/release IDs, paired command
identity, category/tags, split, catalog and renderer hashes, part/instance IDs,
acceptance version/evidence, attempts and recovery-parent provenance.
Assign new train/development/sealed-test membership by scene family before
generation; keep related commands, retries and descendants together. Preserve
the original frozen40/160 and exclude their descendants from new training.

## Quality and failure reuse

Use deterministic physical, identity, synchronized frame/action/label and
single-pass checks. Preserve initial and interval contact evidence, correct
selected-object lift/retention, destination/release and observation usefulness.
No failed prefix or recovered final success is relabeled as clean expert data.

Save failed attempts as honest diagnostics and possible recovery starts.
Recovery SFT uses the saved post-failure state, fresh observation/history and
only successful corrective action targets. Never allow target horizons or
previous-action context to import the failed prefix. Numerically invalid or
interpenetrating states are not automatically useful recovery examples.
`use_recovery_start` validates and restores an exact failure snapshot, preserves
the parent scene family, assigns distinct child identities and refreshes actual
object orientations. Each corrective recording starts with four new neutral-hold
observation frames. Parent rows/actions are never copied into the child.

For each consecutive ten accepted examples in each category, select one
randomly for this worker's rendered wrist/external temporal visual audit.
Persist counters, selected IDs, findings and audit status. Inspect transitions,
not only final poses. Systematic faults trigger generator corrections.
Host event/wake transport is delegated to the Work Scheduler; audit selection,
rendering and judgment remain curriculum responsibilities.
`visual_audit_due` deterministically selects one offset in each per-category
block of ten from the audit seed and block number; persisted accepted counters
allow resumption without selecting different examples.

`so101.curriculum_ledger.CurriculumLedger` persists explicit family membership
and accepted-example counters transactionally in a mutable coordination SQLite
database, separate from immutable episode storage. Membership is reserved before
simulation and cannot change on retries or restart; the caller must retain the
same family for descendants. Only a complete two-command pair can increment
category counters. Replaying the same pair is idempotent, while changed example
identities fail explicitly. Selected audits remain pending until this worker
records an explicit judgment and findings. The seed is fixed when the ledger is
created. This ledger does not invent split proportions, qualify a pair, render
audits, or automatically promote data. Producer integration and protected
historical-family exclusion remain required.

## Execution and publication

Continuously runnable low-priority CPU generation, using all idle CPUs without
artificial time/sample/storage quotas. Host cgroup/service integration is
separate Work; do not implement it here. Graceful interruption should commit
completed attempts and resume matching manifests; changed fingerprints produce
new releases. Missing verified parts is an explicit waiting state, not a reason
to spin or generate placeholder data. Real resource errors must be visible.

Publish new immutable catalogs and SQLite containers using existing storage,
with exact-byte/hash verification and full SFT pair qualification. Never append
to mounted immutable databases or assume the reader federates shards.
Keep original datasets, checkpoints, STOP markers, canonical paths and active
sweep inputs untouched. No model training, GPU or physical robot motion.

`python -m so101.produce_catalog --root /new/producer --catalog /pinned/catalog.json
--appearance /pinned/profile.json --part-id 3005 --seed 20261108` runs the first
resumable development worker. Run from an immutable source snapshot with CPU
renderer/BLAS threads set to one. It uses SCHED_IDLE/nice19, retains explicit
failure receipts, finishes the current attempt on SIGTERM, and resumes an
interrupted complete pair without simulating it again. Recipe changes require
a new root. The loop cycles configuration categories and command tasks; failures
are diagnostics, not accepted examples. All actual paired SFT samples and archive
bytes are checked before separately publishing an immutable release.

This initial worker uses two objects and one process. Pressure-aware multiworker
fan-out, broader validated part mixtures and audit wake/render transport remain
unfinished; it is not yet an all-idle-CPU resource manager. New families are
explicitly development-only, keeping feedback-visible attempts out of sealed
evaluation and training. It never imports historical episodes or assigns their
descendants to training. A future train/sealed producer must implement protected
historical-family exclusions and independent split policy before publication.

`so101.curriculum_workers` supervises multiple identical pinned development
workers with one shared ledger. Global scene sequences are reserved
transactionally before an attempt and retained in each worker's interruption
state; worker scheduling cannot duplicate a scene or reset category audit counts.
Each child uses SCHED_IDLE/nice19 and idle-class I/O, with renderer/BLAS threads
set to one by the runtime environment.

Fan-out is bounded by actual CPU affinity and available memory, not CPU quotas
or time windows. The controller reads host MemAvailable, cgroup memory.max/current
and inactive-file cache, reserves each worker's growth toward the observed peak,
and updates that peak from actual receipts. Inactive file cache is treated as
reclaimable, not resident model memory. New-source expansion waits for a matching
full paired recording/reader measurement; prior measured peaks remain
conservative startup estimates. Host/cgroup memory pressure prevents auxiliary
admission and retires auxiliary workers at attempt boundaries; it does not
churn the sole idle-priority worker when measured memory headroom is ample.
Insufficient reserved growth headroom can retire the primary worker as well.
This is predictive backpressure, not a guaranteed memory cap; unexpected worker
errors remain explicit. Source changes require a separate recorded rollout root.
Existing datasets and historical timestamps are not rewritten.

Example supervisor invocation (peak must come from the matching measured
two-object renderer/reader population, not part archive sizes):

```sh
python -m so101.curriculum_workers --root /new/supervisor \
  --catalog /pinned/catalog.json --appearance /pinned/profile.json \
  --part-id 3005 --seed 20261108 --initial-sequence 0 \
  --worker-peak-kib 1445932
```

The runtime requires `ionice`. Supervisor state, child logs and the shared audit
ledger are durable coordination artifacts; completed release archives remain
immutable and independent. Shutdown requests each child to finish its attempt.
Storage-pressure planning, validated broader worker populations, train/sealed
promotion policy and audit wake delivery are still separate work.
