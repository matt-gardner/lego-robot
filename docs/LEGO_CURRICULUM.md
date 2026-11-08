# Workstream 4: Continuous LEGO data curriculum

Approved by Matt November 8, 2026. This dedicated session owns curriculum,
producer, deterministic checks, recovery extraction and visual sanity audits.
Catalog, dashboard, realism and host integration have separate owners.

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
