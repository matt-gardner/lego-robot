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

Only a future scene/expert adapter can turn a plan into demonstrations. Its
saved settled state must include every dynamic object and velocity, robot
state, labels, camera/renderer identity and physics fingerprints. Namespace
part instances/assets and keep source part identity separate from instance ID.
The current four-object BeamSimulation is not an arbitrary-pile adapter.

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

For each consecutive ten accepted examples in each category, select one
randomly for this worker's rendered wrist/external temporal visual audit.
Persist counters, selected IDs, findings and audit status. Inspect transitions,
not only final poses. Systematic faults trigger generator corrections.
Host event/wake transport is delegated to the Work Scheduler; audit selection,
rendering and judgment remain curriculum responsibilities.
`visual_audit_due` deterministically selects one offset in each per-category
block of ten from the audit seed and block number; persisted accepted counters
allow resumption without selecting different examples.

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
