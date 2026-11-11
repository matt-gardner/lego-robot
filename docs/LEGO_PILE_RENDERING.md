# Saved-pile appearance rendering

`so101.render_pile` rerenders an existing `model.xml` and
`settled-state.npz` without stepping, repositioning or freezing objects.
It is independent of the versioned wrist-camera profiles and does not
change their identities or producer defaults.

```bash
PYTHONPATH=src MUJOCO_GL=osmesa python -m so101.render_pile \
  --source /path/to/saved-pile \
  --output /path/to/new-render
```

Use the repository's isolated Linux simulation environment and installed
OSMesa libraries. Do not rebuild the host-mounted virtual environment.
The output directory must not already exist.

## Appearance profile

The default profile is `src/so101/renderer_profiles/pile_photo_v1.json`.
`--profile` selects an explicit alternative. Published profile/source bytes
must remain pinned; introduce a new version for later release changes.

The renderer fits the full visible pile to the requested frame occupancy,
replaces the checkerboard with deterministic fine surface grain, and adds
plastic specular highlights. Explicit corner normals smooth shallow mesh
faceting while retaining sharp molded edges; mesh vertices and faces are
unchanged. Stratified light samples approximate a broad light source.
Supersampling and slight final softness reduce aliasing without added noise.

## Outputs and checks

- `pile.png`: appearance image.
- `synchronized_labels.npz`: segmentation and metric depth from the same
  geometric camera at the output resolution.
- `receipt.json`: source model/state and profile/implementation hashes,
  physical model hash, actual visible bounding box and frame occupancy.

Compilation must preserve the original physical model fingerprint, and
rendering must preserve the saved qpos/qvel exactly. Clipped pile geometry
and nonfinite depth are errors. Contact geometry, masses, inertia, joints
and solver settings are not restyled.

The first input pile uses detailed catalog visual meshes but convex-hull
collision approximations. Its appearance rerender does not establish native
cavity, clutch or training-data qualification. Lighting is an OpenGL
area-source approximation, not path-traced global illumination. Better
appearance is not camera calibration or evidence of real-robot transfer.

## Fresh photo-style piles

`so101.photo_pile_generator` uses the retained 90-piece photo-demo XML as
immutable body/visual-hull templates, not as a saved-state initializer:

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m so101.photo_pile_generator \
  --source /path/to/b887f204-big-pile-compact-settled \
  --root /path/to/new-photo-pool --pieces 20 --seed 20261111
```

Only 20..90 pieces are supported. Each proposal samples fresh XY/yaw positions
within the original 70 mm-radius footprint and inserts the pieces using low-height
gravity drops. Pending pieces are parked and reset until their insertion.
Inserted bodies are never frozen or restored from an old scene. Original
gravity, visual-hull collisions and solver settings remain unchanged.
The `collective-low-drop-v1` recipe adds a piece every 0.15 simulated seconds
while the existing pile keeps moving. It does not demand stationarity after
each insertion. The final full scene must still meet the unchanged photo-demo
surface-speed limit of 0.5 mm/s continuously for 0.5 seconds.
The original 5 mm penetration cap applies throughout insertion and final
settling; actual maxima are recorded separately. Final settling allows
0.5 simulated seconds per piece. Total simulation work is bounded by 6 CPU
seconds per piece, with a separate 24 wall-seconds per piece safety deadline.
Thus larger piles receive proportionate budgets, and idle-priority scheduling
does not consume their CPU allowance while another process runs. STOP/resource
checks remain active during final settling. Receipts record the recipe,
drop cadence, budgets, and actual CPU/wall time; historical receipts are unchanged.

The producer runs at idle CPU priority, admits work only with 2 GiB available
memory and 10 GiB free disk, and pauses before exhausting a 4 GiB output budget.
It holds a single-writer lock. `STOP` in the output root or its parent, SIGTERM,
or SIGINT stops the producer without modifying completed scenes. An interrupted
unpublished proposal is retried with the same seed. Completed/rejected attempt
receipts advance the seed on restart. The root pins the source model, piece
count, initial seed and implementation hash; changed configurations require
a separate root. `--attempts N` bounds a run; the default continuously generates.

Each successful scene is staged and verified by exact model/full-state restore
and NPZ equality before publication. Failures are explicit rejection receipts,
not partially admitted scenes. Status is in `status.json`. Outputs remain
separate from native task pools, frozen datasets and training acceptance.
The source's historical `catalog-demo` labels are retained, not invented part
identities. Saved models/states are renderable by the appearance command above;
generation itself does not produce a photograph or establish RL/SFT acceptance.

### Robot tasks and capped progression

`python -m so101.photo_pile_tasks --source-root <saved-photo-pool> --root
<new-task-pool> --pieces 20 --scenes 10` selects ten distinct saved model
identities and integrates them with the SO101 robot and visible destination
markers. Original object poses are translated onto the robot workspace table,
then the full scene is dynamically settled with the photo-demo physics.
Sources that fail this additional settling are recorded and skipped. Original
pile files are never modified. The new receipts bind source hashes and the
translation explicitly.

Each admitted scene has four tasks per target (view, grasp, push, place),
persistent body/geometry identities, the existing task predicates, and an exact
robot reset. New task revisions use `independent-robot-starts-v1`: each task
samples its own seeded, joint-limit/workspace/collision-checked robot
configuration, with tool positions at least 5 mm apart within the scene.
The settled pile is preserved, not re-dropped per task. Task-specific full
integration snapshots are embedded in the SHA-bound `problem.json`; resetting
the same task replays exactly the same start across assistance levels.
These are randomized starts, not intentionally constructed obstacle challenges
or verified camera-visible starts.
Published pools use the existing `pile_pool`/`rl_pile_adapter`
contract, with the photo collision profile explicit in scene tags/provenance.
Task solvability and initial visibility remain unmeasured; emitting task goals
is not demonstrating successful policy execution. These pools have no expert
references and cannot support nonzero gold-prefix assistance until genuine
successful references are generated and the trainer's replay contract supports
them. They are not paired SFT recordings.

`python -m so101.photo_pile_tasks --retask-pool <immutable-pool.json> --root
<new-task-root>` revises existing tasks without generating more piles or
modifying the source pool. Models, pile states and family IDs are retained;
scene/task IDs and problem hashes change, invalidating old expert references.
Renderers must group by actual task reset state, not only scene identity.

After verifying all replacement pools, `--select-pools <pool.json> ... --root
<curriculum-root>` atomically writes `current-task-pools.json` with the selected
manifest paths, hashes and populations. Dashboard readers use this explicit
selection, not modification times or recursive discovery of historical subsets.
Pending previews must not reuse images from superseded task/reset identities.

After handing the ten-scene 20-piece pool to the trainer, use
`python -m so101.photo_pile_curriculum --source <original-photo-demo> --root
<new-curriculum-root>` for **30, 40, 60 and 90 pieces**, in that order. Each
stage stops accepting after ten robot-integrated task scenes; it does not
replenish the 20-piece bucket. Piece count is a structural difficulty
progression, not a measured policy-success ordering. Trainer probing supplies
the latter. Each stage has at most 100 proposals; failure to reach quota
publishes the partial pool and an explicit blocked status rather than silently
generating forever or skipping to another size. `pieces-N/handoff.json` binds
each stage's immutable manifest/checksum, count and quota. The same STOP,
idle-priority and resource limits apply. New task roots stay separate from
frozen measurement pools and historical datasets.

For Matt's simple/common-piece preference, add `--sampling common-basic-v1`.
It assigns 90% of each pile (rounded) to ordinary bricks/plates
3001, 3002, 3003, 3004, 3020 and 3022, with the remainder split between
Technic bricks 3700 and 3702. Selection within each group is seeded and allows
repeats; body identities remain unique. This is an explicit basic-piece prior,
not measured sales/set-frequency popularity. The mapping is restricted to the
exact retained photo-model hash and was checked against catalog visual vertices.
Part identities and sampling profile are recorded on new scenes; existing
scene/task receipts are not rewritten.

Larger-stage scenes cycle through single-color, two-color and six-color
palettes by **accepted scene count**, so rejection streaks cannot skip a color
stratum. Each piece remains a separate target for all four task types (for
example, 30 pieces yield 120 tasks). Task descriptors include height rank and
higher overlapping world-AABB candidates to distinguish top/buried candidates.
These are geometric proxies, not measured wrist-camera visibility or proof
that a buried piece is graspable without clearing other pieces. Trainer rollouts
must establish actual solvability/difficulty.

`--prior-root <previous-progression-root>` carries previously admitted task
scenes into a new versioned progression without regenerating them. Their
identities and source data remain unchanged and they count toward the same
ten-scene quota; prior roots are never modified.

## Optional CPU path tracing

`so101.pathtrace_pile` consumes a pinned `scene.json` world-space PLY
export, not arbitrary MuJoCo XML. Install `requirements/pathtracing.txt`
in a separate isolated environment; LLVM must be available to Mitsuba. It uses CPU
`llvm_ad_rgb`, four threads, indirect illumination, a broad area source
and rough plastic. It does not replace the raster or wrist renderer.

```bash
PYTHONPATH=src python -m so101.pathtrace_pile \
  --scene /path/to/export/scene.json --output /path/to/new-render
```

The manifest contains `objects` with `mesh` and normalized sRGB `rgb`,
`table_z_m`, and `camera` with `origin`, `target`, `up`, `fovy`.
Relative mesh paths resolve beside the manifest.
For an explicitly relocated export, `--mesh-root` selects the directory
containing its mesh basenames; resolved meshes are hashed in the receipt.
Export world-space
visual vertices/normals without applying MuJoCo mesh centering twice.
For mono camera matching, average both MuJoCo eye transforms, not just
the left eye. Use the source plane's actual height to avoid hovering.

The pinned profile is `src/so101/renderer_profiles/pile_pathtrace_v1.json`.
The output receipt records mesh, manifest, implementation, profile and
image hashes plus backend/settings. This appearance-only backend does
not produce synchronized labels/depth or simulate physics. Existing
saved-state and collision qualification must accompany the export;
photorealism does not upgrade it. The plastic response is an appearance
assumption, not measured ABS. True molded silhouette bevels, measured
table material and calibrated optics remain absent.

On the saved 90-piece example, a coplanar 80-micrometer normal-bevel
experiment did not visibly improve the molded edges, so it was not
adopted. A 1-millimeter aperture-radius thin-lens probe added foreground
blur; the default retains the inspectable pinhole view rather than
assuming unmeasured optics. Further substantial gains require actual
rounded visual geometry and material/optical measurements, not heavier
blur or arbitrary noise.

## Exact INITIAL external pile cards

`python -m so101.photoreal_task_images --pool <complete-TRAIN-pool>
--output <versioned-durable-pool-directory>` restores each robot-integrated
task-specific `initial_state` (or legacy `integration_state.json`) through
`pile_task_runtime`; it never substitutes the original pre-robot settled pile.
Images are shared only when the actual full integration-state hashes match,
not merely because task descriptors belong to the same scene.
It exports compiled visual vertices and normals
once into binary world-space PLYs, groups them by body/material, and uses the pinned
`pile_pathtrace_v1` CPU recipe. MuJoCo segmentation is captured separately at the
same resolution and mono camera (average of both eye transforms). Sampled
pixel-center Mitsuba primary rays must agree with the segmentation within 1%;
the actual sampled count and mismatch are recorded, not a calibration claim.
Unsupported geometry/materials and task-dependent reset differences fail explicitly.

The output root contains only content-hashed PNGs and per-task image receipts.
`_scenes/<scene-id>/<integration-state-sha>/` retains world exports, segmentation, diagnostic raster RGB and pathtrace
receipts; `_meta/` pins implementation/profile/pool hashes and per-piece-count
visible/not-visible coverage. The dashboard's live-file reader is the ongoing consumer.
Use a new versioned output for changed implementation/profile bytes. A partial
population is not full image coverage or policy qualification. Existing wrist
images remain separate; no historical image recovery is performed.

`python -m so101.task_image_roles --images <captured-pool> --output <roles-v1>`
adds independently measured `source_object` and push/place `destination` roles
without changing RGB. Destination boxes describe visible goal-marker geometry,
not an invented second LEGO object, tolerance region or amodal extent.
The dashboard reads complete atomic receipts and assets directly from the durable
pool directory, preferring enriched `_roles-v1` receipts per task. It validates
the exact pool, problem/source, target and image hashes on read. No image
publication callback or copied measurement manifest is required, and a metrics
refresh cannot overwrite these independent image derivatives.
New captures include both roles in their first complete atomic receipt. Binary
exports hard-link identical content-addressed world meshes in `_meshes/`, so
unchanged pile geometry is not stored again for every robot starting pose.
The storage-corrected full backfill uses `initial-photo-external-v2/<pool-sha>/`;
earlier v1 proofs remain intact. Appearance settings are unchanged.

## Archived curriculum wrist rerenders

`python -m so101.pathtrace_recordings --root <curriculum-root> --since
2026-11-08T08:00:00+00:00 --output <new-derived-root>` creates separate CPU
appearance derivatives of qualified archives. `--inventory-only` measures the
population without rendering. Eligibility requires both paired commands to
have recorded generation dates after Pacific midnight; unknown dates are
reported as exclusions. Attempt/release copies and format migrations are
deduplicated by release identity, not directory timestamps.

The backend reads immutable `EpisodeStorage` containers, including recordings
whose original directories no longer exist. Each image uses its complete saved
`qpos` and the recorded wrist-camera geometry, without physics stepping, robot
translation, trajectory replay or scene resettling. Original CAD jaw visuals
replace cosmetic sleeves; all hidden collision-display groups stay hidden.
Lighting and rough-plastic parameters come from the accepted pathtrace profile,
at the original 320 by 240 recording resolution and 1024 samples per pixel.
Command boxes, visibility and camera depth are recomputed with the same visual
geometry and camera using the existing label helpers.

Each new container retains original model/state, metadata, frames, external
images, splits and historical acceptance provenance. `render_model.xml` and
`render_model.mjb` identify actual new visual geometry;
`appearance_rerender` identifies actual image production separately from the
historical `initial.renderer`. These are appearance derivatives, not newly
recorded or training-promoted episodes. External views are retained original
images, not claimed rerenders. Consumers must use the derivative image/label
identity rather than interpreting the historical model as the new visual model.
No historical acceptance defect is corrected by changing appearance.

`inventory.json`, per-release `progress.json`, image/label checksums and
`completion.json` distinguish queued, partial and verified completed work.
Rerunning the exact same implementation/profile checks completed outputs and
restarts an incomplete episode from its saved poses; it never changes originals.
Mitsuba and the normal MuJoCo/OpenCV runtime are optional rendering dependencies;
use an isolated environment, not a host-mounted virtual environment.

An already-running renderer loads its inventory once; changing that file does
not remove tasks from the active process. For an explicitly requested task
exclusion, stop the owned worker without deleting partial outputs, then run
`so101.pathtrace_queue --output <derived-root> --exclude-task view` against the
same frozen renderer package/profile. The dispatcher preserves the original
inventory, writes `selected_inventory.json`, verifies completed containers and
resumes incomplete non-excluded releases. It rejects mixed-task pairs and any
existing excluded derivatives rather than deleting them. Completion counts
refer to the selected inventory, not the superseded original population.
### Pile-first CPU scheduling

`scripts/render_pile_first.py --root <initial-photo-external-v2> --data
<lego_curriculum>` checks the current immutable pools and atomic
`photo-task-progression-collective-v1/pieces-*/tasks/scene_*/admission.json`
records before every frame. Every admitted pile gets one representative
exact-task-reset image before any additional task images. Newly arriving piles
preempt task expansion at the next frame boundary.

Existing selected-pool receipts remain under their pool SHA. Admissions from
unfinished stages publish separately under `pile-previews-v1/<admissionSHA>`;
these are pile previews, not completed-stage or selected-task coverage.
The queue uses the existing exclusive worker lock and a fresh CPU subprocess
per image, preserving unpublished interrupted exports rather than deleting them.
