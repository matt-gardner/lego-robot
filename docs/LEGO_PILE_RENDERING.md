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
within the original 70 mm footprint and inserts the pieces using low-height
gravity drops. Pending pieces are parked and reset until their insertion.
Inserted bodies are never frozen or restored from an old scene. Original
gravity, visual-hull collisions and solver settings remain unchanged.
Every insertion and the final full scene must meet the photo-demo
surface-speed limit of 0.5 mm/s continuously for 0.5 seconds and the original
5 mm penetration cap; actual maximum insertion and final-settle penetration
are recorded separately. Each insertion has a 10-second simulated timeout;
the insertion sequence also has a 120-second wall-time budget.

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
robot reset. Published pools use the existing `pile_pool`/`rl_pile_adapter`
contract, with the photo collision profile explicit in scene tags/provenance.
Task solvability and initial visibility remain unmeasured; emitting task goals
is not demonstrating successful policy execution. These pools have no expert
references and cannot support nonzero gold-prefix assistance until genuine
successful references are generated and the trainer's replay contract supports
them. They are not paired SFT recordings.

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
