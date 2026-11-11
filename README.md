# SO-101 LeRobot workspace

New expert recordings use `initial.expert_acceptance_version=1`: one grasp
attempt and one push pass, with final task success still required. Failed
attempts are rejected as complete episodes, never trimmed or relabeled as clean
experts. SFT/RL loaders reject retry phases in versioned clean recordings.
Historical unversioned recordings remain unchanged and retain their original
acceptance semantics. Corrected replacements must use a new output dataset
(for example `physical_preview_v3_clean_v1`), preserving source provenance;
older preview recordings are not silently rewritten.

Completed `data/simulation/physical_preview*/collection.json` datasets appear
as supplemental dashboard examples. Search for `physical_preview_v3` to inspect
the collision-correct preview; these examples are not marked as used by historical
training runs. Incomplete previews are not shown.

Both the training Data browser's source examples and Recorded examples sort by
recorded `generated_at` completion timestamps, newest first, with stable identity
ties before pagination. Recorded examples also select the newest dated episode
release, not the most recently copied file. Undated legacy data follows dated
data and is labeled "Generation date not recorded"; its chronology is unknown.
`generated_at` is a timezone-qualified RFC3339 wall-clock timestamp captured by
the producer at recording completion (collection completion for release metadata).
Copies and storage conversions preserve it; qualification dates, seeds, protocol
versions and file modification times are never substituted. Policy rollout and
solved-entry ordering, part catalog selection and provenance are unchanged.

Python 3.12, LeRobot 0.6.1, and the `core_scripts` and `feetech` extras.
Includes calibration, motor setup, teleoperation, recording, replay, and
camera discovery. Training and policy-specific extras are not installed.

Sources:
- https://huggingface.co/docs/lerobot/en/installation
- https://huggingface.co/docs/lerobot/en/so101

## Project layout

The AWS checkout is `/mnt/dockyard/lego-robot`. Persistent datasets, experiment
outputs, and Hugging Face caches live separately under
`/mnt/dockyard/lego-robot-data/{data,outputs,hf}`. The checkout's ignored
`data` and `outputs` symlinks preserve existing relative-path commands.
Set `HF_HOME=/mnt/dockyard/lego-robot-data/hf` when using the migrated cache.
Neither datasets nor model/checkpoint files belong in Git.

The read-only dashboard is served at
`https://server.mgardner.org/lego-robot/`, behind Cloudflare Access. Its
host-managed `so101-web.service` uses this checkout and the persistent
`outputs` directory. Moving code does not authorize training or clearing STOP.
For scoped container restart and GPU job controls, see
[Container control](docs/CONTAINER_CONTROL.md); GPU infrastructure qualification
and launch authorization are separate from implementing the broker.

Movie playback needs MuJoCo (the `simulation` extra), OpenCV, NumPy, an
`ffmpeg` executable, and a headless renderer in the **dashboard's own**
Python environment, not a shared host/training environment. The host
dashboard uses CPU OSMesa (`MUJOCO_GL=osmesa`, `PYOPENGL_PLATFORM=osmesa`).
To avoid XML compilation's memory peak inside the bounded service, prepare
its full, unmodified model outside the service:

```sh
PYTHONPATH=/mnt/dockyard/lego-robot/src uv run --no-project \
  --python /mnt/dockyard/so101-web/venv/bin/python \
  python -m so101.simulation_playback \
  --model-root /mnt/dockyard/lego-robot-data/data/simulation/model \
  --model-cache /mnt/dockyard/so101-web/playback-runtime/so101.mjb \
  --prepare-model-cache
```

Set the service's `SO101_PLAYBACK_MODEL` to that `.mjb` path. Playback checks
the simulator source, generated scene, model files, MuJoCo version, and
binary checksum before loading; rebuild explicitly when a check fails.
Without this setting, existing XML-based simulation/playback is unchanged.
This cache is dashboard-only and does not load a policy or authorize training.

New simulations use scene version 3: Lego props are movable colliding bodies,
studs have contacts without adding duplicate shell mass, and initial solid
placements are checked before and after settling (maximum permitted penetration
0.2 mm). Destination regions are flat, nonphysical annotations, not solid
obstacles. Invalid initial scenes fail explicitly rather than recording an
interpenetrating start. New episode initials retain the scene version and full
joint positions/velocities and actuator targets; recorded frames retain full
joint positions.
Version 3 additionally uses 0.5 ms physics steps and 1 ms contact time constants
to prevent the old soft table contacts from allowing millimeters of penetration
when grasping or releasing a beam. Version 2 remains supported for its recordings.

Untagged historical episodes use version 1 geometry and initialization for
playback and evaluation. Their original wrist JPEGs, actions, labels and metrics
are not rewritten. Legacy episodes without full joint positions show a **legacy
action reconstruction**, not corrected physics or original overview footage.
Changing future physics cannot repair those historical recordings. The existing
cache command compiles version 1; additionally run it with `--scene-version 3`
to compile the sibling `so101-v3.mjb` for new episodes (`--scene-version 2`
compiles the retained version 2 model). Relevant caches must be
rebuilt after simulator source changes before restarting the dashboard.
Scene updates alone do not regenerate data or train a model.

### Corrected training and RL data

Regenerate expert scenarios with the CPU-only, resumable data rebuild:

```sh
uv run --locked --extra simulation python -m so101.rebuild_physical_dataset \
  --source-collection data/simulation/entity_commands_with_difficulty_v1 \
  --output data/simulation/entity_commands_physical_v3 \
  --validation-split outputs/entity_smolvla_trajectory_rl_all_assisted_lr1e4_before_step243_rerun/data_split.json \
  --publish-catalog
```

The rebuild generates fresh images, labels, actions and full physics snapshots;
it does not repair or reuse old actions as ground truth. Task/object/destination
identities and labels are retained, but invalid starts are repaired or replaced
with seeded physical starts. Historical policy recovery prefixes are provenance,
not rerun-policy claims. Each physics substep is checked: at most 1 mm penetration
for any colliding pair and 0.3 mm for table contacts. Failed attempts are reported
and never included as successes. Physics/source fingerprint changes prevent
mixing incompatible resumed data.

Only a fully generated, verified replacement publishes the canonical catalog
used by existing `--collection` paths; a partial rebuild leaves it unchanged.
Original episodes remain untouched and their catalog is retained under
`entity_commands_physical_v3/historical/collection.json`. The mapped frozen split
is `entity_commands_physical_v3/data_split.json`; use it with new SFT/RL runs,
not the historical split or stopped checkpoints. Old metrics and matched-sample
sweep results remain historical and are not comparable to the corrected data.
No training, evaluation policy, or GPU is launched by this command.

Cold movie preparation returns JSON HTTP 202 promptly while one bounded
background renderer works; the browser polls until JSON HTTP 200 reports
ready. Other recordings wait without adding renderer jobs. This avoids the
public gateway's 60-second request timeout without changing its limits or
access rules. Rendering still has a 180-second limit; cached playback is
immediate. Non-JSON gateway/sign-in errors are displayed without parsing HTML.
Opening an example starts preparation immediately, without waiting for Play.
While it prepares, the preview shows the saved source overview snapshot
alongside recorded wrist frames, explicitly labeled as a static reference.
When ready, the synchronized dual-view video loads paused and supports
scrubbing both views; Play does not start a second renderer. Before it is
ready, Play animates the saved wrist frames immediately at their recorded
timing. The combined video takes over at the selected frame, preserving
playing/paused state. Closing or switching examples stops wrist playback.
The host's
dashboard CPU budget is 150% (1.5 CPUs), with the existing single-renderer
and memory/access limits preserved.

Application code is an installable package under `src/so101/`; tests are in
`tests/`. Run commands from the project directory with
`uv run --locked python -m so101.<module>` and run tests with
`uv run --locked pytest -q`. Root-level Python scripts are no longer used.

| Location / modules | Purpose |
| --- | --- |
| `src/so101/robot_control.py`, `robot_chat.py`, `describe_scene.py`, `robot_teacher.py` | Bounded hardware control, Copilot observation/chat, and teacher loop |
| `record_demo.py`, `robot_demo.py`, `real_policy_runner.py`, `target_perception.py` (inside `src/so101/`) | Real demonstrations, dataset export, learned-policy execution, and visual grounding |
| `robot_sim.py`, `collect_*_sim.py`, `build_command_benchmark.py`, `relabel_oriented_boxes.py` | Simulator, expert-data collection, benchmarks, and label conversion |
| `train_*`, `evaluate_command_sim.py`, `evaluate_entity_sim.py`, `jerk_reference.py` | Supervised/RL training, evaluation, and reference-margin calibration |
| `src/so101/control_model/` | Shared command schemas, datasets, SmolVLA architecture, and gradient utilities |
| `src/so101/analysis/`, `src/so101/experiments/` | Offline metric audits/plots and isolated jerk/checkpoint trials; not automatic production training |
| `rl_dashboard.py`, `training_data_browser.py`, `simulation_playback.py`, `rollout_evidence.py`, `visualize_data.py`; `src/so101/dashboard/` | Monitoring, dataset browsing, movie rendering, evidence capture, and packaged web assets |
| `cloud_training.py`, `cloud_worker.py`, `cloud_checkpoint.py`, `cloud_requirements.py`; `cloud/` | Cloud controller, worker, durable backups, dependency conversion, and Azure configuration/templates |
| `scripts/` | Shell entry points for teleoperation and cloud machine/disk setup |
| `tests/` | Automated regression tests, separate from application code |
| `docs/`, `docs/archive/` | Project/control-model plans, Azure/AWS setup checklists and operations guides, and historical proposal |
| `data/`, `outputs/` | Demonstrations, simulator assets, trained checkpoints, logs, and retained recordings; not disposable clutter |

The old learning-rate sweep was removed: it called deleted supervised-loss
options and assumed obsolete checkpoint/log formats. Useful analysis utilities
remain in the package. Plans are design documents, not proof of implemented
behavior; the original proposal is retained only as history.

## Install and run

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) on the
computer connected to the robot, then run from this directory:

```sh
uv sync --locked
uv run lerobot-calibrate --help
uv run lerobot-find-port
```

No environment activation is necessary with `uv run`.

Move or copy this entire project directory, including `uv.lock`, to its final
location and run `uv sync --locked` there. Do not move a virtual environment
between locations or operating systems; recreate it from the lockfile instead.
In Dockyard, uv keeps the Linux environment outside `/mnt/data`, in
container-local storage, to avoid overwriting a host macOS environment.

Linux uses CPU PyTorch intentionally: motor control and recording do not
require CUDA. macOS uses the standard PyTorch wheel, with MPS when available.
GPU policy training needs a separate CUDA-appropriate setup.

For Azure Spot training with local control/dashboard, retained storage and
checkpoint resumption, follow [CLOUD_TRAINING_SETUP.md](docs/CLOUD_TRAINING_SETUP.md).
It is the short account/billing/quota checklist; detailed commands are in
[CLOUD_TRAINING_OPERATIONS.md](docs/CLOUD_TRAINING_OPERATIONS.md).
Paid actions require `--apply`; nothing is provisioned automatically. The fresh
Ubuntu GPU bootstrap is `scripts/setup_cloud_gpu.sh`. Actual GPU/deployment and
eviction qualification are still required. Architecture and limits are in
[CLOUD_TRAINING_PLAN.md](docs/CLOUD_TRAINING_PLAN.md).

For AWS Spot GPUs, start with [AWS_SPOT_SETUP.md](docs/AWS_SPOT_SETUP.md):
the short manual checklist includes copy-paste Copilot prompts for preparation,
approved deployment, qualification, training, and cleanup. Explanations,
Oregon/eastern-region tradeoffs, and persistent-storage/network costs are in
[AWS_SPOT_OPERATIONS.md](docs/AWS_SPOT_OPERATIONS.md). AWS orchestration still
needs implementation and real qualification; the existing controller is Azure-only.

## Live RL / SFT dashboard

```sh
uv run --locked python -m so101.rl_dashboard --port 8765
```

Open **http://127.0.0.1:8765**. The read-only dashboard discovers trajectory
(`trajectory_rl.json`), older chunk-level (`rl_training.json`) RL runs, and
supervised (`sft_training.json`, `kind: "sft"`) runs
under `outputs/`, including new runs created while the server is running.
Use `--root /path/to/outputs` for another run directory. It binds to localhost
by default; it has no authentication and should not be exposed publicly.
No extra dependencies, external scripts, or model loading are needed.

Behind an authenticated reverse proxy, use `--base-path /lego-robot` and preserve that
prefix when proxying to the loopback server. Open `/lego-robot/` for the dashboard
or `/lego-robot/data` for the integrated data browser; assets and API requests stay
under the prefix. The default empty base path preserves direct-root serving.
`/lego` (or `/lego-robot/lego`) separately browses schema-1 LEGO catalogs and
existing immutable `.sqlite` episode containers under the workspace's `data`
directory without extraction. Set `--lego-catalog-root /mnt/data/lego-catalog`
to read externally published catalog releases. Hidden staging paths are excluded
from discovery until publication. Exact artifact and catalog hashes,
part preparation failures, historical acceptance labels and original recorded
frames remain visible; missing external recordings are not reconstructed.
The parts page automatically uses the latest usable catalog, with dropdowns for
training readiness, part type, footprint and decoration. Training readiness means
prepared for simulation-data generation, not robot graspability. Footprints are
parsed from explicit dimension pairs in part descriptions, not inferred from
measured bounding boxes; unlabelled parts remain "not specified". Historical
snapshots and qualification labels remain accessible through the read-only API
and technical part details, not browsing controls.
`/examples` provides a separate original-recording browser. Use
`--lego-episode-root /path/to/lego_curriculum` to discover externally published
immutable SQLite recordings without extracting logical source paths. A release's
`qualification.json`, when present, must match the container byte hash and is
shown without upgrading development diagnostics to training data. Wrist/external
frames play as a synchronized recorded-image sequence; no legacy simulation
reset or compiled-model download is used.
Cards show hash-verified visual-mesh thumbnails. Clicking a
part opens a drag-to-rotate, scroll-to-zoom viewer (arrow keys also rotate).
Previews use embedded visual MJCF meshes, not collision hulls, and need no GPU,
external scripts or new runtime dependency. Unsupported or unprepared geometry
is explicitly unavailable. Print/sticker variants use producer-declared
`base_part_id` relationships; numbering-based declarations are not claims of
qualified geometric equivalence or decorated simulation readiness.
Part/example details offer comments and “looks wrong” feedback when the same-origin
Family feedback capabilities endpoint is enabled and authenticated. Website login
alone is not a Family identity. Submissions pin the immutable release and artifact
hash; example comments can pin the selected original camera, frame and timestamp.
The browser retains the exact payload and UUID in local storage before sending;
after interrupted delivery, check the original receipt or retry that same payload.
The Data created filter matches generated supervised targets by joining archived
episode target identity to scene-plan part IDs, not every part present in a scene.
It matches part IDs across catalog versions and includes development examples;
training eligibility/promotion and actual training use are not inferred. Archives
without catalog provenance or a target-to-part join do not establish coverage.
Owner and monitor receipts/errors are separate. Queued means durable intake;
delivered means owner enqueue, not acknowledgment, resolution or execution approval.
Starting another comment explicitly releases the browser's saved payload; server
receipts remain sender-private. Storage/authentication failures are shown rather
than silently accepting feedback.
Producer-authored immutable `notices/*.json` (schema version 1) annotate an exact
`catalog_id` and `catalog_sha256` with `invalidated` or `superseded`, a unique
`notice_id`, `reason`, `source_work_item`, and an `evidence` object. The browser
keeps `status`/`stored_status` historical and exposes `effective_status` and
`qualification_status` separately; an invalidation takes precedence over
supersession. No matching notice means `not_invalidated`, not a new qualification
or robot-competence claim. Notice byte hashes and reasons appear in release details.
Release discovery discards each catalog's full payload before reading the next;
unchanged artifact summaries are cached after fingerprint verification, while
qualification notices and receipt checks remain fresh on each request.
item browsing loads only the selected catalog. Catalog API requests are serialized
to avoid multiplying the full-catalog working set within the dashboard service.
For example:

```sh
uv run --locked python -m so101.rl_dashboard --root /path/to/outputs --port 8765 --base-path /lego-robot
```

`--notice "Awaiting verified migration; no results loaded"` can label a deliberately
empty staging root while data transfers finish. Do not point at incomplete archive
parts or claim an empty staging dashboard contains the migrated results.

The **RL** and **SFT** tabs filter runs and charts independently. RL selections
and plot behavior are preserved when switching tabs. SFT defaults to an active
run, or the latest discovered SFT run, even before its first validation.
SFT optimizer steps show **total loss**, logged action/box components,
learning rate, gradient norm and step timing; training does not invent rollout
success, completion or jerk. SFT evaluation shows the same unassisted success,
success-conditional completion efficiency, **absolute RMS jerk** and wrong-object
metrics as RL validation, overall and in fixed grasp/view/push/place sections.
Both tabs retain smoothing, shared pan/reset, page scrolling and the timing strip.
SFT recent rows explicitly distinguish loss-only training from the two latest
validation rows. `python -m so101.train_entity_sim` and the exact corresponding
Python script are recognized as active SFT trainers; shell/wrapper mentions are
not. The API exposes `kind: "rl" | "sft"` and scalar plot points, not model or
per-frame payloads.

Select multiple runs to overlay plots; enable archived experiments to compare
saved `attempt*.json` branches. RL plots are grouped into independently collapsible
**Eval metrics**, **Overall training performance**, separate **grasp/view/push/place
training performance**, and **Training health** sections. Open/collapsed choices
are remembered in this browser across full refreshes, along with the active RL/SFT
tab, smoothing slider, archive toggle, search, independent run selections (including
empty selections), and each tab's focused run and chart view. Updates resume live
after a refresh. Eval, overall training and health start open; task sections start
collapsed and only render charts when expanded.
Each training-performance section includes success, completion efficiency, jerk,
wrong-object rate, reward and assistance (including discarded and solved groups).
Training jerk plots subtract **each problem's own expert-reference jerk** from
each rollout's RMS commanded jerk, then average those differences, with a dashed
zero line. Reference commands use the same assistance prefix as that rollout.
There is no pooling across problems or command-level percentile. Historical
plots recompute the matching problem references without rewriting historical
losses. Missing references remain gaps; at least four commands are required,
including failures. Validation retains explicitly labeled absolute RMS jerk.
The old phase/task selectors are replaced by the sections: eval, overall training
and health stay run-level, and each category stays pinned to its task.
Hover for exact raw values and use the smoothing slider (faint lines retain raw
data). Run comparisons, smoothing and shared pan/reset work across sections.
The gradient-norm plot starts at the latest logged optimization change, so
old-scale values do not dwarf calibrated updates. Raw history and other metrics
remain intact, and new gradient spikes are not filtered.
The training step-time plot shows duration in seconds (excluding validation and
checkpoint writes). The top-of-page timing strip shows the last step duration
for selected runs and a live "Since last step" counter only for runs with a live
local trainer process or a fresh cloud
worker heartbeat/contact and updates every
second, including while polling is paused (activity is last observed until polling
resumes). Archived runs never show a counter. New trainer
steps record completion timestamps; older runs use a log-write-time estimate,
marked `~`. Rewriting the same step does not reset that estimate while the
dashboard server remains running.
Cloud history remains available locally during an eviction; worker state, contact
age and checkpoint backup step appear beside the run. `cloud_training.py watch`
maintains the mirror without exposing an unauthenticated dashboard on the VM.
Direct GPU runs use `scripts/sync_gpu_dashboard.py`; trajectory logs are
recognized by their RL estimator, without requiring a particular run name.
Each successful refresh updates both the run heartbeat and mirror status.
For new direct GPU runs, launch the prepared remote Docker command through
`scripts/sync_gpu_dashboard.py --launch-command /remote/trainer-command.json`
with the run's `--host`, `--known-hosts`, `--instance-id`, `--job-id`,
`--container-name`, `--remote-root` and local `--output`. This is the standard
direct-GPU launch path: it automatically starts persistent 20-second dashboard
mirroring, without a separate `--watch` command. Run this launcher as the
persistent background owner, not a one-shot shell. It verifies the named trainer
is running, reuses only an exact matching active command, refuses to overwrite a
stopped/different trainer, and holds the mirror lock before launching. The mirror
exits after completion and surfaces premature trainer exit as a failure.
Scrolling over a plot scrolls the page,
without changing its axes. Drag to pan, and double-click or use **Reset view**
to restore the full history. The recent-step table shows
the latest ten training steps and the two earlier validation-only rows.

Metrics refresh every three seconds. Assisted training and unassisted
validation are labeled separately. Completion efficiency is conditional on
success; jerk includes failures; discarded training groups remain included in
training success. Missing historical metrics are gaps, not zeros. Log update
times show freshness, not proof that a trainer is currently running. A
partially written log retains the last complete snapshot with a visible
warning and is retried automatically. The server does not start, stop, or
modify training.

### Fresh full-module SFT

```sh
uv run --locked python -m so101.train_entity_sim \
  --collection data/simulation/entity_commands_with_difficulty_v1 \
  --output outputs/entity_smolvla_sft_full_triangular_5000 \
  --pretrained lerobot/smolvla_base \
  --steps 5000 --batch-size 2 --depth-mode known \
  --learning-rate 1e-5 --learning-rate-schedule triangular --warmup-fraction .1 \
  --validation-split outputs/entity_smolvla_trajectory_rl_all_assisted_lr1e4_before_step243_rerun/data_split.json \
  --validation-interval 250 --validation-seed 961000 --seed 61000 \
  --training-state outputs/entity_smolvla_sft_full_triangular_5000/training_state.pt \
  --obstacle-input-end-probability 1
```

This starts from pretrained SmolVLA plus fresh entity modules, **not an SFT/RL
checkpoint**, and trains all existing trainable modules (default scope `all`).
The asymmetric triangular learning rate warms up for **10%** of the run, peaks
at **1e-5 at step 500** (not the midpoint), then declines to **zero at step 5000**.
The 2,143-demo corpus excludes the frozen 40 validation demos, leaving **2,103
training demos**; new normalization statistics use training data only.

Validation every 250 steps uses the same frozen problems, original rollout
seeds and success predicates as RL: unassisted SDE sampling with **σ = .1**,
horizon **300** and action chunk **16**. Training and evaluation share one loaded
model; evaluation restores the training RNG afterward. Use the **SFT** dashboard
tab for loss-only training charts (including components); success, completion,
absolute jerk and wrong-object charts remain evaluation-only.

Loss charts show normalized means over valid action/entity labels, not batch
totals; do not divide them by batch size again. All three supervised-loss charts
use logarithmic step and loss axes; nonpositive values appear as gaps.
For optimizer batches larger
than 16, SFT uses activation-checkpointed microbatches of 16. Outputs are joined
before computing the loss, preserving full-batch padding/entity weighting and
one gradient clipping/Adam update per optimizer batch. The log records both the
optimizer and microbatch sizes. With `--gradient-accumulation`, each microbatch
is backpropagated immediately without activation recomputation, freeing its
graph before the next forward pass. Full-batch random inputs and loss
denominators are prepared once, so unequal padding/entity masks retain their
global weighting; clipping and Adam still run once per optimizer batch.
`--micro-batch-size` defaults to 16. Execution-mode changes on resume are
recorded without rewriting earlier updates or changing the sample schedule.
Adaptive LR/batch trials may reuse an evaluated CUDA step-zero checkpoint only
when it has no optimizer updates or Adam moments. They retain identical model,
RNG, and frozen evaluation inputs, record the cached baseline's provenance, and
save their own step-zero checkpoint. Later evaluations still run normally.
Complete exports validate the standalone model, full optimizer/RNG state, frozen
probe, and all retained quarter checkpoints against the untrained trial recipe.
`scripts/verify_adaptive_gpu_exports.py` independently confirms their immutable
S3 versions, sizes, and SHA-256 checksums before instance termination.

For the long B64/LR3e-5 duration experiment,
`scripts/prepare_extended_training.py` extends a completed CUDA/micro8
checkpoint while preserving its exact model, Adam, RNG, and evaluation history.
It records the original recipe and source checksum, and preserves the original
number of warmup updates: continuation neither restarts warmup nor decays LR.
`scripts/stage_long_gpu_training.py` restores version-pinned inputs and launches
this continuation on a new GPU. Its 16,384-update horizon is provisional;
duration conclusions apply only to the frozen sweep dataset, not subsequently
expanded or corrected data. Dashboard sync accepts the new host, instance,
remote output directory, pinned host-key file, and exact container identity.

### Sample-matched local SFT sweeps

```sh
uv run --locked python -m so101.sweep_entity_sft --dry-run
uv run --locked python -m so101.sweep_entity_sft
```

The serial sweep starts only after the original batch comparison has completed
and no trainer is active. It runs batches64/128 crossed with learning rates
3e-6/1e-5/3e-5, each processing24,576 sequence samples (384/192 updates).
New pilots use **10% warm-up followed by constant LR**, intentionally not the
original triangular decay. Checkpoints and frozen40 evaluations occur at equal
sample-budget quarters, not equal optimizer-step intervals.

A deterministic held-out loss probe uses four fixed frames per frozen problem,
training-only normalization, fixed diffusion noise/time, and the original
supervised reductions without gradients. It reuses the live policy and restores
training RNG/mode. Evaluation checkpoints retain full optimizer/RNG/history
under each run's `checkpoints/`; latest resumable state is still overwritten.
New log rows record `samples_seen`, and evaluation rows include supervised loss
and evaluation duration. Contrast loss is no longer trained or plotted.

Results and sample-axis loss/success plots are written under
`outputs/entity_smolvla_sft_sweep_24576`. Existing outputs are preserved, and
rerunning the same command resumes incomplete trials with strict recipe checks.
Equal exposure is only a compute proxy; wall time, optimizer overhead, valid
label counts and sampling streams can differ. Adam betas remain fixed.
The dashboard's SFT plots use logged cumulative `samples_seen` rather than
optimizer step, aligning different batch sizes at equal training exposure.
Validation loss (fixed held-out set) uses the frozen-set supervised loss,
including the zero-sample baseline. Tooltips retain
optimizer step for diagnosis; historical points without sample counts are
omitted explicitly, not estimated from batch size. RL plots keep their step axis.
Timing distinguishes optimizer-only duration from elapsed time since the reported
update, GPU report age, and the last completed evaluation duration. A scheduled
evaluation boundary awaiting a result is labeled pending/in progress, not a
confirmed live phase; fresh explicit phase reports take precedence. Stale GPU
reports make the current phase unknown rather than implying optimizer progress.
The top status strip lists active runs independently of plot selections or tabs,
using compact data-only labels. Run colors are assigned uniquely across the
catalog and retained through refreshes; selections do not change their colors.
Current curriculum recordings in the data viewer offer a text box and Submit
button for whole-example comments. They reuse the authenticated Family transport, immutable archive identity,
durable saved key, receipt checking and exact retries from the catalog viewer.
Legacy directory demonstrations and policy recordings currently lack a supported
immutable feedback target and are explicitly labeled unavailable.
Opening an item shows a fresh comment form. Only unconfirmed sends retain their
exact key and payload for retry, scoped to their original item/version; a durable
intake receipt clears browser storage. Confirmed comments are not kept as history
or shown on other examples. Sign-in appears only on an actual authentication rejection.
Original curriculum recordings use cached native H.264 video combining both
recorded cameras at their recorded frame rate, without MuJoCo reconstruction.
Native preparation retries transient gateway errors twice; a persistent failure
returns the control to Play for an explicit retry rather than showing a false Pause state.
Catalog discovery is serialized across dashboard consumers and streams part
summaries without retaining full part arrays. Selected catalog details still
load the complete metadata, and artifact fingerprints still verify original bytes.
`/api/health` checks HTTP service readiness without loading catalogs or archives;
catalog/data readiness and recorded-video playback require separate verification.
Preparation uses the existing bounded movie worker and byte-range serving; the
frame selector retains exact context for comments without fetching images during
video playback.
Curriculum cards show the task/target, generation category, object count, part mix,
creation date and recording length instead of legacy difficulty or redundant
acceptance labels. Category identifies the proposal recipe, not measured difficulty.
Parts, data, recordings and training share one server-rendered navigation header
and stylesheet, with the current page highlighted and mount-prefixed links.
Frozen40 is used for tuning, not an independent final test set; pilot rankings
need seed confirmation. The sweep does not automatically deploy a winner,
restart RL, or provision cloud resources.

### Long assisted trajectory RL

For a cloud SFT full-state checkpoint and its frozen episode database,
`python scripts/prepare_rl_transition.py --state <training_state.pt> --database
<episodes.sqlite> --destination <new-directory>` exports an initial standalone
policy plus byte-preserved `episode.json`/`frames.jsonl` references, without
copying images or modifying the SFT state. Run RL from that directory with
`--checkpoint initial_policy --collection data/simulation/rl_transition`.
Provide the simulator model separately and set `MUJOCO_GL=egl` before Python
starts on CUDA. This is a new RL optimizer, not a resumption of SFT Adam state.
The transition receipt identifies the exact SFT update and source checksums.
Online rollouts render only the trailing observation history of each action
chunk. Earlier frames would be evicted before the next policy decision;
physics, per-frame success/constraint checks, and optional rollout evidence
remain unchanged. With 16-action chunks and four-frame history this avoids
12 unused RGB renders per complete chunk.
Use `--no-initial-validation` to begin optimizer updates without a preliminary
baseline. Frozen evaluation still runs at `--validation-interval`; early
recoveries record that evaluation has not happened yet, not a synthetic score.

RL defaults to `--rgb-renderer mujoco`. Select the actual Mitsuba CUDA/OptiX
path tracer with `--rgb-renderer gpu-pathtrace --pathtrace-samples 16`, or
the LLVM CPU backend with `--rgb-renderer cpu-pathtrace --pathtrace-threads 4`.
These require the optional Mitsuba/Dr.Jit runtime; GPU mode additionally needs
a compatible NVIDIA driver and CUDA/OptiX runtime and never falls back to CPU.
The default 16 samples is an experimental online budget, not a validated
speed/quality recommendation; historical offline renders used 1024 samples.
CPU thread count does not control GPU rendering. This uses
the historical wrist path-tracing pipeline and original CAD jaw visuals, not
an image filter. All serial, validation and batched simulators use the selected
backend. Online path tracing reuses CAD mesh buffers while updating pose,
normals and material parameters; offline rerendering retains its existing
default construction path. Mesh reuse is recorded in the renderer identity.
Physics, geometric labels and recorded renderer-identity checks remain
unchanged; incompatible recorded appearance identities fail explicitly.
The RGB visual twin follows the recorded camera rather than claiming calibration.
Online visual meshes use in-memory OBJ triangle assets instead of inline XML
vertex/face arrays, avoiding native-scene XML parsing failures without changing
the saved scene, mesh coordinates, triangle indices, or physics model.
The CAD jaws can occlude RGB pixels differently from the legacy simplified jaw
geometry used for segmentation, visibility and rewards. Scoring is deliberately
unchanged for experiment parity; photo RGB is not pixel-aligned occlusion
ground truth and must not be used to claim improved visibility qualification.
`observation_renderer` records the effective profile and implementation hashes.
CPU path tracing can be much slower; compare actual rollout/group wall time,
not offline rendering benchmarks, before using it for training. GPU spare
memory does not establish spare compute: measure rendering contention with
full-model training before claiming an end-to-end speed improvement.
Serialized rollouts include wall-clock `timings` for reset/gold-prefix,
observation construction/rendering, inference through action readback, physics
stepping, and total rollout time before final evidence export. In-loop goal
checks and evidence overhead are included in total but not the physics-step
subtotal. Use these alongside sampled GPU
utilization to identify bottlenecks rather than treating EGL as batched GPU
simulation.
Stochastic denoising retains detached transition tensors on the GPU until the
path is complete, then batches their CPU transfer. Replay still receives the
same CPU tensors and sampled noise, without forcing a GPU/CPU synchronization
between every denoising step.

`--target-success-curriculum` replaces perfect-group mastery checks and
single-group zero-success recovery with a per-task rolling four-group success
window. Above 60% success, gold-prefix assistance decreases by the configured
curriculum decrement; below 40%, it increases, bounded by the existing minimum
and task maximum. A changed level starts a fresh window. Windows and strategy
transitions are checkpointed, preserving model, optimizer and RNG history on
resume. The target encourages mixed binary outcomes, not a guarantee of
improved learning; unassisted fixed held-out evaluation is unchanged.

`--reference-jerk-fallback` calibrates a frozen 75th-percentile margin from
successful training reference trajectories only (never RL rollouts), using
unassisted reference windows and the same five-joint units, horizon and FPS.
The checkpoint retains the calibration and exact reference-file hashes;
`frozen_jerk_calibration.json` exports its provenance. A usable problem-specific
reference still takes precedence; otherwise the pooled margin supplies the
jerk loss. `--reference-free-completion` instead rewards shorter successful
rollouts within each task group, independently normalizing this advantage
among successes. Failed rollouts receive no completion bonus. Historical
reference-relative completion-efficiency metrics remain unchanged.

`--auxiliary-view-manifest` admits a completed image-free harder-VIEW reference
manifest after checking file hashes, successful replay receipts and scene-family
separation from the frozen validation set and between auxiliary splits. Main
VIEW groups sample the auxiliary training references; the original collection
and fixed validation memberships stay unchanged. Auxiliary development success
is logged separately before its first training update and at validation
intervals; successful expert replay alone does not establish policy difficulty.
To explicitly advance an admitted curriculum, provide the new manifest with
`--auxiliary-view-previous-sha256` naming the exact previous admission hash.
The prior manifest must remain intact, cross-stage train/development family
leakage is rejected, and transition and validation history are retained.
The new development set is measured before training on the new references.

`--batch-rollouts` optionally batches inference across a training group's
independent simulator instances on CUDA or CPU. Each trajectory retains its
own observation history and seeded noise generator, including after other
trajectories terminate; recorded denoising paths are split back per trajectory
for unchanged policy and jerk losses. Replay reuses the exact frozen observation
KV features sampled in the batch: recomputing a BF16 prefix individually can
change its numerics enough to violate recorded-action replay. Each simulator's
EGL context and offscreen framebuffer are restored before advancing it.
Physics is still stepped serially;
held-out validation and mastery checks retain the serial evaluation protocol.
Per-trajectory timing in this mode includes time waiting for the other
trajectories, so summed timings overlap; use group/step wall time for throughput.

For off-instance recovery, `scripts/publish_rl_checkpoint.py` validates and
publishes the atomic RL state (all trainable weights, Adam, sampling RNG,
curriculum and history), bound to a dependency tar containing matching source,
`data_split.json`, the transition receipt, original input manifest and
versioned SFT recovery receipt. Reconstruct the initial policy/configuration from
that SFT state before applying the RL state. Historical projection-only states
still require their frozen pretrained weights; neither format replaces tokenizer,
normalization or scene dependencies. Publication accepts explicit full-model
scope and defaults to a 16 GiB state limit (`--max-state-bytes`).
Use a fresh task-owned `--root` per publication and `--profile
lego-artifacts` only when publishing from the authorized operator environment;
worker instances use their instance profile. Keep the resulting receipt,
including both dependency and state-manifest versions, with the run.

```sh
PYTORCH_MPS_HIGH_WATERMARK_RATIO=.65 PYTORCH_MPS_LOW_WATERMARK_RATIO=.5 \
OMP_NUM_THREADS=2 uv run --locked python -m so101.train_trajectory_rl \
  --checkpoint outputs/entity_smolvla_commands_norm_preserved_5000 \
  --collection data/simulation/entity_commands_varied_recovery_intact_2127 \
  --output outputs/entity_smolvla_trajectory_rl_all_assisted_lr1e4 \
  --steps 5000 --learning-rate 1e-4 --curriculum-mode mastery \
  --rl-estimator denoising-grpo --smoothness-advantage-weight 0 \
  --jerk-regularizer-weight .02 --jerk-scale 6 \
  --grasp-curriculum-max .95 --view-curriculum-max .95 \
  --push-curriculum-max .95 --place-curriculum-max .95
```

Gold-prefix assistance now works for all four tasks (grasp/view defaults remain
zero unless explicitly enabled). Each task/obstacle slice independently drops
by five percentage points only after four perfect eight-rollout groups and
a perfect assisted task-wide check at that same level. Unassisted validation
is logged separately. Low success or stalled assistance does not stop training;
non-finite gradients do, before corrupting the last completed state.

Every completed step atomically saves `latest_state.pt`, containing trainable
parameters, optimizer, curriculum/log, and sampling RNG states. Resume with
the same arguments plus `--resume`; the atomic state is authoritative even
if an interruption left the separate JSON incomplete. By default, two
checksummed generations are retained and `latest_state.pt` is a hard link, not
another model/optimizer copy. Evaluation no longer writes additional optimizer
checkpoints unless `--retain-evaluation-checkpoints` is requested. Full final
exports are opt-in (`--save-final-checkpoint`); compact evaluation history stays
in the log. Changing the trainable scope requires a fresh RL optimizer/run.
On Macs, run only one model-loaded trainer or qualification process at a time.
The live worker uses `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.65` and
`PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5` to leave unified-memory headroom; these
limit the MPS allocator, not total process/system memory.
`run_config` in the log records the launch arguments. Updates combine
task-reward policy gradients with a reference-margin jerk regularizer: no supervised demonstration loss or replay
minibatches. Expert episodes still provide scene initialization, gold-prefix
assistance, and reward references; held-out validation episodes remain excluded
from the training pool. Old checkpoints retain their optimizer history, but
receive no new supervised gradients. `training_objective_changes` records the
objective transitions; historical supervised metrics remain intact.
The obsolete `--sft-coefficient`, `--sft-batch-size`, and `--policy-target-kl`
flags are removed.

For automatic harder-data progression, use `--automatic-difficulty
--target-success-curriculum --difficulty-catalog /absolute/current/catalog.json`
with mastery mode and minimum assistance zero. Each task/obstacle slice uses
four training groups: above 60% success reduces assistance; sustained success
at zero assistance advances to an admitted native pile tier. Below 40% restores
available reference assistance, or backs off a reference-free native tier.
Windows reset on transitions; checkpointed slice state and immutable admission
hashes survive resume. Frozen validation/development outcomes never control
generation or advancement.
`--native-pool-manifest /absolute/current-pool.json` also consumes the continuous
producer's admitted snapshots: piece counts define physical tiers, at most
32 scenes are frozen per admitted tier, and later pointer updates can supply
new tiers without replacing already-used data. The run preserves its own
immutable manifests and historical training lookup for recovery/solved sampling.

Exhaustion queues one bounded TRAIN-only `so101.pile_task_worker` request at a
time. Proposals increase piece count from 6 to 128 and vary color diversity;
these are physical recipe tiers, not claims that a setup is solvable or harder
for the policy. Actual learning success controls advancement. Failed requests
retry at most three times with distinct seeds, then record an explicit blocked
state; exhausting generator bounds is explicit too. Worker results must match
request identity and manifest hashes, exclude held-out families, and provide
binary-model/full-integration reset, persistent instance labels and versioned
goals. There is no fallback to easy legacy VIEW reference data.

Native tasks use dynamic object identities, COM positions, real goal markers
and native dwell/contact graders. They do not require expert trajectories or
gold-prefix assistance. Automatic mode uses reference-free completion and
the frozen training-reference jerk calibration when jerk regularization is
enabled. Stored synthetic RGB is unnecessary: observations render live. Native
path tracing builds its visual twin from that problem's XML rather than the
legacy fixed scene. Matching topology/cameras does not qualify CUDA throughput
or RGB-versus-segmentation occlusion; check these on the actual training runtime.

With `--jerk-regularizer-weight .02`, success/completion advantages retain
weights **1 / 0.5**; `--smoothness-advantage-weight` must be **0** to avoid
counting jerk twice. The direct regularizer is
`max(RMS_command_jerk - problem_reference_jerk, 0)^2 / jerk_scale^2`, using the same
recorded-noise denoising derivative as the pilot trials, with observations
held fixed. It is not a gradient through MuJoCo.

Each problem is identified by its source episode, not its command category
(grasp/view/push/place). Its margin is the measured jerk of **that episode's own
successful expert trajectory**, at the exact gold-prefix assistance level.
No P90 or other aggregation across problems is used. Measurements require 10 Hz.
Unlike the fixed-horizon pilot, production stops on success, so reference
suffixes stop at their recorded end (or the configured horizon) without
artificial hold padding. Reference commands are loaded on demand, and margins
are cached as assistance changes. `jerk_reference_margins.json` records the
source catalog, protocol and measured profiles. Missing references disable
only that problem's jerk loss.
An unpadded suffix with fewer than four reference commands is unavailable,
not a measured zero jerk; its loss is disabled explicitly. A genuine zero
measured from at least four commands remains active.

Jerk gradients apply to successful and failed rollouts with at least four
commands **only in groups with at least one successful rollout**.
Zero-success groups are discarded from every learning component, including
jerk and experimental boundary losses; they cannot advance Adam momentum.
Task and jerk gradients are averaged over their own group counts before
combination, shared clipping and full AdamW. Group logs include margins and
losses; step logs include weighted component norms.
Frozen validation also retains each source, noise seed, success, rejection
reason and jerk summary, so comparisons can match successful rollouts instead
of treating early failures as smoothness improvements.

The experimental `--action-boundary-weight` (default **0**) adds a separate feasibility
loss on raw arm targets: squared excess of the rollout's largest arm correction
above the execution guard's **5-degree rejection limit**, divided by
`--action-boundary-scale` squared (default **6 degrees**).
It includes the rejected command even when no command executes, excludes the
gripper, and does not need an expert reference. Its denoising gradient bypasses
clipping, so saturated or rejected proposals can be corrected. Boundary gradients
are zero for all proposals the guard accepts, including permitted clipping.
They are averaged over their own eligible-group count before combination with task
and jerk gradients. The execution guard is unchanged: an arm excursion above
5 degrees still rejects the rollout. Logs record boundary losses, rejection
counts, component norms and objective transitions. This option is not a claim
that pass rates have recovered. The initial mean-excursion prototype penalized
valid clipping as well; its matched trial removed view rejections
but did not restore view passes and degraded other command probes, so it remains
disabled in the canonical recipe.

`--recover-zero-success-groups` makes the mastery curriculum recover when a
group has no task-reward signal: increase that command's gold-prefix assistance
by the existing curriculum increment, capped at its configured maximum.
Perfect-group streaks reset at the new level; the existing mastery check still
controls decreases. Losses and Adam updates are unchanged, and each problem's
jerk reference uses its actual assistance. Recovery events are logged.
Assisted training gains are not evidence of recovery: use the unchanged
unassisted frozen validation to judge improvement.

The legacy/diagnostic smoothness score remains
`min(expert_jerk_p10 / max(rms_jerk, 1e-6), 1)`, not a within-group ranking.
The reference is fixed per task in `data_split.json`; jerk is the RMS third
difference of the five commanded arm-joint positions over the policy suffix.
Completion and the diagnostic smoothness score remain success-only. Without
the new regularizer (weight 0), each component is
independently centered and standardized before combining advantages, with
default success/completion/smoothness weights **1 / 0.5 / 1**. The logged scalar
reward uses those same weights on the unnormalized scores.

An eight-of-eight successful training problem enters the persistent
`solved_problems` bucket, identified by episode source and exact assistance
level. Historical perfect training groups seed the bucket on first resume;
held-out episodes are excluded. Each step samples one additional problem
uniformly from the bucket as it existed before that step, and collects a fresh
group of eight stochastic rollouts. When all succeed, the success advantage
is zero, leaving completion and the jerk regularizer to optimize. If failures
return, their success signal is retained. Solved sampling neither lowers
assistance nor advances mastery; ordinary task sampling continues unchanged.
The bucket and sampling RNG are saved in the atomic resume state. An empty
bucket simply waits for the first perfect group; no solved group is fabricated.
Group logs distinguish `sampling_bucket: main` from `solved`; dashboard training
aggregates include both, including their different applied assistance levels.

### Data browser and task difficulty

The read-only browser is at **http://127.0.0.1:8765/data**, linked from the
dashboard. Browse source demonstrations independently of training runs, filtering
by task, scene difficulty, object count, demonstration result and quality.
Collection references without locally available episode metadata are counted and
explicitly disclosed, not presented as playable examples. Legacy examples remain
**unrated**. Select a training context after opening an item to see that run's
split, usage, policy outcomes and retained trajectory recordings. Training
filters are not exposed at the top level. Discarded groups count as training
attempts; expert success is not a policy pass, and historical solved-bucket
membership is not unassisted mastery.
Current curriculum SQLite examples are included in the same browse results,
deduplicated by archived example identity and ordered by their recorded generation
time. Cards prominently show Date created (or Not recorded for legacy data).
Curriculum details play original wrist/external frames without model reconstruction;
development examples remain labeled development, not training-promoted.

`collect_difficulty_sim.py` generates four authored levels (0 easy, 1 standard,
2 hard, 3 challenge) for each task using the procedural physics expert, without
loading a policy. Levels describe scene factors, not measured policy pass rates:

| Task | Increasing difficulty |
|---|---|
| Grasp | Clear central pickup; outer reach/rotation; distant camera/clutter; obstacle detour with collision-free retention |
| View | Visible/central; off-center; initially outside the image; genuinely occluded initially, then parallax |
| Push | Short straight; diagonal; cross-workspace; long diagonal with tighter destination tolerance |
| Place | Short carry; diagonal carry; cross-workspace carry; longer carry and precise release |

SO-101 has no mobile base: "search" means wrist/arm camera motion within reach.
Search/occluded view scenes suppress unseen entity centers **and depth** so the
policy cannot read the hidden target's ground-truth coordinates. Every scene
saves its initial camera visibility and geometry. New per-scene destination
tolerances and declared collision constraints are restored and enforced by RL;
legacy scene criteria are unchanged.

```sh
# Sequential, model-free generation. Existing directories require explicit --resume.
uv run --locked python -m so101.collect_difficulty_sim \
  --root data/simulation/task_difficulty_v1 --count 1 --attempts-per-case 4 \
  --resume --base-collection data/simulation/entity_commands_varied_recovery_intact_2127 \
  --merged-root data/simulation/entity_commands_with_difficulty_v1
```

Increase `--count` in a new root for a larger range of seeded examples.
Counts are bounded at 100 per cell and 10 attempts per example to keep seed
ranges disjoint. Resume requires the original seed. Missing successful cells
are reported explicitly and the command exits nonzero; incomplete collections
are not exported as merged training collections.
Failed experts and invalid scene attempts remain in the collection report;
only successful, verified demonstrations enter `successful_episodes` and the
optional separate merged collection. The existing run and its frozen validation
cases are **not changed automatically**. Adopting a new corpus requires an
explicit refresh/handoff; refresh also recomputes expert bounds, so compare
jerk in raw units rather than assuming old/new smoothness scores are identical.

Old policy logs contain only outcome metrics, not trajectories: the browser
labels these **summary-only** rather than substituting expert images for policy
footage. Future trainer invocations default to `--record-rollouts`: the first
rollout in each main/solved group saves every executed bounded action, measured
joint states, full physics positions, its seed and actual wrist frames every 16
policy steps plus first/last frames. Open an example and click **Play side-by-side
movie** for synchronized overview/wrist playback. **Previous/Next** arrows move
through every retained recording of that source across training iterations
(including multiple recordings at the same step); **Source expert** returns to
the demonstration. Summary-only and expired iterations are not invented as
movies. The slider still scrubs original wrist frames and plots the action trace.
Recording covers the policy suffix after expert
assistance. `--rollout-retention-steps 20` bounds retained step directories;
expired recordings keep their historical summaries. Use `--no-record-rollouts`
to disable. New recordings render both views from saved full physics positions.
Legacy sources pair original wrist images with an explicitly labeled
action-reconstructed overview; older policy recordings reconstruct both views
from their saved commands and assistance prefix. These reconstructions are not
claimed to be original camera footage. Playback requires `ffmpeg` with H.264
support. It runs one on-demand, policy-free rendering subprocess at a time,
caches at most 20 movies under `outputs/.data-browser-playback`, and supports
seeking. It never loads another neural policy or changes training checkpoints.

### Why jerk can still rise

`uv run --locked python -m so101.analysis.audit_trajectory_jerk <run>/trajectory_rl.json`
audits saved evidence without a model. The current measure is RMS third finite
differences of five **bounded commanded arm-joint positions**, in degrees/frame^3
at 10 Hz; multiply by 1000 for degrees/s^3. It excludes the gripper, assisted
prefix and prefix-to-policy handoff. It is not measured mechanical joint jerk.

The inverse-jerk score is absolute, but its advantage is still independently
standardized within successful rollouts; completion and success are separately
standardized too. Combined advantages are clipped and divided by their group
L1 norm. Thus absolute reward magnitudes do not survive into the update, worse
jerk can coexist with positive combined advantage, and doubling the jerk weight
does not double the final update. The inverse score is also compressed near
zero when policy jerk greatly exceeds the expert reference.
Failures receive zero smoothness reward regardless of jerk, and all-failure
groups are discarded. Dashboard/evaluation jerk nevertheless averages **all**
rollouts, so the plotted population is wider than the smoothness objective.
At evaluation step 240, push had zero successes: none of those unassisted
failures receives a smoothness bonus under this objective.

Before the reference-margin change, fixed unassisted evaluation jerk rose from **6.309 at step
190 to 7.138 at step 240** (+13.1%), so changing training assistance is not the
whole explanation. Across steps 215-243, 142 successful rollouts had below-group-
mean smoothness but positive combined advantages; solved groups were 29/261
accepted groups (11.1%), at mean assistance 62.1%. They have no success-advantage
variation when perfect, but retain completion/smoothness competition and cannot
guarantee the combined update improves either. Fixed stochastic sampling also
introduces noise, but these logs do not isolate noise versus optimizer effects.

Standardization is not ranking, and does not inherently prevent learning a
jerk-only objective. These logs do not establish that scalar-reward centering
is necessary or sufficient to fix the learning signal. Qualify jerk alone
before choosing how to combine it with task objectives.

`experiment_jerk.py` runs isolated real-policy trials from a reference checkpoint,
without touching its weights, optimizer or STOP marker. Fixed trials use exactly
eight source problems (two per task), the same training noise seeds each step,
and independent fixed probe noise. Default pilot scope is 64 commanded actions
after a fixed 80% expert prefix. Rollouts continue after success so an earlier
stop cannot artificially lower jerk; completion uses the first sustained goal
time. Invalid/truncated rollouts are reported and penalized in score-function
trials. This is a short assisted pilot, not a full unassisted-task qualification.

```sh
PYTORCH_MPS_HIGH_WATERMARK_RATIO=.65 PYTORCH_MPS_LOW_WATERMARK_RATIO=.5 \
uv run --locked python -m so101.experiments.experiment_jerk --steps 5 \
  --output outputs/jerk_trials/score --representation negative-rms
uv run --locked python -m so101.analysis.analyze_jerk_experiment outputs/jerk_trials/score/experiment.json
```

`--method pathwise` is an experimental alternative: differentiate a squared
excess-RMS jerk loss of bounded executed commands through the recorded Gaussian draws and all
denoising steps. Reentrant activation checkpointing limits memory. Cross-chunk
jerk derivatives are included; observation/state-feedback derivatives through
MuJoCo are **not**, so this is a local on-policy regularizer, not an unbiased
full simulator gradient. It uses no expert action targets or imitation loss.
The jerk margin is **each problem's own successful reference trajectory jerk**,
not a percentile or pooled command statistic. Reference
commands are measured after the same assistance prefix, over the same fixed
action horizon; if the demonstration ends early, its final command is held to
fill the window. Only recordings at the matching 10 Hz are used. The loss is
`max(RMS_jerk - problem_reference_jerk, 0)^2 / jerk_scale^2`, so there is no smoothing
pressure below that problem's reference. A problem without a usable reference has
its jerk loss **disabled**, not replaced by zero or another problem's margin;
its normal policy gradient remains active. Reports identify each problem's
source, command, margin, and explicit reason when disabled. Old pooled-P90
calibration files are rejected by new trials rather than silently reused.

Pathwise trials calibrate and save `jerk_reference_margins.json` automatically.
To measure once and reuse the exact same calibration:

```sh
uv run --locked python -m so101.jerk_reference \
  --split outputs/entity_smolvla_trajectory_rl_all_assisted_lr1e4/data_split.json \
  --output outputs/jerk_trials/reference_margins.json --horizon 64 --assistance .8
# Pass --jerk-reference-margins outputs/jerk_trials/reference_margins.json to a pathwise trial.
```

Changing the horizon, assistance or frame rate requires a matching calibration.
Earlier zero-margin qualification results predate this reference budget.
Noise recording is opt-in and does not change sampled actions or RNG sequence.
`--combined` retains separately normalized success/completion advantages;
pathwise mode adds their policy gradient to the analytic jerk gradient.
Combined trials log the actual weighted task/jerk gradient norms and their
alignment before shared clipping and Adam preconditioning.
`--varied --fresh-training-noise --steps 30` changes the eight training sources
and noise each step while retaining the fixed probe panel. `--init-state` and
`--continue-optimizer` support an explicit experimental continuation.
Varied-data continuations restore the saved problem-sampling state and advance
training-noise seeds rather than repeating the first batches after a pause.

Run only one trial worker at a time. Experiments use full AdamW, no KL guards,
backtracking or supervised loss. Logs/checkpoints stay in their own output
directory, and no experimental objective becomes the production default
without measured qualification.

For harder data, collect into a separate directory while RL continues.
Once a new merged collection is verified, create the run's `STOP` file.
The trainer finishes and checkpoints its current step, then exits. Remove
`STOP` and resume with the new `--collection`, `--resume --refresh-corpus`.
This rebuilds the training pool and expert bounds while retaining the
same validation cases and learned assistance levels. The new collection must
retain the original episodes as well as the additions.

### Denoising-path policy gradients

Add these flags to the trajectory trainer to use sampled-path likelihoods
instead of the default signed flow-matching MSE:

```sh
--rl-estimator denoising-grpo --denoising-noise-scale .1 --policy-clip .2
```

This keeps the flow model and relative success/completion/jerk advantages,
but samples Gaussian reverse-flow transitions and differentiates their
recorded log probabilities with a clipped policy-ratio objective. Log
densities sum over latent coordinates, never average inside the probability
ratio. A fixed `1 / (denoising_steps * chunk_size * latent_dimensions)`
scales the policy loss outside that ratio, keeping the established gradient
scale. Trajectory scores still sum over decisions
without trajectory-length normalization.
Likelihood recomputation uses the recorded path, not fresh denoising noise.
Training defaults to `--mixed-success-only`: groups with no successes or all
successes still inform difficulty calibration, but are discarded before task,
jerk, or action-boundary backward. They do not consume the accepted-group
quota; another problem is sampled within the configured attempt limit.
With `--calibrate-assistance --batch-rollouts --group-size 4`, synchronous
calibration retries the **same problem**: four passes reduce assistance; four
failures increase it; only mixed groups contribute training gradients.
Each attempt updates a task-type assistance EMA (`--assistance-ema-alpha 0.2`);
new problems start from that EMA unless `--initial-curriculum` supplies explicit
per-problem starting assistance. That JSON maps each task to a nonempty list of
`{"source": "TRAIN/episode/path", "assistance": 0.25}` records and restricts
sampling to those frozen TRAIN sources without changing held-out validation.
`--training-tasks grasp push place` restricts update sampling while retaining the
same four-task frozen validation; omitted tasks can be measured by the probe.
Native reference-free problems do not support gold-prefix assistance.
`--max-group-attempts-per-task` bounds total sampling attempts per task/update,
including calibration retries; exhausting it raises rather than silently skipping
an unfinished problem, including at assistance bounds. A single training policy
alternates rollout generation and optimization on the GPU. All gradients within an update use
the policy that sampled those groups, preserving strict cache/action replay
checks. The per-task EMA and normal training RNG are saved in full checkpoints.
An independent `python -m so101.probe_trajectory_rl` process can run one probing
rollout at a time alongside training. Supply `--checkpoint`, `--collection`,
`--split`, `--plan`, `--training-output`, `--output`, and `--model-root`.
The plan is a list of task/source/assistance records. Each problem/assistance
combination receives at least 16 repeats and a Wilson 95% success interval.
The probe pins the latest atomically published full-model snapshot per combination
(or the initial SFT before the first checkpoint), records its iteration, and never
mixes measurements across policy versions or constructs an optimizer. These are
lagged difficulty estimates, not training rollouts or automatic bucket admission.
For assistance calibration on new photo piles, use `--native-only --initial-only`
with `--native-pool` and `--native-references`. This rejects legacy problem sources,
pins the untouched initial policy, pairs rollout seeds across assistance levels,
and verifies the policy digest is unchanged. `--batch-rollouts 4` batches inference
without retaining training decisions; the default remains one serial probe.
`--calibrate-assistance` stops additional levels for a problem after 16 or more
repeats show 25--75% success, or all pass without assistance. Unmeasured levels
remain explicitly unmeasured, and all-pass problems are marked too easy, not
qualified training targets. Use 32 repeats to refine uncertain estimates.

Generate reference overlays with `python -m so101.collect_pile_references
--pool <immutable-pool.json> --model-root <model> --output <fresh-directory>`.
Attempts are balanced across scene families and default to 20 per task type.
The collector executes numeric policy actions, restores the exact integration
state, replays every action, and admits a reference only when the native dwell
grader succeeds at the end and maximum replay qpos error is at most 1e-7.
Scripted IK failures and failed predicates are reported, not called infeasible.
The completed overlay binds the original pool, scene files and action hashes;
it never changes the producer's reference-free capabilities or immutable scenes.
For task visualization, `python -m so101.capture_pile_task_images --pool <pool>
--model-root <model> --output <fresh-directory> --camera external` captures a
renderer-only external view of the actual unchanged reset. The default
`--camera wrist_cam` instead captures CPU16spp wrist RGB. Both require a working
MuJoCo GL context and bind per-target segmentation boxes to each image SHA.
Boxes use exclusive pixel maxima; invisible targets have a null box and zero
visible pixels. External visibility is not policy-wrist visibility or solvability.
`python -m so101.publish_photo_calibration --pool <pool> --references <manifest>
--output <dashboard-root> --images <external-directory> --images <wrist-directory>`
publishes all admitted tasks, preparation progress, and ordered camera images.
Add `--measurements <probe-results.json>` only for actual new-pile policy results.
Reference acceptance never supplies model pass rates; historical groups/images
are not merged into the new population.

The historical `python -m so101.calibrate_trajectory_rl` screening command
performs inference only on the final SFT checkpoint, with no optimizer; it is
not the repeated new-photo assistance calibration described above.
Supply `--checkpoint`, `--collection`, the existing frozen `--split`,
`--native-pool`, `--model-root`, `--source-commit`, and a fresh `--output` JSON.
It reuses five distinct TRAIN problems per bucket/task across assistance levels
0, 0.5, 0.95, then 0.25/0.75 only if needed. Cells stop after a majority of
problems yield mixed four-rollout groups, or all five pass at zero assistance.
Native scenes are measured separately by piece count at zero assistance only.
Scarce buckets report their actual population instead of duplicating problems.
The JSON and adjacent Markdown table update after every inference group;
the final parameter digest must match the initial digest.
Discard reasons and the sampled success histogram remain in the training log.
`--no-mixed-success-only` explicitly restores historical all-pass regularization
for controlled comparisons; all-fail groups remain discarded.
RL defaults to `--trainable-scope all`: all policy parameters are explicitly
unfrozen and optimized with FP32 weights, so small AdamW updates do not round
away in BF16 storage. The first active update verifies nonzero gradients and
actual parameter changes through vision, language backbone, history, entities
and the action expert/projections. Heads unused by the RL objective need not
receive gradients. Replay accumulates observation-cache derivatives across
denoising transitions. Within each training gradient component, row adjoints
stay on CPU until all trajectories have replayed; each original rollout batch
is then recomputed once and receives their combined vector-Jacobian product.
This preserves full-model gradients without retaining prefix graphs between
decisions or repeating the vision/text prefix for each row and denoising step.
Replay must still match recorded features within absolute
`1e-4`. Streaming likelihood losses must be backpropagated before advancing the
iterator; pathwise replay finalizes its prefix after loss backward. The group
prefix context must finish before measuring component gradients or updating
parameters. Standalone replay still finalizes its own prefix immediately. A changed
replay implementation triggers another parameter-update audit, retaining the
earlier audit. Each update records rollout, jerk-backward, task-backward,
optimizer and remaining wall time in `wall_phase_seconds`. Regularizer component
buffers live on CPU in full-model mode to avoid several full-model GPU copies.
The historical `--trainable-scope action-projections` remains explicit opt-in:
its vision/text prefix is frozen and may remain BF16, while its action expert
uses FP32. Scope, parameter count and gradient audit are recorded in the log.
Full entity checkpoints also record and restore the expert dtype before
loading weights; older checkpoints without that metadata keep legacy
loading behavior.

The final transition marginalizes noise in output coordinates that cannot
reach the simulator (padded dimensions and action slots beyond the execution
window). Earlier transitions retain their full likelihood because those
latents can influence later physical actions. This removes nuisance score
variance without changing the expected reward gradient; even invalid first
actions remain inside the scored execution window.

Updates apply the full AdamW proposal at the configured learning rate after
ordinary gradient-norm clipping. There are no conditional-KL probes,
backtracking, proposal-fraction scaling, or KL-based update rejection.
The policy-ratio clipping objective remains; it does not constrain the first
Adam step on fresh data.
Logs include policy/combined gradient norms (identical without imitation),
zero replay-gradient norm for historical compatibility, and whether an optimizer
update was applied. If no groups are accepted, the trainer
explicitly skips the optimizer update: no Adam momentum, weight decay, or
optimizer-counter advancement is applied.

The sampler uses `sigma(t) = noise_scale * sqrt(t)`, with the corresponding
reverse-SDE drift correction. This avoids endpoint singularities while
keeping positive variance in every sampled transition, including the last
transition to zero. Zero noise recovers the original Euler ODE sampler, but
likelihood-based training requires a positive noise scale.

Training, mastery checks, and trainer validation all use the selected
sampler. Other existing evaluation/control entry points still use the
original ODE sampler; their results are not SDE-policy evaluations.
Deterministic inference after SDE training needs separate evaluation.
Resume with the same sampler flags as the checkpoint's `run_config`.
For an estimator handoff, first request a graceful `STOP`, then resume
the latest atomic state with the new flags. Estimator changes are logged;
the optimizer, assistance levels, validation split, and RNG state survive.

## Video dependencies

On Linux ARM64, this project excludes TorchCodec using a platform-specific uv
override. Its published 0.11.1 ARM64 wheel requires CUDA libraries even from
the CPU index. LeRobot automatically falls back to PyAV, whose wheel includes
FFmpeg libraries. Video encoding and decoding work without a system `ffmpeg`.
Do not explicitly select the `torchcodec` video backend on that platform.

For TorchCodec on other platforms, install shared FFmpeg as the upstream
installation guide instructs:

```sh
# macOS Apple Silicon
brew install ffmpeg

# Ubuntu/Debian (on your own machine, not inside Dockyard)
sudo apt install ffmpeg
```

## Connect the robot

Run on the computer with direct USB access to the arm. Dockyard did not expose
any serial ports during setup, so hardware communication and calibration have
not been verified. Cameras also require access from the process running LeRobot.

Discover each controller with `uv run lerobot-find-port` and substitute its
actual port below. Linux ports commonly look like `/dev/ttyACM0`; macOS ports
look like `/dev/tty.usbmodem...`. The examples use Linux port names.
On Linux, your account needs permission to open the serial device, typically
through the `dialout` group (log out and back in after changing group membership).

### Motor setup, only when required

Skip this if the vendor already configured the motors. For unconfigured motors,
follow the SO-101 assembly guide: connect only the single motor requested by
the setup program, not the entire daisy chain.

```sh
uv run lerobot-setup-motors \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0
```

If you also have a leader:

```sh
uv run lerobot-setup-motors \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1
```

### Calibrate

Follow the interactive prompts and the SO-101 guide. Keep the same IDs for
subsequent commands so LeRobot can reuse the saved calibration.

```sh
uv run lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=so101_follower
```

If you also have a leader:

```sh
uv run lerobot-calibrate \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=so101_leader
```

Calibration is stored in LeRobot's user cache, not in this directory. Moving to
another computer therefore requires copying calibration or calibrating again.

### Teleoperate, with a leader and follower

To teleoperate with the USB camera and a live Rerun display, set the verified
arm ports and camera index on the host:

```sh
export FOLLOWER_PORT=/dev/tty.usbmodemFOLLOWER
export LEADER_PORT=/dev/tty.usbmodemLEADER
export CAMERA_INDEX=0
bash scripts/teleoperate.sh
```

Replace the example values using `uv run lerobot-find-port` and
`uv run lerobot-find-cameras opencv`. The script uses 640x480 at 30 FPS for
the camera and a 30 Hz control loop. Extra LeRobot arguments can be passed
to the script, for example `bash scripts/teleoperate.sh --fps=10`.
This command moves the follower; clear the workspace before starting.

Clear the workspace and be ready to disconnect power before starting:

```sh
uv run lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=so101_follower \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=so101_leader
```

If you bought only a follower, leader/follower teleoperation requires a separate
leader or a different supported input device.

For cameras and dataset recording:

```sh
uv run lerobot-find-cameras opencv
uv run lerobot-record --help
uv run lerobot-replay --help
```

See https://huggingface.co/docs/lerobot/en/il_robots for camera configuration
and the recording workflow. No motor setup, calibration, or movement is run
automatically by this project.

## Describe the current scene (camera only)

For a single fresh camera snapshot and a Copilot description, run on the Mac:

```sh
uv run --locked python -m so101.describe_scene
# Select a different camera or model if needed:
uv run --locked python -m so101.describe_scene --camera_index=0 --model=gpt-4.1
```

This opens only the camera, sends its JPEG directly to a tool-free Copilot SDK
session, prints the description, and exits. It never connects to the follower
or changes motor torque. It uses the same Copilot authentication described below.
`CAMERA_INDEX` sets the default camera index. Close other camera apps first and
allow camera access for the terminal if macOS prompts. The image leaves the Mac
for remote inference. Capture and SDK failures are reported as errors, not scene
descriptions.

Verified on this Mac on October 24, 2026: camera 0 supplied a fresh image and
Copilot returned a description of a green studded baseplate and light-gray
bricks, noting blur and tight framing. This camera-only check does not verify
follower feedback, physical motion, or visual accuracy.

## Host-side Copilot robot chat

Run this application **on macOS**, not in Docker. It uses the Copilot Python SDK
with real image tool results and direct LeRobot access to the follower. The
leader is not needed. Stop teleoperation and other camera/robot apps first.

Install/authenticate GitHub Copilot CLI if you have not already:

```sh
brew install copilot-cli
copilot
# Use /login in the CLI if prompted, then exit the CLI.
```

Your account needs Copilot access. The SDK downloads its own pinned runtime on
first use and uses your local GitHub authentication. Camera images and chat
content are sent to the configured Copilot model for inference.

From this directory on your Mac:

```sh
uv sync --locked

# No hardware: simulated joints and an explicitly labeled black test image.
uv run python -m so101.robot_chat --simulate=true --motion=true

# Real camera and position feedback, no commanded motion.
uv run python -m so101.robot_chat

# Real camera plus small follower joint moves.
uv run python -m so101.robot_chat --motion=true
```

Defaults are your follower port `/dev/tty.usbmodem5B901012531`, camera `0`, and
model `gpt-4.1`. Override with `FOLLOWER_PORT` / `CAMERA_INDEX`, or CLI options:

```sh
uv run python -m so101.robot_chat \
  --follower_port=/dev/tty.usbmodem5B901012531 \
  --camera_index=0 \
  --model=gpt-4.1 \
  --motion=true
```

Example conversation:

```text
Look at the camera and describe the workspace. Do not move.
Read the current joint positions.
Move wrist_roll by +2 degrees, then look again and describe the change.
```

To run one instruction and exit instead of opening an interactive prompt:

```sh
uv run --locked python -m so101.robot_chat --motion=true \
  --prompt="Observe the scene and read joint positions. Move wrist_roll by +1 degree exactly once, then read positions and observe again. Report measured feedback. Do not move any other joint."
```

One-shot mode uses the same motion limits and shutdown behavior as interactive
mode, including releasing torque on exit. Status and motion feedback are logged
to the terminal independently of the model's description.

The tools are `observe_scene`, `robot_status`, `nudge_joint`, and `stop_robot`.
Only those tools are allowlisted; shell/filesystem permissions are denied.
`observe_scene` returns JPEG bytes to the model, not a path it cannot read.
Images are captured on demand, not streamed. Tool calls and errors are logged
to the terminal. `/stop` latches motion off between turns without asking the
model. `/quit` exits. Ctrl-C interrupts the app, including during a model turn;
pending motion checks the stop flag between trajectory steps.

### Motion limits and physical precautions

Support the follower in a clear, stable starting pose **before launching**.
Startup disables torque and validates existing calibration; it does not
recalibrate. Even the observation-only mode connects to the motor bus and
disables torque, so an unsupported arm can fall. Torque is enabled on the first
motion request, after priming all goal positions from fresh joint feedback.

The chat and teacher tools currently expose single-joint relative nudges:

- At most 5 degrees per request, at a commanded 5 degrees/second.
- Gripper uses 0-100 percentage units instead of degrees, with the same step/rate limits.
- Targets stay within 15 units of each joint's startup position.
- At most three moves per user turn.
- Motion is serialized, completion is checked from feedback, and hardware faults
  latch stop until restart.

Direct control can also call `MotionController.move_to({...})` with absolute
joint targets. This is one synchronized multi-joint trajectory and consumes one
move, regardless of its distance within the configured bounds. For example,
`controller.move_to({"shoulder_pan": 20, "shoulder_lift": 60})` moves both joints
together, holding unspecified joints at their measured starting positions.
Each joint advances at at most five units/second; intermediate servo updates do
not split the request into separate nudges or require camera checks. Feedback
checks enforce joint bounds, an eight-unit maximum tracking error, and a
two-unit limit on uncommanded drift or overshoot beyond the planned path.
Completion requires measured progress from every moving joint. Opt-in partial
results identify unfinished targets explicitly. Configure bounds that contain
the intended pose; the default startup envelope remains +/-15 units.

These bounds are conservative starting values, **not collision avoidance,
certified velocity limits, or proof that a pose is safe**. There is no inverse
kinematics, depth estimation, tabletop calibration, or pick-and-place skill yet.
The model can inspect and nudge; it cannot reliably sort Lego pieces yet.

The software stop prevents further targets but may leave torque enabled and
the arm pursuing its last target. Normal shutdown attempts to disable torque;
communication failure can prevent that. Support the arm on exit and keep its
physical power cutoff accessible. Do not rely on a model tool or Ctrl-C as a
hardware emergency stop. The app lock excludes other copies of this app, not
independent LeRobot programs.

### Verification and handoff

```sh
uv run pytest -q tests/test_robot_control.py
```

Container verification covers simulated movement limits, stop/fault behavior,
SDK image tool results, runtime startup, and restricted session creation.
An authenticated model identified a synthetic image's red/green halves and
executed a simulated wrist nudge through the SDK tools. The camera-only command
above has also been exercised on the Mac with a real image.

On October 24, 2026, a real one-shot Copilot session observed the scene, read
follower feedback, requested one +1-degree wrist-roll nudge, and reobserved.
The session reported wrist roll changing from approximately 109.23 to 109.85
degrees toward a 110.23-degree target, with other joints unchanged. No clear
camera-image change was distinguishable. This verifies one small-motion
integration trial, not precise one-degree positioning or general motion safety.
The controller's completion tolerance is one unit, so a completed nudge can
still fall short of its requested delta.

The user did not observe the wrist motion, so that trial is not considered
camera-verified. A subsequent direct test bypassed Copilot, captured a baseline
image, commanded two +5-degree shoulder-pan nudges, and captured images after
each nudge while torque remained enabled. Direct inspection showed a substantial
left/up shift of the baseplate and bricks in the image. Shoulder-pan feedback
changed from -6.46 to +1.98 degrees (approximately +8.44 degrees).
The evidence is saved in `outputs/direct_motion/before.jpg`,
`outputs/direct_motion/after_step_1.jpg`, and
`outputs/direct_motion/after_step_2.jpg`. This verifies camera-visible direct
movement, not the earlier wrist-motion claim or general manipulation capability.

For a new setup, start in simulation, then observation-only mode, then try one
small wrist movement with a clear workspace.

Later on October 24, direct assistant control using wrist camera 0 and laptop
overview camera 1 picked up one long gray piece with coordinated `move_to`
trajectories. Both cameras showed it held off the table at the lift endpoint and
again 66.6 seconds later; it was then lowered, released, and the controller closed.
Four earlier closures/lifts were empty and are retained in the trace. Evidence
and measured feedback are in `data/direct_grasp/pickup_pose_20261024_1118/`,
especially `overview_0023.jpg`, `overview_0024.jpg`, and `outcome.json`.
This is a single direct pickup, not a verified autonomous SDK grasp, a repeatable
policy, or a synchronized training dataset.

Subsequent near-vertical approaches produced two further verified pickups of the
large studded gray beam, including a changed starting arm pose and object
location (`data/direct_grasp/repeatability/`, observations `0017`, `0018`, and
`0026`). The other large, hollow-side-up gray beam has not been picked up;
alternation and arbitrary-start reliability remain unverified. The latest
attempt stopped on a joint-progress fault, without closing on or lifting a
piece; see `data/direct_grasp/alternating_20261024_1316/status.json`.
Demonstration collection remains paused.

When `record_demo.py` is used without recording, low-light observations also
produce brightness-corrected, averaged stationary-frame diagnostic JPEGs.
Original snapshots are preserved. These derived images and the laptop overview
remain outside demonstration episodes; recording uses only the wrist stream.

Between moves, the controller preserves previously commanded holding goals,
including a closed gripper's goal, rather than resetting them to measured
positions that may lag under load. New trajectories start from those command
goals. Unexpected movement between requests still latches stop; tracking,
progress, speed, and travel guards are unchanged.

## Wrist-only simulation and SmolVLA

`robot_sim.py` runs entirely without USB access or real cameras. It uses the
Apache-2.0 SO-101 model from MuJoCo Menagerie, pinned to
`f3b6013629c7b94f8ee93fd977b161fa8350f9bc`; `setup` downloads the model,
referenced meshes, and upstream license under `data/simulation/model/`.

```sh
uv sync --extra simulation --extra training
uv run --locked --extra simulation python -m so101.robot_sim setup
uv run --locked --extra simulation python -m so101.robot_sim collect \
  --root=data/simulation/gray_beams --episodes=50 --attempts=150 --seed=0
uv run --locked --extra simulation python -m so101.robot_sim export \
  --collection=data/simulation/gray_beams \
  --destination=data/simulation/gray_dataset --repo-id=local/gray_dataset
uv run --locked --extra simulation --extra training python -m so101.robot_sim train \
  --dataset=data/simulation/gray_dataset --output=outputs/smolvla_sim \
  --steps=20000 --batch-size=1 --device=mps
uv run --locked --extra simulation --extra training python -m so101.robot_sim evaluate \
  --checkpoint=outputs/smolvla_sim/checkpoints/last/pretrained_model \
  --root=data/simulation/policy_evaluation --episodes=20 --seed=10000
```

Use fresh collection, dataset, training, and evaluation directories. For a quick
training smoke run, use `--steps=2`; that is not enough to learn the task.
Training saves locally with Hub uploads and W&B disabled. `cpu` and `cuda`
are also supported device choices. Rendering uses MuJoCo's local OpenGL backend.
Training uses four persistent data workers and checkpoints every 1,000 steps
and at completion. New policies predict sixteen actions (1.6 seconds at the
dataset's 10 Hz, rather than the base model's five-second chunk) and replan
every sixteen actions (1.6 simulated seconds); evaluation can override this with
`--action-steps`. Use `train --pretrained=PATH --learning-rate=3e-5` to fine-tune
an existing local checkpoint on a new collection at a lower learning rate.
Evaluation saves wrist-frame trajectories, raw predictions,
applied actions, and contact diagnostics for each rollout, with reproducible
per-episode sampling seeds. Small finite boundary overshoots (at most five
degrees or gripper units) are projected to the joint limits and explicitly
counted in the report; larger or nonfinite predictions stop the rollout.

The scene has white printed arm parts, a table, and two 128 x 16 x 9.6 mm gray
beam proxies: a studded solid beam and an open hollow beam. Studs are visual-only;
the hollow beam's base and walls participate in collisions. These are not
measured CAD replicas of the actual pieces. Initial arm poses, object positions,
and object yaw vary by seed; modest lighting and color variation is applied,
while wrist-camera mounting and field of view stay fixed.
Pass `--varied-layout` to both collection and evaluation to also randomize which
piece is on each side and increase lateral position jitter to +/-2 cm. This
prevents identifying the requested piece from a memorized left/right location.
For corrective demonstrations, add `collect --recovery-checkpoint=PATH`: the
learned policy first reaches an imperfect intermediate state, then the expert
records a verified correction and pickup. The untrusted policy prefix is not
exported as demonstration actions. Repeat `export --collection=PATH` to combine
nominal and corrective collections.

The scripted IK demonstrator alone uses privileged object poses. It approaches,
descends, closes separately, requires contact with **both jaws**, then lifts.
Success requires at least 5 cm elevation and both-jaw retention at every sample
for one simulated second. No welds, object attachment, or object-pose writes
are used after reset. Failed attempts remain in the collection report and cannot
be exported. Targets alternate across independently reset successful episodes.

Only `observation.images.wrist`, six measured joints, and the task condition the
policy. Actions are absolute joint targets in degrees, with gripper travel mapped
to 0-100; samples pair pre-action state/images with the next command at 10 Hz
using the physics clock, not wall time. Diagnostic overviews and privileged
object poses stay outside the dataset inputs. The exporter labels these datasets
`so101_sim` and refuses mixed real/sim collections until their joint conventions
are calibrated. Evaluation reports target-specific physical-contact success on
new seeds, not training loss or expert success.
Lifting the unrequested piece stops the rollout and is reported as a failure.

This is a simulation bootstrap, **not a calibrated digital twin or verified
sim-to-real policy**. Camera extrinsics, servo offsets/compliance, friction, and
piece geometry still need real-unit calibration and transfer evaluation.
Neither the training nor evaluation command can deploy a policy to hardware.

The local bootstrap produced ten verified alternating expert episodes
(1,192 wrist-only frames), and the expert passed another twenty unseen seeds.
A 20-step MPS fine-tuning smoke run saved a checkpoint with changed weights at
`outputs/smolvla_sim_verified/checkpoints/last/pretrained_model`. This verifies
the training path, not a competent learned pickup policy. A subsequent
evaluation with a 30-second budget stopped both unseen rollouts on invalid
predicted joint targets (0/2 pickups). A full training run and real-robot
transfer have not been completed. Detailed artifact checks are in
`data/simulation/verification.json`.

## Persistent Copilot teacher

The experimental teacher tries to generate a grasp demonstration through slow,
camera-guided robot interaction, without requiring leader operation:

```sh
uv run --locked python -m so101.robot_teacher --motion=true
```

It uses a vision-capable SDK model (`gpt-6.1-sol` by default), supplies a fresh
image and joint state each turn, permits one bounded movement, and returns
another image with measured feedback. Override `--model`, `--camera_index`,
`--follower_port`, `--task`, `--max_steps`, or `--max_seconds` as needed.
No motion is enabled without `--motion=true`. For hardware-free integration:

```sh
uv run --locked python -m so101.robot_teacher --simulate=true --motion=true \
  --workspace=data/robot_teacher_sim --max_steps=2 \
  --task="Test one wrist nudge, record a finding, then finish blocked because this is simulation."
```

The default persistent workspace is `data/robot_teacher/` (gitignored).
`findings.jsonl` stores the model's observations and hypotheses across sessions;
`runs/<run-id>/` contains timestamped events, camera JPEGs, and the outcome.
`read_findings` and `record_finding` give the session notebook access without
general filesystem or shell permissions. Notes are untrusted hypotheses, not
verified calibration or instructions. Images and notes supplied to Copilot
leave the Mac for inference. Use a separate workspace for simulated experiments.

Teacher mode deliberately differs from the original chat controller: it allows
at most 40 movements/turns and 900 seconds by default, one move per SDK turn,
with arm joints within +/-20 degrees of startup clipped to saved calibration,
and gripper targets within 0-60 percent. Each move remains at most five units,
at a commanded five units/second. These are not collision-free envelopes.
Target completion uses the same one-unit tolerance as ordinary chat and requires
measured movement in the requested direction; merely being within tolerance while
stationary is not success. Teacher mode can report `outcome: partial` when valid
in-bounds feedback shows progress but the target is not reached within two seconds.
It returns the measured delta, target error, and a fresh image so the session can
learn the response. This is not a success result or permission to force a blocked
direction. No progress, unexpected movement, invalid feedback, communication
errors, and limit violations still latch stop. Ordinary chat retains its strict
timeout behavior.
Hardware/camera errors latch stop; budgets cannot be reset by model tools.
The app holds the same lock as robot chat, but independent teleoperation programs
still do not participate in it. Startup/shutdown torque precautions still apply;
normal exit releases torque, including after a candidate grasp.

The teacher cannot certify its own success. `candidate_grasp` requires recorded
before/after image IDs and is saved with `independently_verified: false`; a human
or separate image review must establish that the object was actually lifted and
held. Simulation cannot claim a candidate grasp. Failure, uncertainty, and physical
or visual blockers should be recorded, not converted into success.

These exploratory per-movement traces are **not yet synchronized LeRobot training
datasets**. SmolVLA base weights have been downloaded to the Hugging Face cache;
policy dependencies, MPS inference/training validation, dataset recording, and
fine-tuning are still pending.

See [PROJECT_PLAN.md](docs/PROJECT_PLAN.md) for the selected SmolVLA direction, teacher loop, visualization,
demonstration collection, and physical evaluation stages.
