# SO-101 project handoff

Updated November 8, 2026. Matt explicitly reaffirmed continuing the whole
approved learning-rate sweep, not stopping after one recovered trial. See
the GPU operating handoff below for current state and execution boundaries.
This document does not authorize additional grids, increased spending limits,
or physical robot motion.

## Project and code

The project controls an SO-101 robot using LeRobot and the Copilot SDK.
Implemented areas include bounded hardware control, scene observation/chat,
demonstration recording/replay, simulation/expert-data collection, SmolVLA
training/evaluation, and a training/data dashboard. See `README.md` for the
module map. Plans under `docs/` describe intent; inspect code before assuming
any proposed feature is complete.

Application code is in `src/so101/`, tests in `tests/`, operational guidance in
`docs/`. Use Python 3.12 and the existing `pyproject.toml`/`uv.lock`.
Typical commands, after the appropriate environment is installed:

```sh
uv run --locked python -m so101.<module>
uv run --locked pytest -q
```

Recreate environments on Linux; never copy the Mac virtual environment.
The lockfile deliberately selects CPU Torch on Linux. Do not use that environment
for GPU training. The digest-pinned CUDA runtime described below has now run
the original SFT workload successfully on a real NVIDIA L4.

## Recent completed work

- Removed contrast loss from the active training code and dashboard plots.
- Dashboard remembers page, smoothing, collapsed sections, and other UI state
  across refreshes. Do not redo this work.
- Implemented a serial, resumable, sample-matched six-cell supervised
  fine-tuning (SFT) sweep, fixed supervised probes, retained full-state
  evaluation checkpoints, and comparison plots.
- Fixed an MPS out-of-memory failure during evaluation by releasing completed
  training graphs/gradients and unused accelerator cache before the fixed
  probe. The previously failing step192 probe and subsequent updates were
  successfully exercised without shrinking the batch/probe or disabling caps.
- Provisioned an AWS CPU host with encrypted persistent storage, Docker,
  Session Manager access, and role-based identities. A real stop/start
  preserved files, Docker image cache, mount, memory limits, and metadata rules.
- Configured certificate-based Roles Anywhere access from the Mac, avoiding
  repeated AWS root/browser logins for ordinary authorized host access.

## Approved experiment and results

The question is how batch size and learning rate affect learning at equal
sample exposure, rather than equal optimizer steps. The user approved this
bounded pilot grid, in this serial order:

| Effective batch | Learning rate | Updates | Samples |
| --- | --- | --- | --- |
| 128 | 1e-5 | 192 | 24,576 |
| 64 | 1e-5 | 384 | 24,576 |
| 128 | 3e-6 | 192 | 24,576 |
| 64 | 3e-6 | 384 | 24,576 |
| 128 | 3e-5 | 192 | 24,576 |
| 64 | 3e-5 | 384 | 24,576 |

Shared recipe: microbatch16, seed61000, 10% warmup then constant learning rate,
training-only normalization, and the same frozen40 tuning problems.
Evaluate at 0/6,144/12,288/18,432/24,576 samples, using validation seed961000,
horizon300, noise0.1, and action16. The deterministic supervised probe uses
160 frames and seed1061000, isolated from training RNG/mode.

Artifacts are under `outputs/entity_smolvla_sft_sweep_24576/`.
The original dataset catalog is
`data/simulation/entity_commands_with_difficulty_v1/collection.json`:
2,143 successful episodes, with the frozen40 excluded from training.

| Trial / exposure | Frozen40 passes | Held-out supervised loss | Completion efficiency |
| --- | --- | --- | --- |
| B128/LR1e-5, final 24,576 samples | 9/40 | 0.737404 | 0.814388 |
| B128/LR1e-5, 18,432 samples | 8/40 | 0.799415 | 0.698405 |
| B64/LR1e-5, 18,432 samples | 10/40 | 0.700926 | 0.767045 |
| B64/LR1e-5, final 24,576 samples, GPU continuation | 11/40 | 0.661915 | 0.541894 |

The first trial's final passes were view7/push2/grasp0/place0.
The second trial's latest evaluation passed view7/push2/grasp1/place0.
Its 10/40 is an intermediate evaluation, not its final result. The November 8
GPU continuation completed update384 and its final frozen40 evaluation.
The remaining four grid cells have not started.

**The goal is all six cells and their sample-matched comparison.** Completing
B64/LR1e-5, publishing a checkpoint, or shutting down one GPU is an intermediate
milestone, not permission to stop the sweep. Continue the four remaining cells
in the table's order, serially. Stop only when the whole approved pilot and
comparison are complete, Matt explicitly pauses it, or a concrete safety or
authority blocker prevents progress. Surface a blocker and its next action;
do not silently park, declare a winner, or mark the sweep Done after one cell.

These are single-seed pilots on a tuning set, not an independent final test.
Do not declare or deploy a winner yet. Compare per-command success, fixed
held-out loss at equal samples, completion, timings, and jerk on common-success
problems only. Use full trajectory durations, not equal cropped windows.
Equal samples approximate compute, not exact FLOPs/wall time; unchanged Adam
betas also imply different momentum half-lives in samples across batch sizes.

## Current stop and recovery state

The paragraphs about update304 below are historical recovery context. The
original CPU/MPS checkpoint and STOP marker remain preserved, but the GPU copy
completed304->384 on November 8. Its complete state and authentic log are now
durable; use the distinct exported GPU checkpoint identified below. Do not
overwrite the original304 checkpoint or clear its local STOP marker merely to
make the local sweep driver accept completion.

The user explicitly stopped local training and requested future runs move to
AWS. Local training is stopped; do not restart it automatically.
The local supervision schedule46 is disabled.

The second trial stopped gracefully after update304/384, or 19,456 samples.
Its final `training_state.pt` contains the model, complete Adam state, sampling
RNG, Python RNG, NumPy RNG, Torch CPU RNG, device RNG, and continuous log.
Finite model/Adam state was verified locally. `model.pt` was also exported.
The `STOP` marker is intentionally preserved.

Earlier, this trial failed during the fixed probe after update192 and resumed
from durable update168. Recovery repeated24 updates/1,536 samples, plus
evaluation overhead. Preserve this caveat in compute/timing comparisons.
The first trial's original driver wall time was 9,928.588 seconds (~2h45m);
do not mistake tiny reused-driver-session timings for whole-trial wall time.

The driver reports failure after the deliberate stop because it rejects an
incomplete trial. This is not a new divergence/OOM. Consult `MIGRATION.json`,
the trainer log, and the full state rather than treating that label as a new
training incident.

## AWS layout and migration status

Current authoritative checkout: `/mnt/dockyard/lego-robot`.
Persistent storage: `/mnt/dockyard/lego-robot-data/{data,outputs,hf}`.
Ignored checkout symlinks preserve `data/` and `outputs/` relative paths.
The migration receipt reports verified completion; a separate host inspection
matched all 315,542 manifest files (22,870,035,633 bytes), the archive checksum,
2,143 episodes, and the stopped update-304 state. Migration paths remain
compatibility links/evidence, not a second working tree to edit.
The checkout preserves the newer dashboard base-path and notice support.
The sections below describe the original transfer and recovery context.

Existing instance: `i-0860eb23354785a2c`, Oregon `us-west-2`, CloudFormation
stack `dockyard-cpu`. It is an On-Demand `r6a.large`: two vCPUs, 16 GiB RAM,
NO GPU. The encrypted disks are 20 GiB root plus 80 GiB retained data mounted
at `/mnt/dockyard`. This mount is ordinary persistent storage; the user
accepted keeping this project's files there.

The separately transferred, extracted, checksum-verified code is:

```text
/mnt/dockyard/so101-migration/code
```

The original bulk migration is still in progress at handoff time. It contains
the exact dataset, simulation assets, frozen split/baseline logs, all sweep
states/exports/evaluation snapshots, and exact public pretrained model caches.
It was over-inclusive: the 16.4 GB compressed archive includes historical
states and public weights that are not all necessary to resume the current
trial. Minimal current-state plus dataset is about6.9 GB before compression.
Do not assume the bulk copy was reduced; it was not.

The receiving directory is `/mnt/dockyard/so101-migration`.
Completed `so101-migration.partNNN` files are checksum-verified archive pieces;
`.part` suffixes denote incomplete pieces. These are NOT an extracted dataset
or usable checkpoint workspace. Avoid starting a duplicate transfer.
The original transfer is driven from the Mac, not by an AWS training job.

On successful assembly/extraction/full file verification, the pipeline writes:

```text
/mnt/dockyard/so101-migration/migration-receipt.json
/mnt/dockyard/so101-migration/workspace/so101
/mnt/dockyard/so101-migration/workspace/hf
```

Check the receipt (`status=verified`) and actual files before claiming the
migration complete. The model-cache location is the future `HF_HOME`.
The separately copied `code` directory is distinct from this full snapshot.
Local originals are retained. This last sentence described the November 2 transfer state. AWS GPU training
has since completed the second cell; current receipts and boundaries follow.

## Next steps and guardrails

1. Keep the authoritative checkout separate from persistent datasets, outputs,
   and model caches. Preserve results, full resumable state, frozen splits,
   normalization, evaluation snapshots, and recovery accounting.
2. Use the qualified GPU architecture below, not the CPU host for training.
   The original continuation is finished, but fresh-cell GPU orchestration,
   registration, and container authority still need explicit integration.
   Read `docs/CONTAINER_CONTROL.md` and `docs/AWS_SPOT_OPERATIONS.md`; older
   statements that no AWS runtime is qualified are historical.
3. Qualify cross-device continuation explicitly. `sweep_entity_sft.py` accepts
   `--device cuda` while retaining MPS as the default and preserving the recipe.
   Both SFT and RL require `--allow-device-rng-migration --device-rng-seed SEED`
   for a backend change. Legacy SFT states identify their backend through saved
   `run_config.device`; new states also record it in RNG metadata. Foreign RNG
   bytes are never applied to the new accelerator. CPU/sampling RNG, model,
   Adam state, sample history, and the continuous log are restored; the new
   device RNG is explicitly seeded and the non-bitwise migration is recorded.
   Source support is not CUDA recovery qualification. Do not clear original
   STOP or restart training without the runtime qualification/handoff above.
4. Verify the two completed cells, integrate the exported GPU result without
   altering the original stopped CPU state, then finish the remaining four
   cells serially through the broker. Never restart the completed second cell
   from304 or clear the original STOP as a bookkeeping shortcut. Do not change
   samples, batch, microbatch, learning rate, seeds, guards, or evaluation
   protocol. No RL, automatic winner deployment, or additional grids.

Never launch a second loaded training policy concurrently. On the Mac,
preserve MPS high/low watermark ratios0.65/0.5 and OMP threads2; never disable
the memory cap to hide an allocation failure.
Do not delete `data/` or `outputs/` as cleanup.
Do not move the physical robot without explicit authorization.

The host is shared with other agent/application work. Do not stop or modify
unrelated services, containers, paths, or infrastructure. No inbound ports
were needed for this work; do not expose a dashboard or SSH publicly.
Host-operator SSM access supports administration, not hostile-code isolation.
Ordinary coding containers must not receive the Docker socket, privileged
mode, host networking, host credentials, or EC2 metadata access.
Use roles and temporary credentials, not AWS root or permanent access keys.
The Mac's private machine certificate keys are not in this project and must
not be copied into agent workspaces.

## GPU operating handoff: November 8

### Ownership and completion condition

Matt requested a new worker to own the **entire remaining sweep**, including
necessary GPU integration, continuous monitoring, durable artifacts, and the
sample-matched final comparison. Use one container implementation worker, not
competing GPU operators. A bounded trusted host activation helper may install
reviewed infrastructure/specification changes; it must not run a second sweep.
The original operator is Dockyard session
`465a15cf-8e65-41d2-a3f2-a2a9e2339e10`. The Lego project is
`3fd44373-a454-4f3f-8d48-d32ee0682e66`.

Do not finish this assignment after one cell, a qualification PASS, a loss row,
checkpoint publication, or VM shutdown. Continue to the next approved cell.
After all six cells, verify the fixed probes, full exposure, durable states and
retained evaluation artifacts, update comparison results, and terminate the
last GPU. No unapproved additional search or winner deployment.

### Actual completed GPU run and authoritative artifacts

The second cell ran on an On-Demand `g6.2xlarge` NVIDIA L4 in `us-west-2d`:
job `92df9f9e5f764b4180e73195090da6f9`, instance `i-02671f30f7db3b6fb`.
Original304->384 training and the final40-problem evaluation completed.
All384 authentic log rows and final evaluation are served by the existing
dashboard, `/lego-robot/`, run ID `bf3598f0d7a5f5c9`.

The complete model/Adam/RNG/log checkpoint is **not** the original CPU file:

```text
/mnt/dockyard/lego-gpu-original-training-export/92df9f9e5f764b4180e73195090da6f9/training_state.pt
s3://so101-training-133961745003-us-west-2/runs/lego-robot/92df9f9e5f764b4180e73195090da6f9/step_000000384/54ab1f5cd97cdd137cc6fea3cb44da0ea69d385aa22e42fcea8c533cbca47867.pt
version: KLG1fO.6BEcnqwiQVeorJbWBoQ6rv3m7
SHA256: 54ab1f5cd97cdd137cc6fea3cb44da0ea69d385aa22e42fcea8c533cbca47867
bytes: 1242249483
```

Its exact version, size, entire SHA, fsynced CPU copy, finite model/Adam state,
RNG, continuous384-step history and unchanged original304-prefix were verified.
The live chart now contains its authentic full `state["log"]`, not fabricated
interpolated rows. The pre-GPU chart backup is beside `sft_training.json`, named
`sft_training.pre-gpu-92df9f9e5f764b4180e73195090da6f9.json`.

The original CPU `training_state.pt`, `STOP`, data and old evaluation snapshots
are preserved. The worker's separate `model.pt`/metadata and retained384
evaluation snapshot were not included in the standard state-only S3 export.
Do not claim those individual artifacts were exported. Integrate/derive needed
exports from the verified full state with provenance, preserving existing
files; do not blindly call the local driver's `verify_state()` against the
still-original304 file or rerun this completed cell.

The actual GPU was directly observed EC2 `terminated` at08:21:10 UTC, with
empty GPU inventory confirmed again during this handoff. All previous GPUs are
terminated. Temporary training grants and the operator's added `gpu.start`
scope were revoked; there is no running sweep or retry cron.

Combined result receipt:

```text
/mnt/dockyard/service-home/.copilot/session-state/a3f8c707-2054-46f0-88bf-72046bf0d460/files/completed-original-gpu-training.json
/var/lib/lego-robot-image/qualification-fast-on-demand-d-tmp-fix-attempt-2/
```

The latter root-owned directory contains the manifest/plan, runtime qualification,
installed aggregate-approval invocation, original stop/export, dashboard and
actual termination receipts. A host helper can provide reviewed nonsecret
copies; a container should not assume it can read root-private files.

### Qualified runtime and immutable inputs

Published Docker recipe fixes are in commit
`767c2c00470dc1a40e7e3e6660dced4c83925b43`. The actual frozen worker source is
`e923bc147a332027b129a836cea277a006f86196`, archive SHA
`b1b5ec888e7c321e9aff4f829ababe3dccb8aaccd95ccf5b2c58c8d3b05b2a5a`.
Do not mistake the recipe commit for a different source archive or relabel an
old image. The qualified image is:

```text
133961745003.dkr.ecr.us-west-2.amazonaws.com/lego-robot-runtime@sha256:33aea54ba958d439aad43a596fc0814cce7a5ef2b627fc66d9991c5c82055179
```

It has a real UID/GID1001 passwd identity, `HOME=/tmp`, and temporary storage
for read-only CPU validators. The training launcher uses its explicit writable
tmpfs. Real CUDA13/L4/BF16/EGL, original full-state restoration and unchanged
trial qualification succeeded. Controlled1.242GB checkpoint publication took
38.74s; complete CPU export, cold-GPU exact next-model/Adam update, independently
scheduled termination and isolated guest self-expiry were genuinely exercised.
The scheduled termination test also had concurrent broker expiry; it was not
an isolated scheduler-only shutdown or a real Spot interruption.

Use `/var/lib/lego-robot-image/registration/consolidated-assets/bundle-tmp-fix.json`
as the operator's immutable input/runtime index. Seven inputs totaled
10,171,367,419 bytes for the completed continuation. Stage at most four exact
versioned objects concurrently to local EBS; verify every version/size/fullSHA
before committing the population receipt. Do not re-extract315k episode files.

The episode SQLite SHA is
`e7fdb523d29c815fafbd54502545b02d5ed1f3cb5643e613be6ca8be903f9ea9`,
size5,968,420,864, version `dpFCeimVpHDmGmiPNjZLkV7_PC5MPHz`.
Mount it read-only at `/opt/so101/data/episodes.sqlite`, retaining the original
collection catalog, ordering, pairing, normalization and split. Equality was
verified for2103 training episodes/304198 samples plus frozen40/160 probes.
Keep `--episode-storage` outside the saved recipe configuration.

The simulation model is also required before the first full SFT update:
original `model.tar`, SHA
`618ab54829ab92f7b38f1343460a7890e3c1832086ed5281b65fadbc396ee122`,
version `9NUSwDrszi20MJArEhwdqAgTNJISk9y6`, 17,459,200 bytes.
Extract its22 members to `data_model`, then mount `data_model/model` read-only
at `/opt/so101/data/simulation/model`. Also preserve the exact frozen split,
collection JSON and HF archive; `HF_HOME` points to extracted `hf_cache`
containing `hub`, not to `hf_cache/hub`. Never package HF tokens.

### Container control versus trusted host administration

**A container can orchestrate GPU jobs through the host HTTPS broker. It is not
an AWS administrator, and a new container is not already authorized.**
The broker process remains on the CPU host; this does not require a continuously
running host AI operator once the reviewed sweep path is installed.

The private endpoint is `https://host.docker.internal:8766`
(`172.17.0.1` on the existing Docker bridge). Verify its TLS certificate/SAN.
An operator must issue a unique client ID and private0600 capability file for
the new container, install the public CA, permit only that container's bridge
IP, and reload the root-owned policy. Do not reuse another session's token,
the host-operator token, or a broad bridge/network allow rule.

The current container client has only `dashboard.status`/`dashboard.restart`.
Only the historical host principal has GPU submit/status/logs/stop/recovery;
`training_authorized_specs` is empty. Merely selecting the Lego container
template does not grant GPU control. Issue only the new worker's necessary
GPU lifecycle scopes, including `gpu.start` and, if implemented, `gpu.resume`.
The exact registered spec needs a separate training grant after genuine
current-worker approval. Preserve job ownership under that new principal.

Once provisioned with its own private files, the container uses:

```sh
PYTHONPATH=src python -m so101.control_broker call gpu.submit \
  --url https://host.docker.internal:8766 \
  --token-file ~/.config/lego-robot/control.token \
  --ca-file ~/.config/lego-robot/control-ca.crt \
  --body '{"spec_id":"OPERATOR_REGISTERED_CELL","spec_sha256":"EXACT_CANONICAL_SHA256"}'
```

Use the returned `job_id` with `gpu.status`, `gpu.logs`, `gpu.start`,
`gpu.stop`, and `gpu.recover`; `gpu.resume` additionally needs a newly verified
registered resume spec and `parent`. Never instantiate a separate
`Broker.submit()` process: its process-local lock raced the live expiry watcher
and previously killed a just-launched VM. All admission/start/stop mutations
must use the existing live HTTPS broker.

Containers cannot register arbitrary specs, change IAM/templates/documents,
stage/publish new immutable inputs using host credentials, install aggregate
approval, edit root training policy, or run general SSM shell commands.
These are bounded trusted activation responsibilities, not reasons to give
the worker sudo, Docker socket, host network, metadata access or AWS secrets.

### Required integration before the four fresh cells

The existing operator scripts and fixed approval document were deliberately
bound to the original304 continuation. `approve-gpu-qualification.py` checks
`original_step == 304`; the existing manifest contains that trial's state.
They are **not a ready-made fresh-cell sweep executor**.

Each remaining cell must initialize from the approved public base and seed,
not from the trained B64/1e-5 checkpoint. Implement and exercise a reviewed
fresh-state path (including correct step0/initial optimizer semantics), per-cell
immutable manifest/spec/recipe, provisioning, actual runtime qualification,
aggregate approval and separate exact-spec training grants. Derive and check
the actual state's step/configuration; do not forge a304 report, reuse another
job's PASS, or relax identity checks to make a fresh trial pass. Changes to
pinned worker code/documents/templates/images require corresponding reviewed
publication and trusted activation, not mismatched source/image labels.

Use `sweep_entity_sft.py` for the approved grid, argument construction,
sample-aligned verification and comparison semantics; integrate GPU execution
and artifact ownership rather than invoking its local subprocess loop on the
CPU host. Reconcile recorded recipes/driver arguments, including obstacle
schedule flags not fully represented in old `run_config`, before constructing
new manifests. Do not silently change experimental conditions.

The current publisher exports one complete stopped state, not automatically
all `model.pt`, metadata and retained evaluation checkpoints. Include those
required sweep artifacts in durable export/verification, with explicit
provenance for anything reconstructed from a verified full state. Preserve
real device migration/replayed-update/startup/evaluation timing caveats; the
first cell was MPS and the second crosses MPS/CUDA, not a clean same-device
wall-time comparison.

### Lease, capacity, budget and continuous monitoring

Keep at most one actual GPU and one loaded training policy. Existing A10G Spot
capacity attempts were explicitly rejected; conditional On-Demand L4 was
authorized. The proven L4 rate is $0.9776/hour. The prior full jobs had two-hour
leases; use only operator-approved templates, purchase options and bounds.
Capacity changes by zone: zoneB was rejected before the successful zoneD run.
Do not assume availability, repeatedly replay tokens, or claim a rejection is
unallocated without actual AWS error/inventory evidence.

The root policy still limits lifetime reservations to $20. At this handoff the
ledger reserves $19.3972, leaving $0.6028, insufficient for another two-hour
$1.9552 reservation. Reservations include historical failed/shortened attempts
and are not actual billed spend. Reconcile only provably unallocated attempts
or conservatively accounted, actually terminated usage using preserved AWS
receipts. Never reset the ledger, erase paid history, shorten a lease merely
to evade accounting, or raise the limit silently. If supported reconciliation
cannot provide the remaining sweep's bounded allowance, surface the precise
budget decision needed from Matt, not a new experiment-choice question.

For every fresh allocation, verify the independent one-shot AWS termination
schedule and the actual guest self-expiry guard before long work. Persist job,
instance, spec, deadline and command intents. Observe fixed SSM commands to
terminal status; SSM online can precede installation of `early.py`. The old
successful path waited three minutes for cloud-final. Retry a missing-handler
provision only after exact FAILED/empty-stdout/never-executed proof. Do not
replay accepted or ambiguous start, approval or stop commands.

Verify real training rows/configuration and full update progress, not optimistic
broker status. Mirror only actual rows while preserving original logs, then
prefer the verified full checkpoint's authentic log. Local checkpoint interval
does **not** imply off-GPU S3 persistence. Request normal checkpoint publication
with enough time for evaluation, validation, upload and full export before the
hard lease; retain all previous durable checkpoints.

After publication, verify exact S3 version/size/SHA, complete CPU download/fsync,
full model/optimizer/RNG/log and required sweep artifacts. A clean container
exit alone is insufficient. The completed run had a controller grace-expiry
cleanup warning despite actual stop SUCCESS and independently verified export;
inspect actual receipts rather than treating either warning or status as proof.

Directly verify EC2 `terminated` and empty relevant GPU inventory before the
next cell. The worker owns bounded watches and failure recovery; the Work
Scheduler does not monitor paid runs. Do not add the removed retry cron.
Continue the next approved cell automatically when safe, and keep the sweep
Work item active until the entire grid/comparison is genuinely finished.
