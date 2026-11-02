# SO-101 project handoff

Updated November 2, 2026. This is a handoff, not permission to launch training,
spend money, or move the physical robot.

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
The lockfile deliberately selects CPU Torch on Linux. CUDA training needs a
separately qualified CUDA environment. No training environment on the AWS host
has been qualified by this handoff.

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

The first trial's final passes were view7/push2/grasp0/place0.
The second trial's latest evaluation passed view7/push2/grasp1/place0.
Its 10/40 is an intermediate evaluation, not its final result.
The remaining four grid cells have not started.

These are single-seed pilots on a tuning set, not an independent final test.
Do not declare or deploy a winner yet. Compare per-command success, fixed
held-out loss at equal samples, completion, timings, and jerk on common-success
problems only. Use full trajectory durations, not equal cropped windows.
Equal samples approximate compute, not exact FLOPs/wall time; unchanged Adam
betas also imply different momentum half-lives in samples across batch sizes.

## Current stop and recovery state

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
Local originals are retained. Nothing has been launched for training on AWS.

## Next steps and guardrails

1. Keep the authoritative checkout separate from persistent datasets, outputs,
   and model caches. Preserve results, full resumable state, frozen splits,
   normalization, evaluation snapshots, and recovery accounting.
2. Resolve and qualify the execution target before resuming. The CPU host is
   suitable for coding agents/orchestration, not a GPU worker. AWS Spot quota
   approval is not a GPU-launch or spending authorization. The controller's
   GPU-launch permissions, trusted credential broker, cost/hour bounds,
   interruption recovery, and independent shutdown backstop are not ready.
   Review `docs/AWS_SPOT_OPERATIONS.md` before implementing those pieces.
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
4. Only after runtime qualification and launch approval, explicitly clear
   `STOP` and resume the second trial from update304, then finish the approved
   grid serially. Do not silently change samples, batch, microbatch, learning
   rate, seeds, guards, or evaluation protocol. No RL, automatic winner
   deployment, or additional sweeps are currently approved.

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
