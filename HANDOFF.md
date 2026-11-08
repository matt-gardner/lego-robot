# Finish the learning-rate sweep

The Work Scheduler should start a **new container session now** to finish
the sweep, including the missing GPU execution support. Use template
**Lego robot - 6.1 sol**, cwd `/mnt/dockyard/lego-robot`.
No previous session is needed.

## Remaining runs

Each run starts fresh from `lerobot/smolvla_base` and sees24,576 samples.
The two LR1e-5 runs are complete; do not rerun them.

| Order | Batch size | Learning rate | Updates |
| --- | --- | --- | --- |
| 1 | 128 | 3e-6 | 192 |
| 2 | 64 | 3e-6 | 384 |
| 3 | 128 | 3e-5 | 192 |
| 4 | 64 | 3e-5 | 384 |

**Allocate one GPU, keep it running across all four runs, then terminate it.**
Run one model at a time. Save/export each run before loading the next fresh
model; do not terminate or reallocate between runs. Retain independent expiry
and emergency cleanup.

## How to finish

1. **Finish the multi-run GPU executor.** Private broker access is automatically
   installed at `~/.config/lego-robot/control.json`. Read
   `docs/CONTAINER_CONTROL.md`; reuse `control_broker.py` and
   `control_workflow.py` for allocation, staging and lifecycle management.
   The installed recipes restore a trained checkpoint at step304, so they
   cannot run these fresh arms unchanged. Implement and activate a fresh-capable
   runtime/spec, genuine step0 qualification, and serial arm transitions on
   the same instance with a bounded whole-sweep lease. Replace the approval
   path's hardcoded304 requirement; do not fake checkpoint state.
2. **Run the four arms.** Use `sweep_entity_sft.trials(24576, ...)` and
   `command(..., device="cuda")` for their arguments. Preserve original
   data/split/physics, seed61000, microbatch16, 10% warmup then constant LR,
   frozen40 evaluation problems and fixed160-frame probe. Evaluate at
   0/6,144/12,288/18,432/24,576 samples. Do not incorporate unrelated data
   workers' simulation changes or run the local CPU sweep as a substitute.
3. **Export and compare.** The current publisher exports only training state.
   Add durable exports of model/config/metadata, full model/Adam/RNG state,
   logs and retained evaluation checkpoints for every arm. Verify complete
   updates, fixed evaluations/probes, finite state and exact artifact hashes.
   Combine these with the two completed runs using the sweep's verification
   and comparison helpers: equal-sample loss/success, per-command results,
   completion, timings and jerk on common-success full trajectories.
4. **Shut down.** Confirm actual EC2 termination after the whole sweep and
   remove finite monitoring watches. The new session owns paid-run monitoring
   and failure recovery, not the Scheduler.

There is **no ready one-command fresh-arm GPU sweep runner yet**: the new
session must finish step1 and then execute the sweep, not stop after setup.
Do not use `gpu_smoke` per arm; it terminates the instance.
Use AWS quota/budget alerts, private credentials, TLS, ownership and idempotent
requests. Do not add software budget/count gates or require per-arm messages
to a host agent.

## Inputs and preservation

- Output root: `outputs/entity_smolvla_sft_sweep_24576/`.
  Dataset: `data/simulation/entity_commands_with_difficulty_v1/`.
  Frozen split is defined by `sweep_entity_sft.SPLIT`.
- Installed broker runtime: `/opt/lego-robot-control/runtime-current`.
  Root recipes/workflows/preset: `/etc/lego-robot-control/`.
  Trusted registration: `/opt/lego-robot-control/session-register`.
  New immutable recipe activation must be durable and self-service.
- A prepared B128/LR3e-6 step0 state exists under
  `/var/lib/lego-robot-image/fresh-self-service-states/b128-lr3e6/`.
  Config/probe binding and GPU qualification still need verification.
  Local image `lego-runtime-b1c9551-self-service:latest` is not yet published
  or activated in GPU specs.
- Completed B64/LR1e-5 step384 state:
  `/mnt/dockyard/lego-gpu-original-training-export/92df9f9e5f764b4180e73195090da6f9/training_state.pt`.
  Integrate missing standalone exports from this verified state with provenance.
  Preserve the original CPU304 checkpoint/STOP and all other workers' files.
- Disclose a comparison caveat: the completed GPU continuation used the
  default obstacle-end probability0.5; the sweep builder explicitly sets1.
  Preserve actual argv/logs rather than claiming identical protocol.
  Device differences also prevent a clean same-device timing comparison.

Done means all six runs have verified artifacts and a sample-matched comparison,
and the shared GPU is terminated. No extra grid, RL, deployment or robot motion.
