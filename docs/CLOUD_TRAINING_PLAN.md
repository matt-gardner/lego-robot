# Locally controlled Spot training

## Status and recommendation

The first Azure implementation is available in `cloud_training.py`,
`cloud_worker.py`, `cloud_checkpoint.py` and `cloud/azure-resources.json`.
Follow [CLOUD_TRAINING_SETUP.md](CLOUD_TRAINING_SETUP.md) for account, billing,
quota, bootstrap and handoff steps. No cloud resources have been deployed or data
uploaded. Local mock/CPU checks are not cloud GPU or real eviction qualification.
AWS and automatic cross-zone/disk-loss restoration remain future work.

The bounded first version restarts the **same deallocated Azure VM**, rather than
creating replacement VMs or moving disks between zones. Sessions expire into a
paused state requiring an explicit new start, and allocation attempts reserve
whole sessions conservatively. Source/data staging is immutable once the cloud
run has committed; updates and permanent VM deletion/recreation remain operator
procedures. This narrows the architecture below to a usable single-zone pilot.

Start with one Azure Spot GPU VM, eviction policy **Deallocate**, and a separately
managed persistent data disk. AWS Spot plus an EBS volume is the same architecture.
"Reserved storage" means persistent storage billed while compute is off, not a
reserved GPU instance or guaranteed Spot capacity. Choose the provider/SKU after a
short CUDA/EGL benchmark; do not assume an A100/H100 host can provide the required
headless renderer until the bootstrap smoke test passes.

```text
This Mac
  Copilot -> local controller -> cloud API / SSH
                    |
                    +-> local mirrored logs + worker status -> dashboard :8765

Cloud
  one Spot VM -> one supervised trainer -> persistent disk
                      |                     |
                      +-> heartbeat         +-> versioned checkpoint backups
                                               in Blob Storage / S3
```

The laptop controls desired state but is not in the training hot path. Closing the
terminal must not stop training. With only a local controller, replacement pauses
while this machine sleeps; an optional tiny always-on CPU controller would remove
that limitation without needing an always-on GPU.

## Durable state and staging

Use an initial 128 GiB persistent disk, monitor free space, and resize after
measuring the CUDA environment/cache and checkpoint retention. A suitable layout is:

```text
/mnt/training/
  workspace/                 # pinned source archive, data/, outputs/
  so101-cloud/
    venv/                    # isolated CUDA environment
    cache/                   # uv and Hugging Face caches
    training.env
    installed-requirements.txt
  control/
    desired-state.json       # run, recipe, budget, pause/stop intent
    worker-status.json       # observed generation, PID, heartbeat, step
    generations/             # committed checkpoint manifests + hashes
```

Provision/mount the data disk independently of the VM's OS/temp disk. Explicitly
retain it on VM deletion; Azure Deallocate retains disks but also continues disk
billing. For AWS, disable DeleteOnTermination on the data volume. Restrict destructive
cloud operations to the worker VM, never its storage by default.

Stage a manifest, not the entire laptop workspace:

- The exact code revision, or a SHA-256-identified source archive (this checkout is
  not currently a Git repository), plus `uv.lock` and dependency metadata.
- Base checkpoint `outputs/entity_smolvla_commands_norm_preserved_5000` and pinned
  model/tokenizer dependencies needed to reconstruct the frozen policy.
- Canonical `latest_state.pt` and `data_split.json`; JSON plot history is included
  for convenience but the atomic model/optimizer/log snapshot is authoritative.
- Collection `collection.json`, and `episode.json` plus `frames.jsonl` for **every
  referenced source directory**, including validation sources. The collection
  references other corpus directories; copying its directory alone is insufficient.
- Pinned MuJoCo model XML, mesh assets, and revision metadata under
  `data/simulation/model`.

The current 2,127-episode corpus requires about **315.7 MiB** of metadata/action
files and the base checkpoint about **0.858 GiB**, excluding model download caches.
The implemented bundle additionally pins and includes the two required cached
Hub snapshots (about **2.74 GiB** here), excludes tokens/unrelated caches, and
runs the worker offline against those exact revisions.
The current reward-only rollout path reconstructs scenes and replays expert
actions; it does not read recorded demonstration images. Preserve relative source
paths and launch from the workspace root. Never copy a macOS virtual environment.
Credentials and unrelated recordings/outputs are excluded from the staging manifest.

## Worker and controller lifecycle

The controller should expose `start`, `pause`, `resume`, `status`, `logs`, and
explicit `destroy-compute` commands from this machine. A normal stop never deletes
storage. It records desired state before making cloud API calls, so a controller
restart does not accidentally restart a deliberately paused job.

1. Reconcile desired vs observed cloud state; enforce one worker generation and
   a configured GPU-hour/price budget. No automatic on-demand fallback.
2. Allocate a Spot VM in the disk's zone, attach the disk, mount by UUID, and run
   the bootstrap against a pinned GPU image. Avoid formatting any existing disk.
3. Validate source/data/checkpoint hashes and free space, acquire an exclusive
   run lock, and start the trainer as a non-root systemd service. SSH disconnects
   must not affect it. The disk lock prevents two local trainer processes; a
   controller lease/generation additionally prevents two replacement VMs writing.
4. Resume the newest verified committed checkpoint with its recorded recipe.
   Preserve weights, Adam moments, curriculum, solved bucket, split, and sampling
   state. Never restart from the supervised base or separate stale model exports.
5. On eviction, preserve desired state as running, mark the worker interrupted,
   wait for the old VM to stop and release the volume, then retry with bounded
   exponential backoff and jitter. Never force-detach an actively written disk.
6. Retry capacity/eviction failures; surface corrupt checkpoints, dependency
   failures, non-finite updates, and resource exhaustion as errors requiring
   attention, rather than endlessly relaunching the same failure.

Set a price ceiling, total GPU-hour budget, and maximum capacity-wait policy.
Keep an audit trail of launches, costs, checkpoint generations, and user commands.
Systemd restart policies must respect pause intent and must not hide fatal errors.

The worker's metadata-event watcher should request `STOP` on an eviction notice.
Azure notice is best-effort up to **30 seconds**; AWS normally provides **two
minutes**, also best-effort. Our current step can take longer than either window.
Therefore **recovery cannot depend on finishing the current step**. Normally it
replays the one in-flight step from the previous completed checkpoint.

## Checkpoint durability and qualification

The existing trainer already saves trainable parameters, optimizer, full log
(including mastery and solved problems), and RNGs at each completed step. Keep
that logical boundary; there is no need to change the RL algorithm.

Implemented locally; validate these on the actual cloud deployment:

- Add file and directory `fsync`, retained checkpoint generations, hashes, and an
  atomically published commit manifest. Rename alone protects readers from partial
  files but is not a full power-loss durability protocol.
- Publish immutable generation objects to Blob/S3, then their commit manifest;
  never advertise a partially uploaded object as resumable. Record both locally
  committed and remotely backed-up steps, and the possible recovery lag.
- Validate the checkpoint structure, model/optimizer finiteness, manifest hashes,
  source revision, and data split before starting a replacement trainer.
- Distinguish user pause from eviction shutdown; clear only the acknowledged
  interruption STOP marker when resuming an intended-running job.
- Tag device RNG state with its backend. Current restore code selects the target
  device but blindly applies the saved device RNG bytes. MPS RNG is not CUDA RNG:
  the initial Mac-to-CUDA transfer needs an explicit, logged CUDA reseed while
  preserving other state. Subsequent CUDA-to-CUDA restarts restore CUDA RNG.
  Cross-device continuation is not bit-exact and must not be presented as such.
- Retain the current reward-only recipe, absolute jerk scores, solved sampling,
  full AdamW, frozen FP32 expert and BF16 prefix. Cloud migration must not restore
  supervised losses, KL/backtracking, or change precision to chase benchmark speed.

EBS/managed disks are zonal resources: switching availability zones cannot simply
reattach the same disk. Start with same-zone recovery. A later multi-zone fallback
must restore a verified disk snapshot/object checkpoint and staged data into a new
volume; its recovery point may lag the last local commit. Object backups protect
against accidental disk loss and permit that fallback, but do not reserve GPU
capacity.

## Dashboard stays on this machine

Keep the current server bound to `127.0.0.1:8765`; do not expose its unauthenticated
HTTP port on the VM. Mirror cloud logs into a separate `outputs/cloud/<run>/`
directory, never over the output of a live local trainer.

The controller synchronizes only changed revisions, using SSH/rsync deltas or
immutable per-step events, and atomically replaces local log files. Do not fetch
the entire growing history over HTTP every three seconds. The browser continues
reading local data normally; plots/history remain usable while the GPU is evicted.

The dashboard now reads an adjacent `remote_status.json` from the controller
mirror. Local runs still use `active_run_outputs()`. Cloud runs require both
fresh contact and a fresh running-worker heartbeat to show the "Since last step"
counter. States include running, checkpointing, waiting for capacity, paused,
interrupted/stale and error. Archived runs stay inactive. Contact age and backed-up
step are shown separately from optimizer-step timing.

SSH uses verified host keys and a dedicated account; cloud permissions are scoped
to the experiment's VM/disk/storage resources. If changing hosts, verify their new
host keys through the cloud control plane rather than disabling host verification.
Application/UI updates are staged as a new source generation and applied through
a checkpointed restart; never hot-edit the active trainer.

An SSH tunnel to a remote dashboard is a useful diagnostic fallback, but is not
the primary design: the dashboard would disappear with the evicted VM.

## Fresh-machine bootstrap (implemented)

Use Ubuntu 24.04 x86_64 on an NVIDIA GPU image with driver **580 or newer**.
The script deliberately does not install/reboot GPU drivers or format disks.
After securely staging the checkout on the mounted persistent disk:

```sh
cd /mnt/training/workspace
bash scripts/setup_cloud_gpu.sh --storage /mnt/training --dry-run
bash scripts/setup_cloud_gpu.sh --storage /mnt/training
source /mnt/training/so101-cloud/training.env
cd "$SO101_ROOT"
"$SO101_PYTHON" train_trajectory_rl.py --help
"$SO101_PYTHON" rl_dashboard.py --help
```

The script installs system libraries, pinned uv, Python 3.12, training/simulation
extras, and an isolated CUDA 13.0 environment. It preserves locked package
versions/markers while replacing CPU build selectors, explicitly asks uv for CUDA
wheels, and saves installed versions. It checks dependency consistency, CUDA
gradients, BF16 support, EGL rendering and trainer imports **without loading a
policy**. Failures are errors, never CPU or software-rendering success fallbacks.
Rerunning reuses the environment/cache; it does not touch the robot/MPS `.venv`.
Use `--skip-system-packages` only on an image with those libraries already present.

Do **not** use plain root-project `uv run`/`uv sync` for the worker: that project's
Linux sources intentionally select CPU PyTorch. Use `SO101_PYTHON`. The generated
CUDA environment is derived from the committed lock plus CUDA runtime dependencies;
the resolved inventory is recorded, but a fully immutable worker still requires
a built image pinned by digest. `training.env` is written only after smoke checks
pass; it is environment configuration, not a resume launcher.

Bootstrap alone does not start training. `cloud_training.py` stages the verified
asset closure; `cloud_worker.py` installs the supervisor, watches eviction events,
backs up generations and resumes the trainer with explicit cross-device RNG
migration. Account setup and actual hardware qualification are still required.

## Implementation sequence and acceptance

| Chunk | Deliverable |
| --- | --- |
| 1 | Bootstrap script and dependency adaptation (implemented); qualify on one rented GPU after authorization, then pin the working image digest. |
| 2 | Durable checkpoint protocol, asset closure/staging, and explicit RNG migration implemented and CPU/mock tested. |
| 3 | Azure provisioning, retained disk, systemd worker/lock, eviction watcher, retries/session reservations and pause intent implemented; cloud qualification pending. |
| 4 | Delta log mirror and dashboard remote activity implemented; real interruption/recovery exercise pending. |

Use one model-loaded process at a time during local development. Do not run a
second qualification model next to the live local trainer. Cloud tests require
explicit provisioning/upload authorization and a capped budget.

Before relying on unattended runs, demonstrate:

- Forced worker termination during a step and during checkpoint publication,
  recovery of the newest complete generation, and no partial checkpoint accepted.
- VM loss/deallocation, disk survival, replacement, and no duplicate writer.
- Preserved optimizer counters/moments, recipe, split, assistance, solved bucket,
  and same-backend RNG; explicitly audited first cross-backend reseed.
- Laptop/controller restart or sleep without duplicate GPU allocation; paused
  experiments remain paused; exhausted Spot capacity is visible rather than hidden.
- Local dashboard history remains accessible throughout an outage, liveness
  becomes stale/waiting, and subsequent resumed steps appear without resetting
  history or misreporting a stopped worker as active.
- Budget enforcement, acceptable disk/checkpoint retention, and recovery from an
  object-store checkpoint if the original zonal disk cannot be reused.

Public provider behavior references:
[Azure Spot and eviction policies](https://learn.microsoft.com/en-us/azure/virtual-machines/spot-vms),
[AWS interruption notices](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/spot-instance-termination-notices.html),
[uv CUDA wheel selection](https://docs.astral.sh/uv/guides/integration/pytorch/).
