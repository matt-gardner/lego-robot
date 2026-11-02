# Running this experiment on an Azure Spot GPU

The first implementation supports **Azure**, one GPU and one retained disk in one
zone. AWS account preparation is described at the end, but the controller does
not implement AWS. You do not need both accounts.

**Nothing has been deployed or uploaded.** Local CPU/mock tests cover checkpoints,
recipe preservation, budgets and dashboard liveness. An actual cloud GPU,
driver/CUDA/EGL rendering, deployment and real eviction recovery still need the
qualification exercise below. Do not trust this unattended until that passes.
The existing Mac trainer is not moved or stopped by installing these files.

## 1. Account, billing and quota: your steps

Create an Azure account/subscription with billing enabled. Use a dedicated
pay-as-you-go subscription or resource group for this experiment, enable MFA and
keep billing notifications on an address you monitor. Free/student/credit
subscriptions can restrict Spot/GPU access; confirm eligibility before spending
time configuring the VM.

In the Azure portal, open **Cost Management → Budgets** at subscription scope.
Create a monthly budget you are comfortable with, for example $100, with actual
alerts at 50%, 80% and 100%, plus a forecast alert at 100%. Also create a smaller
daily alert if available for your offer. Enable Cost Management anomaly alerts.
Check the invoice/cost analysis after the first short session.

**These are alerts, not a hard billing cap.** Pay-as-you-go Azure subscriptions
do not have a configurable hard spending limit. Billing data/alerts are delayed;
budget automation can also be delayed. Do not assume the account will stop at
$100. A credit subscription's spending limit is a different mechanism, equal to
its available credit, not an arbitrary limit you can set.

The controller adds several narrower protections:

| Protection | What it actually covers |
| --- | --- |
| Positive Spot maximum hourly price | Azure evicts/refuses allocation above that compute price; never falls back to on-demand. |
| `max_session_hours` | Fixed lease per allocation, checked on the VM every 20 seconds, plus an Azure-side daily auto-shutdown scheduled at the lease's UTC time. Expiry pauses rather than automatically buying another session. |
| `total_gpu_hours` | Local persistent conservative reservations; each allocation attempt consumes a whole session, including failures/evictions. Failed allocations are not automatically refunded. |
| Storage deletion locks | Accidental deletion protection, **not** cost protection. Storage continues billing. |

These are not a guaranteed monetary/account cap. Bootstrapping, OS disks, managed
data disk, Blob capacity/versions/transactions, public IP, egress, taxes and any
Marketplace image fees are additional. Use an ordinary Ubuntu image rather than
a paid Marketplace image unless you explicitly accept its fees. Shutdown/API/
identity failures can delay enforcement; verify the independent guard and Azure
auto-shutdown. Never start the VM outside the controller: that bypasses reservations.
Do not delete/reset `outputs/cloud/control/<vm>/controller.json` to regain hours.
The 128 GiB Premium data disk is billed for its provisioned capacity even when
almost empty or when the VM is deallocated.

Start with `Standard_NC24ads_A100_v4` (one 80 GB A100). In **Quotas → Compute**,
select your region and request at least **24 Total Regional Spot vCPUs**. Check
that this SKU/zone is available and your subscription is permitted to use it.
`Standard_NC40ads_H100_v5` requires at least 40 Spot vCPUs and a higher price
ceiling if the current Spot price exceeds the example's $1.20/hour. GPU quota,
Spot capacity and CUDA/EGL compatibility are separate questions; none guarantees
the others. These are not reserved GPU instances.

## 2. Local tools and permissions

Run from this project directory on the Mac. Install the Azure CLI if missing,
then sign in interactively. Do not put passwords, Azure keys or tokens in config.

```sh
brew install azure-cli       # only if az is missing
az login
az account list --output table
az account set --subscription YOUR-SUBSCRIPTION-UUID
ssh-keygen -t ed25519 -f ~/.ssh/so101-cloud
cp cloud/azure.example.json cloud/azure.local.json
```

For unattended SSH, use a dedicated key loaded into your local SSH agent. Keep
the private key local; only its public half goes to Azure. The controller does
not disable host verification.

The deployer needs resource creation permissions plus role-assignment and lock
permissions in the **dedicated** resource group, e.g. Contributor plus User
Access Administrator (or an Owner performing initial deployment). An
organization's administrator may have to create the group and grant these.
Do not grant subscription-wide permissions to the VM. The template gives its
managed identity Blob Data Contributor on its backup account, and VM Contributor
only on itself for shutdown. No cloud credentials are copied to the worker.
In **Subscriptions → Resource providers**, ensure `Microsoft.Compute`,
`Microsoft.Network`, `Microsoft.Storage` and `Microsoft.DevTestLab` are registered;
the last supplies auto-shutdown. Subscription-level registration may require an
administrator even if you have resource-group deployment permissions.

Your signed-in local identity additionally needs **Storage Blob Data Contributor**
on the new storage account for asset uploads/downloads. Management-plane Owner
does not automatically grant Blob data access. Ask the administrator to assign
that role after deployment, or assign it yourself if permitted. Allow time for
RBAC propagation.

Edit `cloud/azure.local.json`:

- Fill subscription, region and zone, a dedicated resource group/VM name and a
  globally unique lowercase storage account name (3–24 letters/digits).
- Set `ssh_source_cidr` to your public IPv4 plus `/32`; never `0.0.0.0/0`.
- Set the key paths and choose a **pinned** Ubuntu 24.04 x86_64 image version.
- Review the price, session-hour and total-hour limits. Example values reserve
  at most 20 GPU-hours at $1.20/hour: **$24 compute estimate**, not a $24 bill cap.
- Keep `run` pointing to the existing canonical local output. Recipe, base,
  corpus, seed and all optimization flags are read from its atomic checkpoint.

List available ordinary Ubuntu images and pick an exact version, not `latest`:

```sh
az vm image list --location westus3 --publisher Canonical \
  --offer ubuntu-24_04-lts --sku server --all --output table
az vm list-skus --location westus3 --size Standard_NC24ads_A100_v4 \
  --all --output table
```

A stock Ubuntu image **does not have the GPU driver installed**; install it in
step 4. An existing Ubuntu 24.04 GPU image with NVIDIA driver >=580 is also
acceptable if its licensing, provenance and exact version are understood.
Check current Spot prices in the portal before deployment; previously quoted
prices are not a live quote.

## 3. Review, then explicitly create paid resources

```sh
uv run --locked python -m so101.cloud_training --config cloud/azure.local.json plan
# This next command creates billable resources:
uv run --locked python -m so101.cloud_training --config cloud/azure.local.json --apply deploy
uv run --locked python -m so101.cloud_training --config cloud/azure.local.json --apply pin-host
uv run --locked python -m so101.cloud_training --config cloud/azure.local.json ssh
```

`plan` is offline. `deploy` creates a Spot/Deallocate VM, a 128 GiB retained
managed data disk at LUN 0, private versioned Blob backups, network/IP, SSH-only
NSG restricted to your `/32`, self-shutdown identity permissions and Azure
auto-shutdown. Paid mutations require `--apply`. The dashboard port is never
opened. Inspect **VM → Auto-shutdown** and the deployment result in the portal.
The session clock includes setup time; deallocate if setup is taking too long.

`pin-host` retrieves the SSH host key through authenticated Azure Run Command,
not unauthenticated `ssh-keyscan`. If an address/key changes, it refuses the
change: verify it in Run Command, archive the old controller `known_hosts`, then
explicitly repin. Do not use `StrictHostKeyChecking=no`.

Capacity/quota/image/permissions failures are errors, never on-demand fallback.
A partially failed deployment may leave storage/network resources. Check the
portal and deallocate any VM before troubleshooting; the controller attempts
deallocation on deployment failure. Preserve the budget ledger. After fixing the
cause, an explicit `deploy --apply` retry is allowed for the same unfinished
target, retaining previous reservations and consuming another session; a completed
or staged deployment cannot be redeployed. Do not create a second VM to work
around an error. Initial deployment refuses a nonempty unrelated resource group.

## 4. Prepare the new machine

In the SSH session, inspect `nvidia-smi`. For a stock Ubuntu image, follow
[Azure's Linux NVIDIA driver instructions](https://learn.microsoft.com/en-us/azure/virtual-machines/linux/n-series-driver-setup)
to install a driver **580 or newer** compatible with your GPU/kernel, and reboot.
Ubuntu's `nvidia-driver-580-server` package is one option when available in your
image's repositories; confirm availability first. Do not install an obsolete
CUDA driver merely because an older guide uses it. You do not need a system CUDA
toolkit: the Python environment installs its own CUDA 13 runtime. Secure Boot
requires correctly signed modules; use a compatible image/driver or explicitly
configure the VM accordingly. The setup script deliberately does not install
drivers or reboot for you.

Back on the Mac:

```sh
# FIRST USE ONLY: confirm in Azure this is the newly created blank LUN-0 disk.
uv run --locked python -m so101.cloud_training --config cloud/azure.local.json \
  --apply mount --initialize-empty-disk

# Later mounts/recovery must omit --initialize-empty-disk.
uv run --locked python -m so101.cloud_training --config cloud/azure.local.json --apply bootstrap
```

The mount helper refuses the OS disk, partitions, existing signatures and
unexpected filesystems; existing ext4 disks are mounted, never reformatted.
It persists the mount by UUID. The bootstrap stages code only, installs Ubuntu
libraries, uv/Python and a separate CUDA environment on the data disk, then
checks CUDA gradients, BF16, EGL and imports **without loading a policy**.
Any failure blocks migration. `bootstrap` refuses to overwrite an installed
worker; use the existing pinned workspace on recovery.

Do not use plain `uv run`/`uv sync` on the cloud VM: this repository's Linux
configuration selects CPU Torch. Use `/mnt/training/so101-cloud/venv/bin/python`
or source `/mnt/training/so101-cloud/training.env`.

## 5. Final checkpoint handoff: only when you choose to move training

Do **not** stop the Mac trainer until the previous smoke checks pass. No second
policy is loaded by bootstrap. The periodic Copilot monitor has been disabled;
keep it disabled during migration so it cannot resume a local worker.

Request a graceful local stop and wait for the trainer process to exit after
its completed checkpoint. Do not force-kill it or copy a stale model export.

```sh
touch outputs/entity_smolvla_trajectory_rl_all_assisted_lr1e4/STOP
# Wait for "stopped_at_checkpoint" in outputs/rl_all_assisted_lr1e4.stdout.log
# and confirm the local dashboard no longer shows an active trainer.

uv run --locked python -m so101.cloud_training --config cloud/azure.local.json \
  prepare --bundle outputs/cloud/final-handoff.tar.gz
uv run --locked python -m so101.cloud_training --config cloud/azure.local.json \
  --apply stage --bundle outputs/cloud/final-handoff.tar.gz
uv run --locked python -m so101.cloud_training --config cloud/azure.local.json --apply start
uv run --locked python -m so101.cloud_training --config cloud/azure.local.json status
```

Use a new bundle filename for each preparation. A live snapshot is allowed for
inspection, but first cloud start refuses it if the local atomic state has since
advanced. An unstarted, paused cloud snapshot can be restaged; a committed cloud
run cannot be overwritten with a new bundle.

Staging copies the asset closure, not the whole laptop: code/lockfile, pinned
MuJoCo assets, base checkpoint, every referenced episode's metadata/actions, frozen
split and atomic state/history. It also flattens the existing cached policy/VLM
Hub snapshots, records their commit IDs and pins their cache refs; the training
worker runs Hub/Transformers offline instead of silently following a changed
`main`. Missing cached configurations are an error. Cached model files add to
bundle size/storage; no extra policy is constructed. It excludes recorded images,
`.venv`, tokens/credentials and unrelated caches/outputs. It verifies transferred
hashes and backs the immutable bundle up to Blob before marking the worker ready.
Current metadata-only inventory found all 4,334 workspace assets (about 1.18 GiB)
and 21 selected Hub cache files (about 2.74 GiB), before compression/checkpoint
history; no policy was loaded for this inventory. CUDA dependencies need additional
space on the retained disk.

The worker continues the **same** model/Adam/log/mastery/solved-bucket state and
recipe. Only execution-specific options change: CUDA, three durable checkpoint
generations, resume, no giant final base export, and explicit device-RNG migration.
MPS random-state bytes cannot be restored as CUDA state: the first CUDA device
stream is reseeded from the existing recipe seed, recorded in
`device_rng_migrations`; the continuation is **not bit-exact across backends**.
Subsequent CUDA resumes restore CUDA RNG. No SFT, KL guard or learning-rate
changes are introduced. A local process check refuses start while the matching
Mac trainer is still active; never manually start it again while cloud training
is intended to run.

## 6. Control and local dashboard

```sh
# Run in a separate terminal; Ctrl-C stops this controller, not the remote trainer.
uv run --locked python -m so101.cloud_training --config cloud/azure.local.json --apply watch

uv run --locked python -m so101.cloud_training --config cloud/azure.local.json status
uv run --locked python -m so101.cloud_training --config cloud/azure.local.json logs
uv run --locked python -m so101.cloud_training --config cloud/azure.local.json --apply pause
# Wait until status reports paused; pause alone does not immediately stop compute.
uv run --locked python -m so101.cloud_training --config cloud/azure.local.json --apply deallocate
# Next intentional session (reserves hours if compute was deallocated):
uv run --locked python -m so101.cloud_training --config cloud/azure.local.json --apply resume
```

`watch` reconciles the existing VM, incrementally mirrors changed plot logs and
observes heartbeats every 30 seconds. On capacity eviction within an intended
running session it retries **that same** Spot VM in the disk's zone, with bounded
backoff and fresh conservative reservations. No new zone/VM/on-demand fallback.
Expiry requires an explicit new start; budget exhaustion refuses allocation.
Fatal trainer/dependency/corruption failures are reported, not endlessly restarted.
After deliberate pause, eviction or controller restart, persisted intent controls
whether automatic resumption is allowed.

Systemd and the on-disk exclusive lock keep one cloud trainer; the VM's guard and
Azure auto-shutdown remain independent of this laptop. If the laptop sleeps,
training continues within its lease; automatic allocation retries and dashboard
mirroring pause until `watch` runs again.

The existing dashboard remains **http://127.0.0.1:8765**, reading mirrored history
from `outputs/cloud/<vm>/`. Restart the dashboard once after updating these source
files if it predates cloud support. You can also launch a separate local instance:

```sh
uv run --locked python -m so101.rl_dashboard --port 8766
```

Cloud runs show worker state, last-contact age and last backed-up step. A running
counter requires both recent local contact and a fresh worker heartbeat; copied
logs alone do not make a run active. During outages, history stays accessible and
liveness becomes stale. No public remote dashboard or open port 8765 is needed.

The local data browser is **http://127.0.0.1:8765/data**. Changed cloud logs also
mirror the frozen split and available bounded `rollouts/` evidence, without
copying model weights. Source expert previews use the original local corpus.
Recorded policy imagery is separate: future workers record the first rollout
per training group (actions, states, sparse wrist frames); older logs remain
summary-only. Default retention is 20 training steps on both the persistent
worker disk and local mirror. These optional recordings are not part of the
Blob checkpoint commit; losing them does not prevent checkpoint resumption.
See README for difficulty generation and explicit corpus-refresh semantics.

## 7. Qualification before unattended use

Keep the first session short. Require an actual completed CUDA step with finite
model/Adam tensors, preserved optimizer counters and recipe, one solved group,
unchanged frozen validation and an advanced Blob backup. Inspect `worker.stdout.log`
and `/mnt/training/control/backup.json`; errors must be resolved, not ignored.
Measure actual step time before estimating cost/speed against the Mac.

Test a controller restart, deliberate pause/deallocation/resume, then a Spot
eviction (Azure's `az vm simulate-eviction` on the test VM). Check that the retained
disk survives, there is only one trainer, the newest verified generation resumes,
and the local dashboard reports the outage truthfully. Repeat with the controller
closed and confirm the VM deallocates at its lease deadline. Fatal worker kills
should produce a visible error requiring explicit resume, not automatic loops.

Local checkpoint publication fsyncs file/directory, commits checksummed manifests
and retains three generations. Recovery rejects/quarantines corrupt newer commits
and recomputes from an older verified generation. Blob uploads publish `.pt`,
`.split.json`, `.identity.json`, then the `.json` commit object **last**. Backup
failures are visible; backups may lag the managed disk. A partial upload is not a
resumable checkpoint.

## Storage recovery, shutdown and deletion

The persistent disk is the normal recovery path. Azure Deallocate retains its
VM/disk, but does not guarantee capacity. Logs and state remain at
`/mnt/training/workspace/<run>/`. The worker writes no state to temporary GPU
storage.

Blob objects are private in container `training`:

```text
<vm>/assets/<bundle-sha256>.tar.gz
<vm>/checkpoints/step_000000123.pt
<vm>/checkpoints/step_000000123.split.json
<vm>/checkpoints/step_000000123.identity.json
<vm>/checkpoints/step_000000123.json
```

If the disk is lost, use `az storage blob list/download --auth-mode login` to
retrieve the asset bundle and newest **committed** generation into a separate
recovery directory. Verify bundle SHA-256 against its object name and checkpoint
size/hash against its commit JSON. Restore its split and identity too. Use
`cloud_checkpoint.validate_state` on a CPU-loaded trusted checkpoint and verify
`split_sha256`/`identity_sha256` before resuming. Never deserialize untrusted
checkpoints. Disk-loss/cross-zone restore is **manual** in this first version:
do not rerun stage over an existing run, or resume the base/old local state.

On a separately prepared recovery disk, extract the verified asset bundle and
put the selected backup files in this layout (using their actual step number):

```text
workspace/<run>/generations/step_000000123.pt       <- checkpoint Blob
workspace/<run>/generations/step_000000123.json     <- commit Blob, not split/identity
workspace/<run>/data_split.json                   <- matching .split.json Blob
workspace/<run>/cloud_identity.json               <- matching .identity.json Blob
```

Run `cloud_checkpoint.recover(output_path)` using `SO101_PYTHON` from the restored
workspace; it verifies hashes/identity/finiteness and publishes `latest_state.pt`.
Restore `/mnt/training/control/worker.json` from the local controller's saved
copy, set intent to paused, reinstall the worker service/guard, and verify mounts
and lease before explicitly resuming. Keep the same workspace paths and source
manifest. If moving to a different VM name/account, update worker backup and
shutdown identity configuration deliberately. This is an operator recovery,
not an automatic cross-zone failover claim.

Only stopping the Python process or issuing Linux `shutdown` is insufficient to
guarantee Azure compute billing stops. Use **deallocate**, then confirm the portal
says **Stopped (deallocated)**. The `destroy-compute --apply` command deletes only
the VM, retaining training storage; deletion/recreation is an operator procedure,
not the normal Spot recovery path.

Managed disk, OS disk where retained, public IP and backups cost money while the
GPU is off. Checkpoint Blob objects/versions accumulate: monitor storage costs
and intentionally prune old committed generations after qualification, keeping a
verified recovery point and its asset bundle. Versioning/soft-delete can retain
billed bytes after deletion. This is not an unlimited free archive.

For permanent teardown, first download and verify needed results, then explicitly
remove the **data disk and storage-account deletion locks** in the portal and
delete the dedicated resource group. This is irreversible; the controller never
does it automatically. Check Cost Analysis for lingering resources afterward.

## If you prefer AWS instead

Create a dedicated AWS account with billing enabled, secure the root account with
MFA, use IAM Identity Center/short-lived login for normal work, and enable billing
notifications. Configure AWS Budgets actual/forecast alerts and Cost Anomaly
Detection. AWS Budgets updates can lag by hours; alerts and optional Budget
Actions are **not a guaranteed hard spending cap**.

For an L40S pilot choose one `g6e.2xlarge`, request enough **All G and VT Spot
Instance Requests** vCPU quota in the chosen region (at least eight for this
size), and confirm Spot availability. Use a persistent encrypted EBS data volume
with `DeleteOnTermination=false`, one fixed zone, versioned private S3 backups,
restricted SSH and an instance role rather than static access keys. Storage and
network costs persist when compute is stopped. P-family GPUs need their separate
Spot quota; do not rent an eight-A100 instance for this single-GPU trainer.

The CUDA bootstrap and checkpoint protocol are reusable, but EC2 provisioning,
IAM/IMDS shutdown/eviction and S3 backup adapters are **not implemented** here.
Do not feed AWS credentials/config to the Azure controller. If using the current
implementation now, choose Azure.

## Provider references

- [Azure spending limits](https://learn.microsoft.com/en-us/azure/cost-management-billing/manage/spending-limit)
- [Azure budgets](https://learn.microsoft.com/en-us/azure/cost-management-billing/costs/tutorial-acm-create-budgets)
- [Azure Spot policy, prices and eligibility](https://learn.microsoft.com/en-us/azure/virtual-machines/spot-vms)
- [Azure Linux NVIDIA setup](https://learn.microsoft.com/en-us/azure/virtual-machines/linux/n-series-driver-setup)
- [AWS Budgets and update delays](https://docs.aws.amazon.com/cost-management/latest/userguide/budgets-managing-costs.html)
