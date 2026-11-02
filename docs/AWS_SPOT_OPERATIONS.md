# AWS Spot GPU training: explanations and operations

For the short list of your prerequisites and copy-paste Copilot prompts, use
[AWS_SPOT_SETUP.md](AWS_SPOT_SETUP.md).

**Recommended starting point:** one L4 `g6.2xlarge` or A10G `g5.2xlarge` in
Oregon (`us-west-2`), encrypted gp3 data storage, and S3 recovery checkpoints.
Support multiple Oregon Availability Zones before adding Ohio (`us-east-2`).
Do not pay for two concurrent machines merely to have a fallback.

This is a setup guide, not evidence of a deployed or qualified AWS system.
The existing `so101.cloud_training` controller is Azure-oriented. The Linux
bootstrap is reusable, but AWS provisioning, budgets/leases, interruption
handling, checkpoint upload, and automatic replacement need implementation and
real qualification. Nothing here changes the current Mac experiment.

## 1. Choosing a GPU for this workload

Linux Spot examples retrieved November 1, 2026 from public price trackers:

| Instance | GPU / system RAM | CPUs | Oregon | Ohio | Virginia |
| --- | --- | --- | --- | --- | --- |
| `g6.2xlarge` | L4 24 GB / 32 GB | 8 | $0.430/h | $0.478/h | $0.834/h |
| `g5.2xlarge` | A10G 24 GB / 32 GB | 8 | $0.455/h | $0.642/h | $0.601/h |
| `g6e.xlarge` | L40S 48 GB / 32 GB | 4 | $1.850/h | $1.675/h | $1.362/h |

These are whole-VM prices, not per-GPU add-ons. Actual prices vary by
Availability Zone and time, and published prices do not guarantee capacity.
Refresh them before approving deployment. `us-east-1` is Virginia;
`us-east-2` is Ohio, not another name for the same region.

Our effective batch128 already works on MPS with activation-checkpointed
microbatches16 under an 18.25 GiB allocation cap. That makes 24 GB dedicated
VRAM a credible CUDA candidate, not a demonstrated fit. Keep microbatching;
native batch128 previously ran out of memory. GPU memory and CPU memory are
separate on these VMs.

Prefer the 32 GB system-RAM sizes to the 16 GB xlarge variants: initialization,
CPU checkpoint loading, dataset buffers, and MuJoCo also need memory. Measure
resident RAM and peak VRAM during real training and evaluation. Our Linux
bootstrap requires BF16 support, so the older T4 is not an appropriate drop-in
budget choice with this setup.

Compare **cost per completed experiment**, including initialization, evaluation,
checkpoint writes, interruptions, and repeated work. An L4 and A10G at similar
prices need a workload benchmark, not an inference benchmark or theoretical
FLOPS comparison. RL with many serial simulation rollouts may be CPU-bound.
The A100 remains a higher-headroom alternative, but its speed advantage on our
exact workload has not been measured.

You can inspect current price-history entries with AWS CLI v2:

```sh
aws ec2 describe-spot-price-history \
  --profile YOUR_PROFILE --region us-west-2 \
  --instance-types g6.2xlarge g5.2xlarge \
  --product-descriptions "Linux/UNIX" \
  --start-time "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --query 'SpotPriceHistory[].{AZ:AvailabilityZone,Type:InstanceType,USDPerHour:SpotPrice,Effective:Timestamp}' \
  --output table
```

Repeat with `us-east-2` or `us-east-1` to compare. These are price records, not
an allocation test.

## 2. Account, quota, permissions, and budget

If you do not know whether you have an AWS account, start at step 1 of the
[short checklist](AWS_SPOT_SETUP.md). It explains finding an account, creating
one, choosing the Paid account plan with free Basic support, enabling MFA,
and setting billing alerts before any CLI or GPU configuration.

For a new account, this guide uses
[standard account signup, labeled "advanced" by AWS](https://docs.aws.amazon.com/accounts/latest/reference/getting-started.html).
That label means full account/region control, not that you must already know
AWS. The [simplified new signup](https://docs.aws.amazon.com/accounts/latest/reference/sign-up-for-aws.html)
has different region/access restrictions and project spend-limit features.
Do not assume those features apply to this standard-account setup.

Use MFA and short-lived console/CLI credentials, preferably your organization's
IAM Identity Center/SSO profile. Do not copy passwords or access keys to a VM
or put them in config, chat, logs, or source control.

SSO is not assumed for a personal account. Establish an appropriate non-root
identity and supported short-lived login first; Copilot can guide the console
steps and local CLI setup. You enter payment details, passwords, and MFA in
AWS's own pages, never in chat. Account creation and quota requests do not
launch a GPU.

For a non-root IAM user or federated console role, AWS CLI v2.32 or newer
supports browser login with temporary credentials:

```sh
aws login --profile so101 --region us-west-2
```

The identity needs the AWS-managed `SignInLocalDevelopmentAccess` policy;
have its administrator grant that access if needed. Choose the non-root
identity in the browser, and enter passwords/MFA only there.
IAM Identity Center users should use `aws configure sso --profile so101`
and `aws sso login --profile so101` instead.

That browser-login flow is for human administration, not unattended workers.
Do not create console users or access keys just to identify coding sessions.
AWS-hosted workloads should use IAM roles and automatically refreshed
temporary credentials.

### CPU agent host and GPU worker

The role scaffolding assumes an EC2 CPU host and a separate EC2 GPU worker:

| Role | Binding | Current permissions |
| --- | --- | --- |
| `dockyard-host` | CPU instance profile | Assume only `so101-controller`; Session Manager agent access |
| `so101-controller` | Assumed by the trusted host service | Oregon quota/compute preflight and public price reads |
| `so101-gpu-worker` | GPU instance profile | Read planned project-bucket inputs/recovery and publish run artifacts |
| `dockyard-provisioner` | CloudFormation service role | Approved Oregon `r6a.large`, encrypted 20 GiB root / 80 GiB data, and CPU role binding |
| `dockyard-machine-access` | Mac certificate via Roles Anywhere | Trusted SSM shell/commands on the CPU instance; no direct provisioning grants |

The controller deliberately cannot launch instances yet. The CPU host is approved
and provisioned separately. GPU deployment permissions, the project bucket, and
GPU/storage spending still need a reviewed qualification plan. The controller
credential broker has not yet been implemented.

The CPU host runs coding containers and job coordination. Keep the actual
training/data loading/evaluation process on the GPU VM rather than transferring
every minibatch across the network. Start lean: the approved `r6a.large` has
2 vCPUs and 16 GiB RAM, with a 12 GiB aggregate Docker container pool.
Move to 32 GiB if measured concurrency/peaks require it; 64 GiB is not justified
by the initial workload. Set per-container limits and bound concurrent sessions. Budget its
compute, Docker/cache storage, and recovery separately from GPU hours.
An always-on CPU host can dominate the monthly bill.

Docker containers do not automatically provide separate AWS identities or a
strong boundary against hostile code. Ordinary coding containers must not
reach IMDS or receive the host Docker socket, privileged mode, or host networking.
A trusted host service should assume the controller role and broker only
authorized, budget-checked job operations for the coordinating session.
Credential refresh, container isolation, durable ownership, and an independent
GPU shutdown backstop must be implemented and qualified before unattended use.
If the CPU host is outside AWS, replace its EC2 authentication with reviewed
workload federation or IAM Roles Anywhere.

The CPU template generator is `scripts/aws_cpu_host_template.py`; it embeds
`scripts/setup_dockyard_cpu.sh`. It uses a Canonical Ubuntu 24.04 x86 image,
the existing public subnet, no inbound security-group rules, IMDSv2, and
Session Manager rather than SSH keys. Docker/containerd data and session
workspaces live on the encrypted 80 GiB volume at `/mnt/dockyard`.
The bootstrap matches the EBS volume serial, rejects the OS disk and existing
signatures before initialization, mounts by UUID, and requires the mount before
Docker/containerd start. Ordinary bridge containers are blocked from IMDS;
host-network/privileged containers and Docker-socket exposure remain prohibited.
This prepares the server, not the Dockyard application or credential broker.
The deployed CPU stack is `dockyard-cpu` in Oregon. A real instance stop/start
verified that session files, the Docker image cache, the data mount, and the
container memory/IMDS rules survive. Use the instance ID with Session Manager;
its auto-assigned public IP changes after stop/start.

EBS is separately priced by provisioned capacity, not used bytes, and persists
while the instance is stopped.
The 20 GiB root plus 80 GiB data cost about $8/month at $0.08/GB-month.
At Oregon's verified Linux On-Demand rate of $0.1134/hour, 730 running hours
plus one $0.005/hour public IPv4 total about $94.43/month including those disks.
Internet egress, snapshots, and GPU resources are additional.
Stopping the CPU removes compute charges and releases its auto-assigned public
IP, but retains both disks. Termination deletes the disposable root; the data
volume is retained, including on stack deletion/replacement, and continues
costing about $6.40/month. Reattach it only in the same Availability Zone.
Persistence is not a backup: snapshots have separate stored-data charges and
are not enabled automatically.

### Unattended access from the Mac

The local AWS profile `dockyard-machine` uses IAM Roles Anywhere rather than a
root login, human SSO session, or IAM access key. A local certificate authenticates
to the `dockyard-machine-access` role; the AWS credential helper obtains one-hour
credentials on demand. SDK credential providers refresh them automatically.
The machine certificate expires on November 1, 2027; certificate renewal is
not automatic and must happen before that date.

On the configured Mac, open an interactive session with:

```sh
~/.aws/dockyard-machine/bin/dockyard-ssh
```

Despite its name, this wrapper uses SSM, not SSH. It connects only to
`i-0860eb23354785a2c` in `us-west-2`, using the dedicated
`dockyard-machine-shell` session document. The shell runs as `ubuntu`, starts
in `/mnt/dockyard/sessions`, and supports `sudo`. Sessions have a 20-minute
idle timeout and a 60-minute maximum duration; reconnecting needs no browser
login. No inbound port or SSH key is required.

For automation without an interactive terminal:

```sh
aws ssm send-command --profile dockyard-machine --region us-west-2 \
  --instance-ids i-0860eb23354785a2c \
  --document-name AWS-RunShellScript \
  --parameters 'commands=["id","findmnt /mnt/dockyard"]' \
  --query Command.CommandId --output text

aws ssm get-command-invocation --profile dockyard-machine --region us-west-2 \
  --instance-id i-0860eb23354785a2c --command-id <command-id>
```

Poll the command invocation until it is terminal and require `Status=Success`
and `ResponseCode=0`; command acceptance alone does not mean execution succeeded.
Run Command executes as Linux root. This is trusted host-operator access, not
an untrusted coding-container sandbox. Linux root is distinct from the AWS
account root identity. The access role grants SSM shell/command execution only
on this instance, command-result retrieval (which SSM cannot resource-scope),
and named-session management; it does not grant EC2 provisioning, IAM changes,
or role assumption. A host administrator can still access the host's own EC2
identity, so this is not isolation from that identity.

The helper, plugin, public identity metadata, and PKI material are under
`~/.aws/dockyard-machine`, outside the project. Its directory is owner-only and
private keys are mode 600. The CA private key is also retained there for renewal.
Do not copy those keys into repositories, prompts, containers, or other agents'
workspaces. The profile works for processes running as this Mac's OS user;
containers/other machines do not automatically inherit it. A separate
host-side credential broker or separately provisioned workload identity is
required for those environments; that broker is not installed.

To revoke new credential issuance, an authorized administrator can disable the
Roles Anywhere profile or trust anchor. Their ARNs and certificate expiry are
recorded in `~/.aws/dockyard-machine/identity.json`. Already-issued credentials
can remain valid for up to one hour; urgent revocation also requires denying
the role's active sessions. For renewal, issue a replacement leaf certificate
with subject CN `dockyard-mac` from the existing CA, verify it, replace the local
certificate/key atomically, and verify a fresh `sts get-caller-identity`.
There is no AWS Private CA deployment or associated Private CA subscription.

In Service Quotas -> Amazon EC2, select Oregon and request at least **8 vCPUs**
for **All G and VT Spot Instance Requests**. AWS documents a default of zero.
The quota counts running GPUs and pending requests by vCPU, not GPU count.
Quota is regional; repeat for Ohio if supporting it. On-Demand G/VT and
P-family A100 quotas are separate. Quota approval is not a capacity reservation.

The deployment identity needs permission for the narrowly scoped EC2,
networking, EBS, S3, and IAM resources in the reviewed plan. An organization
may need to create the roles for you. The VM should use an instance role with
only its required bucket/prefix access and any explicitly scoped shutdown
permissions, not administrator access or stored AWS keys.

Create AWS Budget email alerts and tag every resource with project/run IDs.
Alerts are delayed and are **not a hard spending cap**. Before deployment,
agree on a Spot price ceiling, maximum session duration, and total GPU-hour
allowance that includes bootstrap, validation, recovery tests, and retries.

The AWS implementation must enforce those limits, persist consumed allowance,
stop compute on completion/failure, and prevent concurrent writers. A reboot
or replacement must not reset the allowance. Use a VM-local lease plus an
independent cloud-side shutdown backstop; test them. API failures can still
delay shutdown, so this is not a guaranteed account-wide dollar cap.

No automatic On-Demand fallback. A price ceiling can reduce capacity and cause
price-related interruptions; refusing unaffordable capacity is intentional.
Do not treat an omitted price ceiling or the On-Demand rate as a spending cap.

## 3. Storage, networking, and the full bill

The intact 2,143-episode corpus measured on November 1, 2026 contains about
**5.66 GB / 5.27 GiB across 315,364 files**. This includes the frozen40 sources;
training excludes them. The active full training-state file is about 1.24 GB;
the exported model is about 1.11 GB. Packages, pretrained caches, staging copies,
and experiment histories require additional space.

Start with **40 GiB root gp3 plus 100 GiB data gp3**, if the chosen GPU AMI
permits that root size. Check its minimum before approval. Put the workspace,
environment/cache, and outputs on the separate data volume.

US-region planning rates, excluding taxes and discounts:

| Item | Rate | Example |
| --- | --- | --- |
| gp3 EBS storage | $0.08 per provisioned GB-month | 140 GB total: $11.20/month |
| S3 Standard, first storage tier | $0.023 per stored GB-month | 20 GB: $0.46/month |
| Standard EBS snapshots | About $0.05 per stored GB-month | 20 GB billable snapshot data: $1/month |
| Public IPv4 | $0.005 per address-hour | 100 running hours: $0.50 |
| S3 Standard PUT/COPY/POST/LIST | Typically $0.005 per 1,000 requests | 315,364 individual PUTs: about $1.58 |
| S3 Standard GET | Typically $0.0004 per 1,000 requests | Archive/checkpoint requests are small |
| Same-region EC2 <-> S3 transfer | No standard data-transfer charge | NAT processing can still add charges |
| West <-> East transfer | Plan roughly $0.02/GB; verify service/direction | 20 GB transfer: about $0.40 |

Check regional prices in the AWS calculator before spending. EC2 quotes here
are tracker examples; storage/network rates are planning inputs, not a
guaranteed future bill.

EBS is billed for provisioned capacity until deleted, including detached or
stopped-VM disks. S3 is billed for stored bytes, including noncurrent versions.
Snapshots are incremental but retained changed blocks continue billing.
Versioning and snapshots are not free substitutes for retention rules.

Default gp3 includes 3,000 IOPS and 125 MiB/s. Extra provisioned IOPS or
throughput cost more. Start at the baseline, then measure disk throughput and
CPU preprocessing stalls. EFS/FSx is unnecessary for this single-worker setup.

For 100 running hours with 140 GB EBS and 20 GB S3 retained for a full month:

| Oregon configuration | Compute | IPv4 | EBS | S3 | Approximate total |
| --- | --- | --- | --- | --- | --- |
| L4 `g6.2xlarge` | $43.00 | $0.50 | $11.20 | $0.46 | **$55.16** |
| A10G `g5.2xlarge` | $45.50 | $0.50 | $11.20 | $0.46 | **$57.66** |

These exclude snapshots, requests, transfer, image fees, taxes, and any running
hours beyond the assumed 100. Initialization, downloads, evaluation, and idle
time are compute time too. Storage charges accrue even with no running GPU.

Uploads from the laptop are generally free inbound. Internet downloads can
incur outbound transfer charges after the applicable account-wide allowance;
do not assume every run gets a separate allowance. Cross-region replication
also incurs destination storage and request charges, not just transfer.

### Avoid an unnecessary NAT Gateway

A typical US NAT Gateway costs $0.045/hour plus $0.045/GB processed, in
addition to applicable transfer charges. At 730 hours, its base charge is
about $32.85/month even without traffic. It and its public IP can keep billing
after the GPU is stopped.

For the initial single-machine setup, a public subnet with an Internet Gateway,
one public IPv4, and SSH restricted to your IP is simpler and cheaper. Keep
IMDSv2 required and the dashboard bound to localhost. Do not open port8765 to
the internet.

An S3 **gateway** endpoint has no hourly or processing fee. A private-subnet
Session Manager design is possible, but NAT or **interface** endpoints may
add costs; inventory them rather than assuming all endpoints are free.

## 4. Launch and bootstrap

First review the explicit billable-resource plan. Select an AWS-published
Ubuntu 24.04 x86_64 GPU image with a pinned AMI ID and NVIDIA driver >=580.
Our existing bootstrap installs CUDA 13.0 wheels and rejects older drivers.
Verify the image's provenance, software fees, and root-volume minimum.
Do not assume every Deep Learning AMI meets these requirements.

Launch one allowed Spot type in Oregon. For a single launch, pick a qualified
zone; for multi-zone Fleet support, use price-capacity-optimized allocation
and both qualified instance types. Keep one active trainer, no On-Demand
fallback, and the reviewed price/session/hour limits.

Create an encrypted gp3 data EBS volume in the instance's Availability Zone.
Set **Delete on termination = false** for that volume. The disposable root
volume can remain true. Identify the correct device before formatting; never
format a retained disk containing a run. Mount it at `/mnt/training`, with
permissions for the training user and a persistent UUID-based mount entry.
The bootstrap verifies this is a separate mounted filesystem.

Transfer the following, preserving workspace-relative paths:

- Project source, `pyproject.toml`, `uv.lock`, and required scripts; not the Mac
  virtual environment, credentials, or unrelated artifacts.
- The collection manifest and all source episode directories it references.
  The manifest lives in `data/simulation/entity_commands_with_difficulty_v1`,
  but the referenced episodes also live in other corpus directories.
- `data/simulation/model` and its simulator assets.
- The exact frozen evaluation split at
  `outputs/entity_smolvla_trajectory_rl_all_assisted_lr1e4_before_step243_rerun/data_split.json`.

Archive the selected dataset instead of restoring 315,000 S3 objects one by
one. Do not copy all historical corpora or nominal corrupted datasets.
Verify checksums and the exact catalog/split before training. Keep a reproducible
source archive/version and the pretrained-base revision/cache required by
recovery checkpoints.

On the VM, from the copied workspace:

```sh
cd /mnt/training/so101
bash scripts/setup_cloud_gpu.sh --storage /mnt/training
source /mnt/training/so101-cloud/training.env
cd "$SO101_ROOT"
"$SO101_PYTHON" -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'
```

The script builds an isolated Python 3.12 CUDA environment and verifies imports,
CUDA gradients, BF16 support, and headless MuJoCo EGL rendering without loading
a large policy. It does not provision AWS resources or implement recovery.

**Do not use plain `uv run` or `uv sync` for this Linux GPU environment.**
The root project selects CPU Torch wheels on Linux. Source `training.env` and
use `$SO101_PYTHON`. Never copy or overwrite the Mac `.venv`.

## 5. New CUDA SFT recipe

This is an example for a **new qualified AWS run**, not an instruction to launch
another experiment now:

```sh
"$SO101_PYTHON" -u -m so101.train_entity_sim \
  --collection data/simulation/entity_commands_with_difficulty_v1 \
  --output outputs/entity_smolvla_sft_aws_batch128_1000 \
  --pretrained lerobot/smolvla_base \
  --device cuda --steps 1000 --batch-size 128 \
  --history-steps 4 --depth-mode known \
  --learning-rate 1e-5 --learning-rate-schedule triangular \
  --warmup-fraction .1 --seed 61000 \
  --validation-split outputs/entity_smolvla_trajectory_rl_all_assisted_lr1e4_before_step243_rerun/data_split.json \
  --validation-interval 50 --validation-seed 961000 \
  --training-state outputs/entity_smolvla_sft_aws_batch128_1000/training_state.pt \
  --checkpoint-interval 10 --obstacle-input-end-probability 1
```

This preserves effective batch128/microbatch16, peak1e-5 at step100, decay to
zero at1000, frozen40 evaluation every50, and the original learning seed.
The proposed checkpoint interval10 is for the new run and must remain the
same when resuming it. It does not modify the Mac run's interval25.

Use a persistent process supervisor with durable stdout logs. It should keep
the trainer alive across SSH disconnects, but must honor lease expiry,
completion, failure, and STOP. A service that blindly restarts a completed
trainer is not correct lifecycle management.

Before a long run, qualify actual batch128 updates, memory usage, finite
gradients/Adam state, and the frozen evaluation. Measure full workload timing.
Do not load a second policy while the first is active.

**MPS -> CUDA is not an exact supported resume by merely changing `--device`.**
The trainer's strict saved configuration includes the device, and its accelerator
RNG state is backend-specific. Use a distinct fresh CUDA experiment unless
migration is explicitly implemented and approved. Do not bypass the check or
silently reset optimizer/RNG state.

For the dashboard, start it on the VM using `$SO101_PYTHON`:

```sh
"$SO101_PYTHON" -m so101.rl_dashboard --port 8765
```

Keep its localhost binding. From the laptop:

```sh
ssh -L 8766:127.0.0.1:8765 ubuntu@YOUR_VM
```

Open `http://127.0.0.1:8766`; the alternate local port leaves the Mac dashboard's
8765 available. Use a verified SSH host key, not disabled host checking.

## 6. What survives an interruption

| Storage | Survives instance termination? | Recovery limitation |
| --- | --- | --- |
| Root EBS with delete-on-termination true | No | Treat as disposable |
| Retained data EBS | Yes | Attaches only within its Availability Zone |
| Local instance-store SSD | No | Cache/scratch only |
| S3 backup | Yes | Restore into any supported zone/region |
| EBS snapshot | Yes | New volume; copy snapshot for another region |

Persistent EBS is useful but not sufficient for flexible Spot recovery.
If the original zone has no GPU capacity, another zone cannot directly attach
that disk. Use S3 checkpoints or create a new volume from a snapshot.

The trainer's atomic `training_state.pt` contains trainable weights, Adam state,
RNG, and embedded history. Frozen weights are reconstructed from the pretrained
base, so preserve its exact identity. `model.pt` alone is not a full resume.

The AWS uploader must stage a stable copy of a completed atomic checkpoint,
upload required files to a unique prefix, verify checksums, and only then publish
a latest-complete pointer. The checkpoint's embedded log is the authoritative
state for that optimizer step. Blindly synchronizing a changing output directory
can mix checkpoint and metadata versions.

Publish periodically, keep a bounded recovery history, and enforce cleanup.
Unbounded versioning of a 1.24 GB file can dwarf the small corpus's storage bill.
Do not rely only on snapshots or a final shutdown upload.

AWS interruption notices are best effort, normally two minutes before stop or
termination; hibernation has different warning behavior. Monitor via IMDSv2
metadata or EventBridge; AWS recommends checking metadata every five seconds.
On warning, create the trainer's STOP marker, attempt a graceful save, and
upload the latest complete state. Validation may outlast the warning window.

Periodic backups remain the safety net even with no warning. At our observed
Mac speed, interval25 can lose about 13-15 minutes of training plus upload lag.
Interval10 reduces repeated work but increases checkpoint I/O. Measure CUDA
timings and choose the cadence before starting a run.

For compatible CUDA recovery, restore all required inputs and the complete
checkpoint at the same relative paths, clear an intentional STOP marker, and
repeat the original command. The existing trainer automatically resumes an
existing `--training-state`; its recipe must match.

Do not trust unattended Spot until these real tests pass:

1. CUDA/EGL and actual effective batch128 training/evaluation fit.
2. Stop/save/upload/restore produces the next optimizer step with continuous
   history and finite model/Adam state.
3. Termination/replacement recovers from S3 without the original root disk.
4. Price/session/total-hour limits and one-worker protection survive restart,
   failure, and recovery without resetting allowance or buying On-Demand.
5. Completion/failure cancels replacement requests and stops billable compute.

## 7. Multiple zones and an eastern fallback

Start in one Oregon zone with retained EBS, then add other Oregon zones with S3
restore. EC2 Fleet's recommended price-capacity-optimized strategy balances
price with capacity; cheapest-only placement can lead to more repeated work.
Include only GPU types that passed workload qualification.

Ohio offers a separate capacity pool and its L4 example is closer in price
to Oregon than Virginia's. Add it only if availability or interruption costs
justify the extra operational work.

| Requirement | Multiple Oregon zones | Oregon + Ohio |
| --- | --- | --- |
| Quota | One regional quota | Separate approval per region |
| S3 assets | One regional bucket | Copy/restore across regions or another bucket |
| Retained EBS | Separate zone-specific volumes or restore | Regional storage/restore; no direct attachment |
| AMI/network/security config | Per-zone subnets | Region-specific AMIs, VPCs, subnets, security groups |
| Transfer | Same-region EC2/S3 has no standard transfer fee | Cross-region transfer and replication costs |
| Worker ownership | One active writer | One active writer across both regions |

A regional Fleet does not span West and East. Cross-region fallback needs a
controller with reliable shared ownership/allowance state; do not run two
independent auto-restart loops against the same experiment.

A second-region S3 corpus copy adds roughly $0.13/month for its 5.66 GB, plus
transfer and requests. The operational complexity matters more than that
storage cost. Avoid idle GPU standby, duplicate large EBS caches, and duplicated
NAT/interface-endpoint infrastructure.

## 8. Finish without leaving a bill running

Save and verify the final recovery/export artifacts, then cancel persistent
Spot/Fleet requests that could replace a terminated worker. Stop or terminate
GPU compute and confirm no replacement starts.

Inventory retained data and root volumes, snapshots, S3 versions, public IPs,
NAT Gateways, and interface endpoints. Release unused Elastic IPs and delete
unneeded infrastructure only after explicit approval. Never delete the only
recoverable model, optimizer state, or dataset copy as routine cleanup.

Retained storage continues billing after the training task ends. Report its
ongoing monthly cost separately from completed GPU-hour charges.

## References

- [AWS CLI temporary browser login](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-sign-in.html)
- [AWS GPU Spot quotas](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-spot-limits.html)
- [AWS EBS pricing](https://aws.amazon.com/ebs/pricing/)
- [AWS S3 pricing](https://aws.amazon.com/s3/pricing/)
- [AWS VPC, IPv4, NAT pricing](https://aws.amazon.com/vpc/pricing/)
- [AWS EC2 transfer/pricing](https://aws.amazon.com/ec2/pricing/on-demand/)
- [Spot interruption notices](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/spot-instance-termination-notices.html)
- [Fleet allocation strategies](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-fleet-allocation-strategy.html)
- [AWS Pricing Calculator](https://calculator.aws/)
- [L4 tracker quotes](https://www.devzero.io/instances/aws/g6.2xlarge)
- [A10G tracker quotes](https://www.devzero.io/instances/aws/g5.2xlarge)
- [L40S tracker quotes](https://www.devzero.io/instances/aws/g6e.xlarge)
