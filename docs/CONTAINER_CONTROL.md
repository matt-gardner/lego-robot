# Container dashboard and GPU job control

`so101.control_broker` is a host-side HTTPS capability broker, not a general
remote shell. Container clients receive neither AWS credentials, sudo access,
the Docker socket, nor permission to replace the host service implementation.

## Dashboard deployment

The host operator installs **copies** of `src/so101/control_broker.py` and
`scripts/lego-robot-control.service` in `/opt/lego-robot-control/` and
`/etc/systemd/system/`. The installed Python file and every parent directory
must be root-owned and not writable by coding users. Never run this root
service from the writable Git checkout or import that checkout into it.
Python isolated mode (`-I`) prevents user-site/PYTHONPATH imports.

Install the policy under `/etc/lego-robot-control/policy.json`; the policy,
approved specifications, and their parent directories must be root-owned,
not group/world writable. Start with `control-policy.example.json`, GPU
disabled. Generate a distinct random capability per session, place only its
SHA256 in the host policy, and put the secret in that session's **private**
container filesystem, mode 0600. Do not put capabilities on the shared `/mnt`
mount, in Git, in chat, in environment dumps, or in process arguments.
Revocation means removing that client from policy and restarting the broker.
The same client ID must not be assigned to different principals.

Install a host TLS certificate/key whose SAN matches the container-visible
host name (normally `host.docker.internal`). Distribute the public certificate
or issuing CA alongside the private capability file. The CLI verifies TLS;
it refuses plaintext HTTP. The sample unit binds loopback and is deliberately
not container-reachable until the host operator configures a Docker-bridge
binding and narrowly permits that port. Do not publish it through public nginx
or Cloudflare. Do not bind a public interface. Session network policy must
allow only the intended host endpoint.

Example from an authorized container, using its privately installed files:

```sh
PYTHONPATH=src python -m so101.control_broker call dashboard.status \
  --url https://host.docker.internal:8766 \
  --token-file ~/.config/lego-robot/control.token \
  --ca-file ~/.config/lego-robot/control-ca.crt
```

Replace `dashboard.status` with `dashboard.restart` to restart exactly
`so101-web.service`. Both actions check the actual catalog health endpoint;
failed commands return errors, not success-shaped defaults.
Reinstall the root-owned code copy and restart the broker after source changes.

## Immutable GPU jobs

The AWS CLI adapter is implemented, but **GPU provisioning is disabled until
the AWS infrastructure and runtime below are qualified**. Installing the
broker does not launch a machine, authorize training, or qualify CUDA recovery.
The earlier Azure controller remains separate and is not an AWS backend.

The operator registers a JSON specification in the root-owned `specs/`
directory. Containers submit its ID plus the SHA256 of canonical JSON
(sorted keys, compact separators). They cannot submit commands, launch
template overrides, IAM roles, or arbitrary recipes.

Each spec records:

- full Git commit and digest-pinned runtime image;
- dataset and frozen-split SHA256, exact versioned/checksummed resume state;
- frozen recipe and scoped S3 output URI;
- immutable EC2 launch-template version, AMI, instance type, bootstrap SHA256;
- maximum duration and hourly Spot rate;
- immutable SSM start/stop document names and numeric versions, and log group.

The **qualified launch template is the executable implementation of that job**.
Its bootstrap must fetch and verify the registered spec/assets, verify the
runtime image and code commit, stage the dataset/cache/checkpoint, publish logs
and recovery artifacts, and then remain **idle**. It must not clear STOP or
start a policy. The job/digest/deadline are passed as EC2 tags. Bind each
template version to the exact registered spec; qualification must demonstrate
that a mismatched job digest fails before running anything. The broker checks
template AMI, bootstrap hash, instance type, bounded one-time Spot pricing,
IMDSv2 with instance metadata tags enabled, OS-initiated shutdown set to
`terminate`, encrypted ephemeral disks, and disk-size policy before allocation.
This project does not yet include a qualified AWS worker bootstrap or
SSM documents. Do not substitute the Azure worker.

Commands:

- `gpu.submit`: provision one idle worker from an approved spec.
- `gpu.status` / `gpu.logs`: inspect only jobs owned by this capability.
- `gpu.start`: dispatch the approved start document, **only** if both the
  capability and a separate operator grant authorize that exact spec digest.
  An ambiguous start is never automatically retried.
- `gpu.stop`: request checkpoint/upload through the approved stop document,
  then terminate after acknowledgment or the bounded grace period. A hard
  deadline may prevent a final checkpoint; previous durable checkpoints must
  remain recoverable.
- `gpu.resume`: create a new attempt from a newly verified checkpoint spec;
  preserve commit, image, dataset, split, and recipe.

Requests use JSON `--body`, e.g. `{"job_id":"..."}` or
`{"spec_id":"sft-b64","spec_sha256":"..."}`. Resume also requires `parent`.
The persisted SQLite ledger reserves the full compute allowance before
allocation, allows only one nonterminal attempt, and retains reservations
after failures. Compute reservations do **not** cover S3, storage, network,
taxes, or all account spend. Those need independent AWS budget controls.

## AWS prerequisites and shutdown

Keep AWS credentials exclusively in the trusted host service through temporary
roles. Before enabling `gpu_enabled`, configure and qualify:

1. AWS CLI v2, the controller role, artifact bucket/versioning, worker role,
   log groups, pinned launch template/AMI/image, and SSM documents.
2. Explicit policy values: `broker_id`, `aws_cli`, `region` (Oregon only),
   `artifact_prefix` (S3 prefix ending `/`), `instance_types`, `amis`,
   `max_hours`, `max_usd_per_hour`, `total_compute_usd`, `max_disk_gib`,
   `checkpoint_grace_seconds` (1..600), `start_documents`, `stop_documents`,
   `log_prefix`, `schedule_group`, `shutdown_role_arn`, and initially empty
   `training_authorized_specs`. Do not invent spending limits.
3. Tag-scoped controller permissions, narrow `iam:PassRole`, and an independent
   EventBridge Scheduler role allowed to terminate only these worker instances.
   The one-time schedule must work even when this broker/CPU host is down.
4. Worker self-expiration, encrypted ephemeral disks, no inbound ports,
   interruption checkpoint/upload, and durable recovery before terminating.
5. Real CUDA dry-run/recovery qualification and an explicitly authorized,
   bounded paid lifecycle test before claiming the GPU path operational.

### Proposed AWS resource contract

These are **proposed names**, agreed for scoped operator inventory, not a
deployment manifest or evidence of resource existence, permissions, or runtime
qualification. Resolve inventory collisions before creating resources. Keep GPU
jobs disabled until the prerequisites above are satisfied.

| Resource or policy field | Proposed value |
| --- | --- |
| Region | `us-west-2` |
| Worker IAM role and instance profile | `so101-gpu-worker` |
| EC2 launch-template name | `so101-gpu-worker` |
| Dedicated worker security-group name | `so101-gpu-worker` (no inbound rules) |
| SSM start document | `so101-worker-start` |
| SSM checkpoint-stop document | `so101-worker-checkpoint-stop` |
| Scheduler group | `so101-worker` |
| Scheduler termination role | `so101-worker-shutdown` |
| CloudWatch log group / policy `log_prefix` | `/so101/worker` / `/so101/` |
| Log stream | Exact broker job ID |
| Artifact bucket | `so101-training-133961745003-us-west-2` |
| Input object prefix | `inputs/lego-robot/` |
| Recovery/output object prefix | `runs/lego-robot/` |
| Policy `artifact_prefix` | `s3://so101-training-133961745003-us-west-2/runs/lego-robot/` |
| Attempt output prefix | `runs/lego-robot/<job-id>/` |

The bucket name comes from the operator-reported existing worker IAM policy;
bucket existence, ownership, encryption, and versioning still need inventory.
Versioned inputs and recovery require narrowly scoped `GetObjectVersion`, not
just `GetObject`. Worker SSM-agent and log publishing permissions also need
verification; naming agreement grants none of them.

The existing internal broker ID remains `lego-robot`; schedule names remain
`so101-<job-id>`. Do not reuse or modify the shared CPU security group for GPU
workers. AMI ID, CUDA image digest, subnet/security-group IDs, actual launch
template ID and numeric version, and SSM document numeric versions are
**unselected**. Pin them only after inventory and qualification; never assume
version `1`, `$Latest`, or a deployed ARN from a proposed name.

### Offline AWS worker hard lease asset

`src/so101/aws_worker_guard.py` and
`scripts/lego-robot-worker-guard.service` are source assets, not an installed
or qualified worker. The operator must install a root-owned, non-writable copy
at `/opt/lego-robot-worker/aws_worker_guard.py` and the unit on the **GPU
worker**, never the CPU controller. Enable it before staging or training.
Do not run the guard on this coding host: it requests OS poweroff.

The stdlib-only guard uses proxy-free IMDSv2 to read `so101-deadline` and the
instance launch time. It expires at the earlier of the tagged deadline or
launch plus twelve hours; restarting cannot renew that bound. After capture,
both wall and monotonic clocks enforce the limit, so clock rollback cannot
extend it. Missing/invalid/unavailable metadata fails closed to poweroff.
Shutdown errors propagate and the unit retries. Metadata requests are bounded;
the polling interval is five seconds. This is a hard backstop, not a promise
of exact-second termination or successful AWS termination.

The immutable template must expose metadata tags and use
`InstanceInitiatedShutdownBehavior=terminate`; otherwise poweroff may merely
stop a billable-storage instance. The broker now rejects either missing
setting before launch. The worker identity must not permit deadline-tag
changes. Keep the independent controller-created termination schedule.
This asset does not yet implement Spot interruption checkpoint/upload,
bootstrap staging, trainer start/stop, or S3 recovery. Those and live shutdown
qualification remain prerequisites; do not enable GPU jobs from offline tests.

### Controller identity and credential refresh

The broker must not use the CLI's default credential chain. GPU-enabled policy
requires `aws_profile`, `aws_config`, and `controller_role_arn` as well as an
absolute `aws_cli` path. For this deployment, use profile `so101-controller`,
config `/etc/lego-robot-control/aws-config`, and role
`arn:aws:iam::133961745003:role/so101-controller`.

The config and all parent directories must be root-owned and not group/world
writable. Its complete contents are:

```ini
[profile so101-controller]
role_arn = arn:aws:iam::133961745003:role/so101-controller
credential_source = Ec2InstanceMetadata
region = us-west-2
```

This keeps temporary credential acquisition and refresh inside the trusted host
CLI. The broker supplies an isolated environment, disables shared credential
files and automatic API retries, and rejects additional profiles, credential
processes, source profiles, or endpoint overrides. The CLI's temporary role cache
is confined to the root-private `/var/lib/lego-robot-control/.aws/` directory,
which remains writable under the systemd sandbox. It verifies the actual STS
account and assumed controller role before opening the GPU ledger or serving
requests. Dashboard-only startup remains independent of AWS.

After installing AWS CLI v2 and the root-owned config/policy fields, run:

```sh
sudo /usr/bin/python3 -I /opt/lego-robot-control/control_broker.py preflight \
  --policy /etc/lego-robot-control/policy.json
```

This performs only STS identity verification; it neither enables GPU operations
nor creates a ledger or paid resources. Passing it proves authentication, not
launch permissions, runtime qualification, budget controls, or CUDA recovery.
Keep `gpu_enabled=false` until every other prerequisite above passes.

### Independent termination

The broker installs a one-time independent AWS termination schedule after
allocation; if this fails it requests immediate termination and keeps the
attempt in `recovery-required`. Its own guard reconciles ambiguous launches by
durable tags and enforces checkpoint/lease deadlines. A crash between allocation
and schedule creation is another reason worker self-expiration must be
qualified before enabling provisioning. Failed recovery remains visible in
the host journal and blocks another allocation; never delete the ledger to
bypass it. Unresolved resource ownership/termination requires host inspection.

The migrated trial remains stopped at update 304. MPS RNG is not CUDA RNG;
cross-device recovery is not bitwise-identical and must be qualified without
silently changing the recipe.
