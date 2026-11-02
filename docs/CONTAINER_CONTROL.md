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
IMDSv2, encrypted ephemeral disks, and disk-size policy before allocation.
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
