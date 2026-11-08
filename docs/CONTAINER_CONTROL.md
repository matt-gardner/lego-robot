# Container dashboard and GPU job control

`so101.control_broker` is a host-side HTTPS capability broker, not a general
remote shell. Container clients receive neither AWS credentials, sudo access,
the Docker socket, nor permission to replace the host service implementation.

## Automatic session access and GPU preparation

Once the host integration is installed, the saved Lego container template
automatically receives its own private capability, public CA, and
`~/.config/lego-robot/control.json` before the agent starts. Do not contact a
current agent or request a per-job grant. The installed root preset grants
routine lifecycle access; AWS quota and account budget alerts are the admission
controls, not a second software lifetime budget or inventory gate.

`gpu.catalog` lists this session's installed immutable job specifications.
`gpu.prepare` accepts only an owned job ID and `provision`, `qualify`, or
`approve`; the broker obtains all executable document versions and parameters
from root-installed `workflows.json`. Accepted SSM commands and dispatch intents
are persisted. Repeating a request observes the accepted command rather than
dispatching it twice. An uncertain response is reconciled against AWS command
history, never assumed to mean nothing happened.

Allocations accept a client-generated `request_id`; the same session/request
returns the same owned job rather than allocating twice. Failure responses
retain that job receipt and the AWS error code. `gpu.request` retrieves it
without needing a host-side handoff. The lifecycle CLI automatically tries
installed equivalent-zone alternatives only after an explicit, unallocated AWS
capacity rejection; it does not change the model, inputs, or training recipe.

For the end-to-end CUDA lifecycle check, from the repository root:

```sh
PYTHONPATH=src uv run --no-project python -m so101.gpu_smoke \
  --spec-id SPEC_ID_FROM_CATALOG --state-dir outputs/gpu-lifecycle-check
```

This allocates a real GPU, stages the registered inputs, executes CUDA/BF16/EGL
and exact-state recovery checks, then requests stop and observes actual
termination. It does **not** train a sweep cell or claim the sweep is complete.
Its receipts are retained in the chosen state directory. Re-running with the
same keyed intent recovers the same allocation; never clear that intent to
force a second machine after an uncertain response. Old unkeyed intents remain
historical evidence and are not replayed.
Automatic EC2 lease termination remains active if the client disappears.

The installation instructions and historical qualifications below are host
setup/reference material, not instructions to message an existing session.
Fresh-cell training and complete sweep exports require their own installed
workflow; passing the CUDA lifecycle check alone does not prove those ready.

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
`so101-web.service` through the legacy broker. Both actions check the actual catalog health endpoint;
failed commands return errors, not success-shaped defaults.
Reinstall the root-owned code copy and restart the broker after source changes.

The native Dockyard `lego_dashboard_restart` registration instead invokes the
root-installed `scripts/lego_dashboard_deploy.py` copy with Python isolated mode
and no arguments. It deploys current **published main** dashboard modules/assets
before restarting; it never adopts the shared checkout's dirty or unpublished
changes. The generic Dockyard server contains no LEGO-specific deployment code.

Its root-owned `/etc/lego-robot-control/dashboard-deploy.json` manifest selects
`repository` (a private bare published-source mirror), `baseline` and
`baseline_sha256` (a frozen compatible dependency tree), `releases`, `override`,
`python`, `health_url`, and the exact `paths` allowlist. The installed scope is
`rl_dashboard.py`, `training_data_browser.py`, `lego_data_browser.py`,
`part_preview.py`, `generation_order.py`, and the ten `dashboard/` assets.
Producer modules remain frozen; scope/dependency changes require an operator
manifest update. Coding containers cannot replace the helper or manifest.

Deployment pins the fetched commit, stages regular files with SHA-256 evidence,
checks imports as the dashboard user, then atomically installs a PYTHONPATH-only
systemd override. Existing flags, playback runtime, authentication/front proxy,
CPU/memory limits and read-only data mounts remain intact. The helper restarts
and checks the real `/lego-robot/api/lego` JSON API and live process source path.
Staging failures leave the current service unchanged; activation failures restore
the previous override and restart it, explicitly reporting rollback failures.
The native tool's sender-owned durable receipt prevents uncertain-call replay;
inspect `server_tool_status` instead of repeating the request. Successful
releases retain `manifest.json`; `last-success.json` records the PID/source hashes.
The separate `lego_dashboard_status` tool remains read-only.

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
Input staging downloads at most four immutable S3 objects concurrently onto the
worker's local disk. Each object retains its exact version, size and SHA-256
checks; the population receipt is committed only after every download verifies.
Per-object byte counts and elapsed times are logged. A consolidated episode
SQLite artifact is mounted directly via `--episode-storage`, with its small
original catalog mounted at the existing collection path; it must not appear
in the archive-extraction plan. The small-file dataset archives are not needed
for that loader path. Model-cache archives remain separately verified.

Its bootstrap must fetch and verify the registered spec/assets, verify the
runtime image and code commit, stage the dataset/cache/checkpoint, publish logs
and recovery artifacts, and then remain **idle**. It must not clear STOP or
start a policy. The job/digest/deadline are passed as EC2 tags. Bind each
template version to the exact registered spec; qualification must demonstrate
that a mismatched job digest fails before running anything. The broker checks
template AMI, bootstrap hash, instance type, bounded one-time Spot pricing,
IMDSv2 with instance metadata tags enabled, OS-initiated shutdown set to
`terminate`, encrypted ephemeral disks, and disk-size policy before allocation.
The AWS bootstrap and SSM documents are implemented; the original step304
continuation has been qualified on L4. That qualification is not approval for
a fresh sweep cell. Do not substitute the Azure worker.

### Fresh sweep initialization

`so101.train_entity_sim --initialize-only` accepts the normal cell recipe and
saves seeded step-zero model, empty Adam state with parameter groups, training
RNG, normalization and frozen-probe selection. It performs no evaluation or
optimizer update and refuses existing outputs, model resumes or existing
training states. Keep the full intended update budget in `--steps`; do not
change it to zero. The preparation flag is not part of the training recipe.

Stage and publish each cell's initial state as a new immutable input through
trusted activation. Verify its actual step/configuration and qualify that
exact state on the worker. Starting from this state evaluates the frozen
step-zero probe before the first update. An initial state is neither runtime
qualification nor a training grant; the host aggregate-approval path must
support step zero explicitly, not relabel it as the historical step304 PASS.

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
The persisted SQLite ledger retains job ownership, independent deadlines,
allocation intents and estimated compute allowances for audit and recovery.
It does not enforce a cumulative spending cap or count existing GPUs to
duplicate AWS admission. Matt chose AWS GPU quota and account budget alerts
as those controls; budget alerts notify rather than impose a spending cap.
AWS quota/capacity errors remain explicit, and every launched worker still has
independent shutdown and cleanup.

### Trusted future-session provisioning

The maintained root-only registration command is
`so101.control_session`. The trusted Dockyard launch integration validates the
session/container mapping and actual bridge IP, generates the private token
inside that container, and passes only its SHA256 to the installed command:

```sh
/opt/lego-robot-control/session-register register \
  --session-id UUID --container-id FULL_CONTAINER_ID \
  --token-sha256 SHA256 --preset-id lego-approved-sweep
```

`/etc/lego-robot-control/session-presets.json` supplies the fixed repository,
actions and exact approved `spec_sha256` list. Caller-supplied commands,
actions, recipes and tokens are not accepted. Root-private registration state
binds UUID/container/hash/preset, refuses rebinding, and permits an idempotent
same-binding retry after an uncertain broker reload. The host integration
installs the public CA, allows only the actual bridge IP, and invokes
`revoke --session-id UUID --container-id FULL_CONTAINER_ID` on cleanup.
Both actions return nonsecret JSON receipts only after broker reload succeeds.

This is a trusted server integration, not a registration API for containers.
Installing the command alone does not prove automatic session provisioning or
fresh-cell execution; verify those live before claiming self-service delivery.

## AWS prerequisites and shutdown

Keep AWS credentials exclusively in the trusted host service through temporary
roles. Before enabling `gpu_enabled`, configure and qualify:

1. AWS CLI v2, the controller role, artifact bucket/versioning, worker role,
   log groups, pinned launch template/AMI/image, and SSM documents.
2. Explicit policy values: `broker_id`, `aws_cli`, `region` (Oregon only),
   `artifact_prefix` (S3 prefix ending `/`), `instance_types`, `amis`,
   `max_hours`, `max_usd_per_hour`, `max_disk_gib`,
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
the name alone does not establish bucket existence, ownership, encryption, or
versioning. Verify those properties during operator inventory and qualification.
Versioned inputs and recovery require narrowly scoped `GetObjectVersion`, not
just `GetObject`. Worker SSM-agent and log publishing permissions also need
verification; naming agreement grants none of them.

The existing internal broker ID remains `lego-robot`; schedule names remain
`so101-<job-id>`. Do not reuse or modify the shared CPU security group for GPU
workers. AMI ID, CUDA image digest, subnet/security-group IDs, actual launch
template ID and numeric version, and SSM document numeric versions are
**unselected**. Pin them only after inventory and qualification; never assume
version `1`, `$Latest`, or a deployed ARN from a proposed name.

### Offline versioned artifact staging primitive

`so101.aws_artifacts` supplies `Artifact(uri, version, sha256, size)` and
`ArtifactStore.stage(root, job_id, artifacts, max_bytes=...)`. This is a library
asset, **not an installed bootstrap or a broker-wired staging command**. A future
root-installed worker must obtain these references and the byte ceiling from
its verified immutable manifest; a bare checksum does not authenticate a
container-supplied manifest. Use an operator-owned, private staging root.

The store allows only scoped S3 input/recovery prefixes, an explicit bucket
owner, exact non-null object versions, and safe flat destination filenames.
It checks the total declared byte ceiling before any transfer, then HEADs each
exact version/size before GET and verifies both returned metadata and local
SHA256/size. The AWS CLI receives an isolated environment using the worker
instance role, not inherited user/controller credentials or alternate endpoints.
No object content is deserialized or unpacked.

Staging exclusively creates a new mode-0700 `<root>/<job-id>` directory.
Existing attempts, including their STOP markers, are never reused or changed.
Only after every object verifies and files are synced does `.staged.json`
commit the complete staging receipt. An exception leaves an **uncommitted**
attempt (possibly containing partial or individually verified files); never
start from files alone or retry into that directory. A new attempt requires a
new broker job ID. No cleanup or retention deletion is automatic.

The byte ceiling is a local staging bound, **not** enforcement of the approved
$5 storage/network allowance. This primitive does not publish checkpoints,
prove checkpoint training-state completeness, implement S3 retention/recovery
selection, handle Spot notices, authorize a trainer start, or qualify CUDA.
Original dataset/checkpoint/STOP paths remain untouched.

### Offline versioned checkpoint publication primitive

### Operator-only CUDA runtime build inputs

`so101.aws_worker_run` now provides the root execution entrypoint and source SSM
documents in `scripts/so101-worker-{start,checkpoint-stop}.ssm.json`. They are
not registered/installed documents yet. Numeric SSM versions and hashes must be
frozen after qualification. The SSM agent must support `ENV_VAR` interpolation;
an older agent fails the fixed command rather than falling back to shell
substitution.

Root-only `/etc/so101-worker/execution.json` must declare `manifest`,
`manifest_sha256`, `plan`, `plan_sha256`, `staging_root`, `attempts_root`,
`docker_cli`, `grace_seconds`, and `qualification`. Qualification is a trusted
root JSON receipt with `status=PASS` and the exact `spec_sha256`; it must reflect
actual runtime qualification, not fabricated source-test results. The root
Python package must be installed in `/opt/so101-worker/bin/python`'s environment.

The start document accepts `Mode=start` (default), `Mode=qualify`, or the
separate disposable `Mode=spot-qualify` and cold `Mode=spot-recover-check`.
Qualification does **not** require an existing PASS, so the workflow is not
circular. It uses the exact root plan and explicit migration seed, idle receipt,
active guard and remaining lease, and refuses to overlap a started training
attempt. The pinned already-local image runs with only GPU0, offline read-only
checkpoint/model-cache mounts, and at most300 seconds. A timeout verifies the
container's specification label before removing that exact container ID.

`so101.aws_qualify` exercises actual CUDA13/BF16 gradients, EGL RGB rendering,
and tiny-model full-state recovery reproducing the next model/Adam update. It
also reconstructs the original SmolVLA trainable model/Adam state on CUDA from
the untouched checkpoint, using the explicit device migration seed. It does
not advance the original trial or change its file. Missing cached base-model
assets fail offline.

Successful checks produce `RUNTIME_PASS` in the separate task-owned
`<job>-qualification/runtime-qualification.json`, NOT the aggregate training
PASS file. Actual shutdown/independent termination, bounded Spot recovery
throughput and complete worker safety qualification remain necessary before
the operator can install the aggregate `qualification` PASS receipt. Offline
tests and mocked GPU responses are not evidence of these runtime checks passing.

The separately pinned plan has `schema=1`, `spec_sha256`, `argv`,
`readonly_mounts`, and `archives`. `argv` must exactly match `spec.recipe.argv` and contains only
trainer arguments after the fixed `python -u -m so101.train_entity_sim` command.
The saved relative output/training-state paths are preserved, not rewritten in
the checkpoint. Input mounts are root-trusted prepared paths within staged
artifacts.

`so101.aws_worker_prepare.prepare` binds the plan to the committed staging
receipt and verified archives. `archives` maps each staged artifact name to
`destination`, `max_bytes`, and `max_members`. Each destination is a new,
nonoverlapping staged directory. `so101.aws_extract` streams extraction,
rejecting traversal/absolute paths, symlinks/hardlinks, sparse files, devices,
and FIFOs; byte/member bounds are exact. It fsyncs file hashes/receipt and makes
only the new copied files root-owned/readable, not writable, for UID1001 mounts.
Direct file mounts must name a verified staged artifact; preparation explicitly
makes that new copy readable even under umask077. Other mounts must name
extracted inputs, not private staging receipts or arbitrary root paths.

Produce inputs with `python -m so101.aws_archive --root APPROVED_DATA_ROOT
--include RELATIVE_TREE --max-bytes N --max-members N`. Without `--destination`
this inventories only; with a new destination outside the source root it emits
a regular-file tar and SHA256/file inventory on stdout. Data selections must
name explicit approved robot-data paths, not `.` or a home/repository root.
For HF_HOME use `--hf-cache --include hub/models--ORG--REPO` (repeat per approved
model). Only `refs`, `snapshots`, and `blobs` are selected. Internal regular-file
links are flattened; external/directory links and private/auth paths are rejected.
Root `token`, `stored_tokens`, `xet`, and unrelated caches are never selected.
Producer and extractor share a hard ceiling of200,000 members; each operator
plan must still pin an explicit smaller measured budget (for example150,000
for the large approved dataset), never an unbounded tree or implicit merging.
The inventory counts every flattened copy, including repeated snapshot blobs,
not `du`'s unique-inode size. Sources are rechecked by inode, size, mtime, ctime,
and SHA256 during production. A failed command's partial output is not an artifact;
publish only a successful receipt's exact archive SHA/size/version. Large production
packing requires the operator's separate capacity-provisioned builder disk.

### Fresh GPU idle bootstrap (source asset, not live qualification)

`aws_build.prepare` now includes the explicit guard/Spot units, fixed SSM
documents, and `scripts/so101-worker-install.py` in the same immutable `source.tar`
as the runtime code. Rebuild the final image from this exact bundle: its source
SHA changes when these assets change. Record the separate launcher's raw SHA256
from the same commit before using it as root.

The trusted AMI must already provide Python >=3.12, AWS CLI v2, Docker, the NVIDIA
driver and NVIDIA Container Toolkit with Docker's `nvidia` runtime configured.
Bootstrap never installs mutable network packages, assumes a Docker socket is
available only to root, and accepts only the project private ECR digest in the
specified account/Oregon region.

Provision root-owned, non-group/world-writable `/etc/so101-worker/manifest.json`,
`plan.json`, and `execution.json` plus the exact-version source archive and
independently SHA-verified launcher under a private root directory. Execution
configuration has exactly `manifest`, `manifest_sha256`, `plan`, `plan_sha256`,
`staging_root`, `attempts_root`, `docker_cli`, `grace_seconds`, `qualification`.
Use the fixed paths `/etc/so101-worker/{manifest,plan,qualification}.json` and
`/var/lib/so101-worker/{staged,attempts}`; Docker must match the manifest path.
`grace_seconds` is an integer 1..300. Manifest, plan, and execution digests hash
canonical JSON (sorted keys, compact separators, no NaN); source/launcher digests
hash raw file bytes. The qualification path must remain absent at bootstrap.

On the newly launched **GPU only**, with independent scheduled termination
already armed and the exact job/spec/broker/deadline IMDS tags enabled:

```bash
/usr/bin/python3 -I /root/so101-worker-install.py \
  --source-archive /root/source.tar --source-sha256 SOURCE_RAW_SHA256 \
  --execution-sha256 EXECUTION_CANONICAL_SHA256
```

The launcher rejects existing `/opt/so101-worker`, verifies and bounds the
code-only archive before importing it, and installs the private stdlib-only
interpreter at `/opt/so101-worker/bin/python`. Before enabling any service,
the installer verifies the account, Oregon region, approved GPU type, AMI,
job/spec/broker tags, and unexpired lease. CPU/wrong-instance execution is an
error, never a request to power off that host. It enables the standalone lease
guard before creating the interpreter, pulling the image, or staging data.
The Spot timer also starts before long operations. ECR credentials come only
from the instance role and use an automatically removed private Docker config.
Image pull/requests are lease-bounded; archive preparation remains protected
by the active independent guard. Successful `/var/lib/so101-worker/installed.json`
means **prepared-idle**, not training or aggregate PASS. No start is issued.
Installation reports each image-pull, artifact-staging, and archive-preparation
phase with remaining lease seconds. Docker pull output streams without exposing
the temporary registry authorization. Extraction reports member/byte progress
every 10,000 members and checks the lease between members, body chunks, and
directory synchronization; expiration never commits an extraction receipt.
Failed/ambiguous installation leaves an intent and cannot blindly replay or
overwrite existing state; inspect that exact disposable worker before recovery.
Never run this installer/guard on the CPU controller or coding-agent host.

### Noncircular two-phase transport

Do **not** embed the final manifest, plan, execution hashes, or specification
hash into launch-template user data: the specification itself pins that user
data and numeric template version. First generate fixed user data with
`so101.aws_bootstrap_payload.userdata` from the verified source bundle and
only account/region/AMI/type/broker/AWS-CLI identity plus source/launcher raw
hashes. It embeds the standalone early bootstrap and lease guard, checks
GPU identity and valid runtime job/spec/deadline tags, and enables the guard
without downloading anything or starting training. Freeze these exact bytes
and the template version, then construct the specification, then the configs.

After launch, the trusted operator/controller sends the fixed numeric version
of `scripts/so101-worker-provision.ssm.json` with `JobId`, `SpecSHA256`, and
`ProvisionBase64` (base64 of canonical JSON containing exactly `manifest`,
`plan`, and `execution`). This requires separately scoped SSM provisioning
permission; source availability is not evidence that that permission exists.
The provisioner binds expected job/spec to IMDS, hashes actual IMDS user data
against `spec.bootstrap_sha256`, verifies all configuration links, and downloads
only the exact source version using the worker role. The launcher raw hash is
checked before execution. Provisioning is bounded by the lease, leaves an
exclusive intent on failure, and never starts training or writes PASS.
The installer adopts an existing early guard only when its root-owned receipt,
code, unit and early script match this exact worker and pinned source bytes.
Altered/ambiguous existing assets are rejected, not replaced.

### Disposable real Spot qualification

After prepared-idle bootstrap, dispatch the same pinned start document with
`Mode=spot-qualify`. It starts only a new tiny CUDA13/BF16 Linear/Adam workload,
not the SFT trainer, and refuses an existing training attempt. Its zero-filled
CPU padding equals the original staged checkpoint's exact byte size, so the
stop exercises representative I/O volume without reading/advancing that trial.
The workload saves full model/Adam and CPU/CUDA RNG state every five seconds
and exits only after a new-output STOP; an1800-second maximum without a stop
is a failure, not qualification. Check image+archives+expanded inputs+padding
and publication snapshot storage before launching the disposable worker.

Use an actual AWS Spot interruption (for example an operator-authorized,
single-tagged-worker FIS `aws:ec2:send-spot-instance-interruptions` with PT2M);
no fake IMDS notice is implemented. The existing Spot timer reads the real
notice, routes to the isolated `attempts/spot-qualification/JOB` attempt, and
shares the absolute deadline/stop lock. The CPU validator runs as root in the
read-only/networkless pinned image so it can read the root-private stable
snapshot; validation failure never becomes publication success. Successful
state and notice evidence commit under `OUTPUT_URI/qualification/JOB/` and
`OUTPUT_URI/qualification-events/JOB/`, never the normal training recovery
namespace, before requesting poweroff. The worker template must terminate,
not stop, on poweroff; independent scheduled termination remains armed.

The separately allowlisted read-only `gpu.recover_qualification` capability
requires the actual owned worker already `terminated`. It retrieves exact
versioned evidence, verifies the checkpoint/notice/spec links, compares padding
against the original S3 version's byte size, and checks the CloudWatch
`JOB-spot-qualification` checkpoint-stop marker. It returns
`QUALIFICATION_EVIDENCE`, **never aggregate PASS**. Poll observation time is
not AWS's actual termination timestamp: a late poll leaves deadline evidence
unproven, not proof of late termination. Use independently verified AWS event
timestamps when reviewing that uncertainty; do not substitute an invented time.

Download and SHA-verify the recovered disposable checkpoint's exact version
into a durable local copy before a cold GPU recovery test. Bind that same exact
reference as `spot_recovered_state.pt` in the next approved single GPU's root
manifest and stage it through the normal version/SHA-verified input transport.
Dispatch the pinned start document with `Mode=spot-recover-check`; the root
handler uses GPU0, no network, a read-only recovered checkpoint,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, and the fixed runtime command:

```text
python -m so101.aws_spot_workload check-cuda --state /check/state.pt \
  --padding-bytes ORIGINAL_CHECKPOINT_BYTES
```

Each checkpoint stores a next-update model/Adam byte hash forecast on a clone
on the pre-interruption GPU without consuming the running RNG stream.
The recovered CUDA RNG/model/Adam must reproduce that reference twice;
CPU-only validation returns `cuda_next_update_exact=false` and is not this test.
The separate `JOB-spot-recovery/recovered-runtime.json` receipt binds the actual
recovered artifact URI/version/SHA and returns `SPOT_CUDA_RECOVERY_PASS`, not
aggregate PASS. It cannot overlap training or the disposable workload; timeout
cleanup checks the exact owned container label before removal.
Only actual runtime, recovered-GPU, logs, real Spot and independent shutdown
evidence supports the operator's aggregate PASS and unchanged304 continuation.
Evidence retrieval is not a full local checkpoint export or cloud cleanup.
Failure leaves explicit partial task-owned staging, never modifies original
trees, and does not automatically retry or delete.

Model cache producers must flatten only internal approved data links into
regular files, rejecting resolved paths outside that input root. Package only
needed model-cache `hub` data, never HF_HOME token/stored_tokens, host home,
SSH keys or credentials. Existing original caches/datasets remain unchanged.
Collection/split/model mounts are read-only. A new UID1001-owned output receives
only a copy of the checksummed recovery state; original output and STOP are
never mounted writable or cleared. Missing model/data assets fail offline.

Launch requires the matching idle receipt and unexpired lease/active guard,
persists exclusive attempt intent before Docker, and runs the pinned image with
no network, a read-only root, no Linux capabilities, no privilege escalation,
and only GPU device0. Ambiguous starts cannot automatically replay. This is not
yet a complete controller/worker recovery state machine.

Checkpoint-stop writes STOP only in that new output and waits for the exact
committed container to exit. Nonzero exit/timeouts fail explicitly. Clean exit
first produces `exited-awaiting-validation`; the root CLI then validates full
state inside the pinned image on CPU, including original recipe/history,
finite model/Adam state, CPU/sampling RNG and CUDA RNG metadata. It revalidates
the stable snapshot before uploading, confirms both checkpoint and manifest
S3 versions/checksums, and writes a root `recovery.json` before SSM success.
The snapshot validator checks bytes/metadata, not actual CUDA restore capability.

`grace_seconds` now bounds the entire CLI stop/validate/upload operation (at
most300 seconds), not each subprocess independently. It must be shorter than
the broker's remaining termination grace/lease. Per-operation upload/validation
timeouts are further capped by remaining grace and
the absolute `StopDeadline` transmitted by the controller before SSM dispatch,
so dispatch delay cannot reset the stop window. This new parameter requires
coordinated controller and immutable stop-document version activation.
Ambiguous failures preserve
intent and reservations; there is no automatic replay or upload reconciliation.
Actual throughput/recovery qualification under the stop window is still required.
Worker `s3:PutObject` rights for the exact new run namespace are required; no
credentials are supplied to training or validation containers.

`so101.aws_spot` and the source `so101-worker-spot.service`/`.timer` perform
one-shot IMDSv2 checks. HTTP404 is the explicit no-notice case; other metadata
errors propagate. Account/AMI/type and exact job/spec/broker tags must match the
root GPU manifest before interruption action. Notices never extend the earliest
observed deadline and are capped at120 seconds and the original lease.

The monitor persists interruption intent and a shorter shared stop deadline,
then uses the same validated publication path. A root advisory lock prevents
SSM and Spot from publishing concurrently; each subprocess rechecks the
independent deadline and is individually bounded to120 seconds. An already
active stop retains ownership rather than being interrupted by another stop.
Successful publication precedes fixed OS poweroff; failures log and request
poweroff without automatic upload replay. Idle workers terminate without
touching original data. The service/timer are GPU-worker-only source assets,
not installed on the existing CPU host.

Training containers now use Docker's native `awslogs` driver with the exact
registered CloudWatch group and job-ID stream in Oregon. Blocking mode and
one-second flush are explicit; driver startup/permission failures fail container
launch rather than silently falling back to local-only logs. The GPU worker
daemon uses its instance role; no AWS credentials or network are supplied to the
training container. Worker images/daemon configuration must not contain static
credentials or unrelated profiles. Required rights are scoped CreateLogStream
and PutLogEvents on the existing approved group, not CreateLogGroup.

Controller start/stop SSM dispatch also enables CloudWatch command output in
that registered group. Command output uses SSM-managed stream names; training
logs use the exact job stream exposed by `gpu.logs`. Real upload/flush/failure
qualification remains required. Guard/Spot systemd journals are still local.

Controller `gpu.recover` now discovers content-addressed manifests in only the
owned job's registered output prefix, without relying on SSM stdout (which Spot
poweroff can lose). It pins current manifest VersionId with HEAD, then downloads
that exact version and checks filename SHA256/body identity, job/spec/update,
and checkpoint namespace/version/size/server SHA256. A higher checkpoint-only
object is not a committed manifest and does not hide the prior recovery.
Malformed complete manifests, ambiguous latest manifests, or incomplete
listings fail explicitly rather than silently selecting an older attempt.

Verified metadata is persisted privately beside controller state. SSM-success
stop handling verifies recovery before termination; `gpu.recover` also works
after independent/Spot termination. Parent resume specifications must match
the verified checkpoint URI/version/SHA while preserving the frozen code/image,
dataset/split, and recipe. Required controller read-only rights: scoped
ListBucket, GetObject (unversioned HEAD to pin VersionId), and GetObjectVersion
for the exact project run objects. No S3 writes/deletes are added.
This verifies published references/server checksum, not a fresh local full-state
download/export; local durable export is still necessary before temporary cloud
copies can be cleaned up.

Real Spot notice/throughput/termination qualification,
root host bootstrap, and live verification remain unfinished.
Independent lease termination remains necessary on stop failure.

Broker submission no longer scans global GPU inventory as an admission gate.
AWS quota and capacity responses determine allocation; ownership-scoped
inventory remains necessary for status and cleanup. Launches remain Oregon-only.

This supplements, rather than replaces, the broker's serialized one-attempt
ledger. Inventory is a snapshot, not an AWS atomic cross-principal lock. Other
principals, open Spot requests/fleets, or stopped machines started concurrently
can invalidate it. Operator-controlled exclusive GPU launch/start authority and
verification of those external launch sources remain necessary. Quota alone is
not proof of one machine. Unrelated other-region workloads are not inventoried,
so this is not an account-wide guarantee. This source change is not installed
or live-qualified.

`so101.aws_budget.Reservations` is an offline integer-microdollar accounting
primitive. It serializes cumulative reservations with SQLite `BEGIN IMMEDIATE`,
persists them before external operations, retains reservations after ambiguous
failure, and rejects changed contracts for reused IDs. Its ceiling is $5 across
the single ledger, not $5 per worker. Byte-hour storage and both transfer
directions round upward; request and fixed exposure must also be declared.

It is **not wired to the broker, publisher, or worker**, does not authenticate
the policy digest, and does not query prices, enforce deletion, or cap AWS bills.
An operator-owned trusted ledger and authenticated rate/horizon policy are
mandatory before wiring it. Include ALL newly incurred exposure: temporary
artifact versions, manifests, requests, transfer/recovery, logs, image registry,
and worker disks; costs omitted from the contract remain unbounded.

Proposed policy for review, not deletion authorization: bounded temporary NEW
cloud copies, with independently verified durable local export of new full-state
checkpoints before cloud expiry. Original datasets/checkpoints/STOP are never
deleted or changed. Retention after the accounting horizon requires a separate
reservation before that horizon expires. Because retention enforcement and
export recovery are not implemented, no finite-horizon estimate yet establishes
the approved lifetime noncompute bound. Missing policy/rates/export/expiry
verification prevents using this primitive as evidence of a lifetime spending
cap. It is not an additional worker activation prerequisite and is not wired
into admission.

`scripts/Dockerfile.gpu-worker` pins the Linux amd64 Ubuntu24.04 base by digest.
The exact cu130 wheelhouse supplies CUDA runtime libraries; using an additional
NVIDIA CUDA runtime base would duplicate those libraries. Driver capabilities
include compute, utility and graphics for EGL; `NVIDIA_REQUIRE_CUDA=cuda>=13.0`
declares the driver requirement, not proof of a working GPU.
It does not install a host driver, launch training, download models,
or prove GPU qualification. The existing host must later supply NVIDIA driver
580 or newer and NVIDIA Container Toolkit. Building on the existing CPU host
does not itself require a GPU.

The operator must collect a complete Python 3.12 Linux amd64 wheelhouse using
the locked training/simulation export and `cloud_requirements.cuda_requirements`,
with explicit cu130 resolution. The Linux CPU lock is not directly installable
into this image. Collection must include CUDA-specific transitive dependencies;
the wheel inventory helper does not resolve them or certify that every
nonaccelerator version matches `uv.lock`. That comparison remains a build gate.

`python3 -m so101.aws_build --repo REPO --commit FULL_SHA --wheelhouse WHEELS
--destination NEW_CONTEXT` exports only committed runtime source, `pyproject.toml`,
and `uv.lock`; it never exports the mutable checkout, datasets, model caches,
checkpoints, STOP, or credentials. It copies and hashes every wheel and rejects
duplicate packages, wrong-platform wheels, and non-cu130 Torch/Torchvision pins.
Older CPython ABI3 wheels through cp312 are accepted for Python3.12; older
version-specific CPython ABIs and cp313 wheels are rejected.
The generated requirements use exact versions and SHA256 hashes. Package
installation in the Dockerfile is network-free with dependency checking.

On the trusted operator host, pass the manifest's `commit` and `source_sha256`
as `SOURCE_COMMIT` and `SOURCE_SHA256` build arguments. No Docker daemon access
or registry credential belongs in the source engineering container. System apt
packages are not snapshot-pinned: record the resulting image digest and package
inventory; this recipe is not a claim of bit-for-bit reproducible apt layers.
BuildKit read-only bind mounts keep installer wheels out of final image layers;
the host still needs room for the wheelhouse and build cache. The idle image defaults to UID1001,
registered as `so101` in `/etc/passwd`, with `HOME=/tmp` and offline model access.
The launcher supplies writable temporary storage; deterministic Torch/Adam
initialization must be checked as UID1001, not just as root, because Torch's
cache initialization resolves the operating-system username.
The image also declares `/tmp` as a temporary volume so read-only CPU
checkpoint validators have writable scratch storage without extra flags.
Their `--rm` launch removes that anonymous volume on exit; explicit launcher
tmpfs mounts take precedence.
Narrowly owned `/work` and separately staged model
assets are required before a future qualified launcher can use it.

No final wheelhouse, image digest, AMI, or build qualification exists yet.

`so101.aws_checkpoint.CheckpointPublisher` is a source-only library for stable,
opaque checkpoint publication. It is not connected to a trainer, SSM document,
Spot handler, or installed worker. A trusted caller must validate full training
state, actual step/spec identity, runtime authorization, lease, and conservative
compute/noncompute reservations **before** invoking it. No paid/upload action
is authorized by the presence of this library.

The caller supplies a private task-owned publication root, separate from
original outputs. A new job/step directory is created exclusively; existing
steps cannot be replayed automatically. The source is opened read-only without
following a final symlink, must be a regular file, and is copied into a synced
snapshot under an explicit byte ceiling. Size/mtime/ctime changes during the
copy fail before any AWS request. Normal trainer checkpoint writes must still
use atomic replacement, not in-place mutation.

The library persists an intent before upload, uses content-addressed keys under
`runs/lego-robot/.../<job-id>/step_<step>/`, and requires conditional
`PutObject(IfNoneMatch="*")`, explicit bucket owner, SHA256 checksum, AES256
encryption, and non-null version IDs. Each put is confirmed by an exact-version
HEAD with matching size/checksum. Only after the checkpoint version verifies is
a versioned manifest published; it binds job/step/spec to the immutable
checkpoint reference. A local `published.json` receipt appears only after both
versions confirm. The returned manifest reference can be verified using the
existing exact-version staging library; recovery selection and trainer resume
remain separate, unfinished integration.
Only single-part PUTs are implemented; the caller must preflight the applicable
S3 object-size limit. Multipart publication is not supported.

Upload errors/timeouts/checksum mismatches propagate. Partial publication
retains intent/snapshot and possibly a remote uncommitted checkpoint version;
there is no retry, overwrite, deletion, or success-shaped fallback. The operator
must reconcile exact keys/versions and retain cost reservations after ambiguous
outcomes before considering another attempt. The byte ceiling is not a storage,
request, network, retention, or account-cost cap. Full-state validation,
noncompute budget enforcement/retention decisions, interruption uploads, and
real S3 recovery qualification remain prerequisites for operational use.

### Root-manifest-bound idle staging entrypoint

`python3 -I -m so101.aws_worker_bootstrap` adds a **source-only** staging
entrypoint, not a complete or deployed GPU bootstrap. A future operator-owned
image must install the Python package, AWS CLI, and Docker CLI outside writable checkouts,
create a root-owned staging directory with non-writable parents, install the
lease guard, and invoke the entrypoint with fixed `--manifest`,
`--manifest-sha256`, and `--root` arguments from its immutable bootstrap.
Do not invoke it on the CPU controller. No real manifest/AMI/image/template
or SSM payload is selected by this entrypoint.

The root manifest and every parent must be root-owned and not group/world
writable. Its SHA256 is over the broker's canonical JSON representation
(sorted keys, compact separators), **not** a pretty-printed file. Pin that
digest in the qualified bootstrap; never accept it from the coding container.
The exact schema is:

| Field | Meaning |
| --- | --- |
| `schema` | Integer `2`; earlier staging-only schema `1` is rejected |
| `broker_id` | Exact approved broker tag |
| `spec_sha256` | Canonical digest of the registered job specification |
| `spec` | Complete, exact registered specification contents |
| `account_id` | Expected twelve-digit instance account and bucket owner |
| `instance_type` | Approved `g5.2xlarge` or `g6.2xlarge` |
| `ami` | Exact approved AMI ID |
| `aws_cli` | Absolute root-installed AWS CLI path |
| `prefixes` | Nonempty list of approved S3 input/recovery prefixes |
| `max_bytes` | Positive integer staging-byte reservation |
| `artifacts` | Flat filename to `{uri, version, sha256, size}` mapping |
| `runtime` | `{docker_cli, image, commit, source_sha256}` identity contract |

Schema 2 requires `source.tar` and `training_state.pt` artifacts. The manifest's
`spec` must hash to `spec_sha256`; its AMI/instance type must match the manifest,
its image/commit must match `runtime`, and its
checkpoint URI/version/SHA256 must match `training_state.pt`. `source.tar` must
match `runtime.source_sha256`. Other dataset/split/cache references still need
operator validation against the exact recipe and artifact inventory. This is
an intentional source-schema change: regenerate and re-pin any prepared schema-1
manifest before using this entrypoint; no deployed manifest is migrated.

The operator must derive this manifest from the exact frozen job spec and
verified dataset/split/checkpoint inventory; binding a spec digest does not
itself prove the operator chose the right artifacts or recipe. This schema is
a staging contract, not an alternative trainer recipe or training authorization.

Before any transfer, the entrypoint verifies the manifest digest, root staging
directory/AWS CLI/Docker CLI ownership, an active `lego-robot-worker-guard.service`, and
IMDSv2 identity (account, Oregon region, GPU type, AMI). The instance's job,
spec, and broker tags must match. It captures the hard lease before transfer
and rechecks expiry using wall and monotonic clocks, plus guard activity,
before committing `.bootstrap.json`. That receipt binds the manifest/spec/job
to the SHA256 of `.staged.json` and the checked runtime identity; its phase is
only `staged-idle` (binding receipt schema 2).

Before S3 staging, `so101.aws_runtime` inspects **only an already-local image**
via the worker's fixed root-controlled Unix Docker socket. It requires the
exact repository digest, Linux amd64 platform, and both
`org.opencontainers.image.revision` and `io.so101.source-sha256` labels matching
the root manifest. Missing/mismatched images fail; there is no pull, container
creation, or execution, and inherited Docker contexts/endpoints are ignored.
The installed worker Docker socket must never be mounted into coding agents.

These label checks verify **declared source identity**, not the installed code
contents or trusted build provenance. The pinned image digest and verified
source archive bind immutable inputs; a future qualified build still must
prove that the image actually contains/uses those source bytes, the correct
CUDA dependencies, and no secret/data payloads. Runtime CUDA/EGL/full-state
qualification remains separate.

Missing metadata, guard failures, identity mismatches, transfer failures, or
expiry abort visibly. An attempt with `.staged.json` but no `.bootstrap.json`
is not a successfully bound staging attempt. Neither receipt authorizes a
trainer start, proves installed code contents/build provenance, validates full-state
CUDA recovery, or proves independent shutdown works. The entrypoint does not
start services, power off, clear STOP, unpack data, load a model, change file
ownership, or launch training. Guard installation/activation and later
immutable start/stop orchestration still require separate operator integration
and real qualification.

### SFT device continuation source support

The serial SFT driver accepts `--device cpu|mps|cuda` (default `mps`).
Changing this option leaves the grid, sample counts, microbatch, seeds,
learning-rate schedule, frozen probes, evaluation cadence, and retention flags
unchanged. Its dry-run output includes the actual trainer commands.

SFT and RL share backend-aware device RNG recovery. Cross-backend recovery
requires both `--allow-device-rng-migration` and a valid explicit
`--device-rng-seed`; otherwise it fails before restoring model/optimizer state.
The source backend comes from saved RNG metadata, or the legacy saved
`run_config.device`, never from the requested target. Same-backend recovery
restores device RNG bytes without reseeding. A migration preserves CPU and
sampling RNG, model/Adam state and optimizer/sample history, seeds only the
target device RNG, and records `device_rng_migrations` with `bit_exact=false`.
Only the device field may change under this opt-in; unrelated recipe changes
remain rejected. New SFT saves record the RNG backend explicitly.
SFT also accepts its valid pre-first-update checkpoint (step zero, empty update
history and Adam state, initialized parameter groups); trained-state validation
remains the default for other callers. Both paths reject non-finite tensors.

This source support does not load or modify the stopped update-304 checkpoint,
clear STOP, authorize training, or qualify CUDA/full-state recovery on a real
worker. Keep using a separately verified attempt and the existing runtime
handoff/qualification requirements.

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
# Registered purchase modes

The host broker defaults existing specifications to Spot. An operator-created
specification may explicitly set `purchase_option` to `spot` or `on-demand`.
On-demand additionally requires a root policy `on_demand_rates` mapping the
approved instance type to its exact `max_usd_per_hour` reservation rate, and a
pinned launch template with no `InstanceMarketOptions`. Spot retains its bounded
price, one-time, terminate-only template checks. Neither mode authorizes training
or relaxes lease, checkpoint, or runtime qualification requirements.
This host-only adapter change does not require replacing a previously pinned
worker source archive or image.
