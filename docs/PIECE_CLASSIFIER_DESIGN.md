# Piece recognition: SFT feasibility experiment

Work `e2246fc2-64c6-4d14-8384-6d7b957cfa37`. Matt's latest direction:
**run the bounded classification experiment while RL data is being fixed,
using GPU rendering.**
This supersedes the earlier standalone ConvNeXt/16-ID proposal.

## What we will test

Use the existing four-frame entity SmolVLA checkpoint, with one query per
bounding box in the final image. Final VLM query features feed:

- An embedding head matched against **12 canonical rendered views per part**.
  Anchors are mean-pooled vision+connector features from the initial checkpoint,
  frozen before SFT. Retaining separate views avoids averaging away details.
- Shared **family and kind category heads**, so exact-ID uncertainty can
  eventually fall back to a broader category. Initial labels are catalog family
  plus head/gear distinctions from descriptions; they are heuristic, not a
  complete reviewed LEGO hierarchy.

Train contrastive catalog-ID cross-entropy plus category cross-entropy, updating
shared vision/VLM weights but **not the action expert**. No RL or action loss
in this first experiment. `--freeze-backbone` provides a head-only comparison.
Nothing about this synthetic test establishes robot improvement or real transfer.
Raw ranking/category metrics are not calibrated confidence or safe fallback yet.

Queries see four static-pile camera-orbit images, with boxes referring to the
last view. These are multiview inputs, **not recorded robot-motion histories**.
Image-frame embeddings identify the final view. Queries cannot attend one another
or alter image tokens, and share the same position ID to prevent box-list order
from becoming an ID shortcut. Inputs contain no part names/IDs/depth/state.

## Data and split

Search both shared `data/lego_curriculum` and `data/lego-renderer` trees for
supported saved piles: admitted task scenes, settled photo piles and native
settled-pile exports. Do not select only currently active task pools.
Record every discovered model's inclusion/exclusion, exact hashes and labels.
Unknown schemas/identities are explicit exclusions, never invented labels.

Historical `catalog-demo` labels can be recovered only by matching exact visual
mesh signatures against the eight known types in the **hash-pinned retained
photo template**. No substring/body-number guess is allowed. Catalog inputs are
copied under the existing reader lock; the current catalog is never modified.
Missing/unverified exact IDs fail preparation explicitly.

Split approximately **70/15/15 train/development/test by provenance-connected
pile family before rendering**. Group exact models, source receipts and retasked
family relatives together, including raw piles and robot-integrated derivatives.
All eight rendered views and four-view windows of a pile remain in its split.
No class-balancing migration of individual test pieces into train.

Render unchanged saved states with existing MuJoCo raster APIs; never settle,
step or generate a new pile. Annotate every visible piece, not just task targets.
GPU labels use a dedicated zero-MSAA context
(`native-id-segmentation-no-msaa-v1`); the original RGB context and model
MSAA setting are preserved. Multisampled ID colors must not be used for boxes.
Invisible and <64-visible-pixel targets are counted separately, not trained as
confident positives. Dataset size is determined by the saved inventory.
At initial inspection, the raw `photo-piles-*` population had 617 receipts and
the curriculum tree had 80 `problem.json` files; these are candidate counts,
not deduplicated admitted experiment populations, and producers remain active.

Report top-1/3/5, macro/per-ID recall, major confusions, category accuracy,
train frequency and **seen-ID vs unseen-ID** test slices, plus a training-majority
baseline. Select a checkpoint using development only, then evaluate test once.
Canonical geometry for test-only IDs is intentionally available: this tests
render-conditioned retrieval of unseen observed-pile IDs, not geometry-free
zero-shot recognition. Other catalog IDs are not included as distractors yet.
The first experiment does not measure unknown rejection or calibrated abstention.
The split is held out from **this recognition SFT**. An existing VLA checkpoint
may previously have encountered some piles through robot training; that exposure
has not been ruled out. Do not claim previously unseen-scene generalization
without auditing the selected checkpoint's training history.

## Entry points

Run these in the repository's isolated
simulation/training environments. Do not modify the host `.venv`. `EXPERIMENT`
is a new named output directory, shared with the eventual GPU host.

```bash
EXPERIMENT=/mnt/dockyard/lego-robot-data/outputs/piece-recognition-v1
PYTHONPATH=src python -m so101.piece_recognition_data \
  --pile-root /mnt/dockyard/lego-robot-data/data/lego_curriculum \
  --pile-root /mnt/dockyard/lego-robot-data/data/lego-renderer \
  --catalog /mnt/dockyard/lego-robot-data/data/lego_catalog/current/catalog.json \
  --photo-template /mnt/dockyard/lego-robot-data/data/lego-renderer/b887f204-big-pile-compact-settled/model.xml \
  --output "$EXPERIMENT/prepare"

PYTHONPATH=src MUJOCO_GL=egl python -m so101.piece_recognition_render \
  --plan "$EXPERIMENT/prepare/plan.json" --output "$EXPERIMENT/render"
```

Preparation produces a manifest and pinned subset, not rendered images.
Rendering produces clean RGB, same-camera boxes, four-view windows and canonical
part views, not encoded features or training. Both reject existing destinations;
partial directories are preserved for inspection, not overwritten on retry.

For GPU SFT, choose an immutable qualified worker image containing this new
source, an existing **four-frame entity checkpoint** (`model.pt`,
`entity_config.json`, `metadata.json`), and its already-cached base model.
Copy/bind inputs at the exact absolute paths recorded in manifests. Matching
MuJoCo version is required to restore saved states. No checkpoint is selected
or downloaded, no image is built and no GPU is allocated by preparation.

```bash
python scripts/prepare_piece_recognition_launch.py \
  --image "$QUALIFIED_IMAGE_DIGEST" --experiment-root "$EXPERIMENT" \
  --checkpoint "$CHECKPOINT" --cache "$HF_CACHE" \
  --container lego-piece-recognition-v1 --output "$REMOTE_COMMAND_JSON"
```

This writes Docker argv only. Stage that JSON at the same path on the GPU host.
The future **actual** launch must be through the managed mirror:

```bash
python scripts/sync_gpu_dashboard.py \
  --host "$GPU_HOST" --instance-id "$GPU_INSTANCE_ID" \
  --remote-root "$EXPERIMENT/output" --output "$LOCAL_MIRROR_ROOT" \
  --job-id piece-recognition-v1 --container-name lego-piece-recognition-v1 \
  --launch-command "$REMOTE_COMMAND_JSON"
```

The eventual operator must run that launcher persistently and verify its first
successful 20-second refresh, reconcile an existing container before retries,
and never attach by restarting another trainer. The SFT output basename follows
the existing mirror's accepted naming; its objective is explicitly recorded as
recognition-only. The prepared schedule is 500 optimizer steps, one image-window
per microbatch, four accumulation batches, up to 32 queries/window chunk and
learning rate 1e-5. Runtime/VRAM/data volumes remain unmeasured.
`--freeze-backbone` writes to `output-frozen` instead of `output`; point the
managed mirror's remote root there for that separate comparison.

Artifacts: `anchors.pt`, `best.pt`, `sft_training.json`, `test_results.json`.
No automatic resume or overwritten working policy. `STOP` stops this new run.
Model choice uses development macro recall; test stays out of optimization.

## Current boundary

The bounded recognition experiment, GPU rendering and training are authorized;
**RL execution and integration are not**. Existing catalog, renderer, generator
and dashboard surfaces remain owned elsewhere. Record the selected checkpoint,
digest-pinned runtime, shared paths/cache mounts and connection/job arguments
with the execution receipt. Synthetic feasibility is the first question; real images, calibration,
robust category fallback, joint RL scheduling and robot-task regression tests
remain subsequent work.
