# Bounding-box-conditioned robot control plan

## Objective

Build a closed-loop system in which:

1. A planning VLM receives a scene image and a high-level task.
2. It selects one object or region and emits one bounded command:
   - a short skill phrase such as `grasp`, `push`, `view`, or `place`;
   - one or more role-tagged bounding boxes;
   - optionally, camera-relative distance for any entity whose depth is known.
3. A local control model continuously converts the command, current wrist image,
   and robot state into SO-101 action chunks.
4. The control model also predicts where every commanded entity will appear and
   its camera-relative distance after each action in the chunk.
5. The system executes a short prefix, observes again, updates every entity,
   and replans without pausing the control loop.
6. Success is determined from physical evidence, not from the policy's claim.

The first acceptance target is deliberately narrow:

> Given a bounding box around one loose tabletop LEGO in the current wrist
> image and the phrase `grasp`, lift that exact LEGO and hold it visibly above
> the table.

Color-based planning, multi-object sorting, pushing, placing, and a packaged
Copilot session come after this primitive works reliably.

## What stock SmolVLA supports

The installed LeRobot 0.6.1 implementation already supports:

- one or more images;
- robot state;
- tokenized language;
- flow-matching prediction of a fixed-length motor-action chunk;
- queued chunk execution and configurable `n_action_steps`.

It does **not** currently support:

- a structured bounding-box input;
- future bounding-box outputs;
- per-entity depth inputs or predictions;
- temporal image/state history for triangulation;
- an object-visibility output;
- auxiliary tracking or success losses;
- reinforcement-learning updates.

The relevant implementation is
`.venv/lib/python3.12/site-packages/lerobot/policies/smolvla/`.
The policy projects padded state through `state_proj`, embeds image and language
tokens as a prefix, and predicts action flow from expert suffix tokens through
`action_out_proj`. Its training `forward` currently returns only per-action
flow-matching MSE.

We will not edit `.venv` files. We will implement a repository-owned policy
package that subclasses or vendors the minimal SmolVLA pieces we need. This
keeps the uv environment reproducible and makes checkpoints self-describing.

## Command contract

Create `src/so101/control_model/command.py` with validated command and entity types:

```python
CommandEntity(
    entity_id: str,
    role: Literal["object", "destination", "reference", "obstacle"],
    box_cxcywhr: tuple[float, float, float, float, float],
    camera_depth_m: float | None,
)

BaseCommand(
    command_id: str,
    scene_id: str,
    source_camera: str,
    source_timestamp: float,
    phrase: Literal["grasp", "push", "view", "place"],
    entities: tuple[CommandEntity, ...],
)
```

Each entity has a stable identity within the command, an explicit semantic role,
and an oriented box containing normalized center coordinates, canonical major
and minor extents, and rotation. For model input and prediction, convert the
shape to the axis-swap-invariant features
`mean_extent`, `anisotropy*cos(2*rotation)`, and
`anisotropy*sin(2*rotation)`. This representation remains continuous when a
box implementation swaps its two axes. Optional
`camera_depth_m` is the entity reference point's positive distance along the
wrist camera's optical axis. An absent value means unknown, not zero. The
command also carries the source frame identity so stale boxes cannot silently
be applied to a different scene.

Command validation is phrase-specific:

- `grasp`: exactly one `object`;
- `push`: one `object` and one `destination`;
- `view`: one or more `object` or `reference` entities;
- `place`: one `destination`, plus an `object` entity when the held object is
  visible and trackable.

The first implemented skill remains `grasp`, but its representation is already
the one-entity case of the general interface. Unsupported role combinations
fail explicitly rather than being reinterpreted.

Entities are task inputs, not robot motor state. The model receives a
variable-sized entity set represented internally as padded tokens plus an
entity mask. The initial maximum can be four entities, but model parameters are
shared across entities so raising that limit does not introduce role-specific
heads.

## Dataset representation

Each training sample at time `t` contains:

- a temporal window `observation.images.wrist[t-K+1:t]`;
- corresponding six-value SO-101 state history and timestamps;
- previous executed actions for the history window;
- `observation.command_boxes[t]`, padded normalized swap-invariant oriented
  boxes with shape `(max_entities, 5)`;
- `observation.command_role_tokens[t]`, role text tokenized with the same
  SmolVLM tokenizer used for the task phrase;
- `observation.command_role_attention_mask[t]`;
- `observation.command_entity_mask[t]`;
- `observation.command_visibility[t]`;
- `observation.command_depth_m[t]`;
- `observation.command_depth_known[t]`;
- `task`, initially `grasp`;
- `action[t:t+H]`, the normal SmolVLA motor-action chunk;
- `command_boxes_future[t:t+H]`, the box of every entity after each gold action;
- `command_visibility_future[t:t+H]`;
- `command_depth_m_future[t:t+H]`;
- `command_depth_valid_future[t:t+H]`;
- episode outcome and target identity for evaluation, not as an inference input.

Future boxes preserve entity identity. The future `object` box is the same
physical object selected by the planner, not whichever same-colored contour is
largest later. A `destination` box follows the same physical destination region
through camera motion even when that region itself does not move. Simulation
supplies exact identities. Real data uses planner selection, tracking, and
manual review when identity is ambiguous.

Every entity box must be transformed with every image augmentation. Crops,
resize, padding, flips, and rotations that do not transform all boxes
identically are forbidden. Entity order and role labels must remain aligned
through batching and augmentation.

Depth is metric and is not changed by image resize or crop. Camera-coordinate
augmentations that would invalidate metric depth are forbidden unless they
transform the camera model and labels consistently.

## Model development

### Repository-owned entity-conditioned SmolVLA

Add:

- `src/so101/control_model/configuration_entity_smolvla.py`;
- `src/so101/control_model/modeling_entity_smolvla.py`;
- `src/so101/control_model/processor_entity_smolvla.py`.

Use the pretrained SmolVLA image, language, state, and action weights. The skill
phrase and entity roles both use the existing SmolVLM tokenizer and language
embedding space. Each entity is encoded as a small token block:

```text
canonical role text ("object", "destination", ...)
  -> SmolVLM tokenizer and language embeddings

normalized center (2) + mean extent + anisotropy vector (2)
  + visibility + normalized depth + depth-known bit
  -> coordinate/Fourier encoding and MLP
  -> one numeric entity token placed immediately after its role tokens
```

Unknown input depth is represented by a dedicated depth-known embedding and a
zeroed depth value. It is never represented as a measured zero distance.

The observation encoder accepts a configurable temporal window. Each history
step contains:

- wrist image;
- measured robot state;
- previous executed action;
- relative timestamp;
- tracked entity boxes, visibility, and optional measured depth.

Frame/time embeddings preserve temporal order. The model therefore has both
visual parallax and the associated robot/camera motion needed to infer depth.
Start with `K=4` frames at 10 Hz and compare against `K=1`, `K=2`, and `K=8`.

The prefix is ordered as:

```text
[image tokens]
[task phrase language tokens]
[entity 1 role language tokens][entity 1 numeric box token]
[entity 2 role language tokens][entity 2 numeric box token]
...
[robot-state token]
```

Entity and token masks exclude padding. Roles do not use a separate integer
embedding table. Canonical role strings are retained in the dataset, and the
custom processor produces their language token IDs and attention masks. The
same box encoder and tracking head are applied to every entity, which lets the
model generalize from one-box `grasp` commands to two-box `push` commands and
later commands with more entities.

Expose the final expert suffix features currently used by `action_out_proj`.
Add a shared entity tracker:

```text
concatenated per-step action feature and per-entity query
  -> shared interaction MLP
  -> 5 absolute oriented-box values, 1 log-depth value,
     and 1 visibility logit
```

Predict all 16 future boxes in parallel, one for the state following each
action in the chunk. Predict depth in log space. These trajectories handle
both physically moving objects and stationary destinations whose image
coordinates and camera-relative depths change as the wrist camera moves.

The model output is:

```python
ControlPrediction(
    actions: Tensor[B, H, 6],
    entity_boxes: Tensor[B, H, N, 4],
    entity_log_depths: Tensor[B, H, N],
    entity_visibility_logits: Tensor[B, H, N],
    entity_mask: Tensor[B, N],
)
```

### Losses

Use supervised behavior cloning first:

```text
L = L_action
  + lambda_box * masked_oriented_box_loss(predicted_entity_boxes, gold_entity_boxes)
  + lambda_depth * masked_SmoothL1(predicted_log_depth, gold_log_depth)
  + lambda_visible * masked_BCE(predicted_visibility, gold_visibility)
```

- `L_action` remains SmolVLA's flow-matching loss.
- Box loss uses Smooth L1 with `beta=0.02`, weights center error twice as
  strongly as shape error, and uses normalized timestep weights increasing
  linearly from 1 at the first prediction to 2 at the sixteenth.
- Box loss is masked by both entity presence and visibility. The
  swap-invariant anisotropy representation naturally removes orientation loss
  for square boxes.
- Depth loss is masked by entity presence and depth-label validity. It can
  remain active through visual occlusion in simulation when an exact
  camera-frame entity reference point is still available.
- Visibility loss is masked by entity presence and remains active while a
  present entity is occluded.
- Losses are normalized independently before weighting.
- Start with `lambda_box=10`, `lambda_depth=1`, and `lambda_visible=0.1`, then
  tune using validation curves rather than raw loss magnitudes.
- Report metrics separately by role so easy destination tracking cannot hide
  poor object tracking.

Train and compare four depth conditions with the same action/tracking
architecture:

1. **Measured-depth input:** current/history depth supplied when available.
2. **Single-frame depth-blind:** no depth input and `K=1`.
3. **Temporal depth-blind:** no depth input and `K>1`, requiring learned
   parallax/scale inference.
4. **Mixed-depth:** random per-entity depth dropout during training so one model
   handles partial sensor availability.

All variants predict future depth. Input-depth dropout is independent of output
depth supervision. Separate checkpoints and a mixed-depth checkpoint are
evaluated so gains from directly supplied depth are not confused with learned
monocular or temporal estimation.

Add an optional success/contact head only after box tracking is useful. Do not
train it from gripper percentage alone; labels require simulator object state or
overview-verified real suspension.

Reinforcement learning is deferred. Sparse real grasp reward would be
sample-inefficient and would obscure whether failures come from perception,
target tracking, or motor control. Simulation RL is considered only after the
supervised model reproduces gold trajectories but still fails under recoverable
perturbations.

## Gold trajectory generation in simulation

Extend `robot_sim.py` without allowing the learned model to access simulator
object poses.

For every episode:

1. Randomize robot start, target pose, target shape/color, distractors, lighting,
   camera variation, and table appearance.
2. Create a command with exact simulator-backed entities and roles.
3. Render each entity's segmentation ID or projected destination region to
   derive precise wrist-image boxes.
4. Transform each entity reference point into the wrist-camera frame and record
   its positive optical-axis depth in meters.
5. Generate a contact-verified expert trajectory.
6. After every expert action, render every command entity's next box,
   visibility, and camera depth while preserving entity identity.
7. Retain only episodes with verified lift and one-second two-jaw retention.

The simulator depth label is not an arbitrary body-to-robot distance. It is the
entity reference point's `z` coordinate in the calibrated wrist-camera frame.
For an object, the reference point is its defined manipulation center. For a
destination, it is the destination region center. This definition remains
available through occlusion and is reproducible from camera extrinsics.

Include hard cases:

- same-color distractors;
- crossed and partially occluded pieces;
- target near image boundaries;
- target leaving and re-entering the wrist view;
- similar apparent box sizes at different depths;
- controlled wrist/camera motions that produce useful parallax;
- little or no camera motion, exposing unavoidable monocular ambiguity;
- recovery prefixes generated by the current policy;
- different start poses and object orientations.

After grasp is established, add push trajectories containing both an `object`
and `destination`. Include controls where the same object is paired with
different destinations in an identical scene so success requires using both
entity tokens rather than memorizing the image.

Split by complete scene seeds, not frames. No physical layout or trajectory
appears in both train and evaluation.

Initial dataset target:

- 2,000 uncomplicated grasp episodes;
- 1,000 same-color distractor episodes;
- 1,000 policy-recovery episodes;
- balanced hollow/studded shapes and all supported colors.

Dataset quality is measured by target identity, lift, retention, and box-label
continuity—not merely by exporter success.

## Simulation data curriculum

Prioritize object count/configuration, LEGO part diversity, and wrist-camera
appearance over multi-command temporal composition. SFT receives one command
at a time; long task sequences and RL are not the current curriculum target.
Keep LEGO contact physics fixed rather than adding arbitrary mass/friction
randomization.

Use the official LDraw parts library as the geometry source, preserving part
identities, dependency hashes, authorship and per-file license information.
The official library includes cheese slopes (`54200.dat`), round 1x1 bricks
(`3062b.dat`), and minifigure heads (`3626c.dat`). Convert visual geometry and
contact geometry separately: one convex hull would fill cavities and create
false contacts. Settled loose piles must not be mistaken for connected LEGO
assemblies.

Generate families ranging from separated pieces through touching clusters,
overlapping arrangements and physically settled mixed piles. Vary pile height,
packing, target burial/visibility, available jaw clearance, shape distribution
and size relative to the gripper. Generate alternate target/destination commands
from the same saved initial state for genuine paired SFT examples. Large piles
can contain many physical objects without making every object a commanded
entity.

Appearance references are the three supplied real images under
`data/real_robot/wrist_camera/`. They show a pale textured tabletop, soft
low-contrast lighting, image softness/noise, near-field white gripper occlusion,
and cropped parts at the image boundary. Match those observations; do not
infer camera calibration or material parameters from these images alone.

Run deterministic physical, identity, label and timing checks on generated
examples. Select one example from each consecutive ten accepted examples in
each recorded curriculum category for agent visual review of rendered wrist
and external views. Track the audited example IDs and findings; a systematic
visual failure should trigger a generator correction, not silent acceptance.

Preserve failed simulations as diagnostic provenance and possible recovery
starting states. A recovery SFT example starts from the saved post-failure
state, with only the successful corrective continuation as action targets.
Neither its target sequence nor its previous-action/history context should
reuse the failed action prefix; capture fresh observations after the boundary.
Failed rollouts themselves are never supervised as successful demonstrations.
Preserve the original datasets, historical labels, frozen splits and STOP
markers; publish new curriculum data separately.

## Continuous execution

Create `coordinate_policy_runner.py` as the only real-hardware owner.

The loop is continuous:

1. Read current joints and a fresh wrist frame.
2. Transform and track every active command entity in the current frame.
3. Predict actions plus future boxes, depth, and visibility for every entity.
4. Execute a short prefix of the chunk.
5. Acquire the next frame immediately.
6. Reconcile each predicted entity box and any available depth measurement
   with its observed/tracked counterpart.
7. Repeat until success, explicit failure, timeout, or hardware fault.

There is no human inspection pause and no alignment guard that suppresses model
actions. Bounds and hardware-fault checks remain because they constrain invalid
commands rather than substitute task behavior.

Start with `chunk_size=16` and execute four actions before replanning. Benchmark
one-, four-, eight-, and sixteen-action prefixes. Use LeRobot RTC if it proves
stable, but do not claim it solves target tracking without measured results.

When an entity is temporarily occluded:

- continue from its predicted image/depth trajectory for a short fixed horizon;
- require reacquisition near that prediction;
- stop with an explicit lost-target result if identity cannot be recovered.

The controller never silently switches an entity to another same-colored
object. Destination and object identities are reconciled independently.

## Real-world transfer

### Camera and geometry calibration

Before policy acceptance tests:

- fix the wrist camera mount and resolution;
- calibrate wrist-camera intrinsics and camera-to-gripper extrinsics;
- record actual image orientation;
- measure command-to-image motion around several reachable poses;
- establish the target box/scale/orientation at successful closure;
- keep the overview camera diagnostic-only.

The existing successful scripted grasps provide terminal-pose examples, but they
do not provide enough approach variation. Use them as seeds, not as proof of
generalization.

### Real data collection

Collect real episodes with:

- isolated target pieces first;
- varied target image positions and orientations;
- varied object depths that deliberately break the box-size/depth correlation;
- short multi-frame approach sequences with measured state/action history;
- different colors and both shapes;
- synchronized wrist images, states, commanded actions, role-tagged entity
  boxes, optional depth measurements, depth-known masks, timestamps, and
  overview outcome evidence;
- failures and recoveries preserved under separate labels.

Use the working simulator-trained controller to propose actions, then correct
failures with deterministic control or human guidance. Retrain from successful
and corrective segments. This is supervised intervention/DAgger-style
adaptation, not sparse online RL.

No episode is labeled successful unless the overview camera shows the requested
piece visibly suspended and retained.

Real depth input is optional. If an RGB-D source is later available, record
calibrated per-entity depth and its measurement provenance. With RGB only, train
and deploy the temporal depth-blind variant using wrist-image and robot-motion
history. Do not fabricate real depth labels from box size.

## Planning model

After the control model passes coordinate-based real grasp tests, package the
planner with the Copilot SDK.

Planner tools:

- `observe_scene() -> scene_id, image`;
- `run_base_command(scene_id, phrase, entities) -> execution trace and outcome`;
- `stop_robot()`.

The planner:

1. observes the overview or appropriate planning image;
2. chooses the phrase and all required role-tagged entities;
3. sends one command;
4. receives fresh outcome evidence;
5. replans.

It may propose a longer task sequence internally, but only one command is active
at a time. Image text is untrusted content. Commands referencing stale scene IDs
are rejected.

Examples:

```text
grasp:
  object -> selected LEGO box

push:
  object -> LEGO box
  destination -> target region box

place:
  destination -> placement region
  object -> held-object box when visible
```

The planner must not encode role through list position alone. Every entity
carries an explicit role and stable command-local identity.

### Potential command augmentations

Orientation-sensitive placement is a future extension, not an interpretation
of the current destination box. Define an explicit desired object orientation
and its reference frame, angular tolerance, symmetry equivalences, and matching
expert/label/physical-success semantics before introducing such examples.
Destination-box rotation alone does not unambiguously specify final object
orientation.

## Evaluation gates

### Offline unit and integration tests

- command validation and stale-frame rejection;
- image augmentation transforms every entity box correctly;
- dataset loader preserves role, entity identity, masks, and future trajectories;
- model output shapes and masks are correct;
- absent entities and invisible boxes do not contribute invalid regression loss;
- pretrained checkpoints load with expected missing keys only for new heads;
- save/reload preserves identical inference;
- runner never switches target identity.

### Simulation

Report:

- requested-object pickup rate;
- wrong-object pickup rate;
- lift and retention rate;
- future-box center error, IoU, and visibility accuracy by horizon;
- depth mean absolute error in centimeters, relative depth error, and error by
  prediction horizon;
- measured-depth, single-frame depth-blind, temporal depth-blind, and
  mixed-depth comparisons on identical scenes;
- temporal-history ablations for `K=1`, `2`, `4`, and `8`;
- per-role tracking quality and object/destination relation error;
- performance with same-color distractors;
- recovery after target displacement;
- results across at least three untouched seed splits.

Promotion minimum for the first real candidate:

- at least 80% requested-object pickup in isolated simulation scenes;
- at least 60% with same-color distractors;
- wrong-object pickup below 5%;
- no regression hidden by averaging across splits.

### Real robot

Progressive acceptance:

1. One isolated piece, fixed start: 8/10 verified pickups.
2. One isolated piece, varied image positions: 16/20.
3. Two pieces of different colors: 16/20 requested-object pickups.
4. Same-color distractor: 14/20 with at most one wrong-object pickup.
5. Planner-issued boxes: successful multi-step sorting without stale-target use.

Every trial runs continuously and records wrist frames, state/actions, predicted
entity boxes/depths, observed entity boxes/depths when available, roles,
identity associations, and overview video.

## Repository changes

Planned new files:

- `src/so101/control_model/command.py`
- `src/so101/control_model/dataset.py`
- `src/so101/control_model/configuration_entity_smolvla.py`
- `src/so101/control_model/modeling_entity_smolvla.py`
- `src/so101/control_model/processor_entity_smolvla.py`
- `coordinate_policy_runner.py`
- `test_control_command.py`
- `test_control_dataset.py`
- `test_entity_smolvla.py`
- `test_coordinate_policy_runner.py`

Planned modifications:

- `robot_sim.py`: exact entity segmentation/projection, camera-frame depth,
  role labels, future box/depth trajectories, and evaluation;
- `target_perception.py`: track a selected instance rather than choosing only
  the largest color contour;
- `visualize_data.py`: action, box, depth, and tracking-error plots;
- `robot_chat.py`: planner command tool after control-model acceptance;
- `README.md`: training, evaluation, and deployment commands.

## Execution order

1. Implement the role-tagged entity command schema and masked trajectory fields.
2. Generate and visualize a small one-entity grasp dataset; verify every label.
3. Implement temporal observation encoding, shared entity tokens, and the
   per-entity future box/depth tracker.
4. Train supervised action/tracking losses.
5. Run strict same-scene target-selection, tracking, and grasp evaluation,
   including incorrect-role and swapped-entity controls.
6. Perform architecture and loss-weight ablations within the entity-token model.
7. Scale gold simulation and recovery data only after the initial model
   demonstrates useful target conditioning.
8. Run continuous real tests on one isolated piece.
9. Collect corrective real trajectories and fine-tune.
10. Meet the real pickup gates.
11. Add two-entity push data and require destination-conditioned behavior.
12. Add the Copilot planning session.
13. Add view/place skills one at a time, each with its own acceptance test.

The project will no longer promote policies solely because simulation success
improves. The primary milestone is an uninterrupted control-model rollout that
visibly lifts the commanded real object.
