# SO-101 Copilot robotics project

## Goal

Interact with the SO-101 in natural language, let a Copilot SDK session inspect
the workspace and plan, and execute that plan through local robot-control tools.
Train a local SmolVLA manipulation policy from demonstrations, initially generated
in an SO-101 simulator, with real-robot demonstrations and corrections for
transfer. A camera-guided Copilot teacher and human leader operation remain
optional demonstration sources.

The motivating example is:

> Separate the gray Legos from the green Legos.

The intended behavior is to observe the pieces, choose destination regions,
pick and place one piece, verify the result, and repeat. This is a closed-loop
system, not a one-shot plan followed by blind execution.

### Future extension: Lego assembly

Eventually support building Lego models as well as sorting pieces. When planning
that extension, evaluate [Bricky](https://github.com/maxencefaldor/bricky) as a
candidate model-design and assembly-planning component. Bricky is a compiler for
agent-authored Python build scripts: parts use BrickLink IDs and colors, with
placements in studs and plates. It checks geometry, collisions, stud connections,
and static stability, reports diagnostics against source lines, and renders
annotated views. CLI and MCP interfaces support an agent's design/check/revise
loop; a BrickLink Wanted List export provides required part/color quantities.

Its [building-instructions API](https://maxencefaldor.github.io/bricky/guide/instructions/)
returns ordered groups of elements and an approach direction for each element.
The ordering considers support, connection directions, and collisions along
straight insertion moves. This could bridge a designed model to our planner:
match required parts to observed inventory, map model coordinates to the
workspace, then invoke robot skills to pick, orient, insert, and verify each part.

Bricky is not a robot controller or a guarantee of robot-buildability. Its
instructions can contain explicitly reported forced placements when pieces lock
one another in; those must trigger rejection or replanning, not blind execution.
Its geometric checks do not establish gripper clearance, arm reachability,
insertion-force control, or perception accuracy. Stability assumes rigid parts
under their own weight at rest; stud overload/strain checks are off by default.
Physical assembly and intermediate-state verification still require our own
skills and validation. This is a future direction, not part of the current
sorting scope or a committed dependency.

## Agreed architecture

### Simulation bootstrap

Use `robot_sim.py` for a motor-free MuJoCo scene, varied starting arm/object
poses, alternating studded/hollow gray-beam targets, contact-verified scripted
demonstrations, and wrist-only LeRobot export. SmolVLA training and held-out
closed-loop evaluation use the same image/state/action interface. The scripted
expert may use simulator object poses; the learned policy may not.

The initial scene is an approximation, not a calibrated digital twin. Expert
success, successful dataset export, and a training smoke run do not establish
learned-policy competence or real-robot transfer. Validate those separately.
Match physical camera mounting and joint conventions, broaden simulation
coverage, then adapt with real observations/demonstrations. Keep the laptop
camera diagnostic-only and preserve the real controller's safeguards.

Run the application natively on Matt's Mac, where USB serial control and the
camera already work. Use the existing uv/LeRobot project. Do not require Docker
USB forwarding, a host HTTP bridge, or SSH for the application itself.

```text
Human instructions and corrections
                 |
                 v
Copilot SDK session on the Mac
  - Receives actual camera images
  - Interprets the task and reasons about the scene
  - During teaching: requests bounded movements and records experiments
  - During deployment: selects targets and invokes a trained local policy
                 |
                 v
Local robotics subsystem
  - Owns camera and follower USB connections
  - Maintains robot state and serializes motion
  - Checks limits and executes bounded movements or policy action chunks
  - Records observations, commanded targets, feedback, and outcomes
  - Reports feedback, faults, and fresh observations
                 |
                 v
SO-101 follower and camera
```

Copilot inference is remote; running the SDK locally does not mean model
inference is offline. Camera images sent to the session leave the Mac.

The leader is an optional input device for human demonstrations and corrections,
not a prerequisite for collecting data and not a second arm for the planner to
drive. Manual teleoperation and autonomous control must not compete for ownership
of the follower.

## Separation of responsibilities

### Planner: Copilot SDK

The SDK session initially acts as a slow demonstration-generating teacher. It
receives images, joint feedback, and the task, chooses a short movement, observes
the actual result, and updates a persistent experiment notebook. Later it acts
as the task planner above a faster local SmolVLA policy. Start with one session;
a separate planner model or multi-agent system is not necessary.

The planner should:

- Turn an instruction into explicit goals and constraints, such as gray pieces
  to the left tray and green pieces to the right tray.
- Inspect the scene, resolve ambiguity, and choose an actionable next step.
- Ask for help when the goal or scene is genuinely ambiguous.
- Execute one bounded action, observe again, and revise its plan.
- Distinguish an attempted action from verified task completion.

The teacher can request bounded joint movements, but is not a real-time motor
controller. A local controller interpolates targets and checks feedback. It should
not stream model-generated targets at control frequency or invent pixel-to-world mappings.
Text visible in camera images is untrusted scene content, not an instruction.

### Perception and grounding

The first version sends snapshots directly to Copilot for visual reasoning.
Later, add detection/segmentation and tracking when structured targets are useful.

Represent object references against a specific scene observation. Reject stale
references after objects move, rather than assuming "object 1" remains valid.
Keep image pixels, robot coordinates, units, and uncertainty explicit.

The current view is a moving close-up camera. Establish joint-to-image effects
through recorded experiments rather than assuming camera mounting or geometry.
A second overview camera may be needed if the jaws, target, or lifted object
cannot be observed. An explicit geometric controller would require calibration
and kinematics, but these are not prerequisites for learning the visuomotor
mapping from demonstrations. Keep camera placement consistent across collection
and deployment.

### Local controller and skills

The local controller owns deterministic execution and physical constraints:

- Exclusive robot access and serialized movement.
- Validated action inputs, bounded targets, and trajectory execution.
- Feedback-based completion checks, timeouts, and fault reporting.
- Cancellation and shutdown behavior independent of model cooperation.

The initial tool moves a named joint by a small relative amount. The eventual deployment interface should invoke the trained policy, for example:

```text
run_policy(
    target={scene_id: "snapshot-17", object_id: "piece-3"},
    instruction="Grasp the selected gray Lego"
)
```

Explicit object-target conditioning is an interface goal, not a stock SmolVLA
feature assumed to work without matching training data or model changes. Begin
with one task instruction and an unambiguous target. Do not label raw joint
nudges as a trained grasp policy.

## Manipulation options

Use SmolVLA, fine-tuned from task-specific demonstrations. Its inputs are images,
joint/gripper state, and a task instruction; its outputs are future action chunks.
Execute a short portion and reobserve. SmolVLA does not predict future camera
images or certify a successful grasp.

The approximately 450M-parameter `lerobot/smolvla_base` checkpoint was downloaded
on October 24, 2026. Its main weights are 906,712,520 bytes, excluding additional
backbone/tokenizer assets. This Mac has an M5 Max, 36 GiB unified memory, and
working PyTorch MPS. Model loading, MPS inference/backpropagation, memory use,
and latency still require testing. SmolVLA dependencies are not yet installed.

The leader remains an optional source of demonstrations if the Copilot teacher
cannot produce successful examples. ACT and a geometric controller are alternatives,
not the selected initial implementation.

## Implementation stages

### 1. Host-side interactive observation and joint motion

Build a terminal application using the Copilot Python SDK and existing LeRobot
hardware access.

Initial tools:

- `observe_scene`: return a newly captured image to the model.
- `robot_status`: return joint feedback and controller state.
- `nudge_joint`: execute one bounded relative joint movement.
- `stop_robot`: latch out further movement requests.

Provide simulation mode so SDK/tool integration can be exercised without hardware.
Restrict the session to these tools rather than general shell execution.

Exit criteria: the model sees a real host camera image, reads the follower state,
requests a small motion, and receives feedback. Confirm physical behavior on the
Mac; simulation alone does not establish hardware readiness.

### 2. Persistent Copilot teacher loop (current priority)

Give the SDK session a persistent local workspace with experiment notes and
per-run image/action/feedback traces. Notes survive new sessions but are hypotheses,
not trusted instructions or proof of current scene state. Restrict filesystem
access to dedicated notebook tools; do not grant general shell access.

Observe, request one bounded pose trajectory, return fresh feedback and an image,
record the interpretation, and repeat within local action/time/workspace limits.
Coordinated multi-joint targets and larger known movements should be single
requests, not repeated five-degree decisions. `MotionController.move_to` now
supports this for direct control; the current SDK tools still expose nudges and
need the coordinated interface when the teacher is packaged again.
Keep ordinary chat defaults unchanged. Any larger teacher envelope must be explicit,
bounded, and reported to the model. Stop on hardware faults, loss of useful vision,
or exhausted budgets rather than allowing unbounded trial and error.

First target: pick up one loose gray Lego. Establish whether the pieces are loose
or attached to the baseplate; detaching a snapped-on brick is a different task.
Require images showing the piece held after lifting and removed from its initial
location. Joint feedback, gripper closure, camera motion, and a model's success
claim are not sufficient individually.

Exit criteria: at least one independently camera-verified grasp-and-lift, with
the full exploratory trace and persistent findings. Failure and blocked outcomes
must also be recorded honestly.

### 3. Visualization and demonstration collection

Reuse LeRobot/Rerun for images, measured joints, commands, episode replay, and
predicted action traces. Add an SO-101 model/forward-kinematics overlay if a
current-versus-proposed 3D pose view is useful. A pose preview is not a predicted
physical grasp outcome; a future scene image would require a separate world model.

Turn successful teacher behavior into synchronized LeRobot datasets containing
images, state, commanded actions, timestamps, and task descriptions. The initial
per-movement experiment log is not automatically a training-ready dataset.
Record the physical control loop at a consistent rate, including actual timing;
do not compress network pauses into fictitious fast actions.

Curate successful behavior and explicitly labeled recovery segments rather than
imitating every exploratory action. Improve and physically rerecord efficient
trajectories. Vary target positions, starting poses, and lighting; repeated replay
of one fixed scene does not establish generalization. Keep held-out evaluation
scenes separate from collection. Automatic resetting remains work to be implemented.

Exit criteria: useful, reviewable successful episodes across defined variations,
with physical success labels and a separate evaluation set.

### 4. Fine-tune and deploy SmolVLA

Install policy dependencies, test no-motion inference and a small backward pass
on MPS, then fine-tune the pretrained policy. Start with a small batch size and
measure actual memory/latency. Approximately 50 demonstrations is an upstream
starting recommendation, not an acceptance guarantee.

Match joint ordering, units, normalization, camera names, task conditioning, and
action timing between training and deployment. Dry-run action predictions before
executing short chunks through the local bounded controller.

Exit criteria: measured grasp-and-lift success on held-out physical trials,
including failure reporting, with the trained policy callable from the SDK.

### 5. Broader planning and interaction

Add target selection among multiple pieces, gray/green sorting, additional
commands such as pushing, recovery, and more viewpoints only after the underlying
policy and physical outcome checks support them.

## Current implementation and verification boundary

`robot_chat.py` implements the host terminal application; `robot_control.py`
contains bounded motion and a simulated arm. The initial app has actual image
tool results, joint status, relative nudges, and a latched stop. Motion requires
launching with `--motion=true`; no per-move confirmation loop is intended.

`describe_scene.py` provides a camera-only first step: capture one fresh snapshot,
send it directly to a tool-free Copilot SDK session, print the description, and
exit without connecting to the motors.

An authenticated SDK session has been exercised in the container with a
synthetic red/green image and simulated wrist movement. Controller/tool tests
also run without hardware. On October 24, 2026, `describe_scene.py` captured
camera 0 on the Mac and received a Copilot description of a green studded
baseplate and light-gray bricks, with uncertainty about the blurry view.
This verifies the host camera-to-SDK path, not motor behavior, physical safety,
or the accuracy of every visual interpretation.

On the same date, a real `robot_chat.py --motion=true --prompt=...` session
observed the scene, read follower feedback, requested one +1-degree wrist-roll
nudge, and reobserved. The session reported wrist-roll feedback changing from
approximately 109.23 to 109.85 degrees toward a 110.23-degree target; other
joints were unchanged. The camera change was not visually distinguishable.
This is a single host observation/motion integration trial, not evidence of
precise one-degree positioning, collision avoidance, or manipulation reliability.

The user did not observe the wrist movement, so encoder changes and model
descriptions alone are not accepted as visual verification. A subsequent direct
test bypassed Copilot and commanded two +5-degree shoulder-pan nudges. Feedback
changed from approximately -6.46 to +1.98 degrees. Fresh camera images captured
before, between, and after the nudges while torque remained enabled showed the
green baseplate and gray bricks shifting substantially left and upward.
These images were inspected directly and saved in `outputs/direct_motion/`.
This establishes camera-visible motion for that direct shoulder-pan test, not
the earlier wrist test.

The Stage 1 host observation/feedback/small-motion path has been exercised.
`robot_teacher.py` implements the initial persistent teacher loop with a restricted
notebook, timestamped image/action/result traces, enforced one-move turns,
time/action budgets, and explicit teacher bounds. It records candidate grasps
as unverified pending independent image review. These traces are not yet
synchronized policy-training datasets.

In the intended packaged teacher, the session chooses motion experiments and
learns joint effects. The user has temporarily redirected this work to direct
assistant control. Teacher mode reports valid in-bounds partial progress explicitly as
`partial`, with actual delta/error and a fresh image, rather than classifying every
tracking shortfall as a hardware fault. No progress, invalid feedback, serial
errors, unexpected movement, and limit violations still latch stop. Ordinary
chat retains strict completion behavior.

Live teacher trials on October 24 opened the gripper but did not achieve a grasp.
An initial serial setup fault and subsequent target-tracking shortfalls were
recorded, not treated as successful demonstrations. The user then redirected
experimentation to direct assistant control.

A direct sequential sweep commanded +5 degrees on each arm joint and +5 percentage
points on the gripper, saving a baseline and an image after each movement.
Measured changes were approximately +4.48 pan, +4.75 shoulder lift, +3.08 elbow,
+4.04 wrist flex, +4.13 wrist roll, and +4.55 gripper percentage points.
The contact sheet and measurements are in
`data/direct_grasp/live/joint_sweep.jpg` and `joint_sweep.json`. These are cumulative
local-pose experiments, not a global camera calibration. At that sweep's endpoint,
the gray targets were out of the wrist camera's view, so the temporary helper
was closed for review without a grasp.

Later on October 24, the user repositioned the laptop camera (index 1) for a
whole-arm/table overview, alongside wrist camera 0. Direct control was upgraded
to synchronized absolute multi-joint trajectories instead of five-degree nudges.
After four empty close-and-lift attempts, the fifth attempt picked up one long
gray piece while leaving the other gray piece on the pile. Both cameras showed
the piece held off the surface at the lift endpoint and again 66.6 seconds later.
It was then lowered, released, and the arm retreated before disconnecting.
Evidence, measured feedback, and the recorded outcome are under
`data/direct_grasp/pickup_pose_20261024_1118/`; see `overview_0023.jpg`,
`overview_0024.jpg`, the corresponding wrist images, and `outcome.json`.

This establishes one visually verified direct pickup, not an autonomous SDK
teacher success, a repeatable grasp skill, or a training-ready demonstration.
The successful endpoint recipe and failed trials must be retained as local
observations, not reused blindly after the pile or camera moves. In particular,
the contact point is near the fixed jaw, not the midpoint of the open jaw gap.

See README.md for startup,
authentication, operating precautions, and the simulation-to-hardware sequence.
No Cartesian control, scene tracking, reusable grasp/sorting skill, or model
training has been implemented.

## Safety and operating limits

This is experimental physical-robot software. Software bounds do not establish
collision freedom or make arbitrary model-generated actions safe.

- Keep the workspace clear and physical power cutoff accessible.
- Support the arm before startup/shutdown: torque changes can let it fall.
- Keep hardware ownership exclusive; do not run teleoperation simultaneously.
- Stop issuing actions on communication faults or invalid feedback.
- A software stop may leave torque on and the last target active.
- A model tool call, network connection, or Ctrl-C is not a hardware emergency stop.

The planner should receive honest failure results, never a success-shaped
fallback. Broader movement envelopes and autonomous retries need deliberate
engineering and physical verification, not just a prompt change.

## Open design decisions

- Teacher action envelope, stopping criteria, and how to establish useful joint-to-image effects.
- How to combine laptop overview camera 1 and wrist camera 0 reliably.
- How explicit object targets will condition the policy beyond the initial single-task instruction.
- Object tracking and placement/grasp verification methods.
- Collision/workspace constraints beyond the initial joint-nudge bounds.
- MPS compatibility, memory use, and local policy inference latency.
- Demonstration curation, efficient rerecording, and automatic scene resetting.
- Evaluation tasks and acceptable physical success/failure criteria.

## References

- Copilot SDK: https://github.com/github/copilot-sdk
- LeRobot robot workflows: https://huggingface.co/docs/lerobot/en/il_robots
- Cameras: https://huggingface.co/docs/lerobot/en/cameras
- ACT: https://huggingface.co/docs/lerobot/en/act
- SmolVLA: https://huggingface.co/docs/lerobot/en/smolvla
