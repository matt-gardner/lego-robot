# Original control-model proposal (historical)

Preserved as design history. See the developed
[control-model plan](../CONTROL_MODEL_PLAN.md) for the architecture proposal.

ok, here's a proposal.  i'm not sure how much access we have to changing smolvla architecture; presumably we can add
auxiliary heads / losses as necessary, but you'll have to keep me honest, because i don't know much about the underlying
details of what we're working with.

components:
- a "planning model", which is an API-based VLM (you, in this context, but eventually you encode this into a copilot
  session that can run outside of your loop).  this predicts a sequence of "base" actions that are carried out by the
  robot's "control model" (described next).  the "base" actions are defined in terms of a bounding box and single word
  or short phrase, like "grasp", "push", "view", "place", or similar.  the planning model will, given a high-level task and an
  image of the scene (like "sort the legos by color"), figure out the next command (phrase + bounding box) to give to the
  control model.  it can predict the whole sequence up front, but it will only send one command at a time to the control
  model, and it will replan regularly.

- the "control model" will translate those base commands into motor movements.  its input should be (1) the wrist image,
  (2) the current robot state, (3) the bounding box, and (4) the phrase.  it should predict (1) the next sequence of
  motor movements, and (2) the updated position of the bounding box after each movement.  there might be some edge cases
  where the bounding box goes off screen while trying to do a movement; we'll worry about that later.

we can train the control model with a combination of reconstruction error on the updated bounding box position and RL
reward on successfully picking up the object.  if we can create gold trajectories in simulation, we don't even need RL,
we just train directly on the next action sequence.

give me a detailed, concrete plan, in a new markdown file, for how you are going to build this.
