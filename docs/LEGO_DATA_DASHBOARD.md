# Workstream 2: Interactive catalog and curriculum dashboard

Approved by Matt November 8, 2026. A dedicated worker owns dashboard discovery,
browsing, feedback and activation, not collection or part qualification.

## Outcome

Show the available parts catalog and current/historical data generations.
Use `LEGO_PARTS_CATALOG.md` for the catalog contract and
`LEGO_CURRICULUM.md` for scene/example metadata. Show imported, prepared,
verified and unavailable parts separately; loading a mesh is not physical
qualification or demonstrated robot competence.

Each example needs a stable ID, generation/release ID, category, task, part IDs,
scene family, split, renderer/catalog fingerprints, geometry/configuration
tags, expert acceptance and visual-audit status. Filter by those fields and
show current work, rejected attempts and prior immutable generations without
rewriting historical labels. Distinguish legacy final-success, clean expert,
post-failure recovery and not-in-training examples.

Expose wrist and external video, selected-object identity, acceptance evidence,
and the failed parent of a recovery without presenting failed actions as SFT
targets. Support parts previews and explicit preparation failure reasons.

## Feedback

Attach comments and "looks wrong" feedback to exact immutable part/example IDs
and versions, optionally a frame/time and issue category. Persist feedback
with delivery state and idempotency identity. Route through the request monitor
to the appropriate catalog, realism or curriculum owner; confirm delivery or
show the actual failure. Do not silently acknowledge undelivered comments.

Preserve existing access controls. Coordinate missing Dockyard/request-monitor
functionality with the Work Scheduler rather than inventing an unauthenticated
agent-message endpoint or writing under Matt's identity.

## Implementation and activation

Reuse the existing dashboard and data browser, keeping current run views and
labels intact. The browser reads directory episode metadata and immutable EpisodeStorage
SQLite releases through its read-only adapter; hidden staging paths are excluded.

Physical-pair raster derivatives retain source example/family IDs and splits,
but have distinct artifact-bound release and dashboard view IDs. They must carry
`physical_pair_render` provenance and a matching `qualification.json` with
image/sample validation and storage hash. These views are labeled **Derived CPU
raster**, never original camera recordings or additional unique training data.
Physical source acceptance, image/sample qualification, visual audit and training
admission remain separate claims.

Publication verification uses `/api/lego?release=<artifact-bound-release-id>&item=<source-example-id>`
for provenance/frame identity, and `/api/data/all?archive_release=<artifact-bound-release-id>`
for the exact derivative-only population. Its returned dashboard item ID selects
details and native video through the existing `view=examples&item=<dashboard-id>`
endpoints. Native encoding reads archived derivative images without rerunning
physics. Original releases and their IDs/labels remain unchanged. A published
acknowledgment requires immutable publication and live projection/media evidence;
staging or physical success alone does not satisfy it.

Own frontend/backend dashboard changes, focused tests, managed activation and
live verification. Coordinate file ownership with realism and curriculum
workers. No GPU, training, hardware motion or deletion of historical data.
