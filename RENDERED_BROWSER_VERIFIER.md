# Rendered-pair browser verifier

`src/so101/verify_rendered_browser.py` is a standalone standard-library
host consumer. It does not render, restart the dashboard, change source
admission, or acknowledge publication. The renderer owns those acknowledgments.

The fixed queue is
`/mnt/dockyard/lego-robot-data/data/lego_curriculum/rendered-physics-first-6d8de05`.
The renderer writes immutable `publication_requests/<source-release>.json`
referencing `releases/<derivative-release>/episodes.sqlite` and its published
qualification receipt. Requests use schema version 1 and kind
`lego-physical-render-publication-request`, two source example IDs, two dashboard
item IDs, frame counts, and two qualified frame-zero PNG hashes per command.

The verifier rejects symlinks, escaping/non-publication paths, mismatched
identities, checksums, qualifications, family/split or paired-frame populations.
It reconstructs and compares every URL against the fixed
`http://127.0.0.1:8765/lego-robot` base: two LEGO details, two data details,
and four frame-zero PNGs. No arbitrary URL, redirect, proxy, authentication,
movie preparation or inventory endpoint is accepted.

Each sequential GET has the existing four-second timeout and two-MiB body
limit. Requests are at most 64 KiB; complete receipts are at most eight MiB.
Archive/qualification/request bytes are rechecked before committing a receipt.
Success is an atomic `browser_receipts/<source-release>.json` with schema 1,
kind `lego-physical-render-browser-receipt`, status `verified`, request SHA,
timezone-aware verification time, and eight exact response entries. Metadata
bodies are retained; PNG entries retain only content type and byte SHA.
The renderer must read and validate this receipt before its `published` ack.
Endpoint evidence does not prove human browser interaction or unique data.

Failures are logged and written to `browser_failures/<source-release>.json`
with status `failed`, the request SHA when validated, and a 60-second retry
backoff. They never become success receipts. A five-second scan is serialized
by a file lock; unchanged successfully validated receipts are cached by
inode/size/mtime/ctime, so idle scans do not rehash large archives. A process
restart revalidates existing success receipts once. No cache/input cleanup
is performed.

The committed `scripts/lego-rendered-browser-verifier.service` runs the exact
installed script as `ubuntu`, outside the dashboard and renderer cgroups,
with 192-MiB memory, 25% CPU, eight tasks and loopback-only network access.
Only the receipt and failure directories are writable. Installation must
create those directories as `ubuntu:ubuntu` mode 0755; renderer-owned inputs
remain unchanged. Install the exact published script root-owned under
`/opt/lego-rendered-browser-verifier`, and record its commit/file SHA. The
service is enabled for ongoing queued candidates, not a scheduler cron.

Focused validation: `pytest tests/test_verify_rendered_browser.py`.
First-live proof must include the exact candidate request/archive SHA,
eight response identities and byte hashes, installed service PID/resource
evidence, and actual renderer receipt consumption. A frozen regression pair
is not a new unique-data claim.
