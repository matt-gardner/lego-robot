"""Build and optionally serve a searchable catalog of SO-101 experiment data."""

from __future__ import annotations

import argparse
import html
import json
import os
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUTPUT = DATA / "visualization" / "index.html"


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def relative(path: Path, output: Path) -> str:
    return Path(os.path.relpath(path, output.parent)).as_posix()


def first_media(directory: Path, patterns: tuple[str, ...]) -> Path | None:
    for pattern in patterns:
        files = sorted(directory.glob(pattern))
        if files:
            return files[0]
    return None


def episode_record(path: Path, output: Path) -> dict:
    metadata = load_json(path)
    directory = path.parent
    preview = first_media(
        directory,
        ("diagnostic_overview.jpg", "overview*.jpg", "*overview*.jpg", "wrist/*.jpg", "*.jpg"),
    )
    video = first_media(directory, ("overview.mp4", "*.mp4"))
    success = metadata.get("success")
    if success is None:
        success = metadata.get("outcome") == "success" if "outcome" in metadata else None
    verification = metadata.get("verification")
    if success is None and isinstance(verification, dict):
        success = verification.get("success")
    task = metadata.get("task") or metadata.get("target_label") or metadata.get("target") or ""
    frames = metadata.get("frames")
    if frames is None and isinstance(metadata.get("episode"), dict):
        frames = metadata["episode"].get("frames")
    return {
        "kind": "episode",
        "name": directory.relative_to(ROOT).as_posix(),
        "group": directory.relative_to(DATA).parts[0],
        "task": task,
        "status": "success" if success is True else "failure" if success is False else "unlabeled",
        "frames": frames or "",
        "metadata": relative(path, output),
        "preview": relative(preview, output) if preview else "",
        "video": relative(video, output) if video else "",
    }


def dataset_record(path: Path, output: Path) -> dict:
    info = load_json(path)
    directory = path.parent.parent
    return {
        "kind": "dataset",
        "name": directory.relative_to(ROOT).as_posix(),
        "group": directory.relative_to(DATA).parts[0],
        "task": info.get("repo_id", ""),
        "status": "dataset",
        "frames": info.get("total_frames", ""),
        "episodes": info.get("total_episodes", ""),
        "fps": info.get("fps", ""),
        "metadata": relative(path, output),
        "preview": "",
        "video": "",
    }


def evaluation_record(path: Path, output: Path) -> dict:
    report = load_json(path)
    episodes = report.get("episodes", [])
    wrong = sum(
        isinstance(row.get("error"), str) and "unrequested" in row["error"]
        for row in episodes
    )
    return {
        "kind": "benchmark",
        "name": path.parent.relative_to(ROOT).as_posix(),
        "group": "benchmarks",
        "task": f"{report.get('success_rate', 0):.1%} success; {wrong} wrong-object lifts",
        "status": "success" if report.get("success_rate", 0) >= .5 else "failure",
        "frames": len(episodes),
        "episodes": "",
        "fps": report.get("fps", ""),
        "metadata": relative(path, output),
        "preview": relative(first_media(path.parent, ("overview_*.jpg",)), output)
        if first_media(path.parent, ("overview_*.jpg",)) else "",
        "video": "",
    }


def checkpoint_record(path: Path, output: Path) -> dict:
    checkpoint = path.parent.name
    return {
        "kind": "checkpoint",
        "name": path.relative_to(ROOT).as_posix(),
        "group": "outputs",
        "task": f"training checkpoint {checkpoint}",
        "status": "checkpoint",
        "frames": "",
        "episodes": "",
        "fps": "",
        "metadata": relative(path / "config.json", output)
        if (path / "config.json").exists() else relative(path, output),
        "preview": "",
        "video": "",
    }


def build(output: Path = OUTPUT) -> tuple[int, int, int, int]:
    output.parent.mkdir(parents=True, exist_ok=True)
    episode_paths = sorted({
        *DATA.glob("**/episode.json"),
        *DATA.glob("**/result.json"),
    })
    dataset_paths = sorted(DATA.glob("**/meta/info.json"))
    evaluation_paths = sorted(DATA.glob("**/evaluation.json"))
    checkpoint_paths = sorted((ROOT / "outputs").glob("**/pretrained_model"))
    records = [episode_record(path, output) for path in episode_paths]
    records.extend(dataset_record(path, output) for path in dataset_paths)
    records.extend(evaluation_record(path, output) for path in evaluation_paths)
    records.extend(checkpoint_record(path, output) for path in checkpoint_paths)
    records.sort(key=lambda record: (record["group"], record["kind"], record["name"]))
    payload = json.dumps(records).replace("</", "<\\/")
    rerun_example = (
        "uv run --with rerun-sdk python -m lerobot.scripts.lerobot_dataset_viz "
        "--repo-id local/DATASET --root /absolute/path/to/dataset --episode-index 0"
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SO-101 Data Catalog</title>
<style>
:root {{ color-scheme: dark; font-family: ui-sans-serif, system-ui, sans-serif; }}
body {{ margin: 0; background: #0b0f14; color: #e6edf3; }}
header {{ position: sticky; top: 0; z-index: 2; padding: 16px 22px; background: #111820ee; border-bottom: 1px solid #30363d; }}
h1 {{ margin: 0 0 10px; font-size: 22px; }}
.controls {{ display: flex; gap: 8px; flex-wrap: wrap; }}
input, select {{ background: #0d1117; color: inherit; border: 1px solid #30363d; border-radius: 6px; padding: 8px; }}
input {{ min-width: min(520px, 75vw); }}
.summary {{ margin-top: 9px; color: #9da7b3; font-size: 13px; }}
main {{ padding: 18px 22px; }}
.help {{ background: #111820; border: 1px solid #30363d; border-radius: 8px; padding: 12px; margin-bottom: 14px; }}
code {{ color: #79c0ff; overflow-wrap: anywhere; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th {{ position: sticky; top: 112px; background: #111820; text-align: left; }}
th, td {{ border-bottom: 1px solid #21262d; padding: 7px; vertical-align: middle; }}
tr:hover {{ background: #111820; }}
img {{ width: 120px; height: 80px; object-fit: cover; border-radius: 4px; background: #161b22; }}
a {{ color: #58a6ff; text-decoration: none; }}
.success {{ color: #3fb950; }} .failure {{ color: #f85149; }} .dataset, .checkpoint {{ color: #d2a8ff; }}
.unlabeled {{ color: #8b949e; }}
</style></head><body>
<header><h1>SO-101 experiment and training data</h1>
<div class="controls"><input id="query" placeholder="Filter by path, task, target, or status">
<select id="group"><option value="">All groups</option></select>
<select id="kind"><option value="">All entry types</option><option>episode</option><option>dataset</option><option>benchmark</option><option>checkpoint</option></select></div>
<div class="summary" id="summary"></div></header>
<main><div class="help"><b>Interactive timelines:</b> LeRobot includes a Rerun visualizer for exported datasets.
Run <code>{html.escape(rerun_example)}</code>. This catalog covers raw attempts, diagnostics, videos, and datasets that Rerun does not index.</div>
<table><thead><tr><th>Preview</th><th>Type</th><th>Status</th><th>Task / target</th><th>Frames</th><th>Path and files</th></tr></thead>
<tbody id="rows"></tbody></table></main>
<script>
const records={payload};
const rows=document.querySelector("#rows"), query=document.querySelector("#query");
const group=document.querySelector("#group"), kind=document.querySelector("#kind"), summary=document.querySelector("#summary");
for (const value of [...new Set(records.map(r=>r.group))].sort()) {{
  const option=document.createElement("option"); option.value=option.textContent=value; group.append(option);
}}
function esc(value) {{ const e=document.createElement("span"); e.textContent=String(value ?? ""); return e.innerHTML; }}
function render() {{
  const needle=query.value.toLowerCase();
  const shown=records.filter(r=>(!group.value||r.group===group.value)&&(!kind.value||r.kind===kind.value)&&
    (!needle||JSON.stringify(r).toLowerCase().includes(needle)));
  rows.innerHTML=shown.map(r=>`<tr>
    <td>${{r.preview?`<a href="${{esc(r.preview)}}" target="_blank"><img loading="lazy" src="${{esc(r.preview)}}"></a>`:""}}</td>
    <td>${{esc(r.kind)}}</td><td class="${{esc(r.status)}}">${{esc(r.status)}}</td>
    <td>${{esc(r.task)}}</td><td>${{esc(r.frames)}}${{r.episodes?` / ${{esc(r.episodes)}} episodes`:""}}</td>
    <td><div>${{esc(r.name)}}</div><a href="${{esc(r.metadata)}}" target="_blank">metadata</a>
    ${{r.video?` · <a href="${{esc(r.video)}}" target="_blank">video</a>`:""}}</td></tr>`).join("");
  const successes=shown.filter(r=>r.status==="success").length, failures=shown.filter(r=>r.status==="failure").length;
  summary.textContent=`${{shown.length.toLocaleString()}} / ${{records.length.toLocaleString()}} entries · ${{successes}} successes · ${{failures}} failures · generated catalog`;
}}
for (const element of [query,group,kind]) element.addEventListener("input",render);
render();
</script></body></html>"""
    output.write_text(document)
    return (
        len(episode_paths), len(dataset_paths), len(evaluation_paths), len(checkpoint_paths)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    episodes, datasets, evaluations, checkpoints = build(args.output.resolve())
    print(
        f"Wrote {args.output} with {episodes} episodes, {datasets} datasets, "
        f"{evaluations} benchmarks, and {checkpoints} checkpoints"
    )
    if args.serve:
        os.chdir(ROOT)
        print(f"Open http://127.0.0.1:{args.port}/{args.output.resolve().relative_to(ROOT)}")
        ThreadingHTTPServer(("127.0.0.1", args.port), SimpleHTTPRequestHandler).serve_forever()


if __name__ == "__main__":
    main()
