"use strict";
const $ = id => document.getElementById(id);
const basePath = document.documentElement.dataset.basePath || "";
let offset = 0, total = 0, request = 0, detail = null;
let detailRequest = 0, previousTrajectory = null, nextTrajectory = null;
const limit = 24;
const filters = ["view", "task", "difficulty", "usage", "outcome", "bucket"];
const notes = {
  examples: "Overview and wrist previews are source expert demonstrations, NOT policy rollouts. Policy usage includes discarded groups; expert success and policy success are separate.",
  rollouts: "Newest policy rollout summaries first. Historical runs did not record actions/images. Future runs record the first rollout per training group, with sparse actual wrist frames and bounded retention; other rollouts remain summary-only.",
  solved: "One entry per source AND exact assistance level that previously passed every rollout. Membership is historical, not proof it remains solved or works unassisted. Counts below are for that assistance level.",
};
async function get(url) {
  const response = await fetch(url);
  if (!response.headers.get("content-type")?.includes("application/json")) {
    const message = response.redirected
      ? "Sign-in may have expired. Refresh the page and sign in again."
      : `Gateway returned HTTP ${response.status} instead of JSON. Retry or refresh the page.`;
    throw new Error(message);
  }
  const body = await response.json();
  if (!response.ok) {
    throw new Error(body.error || `HTTP ${response.status}`);
  }
  return body;
}
function error(message) { $("error").textContent = message; $("error").hidden = !message; }
function endpoint(params) {
  return `${basePath}/api/data/${encodeURIComponent($("data-run").value)}?${new URLSearchParams(params)}`;
}
function text(tag, value) { const node = document.createElement(tag); node.textContent = value; return node; }
const percent = value => `${Math.round(100 * value)}%`;
function image(item, view, frame) { return endpoint({view, item: item.id, image: frame}); }
function preview(item, view, frame, caption) {
  const figure = document.createElement("figure"), img = document.createElement("img");
  img.loading = "lazy"; img.alt = caption; img.src = image(item, view, frame);
  img.addEventListener("error", () => { img.removeAttribute("src"); img.alt = `${caption}: unavailable`; }, {once: true});
  figure.append(img, text("figcaption", caption)); return figure;
}
function renderCard(item, view) {
  const card = document.createElement("button"); card.className = "data-example";
  const difficulty = item.difficulty ? `${item.difficulty.level} - ${item.difficulty.name}` : "Unrated legacy";
  card.append(text("h3", `${item.task} / ${item.target}`), text("p", `${difficulty} | ${item.split}`));
  const images = document.createElement("div"); images.className = "previews";
  if (view === "rollouts" && item.evidence_status === "available") {
    images.append(preview(item, view, "0", "Actual policy wrist frame"));
  } else {
    const source = {...item, id: item.example_id || item.id};
    images.append(preview(source, "examples", "overview", view === "rollouts"
      ? "Expert scene reference (NOT this rollout)"
      : item.difficulty?.provenance === "task-difficulty-v1" ? "Initial scene overview" : "Expert diagnostic overview"));
    if (view !== "rollouts") images.append(preview(source, "examples", "initial", "Source initial wrist view"));
  }
  card.append(images);
  if (view === "rollouts") {
    card.append(text("p", `Step ${item.step} | ${item.success ? "PASS" : "FAIL"} | ${item.bucket} | assistance ${percent(item.assistance)}${item.discarded ? item.jerk_regularizer_applied ? " | task-policy group skipped; jerk loss applied" : " | discarded group" : ""}`));
    card.append(text("p", `Jerk ${Number(item.rms_jerk).toFixed(3)} | ${item.steps} policy steps | ${item.evidence_status === "available" ? "Recorded policy trajectory" : item.evidence_status === "expired" ? "Recording expired; summary retained" : "Summary only: no policy images/actions recorded"}`));
  } else {
    card.append(text("p", `Expert: ${item.expert_outcome} | Policy: ${item.passed}/${item.rollouts} passed | ${item.failed} failed`));
    card.append(text("p", view === "solved" ? `Perfect at step ${item.solved_at_step} with ${percent(item.assistance)} assistance`
      : `${item.used ? `Last sampled at step ${item.last_step}` : "Not sampled in this training log"} | ${item.solved_entries.length} solved assistance entries`));
    if (item.difficulty) card.append(text("p", item.difficulty.factors.join("; ")));
  }
  card.append(text("p", item.source));
  card.addEventListener("click", () => showDetail(item, view).catch(e => error(e.message)));
  return card;
}
async function load() {
  if (!$("data-run").value) return;
  const version = ++request, params = {offset, limit, search: $("data-search").value};
  filters.forEach(id => { params[id] = $(id).value; });
  error(""); $("mode-note").textContent = notes[params.view];
  try {
    const result = await get(endpoint(params));
    if (version !== request) return;
    total = result.total;
    $("data-grid").replaceChildren(...result.items.map(item => renderCard(item, params.view)));
    $("data-count").textContent = `${total.toLocaleString()} matching ${params.view} / step ${result.last_step ?? "none"}`;
    $("page-status").textContent = total ? `${offset + 1}-${Math.min(offset + limit, total)} of ${total.toLocaleString()}` : "No matches";
    $("previous").disabled = offset === 0;
    $("next").disabled = offset + limit >= total;
  } catch (e) { if (version === request) error(e.message); }
}
function showFrame(seek = true) {
  const frame = Number($("frame").value);
  $("detail-image").src = image(detail, detail.view, frame);
  $("frame-value").textContent = `${frame + 1}/${detail.frames.length} | policy/source step ${detail.frames[frame].step}`;
  if (seek && Number.isFinite($("detail-movie").duration)) {
    $("detail-movie").currentTime = Math.min($("detail-movie").duration,
      detail.frames[frame].step / (detail.fps || detail.metadata.fps || 10));
  }
}
function stopMovie() {
  const video = $("detail-movie");
  video.pause(); video.removeAttribute("src"); video.load(); video.hidden = true;
  $("play-movie").textContent = "Play side-by-side movie";
}
function renderTrajectoryNavigation() {
  const source = {id: detail.example_id, task: detail.task, target: detail.target, view: "examples"};
  const recordings = detail.recorded_trajectories.map(item => ({...item, view: "rollouts"}));
  const entries = [source, ...recordings];
  const position = detail.view === "rollouts" ? entries.findIndex(item => item.id === detail.id) : 0;
  previousTrajectory = position > 0 ? entries[position - 1] : null;
  nextTrajectory = position >= 0 ? entries[position + 1] : recordings.find(item => item.step > detail.step);
  if (position < 0) previousTrajectory = entries.filter(item => item.view === "examples" || item.step <= detail.step).at(-1);
  $("previous-trajectory").disabled = !previousTrajectory;
  $("next-trajectory").disabled = !nextTrajectory;
  $("source-expert").disabled = detail.view !== "rollouts";
  $("source-expert").onclick = () => showDetail(source, "examples").catch(e => error(e.message));
  const label = position === 0 ? "Source expert" : position > 0
    ? `Recording ${position}/${recordings.length} | step ${detail.step} | rollout ${detail.trajectory_index + 1}`
    : `Summary only | step ${detail.step}`;
  $("trajectory-position").textContent = `${label} | ${recordings.length} recorded / ${detail.source_rollout_total} logged policy rollouts`;
}
async function playMovie() {
  const version = detailRequest, selected = detail, video = $("detail-movie");
  if (video.getAttribute("src")) {
    if (video.paused) await video.play(); else video.pause();
    return;
  }
  $("play-movie").disabled = true;
  $("movie-status").textContent = "Preparing synchronized views (first playback may take up to 3 minutes)...";
  try {
    const prepareURL = endpoint({view: selected.view, item: selected.id, movie: "prepare"});
    const deadline = Date.now() + 6 * 60 * 1000;
    while (true) {
      const status = await get(prepareURL);
      if (version !== detailRequest || !$("detail").open) return;
      if (status.ready) break;
      if (Date.now() >= deadline) throw new Error("Playback preparation timed out. Retry playback.");
      $("movie-status").textContent = status.status === "waiting"
        ? "Waiting for another recording to finish preparing..."
        : "Preparing synchronized views (first playback may take up to 3 minutes)...";
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
    if (version !== detailRequest || !$("detail").open) return;
    video.src = endpoint({view: selected.view, item: selected.id, movie: "1"});
    video.hidden = false; $("detail-image").hidden = true;
    $("movie-status").textContent = selected.playback_description;
    await video.play();
  } catch (e) {
    if (version === detailRequest) {
      $("movie-status").textContent = `Playback unavailable: ${e.message}`;
      video.removeAttribute("src"); video.load(); video.hidden = true;
      $("detail-image").hidden = !selected.frames.length;
    }
  } finally {
    if (version === detailRequest) $("play-movie").disabled = !selected.movie_available;
  }
}
function plotActions(actions) {
  const svg = $("action-plot"); svg.replaceChildren(); svg.toggleAttribute("hidden", !actions.length);
  if (!actions.length) return;
  const values = actions.flatMap(action => action.slice(0, 5));
  let low = Infinity, high = -Infinity;
  for (const value of values) { low = Math.min(low, value); high = Math.max(high, value); }
  for (let joint = 0; joint < 5; joint++) {
    const line = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
    line.setAttribute("class", `joint-${joint}`); line.setAttribute("fill", "none");
    line.setAttribute("stroke-width", "1.5");
    line.setAttribute("points", actions.map((action, i) =>
      `${10 + 780 * i / Math.max(1, actions.length - 1)},${170 - 160 * (action[joint] - low) / Math.max(1, high - low)}`).join(" "));
    svg.append(line);
  }
}
async function showDetail(item, view) {
  const version = ++detailRequest;
  stopMovie();
  const result = await get(endpoint({view, item: item.id}));
  if (version !== detailRequest) return;
  detail = {...result, id: item.id, view};
  renderTrajectoryNavigation();
  $("play-movie").disabled = !result.movie_available;
  $("movie-status").textContent = result.movie_available ? result.playback_description : "No retained trajectory recording is available for playback.";
  $("detail-title").textContent = `${item.task} / ${item.target}`;
  $("detail-kind").textContent = view === "rollouts"
    ? result.evidence_status === "available" ? `ACTUAL POLICY ROLLOUT | Step ${item.step} | assistance ${percent(item.assistance)} | ${result.scope}`
      : result.evidence_status === "expired" ? "POLICY SUMMARY ONLY: recording expired under retention limit."
        : "POLICY SUMMARY ONLY: this historical rollout did not record images or actions."
    : "SOURCE EXPERT DEMONSTRATION - not a learned-policy trajectory";
  $("frame-label").hidden = !result.frames.length;
  $("detail-image").hidden = !result.frames.length;
  if (result.frames.length) {
    $("frame").max = result.frames.length - 1; $("frame").value = 0; showFrame();
  } else $("detail-image").removeAttribute("src");
  plotActions(result.actions);
  $("trace-note").textContent = result.actions.length ? "Commanded arm joint positions (degrees), five joints. Gripper excluded. The slider scrubs original recorded wrist frames; Play shows synchronized overview and wrist views. Reconstructed views are explicitly labeled." : "No historical trajectory can be reconstructed from summary metrics alone.";
  const {metadata, source_history, frames, actions, states, qpos, recorded_trajectories, ...summary} = result;
  $("detail-summary").textContent = JSON.stringify(summary, null, 2);
  $("detail-history").textContent = JSON.stringify(source_history, null, 2);
  $("detail-metadata").textContent = JSON.stringify(metadata, null, 2);
  if (!$("detail").open) $("detail").showModal();
}
filters.forEach(id => $(id).addEventListener("change", () => { offset = 0; load(); }));
$("data-run").addEventListener("change", () => { $("detail").close(); ++detailRequest; stopMovie(); offset = 0; load(); });
let searchTimer;
$("data-search").addEventListener("input", () => {
  clearTimeout(searchTimer); searchTimer = setTimeout(() => { offset = 0; load(); }, 250);
});
$("previous").addEventListener("click", () => { offset = Math.max(0, offset - limit); load(); });
$("next").addEventListener("click", () => { offset += limit; load(); });
$("refresh-data").addEventListener("click", load);
$("close-detail").addEventListener("click", () => $("detail").close());
$("detail").addEventListener("close", () => { ++detailRequest; stopMovie(); });
$("previous-trajectory").addEventListener("click", () => {
  if (previousTrajectory) showDetail(previousTrajectory, previousTrajectory.view).catch(e => error(e.message));
});
$("next-trajectory").addEventListener("click", () => {
  if (nextTrajectory) showDetail(nextTrajectory, nextTrajectory.view).catch(e => error(e.message));
});
$("play-movie").addEventListener("click", () => playMovie().catch(e => { $("movie-status").textContent = e.message; }));
$("detail-movie").addEventListener("play", () => { $("play-movie").textContent = "Pause movie"; });
$("detail-movie").addEventListener("pause", () => { $("play-movie").textContent = "Play side-by-side movie"; });
$("detail-movie").addEventListener("error", () => {
  if ($("detail-movie").getAttribute("src")) $("movie-status").textContent = "Movie could not be loaded or decoded. Retry playback or refresh the example.";
});
$("detail-movie").addEventListener("timeupdate", () => {
  if (!detail?.frames.length) return;
  const step = $("detail-movie").currentTime * (detail.fps || detail.metadata.fps || 10);
  let frame = 0;
  for (let i = 1; i < detail.frames.length && detail.frames[i].step <= step; i++) frame = i;
  if (Number($("frame").value) !== frame) { $("frame").value = frame; showFrame(false); }
});
$("frame").addEventListener("input", showFrame);
$("detail-image").addEventListener("error", () => error("Recorded frame unavailable. Refresh to check recording retention or source files."));
get(`${basePath}/api/runs`).then(result => {
  const runs = result.runs.filter(run => !run.error);
  for (const run of runs) {
    const option = text("option", run.name); option.value = run.id; $("data-run").append(option);
  }
  const canonical = runs.find(run => !run.archived && run.path.includes("trajectory_rl_all_assisted_lr1e4/"));
  if (canonical) $("data-run").value = canonical.id;
  if (!runs.length) throw new Error("No training runs found");
  return load();
}).catch(e => error(e.message));
