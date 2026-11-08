"use strict";
const $ = id => document.getElementById(id);
const basePath = document.documentElement.dataset.basePath || "";
let offset = 0, total = 0, request = 0, detail = null;
let detailRequest = 0, previousTrajectory = null, nextTrajectory = null;
let moviePreparation = null;
let wristPlayback = null, playbackWanted = false;
const limit = 24;
let trainingContext = "";
const filters = ["task", "difficulty", "object_count", "expert_outcome", "expert_acceptance"];
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
function endpoint(params, run = trainingContext) {
  return `${basePath}/api/data/${run ? encodeURIComponent(run) : "all"}?${new URLSearchParams(params)}`;
}
function text(tag, value) { const node = document.createElement(tag); node.textContent = value; return node; }
const percent = value => `${Math.round(100 * value)}%`;
function image(item, view, frame) { return endpoint({view, item: item.id, image: frame}); }
function preview(item, view, frame, caption) {
  const figure = document.createElement("figure"), img = document.createElement("img");
  img.loading = "lazy"; img.alt = caption; img.src = endpoint({view, item: item.id, image: frame}, "");
  img.addEventListener("error", () => { img.removeAttribute("src"); img.alt = `${caption}: unavailable`; }, {once: true});
  figure.append(img, text("figcaption", caption)); return figure;
}
function renderCard(item, view) {
  const card = document.createElement("button"); card.className = "data-example";
  const difficulty = item.difficulty ? `${item.difficulty.level} - ${item.difficulty.name}` : "Unrated legacy";
  card.append(text("h3", `${item.task} / ${item.target}`), text("p", `${difficulty} | ${item.object_count} objects`));
  const images = document.createElement("div"); images.className = "previews";
  if (view === "rollouts" && item.evidence_status === "available") {
    images.append(preview(item, view, "0", "Actual policy wrist frame"));
  } else {
    if (view === "examples") card.append(text("p", item.generated_at
      ? `Generated: ${item.generated_at}` : "Generation date not recorded"));
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
    card.append(text("p", `Demonstration: ${item.expert_outcome}`));
    if (item.expert_acceptance) card.append(text("p", item.expert_acceptance === "single-pass-v1"
      ? "Clean single-pass demonstration" : "Legacy demonstration (may include retries)"));
    if (item.difficulty) card.append(text("p", item.difficulty.factors.join("; ")));
  }
  card.addEventListener("click", () => {
    trainingContext = ""; $("data-run").value = "";
    showDetail(item, view).catch(e => error(e.message));
  });
  return card;
}
async function load() {
  const version = ++request, params = {offset, limit, view: "examples"};
  filters.forEach(id => { params[id] = $(id).value || "all"; });
  error(""); $("mode-note").textContent = "Newest dated examples first; legacy examples with no generation date follow. Previews show source demonstrations, not policy rollouts.";
  try {
    const result = await get(endpoint(params, ""));
    if (version !== request) return;
    total = result.total;
    $("data-grid").replaceChildren(...result.items.map(item => renderCard(item, params.view)));
    $("data-count").textContent = `${total.toLocaleString()} matching examples`;
    if (result.unavailable_source_count) $("mode-note").textContent +=
      ` ${result.unavailable_source_count.toLocaleString()} catalog references have no locally available episode metadata and are not shown.`;
    const labels = {"single-pass-v1": "Clean single-pass demonstration",
      "legacy final-success (may include retries)": "Legacy demonstration (may include retries)",
      success: "Successful", failed: "Failed", unknown: "Not recorded"};
    for (const field of ["object_count", "expert_outcome", "expert_acceptance"]) {
      const selected = $(field).value || "all";
      $(field).replaceChildren(...[["all", {object_count: "Any number", expert_outcome: "All results", expert_acceptance: "All qualities"}[field]],
        ...(result.facets?.[field] || []).map(value => [value, labels[value] || value])].map(([value, label]) => {
          const option = text("option", label); option.value = value; return option;
        }));
      $(field).value = selected;
    }
    $("page-status").textContent = total ? `${offset + 1}-${Math.min(offset + limit, total)} of ${total.toLocaleString()}` : "No matches";
    $("previous").disabled = offset === 0;
    $("next").disabled = offset + limit >= total;
  } catch (e) { if (version === request) error(e.message); }
}
function showFrame(seek = true) {
  const frame = Number($("frame").value);
  const wrist = $("detail-image"), source = image(detail, detail.view, frame);
  if (!$("detail-preview").hidden && (!wrist.getAttribute("src") || wrist.complete !== false)) {
    if (wrist.getAttribute("src") !== source) wrist.src = source;
  }
  $("frame-value").textContent = `${frame + 1}/${detail.frames.length} | policy/source step ${detail.frames[frame].step}`;
  if (seek && wristPlayback) {
    wristPlayback.originTime = Date.now();
    wristPlayback.originStep = detail.frames[frame].step;
  }
  if (seek && Number.isFinite($("detail-movie").duration)) {
    $("detail-movie").currentTime = Math.min($("detail-movie").duration,
      detail.frames[frame].step / (detail.fps || detail.metadata.fps || 10));
  }
}
function updatePlayButton() {
  const video = $("detail-movie"), combined = !!video.getAttribute("src");
  $("play-movie").disabled = combined ? !detail?.movie_available : !detail?.frames.length;
  $("play-movie").textContent = `${playbackWanted ? "Pause" : "Play"} ${combined ? "combined view" : "wrist view"}`;
}
function stopWristPlayback() {
  if (wristPlayback) clearTimeout(wristPlayback.timer);
  wristPlayback = null;
}
function startWristPlayback() {
  if (!detail?.frames.length) {
    playbackWanted = false;
    updatePlayButton();
    return;
  }
  if (Number($("frame").value) >= detail.frames.length - 1) $("frame").value = 0;
  const selected = detail, version = detailRequest;
  const fps = selected.fps || selected.metadata.fps || 10;
  const state = {
    originTime: Date.now(), originStep: selected.frames[Number($("frame").value)].step, timer: null,
  };
  wristPlayback = state;
  function tick() {
    if (version !== detailRequest || !$("detail").open || wristPlayback !== state) return;
    const step = state.originStep + (Date.now() - state.originTime) * fps / 1000;
    let frame = 0;
    while (frame + 1 < selected.frames.length && selected.frames[frame + 1].step <= step) frame++;
    if (Number($("frame").value) !== frame || !state.timer) {
      $("frame").value = frame;
      showFrame(false);
    }
    if (frame === selected.frames.length - 1) {
      stopWristPlayback();
      playbackWanted = false;
      $("detail-image").removeAttribute("src");
      updatePlayButton();
      return;
    }
    state.timer = setTimeout(tick, Math.max(16, Math.min(100,
      (selected.frames[frame + 1].step - step) * 1000 / fps)));
  }
  tick();
  updatePlayButton();
}
function stopMovie() {
  const video = $("detail-movie");
  moviePreparation = null;
  stopWristPlayback();
  playbackWanted = false;
  video.onloadedmetadata = null;
  video.pause(); video.removeAttribute("src"); video.load(); video.hidden = true;
  updatePlayButton();
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
function prepareMovie() {
  if (moviePreparation?.version === detailRequest) return moviePreparation.promise;
  const version = detailRequest, selected = detail, video = $("detail-movie");
  const promise = loadMovie(version, selected, video);
  moviePreparation = {version, promise};
  return promise;
}
async function playMovie() {
  const video = $("detail-movie");
  if (video.getAttribute("src")) {
    if (playbackWanted || !video.paused) {
      playbackWanted = false;
      video.pause();
    } else {
      playbackWanted = true;
      try {
        await video.play();
      } catch (e) {
        playbackWanted = false;
        throw e;
      }
    }
    updatePlayButton();
    return;
  }
  if (wristPlayback) {
    stopWristPlayback();
    playbackWanted = false;
    updatePlayButton();
  } else {
    playbackWanted = true;
    startWristPlayback();
  }
  if (detail.movie_available && !moviePreparation) prepareMovie();
}
async function loadMovie(version, selected, video) {
  updatePlayButton();
  $("movie-status").textContent = "Preparing synchronized views (first playback may take up to 3 minutes)...";
  try {
    const prepareURL = endpoint({view: selected.view, item: selected.id, movie: "prepare"});
    const deadline = Date.now() + 6 * 60 * 1000;
    while (true) {
      const status = await get(prepareURL);
      if (version !== detailRequest || !$("detail").open) return false;
      if (status.ready) break;
      if (Date.now() >= deadline) throw new Error("Playback preparation timed out. Retry playback.");
      $("movie-status").textContent = status.status === "waiting"
        ? "Waiting for another recording to finish preparing..."
        : "Preparing synchronized views (first playback may take up to 3 minutes)...";
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
    if (version !== detailRequest || !$("detail").open) return false;
    video.onloadedmetadata = () => {
      if (version !== detailRequest) return;
      const frame = selected.frames[Number($("frame").value)];
      video.currentTime = Math.min(video.duration,
        (frame?.step || 0) / (selected.fps || selected.metadata?.fps || 10));
      if (playbackWanted) video.play().catch(e => {
        if (version !== detailRequest) return;
        playbackWanted = false;
        $("movie-status").textContent = `Combined view ready, but playback could not resume: ${e.message}. Press Play to continue.`;
        updatePlayButton();
      });
    };
    stopWristPlayback();
    video.src = endpoint({view: selected.view, item: selected.id, movie: "1"});
    video.load();
    video.hidden = false; $("detail-preview").hidden = true;
    $("movie-status").textContent = selected.playback_description;
    updatePlayButton();
    return true;
  } catch (e) {
    if (version === detailRequest) {
      $("movie-status").textContent = `Combined view unavailable: ${e.message}. Recorded wrist playback remains available.`;
      video.onloadedmetadata = null;
      video.removeAttribute("src"); video.load(); video.hidden = true;
      $("detail-preview").hidden = !selected.frames.length;
      moviePreparation = null;
    }
    return false;
  } finally {
    if (version === detailRequest) updatePlayButton();
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
  updatePlayButton();
  $("movie-status").textContent = result.movie_available ? result.playback_description : "No retained trajectory recording is available for playback.";
  $("detail-title").textContent = `${item.task} / ${item.target}`;
  $("detail-kind").textContent = view === "rollouts"
    ? result.evidence_status === "available" ? `ACTUAL POLICY ROLLOUT | Step ${item.step} | assistance ${percent(item.assistance)} | ${result.scope}`
      : result.evidence_status === "expired" ? "POLICY SUMMARY ONLY: recording expired under retention limit."
        : "POLICY SUMMARY ONLY: this historical rollout did not record images or actions."
    : trainingContext ? "SOURCE DEMONSTRATION - training history below is for the selected run."
      : "SOURCE DEMONSTRATION - select a training context below to inspect run usage and policy recordings.";
  $("frame-label").hidden = !result.frames.length;
  $("detail-preview").hidden = !result.frames.length;
  $("detail-image").hidden = !result.frames.length;
  $("detail-overview").src = image({id: result.example_id}, "examples", "overview");
  $("overview-caption").textContent = view === "rollouts"
    ? "Source expert overview reference - not this rollout"
    : "Saved overview snapshot - static until synchronized video is ready";
  if (result.frames.length) {
    $("frame").max = result.frames.length - 1; $("frame").value = 0; showFrame();
  } else $("detail-image").removeAttribute("src");
  plotActions(result.actions);
  $("trace-note").textContent = result.actions.length ? "Commanded arm joint positions (degrees), five joints. Gripper excluded. Play the recorded wrist view immediately while synchronized video prepares. The source overview stays static until the combined video is ready; playback then continues there at the selected position. Reconstructed views are explicitly labeled." : "No historical trajectory can be reconstructed from summary metrics alone.";
  const {metadata, source_history, frames, actions, states, qpos, recorded_trajectories, ...summary} = result;
  $("detail-summary").textContent = JSON.stringify(summary, null, 2);
  $("detail-history").textContent = JSON.stringify(source_history, null, 2);
  $("detail-metadata").textContent = JSON.stringify(metadata, null, 2);
  if (!$("detail").open) $("detail").showModal();
  if (result.movie_available) prepareMovie();
}
filters.forEach(id => $(id).addEventListener("change", () => { offset = 0; load(); }));
$("data-run").addEventListener("change", () => {
  trainingContext = $("data-run").value;
  if (detail) showDetail({id: detail.example_id, task: detail.task, target: detail.target}, "examples").catch(e => error(e.message));
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
$("detail-movie").addEventListener("play", () => { playbackWanted = true; updatePlayButton(); });
$("detail-movie").addEventListener("pause", () => { playbackWanted = false; updatePlayButton(); });
$("detail-movie").addEventListener("ended", () => { playbackWanted = false; updatePlayButton(); });
$("detail-movie").addEventListener("error", () => {
  const video = $("detail-movie");
  if (video.getAttribute("src")) {
    $("movie-status").textContent = "Movie could not be loaded or decoded. Retry playback or refresh the example.";
    video.onloadedmetadata = null;
    video.removeAttribute("src"); video.load(); video.hidden = true;
    $("detail-preview").hidden = !detail?.frames.length;
    moviePreparation = null;
    if (detail?.frames.length) showFrame(false);
    if (playbackWanted) startWristPlayback();
    updatePlayButton();
  }
});
$("detail-movie").addEventListener("timeupdate", () => {
  if (!detail?.frames.length) return;
  const step = $("detail-movie").currentTime * (detail.fps || detail.metadata.fps || 10);
  let frame = 0;
  for (let i = 1; i < detail.frames.length && detail.frames[i].step <= step; i++) frame = i;
  if (Number($("frame").value) !== frame) { $("frame").value = frame; showFrame(false); }
});
$("frame").addEventListener("input", showFrame);
$("detail-image").addEventListener("load", () => {
  if (!detail?.frames.length || $("detail-preview").hidden) return;
  const wrist = $("detail-image"), source = image(detail, detail.view, Number($("frame").value));
  if (wrist.getAttribute("src") !== source) wrist.src = source;
});
$("detail-image").addEventListener("error", () => error("Recorded frame unavailable. Refresh to check recording retention or source files."));
$("detail-overview").addEventListener("error", () => {
  $("detail-overview").removeAttribute("src");
  $("detail-overview").alt = "Saved overview snapshot unavailable";
});
get(`${basePath}/api/runs`).then(result => {
  const runs = result.runs.filter(run => !run.error);
  for (const run of runs) {
    const option = text("option", run.name); option.value = run.id; $("data-run").append(option);
  }
}).catch(e => error(e.message));
load();
