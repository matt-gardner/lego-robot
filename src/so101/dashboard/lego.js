"use strict";

const legoElement = id => document.getElementById(`lego-${id}`);
const legoBase = document.documentElement.dataset.basePath || "";
const legoExamples = document.documentElement.dataset.browserMode === "examples";
const legoViewer = new PartViewer(legoElement("geometry"));
let legoOffset = 0;
let legoRequest = 0;
let legoDetailRequest = 0;
let legoRelease = "";
let legoPlayback = null;
let legoFeedbackDetail = null;
let legoFeedbackBusy = false;
const legoFeedbackEndpoint = "/family/api/lego/feedback";
const legoFeedbackStorage = "lego-feedback-pending-v1";
async function legoFeedbackFetch(path, options = {}) {
  const response = await fetch(`${legoFeedbackEndpoint}${path}`, {
    credentials: "same-origin", cache: "no-store", ...options,
  });
  const value = await response.json();
  if (!response.ok) {
    const messages = {401: "Sign in with a Family identity; website login alone is not enough.",
      403: "Feedback origin authorization failed.", 404: "Feedback is disabled or this identity has no saved receipt.",
      409: "This key already belongs to different feedback. Do not send a replacement key.",
      422: "Feedback identity or recorded context was rejected.", 503: "Feedback target validation is unavailable."};
    throw new Error(`${messages[response.status] || `HTTP ${response.status}`} ${JSON.stringify(value.detail || value.error || value)}`);
  }
  return value;
}
function legoFeedbackPending() {
  const saved = localStorage.getItem(legoFeedbackStorage);
  return saved ? JSON.parse(saved) : null;
}
function legoFeedbackReceipt(receipt) {
  if (!receipt.request_id || !["queued", "delivered", "failed"].includes(receipt.status)) {
    throw new Error("Feedback returned an invalid receipt; keep the saved key and check again.");
  }
  legoElement("feedback-status").textContent =
    `Saved request ${receipt.request_id}. Owner: ${receipt.owner_delivery}; monitor: ${receipt.monitor_delivery}. ` +
    (receipt.error || receipt.monitor_error || "Enqueue is not acknowledgment or resolution.");
  legoElement("feedback-receipt").textContent = JSON.stringify(receipt, null, 2);
  legoElement("feedback-new").hidden = false;
}
async function legoFeedbackRecover(retry = false) {
  if (legoFeedbackBusy) return;
  legoFeedbackBusy = true;
  try {
    const pending = legoFeedbackPending();
    if (!pending) throw new Error("No saved feedback key. Open the part or example again.");
    legoElement("feedback-status").textContent = "Checking saved feedback...";
    const receipt = retry
      ? await legoFeedbackFetch("", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(pending)})
      : await legoFeedbackFetch(`/${pending.idempotency_uuid}`);
    legoFeedbackReceipt(receipt);
  } catch (error) {
    legoElement("feedback-status").textContent = `${error.message} Saved feedback has not been discarded. Check the original receipt or retry the exact payload; do not create a duplicate.`;
  } finally {
    legoFeedbackBusy = false;
  }
}
async function legoFeedbackOpen(detail, request) {
  legoFeedbackDetail = detail;
  legoElement("feedback-form").hidden = true;
  legoElement("feedback-receipt").textContent = "";
  legoElement("feedback-status").textContent = "Checking authenticated feedback availability...";
  try {
    const pending = legoFeedbackPending();
    for (const name of ["check", "retry"]) legoElement(`feedback-${name}`).hidden = !pending;
    legoElement("feedback-new").hidden = true;
    if (pending) {
      legoElement("feedback-receipt").textContent = JSON.stringify(pending, null, 2);
      legoElement("feedback-status").textContent = "A saved comment awaits receipt checking. Its original target and context are preserved, even if a different item is open.";
      return;
    }
    const capabilities = await legoFeedbackFetch("/capabilities");
    if (request !== legoDetailRequest) return;
    const kind = detail.release.kind === "parts" ? "part" : "example";
    if (!capabilities.supported_target_kinds?.includes(kind) || !capabilities.issue_categories?.length) {
      throw new Error("Feedback for this target is unavailable.");
    }
    legoElement("feedback-category").replaceChildren(...capabilities.issue_categories.map(category => {
      const option = legoText("option", category);
      option.value = category;
      return option;
    }));
    legoElement("feedback-category").value = kind === "part" ? "catalog" : "task";
    legoElement("feedback-target").textContent = `${kind} ${detail.item.id} / immutable release ${detail.release.id}`;
    legoElement("feedback-camera-label").hidden = kind !== "example";
    legoElement("feedback-camera").value = "";
    legoElement("feedback-text").value = "";
    legoElement("feedback-form").hidden = false;
    legoElement("feedback-status").textContent = "Feedback routes through the request monitor to the responsible worker.";
  } catch (error) {
    if (request === legoDetailRequest) legoElement("feedback-status").textContent = error.message;
  }
}
async function legoFeedbackSubmit(event) {
  event.preventDefault();
  if (legoFeedbackBusy) return;
  try {
    if (legoFeedbackPending()) throw new Error("Check the saved feedback before starting another comment.");
    const text = legoElement("feedback-text").value;
    if (!text.trim() || text.length > 8000) throw new Error("Enter a nonblank comment of at most 8000 characters.");
    const {release, item, frames} = legoFeedbackDetail;
    const payload = {idempotency_uuid: crypto.randomUUID(),
      target: {kind: release.kind === "parts" ? "part" : "example", id: item.id,
        release_id: release.id, artifact_sha256: release.artifact_sha256},
      kind: legoElement("feedback-kind").value, text,
      issue_category: legoElement("feedback-category").value};
    const camera = legoElement("feedback-camera").value;
    if (release.kind === "examples" && camera) {
      const frame = frames.find(value => value.frame_index === Number(legoElement("frame").value));
      if (!frame?.[camera]) throw new Error("The selected camera is unavailable at this recorded frame.");
      payload.frame_index = frame.frame_index;
      if (frame.timestamp != null) payload.timestamp = frame.timestamp;
      payload.camera = camera;
    }
    localStorage.setItem(legoFeedbackStorage, JSON.stringify(payload));
    legoElement("feedback-form").hidden = true;
    for (const name of ["check", "retry"]) legoElement(`feedback-${name}`).hidden = false;
    await legoFeedbackRecover(true);
  } catch (error) {
    legoElement("feedback-status").textContent = error.message;
  }
}
const legoMedia = query => `${legoBase}/api/lego?${new URLSearchParams(query)}`;
async function legoFetch(query) {
  const response = await fetch(legoMedia(query));
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || `HTTP ${response.status}`);
  return value;
}
function legoText(tag, text, className = "") {
  const node = document.createElement(tag);
  node.textContent = text;
  node.className = className;
  return node;
}
async function legoOpen(release, item) {
  clearInterval(legoPlayback);
  legoPlayback = null;
  const request = ++legoDetailRequest;
  legoFeedbackDetail = null;
  legoElement("feedback-form").hidden = true;
  for (const name of ["check", "retry", "new"]) legoElement(`feedback-${name}`).hidden = true;
  legoElement("feedback-status").textContent = "";
  legoElement("feedback-receipt").textContent = "";
  legoElement("geometry").hidden = true;
  legoElement("recording").hidden = true;
  legoElement("variants").replaceChildren();
  legoElement("part-title").textContent = "Loading part";
  legoElement("preview-status").textContent = "Loading hash-verified visual geometry...";
  if (!legoElement("detail").open) legoElement("detail").showModal();
  try {
    const detail = await legoFetch({release, item});
    if (request !== legoDetailRequest) return;
    const row = detail.item;
    legoFeedbackOpen(detail, request);
    legoElement("metadata").textContent = JSON.stringify(detail, null, 2);
    legoElement("part-title").textContent = row.description || row.task || row.id;
    const dimensions = row.dimensions_m?.map(value => (value * 1000).toFixed(1)).join(" x ");
    legoElement("part-summary").textContent = `Part ${row.part_id || row.id}` +
      (dimensions ? ` / ${dimensions} mm` : "") +
      (row.decoration ? ` / ${row.decoration}` : "") +
      (row.readiness ? ` / ${row.readiness === "ready" ? "Training ready" : "Not training ready"}` : "");
    legoElement("variant-note").textContent = detail.variants.length
      ? "LDraw-declared variants of this base part. Geometric equivalence and decorated simulation preparation are not qualified."
      : "No declared print or sticker variants are linked in this catalog yet.";
    legoElement("variants").replaceChildren(...detail.variants.map(variant => {
      const button = legoText("button", `${variant.description} / ${variant.part_id} / ${variant.decoration} / ${variant.status}`);
      button.addEventListener("click", () => legoOpen(release, variant.part_id));
      return button;
    }));
    if (detail.release.kind === "parts") {
      try {
        const geometry = await legoFetch({release, item, preview: "geometry"});
        if (request !== legoDetailRequest) return;
        legoElement("geometry").hidden = false;
        legoViewer.setGeometry(geometry);
        legoElement("preview-status").textContent = "Drag to rotate. Scroll to zoom. Arrow keys also rotate. Visual mesh, not a contact-geometry or graspability claim.";
      } catch (error) {
        if (request === legoDetailRequest) legoElement("preview-status").textContent = `Geometry unavailable: ${error.message}`;
      }
    } else {
      legoElement("preview-status").textContent = "Recorded example";
      legoElement("recording").hidden = false;
      legoElement("example-scope").textContent = `Split: ${row.split}. ${row.expert_acceptance}. ` +
        (detail.release.qualification?.scope || "Training membership is not inferred from success.");
      legoElement("frame").replaceChildren(...detail.frames.map(frame => {
        const option = legoText("option", `${frame.frame_index} / timestamp ${frame.timestamp ?? "not recorded"}`);
        option.value = frame.frame_index;
        return option;
      }));
      const showFrame = () => {
        const frame = detail.frames[Number(legoElement("frame").value)];
        legoElement("frame-identity").textContent = frame
          ? `${release} / ${row.id} / frame ${frame.frame_index} / timestamp ${frame.timestamp ?? "not recorded"}`
          : "No recorded frames. No substitute rendering.";
        for (const camera of ["wrist", "external"]) {
          const image = legoElement(camera);
          image.hidden = !frame?.[camera];
          image.removeAttribute("src");
          image.onerror = () => { legoElement("error").textContent = `Unable to load recorded ${camera} frame`; };
          if (frame?.[camera]) image.src = legoMedia({release, item, frame: frame.frame_index, camera});
        }
      };
      legoElement("frame").onchange = showFrame;
      legoElement("play-frames").onclick = () => {
        if (legoPlayback) {
          clearInterval(legoPlayback);
          legoPlayback = null;
          legoElement("play-frames").textContent = "Play recorded views";
          return;
        }
        legoElement("play-frames").textContent = "Pause";
        legoPlayback = setInterval(() => {
          const index = Number(legoElement("frame").value) + 1;
          if (index >= detail.frames.length) {
            clearInterval(legoPlayback);
            legoPlayback = null;
            legoElement("play-frames").textContent = "Play recorded views";
            return;
          }
          legoElement("frame").value = String(index);
          showFrame();
        }, 1000 / (row.fps || 10));
      };
      showFrame();
    }
  } catch (error) {
    if (request === legoDetailRequest) legoElement("preview-status").textContent = error.message;
  }
}
async function legoLoad() {
  const request = ++legoRequest;
  try {
    const release = legoRelease;
    if (!release) return;
    const query = legoExamples ? {release, offset: legoOffset} :
      {release, readiness: legoElement("readiness").value,
        family: legoElement("family").value, footprint: legoElement("footprint").value,
        decoration: legoElement("decoration").value, offset: legoOffset};
    const result = await legoFetch(query);
    if (request !== legoRequest) return;
    legoElement("error").textContent = "";
    const notices = result.release.qualification_notices;
    legoElement("catalog-note").textContent = legoExamples
      ? "Newest recorded generation dates first; undated legacy recordings follow in stable identity order. Original wrist and external recordings. Development diagnostics are not promoted to training data."
      : notices.length
      ? notices.map(notice => `${notice.status}: ${notice.reason}`).join(" ")
      : "Training ready means prepared for simulation-data generation, not proven robot graspability. Footprints follow the part description; they are not measured bounding boxes.";
    for (const [field, all] of [["family", "All types"], ["footprint", "All footprints"], ["decoration", "All decorations"]]) {
      const selected = legoElement(field).value;
      const options = [["", all], ...(result.facets?.[field] || []).map(value => [value,
        ({brick: "Bricks", plate: "Plates", tile: "Tiles", round: "Round parts", slope: "Slopes",
          technic: "Technic", minifigure: "Minifigure parts", other: "Other parts",
          undecorated: "Plain", print: "Printed", sticker: "Stickered"})[value] || value])];
      legoElement(field).replaceChildren(...options.map(([value, label]) => {
        const option = legoText("option", label);
        option.value = value;
        return option;
      }));
      legoElement(field).value = selected;
    }
    legoElement("count").textContent = `${result.total} ${legoExamples ? "recorded examples" : "parts"}`;
    legoElement("grid").replaceChildren(...result.items.map(row => {
      const button = document.createElement("button");
      button.className = "part-card";
      if (row.mjcf || legoExamples) {
        const image = document.createElement("img");
        image.loading = "lazy";
        image.alt = legoExamples ? `Original wrist image: ${row.target_label || row.task}` :
          `${row.description || row.part_id} visual geometry`;
        image.src = legoMedia(legoExamples ? {release, item: row.id, frame: 0, camera: "wrist"} :
          {release, item: row.id, preview: "image"});
        image.onerror = () => {
          image.replaceWith(legoText("span", "Preview unavailable", "part-placeholder"));
          legoElement("error").textContent = `Unable to load geometry preview for part ${row.part_id}`;
        };
        button.append(image);
      } else button.append(legoText("span", "No prepared geometry", "part-placeholder"));
      const info = legoText("span", "", "part-info");
      info.append(legoText("strong", row.description || row.task || row.id),
        legoText("span", legoExamples ? row.target_label || row.id : `Part ${row.part_id || row.id}`, "part-id"),
        legoText("span", legoExamples ? `Split: ${row.split}` :
          row.readiness === "ready" ? "Training ready" : "Not training ready", "part-badge"));
      if (legoExamples) info.append(legoText("span", row.generated_at
        ? `Generated: ${row.generated_at}` : "Generation date not recorded"));
      button.append(info);
      button.addEventListener("click", () => legoOpen(release, row.id));
      return button;
    }));
    legoElement("previous").disabled = legoOffset === 0;
    legoElement("next").disabled = legoOffset + result.limit >= result.total;
  } catch (error) {
    if (request === legoRequest) legoElement("error").textContent = error.message;
  }
}
async function legoRefresh() {
  try {
    const result = await legoFetch({});
    const usable = result.releases.filter(release => legoExamples ? release.kind === "examples" :
      release.kind === "parts" && release.validated_parts > 0 && !release.qualification_notices.length);
    const latest = legoExamples ? usable[0] :
      usable.sort((a, b) => b.publication_mtime_ns - a.publication_mtime_ns)[0];
    legoRelease = latest?.id || "";
    if (!legoRelease) {
      legoElement("count").textContent = legoExamples ? "No published recordings yet." : "No usable parts catalog is available yet.";
      legoElement("grid").replaceChildren();
      return;
    }
    if (!result.releases.length) {
      legoElement("count").textContent = "No published catalogs yet.";
      legoElement("grid").replaceChildren();
    }
    legoOffset = 0;
    await legoLoad();
  } catch (error) { legoElement("error").textContent = error.message; }
}
for (const id of ["readiness", "family", "footprint", "decoration"]) {
  legoElement(id).addEventListener("change", () => { legoOffset = 0; legoLoad(); });
}
legoElement("refresh").addEventListener("click", legoRefresh);
legoElement("previous").addEventListener("click", () => { legoOffset = Math.max(0, legoOffset - 24); legoLoad(); });
legoElement("next").addEventListener("click", () => { legoOffset += 24; legoLoad(); });
function legoStopPlayback() { clearInterval(legoPlayback); legoPlayback = null; ++legoDetailRequest; }
legoElement("close").addEventListener("click", () => { legoStopPlayback(); legoElement("detail").close(); });
legoElement("detail").addEventListener("cancel", legoStopPlayback);
legoElement("feedback-form").addEventListener("submit", legoFeedbackSubmit);
legoElement("feedback-check").addEventListener("click", () => legoFeedbackRecover());
legoElement("feedback-retry").addEventListener("click", () => legoFeedbackRecover(true));
legoElement("feedback-new").addEventListener("click", () => {
  if (legoFeedbackBusy || !legoFeedbackDetail) return;
  try {
    localStorage.removeItem(legoFeedbackStorage);
    legoFeedbackOpen(legoFeedbackDetail, legoDetailRequest);
  } catch (error) { legoElement("feedback-status").textContent = error.message; }
});
if (legoExamples) {
  document.querySelector(".catalog-toolbar").hidden = true;
  document.querySelector(".context").textContent = "Browse original recorded examples and their exact scene, command and acceptance evidence.";
}
legoRefresh();
