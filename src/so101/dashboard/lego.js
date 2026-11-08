"use strict";

const legoElement = id => document.getElementById(`lego-${id}`);
const legoBase = document.documentElement.dataset.basePath || "";
const legoViewer = new PartViewer(legoElement("geometry"));
let legoOffset = 0;
let legoRequest = 0;
let legoDetailRequest = 0;
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
  const request = ++legoDetailRequest;
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
    legoElement("metadata").textContent = JSON.stringify(detail, null, 2);
    legoElement("part-title").textContent = row.description || row.task || row.id;
    const dimensions = row.dimensions_m?.map(value => (value * 1000).toFixed(1)).join(" x ");
    legoElement("part-summary").textContent = `Part ${row.part_id || row.id}` +
      (dimensions ? ` / ${dimensions} mm` : "") +
      (row.decoration ? ` / ${row.decoration}` : "") +
      (row.effective_status ? ` / ${row.effective_status === "verified" ? "Validated" : row.effective_status}` : "");
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
      showFrame();
    }
  } catch (error) {
    if (request === legoDetailRequest) legoElement("preview-status").textContent = error.message;
  }
}
async function legoLoad() {
  const request = ++legoRequest;
  try {
    const release = legoElement("release").value;
    if (!release) return;
    const validated = legoElement("status").value === "verified";
    const result = await legoFetch({release, search: legoElement("search").value,
      effective_status: validated ? "verified" : "",
      status: validated ? "" : legoElement("status").value,
      qualification_status: legoElement("qualification").value, offset: legoOffset});
    if (request !== legoRequest) return;
    legoElement("error").textContent = "";
    const notices = result.release.qualification_notices;
    legoElement("identity").textContent = `Pinned identity: ${release}`;
    legoElement("catalog-note").textContent = notices.length
      ? notices.map(notice => `${notice.status}: ${notice.reason}`).join(" ")
      : "Showing the selected catalog snapshot. Validated parts exclude invalidated historical entries.";
    legoElement("count").textContent = `${result.total} ${validated ? "validated parts" : "matching entries"}`;
    legoElement("grid").replaceChildren(...result.items.map(row => {
      const button = document.createElement("button");
      button.className = "part-card";
      if (row.mjcf) {
        const image = document.createElement("img");
        image.loading = "lazy";
        image.alt = `${row.description || row.part_id} visual geometry`;
        image.src = legoMedia({release, item: row.id, preview: "image"});
        image.onerror = () => {
          image.replaceWith(legoText("span", "Preview unavailable", "part-placeholder"));
          legoElement("error").textContent = `Unable to load geometry preview for part ${row.part_id}`;
        };
        button.append(image);
      } else button.append(legoText("span", "No prepared geometry", "part-placeholder"));
      const info = legoText("span", "", "part-info");
      info.append(legoText("strong", row.description || row.task || row.id),
        legoText("span", `Part ${row.part_id || row.id}`, "part-id"),
        legoText("span", row.effective_status === "verified" ? "Validated" :
          row.effective_status || row.expert_acceptance || "Not recorded", "part-badge"));
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
    const previous = legoElement("release").value;
    const usable = result.releases.filter(release => release.kind === "parts" &&
      release.validated_parts > 0 && !release.qualification_notices.length);
    const latest = usable.sort((a, b) => b.publication_mtime_ns - a.publication_mtime_ns)[0];
    legoElement("release").replaceChildren(...result.releases.map(release => {
      const option = legoText("option", `${release.version} (${release.validated_parts} stored validated parts)` +
        (release.qualification_notices.length ? " - historical warning" : ""));
      option.value = release.id;
      return option;
    }));
    if (result.releases.some(release => release.id === previous)) legoElement("release").value = previous;
    else if (latest) legoElement("release").value = latest.id;
    legoElement("feedback").textContent = `Feedback unavailable: ${result.feedback.reason}`;
    if (!result.releases.length) {
      legoElement("count").textContent = "No published catalogs yet.";
      legoElement("grid").replaceChildren();
    }
    legoOffset = 0;
    await legoLoad();
  } catch (error) { legoElement("error").textContent = error.message; }
}
for (const id of ["release", "search", "status", "qualification"]) {
  legoElement(id).addEventListener("change", () => { legoOffset = 0; legoLoad(); });
}
legoElement("refresh").addEventListener("click", legoRefresh);
legoElement("previous").addEventListener("click", () => { legoOffset = Math.max(0, legoOffset - 24); legoLoad(); });
legoElement("next").addEventListener("click", () => { legoOffset += 24; legoLoad(); });
legoElement("close").addEventListener("click", () => { ++legoDetailRequest; legoElement("detail").close(); });
legoElement("detail").addEventListener("cancel", () => { ++legoDetailRequest; });
legoRefresh();
