"use strict";

const legoElement = id => document.getElementById(`lego-${id}`);
const legoBase = document.documentElement.dataset.basePath || "";
let legoOffset = 0;
let legoRequest = 0;
async function legoFetch(query) {
  const response = await fetch(`${legoBase}/api/lego?${new URLSearchParams(query)}`);
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || `HTTP ${response.status}`);
  return value;
}
async function legoLoad() {
  const request = ++legoRequest;
  try {
    const release = legoElement("release").value;
    if (!release) return;
    const result = await legoFetch({release, search: legoElement("search").value,
      status: legoElement("status").value, offset: legoOffset});
    if (request !== legoRequest) return;
    legoElement("error").textContent = "";
    legoElement("identity").textContent = `Pinned catalog/generation identity: ${release}`;
    legoElement("count").textContent = `${result.total} matching entries`;
    legoElement("grid").replaceChildren(...result.items.map(row => {
      const button = document.createElement("button");
      button.textContent = `${row.part_id || row.id} - ${row.description || row.task || ""} - ${row.status || row.expert_acceptance || "not recorded"}`;
      button.addEventListener("click", async () => {
        try {
          const detail = await legoFetch({release, item: row.id});
          legoElement("metadata").textContent = JSON.stringify(detail, null, 2);
          legoElement("frame").replaceChildren(...detail.frames.map(frame => {
            const option = document.createElement("option");
            option.value = frame.frame_index;
            option.textContent = `${frame.frame_index} / timestamp ${frame.timestamp ?? "not recorded"}`;
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
              if (frame?.[camera]) image.src = `${legoBase}/api/lego?${new URLSearchParams({
                release, item: row.id, frame: frame.frame_index, camera,
              })}`;
            }
          };
          legoElement("frame").onchange = showFrame;
          showFrame();
          legoElement("detail").showModal();
        } catch (error) { legoElement("error").textContent = error.message; }
      });
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
    legoElement("release").replaceChildren(...result.releases.map(release => {
      const option = document.createElement("option");
      option.value = release.id;
      option.textContent = `${release.kind} / ${release.version} / ${release.path}`;
      return option;
    }));
    if (result.releases.some(release => release.id === previous)) legoElement("release").value = previous;
    legoElement("feedback").textContent = `Feedback unavailable: ${result.feedback.reason}`;
    if (!result.releases.length) legoElement("count").textContent = "No published catalogs or SQLite releases found.";
    legoOffset = 0;
    await legoLoad();
  } catch (error) { legoElement("error").textContent = error.message; }
}
for (const id of ["release", "search", "status"]) {
  legoElement(id).addEventListener("change", () => { legoOffset = 0; legoLoad(); });
}
legoElement("refresh").addEventListener("click", legoRefresh);
legoElement("previous").addEventListener("click", () => { legoOffset = Math.max(0, legoOffset - 24); legoLoad(); });
legoElement("next").addEventListener("click", () => { legoOffset += 24; legoLoad(); });
legoElement("close").addEventListener("click", () => legoElement("detail").close());
legoRefresh();
