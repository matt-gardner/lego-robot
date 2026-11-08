"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

class Element {
  constructor(tag = "div") {
    this.tagName = tag;
    this.children = [];
    this.events = {};
    this.value = "";
    this.hidden = false;
    this.width = 800;
    this.height = 600;
    this.drawn = [];
  }
  set textContent(value) { this.content = `${value}`; this.children = []; }
  get textContent() { return (this.content || "") + this.children.map(child => child.textContent).join(""); }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) {
    this.children = children;
    if (children[0]?.tagName === "option") this.value = children[0].value;
  }
  addEventListener(name, callback) { this.events[name] = callback; }
  removeAttribute() {}
  setPointerCapture() {}
  showModal() { this.open = true; }
  close() { this.open = false; }
  getContext() {
    return {fillRect() {}, beginPath() {}, closePath() {}, fill() {}, stroke() {},
      moveTo: (x, y) => this.drawn.push([x, y]), lineTo: (x, y) => this.drawn.push([x, y])};
  }
}

async function catalog(feedback = null, detail = null, examples = false) {
  const elements = new Map(), requests = [];
  const storage = new Map();
  const get = id => {
    if (!elements.has(id)) elements.set(id, new Element());
    return elements.get(id);
  };
  get("lego-readiness").value = "ready";
  const context = vm.createContext({
    document: {documentElement: {dataset: {basePath: "/lego-robot", browserMode: examples ? "examples" : ""}},
      querySelector: () => new Element(),
      getElementById: get, createElement: tag => new Element(tag)},
    URLSearchParams, console, setInterval, clearInterval,
    crypto: {randomUUID: () => "3699f611-928c-4329-a839-927c141cd123"},
    localStorage: {getItem: key => storage.get(key) || null,
      setItem: (key, value) => storage.set(key, value), removeItem: key => storage.delete(key)},
    fetch: async (route, options) => {
      const query = new URL(route, "http://localhost").searchParams;
      requests.push({route, query, options});
      if (route.startsWith("/family/")) {
        return feedback ? feedback(route, options) : {ok: false, status: 401, json: async () => ({detail: "Sign in"})};
      }
      let value;
      if (!query.has("release")) value = {
        releases: examples ? [
          {id: "generated-new", kind: "examples", generated_at: "2026-11-08T18:00:00Z", publication_mtime_ns: 1},
          {id: "copied-old", kind: "examples", generated_at: "2026-11-08T17:00:00Z", publication_mtime_ns: 99},
        ] : [
          {id: "old", version: "old", kind: "parts", validated_parts: 7, publication_mtime_ns: 1, qualification_notices: []},
          {id: "invalid", version: "bad", kind: "parts", validated_parts: 40, publication_mtime_ns: 9, qualification_notices: [{status: "invalidated"}]},
          {id: "latest", version: "current", kind: "parts", validated_parts: 29, publication_mtime_ns: 2, qualification_notices: []},
        ], feedback: {reason: "Not configured"},
      };
      else if (query.get("preview") === "geometry") value = {
        meshes: [{vertices: [[0, 0, 0], [1, 0, 0], [0, 1, 1]], faces: [[0, 1, 2]], color: [0.8, 0.25, 0.1, 1]}],
      };
      else if (query.has("item")) value = detail || {
        release: {kind: "parts", id: "a".repeat(64), artifact_sha256: "b".repeat(64)}, item: {id: "3005", part_id: "3005", description: "Brick", effective_status: "verified"},
        variants: [], frames: [],
      };
      else value = {
        release: {qualification_notices: []}, facets: {family: ["brick", "plate"], footprint: ["1 x 1"], decoration: ["undecorated"]},
        items: [{id: "3005", part_id: "3005", description: "Brick", mjcf: "part.xml", effective_status: "verified", readiness: "ready"}],
        total: 1, limit: 24,
      };
      return {ok: true, json: async () => value};
    },
  });
  for (const filename of ["part-viewer.js", "lego.js"]) {
    vm.runInContext(fs.readFileSync(path.join(__dirname, "../src/so101/dashboard", filename), "utf8"), context);
  }
  await new Promise(resolve => setImmediate(resolve));
  return {get, requests, storage, evaluate: code => vm.runInContext(code, context)};
}

test("defaults to latest usable catalog and effective validated parts with image cards", async () => {
  const ui = await catalog();
  assert.equal(ui.evaluate("legoRelease"), "latest");
  const query = ui.requests.find(request => request.query.get("release") === "latest").query;
  assert.equal(query.get("readiness"), "ready");
  assert.equal(ui.get("lego-family").children[1].textContent, "Bricks");
  const card = ui.get("lego-grid").children[0];
  assert.equal(card.children[0].tagName, "img");
  assert.match(card.children[0].src, /^\/lego-robot\/api\/lego\?/);
  assert.match(card.children[0].src, /preview=image/);
});

test("recorded examples preserve generation order instead of selecting latest file copy", async () => {
  const ui = await catalog(null, null, true);
  assert.equal(ui.evaluate("legoRelease"), "generated-new");
  assert.match(ui.get("lego-catalog-note").textContent, /undated legacy/);
  assert.match(ui.get("lego-grid").textContent, /Generation date not recorded/);
});

test("type, footprint and decoration filters combine and reset pagination", async () => {
  const ui = await catalog();
  ui.get("lego-readiness").value = "not_ready";
  ui.get("lego-family").value = "plate";
  ui.get("lego-footprint").value = "1 x 1";
  ui.get("lego-decoration").value = "undecorated";
  await ui.get("lego-family").events.change();
  await new Promise(resolve => setImmediate(resolve));
  const query = ui.requests.at(-1).query;
  assert.equal(query.get("readiness"), "not_ready");
  assert.equal(query.get("family"), "plate");
  assert.equal(query.get("footprint"), "1 x 1");
  assert.equal(query.get("decoration"), "undecorated");
  assert.equal(query.get("offset"), "0");
});

test("part click opens real geometry, dragging changes projection, and missing variants are explicit", async () => {
  const ui = await catalog();
  await ui.get("lego-grid").children[0].events.click();
  assert.equal(ui.get("lego-detail").open, true);
  assert.equal(ui.get("lego-geometry").hidden, false);
  assert.match(ui.get("lego-variant-note").textContent, /No declared print or sticker/);
  const canvas = ui.get("lego-geometry");
  const initial = JSON.stringify(canvas.drawn);
  canvas.events.pointerdown({button: 0, clientX: 0, clientY: 0, pointerId: 1});
  canvas.events.pointermove({clientX: 100, clientY: 40, pointerId: 1});
  assert.notEqual(JSON.stringify(canvas.drawn), initial);
  const yaw = ui.evaluate("legoViewer.yaw");
  canvas.events.pointercancel();
  canvas.events.pointermove({clientX: 200, clientY: 80, pointerId: 1});
  assert.equal(ui.evaluate("legoViewer.yaw"), yaw);
});

const feedbackResponse = value => ({ok: true, json: async () => value});
const capabilities = {supported_target_kinds: ["part", "example"], issue_categories: ["catalog", "geometry", "task", "camera"]};
const queuedReceipt = {request_id: "tracked", status: "queued", owner_delivery: "queued", monitor_delivery: "delivered"};
const flush = () => new Promise(resolve => setImmediate(resolve));

test("feedback requires Family identity instead of assuming website login", async () => {
  const ui = await catalog();
  await ui.get("lego-grid").children[0].events.click();
  await flush();
  assert.equal(ui.get("lego-feedback-form").hidden, true);
  assert.match(ui.get("lego-feedback-status").textContent, /Family identity/);
});

test("uncertain submission retains exact payload and key; checking and replay do not create duplicates", async () => {
  let posts = 0;
  const ui = await catalog(async (route, options) => {
    if (route.endsWith("/capabilities")) return feedbackResponse(capabilities);
    if (options.method === "POST" && ++posts === 1) throw new Error("Connection lost");
    return feedbackResponse(queuedReceipt);
  });
  await ui.get("lego-grid").children[0].events.click();
  await flush();
  ui.get("lego-feedback-text").value = "Geometry looks wrong";
  await ui.get("lego-feedback-form").events.submit({preventDefault() {}});
  assert.match(ui.get("lego-feedback-status").textContent, /Saved feedback has not been discarded/);
  const saved = JSON.parse([...ui.storage.values()][0]);
  assert.equal(saved.target.id, "3005");
  assert.equal(saved.target.release_id, "a".repeat(64));
  assert.equal(saved.target.artifact_sha256, "b".repeat(64));
  await ui.get("lego-feedback-check").events.click();
  assert.match(ui.get("lego-feedback-status").textContent, /Owner: queued; monitor: delivered/);
  await ui.get("lego-feedback-retry").events.click();
  const bodies = ui.requests.filter(request => request.options?.method === "POST").map(request => request.options.body);
  assert.equal(bodies.length, 2);
  assert.equal(bodies[0], bodies[1]);
  assert.equal(ui.storage.size, 1);
});

test("example feedback pins actual selected frame, timestamp and camera; missing cameras cannot submit", async () => {
  const detail = {release: {kind: "examples", id: "c".repeat(64), artifact_sha256: "d".repeat(64)},
    item: {id: "episode", task: "pick", split: "diagnostic", expert_acceptance: "diagnostic"}, variants: [],
    frames: [{frame_index: 0, timestamp: 0.15, wrist: "image.jpg"}]};
  const ui = await catalog(async route => feedbackResponse(route.endsWith("/capabilities") ? capabilities : queuedReceipt), detail);
  await ui.get("lego-grid").children[0].events.click();
  await flush();
  ui.get("lego-feedback-text").value = "Camera looks wrong";
  ui.get("lego-feedback-camera").value = "external";
  await ui.get("lego-feedback-form").events.submit({preventDefault() {}});
  assert.equal(ui.storage.size, 0);
  assert.match(ui.get("lego-feedback-status").textContent, /unavailable/);
  ui.get("lego-feedback-camera").value = "wrist";
  await ui.get("lego-feedback-form").events.submit({preventDefault() {}});
  const saved = JSON.parse([...ui.storage.values()][0]);
  assert.equal(saved.target.kind, "example");
  assert.equal(saved.frame_index, 0);
  assert.equal(saved.timestamp, 0.15);
  assert.equal(saved.camera, "wrist");
});

for (const status of [401, 403, 404, 409, 422, 503]) {
  test(`feedback HTTP ${status} stays visible and preserves the original saved key`, async () => {
    const ui = await catalog(async route => route.endsWith("/capabilities") ? feedbackResponse(capabilities) :
      {ok: false, status, json: async () => ({detail: "Rejected"})});
    await ui.get("lego-grid").children[0].events.click();
    await flush();
    ui.get("lego-feedback-text").value = "Check this";
    await ui.get("lego-feedback-form").events.submit({preventDefault() {}});
    assert.match(ui.get("lego-feedback-status").textContent, /Rejected/);
    assert.equal(ui.storage.size, 1);
    assert.equal(ui.get("lego-feedback-new").hidden, true);
  });
}

test("delivered and failed receipts distinguish owner enqueue from resolution and retain errors", async () => {
  const ui = await catalog(async route => feedbackResponse(route.endsWith("/capabilities") ? capabilities :
    {request_id: "tracked", status: "failed", owner_delivery: "failed", monitor_delivery: "delivered", error: "Owner unavailable"}));
  await ui.get("lego-grid").children[0].events.click();
  await flush();
  ui.get("lego-feedback-text").value = "Check this";
  await ui.get("lego-feedback-form").events.submit({preventDefault() {}});
  assert.match(ui.get("lego-feedback-status").textContent, /Owner: failed.*Owner unavailable/);
  ui.evaluate('legoFeedbackReceipt({request_id: "tracked", status: "delivered", owner_delivery: "delivered", monitor_delivery: "queued"})');
  assert.match(ui.get("lego-feedback-status").textContent, /not acknowledgment or resolution/);
});

test("storage failure prevents sending, and navigating preserves a saved comment's original target", async () => {
  const ui = await catalog(async route => feedbackResponse(route.endsWith("/capabilities") ? capabilities : queuedReceipt));
  await ui.get("lego-grid").children[0].events.click();
  await flush();
  ui.get("lego-feedback-text").value = "Keep this exact comment";
  ui.evaluate('localStorage.setItem = () => { throw new Error("Storage blocked"); }');
  await ui.get("lego-feedback-form").events.submit({preventDefault() {}});
  assert.match(ui.get("lego-feedback-status").textContent, /Storage blocked/);
  assert.equal(ui.requests.filter(request => request.options?.method === "POST").length, 0);
  const pending = {idempotency_uuid: "saved", target: {id: "prior-part"}, text: "Prior comment"};
  ui.storage.set("lego-feedback-pending-v1", JSON.stringify(pending));
  await ui.get("lego-grid").children[0].events.click();
  await flush();
  assert.equal(ui.get("lego-feedback-form").hidden, true);
  assert.match(ui.get("lego-feedback-receipt").textContent, /prior-part/);
  assert.match(ui.get("lego-feedback-status").textContent, /original target and context are preserved/);
});
