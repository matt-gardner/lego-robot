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

async function catalog() {
  const elements = new Map(), requests = [];
  const get = id => {
    if (!elements.has(id)) elements.set(id, new Element());
    return elements.get(id);
  };
  get("lego-readiness").value = "ready";
  const context = vm.createContext({
    document: {documentElement: {dataset: {basePath: "/lego-robot"}},
      getElementById: get, createElement: tag => new Element(tag)},
    URLSearchParams, console, setInterval, clearInterval,
    fetch: async route => {
      const query = new URL(route, "http://localhost").searchParams;
      requests.push({route, query});
      let value;
      if (!query.has("release")) value = {
        releases: [
          {id: "old", version: "old", kind: "parts", validated_parts: 7, publication_mtime_ns: 1, qualification_notices: []},
          {id: "invalid", version: "bad", kind: "parts", validated_parts: 40, publication_mtime_ns: 9, qualification_notices: [{status: "invalidated"}]},
          {id: "latest", version: "current", kind: "parts", validated_parts: 29, publication_mtime_ns: 2, qualification_notices: []},
        ], feedback: {reason: "Not configured"},
      };
      else if (query.get("preview") === "geometry") value = {
        meshes: [{vertices: [[0, 0, 0], [1, 0, 0], [0, 1, 1]], faces: [[0, 1, 2]], color: [0.8, 0.25, 0.1, 1]}],
      };
      else if (query.has("item")) value = {
        release: {kind: "parts"}, item: {id: "3005", part_id: "3005", description: "Brick", effective_status: "verified"},
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
  return {get, requests, evaluate: code => vm.runInContext(code, context)};
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
