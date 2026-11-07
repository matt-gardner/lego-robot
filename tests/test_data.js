"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

function page() {
  const elements = new Map();
  const element = id => {
    if (!elements.has(id)) elements.set(id, {
      value: "run", open: true, disabled: false, hidden: false, attributes: {}, events: {},
      playCount: 0, duration: NaN, paused: true,
      addEventListener(name, callback) { this.events[name] = callback; },
      pause() { this.paused = true; }, load() {},
      async play() { this.playCount++; this.paused = false; },
      showModal() { this.open = true; },
      replaceChildren() {}, toggleAttribute() {},
      get src() { return this.attributes.src; },
      set src(value) { this.attributes.src = value; },
      getAttribute(name) { return this.attributes[name]; },
      removeAttribute(name) { delete this.attributes[name]; },
    });
    return elements.get(id);
  };
  const context = vm.createContext({
    document: {documentElement: {dataset: {basePath: "/lego-robot"}}, getElementById: element},
    fetch: () => new Promise(() => {}),
    setTimeout: callback => callback(), clearTimeout() {}, Date, console, URLSearchParams,
  });
  vm.runInContext(fs.readFileSync(
    process.env.DATA_BROWSER_SOURCE || path.join(__dirname, "../src/so101/dashboard/data.js"),
    "utf8"), context);
  return {context, element, run: code => vm.runInContext(code, context)};
}

const response = (status, body, type = "application/json", redirected = false) => ({
  ok: status >= 200 && status < 300, status, redirected,
  headers: {get: () => type}, json: async () => body,
});

test("HTML gateway and sign-in responses are not parsed as JSON", async () => {
  const p = page();
  p.context.fetch = async () => ({
    ...response(504, null, "text/html"),
    json: () => assert.fail("Must not parse HTML"),
  });
  await assert.rejects(p.run('get("/test")'), /Gateway returned HTTP 504 instead of JSON/);
  p.context.fetch = async () => response(200, null, "text/html", true);
  await assert.rejects(p.run('get("/test")'), /Sign-in may have expired/);
});

test("JSON renderer errors retain their meaningful message", async () => {
  const p = page();
  p.context.fetch = async () => response(500, {error: "Playback rendering exceeded its 180-second limit"});
  await assert.rejects(p.run('get("/test")'), /180-second limit/);
});

test("playback polls pending JSON before loading the exact selected recording", async () => {
  const p = page();
  const routes = [];
  const replies = [
    response(202, {ready: false, status: "preparing"}),
    response(202, {ready: false, status: "waiting"}),
    response(200, {ready: true, status: "ready"}),
  ];
  p.context.fetch = async route => { routes.push(route); return replies.shift(); };
  p.run('detail = {id: "e0b1925f38c0f55b2e70", view: "examples", movie_available: true, frames: [1], playback_description: "Views"};');
  await p.run("playMovie()");
  assert.equal(routes.length, 3);
  assert.ok(routes.every(route => route.includes("item=e0b1925f38c0f55b2e70&movie=prepare")));
  assert.ok(p.element("detail-movie").src.endsWith("item=e0b1925f38c0f55b2e70&movie=1"));
  assert.equal(p.element("play-movie").disabled, false);
});

test("closing the detail stops polling without starting a movie", async () => {
  const p = page();
  let calls = 0;
  p.context.fetch = async () => {
    calls++;
    p.element("detail").open = false;
    return response(202, {ready: false, status: "preparing"});
  };
  p.run('detail = {id: "original", view: "examples", movie_available: true};');
  await p.run("playMovie()");
  assert.equal(calls, 1);
  assert.equal(p.element("detail-movie").src, undefined);
});

function recording(id = "e0b1925f38c0f55b2e70") {
  return {
    id, example_id: id, task: "grasp", target: "brown wide beam",
    movie_available: true, frames: [{step: 0}, {step: 1}],
    actions: [], metadata: {fps: 10}, recorded_trajectories: [],
    source_rollout_total: 0, source_history: [], playback_description: "Reconstructed overview and recorded wrist",
  };
}

test("opening an example starts preparation and immediately shows both saved views", async () => {
  const p = page();
  const routes = [];
  let complete;
  p.context.fetch = async route => {
    routes.push(route);
    if (route.includes("movie=prepare")) return new Promise(resolve => { complete = resolve; });
    return response(200, recording());
  };
  await p.run('showDetail({id: "e0b1925f38c0f55b2e70", task: "grasp", target: "brown wide beam"}, "examples")');
  assert.equal(routes.length, 2);
  assert.ok(routes[1].includes("movie=prepare"));
  assert.equal(p.element("detail-preview").hidden, false);
  assert.ok(p.element("detail-overview").src.includes("view=examples"));
  assert.ok(p.element("detail-overview").src.endsWith("image=overview"));
  assert.ok(p.element("detail-image").src.endsWith("image=0"));
  assert.match(p.element("overview-caption").textContent, /static/);
  assert.equal(p.element("detail-movie").playCount, 0);
  complete(response(200, {ready: true, status: "ready"}));
  await p.run("moviePreparation.promise");
  assert.equal(p.element("detail-preview").hidden, true);
  assert.equal(p.element("detail-movie").hidden, false);
  assert.equal(p.element("detail-movie").playCount, 0);
  await p.run("playMovie()");
  assert.equal(routes.length, 2);
  assert.equal(p.element("detail-movie").playCount, 1);
});

test("ready paused preview seeks both video views to the selected frame", async () => {
  const p = page();
  p.context.fetch = async route => response(200, route.includes("movie=prepare")
    ? {ready: true} : recording());
  await p.run('showDetail({id: "e0b1925f38c0f55b2e70"}, "examples")');
  await p.run("moviePreparation.promise");
  p.element("detail-movie").duration = 10;
  p.element("frame").value = 1;
  p.element("detail-movie").onloadedmetadata();
  assert.equal(p.element("detail-movie").currentTime, .1);
  await p.run("showFrame()");
  assert.equal(p.element("detail-movie").currentTime, .1);
});

test("policy preview labels the source overview as a reference, not this rollout", async () => {
  const p = page();
  const item = {...recording("rollout"), example_id: "source", evidence_status: "available"};
  p.context.fetch = async route => response(200, route.includes("movie=prepare") ? {ready: true} : item);
  await p.run('showDetail({id: "rollout", task: "grasp", target: "brown wide beam"}, "rollouts")');
  await p.run("moviePreparation.promise");
  assert.ok(p.element("detail-overview").src.includes("view=examples&item=source"));
  assert.match(p.element("overview-caption").textContent, /not this rollout/);
});

test("summary-only recordings do not start a renderer", async () => {
  const p = page();
  const routes = [];
  p.context.fetch = async route => {
    routes.push(route);
    return response(200, {...recording(), movie_available: false, frames: []});
  };
  await p.run('showDetail({id: "summary"}, "rollouts")');
  assert.equal(routes.length, 1);
  assert.equal(p.element("play-movie").disabled, true);
});

test("a failed eager render keeps both previews visible and permits retry", async () => {
  const p = page();
  p.context.fetch = async route => response(200, route.includes("movie=prepare")
    ? {ready: true} : recording());
  await p.run('showDetail({id: "e0b1925f38c0f55b2e70"}, "examples")');
  await p.run("moviePreparation.promise");
  p.element("detail-movie").events.error();
  assert.equal(p.element("detail-preview").hidden, false);
  assert.equal(p.element("detail-movie").hidden, true);
  assert.equal(p.element("detail-movie").src, undefined);
  assert.equal(p.run("moviePreparation"), null);
});


test("switching examples prevents the previous preparation from replacing the preview", async () => {
  const p = page();
  let completeFirst;
  p.context.fetch = async route => {
    if (route.includes("movie=prepare")) {
      if (route.includes("item=first")) return new Promise(resolve => { completeFirst = resolve; });
      return response(200, {ready: true});
    }
    return response(200, recording(route.includes("item=first") ? "first" : "second"));
  };
  await p.run('showDetail({id: "first"}, "examples")');
  const firstPreparation = p.run("moviePreparation.promise");
  await p.run('showDetail({id: "second"}, "examples")');
  await p.run("moviePreparation.promise");
  completeFirst(response(200, {ready: true}));
  assert.equal(await firstPreparation, false);
  assert.ok(p.element("detail-movie").src.includes("item=second"));
});
