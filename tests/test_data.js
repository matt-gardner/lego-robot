"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

function page() {
  const elements = new Map();
  const timers = new Map();
  const storage = new Map();
  let now = 0, timerId = 0;
  const element = id => {
    if (!elements.has(id)) elements.set(id, {
      value: "run", open: true, disabled: false, hidden: false, attributes: {}, events: {},
      playCount: 0, duration: NaN, paused: true,
      addEventListener(name, callback) { this.events[name] = callback; },
      pause() { this.paused = true; }, load() {},
      async play() { this.playCount++; this.paused = false; },
      showModal() { this.open = true; },
      replaceChildren() {}, toggleAttribute() {}, append() {},
      get src() { return this.attributes.src; },
      set src(value) { this.attributes.src = value; },
      getAttribute(name) { return this.attributes[name]; },
      removeAttribute(name) { delete this.attributes[name]; },
    });
    return elements.get(id);
  };
  const context = vm.createContext({
    document: {
      documentElement: {dataset: {basePath: "/lego-robot", browserMode: "data"}}, getElementById: element,
      createElement: tag => ({tagName: tag, value: "", textContent: ""}),
      createElementNS: () => ({setAttribute() {}}),
    },
    fetch: () => new Promise(() => {}),
    setTimeout(callback, delay) {
      if (delay === 1000) return callback();
      const id = ++timerId;
      timers.set(id, callback);
      return id;
    },
    clearTimeout: id => timers.delete(id),
    Date: class extends Date { static now() { return now; } },
    console, URLSearchParams,
    crypto: {randomUUID: () => "3699f611-928c-4329-a839-927c141cd123"},
    localStorage: {getItem: key => storage.get(key) || null,
      setItem: (key, value) => storage.set(key, value), removeItem: key => storage.delete(key)},
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, "../src/so101/dashboard/lego.js"), "utf8"), context);
  vm.runInContext(fs.readFileSync(
    process.env.DATA_BROWSER_SOURCE || path.join(__dirname, "../src/so101/dashboard/data.js"),
    "utf8"), context);
  return {context, element, storage, run: code => vm.runInContext(code, context),
    advance(ms) {
      now += ms;
      const callbacks = [...timers.values()];
      timers.clear();
      callbacks.forEach(callback => callback());
    },
    timers,
  };
}

const response = (status, body, type = "application/json", redirected = false) => ({
  ok: status >= 200 && status < 300, status, redirected,
  headers: {get: () => type}, json: async () => body,
});

test("data viewer comments use immutable archive identity and actual sparse frame without keeping confirmed history", async () => {
  const p = page(), submissions = [];
  p.context.fetch = async (route, options) => {
    if (route.endsWith("/capabilities")) return response(200, {
      supported_target_kinds: ["example"], issue_categories: ["task", "camera"],
    });

    submissions.push(JSON.parse(options.body));
    return response(200, {request_id: "saved", status: "queued", owner_delivery: "queued", monitor_delivery: "delivered"});
  };
  await p.run(`legoFeedbackDataOpen({
    release: {kind: "examples", id: "release-hash", artifact_sha256: "artifact-hash"},
    item: {id: "original-example"},
    frames: [{frame_index: 0, timestamp: 0, wrist: true},
      {frame_index: 149, timestamp: 14.9, wrist: true, external: true}]
  })`);
  assert.equal(p.element("lego-feedback-status").textContent, "");
  p.element("frame").value = "1";
  p.element("lego-feedback-camera").value = "external";
  p.element("lego-feedback-kind").value = "comment";
  p.element("lego-feedback-text").value = "Only a few frames were recorded";
  await p.element("lego-feedback-form").events.submit({preventDefault() {}});
  assert.deepEqual(submissions[0].target, {kind: "example", id: "original-example",
    release_id: "release-hash", artifact_sha256: "artifact-hash"});
  assert.equal(submissions[0].frame_index, 149);
  assert.equal(submissions[0].timestamp, 14.9);
  assert.equal(submissions[0].camera, "external");
  p.run("legoFeedbackDataOpen(null)");
  assert.equal(p.element("lego-feedback-form").hidden, true);
  assert.equal(p.element("lego-feedback-retry").hidden, true);
  assert.equal(p.storage.size, 0);
});

test("an old delivered catalog comment disappears rather than appearing on a new example", async () => {
  const p = page(), routes = [];
  p.storage.set("lego-feedback-pending-v1", JSON.stringify({idempotency_uuid: "old-comment",
    target: {kind: "part", id: "11211", release_id: "catalog-release", artifact_sha256: "catalog-hash"},
    text: "An earlier catalog request"}));
  p.context.fetch = async route => {
    routes.push(route);
    return response(200, route.endsWith("/capabilities")
      ? {supported_target_kinds: ["example"], issue_categories: ["task"]}
      : {request_id: "earlier-request", status: "delivered", owner_delivery: "delivered", monitor_delivery: "delivered"});
  };
  await p.run('legoFeedbackDataOpen({release:{kind:"examples",id:"new-release",artifact_sha256:"new-hash"},item:{id:"new-example"},frames:[]})');
  assert.ok(routes.includes("/family/api/lego/feedback/old-comment"));
  assert.equal(p.element("lego-feedback-form").hidden, false);
  assert.equal(p.element("lego-feedback-retry").hidden, true);
  assert.equal(p.element("lego-feedback-receipt").textContent, "");
  assert.doesNotMatch(p.element("lego-feedback-status").textContent, /earlier-request|previous|saved/i);
  assert.equal(p.storage.size, 0);
});

test("unconfirmed feedback stays with its exact version and does not block a different version", async () => {
  const p = page();
  p.context.fetch = async route => response(200, {supported_target_kinds: ["example"], issue_categories: ["task"]});
  await p.run('legoFeedbackDataOpen({release:{kind:"examples",id:"release-A",artifact_sha256:"hash-A"},item:{id:"same-example"},frames:[]})');
  p.element("lego-feedback-text").value = "Unconfirmed comment";
  p.context.fetch = async route => route.endsWith("/capabilities")
    ? response(200, {supported_target_kinds: ["example"], issue_categories: ["task"]})
    : response(503, {detail: "Unavailable"});
  await p.element("lego-feedback-form").events.submit({preventDefault() {}});
  const pending = p.storage.get("lego-feedback-pending-v1");
  await p.run('legoFeedbackDataOpen({release:{kind:"examples",id:"release-B",artifact_sha256:"hash-B"},item:{id:"same-example"},frames:[]})');
  assert.equal(p.element("lego-feedback-form").hidden, false);
  assert.equal(p.element("lego-feedback-retry").hidden, true);
  assert.equal(p.storage.get("lego-feedback-pending-v1"), pending);
});

test("original paired camera playback buffers slow images without skipping or canceling frames", async () => {
  const p = page();
  p.context.fetch = async () => response(200, {...recording("current"), movie_available: false,
    original_recording: true, frames: [{step: 0}, {step: 1}, {step: 2}]});
  await p.run('showDetail({id: "current"}, "examples")');
  const wrist = p.element("detail-image"), external = p.element("detail-overview");
  wrist.complete = false;
  external.complete = false;
  await p.run("playMovie()");
  p.advance(5000);
  assert.equal(p.element("frame").value, 0);
  assert.ok(wrist.src.endsWith("image=0"));
  assert.ok(external.src.endsWith("image=external%3A0"));
  assert.match(p.element("movie-status").textContent, /Buffering.*no frames are skipped/);
  wrist.complete = true;
  p.advance(100);
  assert.equal(p.element("frame").value, 0);
  external.complete = true;
  p.advance(100);
  p.advance(100);
  assert.equal(p.element("frame").value, 1);
  p.advance(100);
  assert.equal(p.element("frame").value, 2);
  assert.ok(wrist.src.endsWith("image=2"));
  assert.ok(external.src.endsWith("image=external%3A2"));
  assert.equal(p.run("wristPlayback"), null);
  assert.equal(p.element("play-movie").textContent, "Play recorded views");
});

test("original archive prepares native video, autoplays on readiness and avoids per-frame HTTP while playing", async () => {
  const p = page(), routes = [];
  let ready;
  p.context.fetch = async route => {
    routes.push(route);
    if (route.includes("movie=prepare")) return new Promise(resolve => { ready = resolve; });
    return response(200, {...recording("native"), original_recording: true});
  };
  await p.run('showDetail({id:"native"}, "examples")');
  await p.run("playMovie()");
  assert.equal(p.run("wristPlayback"), null);
  assert.match(p.element("movie-status").textContent, /Preparing native video/);
  ready(response(200, {ready: true}));
  await p.run("moviePreparation.promise");
  p.element("detail-movie").duration = 2;
  p.element("detail-movie").onloadedmetadata();
  assert.equal(p.element("detail-movie").playCount, 1);
  const before = p.element("detail-overview").src;
  p.element("frame").value = 1;
  p.run("showFrame(false)");
  assert.equal(p.element("detail-overview").src, before);
  assert.equal(routes.filter(route => route.includes("movie=prepare")).length, 1);
});

test("native preparation recovers a transient gateway error without per-frame playback", async () => {
  const p = page();
  let attempts = 0;
  p.context.fetch = async route => route.includes("movie=prepare")
    ? ++attempts === 1 ? response(502, null, "text/html") : response(200, {ready: true})
    : response(200, {...recording("native"), original_recording: true});
  await p.run('showDetail({id:"native"}, "examples")');
  await p.run("moviePreparation.promise");
  assert.equal(attempts, 2);
  assert.equal(p.element("detail-movie").hidden, false);
  assert.equal(p.run("wristPlayback"), null);
});

test("failed native preparation resets Pause to Play and allows an exact retry", async () => {
  const p = page();
  let resolve;
  p.context.fetch = async route => route.includes("movie=prepare")
    ? new Promise(done => { resolve = done; })
    : response(200, {...recording("native"), original_recording: true});
  await p.run('showDetail({id:"native"}, "examples")');
  await p.run("playMovie()");
  const pending = p.run("moviePreparation.promise");
  resolve(response(500, {error: "encoder failed"}));
  await pending;
  assert.equal(p.run("playbackWanted"), false);
  assert.equal(p.run("moviePreparation"), null);
  assert.match(p.element("play-movie").textContent, /^Play/);
  assert.match(p.element("movie-status").textContent, /encoder failed.*Press Play to retry/);
  await p.run("playMovie()");
  assert.notEqual(p.run("moviePreparation"), null);
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

test("top-level browsing uses dataset features without a training run", async () => {
  const p = page(), routes = [];
  p.context.document.createElement = () => ({value: "", textContent: ""});
  for (const id of ["task", "difficulty", "object_count", "expert_outcome", "expert_acceptance"]) {
    p.element(id).value = "all";
  }
  p.element("task").value = "push";
  p.element("data-run").value = "irrelevant-training-run";
  p.context.fetch = async route => {
    routes.push(route);
    return response(200, {items: [], total: 0, facets: {}});
  };
  await p.run("load()");
  assert.match(routes[0], /^\/lego-robot\/api\/data\/all\?/);
  assert.match(routes[0], /task=push/);
  assert.doesNotMatch(routes[0], /[?&](usage|bucket|outcome)=|irrelevant-training-run/);
  assert.equal(p.element("data-count").textContent, "0 matching examples");
});

test("cards prominently label actual creation time and curriculum scope without inventing legacy dates", () => {
  const p = page();
  p.context.document.createElement = () => ({
    children: [], textContent: "", append(...nodes) { this.children.push(...nodes); },
    addEventListener() {},
  });
  p.run(`globalThis.card = renderCard({id:"new", task:"view", target:"teal LEGO 3005",
    object_count:2, split:"development", origin:"curriculum",
    tags:{category:"overlap",families:["brick","plate"],part_ids:["3005","3024"]}, frames:31,duration:3.1,
    generated_at:"2026-11-08T20:00:00Z", expert_outcome:"success"}, "examples")`);
  const created = p.run('card.children.find(node => node.className === "data-created")');
  assert.match(created.textContent, /^Date created:/);
  assert.equal(created.title, "2026-11-08T20:00:00Z");
  const labels = p.run("card.children.map(node => node.textContent)");
  assert.ok(labels.includes("Generated example · Development only"));
  assert.ok(labels.includes("Look at teal LEGO 3005"));
  assert.ok(labels.includes("Overlap recipe | 2 objects"));
  assert.ok(labels.includes("Bricks, Plates | Parts 3005, 3024"));
  assert.ok(labels.includes("3.1 seconds | 31 recorded frames"));
  assert.ok(!labels.some(value => /legacy|Demonstration:|single-pass/.test(value || "")));
  p.run('globalThis.legacy = renderCard({id:"old", task:"push", target:"beam", object_count:4}, "examples")');
  assert.equal(p.run('legacy.children.find(node => node.className === "data-created").textContent'),
    "Date created: not recorded");
});

test("preparation polls pending JSON before loading the exact selected recording", async () => {
  const p = page();
  const routes = [];
  const replies = [
    response(202, {ready: false, status: "preparing"}),
    response(202, {ready: false, status: "waiting"}),
    response(200, {ready: true, status: "ready"}),
  ];
  p.context.fetch = async route => { routes.push(route); return replies.shift(); };
  p.run('detail = {id: "e0b1925f38c0f55b2e70", view: "examples", movie_available: true, frames: [1], playback_description: "Views"};');
  await p.run("prepareMovie()");
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
  p.run('detail = {id: "original", view: "examples", movie_available: true, frames: []};');
  await p.run("prepareMovie()");
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

test("Play immediately animates wrist frames while combined preparation is still pending", async () => {
  const p = page();
  const routes = [];
  const item = {...recording(), frames: [{step: 0}, {step: 1}, {step: 2}, {step: 3}]};
  p.context.fetch = async route => {
    routes.push(route);
    if (route.includes("movie=prepare")) return new Promise(() => {});
    return response(200, item);
  };
  await p.run('showDetail({id: "original"}, "examples")');
  assert.equal(p.element("play-movie").disabled, false);
  assert.equal(p.element("play-movie").textContent, "Play wrist view");
  await p.run("playMovie()");
  p.advance(100);
  assert.equal(p.element("frame").value, 1);
  assert.ok(p.element("detail-image").src.endsWith("image=1"));
  assert.equal(p.element("play-movie").textContent, "Pause wrist view");
  assert.equal(routes.length, 2);
  await p.run("playMovie()");
  p.advance(100);
  assert.equal(p.element("frame").value, 1);
  assert.equal(p.element("play-movie").textContent, "Play wrist view");
});

test("combined video takes over playing wrist footage at the current position", async () => {
  const p = page();
  let complete;
  const item = {...recording(), frames: [{step: 0}, {step: 1}, {step: 2}, {step: 3}]};
  p.context.fetch = async route => route.includes("movie=prepare")
    ? new Promise(resolve => { complete = resolve; }) : response(200, item);
  await p.run('showDetail({id: "original"}, "examples")');
  await p.run("playMovie()");
  p.advance(200);
  assert.equal(p.element("frame").value, 2);
  complete(response(200, {ready: true}));
  await p.run("moviePreparation.promise");
  assert.equal(p.run("wristPlayback"), null);
  assert.equal(p.timers.size, 0);
  p.element("detail-movie").duration = 10;
  p.element("detail-movie").onloadedmetadata();
  assert.equal(p.element("detail-movie").currentTime, .2);
  assert.equal(p.element("detail-movie").playCount, 1);
  assert.equal(p.element("play-movie").textContent, "Pause combined view");
});

test("combined video preserves a paused wrist position without autoplay", async () => {
  const p = page();
  let complete;
  p.context.fetch = async route => route.includes("movie=prepare")
    ? new Promise(resolve => { complete = resolve; }) : response(200, recording());
  await p.run('showDetail({id: "original"}, "examples")');
  p.element("frame").value = 1;
  p.run("showFrame()");
  complete(response(200, {ready: true}));
  await p.run("moviePreparation.promise");
  p.element("detail-movie").duration = 10;
  p.element("detail-movie").onloadedmetadata();
  assert.equal(p.element("detail-movie").currentTime, .1);
  assert.equal(p.element("detail-movie").playCount, 0);
  assert.equal(p.element("play-movie").textContent, "Play combined view");
});

test("sparse recorded wrist frames follow their recorded step timing", async () => {
  const p = page();
  p.run('detail = {frames: [{step: 0}, {step: 16}, {step: 32}], metadata: {fps: 10}, view: "rollouts", id: "sparse"};');
  p.element("frame").value = 0;
  await p.run("playMovie()");
  p.advance(1500);
  assert.equal(p.element("frame").value, 0);
  p.advance(100);
  assert.equal(p.element("frame").value, 1);
  p.element("frame").value = 2;
  p.run("showFrame()");
  p.advance(100);
  assert.equal(p.element("frame").value, 2);
  assert.equal(p.run("wristPlayback"), null);
});

test("closing an example cancels wrist animation", async () => {
  const p = page();
  p.run('detail = {frames: [{step: 0}, {step: 1}, {step: 2}], metadata: {fps: 10}, view: "examples", id: "original"};');
  p.element("frame").value = 0;
  await p.run("playMovie()");
  p.element("detail").open = false;
  p.element("detail").events.close();
  p.advance(200);
  assert.equal(p.element("frame").value, 0);
  assert.equal(p.run("wristPlayback"), null);
  assert.equal(p.timers.size, 0);
});

test("failed combined preparation does not stop wrist playback", async () => {
  const p = page();
  let fail;
  p.context.fetch = async route => route.includes("movie=prepare")
    ? new Promise(resolve => { fail = resolve; })
    : response(200, {...recording(), frames: [{step: 0}, {step: 1}, {step: 2}]});
  await p.run('showDetail({id: "original"}, "examples")');
  await p.run("playMovie()");
  fail(response(500, {error: "Renderer failed"}));
  await p.run("moviePreparation.promise");
  p.advance(100);
  assert.equal(p.element("frame").value, 1);
  assert.match(p.element("movie-status").textContent, /wrist playback remains available/);
  assert.equal(p.element("play-movie").disabled, false);
});

test("actual saved recording plays wrist frames and hands off at the same position", {
  skip: !process.env.LIVE_RECORDING_JSON,
}, async () => {
  const p = page();
  const item = JSON.parse(fs.readFileSync(process.env.LIVE_RECORDING_JSON, "utf8"));
  let complete;
  p.context.fetch = async route => route.includes("movie=prepare")
    ? new Promise(resolve => { complete = resolve; }) : response(200, item);
  await p.run(`showDetail(${JSON.stringify({id: item.id, task: item.task, target: item.target})}, "examples")`);
  await p.run("playMovie()");
  p.advance(1000);
  const frame = Number(p.element("frame").value);
  assert.ok(frame > 0);
  assert.ok(p.element("detail-image").src.endsWith(`image=${frame}`));
  complete(response(200, {ready: true}));
  await p.run("moviePreparation.promise");
  p.element("detail-movie").duration = item.frames.length / item.metadata.fps;
  p.element("detail-movie").onloadedmetadata();
  assert.equal(p.element("detail-movie").currentTime, item.frames[frame].step / item.metadata.fps);
  assert.equal(p.element("detail-movie").playCount, 1);
});

test("slow wrist frames are allowed to finish rather than being repeatedly canceled", async () => {
  const p = page();
  p.run('detail = {frames: [{step: 0}, {step: 1}, {step: 2}, {step: 3}], metadata: {fps: 10}, view: "examples", id: "original"};');
  p.element("frame").value = 0;
  await p.run("playMovie()");
  const wrist = p.element("detail-image");
  wrist.complete = false;
  p.advance(100);
  assert.equal(p.element("frame").value, 1);
  assert.ok(wrist.src.endsWith("image=0"));
  p.advance(100);
  assert.ok(wrist.src.endsWith("image=0"));
  wrist.complete = true;
  wrist.events.load();
  assert.ok(wrist.src.endsWith("image=2"));
  p.run("stopMovie()");
});
