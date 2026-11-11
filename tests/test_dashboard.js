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
    this.attributes = {};
    this.dataset = {};
    this.style = {setProperty() {}};
    this.value = "";
    this.checked = false;
    this.hidden = false;
  }
  set textContent(value) { this.content = `${value}`; this.children = []; }
  get textContent() { return (this.content || "") + this.children.map(child => child.textContent).join(""); }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.content = ""; this.children = [...children]; }
  setAttribute(name, value) { this.attributes[name] = value; }
  addEventListener(name, callback) { this.events[name] = callback; }
  querySelector(tag) {
    for (const child of this.children) {
      if (child.tagName === tag) return child;
      const found = child.querySelector(tag);
      if (found) return found;
    }
    return null;
  }
  getBoundingClientRect() { return {width: 0, height: 0}; }
  focus() {}
}

function run(id, kind, active = false) {
  return {
    id, kind, name: `${kind}-${id}`, active, archived: false, revision: "1",
    axis: "Optimizer step", config: {}, last_step_at: Date.now() / 1000 - 3,
    latest_train: {step: 10, samples_seen: 1280, learning_rate: 1e-5, step_seconds: 2,
      ...(kind === "sft" ? {loss: 0.4, action_loss: 0.3} : {success_rate: 0.5, jerk_margin_gap: -1})},
    latest_eval: {step: 10, samples_seen: 1280, success_rate: 0.5, completion_efficiency: 0.7, rms_jerk: 3, wrong_object_rate: 0},
    series: [],
  };
}

async function dashboard(catalog, saved = {}, options = {}) {
  const elements = new Map();
  const requests = [];
  const basePath = options.basePath || "";
  const storage = new Map(Object.entries(saved));
  for (const run of catalog) run.series = [
    {phase: "train", task: "all", points: [run.latest_train]},
    {phase: "eval", task: "all", points: [run.latest_eval]},
    {phase: "eval", task: "push", points: [run.latest_eval]},
  ];
  const context = vm.createContext({
    document: {
      documentElement: {dataset: {basePath}},
      getElementById(id) {
        if (!elements.has(id)) elements.set(id, new Element());
        return elements.get(id);
      },
      createElement: tag => new Element(tag),
      createDocumentFragment: () => new Element("fragment"),
      createTextNode: value => Object.assign(new Element("text"), {textContent: value}),
    },
    localStorage: options.localStorage || {
      getItem: key => storage.get(key), setItem: (key, value) => storage.set(key, value),
    },
    ResizeObserver: class { observe() {} },
    setInterval() {},
    AbortSignal, Date, console,
    fetch: async route => {
      requests.push(route);
      return {
      ok: true,
      json: async () => route === `${basePath}/api/runs`
        ? {runs: catalog, server_time: new Date().toISOString()}
        : catalog.find(run => route.endsWith(run.id)),
      };
    },
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, "../src/so101/dashboard/dashboard.js"), "utf8"), context);
  await new Promise(resolve => setImmediate(resolve));
  return {
    elements, storage, requests, eval: code => vm.runInContext(code, context),
    reload: () => dashboard(catalog, Object.fromEntries(storage), options),
  };
}

test("prefixed dashboard fetches catalog and run details without touching root API", async () => {
  const page = await dashboard([run("11111111", "rl")], {}, {basePath: "/lego-robot"});
  assert.ok(page.requests.includes("/lego-robot/api/runs"));
  assert.ok(page.requests.includes("/lego-robot/api/runs/11111111"));
  assert.ok(page.requests.every(route => route.startsWith("/lego-robot/api/")));
});

test("recognition runs use a separate tab and sample-matched loss series", async () => {
  const recognition=run("recognition1","recognition",true);
  recognition.latest_train.loss=2;
  const page=await dashboard([run("action1","sft"),recognition]);
  await page.elements.get("tab-recognition").events.click();
  assert.equal(page.eval("chosen().map(r=>r.id).join(',')"),"recognition1");
  assert.equal(page.elements.get("plots").children.find(section=>section.id==="section-recognition-loss").hidden,false);
  assert.equal(page.elements.get("plots").children.find(section=>section.id==="section-sft-loss").hidden,true);
  const values=page.eval('seriesFor({key:"loss",phase:"train",task:"all"})');
  assert.equal(values.length,1);
  assert.equal(values[0].values[0].x,1280);
  assert.equal(values[0].values[0].value,2);
});

test("recognition shows dev, minibatch accuracy and final test without action labels or fabricated accuracy", async () => {
  const recognition = run("recognition1", "recognition", true);
  recognition.latest_train = {step: 100, samples_seen: 400, loss: 2, train_top1: .6, train_accuracy_n: 16};
  recognition.latest_eval = {step: 100, samples_seen: 400, top1: .5, top3: .7, top5: .8,
    n: 20, families: 3, macro_top1: .3, seen_id_n: 12, unseen_id_n: 8, seen_id_top1: .7, unseen_id_top1: .2,
    category_family_accuracy: .6, category_kind_accuracy: .7};
  recognition.final_test = {...recognition.latest_eval, top1: .25, selected_step: 50, selection: "development macro_top1"};
  recognition.training_evaluation = [{...recognition.latest_eval, n: 40}];
  const page = await dashboard([recognition]);
  await page.elements.get("tab-recognition").events.click();
  assert.match(page.elements.get("cards").textContent, /Latest dev top1/);
  assert.doesNotMatch(page.elements.get("cards").textContent, /success|jerk|unassisted/i);
  assert.match(page.elements.get("recognition-results").textContent, /Final held-out test/);
  assert.match(page.elements.get("recognition-results").textContent, /25\.0%/);
  assert.match(page.elements.get("recognition-results").textContent, /Training set \(selected checkpoint\)/);
  assert.doesNotMatch(page.elements.get("recent-head").textContent, /Success|Jerk|Efficiency/);
  const dev = page.eval('seriesFor(recognitionSections.find(section=>section.id==="recognition-dev").metrics[0])');
  assert.equal(dev[0].values[0].value, 50);
  const train = page.eval('seriesFor(recognitionSections.find(section=>section.id==="recognition-training-accuracy").metrics[0])');
  assert.equal(train[0].values[0].value, 60);
  page.eval('state.runs.get("recognition1").latest_train.train_top1 = undefined; state.runs.get("recognition1").series[0].points[0].train_top1 = undefined');
  assert.equal(page.eval('seriesFor({key:"train_top1",phase:"train",task:"all"})').length, 0);
});

test("SFT plots align different batch sizes by logged samples, retaining optimizer steps in tooltips", async () => {
  const page = await dashboard([run("11111111", "sft"), run("22222222", "sft")]);
  await page.elements.get("tab-sft").events.click();
  page.eval(`state.kind = "sft"; state.selected = new Set(["11111111", "22222222"]);
    state.runs.set("11111111", {...state.catalog[0], series: [{phase:"train",task:"all",points:[
      {step:2,samples_seen:128,loss:2}, {step:3,loss:1}]}]});
    state.runs.set("22222222", {...state.catalog[1], series: [{phase:"train",task:"all",points:[
      {step:1,samples_seen:128,loss:3}]}]});`);
  const values = page.eval('seriesFor({key:"loss",phase:"train",task:"all"})');
  assert.equal(values.length, 2);
  assert.equal(values[0].values[0].x, 128);
  assert.equal(values[1].values[0].x, 128);
  assert.equal(values[0].values[0].step, 2);
  assert.equal(values[1].values[0].step, 1);
  assert.equal(values[0].values[1].value, null);
  assert.match(values[0].axis, /samples seen/);
  page.eval(`state.runs.get("11111111").series = [{phase:"eval",task:"all",points:[
    {step:0,samples_seen:0,supervised_loss:4}]}];`);
  assert.equal(page.eval('seriesFor({key:"supervised_loss",phase:"eval",task:"all"})[0].values[0].x'), 0);
  const validation = page.eval("state.charts.find(c => c.metric.key === 'supervised_loss')");
  assert.equal(validation.section.hidden, false);
  assert.equal(validation.section.open, true);
  assert.equal(validation.metric.title, "Validation loss");
  assert.equal(page.eval("sftSections.find(s => s.id === 'sft-frozen-loss').title"),
    "Validation loss (fixed held-out set)");
});

test("data browser prefixes catalog requests and image/movie endpoints", async () => {
  const elements = new Map(), requests = [];
  const context = vm.createContext({
    document: {
      documentElement: {dataset: {basePath: "/lego-robot"}},
      getElementById(id) {
        if (!elements.has(id)) elements.set(id, new Element());
        return elements.get(id);
      },
    },
    URLSearchParams,
    fetch: async route => {
      requests.push(route);
      return {ok: true, json: async () => ({runs: []})};
    },
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, "../src/so101/dashboard/data.js"), "utf8"), context);
  await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(requests, ["/lego-robot/api/runs",
    "/lego-robot/api/data/all?offset=0&limit=24&view=examples&task=all&difficulty=all&object_count=all&expert_outcome=all&expert_acceptance=all"]);
  vm.runInContext('trainingContext = "run/id"', context);
  assert.equal(vm.runInContext('endpoint({image: "overview"})', context),
    "/lego-robot/api/data/run%2Fid?image=overview");
  assert.equal(vm.runInContext('endpoint({movie: "prepare"})', context),
    "/lego-robot/api/data/run%2Fid?movie=prepare");
});

test("full reload restores controls and independent RL/SFT selections, focus, chart views and sections", async () => {
  const rl = run("11111111", "rl"), otherRl = run("44444444", "rl");
  const sft = run("22222222", "sft"), otherSft = run("33333333", "sft");
  otherSft.archived = true;
  let page = await dashboard([rl, otherRl, sft, otherSft], {
    "rl-selected-runs": JSON.stringify([rl.id, otherRl.id]),
    "sft-selected-runs": JSON.stringify([sft.id, otherSft.id]),
  });
  page.elements.get("smoothing").value = "65";
  page.elements.get("smoothing").events.input({target: page.elements.get("smoothing")});
  page.elements.get("archives").checked = true;
  page.elements.get("archives").events.change();
  page.elements.get("search").value = "333";
  page.elements.get("search").events.input();
  page.elements.get("focus").events.change({target: {value: otherRl.id}});
  page.eval("state.zoom = [2, 8]");
  page.eval("state.charts[0].canvas.events.pointerup()");
  page.eval(`
    const training = state.charts.find(chart => chart.section.id === "section-training").section;
    training.open = false;
    training.events.toggle();
  `);
  await page.elements.get("tab-sft").events.click();
  page.elements.get("focus").events.change({target: {value: otherSft.id}});
  page.eval("state.zoom = [1, 100]; state.charts.find(c => c.metric.logLog).canvas.events.pointerup()");
  page.eval(`
    const loss = state.charts.find(chart => chart.metric.logLog).section;
    loss.open = false;
    loss.events.toggle();
  `);
  page.elements.get("pause").events.click();
  page = await page.reload();
  assert.equal(page.eval("state.kind"), "sft");
  assert.equal(page.eval("state.smoothing"), .65);
  assert.equal(page.elements.get("smoothing").value, "65");
  assert.equal(page.elements.get("smooth-value").textContent, "65%");
  assert.equal(page.elements.get("archives").checked, true);
  assert.equal(page.elements.get("search").value, "333");
  assert.equal(page.elements.get("run-count").textContent, "1");
  assert.equal(page.elements.get("focus").value, otherSft.id);
  assert.equal(page.eval("JSON.stringify([...state.selected])"), JSON.stringify([sft.id, otherSft.id]));
  assert.equal(page.eval("JSON.stringify(state.zoom)"), "[1,100]");
  assert.equal(page.eval("state.charts.find(c => c.metric.logLog).section.open"), false);
  assert.equal(page.eval("state.paused"), false, "refresh must reconnect to live updates");
  await page.elements.get("tab-rl").events.click();
  assert.equal(page.elements.get("focus").value, otherRl.id);
  assert.equal(page.eval("JSON.stringify([...state.selected])"), JSON.stringify([rl.id, otherRl.id]));
  assert.equal(page.eval("JSON.stringify(state.zoom)"), "[2,8]");
  assert.equal(page.eval("state.charts.find(c => c.section.id === 'section-training').section.open"), false);
  page.elements.get("reset").events.click();
  page = await page.reload();
  assert.equal(page.eval("state.kind"), "rl");
  assert.equal(page.eval("state.zoom"), null);
  await page.elements.get("tab-sft").events.click();
  assert.equal(page.eval("JSON.stringify(state.zoom)"), "[1,100]");
  page.eval("state.charts.find(c => c.metric.logLog).canvas.events.dblclick()");
  page = await page.reload();
  assert.equal(page.eval("state.zoom"), null);
});

test("explicit empty selections survive reload without selecting defaults in either tab", async () => {
  let page = await dashboard([run("11111111", "rl"), run("22222222", "sft", true)]);
  const unselect = async () => {
    const checkbox = page.elements.get("runs").querySelector("input");
    checkbox.checked = false;
    await checkbox.events.change();
  };
  await unselect();
  await page.elements.get("tab-sft").events.click();
  await unselect();
  page = await page.reload();
  assert.equal(page.eval("state.kind"), "sft");
  assert.equal(page.eval("chosen().length"), 0);
  assert.equal(page.elements.get("focus").value, "");
  await page.elements.get("tab-rl").events.click();
  assert.equal(page.eval("chosen().length"), 0);
  assert.equal(page.storage.get("rl-selected-runs"), "[]");
  assert.equal(page.storage.get("sft-selected-runs"), "[]");
});

test("corrupt, obsolete and invalid preferences or unavailable storage do not break startup", async () => {
  const catalog = [run("11111111", "rl"), run("22222222", "sft")];
  for (const preferences of ["{", "null", "[]",
    JSON.stringify({version: 0, kind: "sft", smoothing: 65, search: "obsolete"}),
    JSON.stringify({version: 1, kind: "obsolete", smoothing: 100, search: {}, archives: "true",
      views: {rl: {focus: "missing", zoom: [8, 2]}, sft: {zoom: [1, "100"]}}}),
    JSON.stringify({version: 1, smoothing: 12, views: {rl: {focus: 123, zoom: [0, null]}}}),
  ]) {
    let page = await dashboard(catalog, {
      "so101-dashboard-preferences": preferences,
      "rl-selected-runs": '["missing"]', "sft-selected-runs": "{}",
      "rl-section-open": '{"training":"false","obsolete":false}',
    });
    assert.equal(page.eval("state.kind"), "rl");
    assert.equal(page.eval("state.smoothing"), 0);
    assert.equal(page.elements.get("smoothing").value, "0");
    assert.equal(page.elements.get("smooth-value").textContent, "Off");
    assert.equal(page.elements.get("archives").checked, false);
    assert.equal(page.elements.get("search").value, "");
    assert.equal(page.eval("state.zoom"), null);
    assert.equal(page.eval("state.focus"), catalog[0].id);
    assert.equal(page.eval("chosen()[0].id"), catalog[0].id);
    assert.equal(page.eval("state.charts.find(c => c.section.id === 'section-training').section.open"), true);
    await page.elements.get("tab-sft").events.click();
    assert.equal(page.eval("state.zoom"), null);
    assert.equal(page.eval("chosen()[0].id"), catalog[1].id);
    page = await page.reload();
    assert.equal(page.eval("state.kind"), "sft");
    assert.equal(page.eval("state.zoom"), null);
  }
  const page = await dashboard(catalog, {}, {
    localStorage: {
      getItem() { throw new Error("Storage disabled"); },
      setItem() { throw new Error("Storage disabled"); },
    },
  });
  await page.elements.get("tab-sft").events.click();
  page.elements.get("smoothing").value = "95";
  assert.doesNotThrow(() => page.elements.get("smoothing").events.input({target: page.elements.get("smoothing")}));
  assert.equal(page.eval("state.smoothing"), .95);
  assert.equal(page.eval("chosen()[0].id"), catalog[1].id);
  await page.reload();
});

test("RL/SFT interaction preserves RL selection, zoom and metrics; SFT uses loss-only rows", async () => {
  const rl = run("11111111", "rl");
  const sft = run("22222222", "sft", true);
  const newerSft = run("33333333", "sft");
  const page = await dashboard([newerSft, sft, rl], {
    "rl-selected-runs": JSON.stringify([rl.id]),
    "rl-section-open": JSON.stringify({training: false}),
  });
  assert.equal(page.eval("chosen()[0].id"), rl.id);
  assert.equal(page.elements.get("recent-head").textContent.includes("Loss"), false);
  assert.equal(page.eval("seriesFor(sections[1].metrics[2])[0].values[0].raw"), -1);
  page.eval("state.zoom = [2, 8]");
  await page.elements.get("tab-sft").events.click();
  assert.equal(page.eval("chosen()[0].id"), sft.id);
  assert.equal(page.eval("state.zoom"), null);
  assert.equal(page.elements.get("run-count").textContent, "2");
  assert.equal(page.elements.get("observatory-title").textContent, "SFT Observatory");
  assert.equal(page.elements.get("tab-sft").attributes["aria-selected"], "true");
  assert.equal(page.elements.get("training-panel").attributes["aria-labelledby"], "tab-sft");
  assert.match(page.elements.get("recent-head").textContent, /LossSuccessEfficiencyEval RMS jerk/);
  const rows = page.elements.get("recent").children;
  assert.equal(rows.length, 2);
  assert.equal(rows[0].children[1].textContent, "SFT loss");
  assert.equal(rows[0].children[2].textContent, "0.400");
  assert.equal(rows[0].children[3].textContent, "—");
  assert.equal(rows[1].children[1].textContent, "Validation only");
  assert.equal(rows[1].children[2].textContent, "—");
  assert.equal(rows[1].children[3].textContent, "50.0%");
  assert.equal(page.eval("seriesFor(sftSections[0].metrics[0])[0].values[0].raw"), 0.4);
  assert.equal(page.eval("seriesFor(sftSections[1].metrics[2])[0].values[0].raw"), 3);
  assert.match(page.elements.get("step-timing").textContent, /optimizer 2s.*since update/);
  assert.equal(page.eval("state.charts.filter(c => !c.section.hidden).every(c => !c.canvas.events.wheel)"), true);
  assert.equal(page.eval("state.charts.filter(c => c.section.dataset.kind === 'rl').every(c => c.section.hidden)"), true);
  await page.elements.get("tab-rl").events.click();
  assert.equal(page.eval("chosen()[0].id"), rl.id);
  assert.equal(page.eval("JSON.stringify(state.zoom)"), "[2,8]");
  assert.equal(page.eval("state.charts.find(c => c.section.id === 'section-training').section.open"), false);
  assert.equal(page.elements.get("recent-head").children.length, 9);
  assert.equal(page.elements.get("recent").children[0].children[1].textContent, "Training");
  assert.equal(page.storage.get("rl-selected-runs"), JSON.stringify([rl.id]));
});

test("a newly discovered SFT run defaults before validation without altering RL", async () => {
  const rl = run("11111111", "rl");
  const catalog = [rl];
  const page = await dashboard(catalog);
  await page.elements.get("tab-sft").events.click();
  assert.equal(page.eval("chosen().length"), 0);
  const sft = run("22222222", "sft");
  sft.latest_eval = null;
  sft.series = [{phase: "train", task: "all", points: [sft.latest_train]}];
  catalog.unshift(sft);
  await page.eval("refresh()");
  assert.equal(page.eval("chosen()[0].id"), sft.id);
  assert.equal(page.elements.get("step-timing").hidden, true);
  page.eval("state.selected.clear(); saveSelection()");
  await page.eval("refresh()");
  assert.equal(page.eval("chosen().length"), 0, "explicit deselection must persist");
  assert.equal(page.storage.get("sft-selected-runs"), "[]");
  await page.elements.get("tab-rl").events.click();
  assert.equal(page.eval("chosen()[0].id"), rl.id);
});

test("timing separates evaluation boundaries from reporting delay and stale phase evidence", async () => {
  const sft = run("22222222", "sft", true);
  sft.config = {run_config: {validation_interval: 48}};
  sft.latest_train.step = 96;
  sft.latest_train.step_seconds = 13.68;
  sft.latest_eval.step = 48;
  sft.latest_eval.evaluation_seconds = 198;
  sft.last_step_at = Date.now() / 1000 - 150;
  sft.remote = {state: "running", observed_at: Date.now() / 1000 - 5};
  const page = await dashboard([sft]);
  await page.elements.get("tab-sft").events.click();
  const timing = page.elements.get("step-timing");
  assert.match(timing.textContent, /evaluation pending.*step 96/);
  assert.match(timing.textContent, /optimizer 13s/);
  assert.match(timing.textContent, /since update 2m 30s/);
  assert.match(timing.textContent, /report 5s/);
  assert.match(timing.textContent, /last eval 3m 18s/);
  sft.remote.observed_at -= 200;
  page.eval("updateStepCounters()");
  assert.match(timing.textContent, /stale/);
  assert.doesNotMatch(timing.textContent, /evaluation pending/);
  sft.remote = {state: "running", phase: "evaluating", observed_at: Date.now() / 1000,
    phase_observed_at: Date.now() / 1000};
  page.eval("updateStepCounters()");
  assert.match(timing.textContent, /evaluating/);
  sft.remote.phase_observed_at -= 200;
  page.eval("updateStepCounters()");
  assert.doesNotMatch(timing.textContent, /evaluating/);
});

test("top status loads only active runs independently of selection and training tab", async () => {
  const active = run("11111111", "sft", true), completed = run("22222222", "sft");
  const page = await dashboard([active, completed], {
    "rl-selected-runs": "[]", "sft-selected-runs": JSON.stringify([completed.id]),
  });
  const timing = page.elements.get("step-timing");
  assert.ok(page.requests.includes("/api/runs/11111111"));
  assert.match(timing.textContent, /sft-11111111/);
  assert.doesNotMatch(timing.textContent, /sft-22222222/);
  await page.elements.get("tab-sft").events.click();
  assert.match(timing.textContent, /sft-11111111/);
  assert.doesNotMatch(timing.textContent, /sft-22222222|excludes|phase not|Enqueue/);
  active.active = false;
  await page.eval("refresh()");
  assert.equal(timing.hidden, true);
});

test("formerly colliding run IDs receive distinct stable colors across selection and refresh", async () => {
  const runs = Array.from({length: 9}, (_, i) => run((7 * i).toString(16).padStart(8, "0"), "sft"));
  const page = await dashboard(runs);
  const assigned = page.eval("state.catalog.map(run => color(run.id))");
  assert.equal(new Set(assigned).size, runs.length);
  await page.elements.get("tab-sft").events.click();
  page.eval("state.selected.clear(); render()");
  await page.eval("refresh()");
  assert.deepEqual(page.eval("state.catalog.map(run => color(run.id))"), assigned);
});

test("SFT loss plots use log-log coordinates, positive gaps and logarithmic interaction", async () => {
  const sft = run("22222222", "sft", true);
  const page = await dashboard([sft]);
  await page.elements.get("tab-sft").events.click();
  assert.equal(page.eval("JSON.stringify(sftSections[0].metrics.map(metric => metric.key))"),
    '["loss","action_loss","box_loss"]');
  assert.equal(page.eval("sftSections[0].metrics.every(metric => metric.logLog)"), true);
  assert.equal(page.eval("sections.flatMap(section => section.metrics).some(metric => metric.logLog)"), false);
  assert.equal(page.eval("sftSections.slice(1).flatMap(section => section.metrics).some(metric => metric.logLog)"), false);
  page.eval(`
    globalThis.devicePixelRatio = 1;
    globalThis.innerWidth = 1000;
    globalThis.innerHeight = 700;
    state.smoothing = 0;
    state.zoom = null;
    const chart = state.charts.find(chart => chart.metric.key === "loss");
    const coordinates = [];
    chart.canvas.getBoundingClientRect = () => ({width: 700, height: 260, left: 0});
    chart.canvas.getContext = () => ({
      scale() {}, setLineDash() {}, beginPath() {}, stroke() {},
      save() {}, clip() {}, restore() {}, fill() {}, fillText() {},
      moveTo(...args) { coordinates.push(...args); },
      lineTo(...args) { coordinates.push(...args); },
      rect(...args) { coordinates.push(...args); },
      arc(...args) { coordinates.push(...args); },
    });
    chart.canvas.setPointerCapture = () => {};
    const series = state.runs.get("22222222").series.find(series => series.phase === "train");
    series.points = [
      {step: 0, loss: 100}, {step: 1, loss: 10}, {step: 5, loss: 0},
      {step: 10, loss: 1}, {step: 50, loss: -1}, {step: 100, loss: .1},
      {step: 1000, loss: .01},
    ].map(point => ({...point, samples_seen: point.step}));
    chart.update();
  `);
  assert.equal(page.eval("Math.abs((chart.x(10) - chart.x(1)) - (chart.x(100) - chart.x(10))) < 1e-9"), true);
  assert.equal(page.eval("Math.abs((chart.y(.1) - chart.y(1)) - (chart.y(1) - chart.y(10))) < 1e-9"), true);
  assert.equal(page.eval("Math.abs(chart.inverseX(chart.x(100)) - 100) < 1e-9"), true);
  assert.equal(page.eval("coordinates.every(Number.isFinite)"), true);
  assert.equal(page.eval("JSON.stringify(chart.series[0].values.filter(p => chart.plottable(p)).map(p => p.step))"), "[1,10,100,1000]");
  const tooltip = page.elements.get("tooltip");
  tooltip.offsetWidth = 120;
  tooltip.offsetHeight = 80;
  page.eval("chart.hover({clientX: chart.x(100), clientY: 100})");
  assert.match(tooltip.textContent, /step 100: 0\.10/);
  page.eval(`
    chart.drag = {x: 0, zoom: [1, 1000]};
    chart.hover({clientX: (chart.area.right - chart.area.left) / 3});
  `);
  assert.equal(page.eval("Math.abs(state.zoom[0] - .1) < 1e-9 && Math.abs(state.zoom[1] - 100) < 1e-9"), true);
  page.eval("chart.drag = null; state.zoom = null; state.smoothing = .5; chart.update()");
  assert.equal(page.eval("chart.series[0].values.find(p => p.step === 5).value"), null);
  assert.equal(page.eval("chart.series[0].values.find(p => p.step === 10).value"), 1);
  page.eval("state.runs.get('22222222').series.find(s => s.phase === 'train').points = [{step: 1, loss: 0}]; chart.update()");
  assert.equal(page.eval("chart.domain"), null);
  assert.doesNotThrow(() => page.eval("chart.canvas.events.pointerdown({clientX: 0, pointerId: 1})"));
});
