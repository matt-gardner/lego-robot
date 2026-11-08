"use strict";

const $ = id => document.getElementById(id);
const basePath = document.documentElement.dataset.basePath || "";
const colors = ["#4c9dff", "#ec599b", "#f3ce48", "#8ddc91", "#a68cff", "#55d0cf", "#ff9961"];
const performanceMetrics = [
  {key: "success_rate", title: "Success rate", unit: "%", percent: true, bounds: [0, 100],
    note: "All sampled trajectories, including discarded and solved groups."},
  {key: "completion_efficiency", title: "Completion efficiency", unit: "%", percent: true,
    note: "Successful trajectories only; relative to the assistance-adjusted expert reference. Higher is better."},
  {key: "jerk_margin_gap", title: "Jerk − reference margin", unit: "deg/frame³", zeroLine: true,
    note: "For each rollout: commanded RMS jerk minus that problem's own expert-reference jerk at matching assistance; then average. No command-level pooling or percentiles. Includes failures and discarded/solved groups with ≥4 commands. Historical gaps are recomputed from references, not historical loss margins."},
  {key: "wrong_object_rate", title: "Wrong-object rate", unit: "%", percent: true,
    note: "Fraction of trajectories manipulating the wrong object. Lower is better."},
  {key: "reward", title: "Trajectory reward", unit: "",
    note: "Success, completion, and smoothness retain separate advantages."},
  {key: "assistance", title: "Gold-prefix assistance", unit: "%", percent: true, bounds: [0, 100],
    note: "Mean applied assistance, including discarded groups and solved problems at their saved assistance."},
];
const sections = [
  {id: "eval", title: "Eval metrics", note: "Unassisted validation · all tasks", open: true,
    metrics: performanceMetrics.slice(0, 4).map(metric => ({
      ...metric, phase: "eval", task: "all",
      ...(metric.key === "jerk_margin_gap" ? {
        key: "rms_jerk", title: "Validation RMS arm jerk", zeroLine: false,
      } : {}),
      note: metric.key === "success_rate" ? "Unassisted validation trajectories only." :
        metric.key === "completion_efficiency" ? "Successful validation trajectories only; relative to expert task optimum. Higher is better." :
          metric.key === "jerk_margin_gap" ? "Unassisted validation RMS jerk. Validation logs do not record reference margins." : metric.note,
    }))},
  {id: "training", title: "Overall training performance", note: "All tasks · all sampled training groups", open: true,
    metrics: performanceMetrics.map(metric => ({...metric, phase: "train", task: "all"}))},
  ...["grasp", "view", "push", "place"].map(task => ({
    id: `training-${task}`, title: `${task[0].toUpperCase()}${task.slice(1)} training performance`,
    note: `${task} only · includes discarded and solved groups`, open: false,
    metrics: performanceMetrics.map(metric => ({...metric, phase: "train", task})),
  })),
  {id: "health", title: "Training health", note: "Run-level optimizer and timing metrics", open: true, metrics: [
    {key: "learning_rate", title: "Learning rate", unit: "", scientific: true,
    note: "Training only. Actual per-step rate when logged; historical rates are not guessed."},
    {key: "gradient_norm", title: "Pre-clip gradient norm", unit: "",
    note: "Training only, since the latest optimization change. Earlier-scale history is hidden; new spikes remain visible."},
    {key: "step_seconds", title: "Training step time", unit: "s",
    note: "Wall time per training step, excluding validation and checkpoint writes."},
  ].map(metric => ({...metric, phase: "train", task: "all"}))},
];
const sftSections = [
  {id: "sft-loss", title: "Sample-matched training loss", note: "Logged cumulative training samples · log-log axes · mean losses, not batch totals", open: true,
    metrics: [
      {key: "loss", title: "Mean total loss"},
      {key: "action_loss", title: "Mean action loss"},
      {key: "box_loss", title: "Mean box loss"},
    ].map(metric => ({...metric, phase: "train", task: "all", unit: "", logLog: true,
      note: "Losses are normalized by valid action/entity labels, not summed across the batch. No additional batch-size division. Missing components remain gaps, not zeros."}))},
  {...sections[0], id: "sft-eval"},
  {id: "sft-frozen-loss", title: "Sample-matched frozen-set loss",
    note: "Same frozen evaluation set · logged training exposure · baseline at zero samples", open: true,
    metrics: [{key: "supervised_loss", title: "Frozen supervised loss", phase: "eval", task: "all", unit: "",
      note: "Fixed evaluation examples, diffusion noise and time. Equal training samples are a compute proxy, not equal FLOPs or elapsed time."}]},
  ...["grasp", "view", "push", "place"].map(task => ({
    ...sections[0], id: `sft-eval-${task}`, title: `${task[0].toUpperCase()}${task.slice(1)} eval metrics`,
    note: `Unassisted validation · ${task} only`, open: false,
    metrics: sections[0].metrics.map(metric => ({...metric, task})),
  })),
  {...sections.at(-1), id: "sft-health",
    metrics: sections.at(-1).metrics.map(metric => metric.key === "gradient_norm"
      ? {...metric, note: "Logged pre-clip gradient norm for supervised training."} : metric)},
];
function readStorage(key) {
  try { return JSON.parse(localStorage.getItem(key)); } catch { return null; }
}
const finite = value => typeof value === "number" && Number.isFinite(value);
const record = value => value !== null && typeof value === "object" && !Array.isArray(value);
const savedSelections = Object.fromEntries(["rl", "sft"].map(kind => {
  const value = readStorage(`${kind}-selected-runs`);
  return [kind, Array.isArray(value) && value.every(id => typeof id === "string" && id.length > 0) ? value : null];
}));
const savedSections = readStorage("rl-section-open");
const sectionOpen = Object.fromEntries([...sections, ...sftSections]
  .filter(section => record(savedSections) && typeof savedSections[section.id] === "boolean")
  .map(section => [section.id, savedSections[section.id]]));
const storedPreferences = readStorage("so101-dashboard-preferences");
const preferences = record(storedPreferences) && storedPreferences.version === 1 ? storedPreferences : {};
const tabViews = Object.fromEntries(["rl", "sft"].map(kind => {
  const view = record(preferences.views) && record(preferences.views[kind]) ? preferences.views[kind] : {};
  return [kind, {
    focus: typeof view.focus === "string" && view.focus.length > 0 ? view.focus : null,
    zoom: Array.isArray(view.zoom) && view.zoom.length === 2 && view.zoom.every(finite)
      && view.zoom[0] < view.zoom[1] ? view.zoom : null,
  }];
}));
const initialKind = preferences.kind === "sft" ? "sft" : "rl";
const selections = {rl: new Set(savedSelections.rl || []), sft: new Set(savedSelections.sft || [])};
const state = {
  catalog: [], selected: selections[initialKind], runs: new Map(),
  initialized: false, smoothing: finite(preferences.smoothing) && preferences.smoothing >= 0
    && preferences.smoothing <= 95 && preferences.smoothing % 5 === 0 ? preferences.smoothing / 100 : 0,
  paused: false, ...tabViews[initialKind], loading: false, errors: [], charts: [],
  kind: initialKind, sftInitialized: false,
};
const pct = value => finite(value) ? `${(value * 100).toFixed(1)}%` : "—";
const num = value => finite(value) ? value.toFixed(3) : "—";
const exp = value => finite(value) ? value.toExponential(1) : "—";
const shortName = run => run.name.replaceAll("entity_smolvla_", "").replaceAll("trajectory_rl_", "");
const color = id => colors[parseInt(id.slice(0, 8), 16) % colors.length];
const runKind = run => run.kind || "rl";
const tabCatalog = () => state.catalog.filter(run => runKind(run) === state.kind);
const chosen = () => tabCatalog().filter(run => state.selected.has(run.id));
const points = (run, phase, task) => run?.series.find(s => s.phase === phase && s.task === task)?.points || [];
const duration = seconds => {
  if (!finite(seconds)) return "—";
  const total = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(total / 3600), minutes = Math.floor(total % 3600 / 60);
  return `${hours ? `${hours}h ` : ""}${hours || minutes ? `${minutes}m ` : ""}${total % 60}s`;
};

function updateStepCounters() {
  const container = $("step-timing");
  if (!container) return;
  const selected = chosen();
  container.replaceChildren(...selected.map(summary => {
    const run = state.runs.get(summary.id);
    const elapsed = finite(run?.last_step_at) ? Date.now() / 1000 - run.last_step_at : null;
    const prefix = selected.length > 1 ? `${shortName(summary)} · ` : "";
    const since = summary.active === true
      ? `Since last step: ${run?.last_step_time_estimated ? "~" : ""}${duration(elapsed)}`
      : "Not running";
    const counter = text("span",
      `${prefix}Last step time: ${duration(run?.latest_train?.step_seconds)} · ${since}`);
    counter.title = `Latest loaded step: ${run?.latest_train?.step ?? "none"}. ` +
      (run?.last_step_time_estimated
        ? "Estimated from the log write time; this run has no completion timestamp."
        : "Elapsed since the recorded training-step completion.");
    return counter;
  }));
}

function text(tag, value, className = "") {
  const element = document.createElement(tag);
  element.textContent = value;
  element.className = className;
  return element;
}

function saveSelection() {
  savedSelections[state.kind] = [...state.selected];
  try { localStorage.setItem(`${state.kind}-selected-runs`, JSON.stringify(savedSelections[state.kind])); } catch {
    // Private browser modes can disable storage; selection still works in memory.
  }
}

function savePreferences() {
  tabViews[state.kind] = {focus: state.focus, zoom: state.zoom};
  try {
    localStorage.setItem("so101-dashboard-preferences", JSON.stringify({
      version: 1, kind: state.kind, smoothing: Math.round(state.smoothing * 100),
      search: $("search").value, archives: $("archives").checked, views: tabViews,
    }));
  } catch {
    // Preferences still work in memory when browser storage is unavailable.
  }
}

function showErrors() {
  const errors = [...state.errors, ...chosen().flatMap(run =>
    [run.error, run.activity_error].filter(Boolean).map(error => `${shortName(run)}: ${error}`))];
  $("error").hidden = !errors.length;
  $("error").textContent = errors.join(" · ");
}

function renderRuns() {
  const fragment = document.createDocumentFragment();
  const query = $("search").value.toLowerCase();
  const visible = tabCatalog().filter(run =>
    ($("archives").checked || !run.archived) && run.name.toLowerCase().includes(query));
  $("run-count").textContent = `${visible.length}`;
  for (const run of visible) {
    const wrapper = text("div", "", `run${state.selected.has(run.id) ? " selected" : ""}`);
    wrapper.style.setProperty("--run-color", color(run.id));
    const label = document.createElement("label");
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = state.selected.has(run.id);
    checkbox.addEventListener("change", async () => {
      if (checkbox.checked) state.selected.add(run.id);
      else state.selected.delete(run.id);
      saveSelection();
      await loadDetails();
      render();
    });
    const title = text("span", shortName(run));
    label.append(checkbox, title);
    const meta = text("div", "", "run-meta");
    const swatch = text("span", "", "swatch");
    const latest = run.latest_train?.step ?? "—";
    meta.append(swatch, document.createTextNode(
      `step ${latest}${state.kind === "sft" ? ` · loss ${num(run.latest_train?.loss)}` : ""} · eval ${pct(run.latest_eval?.success_rate)}`));
    meta.append(document.createElement("br"), document.createTextNode(
      `${run.archived ? "Archive · " : ""}Log updated ${run.updated_at ? new Date(run.updated_at).toLocaleString() : "unavailable"}`));
    if (run.remote) {
      meta.append(document.createElement("br"), document.createTextNode(
        `Cloud: ${run.remote.state} · contact ${duration(run.remote.contact_age_seconds)} ago` +
        ` · backup step ${run.remote.backup?.step ?? "none"}`));
    }
    wrapper.append(label, meta);
    fragment.append(wrapper);
  }
  if (!visible.length) fragment.append(text("p", `No matching ${state.kind.toUpperCase()} logs.`, "hint"));
  $("runs").replaceChildren(fragment);
}

function renderDetails() {
  const selected = chosen();
  if (!selected.some(run => run.id === state.focus)) state.focus = selected[0]?.id ?? null;
  $("focus").replaceChildren(...selected.map(run => {
    const option = text("option", shortName(run));
    option.value = run.id;
    return option;
  }));
  $("focus").value = state.focus || "";
  const run = state.runs.get(state.focus);
  const latest = run?.latest_eval;
  const sft = state.kind === "sft";
  const cards = [
    ["Latest validation success", pct(latest?.success_rate), latest ? `Step ${latest.step} · unassisted` : "No validation logged"],
    ["Best validation success", pct(run?.best_eval?.success_rate), run?.best_eval ? `Step ${run.best_eval.step}` : "No validation logged"],
    ["Latest optimizer step", `${run?.latest_train?.step ?? "—"}`, run?.latest_train
      ? `${sft ? `Loss ${num(run.latest_train.loss)} · ` : ""}LR ${exp(run.latest_train.learning_rate)}`
      : "No training steps logged"],
    ["Latest validation jerk", num(latest?.rms_jerk), "deg/frame³ · all validation trajectories"],
  ];
  $("cards").replaceChildren(...cards.map(([label, value, note]) => {
    const card = text("div", "", "card");
    card.append(text("div", label, "card-label"), text("div", value, "card-value"), text("div", note, "card-note"));
    return card;
  }));
  const training = points(run, "train", "all").slice(-10);
  const cutoff = training[0]?.step ?? Infinity;
  const preceding = points(run, "eval", "all").filter(row => sft || row.step < cutoff).slice(-2);
  const recent = [
    ...preceding.map(row => ({...row, phase: "eval"})),
    ...training.map(row => ({...row, phase: "train"})),
  ];
  if (sft) recent.sort((a, b) => a.step - b.step || (a.phase === "eval" ? 1 : -1));
  const headings = sft
    ? ["Step", "Phase", "Loss", "Success", "Efficiency", "Eval RMS jerk", "Wrong object", "LR", "Grad norm", "Step time (s)"]
    : ["Step", "Phase", "Success", "Efficiency", "Jerk gap / eval RMS", "Wrong object", "LR", "Grad norm", "Assistance"];
  $("recent-head").replaceChildren(...headings.map(label => {
    const th = text("th", label);
    if (label === "Jerk gap / eval RMS") th.title = "Training: jerk minus reference margin. Validation: absolute RMS jerk (no margin logged).";
    return th;
  }));
  $("recent-note").textContent = sft
    ? "Most recent 10 loss-only optimizer steps and two latest validation rows. Success, efficiency, absolute RMS jerk and wrong-object metrics are evaluation only."
    : "Most recent 10 optimizer steps, plus the two preceding validation-only rows. Training success includes assisted and discarded trajectories; validation is unassisted.";
  $("recent").replaceChildren(...recent.map(row => {
    const tr = text("tr", "", row.phase === "eval" ? "eval-row" : "");
    const values = sft ? [
      row.step, row.phase === "eval" ? "Validation only" : "SFT loss",
      row.phase === "train" ? num(row.loss) : "—",
      pct(row.success_rate), pct(row.completion_efficiency), num(row.rms_jerk),
      pct(row.wrong_object_rate), exp(row.learning_rate), num(row.gradient_norm), num(row.step_seconds),
    ] : [
      row.step, row.phase === "eval" ? "Validation only" : "Training",
      pct(row.success_rate), pct(row.completion_efficiency),
      num(row.phase === "eval" ? row.rms_jerk : row.jerk_margin_gap),
      pct(row.wrong_object_rate), exp(row.learning_rate), num(row.gradient_norm), pct(row.assistance),
    ];
    for (const value of values) tr.append(text("td", `${value}`));
    return tr;
  }));
  if (!recent.length) {
    const row = document.createElement("tr");
    const cell = text("td", "Select a run with logged metrics.");
    cell.colSpan = headings.length;
    row.append(cell);
    $("recent").append(row);
  }
  $("config").textContent = run ? JSON.stringify({
    path: run.path, updated_at: run.updated_at, axis: run.axis,
    configuration: run.config, assistance_curriculum: run.curriculum,
  }, null, 2) : "No run selected.";
}

function seriesFor(metric) {
  const result = [];
  const phase = metric.phase;
  const task = metric.task;
  for (const summary of chosen()) {
    const run = state.runs.get(summary.id);
    if (!run) continue;
    for (const series of run.series) {
      if (series.phase !== phase) continue;
      if (series.task !== task) continue;
      let previous = null;
      const gradientStart = run.config.policy_optimization_changes?.at(-1)?.start_step;
      const visiblePoints = metric.key === "gradient_norm" && finite(gradientStart)
        ? series.points.filter(point => point.step >= gradientStart) : series.points;
      const values = visiblePoints.map(point => {
        const x = state.kind === "sft" ? point.samples_seen : point.step;
        const raw = finite(point[metric.key]) ? point[metric.key] * (metric.percent ? 100 : 1) : null;
        if (!finite(x) || x < 0 || raw === null || (metric.logLog && (raw <= 0 || x <= 0))) {
          previous = null;
          return {...point, x, raw, value: null};
        }
        previous = previous === null ? raw : state.smoothing * previous + (1 - state.smoothing) * raw;
        return {...point, x, raw, value: previous};
      });
      if (!values.some(point => finite(point.value))) continue;
      result.push({
        label: `${shortName(run)} · ${series.task} · ${series.phase === "eval" ? "validation" : "train"}`,
        values, color: color(run.id), axis: state.kind === "sft" ? "Training samples seen (compute proxy)" : run.axis,
        dash: series.phase === "eval" ? [6, 4] : [],
      });
    }
  }
  return result;
}

function logTicks(min, max) {
  const first = Math.floor(Math.log10(min)), last = Math.ceil(Math.log10(max));
  const factors = last - first > 2 ? [1] : [1, 2, 5];
  const ticks = [];
  for (let exponent = first; exponent <= last; exponent++) {
    for (const factor of factors) {
      const value = factor * 10 ** exponent;
      if (value >= min && value <= max) ticks.push(value);
    }
  }
  return ticks.length ? ticks : [min, max];
}

class Chart {
  constructor(metric, container, section) {
    this.metric = metric;
    this.section = section;
    this.panel = text("section", "", "panel plot");
    this.canvas = document.createElement("canvas");
    this.canvas.setAttribute("role", "img");
    this.canvas.setAttribute("aria-label", `${section.querySelector("h2").textContent}: ${metric.title} versus ${section.dataset.kind === "sft" ? "training samples seen" : "optimizer step"}${metric.logLog ? ", both axes logarithmic" : ""}`);
    this.legend = text("div", "", "legend");
    const heading = text("h2", metric.title);
    this.panel.append(heading);
    this.panel.append(text("div", metric.note, "subtitle"), this.canvas, this.legend);
    container.append(this.panel);
    new ResizeObserver(() => this.draw()).observe(this.canvas);
    this.canvas.addEventListener("pointermove", event => this.hover(event));
    this.canvas.addEventListener("pointerleave", () => { $("tooltip").hidden = true; this.crosshair = null; this.draw(); });
    this.canvas.addEventListener("dblclick", () => { state.zoom = null; savePreferences(); drawCharts(); });
    this.canvas.addEventListener("pointerdown", event => {
      if (!this.domain) return;
      this.drag = {x: event.clientX, zoom: [...this.domain]};
      this.canvas.setPointerCapture(event.pointerId);
    });
    this.canvas.addEventListener("pointerup", () => { this.drag = null; savePreferences(); });
  }

  plottable(point, key = "value") {
    return finite(point.x) && finite(point[key]) && (!this.metric.logLog || (point[key] > 0 && point.x > 0));
  }

  update() {
    if (this.section.hidden || !this.section.open) return;
    this.series = seriesFor(this.metric);
    this.legend.replaceChildren(...this.series.map(series => {
      const label = text("span", "");
      const line = document.createElement("i");
      line.style.setProperty("--run-color", series.color);
      if (series.dash.length) line.style.borderTopStyle = "dashed";
      label.append(line, document.createTextNode(series.label));
      return label;
    }));
    this.draw();
  }

  draw() {
    if (this.section.hidden || !this.section.open || !this.series) return;
    const rect = this.canvas.getBoundingClientRect();
    if (!rect.width) return;
    const dpr = devicePixelRatio || 1;
    this.canvas.width = Math.round(rect.width * dpr);
    this.canvas.height = Math.round(rect.height * dpr);
    const ctx = this.canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    const W = rect.width, H = rect.height;
    const left = this.metric.scientific ? 70 : 53, right = W - 17, top = 14, bottom = H - 35;
    this.area = {left, right, top, bottom};
    const valid = this.series.flatMap(s => s.values.filter(p => this.plottable(p)));
    if (!valid.length) {
      this.domain = null;
      ctx.fillStyle = "#8893a4";
      ctx.font = "12px -apple-system, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("No data for this phase / task selection", W / 2, H / 2);
      return;
    }
    const steps = valid.map(p => p.x);
    const minStep = Math.min(...steps), maxStep = Math.max(...steps);
    let [x0, x1] = state.zoom || [minStep, maxStep === minStep ? (this.metric.logLog ? minStep * 10 : minStep + 1) : maxStep];
    if (this.metric.logLog) {
      if (x0 <= 0) x0 = minStep;
      if (x1 <= x0) x1 = x0 * 10;
    }
    this.domain = [x0, x1];
    const ys = valid.filter(p => p.x >= x0 && p.x <= x1).flatMap(p => [p.value, p.raw])
      .filter(value => finite(value) && (!this.metric.logLog || value > 0));
    let ymin = 0, ymax = 1;
    if (this.metric.logLog) {
      ymin = ys.length ? Math.log10(Math.min(...ys)) : 0;
      ymax = ys.length ? Math.log10(Math.max(...ys)) : 1;
      const pad = (ymax - ymin || 1) * .1;
      [ymin, ymax] = [10 ** (ymin - pad), 10 ** (ymax + pad)];
    } else if (this.metric.bounds) [ymin, ymax] = this.metric.bounds;
    else if (ys.length) {
      ymin = Math.min(0, ...ys);
      ymax = this.metric.zeroLine ? Math.max(0, ...ys) : Math.max(...ys);
      const pad = (ymax - ymin || Math.abs(ymax) || 1) * .1;
      ymax += pad;
      if (ymin < 0) ymin -= pad;
    }
    const transform = value => this.metric.logLog ? Math.log10(value) : value;
    const x = step => left + (transform(step) - transform(x0)) / (transform(x1) - transform(x0)) * (right - left);
    const y = value => bottom - (transform(value) - transform(ymin)) / (transform(ymax) - transform(ymin)) * (bottom - top);
    this.x = x;
    this.y = y;
    this.inverseX = pixel => {
      const value = transform(x0) + (pixel - left) / (right - left) * (transform(x1) - transform(x0));
      return this.metric.logLog ? 10 ** value : value;
    };
    ctx.font = "11px -apple-system, sans-serif";
    ctx.textAlign = "right";
    const yTicks = this.metric.logLog ? logTicks(ymin, ymax) :
      Array.from({length: 6}, (_, i) => ymin + (ymax - ymin) * i / 5);
    for (const value of yTicks) {
      const py = y(value);
      ctx.strokeStyle = "#292d33"; ctx.lineWidth = 1; ctx.setLineDash([]);
      ctx.beginPath(); ctx.moveTo(left, py); ctx.lineTo(right, py); ctx.stroke();
      ctx.fillStyle = "#8b95a5";
      const formatted = this.metric.logLog ? Number(value.toPrecision(2)).toString() :
        this.metric.scientific ? value.toExponential(1) :
        Math.abs(value) >= 100 ? value.toFixed(0) : Math.abs(ymax) < .1 ? value.toPrecision(2) : value.toFixed(1);
      ctx.fillText(formatted, left - 9, py + 4);
    }
    ctx.textAlign = "center";
    const ticks = Math.max(2, Math.floor((right - left) / 85));
    const xTicks = this.metric.logLog ? logTicks(x0, x1) :
      Array.from({length: ticks + 1}, (_, i) => x0 + (x1 - x0) * i / ticks);
    for (const step of xTicks) {
      ctx.strokeStyle = "#24272d";
      ctx.beginPath(); ctx.moveTo(x(step), top); ctx.lineTo(x(step), bottom); ctx.stroke();
      ctx.fillStyle = "#8b95a5"; ctx.fillText(this.metric.logLog ? Number(step.toPrecision(2)).toString() :
        Number.isInteger(step) ? `${step}` : step.toFixed(1), x(step), bottom + 18);
    }
    ctx.save(); ctx.beginPath(); ctx.rect(left, top, right - left, bottom - top); ctx.clip();
    if (this.metric.zeroLine) {
      ctx.strokeStyle = "#b4bfce"; ctx.lineWidth = 1.5; ctx.setLineDash([5, 4]);
      ctx.beginPath(); ctx.moveTo(left, y(0)); ctx.lineTo(right, y(0)); ctx.stroke();
    }
    const stroke = (series, key, alpha, width) => {
      ctx.strokeStyle = series.color; ctx.fillStyle = series.color; ctx.globalAlpha = alpha;
      ctx.lineWidth = width; ctx.setLineDash(series.dash);
      ctx.beginPath();
      let started = false;
      for (const point of series.values) {
        if (!this.plottable(point, key)) { started = false; continue; }
        if (!started) { ctx.moveTo(x(point.x), y(point[key])); started = true; }
        else ctx.lineTo(x(point.x), y(point[key]));
      }
      ctx.stroke();
      if (series.values.filter(p => this.plottable(p, key)).length === 1) {
        const point = series.values.find(p => this.plottable(p, key));
        ctx.beginPath(); ctx.arc(x(point.x), y(point[key]), 3, 0, Math.PI * 2); ctx.fill();
      }
    };
    for (const series of this.series) {
      if (state.smoothing) stroke(series, "raw", .22, 1);
      stroke(series, "value", 1, 1.8);
    }
    ctx.globalAlpha = 1;
    if (finite(this.crosshair)) {
      ctx.strokeStyle = "#8491a5"; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(this.crosshair, top); ctx.lineTo(this.crosshair, bottom); ctx.stroke();
    }
    ctx.restore();
    ctx.fillStyle = "#737e8f"; ctx.font = "10px -apple-system, sans-serif"; ctx.textAlign = "right";
    const axes = [...new Set(this.series.map(s => s.axis))];
    ctx.fillText((axes.length === 1 ? axes[0] : "Step / episode index (mixed legacy logs)") +
      (this.metric.logLog ? " (log scale)" : ""), right, H - 1);
  }

  hover(event) {
    if (!this.area || !this.domain) return;
    const rect = this.canvas.getBoundingClientRect();
    if (this.drag) {
      const transform = value => this.metric.logLog ? Math.log10(value) : value;
      const span = transform(this.drag.zoom[1]) - transform(this.drag.zoom[0]);
      const dx = (event.clientX - this.drag.x) / (this.area.right - this.area.left) * span;
      state.zoom = this.drag.zoom.map(value => this.metric.logLog ? 10 ** (transform(value) - dx) : value - dx);
      drawCharts();
      return;
    }
    const pixel = event.clientX - rect.left;
    this.crosshair = pixel; this.draw();
    const tooltip = $("tooltip");
    tooltip.replaceChildren(text("strong", this.metric.title));
    for (const series of this.series) {
      const values = series.values.filter(p => this.plottable(p));
      if (!values.length) continue;
      const point = values.reduce((best, p) => Math.abs(this.x(p.x) - pixel) < Math.abs(this.x(best.x) - pixel) ? p : best);
      const value = this.metric.scientific ? point.raw.toExponential(2) : point.raw.toFixed(2);
      const line = text("div", `${series.label} · ${state.kind === "sft" ? `${point.x} samples · ` : ""}step ${point.step}: ${value}${this.metric.unit === "%" ? "%" : ` ${this.metric.unit}`}`);
      line.style.color = series.color;
      tooltip.append(line);
    }
    tooltip.hidden = false;
    tooltip.style.left = `${Math.max(4, Math.min(event.clientX + 16, innerWidth - tooltip.offsetWidth - 10))}px`;
    tooltip.style.top = `${Math.max(4, Math.min(event.clientY + 12, innerHeight - tooltip.offsetHeight - 10))}px`;
  }

}

function drawCharts() { for (const chart of state.charts) chart.draw(); }
function render() {
  for (const kind of ["rl", "sft"]) {
    $(`tab-${kind}`).setAttribute("aria-selected", `${state.kind === kind}`);
    $(`tab-${kind}`).tabIndex = state.kind === kind ? 0 : -1;
  }
  $("training-panel").setAttribute("aria-labelledby", `tab-${state.kind}`);
  $("observatory-title").textContent = `${state.kind.toUpperCase()} Observatory`;
  for (const section of $("plots").children) section.hidden = section.dataset.kind !== state.kind;
  renderRuns();
  renderDetails();
  savePreferences();
  for (const chart of state.charts) chart.update();
  updateStepCounters();
  $("context").textContent = !chosen().length ? "Select one or more runs to compare." :
    `${chosen().length} run(s) selected · Cards and table: ${shortName(chosen().find(r => r.id === state.focus) || chosen()[0])} · Eval is unassisted. ` +
    (state.kind === "sft" ? "Plots compare logged training samples seen, not optimizer steps. Equal exposure is a compute proxy, not equal FLOPs or wall time. Runs/points without sample counts are omitted, not guessed. SFT training is loss-only; rollout metrics are evaluation only."
      : "Training includes every sampled group; each section has a fixed task scope.");
  showErrors();
}

async function request(path) {
  const response = await fetch(`${basePath}${path}`, {cache: "no-store", signal: AbortSignal.timeout(10000)});
  if (!response.ok) throw new Error(`HTTP ${response.status}: ${path}`);
  return response.json();
}

async function loadDetails() {
  await Promise.all(chosen().map(async run => {
    const previous = state.runs.get(run.id);
    if (previous?.revision === run.revision && !run.error && !previous.error) return;
    try { state.runs.set(run.id, await request(`/api/runs/${run.id}`)); }
    catch (error) { state.errors.push(`${shortName(run)}: ${error.message}`); }
  }));
}

async function refresh() {
  if (state.paused || state.loading) return;
  state.loading = true;
  state.errors = [];
  try {
    const result = await request("/api/runs");
    state.catalog = result.runs;
    if (!state.initialized) {
      const rlCatalog = state.catalog.filter(run => runKind(run) === "rl");
      const existing = new Set(rlCatalog.map(run => run.id));
      for (const id of selections.rl) if (!existing.has(id)) selections.rl.delete(id);
      if (!selections.rl.size && savedSelections.rl?.length !== 0) {
        const first = rlCatalog.find(run => !run.archived && run.latest_eval);
        if (first) selections.rl.add(first.id);
      }
      state.initialized = true;
    }
    if (!state.sftInitialized) {
      const sftCatalog = state.catalog.filter(run => runKind(run) === "sft");
      const existing = new Set(sftCatalog.map(run => run.id));
      for (const id of selections.sft) if (!existing.has(id)) selections.sft.delete(id);
      if (!selections.sft.size && savedSelections.sft?.length !== 0) {
        const first = sftCatalog.find(run => !run.archived && run.active === true)
          || sftCatalog.find(run => !run.archived) || sftCatalog[0];
        if (first) selections.sft.add(first.id);
      }
      state.sftInitialized = sftCatalog.length > 0;
    }
    await loadDetails();
    const warnings = state.errors.length || chosen().some(run => run.error || run.activity_error);
    $("connection").textContent = warnings ? "Log read warning" : "Live log sync";
    $("updated").textContent = `Synced ${new Date(result.server_time).toLocaleTimeString()}`;
    render();
  } catch (error) {
    state.errors = [`Updates disconnected: ${error.message}. Retaining the last loaded data; retrying every 3 s.`];
    $("connection").textContent = "Disconnected";
    showErrors();
  } finally { state.loading = false; }
}

async function switchTab(kind) {
  if (state.kind === kind) return;
  tabViews[state.kind] = {focus: state.focus, zoom: state.zoom};
  state.kind = kind;
  state.selected = selections[kind];
  Object.assign(state, tabViews[kind]);
  $("tooltip").hidden = true;
  for (const chart of state.charts) { chart.crosshair = null; chart.drag = null; }
  render();
  await loadDetails();
  render();
}
for (const kind of ["rl", "sft"]) {
  $(`tab-${kind}`).addEventListener("click", () => switchTab(kind));
  $(`tab-${kind}`).addEventListener("keydown", event => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const target = event.key === "Home" ? "rl" : event.key === "End" ? "sft" : kind === "rl" ? "sft" : "rl";
    $(`tab-${target}`).focus();
    switchTab(target);
  });
}
$("search").value = typeof preferences.search === "string" ? preferences.search : "";
$("archives").checked = preferences.archives === true;
$("smoothing").value = `${Math.round(state.smoothing * 100)}`;
$("smooth-value").textContent = state.smoothing ? `${$("smoothing").value}%` : "Off";
$("search").addEventListener("input", () => { savePreferences(); renderRuns(); });
$("archives").addEventListener("change", () => { savePreferences(); renderRuns(); });
$("smoothing").addEventListener("input", event => {
  state.smoothing = Number(event.target.value) / 100;
  $("smooth-value").textContent = state.smoothing ? `${event.target.value}%` : "Off";
  render();
});
$("focus").addEventListener("change", event => { state.focus = event.target.value; render(); });
$("reset").addEventListener("click", () => { state.zoom = null; savePreferences(); drawCharts(); });
$("pause").addEventListener("click", () => {
  state.paused = !state.paused;
  $("pause").textContent = state.paused ? "Resume updates" : "Pause updates";
  $("pause").setAttribute("aria-pressed", `${state.paused}`);
  $("connection").textContent = state.paused ? "Paused" : "Reconnecting";
  if (!state.paused) refresh();
});
for (const definition of [
  ...sections.map(section => ({...section, kind: "rl"})),
  ...sftSections.map(section => ({...section, kind: "sft"})),
]) {
  const section = document.createElement("details");
  section.id = `section-${definition.id}`;
  section.className = "metric-section";
  section.dataset.kind = definition.kind;
  section.hidden = definition.kind !== state.kind;
  section.open = typeof sectionOpen[definition.id] === "boolean" ? sectionOpen[definition.id] : definition.open;
  const summary = document.createElement("summary");
  summary.append(text("h2", definition.title), text("span", definition.note, "section-note"));
  const plots = text("div", "", "plots");
  section.append(summary, plots);
  $("plots").append(section);
  const charts = definition.metrics.map(metric => new Chart(metric, plots, section));
  state.charts.push(...charts);
  section.addEventListener("toggle", () => {
    sectionOpen[definition.id] = section.open;
    try { localStorage.setItem("rl-section-open", JSON.stringify(sectionOpen)); } catch {}
    $("tooltip").hidden = true;
    for (const chart of charts) {
      chart.crosshair = null;
      chart.drag = null;
      if (section.open) chart.update();
      else {
        chart.series = null;
        chart.canvas.width = 0;
        chart.canvas.height = 0;
      }
    }
    updateStepCounters();
  });
}
refresh();
setInterval(refresh, 3000);
setInterval(updateStepCounters, 1000);
