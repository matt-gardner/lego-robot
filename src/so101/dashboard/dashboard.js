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
  {id: "sft-loss", title: "Sample-matched training loss", note: "Training samples · log-log", open: true,
    metrics: [
      {key: "loss", title: "Mean total loss"},
      {key: "action_loss", title: "Mean action loss"},
      {key: "box_loss", title: "Mean box loss"},
    ].map(metric => ({...metric, phase: "train", task: "all", unit: "", logLog: true,
      note: "Losses are normalized by valid action/entity labels, not summed across the batch. No additional batch-size division. Missing components remain gaps, not zeros."}))},
  {...sections[0], id: "sft-eval"},
  {id: "sft-frozen-loss", title: "Validation loss (fixed held-out set)",
    note: "Fixed held-out set · training samples", open: true,
    metrics: [{key: "supervised_loss", title: "Validation loss", phase: "eval", task: "all", unit: "",
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
const evaluationSections = ["recovery-heldout", "recovery-training"].map(task => ({
  id: `evaluation-${task}`, title: `${task} policy success`, open: true,
  note: "Teacher-free inference · recovery-scene partition, not an original-SFT test split · incomplete cells are not failures.",
  metrics: [
    {key: "success_rate", title: "Success rate", unit: "%", percent: true, bounds: [0, 100]},
    {key: "samples", title: "Completed rollouts", unit: ""},
  ].map(metric => ({...metric, phase: "eval", task})),
}));
const runKinds = ["rl", "sft", "recognition", "evaluation"];
const supervised = kind => kind === "sft" || kind === "recognition";
const recognitionSections = [
  {id: "recognition-training-accuracy", title: "Training minibatch accuracy", open: true,
    note: "Query-weighted over the logged optimizer microbatches. Missing measurements stay absent; loss is not accuracy.",
    metrics: ["top1", "top3", "top5"].map(key => ({
      key: `train_${key}`, title: `Training minibatch ${key}`, phase: "train", task: "all", unit: "%", percent: true,
    }))},
  {id: "recognition-dev", title: "Recognition dev evaluation", open: true,
    note: "Visible box-view observations, not independent pieces. Raw synthetic recognition, not action success or real-transfer qualification.",
    metrics: [
      ...["top1", "top3", "top5", "macro_top1"].map(key => ({
        key, title: `Dev ${key.replace("_", " ")}`, phase: "eval", task: "all", unit: "%", percent: true,
        note: key === "macro_top1" ? "Unweighted mean across part IDs." : "Retrieval accuracy on the dev split.",
      })),
      ...["family", "kind"].map(name => ({
        key: `category_${name}_accuracy`, title: `Dev ${name} category accuracy`, phase: "eval", task: "all", unit: "%", percent: true,
      })),
      ...["seen_id", "unseen_id"].flatMap(name => ["top1", "top3", "top5"].map(key => ({
        key: `${name}_${key}`, title: `Dev ${name.replace("_", " ")} ${key}`, phase: "eval", task: "all", unit: "%", percent: true,
      }))),
    ]},
  {id: "recognition-loss", title: "Recognition-only training loss", note: "No action SFT or RL · training samples", open: true,
    metrics: [{key: "loss", title: "Recognition loss", phase: "train", task: "all", unit: "", logLog: true,
      note: "Recognition embedding and category objective. Not comparable to action SFT loss."}]},
  {...sftSections.at(-1), id: "recognition-health"},
];
function readStorage(key) {
  try { return JSON.parse(localStorage.getItem(key)); } catch { return null; }
}
const finite = value => typeof value === "number" && Number.isFinite(value);
const record = value => value !== null && typeof value === "object" && !Array.isArray(value);
const savedSelections = Object.fromEntries(runKinds.map(kind => {
  const value = readStorage(`${kind}-selected-runs`);
  return [kind, Array.isArray(value) && value.every(id => typeof id === "string" && id.length > 0) ? value : null];
}));
const savedSections = readStorage("rl-section-open");
const sectionOpen = Object.fromEntries([...sections, ...sftSections, ...recognitionSections, ...evaluationSections]
  .filter(section => record(savedSections) && typeof savedSections[section.id] === "boolean")
  .map(section => [section.id, savedSections[section.id]]));
const storedPreferences = readStorage("so101-dashboard-preferences");
const preferences = record(storedPreferences) && storedPreferences.version === 1 ? storedPreferences : {};
const tabViews = Object.fromEntries(runKinds.map(kind => {
  const view = record(preferences.views) && record(preferences.views[kind]) ? preferences.views[kind] : {};
  return [kind, {
    focus: typeof view.focus === "string" && view.focus.length > 0 ? view.focus : null,
    zoom: Array.isArray(view.zoom) && view.zoom.length === 2 && view.zoom.every(finite)
      && view.zoom[0] < view.zoom[1] ? view.zoom : null,
  }];
}));
const initialKind = runKinds.includes(preferences.kind) ? preferences.kind : "rl";
const selections = Object.fromEntries(runKinds.map(kind => [kind, new Set(savedSelections[kind] || [])]));
const state = {
  catalog: [], selected: selections[initialKind], runs: new Map(),
  initialized: false, smoothing: finite(preferences.smoothing) && preferences.smoothing >= 0
    && preferences.smoothing <= 95 && preferences.smoothing % 5 === 0 ? preferences.smoothing / 100 : 0,
  paused: false, ...tabViews[initialKind], loading: false, errors: [], charts: [],
  kind: initialKind, sftInitialized: false, recognitionInitialized: false,
};
const pct = value => finite(value) ? `${(value * 100).toFixed(1)}%` : "—";
const num = value => finite(value) ? value.toFixed(3) : "—";
const exp = value => finite(value) ? value.toExponential(1) : "—";
const shortName = run => run.name.replaceAll("entity_smolvla_", "").replaceAll("trajectory_rl_", "");
const runColors = new Map();
function color(id) {
  if (!runColors.has(id)) {
    const index = runColors.size;
    runColors.set(id, colors[index] || `hsl(${(index * 137.508) % 360} 70% 65%)`);
  }
  return runColors.get(id);
}
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
  const active = state.catalog.filter(run => run.active === true);
  container.hidden = !active.length;
  container.replaceChildren(...active.map(summary => {
    const run = state.runs.get(summary.id);
    const elapsed = finite(run?.last_step_at) ? Date.now() / 1000 - run.last_step_at : null;
    const prefix = `${shortName(summary).split("/").at(-1)} · `;
    const remote = summary.remote;
    const reportAge = finite(remote?.observed_at) ? Math.max(0, Date.now() / 1000 - remote.observed_at) : null;
    const freshReport = finite(reportAge) && reportAge <= 90 && remote.state !== "stale";
    const phaseAge = finite(remote?.phase_observed_at) ? Date.now() / 1000 - remote.phase_observed_at : null;
    const interval = run?.config?.run_config?.validation_interval;
    const latestStep = run?.latest_train?.step;
    const evaluationDue = supervised(runKind(summary)) && finite(interval) && interval > 0
      && finite(latestStep) && latestStep > 0 && latestStep % interval === 0
      && (run.latest_eval?.step ?? -1) < latestStep;
    let phase = "running";
    if (remote && !freshReport) phase = "stale";
    else if (freshReport && finite(phaseAge) && phaseAge >= -30 && phaseAge <= 90
      && ["training", "evaluating", "checkpointing"].includes(remote.phase)) {
      phase = remote.phase;
    } else if (summary.active === true && evaluationDue) {
      phase = "evaluation pending";
    }
    const since = summary.active === true
      ? `since update ${run?.last_step_time_estimated ? "~" : ""}${duration(elapsed)}`
      : "";
    if (runKind(summary) === "evaluation") {
      return text("span", `${prefix}teacher-free evaluation · ${summary.progress?.completed ?? 0}/${summary.progress?.expected ?? "?"} completed rollouts` +
        (remote ? ` · report ${duration(reportAge)}` : ""));
    }
    const counter = text("span",
      `${prefix}${phase} · step ${latestStep ?? "—"} · optimizer ${duration(run?.latest_train?.step_seconds)}` +
      (since ? ` · ${since}` : "") +
      (remote ? ` · report ${duration(reportAge)}` : "") +
      (finite(run?.latest_eval?.evaluation_seconds) ? ` · last eval ${duration(run.latest_eval.evaluation_seconds)}` : ""));
    counter.title = summary.path || summary.name;
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
      state.kind === "evaluation"
        ? `${run.progress?.completed ?? 0}/${run.progress?.expected ?? "?"} completed rollouts`
        : `step ${latest}${supervised(state.kind) ? ` · loss ${num(run.latest_train?.loss)}` : ""} · ${state.kind === "recognition" ? "dev top1" : "eval"} ${pct(state.kind === "recognition" ? run.latest_eval?.top1 : run.latest_eval?.success_rate)}`));
    meta.append(document.createElement("br"), document.createTextNode(
      `${run.archived ? "Archive · " : ""}Log updated ${run.updated_at ? new Date(run.updated_at).toLocaleString() : "unavailable"}`));
    if (run.remote) {
      meta.append(document.createElement("br"), document.createTextNode(
        `Cloud: ${run.remote.state} · contact ${duration(run.remote.contact_age_seconds)} ago` +
        (state.kind === "evaluation" ? "" : ` · backup step ${run.remote.backup?.step ?? "none"}`)));
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
  const sft = supervised(state.kind);
  const recognition = state.kind === "recognition";
  const evaluation = state.kind === "evaluation";
  $("recent-title").textContent = evaluation ? "Measured assistance levels" : "Recent steps";
  const cards = evaluation ? [
    ["Completed rollouts", `${run?.progress?.completed ?? 0}/${run?.progress?.expected ?? "?"}`,
      run?.progress?.complete ? "Evaluation complete" : "Partial population; incomplete cells are not failures"],
    ["Recovery-heldout success", pct(latest?.success_rate),
      latest ? `${latest.step}% saved-state assistance · n=${latest.samples}` : "No recovery-heldout rollouts completed"],
    ["Policy updates", "None", "Teacher-free inference with a frozen checkpoint"],
    ["Scene split", "Recovery-specific", "Recovery-heldout is not an original-SFT held-out benchmark"],
  ] : recognition ? [
    ["Latest dev top1 / top3 / top5", latest ? `${pct(latest.top1)} / ${pct(latest.top3)} / ${pct(latest.top5)}` : "—",
      latest ? `Step ${latest.step} · ${latest.n} box-view observations` : "No dev evaluation logged"],
    ["Dev macro top1", pct(latest?.macro_top1), latest ? `${latest.families} part IDs · unweighted` : "No dev evaluation logged"],
    ["Dev seen / unseen top1", latest ? `${pct(latest.seen_id_top1)} / ${pct(latest.unseen_id_top1)}` : "—",
      latest ? `n=${latest.seen_id_n} / ${latest.unseen_id_n} box-view observations` : "No dev evaluation logged"],
    ["Dev category family / kind", latest ? `${pct(latest.category_family_accuracy)} / ${pct(latest.category_kind_accuracy)}` : "—",
      latest ? `Majority baseline top1 ${pct(latest.majority_baseline_top1)}` : "No dev evaluation logged"],
  ] : [
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
  const results = $("recognition-results");
  results.hidden = !recognition;
  results.replaceChildren();
  if (recognition) {
    results.append(text("h2", `Selected-checkpoint evaluation${run ? ` · ${shortName(run)}` : ""}`),
      text("p", "Full training-set measurements and final held-out test are separate from minibatch training and dev. Counts are visible box-view observations, not independent pieces.", "hint"));
    const table = document.createElement("table");
    const head = document.createElement("tr");
    const columns = ["Split", "Checkpoint step", "Observations", "Top1", "Top3", "Top5", "Macro top1",
      "Seen n", "Seen top1 / top3 / top5", "Unseen n", "Unseen top1 / top3 / top5", "Category family / kind"];
    head.append(...columns.map(name => text("th", name)));
    table.append(head);
    for (const [split, row] of [
      ...(run?.training_evaluation || []).map(row => ["Training set (selected checkpoint)", row]),
      ...(run?.final_test ? [["Final held-out test", run.final_test]] : []),
    ]) {
      const tr = document.createElement("tr");
      tr.append(...[split, row.selected_step ?? row.step, row.n, pct(row.top1), pct(row.top3), pct(row.top5),
        pct(row.macro_top1), row.seen_id_n,
        `${pct(row.seen_id_top1)} / ${pct(row.seen_id_top3)} / ${pct(row.seen_id_top5)}`,
        row.unseen_id_n, `${pct(row.unseen_id_top1)} / ${pct(row.unseen_id_top3)} / ${pct(row.unseen_id_top5)}`,
        `${pct(row.category_family_accuracy)} / ${pct(row.category_kind_accuracy)}`,
      ].map(value => text("td", `${value ?? "—"}`)));
      table.append(tr);
    }
    const wrap = text("div", "", "table-wrap");
    wrap.append(table);
    results.append(wrap,
      text("p", run?.final_test ? `Test checkpoint selected by ${run.final_test.selection}. ${run.final_test.scope || ""}`
        : "Final held-out test not published for this run.", "hint"));
    if (!run?.training_evaluation?.length) results.append(text("p", "Full training-set checkpoint accuracy not published. No historical training accuracy is inferred.", "hint"));
  }
  const training = points(run, "train", "all").slice(-10);
  const cutoff = training[0]?.step ?? Infinity;
  const preceding = points(run, "eval", "all").filter(row => sft || row.step < cutoff).slice(-2);
  const recent = evaluation ? (run?.series || []).flatMap(series =>
    series.points.map(row => ({...row, phase: series.task}))) : [
    ...preceding.map(row => ({...row, phase: "eval"})),
    ...training.map(row => ({...row, phase: "train"})),
  ];
  if (sft) recent.sort((a, b) => a.step - b.step || (a.phase === "eval" ? 1 : -1));
  const headings = evaluation
    ? ["Saved-state assistance (%)", "Recovery partition", "Success", "Completed rollouts"]
    : recognition
    ? ["Step", "Phase", "Loss", "Top1", "Top3", "Top5", "Macro top1", "Observations", "LR"]
    : sft
    ? ["Step", "Phase", "Loss", "Success", "Efficiency", "Eval RMS jerk", "Wrong object", "LR", "Grad norm", "Step time (s)"]
    : ["Step", "Phase", "Success", "Efficiency", "Jerk gap / eval RMS", "Wrong object", "LR", "Grad norm", "Assistance"];
  $("recent-head").replaceChildren(...headings.map(label => {
    const th = text("th", label);
    if (label === "Jerk gap / eval RMS") th.title = "Training: jerk minus reference margin. Validation: absolute RMS jerk (no margin logged).";
    return th;
  }));
  $("recent-note").textContent = evaluation
    ? "Teacher-free completed rollouts only. Recovery-training and recovery-heldout scenes remain separate; no optimizer steps or unassisted-validation claim."
    : recognition
    ? latest?.scope || "Recognition-only training and dev evaluation. Counts are box-view observations, not independent pieces."
    : sft
    ? "Most recent 10 loss-only optimizer steps and two latest validation rows. Success, efficiency, absolute RMS jerk and wrong-object metrics are evaluation only."
    : "Most recent 10 optimizer steps, plus the two preceding validation-only rows. Training success includes assisted and discarded trajectories; validation is unassisted.";
  $("recent").replaceChildren(...recent.map(row => {
    const tr = text("tr", "", row.phase === "eval" ? "eval-row" : "");
    const values = evaluation ? [
      row.step, row.phase, pct(row.success_rate), row.samples,
    ] : recognition ? [
      row.step, row.phase === "eval" ? "Dev evaluation" : "Recognition loss",
      row.phase === "train" ? num(row.loss) : "—",
      pct(row.top1), pct(row.top3), pct(row.top5), pct(row.macro_top1), row.n ?? "—", exp(row.learning_rate),
    ] : sft ? [
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
        const x = supervised(state.kind) ? point.samples_seen : point.step;
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
        label: `${shortName(run)} · ${series.task} · ${series.phase === "eval" ? (state.kind === "recognition" ? "dev" : state.kind === "evaluation" ? "teacher-free" : "validation") : "train"}`,
        values, color: color(run.id), axis: supervised(state.kind) ? "Training samples seen" : run.axis,
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
    this.canvas.setAttribute("aria-label", `${section.querySelector("h2").textContent}: ${metric.title} versus ${supervised(section.dataset.kind) ? "training samples seen" : section.dataset.kind === "evaluation" ? "saved-state assistance percentage" : "optimizer step"}${metric.logLog ? ", both axes logarithmic" : ""}`);
    this.legend = text("div", "", "legend");
    const heading = text("h2", metric.title);
    this.panel.append(heading);
    this.panel.append(this.canvas, this.legend);
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
      ctx.fillText("No data", W / 2, H / 2);
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
      const position = state.kind === "evaluation" ? `${point.step}% assistance · n=${point.samples}` :
        `${supervised(state.kind) ? `${point.x} samples · ` : ""}step ${point.step}`;
      const line = text("div", `${series.label} · ${position}: ${value}${this.metric.unit === "%" ? "%" : ` ${this.metric.unit}`}`);
      if (state.kind === "recognition") line.append(document.createTextNode(
        ` · n=${point.n ?? point.train_accuracy_n ?? "unreported"} box-view observations`));
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
  for (const kind of runKinds) {
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
  $("context").textContent = !chosen().length ? "No runs selected" :
    `${chosen().length} selected · ${shortName(chosen().find(r => r.id === state.focus) || chosen()[0])}`;
  showErrors();
}

async function request(path) {
  const response = await fetch(`${basePath}${path}`, {cache: "no-store", signal: AbortSignal.timeout(10000)});
  if (!response.ok) throw new Error(`HTTP ${response.status}: ${path}`);
  return response.json();
}

async function loadDetails() {
  await Promise.all(state.catalog.filter(run => state.selected.has(run.id) && runKind(run) === state.kind
    || run.active === true).map(async run => {
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
    for (const run of [...state.catalog].sort((a, b) => a.id.localeCompare(b.id))) color(run.id);
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
    for (const kind of ["sft", "recognition", "evaluation"]) if (!state[`${kind}Initialized`]) {
      const sftCatalog = state.catalog.filter(run => runKind(run) === kind);
      const existing = new Set(sftCatalog.map(run => run.id));
      for (const id of selections[kind]) if (!existing.has(id)) selections[kind].delete(id);
      if (!selections[kind].size && savedSelections[kind]?.length !== 0) {
        const first = sftCatalog.find(run => !run.archived && run.active === true)
          || sftCatalog.find(run => !run.archived) || sftCatalog[0];
        if (first) selections[kind].add(first.id);
      }
      state[`${kind}Initialized`] = sftCatalog.length > 0;
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
for (const kind of runKinds) {
  $(`tab-${kind}`).addEventListener("click", () => switchTab(kind));
  $(`tab-${kind}`).addEventListener("keydown", event => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const target = event.key === "Home" ? runKinds[0] : event.key === "End" ? runKinds.at(-1) :
      runKinds[(runKinds.indexOf(kind) + (event.key === "ArrowRight" ? 1 : runKinds.length - 1)) % runKinds.length];
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
  ...recognitionSections.map(section => ({...section, kind: "recognition"})),
  ...evaluationSections.map(section => ({...section, kind: "evaluation"})),
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
