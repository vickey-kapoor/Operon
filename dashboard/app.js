// Operon eval dashboard: renders data/index.json (one entry per benchmark run) and
// loads data/reports/<key>.json for a run's per-task results. No build step; all
// text from the data goes into the DOM through textContent.
"use strict";

const SVG_NS = "http://www.w3.org/2000/svg";
const BACKEND_NAMES = { claude_computer_use: "Claude", computer_use: "Gemini", json: "Gemini" };
// Colors follow the configuration, never its rank: known ones get fixed slots,
// anything new takes the next free slot in order of first appearance.
const FIXED_SLOTS = { "Claude": 1, "Gemini": 2, "Claude · thin": 3 };
const MAX_SLOTS = 6;
const OUTCOME_LABELS = {
  success: "pass", false_success: "false pass", missed: "missed",
  failure: "fail", timeout: "timeout", error: "error",
};

const state = {
  runs: [],
  tab: "evals",
  mode: "browser",
  suite: "full",
  showFlagged: false,
  selected: null,
  costSelected: null,
  slots: {},
  reports: {},
};

// ---------- small helpers ----------

function el(tag, attrs, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "style") node.style.cssText = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function svg(tag, attrs) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) node.setAttribute(k, v);
  return node;
}

function store(key, value) {
  try {
    if (value === undefined) return localStorage.getItem(key);
    localStorage.setItem(key, value);
  } catch (_) { /* storage can be unavailable; the page works without it */ }
  return null;
}

function parseStamp(stamp) {
  const m = /^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z$/.exec(stamp || "");
  return m ? new Date(Date.UTC(+m[1], m[2] - 1, +m[3], +m[4], +m[5], +m[6])) : new Date(stamp);
}

const dateFmt = new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric" });
const timeFmt = new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
const fmtPct = (v) => (v === null || v === undefined ? "–" : `${Math.round(v * 100)}%`);
const fmtSec = (v) => (v === null || v === undefined ? "–" : `${Number(v).toFixed(v >= 100 ? 0 : 1)} s`);
const fmtNum = (v) => (v === null || v === undefined ? "–" : String(v));
function fmtUsd(v) {
  if (v === null || v === undefined) return "–";
  if (v === 0) return "$0";
  if (v < 0.01) return `$${v.toFixed(4)}`;
  if (v < 10) return `$${v.toFixed(3)}`;
  return `$${v.toFixed(2)}`;
}
function fmtTokens(v) {
  if (v === null || v === undefined) return "–";
  if (v >= 1e6) return `${(v / 1e6).toFixed(2)}M`;
  if (v >= 1e3) return `${(v / 1e3).toFixed(1)}k`;
  return String(v);
}

function configLabel(run) {
  const c = run.config || {};
  const name = BACKEND_NAMES[c.backend] || c.backend || "Unknown backend";
  return (c.harness_profile || "full") === "thin" ? `${name} · thin` : name;
}

function seriesColor(label) {
  return `var(--series-${state.slots[label] || MAX_SLOTS})`;
}

function assignSlots(runs) {
  const used = new Set(Object.values(FIXED_SLOTS));
  let next = 1;
  for (const run of runs) {
    const label = configLabel(run);
    if (state.slots[label]) continue;
    if (FIXED_SLOTS[label]) { state.slots[label] = FIXED_SLOTS[label]; continue; }
    while (used.has(next) && next < MAX_SLOTS) next++;
    state.slots[label] = next;
    used.add(next);
  }
}

function key(label, cls) {
  return el("span", { class: cls || "key", style: `background:${seriesColor(label)}` });
}

function configCell(run) {
  return el("span", { class: "config-cell" }, key(configLabel(run)), configLabel(run));
}

function runLink(run, text) {
  const url = run.config && run.config.github_run_url;
  return url ? el("a", { href: url, target: "_blank", rel: "noopener" }, text || "run ↗") : "–";
}

function commitLink(run) {
  const sha = run.config && run.config.git_sha;
  return sha
    ? el("a", { class: "sha", href: `https://github.com/vickey-kapoor/Operon/commit/${sha}`, target: "_blank", rel: "noopener" }, sha.slice(0, 7))
    : "–";
}

function pill(outcome) {
  return el("span", { class: `pill ${outcome || "none"}` }, OUTCOME_LABELS[outcome] || outcome || "–");
}

// ---------- data selection ----------

function filteredRuns() {
  return state.runs.filter((run) =>
    run.mode === state.mode &&
    (run.suite || "").replace(/^desktop-/, "") === state.suite &&
    (state.showFlagged || !(run.config && run.config.flagged)));
}

function latestByConfig(runs) {
  const latest = new Map();
  for (const run of runs) {
    if (run.config && run.config.flagged) continue;
    const label = configLabel(run);
    const have = latest.get(label);
    if (!have || run.started_at > have.started_at) latest.set(label, run);
  }
  return [...latest.values()].sort((a, b) => (state.slots[configLabel(a)] || 9) - (state.slots[configLabel(b)] || 9));
}

function seriesFor(runs, value) {
  const groups = new Map();
  for (const run of runs) {
    const y = value(run);
    if (y === null || y === undefined || Number.isNaN(y)) continue;
    const label = configLabel(run);
    if (!groups.has(label)) groups.set(label, []);
    groups.get(label).push({ x: parseStamp(run.started_at), y, run });
  }
  return [...groups.entries()]
    .map(([label, points]) => ({ label, color: seriesColor(label), points: points.sort((a, b) => a.x - b.x) }))
    .sort((a, b) => (state.slots[a.label] || 9) - (state.slots[b.label] || 9));
}

async function loadReport(run) {
  if (!state.reports[run.key]) {
    const response = await fetch(`data/${run.file}`, { cache: "no-cache" });
    if (!response.ok) throw new Error(`${run.file}: ${response.status}`);
    state.reports[run.key] = await response.json();
  }
  return state.reports[run.key];
}

// ---------- line chart ----------

function niceMax(max) {
  if (max <= 0) return 1;
  const step = Math.pow(10, Math.floor(Math.log10(max)));
  for (const m of [1, 2, 2.5, 5, 10]) if (m * step >= max) return m * step;
  return 10 * step;
}

function lineChart(host, legendHost, series, opts) {
  host.replaceChildren();
  legendHost.replaceChildren();
  if (!series.length) {
    host.append(el("div", { class: "empty" }, "No runs for this selection."));
    return;
  }
  if (series.length > 1) {
    for (const s of series) legendHost.append(el("span", {}, key(s.label, "line-key"), s.label));
  }

  const width = Math.max(host.clientWidth, 280);
  const height = 260;
  const m = { l: 48, r: 52, t: 12, b: 28 };
  const all = series.flatMap((s) => s.points);
  let x0 = Math.min(...all.map((p) => +p.x));
  let x1 = Math.max(...all.map((p) => +p.x));
  const pad = Math.max((x1 - x0) * 0.06, 6 * 3600e3);
  x0 -= pad; x1 += pad;
  const yMax = opts.yMax || niceMax(Math.max(...all.map((p) => p.y)) * 1.1);
  const sx = (x) => m.l + ((+x - x0) / (x1 - x0)) * (width - m.l - m.r);
  const sy = (y) => m.t + (1 - y / yMax) * (height - m.t - m.b);

  const root = svg("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": opts.title || "chart" });
  const axis = svg("g", { class: "axis" });
  for (let i = 0; i <= 4; i++) {
    const v = (yMax * i) / 4;
    axis.append(svg("line", { class: i === 0 ? "baseline" : "gridline", x1: m.l, x2: width - m.r, y1: sy(v), y2: sy(v) }));
    const t = svg("text", { x: m.l - 8, y: sy(v) + 4, "text-anchor": "end" });
    t.textContent = opts.fmt(v, true);
    axis.append(t);
  }
  // One tick per day, thinned so labels never collide.
  const day = 86400e3;
  const firstDay = new Date(x0); firstDay.setHours(0, 0, 0, 0);
  const days = [];
  for (let d = +firstDay + day; d <= x1; d += day) days.push(d);
  const every = Math.max(1, Math.ceil(days.length / Math.max(1, Math.floor((width - m.l - m.r) / 70))));
  days.forEach((d, i) => {
    if (i % every) return;
    const t = svg("text", { x: sx(d), y: height - 8, "text-anchor": "middle" });
    t.textContent = dateFmt.format(new Date(d));
    axis.append(t);
  });
  root.append(axis);

  for (const s of series) {
    if (s.points.length > 1) {
      const d = s.points.map((p, i) => `${i ? "L" : "M"}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join("");
      root.append(svg("path", { d, fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }));
    }
    for (const p of s.points) {
      root.append(svg("circle", { cx: sx(p.x), cy: sy(p.y), r: 4, fill: s.color, stroke: "var(--surface-1)", "stroke-width": 2 }));
    }
  }
  // Direct labels: the latest value of each series, nudged apart vertically.
  const labels = series.map((s) => {
    const last = s.points[s.points.length - 1];
    return { s, x: sx(last.x), y: sy(last.y), text: opts.fmt(last.y) };
  }).sort((a, b) => a.y - b.y);
  for (let i = 1; i < labels.length; i++) {
    if (labels[i].y - labels[i - 1].y < 13) labels[i].y = labels[i - 1].y + 13;
  }
  for (const l of labels) {
    const t = svg("text", { class: "dlabel", x: l.x + 8, y: l.y + 4 });
    t.textContent = l.text;
    root.append(t);
  }

  const cross = svg("line", { class: "crosshair", y1: m.t, y2: height - m.b, visibility: "hidden" });
  root.append(cross);
  const hit = svg("rect", { x: m.l, y: 0, width: width - m.l - m.r, height, fill: "transparent", tabindex: 0 });
  root.append(hit);
  host.append(root);
  const tip = el("div", { class: "tooltip" });
  host.append(tip);

  const sorted = [...all].sort((a, b) => a.x - b.x);
  let focusIndex = sorted.length - 1;
  function show(point) {
    const scale = root.getBoundingClientRect().width / width;
    const px = sx(point.x);
    cross.setAttribute("x1", px); cross.setAttribute("x2", px);
    cross.setAttribute("visibility", "visible");
    // Runs launched together (one workflow) land within a few hours of each other.
    const near = series.map((s) => {
      const best = s.points.reduce((b, p) => (Math.abs(p.x - point.x) < Math.abs(b.x - point.x) ? p : b));
      return Math.abs(best.x - point.x) <= 3 * 3600e3 ? { s, p: best } : null;
    }).filter(Boolean);
    tip.replaceChildren(el("div", { class: "t-head" }, timeFmt.format(point.x)));
    for (const { s, p } of near) {
      tip.append(el("div", { class: "t-row" }, key(s.label, "line-key"), el("strong", {}, opts.fmt(p.y)),
        el("span", {}, opts.detail ? `${s.label} · ${opts.detail(p.run)}` : s.label)));
    }
    tip.style.display = "block";
    const left = px * scale + 12;
    const maxLeft = host.clientWidth - tip.offsetWidth - 4;
    tip.style.left = `${Math.max(0, left > maxLeft ? px * scale - tip.offsetWidth - 12 : left)}px`;
    tip.style.top = `${Math.max(0, sy(point.y) * scale - tip.offsetHeight - 10)}px`;
  }
  function hide() { tip.style.display = "none"; cross.setAttribute("visibility", "hidden"); }
  hit.addEventListener("pointermove", (event) => {
    const box = root.getBoundingClientRect();
    const x = ((event.clientX - box.left) / box.width) * width;
    let best = 0;
    sorted.forEach((p, i) => { if (Math.abs(sx(p.x) - x) < Math.abs(sx(sorted[best].x) - x)) best = i; });
    focusIndex = best;
    show(sorted[best]);
  });
  hit.addEventListener("pointerleave", hide);
  hit.addEventListener("focus", () => show(sorted[focusIndex]));
  hit.addEventListener("blur", hide);
  hit.addEventListener("keydown", (event) => {
    if (event.key === "ArrowLeft") focusIndex = Math.max(0, focusIndex - 1);
    else if (event.key === "ArrowRight") focusIndex = Math.min(sorted.length - 1, focusIndex + 1);
    else return;
    event.preventDefault();
    show(sorted[focusIndex]);
  });
}

// ---------- evals tab ----------

function tile(run, { value, of, delta, meter, stats }) {
  const label = configLabel(run);
  return el("div", { class: "tile", style: `--tile-color:${seriesColor(label)}` },
    el("div", { class: "who" }, key(label), label, el("span", { class: "model" }, (run.config && run.config.model) || "")),
    el("div", { class: "big" }, el("strong", {}, value), of ? el("span", { class: "of" }, of) : null, delta || null),
    meter === undefined ? null : el("div", { class: "meter", role: "img", "aria-label": fmtPct(meter) },
      el("span", { style: `width:${Math.max(0, Math.min(1, meter)) * 100}%` })),
    el("dl", {}, stats.map(([dt, dd, style]) => el("div", {}, el("dt", {}, dt), el("dd", { style: style || null }, dd)))),
    el("div", { class: "when" }, el("span", {}, timeFmt.format(parseStamp(run.started_at))), runLink(run, "View run ↗")));
}

// Change in pass rate since this configuration's previous comparable run.
function passDelta(run, runs) {
  const label = configLabel(run);
  const earlier = runs.filter((r) => configLabel(r) === label && r.started_at < run.started_at && !(r.config && r.config.flagged));
  if (!earlier.length) return null;
  const prev = earlier[earlier.length - 1];
  const points = Math.round((run.summary.success_rate - prev.summary.success_rate) * 100);
  const cls = points > 0 ? "delta up" : points < 0 ? "delta down" : "delta";
  const text = points > 0 ? `▲ ${points} pts` : points < 0 ? `▼ ${-points} pts` : "no change";
  return el("span", { class: cls, title: `vs ${timeFmt.format(parseStamp(prev.started_at))}` }, text);
}

function renderTiles(runs) {
  const host = document.getElementById("tiles");
  host.replaceChildren();
  const latest = latestByConfig(runs);
  if (!latest.length) {
    host.append(el("div", { class: "card empty" }, "No runs yet for this environment and suite."));
    return;
  }
  for (const run of latest) {
    const s = run.summary;
    host.append(tile(run, {
      value: fmtPct(s.success_rate),
      of: `${s.success} of ${s.tasks} tasks`,
      delta: passDelta(run, runs),
      meter: s.success_rate,
      stats: [
        ["Per task", fmtSec(s.mean_seconds)],
        ["Per step", fmtSec(s.seconds_per_step)],
        ["Steps", fmtNum(s.mean_steps)],
        ["False pass", fmtNum(s.false_success), s.false_success ? "color:var(--critical)" : ""],
      ],
    }));
  }
}

function renderCharts(runs) {
  const detail = (run) => `${run.summary.success}/${run.summary.tasks}`;
  lineChart(document.getElementById("ch-pass"), document.getElementById("lg-pass"),
    seriesFor(runs, (r) => r.summary.success_rate),
    { yMax: 1, fmt: (v) => fmtPct(v), detail, title: "Pass rate by run" });
  lineChart(document.getElementById("ch-speed"), document.getElementById("lg-speed"),
    seriesFor(runs, (r) => r.summary.mean_seconds),
    { fmt: (v, axis) => (axis ? `${Math.round(v)}` : fmtSec(v)), detail: (r) => `${r.summary.mean_steps} steps`, title: "Seconds per task by run" });
}

async function renderMatrix(runs) {
  const table = document.getElementById("matrix");
  const columns = latestByConfig(runs);
  const token = (state.matrixToken = (state.matrixToken || 0) + 1);
  if (!columns.length) { table.replaceChildren(); return; }
  const reports = await Promise.all(columns.map((run) => loadReport(run).catch(() => null)));
  if (token !== state.matrixToken) return;  // a newer render owns the table
  table.replaceChildren();
  const tasks = [];
  const byColumn = reports.map((report) => {
    const map = new Map();
    for (const r of (report && report.results) || []) {
      if (!tasks.some((t) => t.task === r.task)) tasks.push({ task: r.task, category: r.category });
      map.set(r.task, r);
    }
    return map;
  });
  table.append(el("thead", {}, el("tr", {}, el("th", {}, "Task"), el("th", {}, "Category"),
    columns.map((run) => el("th", { class: "col" }, configCell(run))))));
  const body = el("tbody");
  const rows = tasks.map((t) => {
    const results = byColumn.map((m) => m.get(t.task));
    const outcomes = new Set(results.filter(Boolean).map((r) => r.outcome));
    return { t, results, differs: outcomes.size > 1, allPass: outcomes.size === 1 && outcomes.has("success") };
  });
  // Tasks someone got wrong first; the ones every configuration passed collapse.
  rows.sort((a, b) => a.allPass - b.allPass);
  const passed = rows.filter((r) => r.allPass).length;
  const more = document.getElementById("matrix-more");
  more.hidden = passed === 0 || passed === rows.length;
  more.textContent = state.matrixAll ? "Hide tasks every configuration passed" : `Show ${passed} tasks every configuration passed`;
  more.onclick = () => { state.matrixAll = !state.matrixAll; renderMatrix(filteredRuns()); };
  for (const { t, results, differs, allPass } of rows) {
    if (allPass && !state.matrixAll && passed !== rows.length) continue;
    body.append(el("tr", { class: differs ? "differs" : "" },
      el("td", {}, t.task), el("td", {}, t.category || ""),
      results.map((r) => el("td", { class: "cell", title: r ? `${r.steps} steps, ${r.seconds} s — ${r.judge_reason || ""}` : "not run" },
        r ? pill(r.outcome) : "–"))));
  }
  table.append(body);
}

function renderRuns(runs) {
  const table = document.getElementById("runs");
  table.replaceChildren(el("thead", {}, el("tr", {},
    ["When", "Configuration", "Model", "Passed", "Rate", "False pass", "s / task", "s / step", "Steps", "Commit", "Run"]
      .map((h, i) => el("th", { class: i >= 3 && i <= 8 ? "num" : "" }, h)))));
  const body = el("tbody");
  for (const run of [...runs].reverse()) {
    const s = run.summary;
    const flagged = run.config && run.config.flagged;
    body.append(el("tr", {
      class: `clickable${state.selected === run.key ? " selected" : ""}`,
      tabindex: 0,
      onclick: () => selectRun(run),
      onkeydown: (e) => { if (e.key === "Enter") selectRun(run); },
    },
      el("td", {}, timeFmt.format(parseStamp(run.started_at)), flagged ? el("div", { class: "flag" }, `⚑ ${flagged}`) : null),
      el("td", {}, configCell(run)),
      el("td", {}, (run.config && run.config.model) || "–"),
      el("td", { class: "num" }, `${s.success}/${s.tasks}`),
      el("td", { class: "num" }, fmtPct(s.success_rate)),
      el("td", { class: "num" }, fmtNum(s.false_success)),
      el("td", { class: "num" }, fmtSec(s.mean_seconds)),
      el("td", { class: "num" }, fmtSec(s.seconds_per_step)),
      el("td", { class: "num" }, fmtNum(s.mean_steps)),
      el("td", {}, commitLink(run)),
      el("td", { onclick: (e) => e.stopPropagation() }, runLink(run))));
  }
  table.append(body);
}

async function selectRun(run) {
  state.selected = run.key;
  renderRuns(filteredRuns());
  const card = document.getElementById("detail");
  card.hidden = false;
  document.getElementById("detail-title").textContent = `${configLabel(run)} · ${timeFmt.format(parseStamp(run.started_at))}`;
  const s = run.summary;
  document.getElementById("detail-sub").textContent =
    `${s.success}/${s.tasks} passed · ${s.false_success} false pass · ${s.missed} missed · ${s.failure} failed · ` +
    `${s.timeout} timeout · ${s.error} error · judge ${run.judge_model || "–"}`;
  const table = document.getElementById("detail-table");
  table.replaceChildren(el("tbody", {}, el("tr", {}, el("td", { class: "empty" }, "Loading…"))));
  try {
    const report = await loadReport(run);
    const order = { false_success: 0, failure: 1, missed: 2, timeout: 3, error: 4, success: 5 };
    const rows = [...report.results].sort((a, b) => (order[a.outcome] ?? 9) - (order[b.outcome] ?? 9));
    table.replaceChildren(
      el("thead", {}, el("tr", {}, ["Task", "Category", "Outcome", "Steps", "Seconds", "Stop reason", "Judge"]
        .map((h, i) => el("th", { class: i === 3 || i === 4 ? "num" : "" }, h)))),
      el("tbody", {}, rows.map((r) => el("tr", {},
        el("td", {}, r.task), el("td", {}, r.category), el("td", {}, pill(r.outcome)),
        el("td", { class: "num" }, fmtNum(r.steps)), el("td", { class: "num" }, fmtNum(r.seconds)),
        el("td", {}, r.stop_reason || "–"), el("td", { class: "wrap-text" }, r.judge_reason || "")))));
  } catch (err) {
    table.replaceChildren(el("tbody", {}, el("tr", {}, el("td", { class: "empty" }, `Could not load the report: ${err.message}`))));
  }
  card.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ---------- cost tab ----------

function hasCost(run) {
  return run.summary && run.summary.cost_usd !== undefined && run.summary.cost_usd !== null;
}

function costPerSuccess(run) {
  const s = run.summary;
  if (s.cost_per_success_usd !== undefined) return s.cost_per_success_usd;
  return s.success ? s.cost_usd / s.success : null;
}

function costPerTask(run) {
  const s = run.summary;
  if (s.cost_per_task_usd !== undefined) return s.cost_per_task_usd;
  return s.tasks ? s.cost_usd / s.tasks : null;
}

function renderCost() {
  const runs = filteredRuns().filter(hasCost);
  document.getElementById("cost-empty").hidden = runs.length > 0;
  document.getElementById("cost-body").hidden = runs.length === 0;
  if (!runs.length) return;

  const tiles = document.getElementById("cost-tiles");
  tiles.replaceChildren();
  for (const run of latestByConfig(runs)) {
    const s = run.summary;
    tiles.append(tile(run, {
      value: fmtUsd(costPerTask(run)),
      of: "per task",
      stats: [
        ["Per success", fmtUsd(costPerSuccess(run))],
        ["Run total", fmtUsd(s.cost_usd)],
        ["Judge", fmtUsd(s.judge_cost_usd)],
        ["Passed", `${s.success}/${s.tasks}`],
      ],
    }));
  }

  lineChart(document.getElementById("ch-cpt"), document.getElementById("lg-cpt"), seriesFor(runs, costPerTask),
    { fmt: (v) => fmtUsd(v), detail: (r) => `${fmtUsd(r.summary.cost_usd)} total`, title: "Cost per task by run" });
  lineChart(document.getElementById("ch-cps"), document.getElementById("lg-cps"), seriesFor(runs, costPerSuccess),
    { fmt: (v) => fmtUsd(v), detail: (r) => `${r.summary.success}/${r.summary.tasks} passed`, title: "Cost per successful task by run" });

  const table = document.getElementById("cost-runs");
  table.replaceChildren(el("thead", {}, el("tr", {},
    ["When", "Configuration", "Passed", "Agent cost", "Judge cost", "Per task", "Per success", "Input tokens", "Output tokens", "Cache reads"]
      .map((h, i) => el("th", { class: i >= 2 ? "num" : "" }, h)))));
  const body = el("tbody");
  for (const run of [...runs].reverse()) {
    const s = run.summary;
    body.append(el("tr", {
      class: `clickable${state.costSelected === run.key ? " selected" : ""}`,
      tabindex: 0,
      onclick: () => selectCostRun(run),
      onkeydown: (e) => { if (e.key === "Enter") selectCostRun(run); },
    },
      el("td", {}, timeFmt.format(parseStamp(run.started_at))),
      el("td", {}, configCell(run)),
      el("td", { class: "num" }, `${s.success}/${s.tasks}`),
      el("td", { class: "num" }, fmtUsd(s.cost_usd)),
      el("td", { class: "num" }, fmtUsd(s.judge_cost_usd)),
      el("td", { class: "num" }, fmtUsd(costPerTask(run))),
      el("td", { class: "num" }, fmtUsd(costPerSuccess(run))),
      el("td", { class: "num" }, fmtTokens(s.input_tokens)),
      el("td", { class: "num" }, fmtTokens(s.output_tokens)),
      el("td", { class: "num" }, fmtTokens(s.cache_read_tokens))));
  }
  table.append(body);
}

async function selectCostRun(run) {
  state.costSelected = run.key;
  renderCost();
  const wrap = document.getElementById("cost-detail");
  wrap.hidden = false;
  document.getElementById("cost-detail-sub").textContent =
    `${configLabel(run)} · ${timeFmt.format(parseStamp(run.started_at))} · ${fmtUsd(run.summary.cost_usd)} agent, ${fmtUsd(run.summary.judge_cost_usd)} judge`;
  const models = document.getElementById("cost-models");
  const byModel = Object.entries(run.summary.by_model || {}).sort((a, b) => (b[1].cost_usd || 0) - (a[1].cost_usd || 0));
  models.replaceChildren(
    el("thead", {}, el("tr", {}, ["Model", "Requests", "Input", "Output", "Cache read", "Cache write", "Cost"]
      .map((h, i) => el("th", { class: i ? "num" : "" }, h)))),
    el("tbody", {}, byModel.length
      ? byModel.map(([name, m]) => el("tr", {}, el("td", {}, name), el("td", { class: "num" }, fmtNum(m.requests)),
        el("td", { class: "num" }, fmtTokens(m.input_tokens)), el("td", { class: "num" }, fmtTokens(m.output_tokens)),
        el("td", { class: "num" }, fmtTokens(m.cache_read_tokens)), el("td", { class: "num" }, fmtTokens(m.cache_write_tokens)),
        el("td", { class: "num" }, fmtUsd(m.cost_usd))))
      : el("tr", {}, el("td", { class: "empty", colspan: 7 }, "No per-model breakdown in this report."))));
  const tasks = document.getElementById("cost-tasks");
  tasks.replaceChildren(el("tbody", {}, el("tr", {}, el("td", { class: "empty" }, "Loading…"))));
  try {
    const report = await loadReport(run);
    const rows = [...report.results].sort((a, b) => (b.cost_usd || 0) - (a.cost_usd || 0));
    tasks.replaceChildren(
      el("thead", {}, el("tr", {}, ["Task", "Outcome", "Steps", "Cost", "Input", "Output"]
        .map((h, i) => el("th", { class: i >= 2 ? "num" : "" }, h)))),
      el("tbody", {}, rows.map((r) => el("tr", {}, el("td", {}, r.task), el("td", {}, pill(r.outcome)),
        el("td", { class: "num" }, fmtNum(r.steps)), el("td", { class: "num" }, fmtUsd(r.cost_usd)),
        el("td", { class: "num" }, fmtTokens(r.input_tokens)), el("td", { class: "num" }, fmtTokens(r.output_tokens))))));
  } catch (err) {
    tasks.replaceChildren(el("tbody", {}, el("tr", {}, el("td", { class: "empty" }, `Could not load the report: ${err.message}`))));
  }
}

// ---------- shell ----------

function render() {
  for (const [id, value] of [["f-mode", state.mode], ["f-suite", state.suite]]) {
    for (const b of document.getElementById(id).querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.v === value));
  }
  document.getElementById("f-flagged").checked = state.showFlagged;
  for (const b of document.querySelectorAll("nav.tabs button")) b.setAttribute("aria-selected", String(b.dataset.tab === state.tab));
  document.getElementById("tab-evals").hidden = state.tab !== "evals";
  document.getElementById("tab-cost").hidden = state.tab !== "cost";
  if (state.tab === "evals") {
    const runs = filteredRuns();
    renderTiles(runs);
    renderCharts(runs);
    renderRuns(runs);
    renderMatrix(runs);
    const selected = runs.find((r) => r.key === state.selected);
    document.getElementById("detail").hidden = !selected;
  } else {
    renderCost();
    document.getElementById("cost-detail").hidden = !filteredRuns().some((r) => r.key === state.costSelected);
  }
}

const THEME_ICONS = {
  auto: [["M12 3a9 9 0 1 0 0 18a9 9 0 1 0 0-18z"], ["M12 3a9 9 0 0 0 0 18z", true]],
  light: [["M12 8a4 4 0 1 0 0 8a4 4 0 1 0 0-8z M12 2v2 M12 20v2 M4.9 4.9l1.4 1.4 M17.7 17.7l1.4 1.4 M2 12h2 M20 12h2 M4.9 19.1l1.4-1.4 M17.7 6.3l1.4-1.4"]],
  dark: [["M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"]],
};

function themeIcon(theme) {
  const icon = svg("svg", { viewBox: "0 0 24 24", "aria-hidden": "true" });
  for (const [d, filled] of THEME_ICONS[theme]) icon.append(svg("path", filled ? { d, fill: "currentColor" } : { d }));
  return icon;
}

function bindControls() {
  for (const id of ["f-mode", "f-suite"]) {
    document.getElementById(id).addEventListener("click", (event) => {
      const value = event.target.dataset && event.target.dataset.v;
      if (!value) return;
      state[id === "f-mode" ? "mode" : "suite"] = value;
      store(`operon-dash-${id}`, value);
      render();
    });
  }
  document.getElementById("f-flagged").addEventListener("change", (event) => {
    state.showFlagged = event.target.checked;
    render();
  });
  for (const b of document.querySelectorAll("nav.tabs button")) {
    b.addEventListener("click", () => { state.tab = b.dataset.tab; store("operon-dash-tab", state.tab); render(); });
  }
  const themes = ["auto", "light", "dark"];
  const themeBtn = document.getElementById("theme");
  let theme = store("operon-dash-theme") || "auto";
  const applyTheme = () => {
    if (theme === "auto") document.documentElement.removeAttribute("data-theme");
    else document.documentElement.setAttribute("data-theme", theme);
    themeBtn.replaceChildren(themeIcon(theme));
    themeBtn.title = `Theme: ${theme}`;
    themeBtn.setAttribute("aria-label", `Theme: ${theme}`);
  };
  themeBtn.addEventListener("click", () => {
    theme = themes[(themes.indexOf(theme) + 1) % themes.length];
    store("operon-dash-theme", theme);
    applyTheme();
    render();
  });
  applyTheme();
  document.getElementById("detail-close").addEventListener("click", () => {
    state.selected = null;
    render();
  });
  let resizeTimer;
  window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(render, 150); });
}

async function main() {
  state.mode = store("operon-dash-f-mode") || state.mode;
  state.suite = store("operon-dash-f-suite") || state.suite;
  state.tab = store("operon-dash-tab") || state.tab;
  bindControls();
  try {
    const response = await fetch("data/index.json", { cache: "no-cache" });
    if (!response.ok) throw new Error(`data/index.json: ${response.status}`);
    const index = await response.json();
    state.runs = index.runs || [];
    assignSlots(state.runs);
    const generated = index.generated_at ? new Date(index.generated_at) : null;
    document.getElementById("meta").textContent =
      `${state.runs.length} benchmark runs` + (generated ? ` · updated ${timeFmt.format(generated)}` : "");
  } catch (err) {
    document.getElementById("meta").textContent = `Could not load benchmark data: ${err.message}`;
  }
  render();
}

main();
