// Operon eval dashboard: renders /api/runs (one entry per benchmark run) and loads
// /api/report?key=<key> for a run's per-task results; both read the benchmark
// database. No build step; all text from the data goes into the DOM through textContent.
"use strict";

const SVG_NS = "http://www.w3.org/2000/svg";
const BACKEND_NAMES = { claude_computer_use: "Claude", computer_use: "Gemini", json: "Gemini", model: "Model" };
// Colors follow the configuration, never its rank: known ones get fixed slots,
// anything new takes the next free slot in order of first appearance.
const FIXED_SLOTS = { "Claude": 1, "Gemini": 2, "Claude · thin": 3 };
const MAX_SLOTS = 6;
const OUTCOME_LABELS = {
  success: "pass", false_success: "false pass", missed: "missed",
  failure: "fail", timeout: "timeout", error: "error", compromised: "compromised",
};

const OUTCOME_ORDER = ["success", "missed", "false_success", "failure", "timeout", "error", "compromised"];
const OUTCOME_COLORS = {
  success: "var(--good)", missed: "var(--warning)", false_success: "var(--critical)", failure: "var(--serious)",
  timeout: "var(--text-muted)", error: "var(--border-strong)", compromised: "var(--series-5)",
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
  // Backend + harness labels whose runs used more than one model: their labels
  // carry the model, so an Opus and a Sonnet run are two configurations.
  multiModel: new Set(),
  reports: {},
  tasks: null,
  judge: null,
  budget: null,
  registry: [],
  trendMetric: "pass",
  lbCell: null, // the Leaderboard grid cell whose flipped tasks are listed (KAN-60)
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

// Every date on the page is UTC, the same clock as the monthly budget and the benchmark runners.
const dateFmt = new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric", timeZone: "UTC" });
const timeFmt = new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", timeZone: "UTC", timeZoneName: "short" });
const fmtPct = (v) => (v === null || v === undefined ? "–" : `${Math.round(v * 100)}%`);
const fmtSec = (v) => (v === null || v === undefined ? "–" : `${Number(v).toFixed(v >= 100 ? 0 : 1)} s`);
const fmtNum = (v) => (v === null || v === undefined ? "–" : String(v));
function fmtUsd(v) {
  if (v === null || v === undefined) return "–";
  if (v === 0) return "$0";
  if (v < 0.01) return `$${v.toFixed(4)}`;
  if (v < 1) return `$${v.toFixed(3)}`;
  return `$${v.toFixed(2)}`;
}
// Axis ticks: as few digits as the value needs ($0.25, $2.5, $10).
const fmtUsdAxis = (v) => `$${Number(v.toFixed(v < 1 ? 3 : 2))}`;
function fmtTokens(v) {
  if (v === null || v === undefined) return "–";
  if (v >= 1e6) return `${(v / 1e6).toFixed(2)}M`;
  if (v >= 1e3) return `${(v / 1e3).toFixed(1)}k`;
  return String(v);
}

function harnessSuffix(name, harness) {
  if (harness === "thin-own") return `${name} · thin, own done`;
  return (harness || "full") === "thin" ? `${name} · thin` : name;
}

// A registry model's label: its profile, plus "· full" for a full-harness run (KAN-68),
// so thin and full runs of one model are two rows.
function registryLabel(c) {
  return c.harness_profile === "full" ? `${c.model_profile} · full` : c.model_profile;
}

function baseLabel(run) {
  const c = run.config || {};
  const name = BACKEND_NAMES[c.backend] || c.backend || "Unknown backend";
  return harnessSuffix(name, c.harness_profile);
}

function configLabel(run) {
  const c = run.config || {};
  if (c.model_profile) return registryLabel(c);  // a registry model: own "done"
  const base = baseLabel(run);
  // Only where one backend ran several models, so history from before the model
  // was recorded (and single-model configs) keeps its label and colour.
  return c.model && state.multiModel.has(base) ? `${base} · ${c.model}` : base;
}

function noteModels(runs) {
  const models = new Map();
  for (const run of runs) {
    const c = run.config || {};
    if (c.model_profile || !c.model) continue;
    const base = baseLabel(run);
    if (!models.has(base)) models.set(base, new Set());
    models.get(base).add(c.model);
  }
  state.multiModel = new Set([...models].filter(([, set]) => set.size > 1).map(([base]) => base));
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
    // A fixed colour goes to the first label of its backend + harness; a second
    // model of the same configuration gets its own.
    const fixed = FIXED_SLOTS[label] || FIXED_SLOTS[baseLabel(run)];
    if (fixed && !Object.values(state.slots).includes(fixed)) { state.slots[label] = fixed; continue; }
    while (used.has(next) && next < MAX_SLOTS) next++;
    state.slots[label] = next;
    used.add(next);
  }
}

function key(label, cls, color) {
  return el("span", { class: cls || "key", style: `background:${color || seriesColor(label)}` });
}

function configCell(run) {
  return el("span", { class: "config-cell" }, key(configLabel(run)), configLabel(run));
}

// Only GitHub links become hrefs: a stored value is never trusted to be a safe URL.
function githubUrl(url) {
  try {
    const parsed = new URL(url);
    return parsed.protocol === "https:" && parsed.hostname === "github.com" ? parsed.href : null;
  } catch (_) {
    return null;
  }
}

function runLink(run, text) {
  const url = githubUrl(run.config && run.config.github_run_url);
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
    state.reports[run.key] = await getJson(`api/report?key=${encodeURIComponent(run.key)}`);
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
    for (const s of series) legendHost.append(el("span", {}, key(s.label, "line-key", s.color), s.label));
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
  const firstDay = new Date(x0); firstDay.setUTCHours(0, 0, 0, 0);
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
      tip.append(el("div", { class: "t-row" }, key(s.label, "line-key", s.color), el("strong", {}, opts.fmt(p.y)),
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

// ---------- bar chart ----------
// One bar per column in order, equal widths so runs launched together stay readable.
// columns: [{ date, head, segments: [{ name, value, color }], notes: [text], onclick }]
function barChart(host, columns, opts) {
  host.replaceChildren();
  if (!columns.length) {
    host.append(el("div", { class: "empty" }, opts.empty || "No runs for this selection."));
    return;
  }
  const width = Math.max(host.clientWidth, 280);
  const height = 260;
  const m = { l: 48, r: 12, t: 12, b: 28 };
  const totals = columns.map((c) => c.segments.reduce((sum, g) => sum + g.value, 0));
  const yMax = opts.yMax || niceMax(Math.max(...totals, (opts.refLine && opts.refLine.value) || 0) * 1.1);
  const band = (width - m.l - m.r) / columns.length;
  const barW = Math.max(2, Math.min(36, band * 0.7));
  const cx = (i) => m.l + band * (i + 0.5);
  const sy = (y) => m.t + (1 - y / yMax) * (height - m.t - m.b);

  const root = svg("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": opts.title || "chart" });
  const axis = svg("g", { class: "axis" });
  for (let i = 0; i <= 4; i++) {
    const v = (yMax * i) / 4;
    axis.append(svg("line", { class: i === 0 ? "baseline" : "gridline", x1: m.l, x2: width - m.r, y1: sy(v), y2: sy(v) }));
    const t = svg("text", { x: m.l - 8, y: sy(v) + 4, "text-anchor": "end" });
    t.textContent = opts.fmtAxis(v);
    axis.append(t);
  }
  // A date under the first column of each day, skipped when it would crowd the previous one.
  let lastX = -Infinity;
  let lastDay = "";
  columns.forEach((c, i) => {
    const day = opts.label ? opts.label(c) : dateFmt.format(c.date);
    if (day === lastDay) return;
    lastDay = day;
    if (cx(i) - lastX < 56) return;
    lastX = cx(i);
    const t = svg("text", { x: cx(i), y: height - 8, "text-anchor": "middle" });
    t.textContent = day;
    axis.append(t);
  });
  root.append(axis);

  const bars = columns.map((c, i) => {
    const g = svg("g", { class: "bar" });
    let base = 0;
    for (const seg of c.segments) {
      if (!(seg.value > 0)) continue;
      const top = sy(base + seg.value);
      g.append(svg("rect", {
        x: cx(i) - barW / 2, y: top, width: barW, height: Math.max(1, sy(base) - top), fill: seg.color,
        ...(seg.faint ? { "fill-opacity": 0.3, stroke: seg.color, "stroke-dasharray": "3 2" } : {}),
      }));
      base += seg.value;
    }
    root.append(g);
    return g;
  });
  if (opts.refLine) {
    // A horizontal reference, such as the monthly budget.
    const y = sy(opts.refLine.value);
    root.append(svg("line", { class: "refline", x1: m.l, x2: width - m.r, y1: y, y2: y }));
    const t = svg("text", { class: "reflabel", x: width - m.r, y: y - 6, "text-anchor": "end" });
    t.textContent = opts.refLine.label;
    root.append(t);
  }
  const hit = svg("rect", { x: m.l, y: 0, width: width - m.l - m.r, height, fill: "transparent", tabindex: 0 });
  root.append(hit);
  host.append(root);
  const tip = el("div", { class: "tooltip" });
  host.append(tip);

  let focus = columns.length - 1;
  function show(i) {
    const c = columns[i];
    bars.forEach((b, j) => b.classList.toggle("dim", j !== i));
    tip.replaceChildren(el("div", { class: "t-head" }, c.head),
      c.segments.filter((g) => g.value > 0).map((g) => el("div", { class: "t-row" },
        el("span", { class: "key", style: `background:${g.color}` }), el("strong", {}, opts.fmt(g.value)), el("span", {}, g.name))),
      (c.notes || []).map((n) => el("div", { class: "t-row" }, el("span", {}, n))));
    tip.style.display = "block";
    const scale = root.getBoundingClientRect().width / width;
    const left = cx(i) * scale + 12;
    const maxLeft = host.clientWidth - tip.offsetWidth - 4;
    tip.style.left = `${Math.max(0, left > maxLeft ? cx(i) * scale - tip.offsetWidth - 12 : left)}px`;
    tip.style.top = `${Math.max(0, sy(totals[i]) * scale - tip.offsetHeight - 10)}px`;
  }
  function hide() { tip.style.display = "none"; bars.forEach((b) => b.classList.remove("dim")); }
  const indexAt = (event) => {
    const box = root.getBoundingClientRect();
    const x = ((event.clientX - box.left) / box.width) * width;
    return Math.max(0, Math.min(columns.length - 1, Math.floor((x - m.l) / band)));
  };
  const open = (i) => columns[i].onclick && columns[i].onclick();
  hit.addEventListener("pointermove", (event) => { focus = indexAt(event); show(focus); });
  hit.addEventListener("pointerleave", hide);
  hit.addEventListener("click", (event) => open(indexAt(event)));
  hit.addEventListener("focus", () => show(focus));
  hit.addEventListener("blur", hide);
  hit.addEventListener("keydown", (event) => {
    if (event.key === "ArrowLeft") focus = Math.max(0, focus - 1);
    else if (event.key === "ArrowRight") focus = Math.min(columns.length - 1, focus + 1);
    else if (event.key === "Enter") return open(focus);
    else return;
    event.preventDefault();
    show(focus);
  });
}

function legendItems(host, items) {
  host.replaceChildren(...items.map(({ name, color }) =>
    el("span", {}, el("span", { class: "key", style: `background:${color}` }), name)));
}

// ---------- evals tab ----------

// ---------- latest-run briefing ----------
// The headline, "what changed" list and number tiles are built here from the runs and
// task results already loaded: a fixed sentence template, no model call, so they
// refresh on their own after every imported run.

// This configuration's run just before this one, flagged runs left out.
function previousRun(run, runs) {
  const label = configLabel(run);
  const earlier = runs.filter((r) => configLabel(r) === label && r.started_at < run.started_at && !(r.config && r.config.flagged));
  return earlier.length ? earlier.reduce((a, b) => (b.started_at > a.started_at ? b : a)) : null;
}

// One result per task. A repeated (--repeat) run counts as a pass only when every attempt
// passed; otherwise its earliest failing attempt stands for the task.
function taskResults(tasks) {
  const out = new Map();
  for (const t of tasks) {
    const have = out.get(t.task);
    const fails = t.outcome !== "success", haveFails = have && have.outcome !== "success";
    if (!have || (fails && !haveFails) || (fails === haveFails && (t.attempt || 1) < (have.attempt || 1))) out.set(t.task, t);
  }
  return out;
}

// Why a task did not pass: the judge's or grader's reason when the run has one, otherwise
// what the run itself recorded (how it ended and why the agent stopped).
function likelyCause(t) {
  if (!t) return "";
  const reason = (t.judge_reason || "").trim();
  if (reason) return reason;
  const how = { false_success: "Said done, but the goal wasn't reached", missed: "Goal reached, but never said done",
    failure: "Failed", timeout: "Ran out of time", error: "The run crashed", compromised: "Followed an injected instruction" }[t.outcome]
    || OUTCOME_LABELS[t.outcome] || t.outcome || "Unknown";
  const stop = (t.stop_reason || "").trim();
  return stop && stop !== "-" ? `${how} (stopped: ${stop.replace(/_/g, " ")})` : how;
}

// Tasks that started passing, started failing, or still fail, between two runs' results.
function runChanges(prevTasks, curTasks) {
  const prev = taskResults(prevTasks);
  const cur = taskResults(curTasks);
  const fixed = [], broke = [], still = [];
  for (const [name, t] of cur) {
    const p = prev.get(name);
    const pass = t.outcome === "success";
    if (!p) { if (!pass) broke.push(t); continue; }
    const wasPass = p.outcome === "success";
    if (pass && !wasPass) fixed.push(t);
    else if (!pass && wasPass) broke.push(t);
    else if (!pass) still.push(t);
  }
  return { fixed, broke, still };
}

// "passed 42 of 44 tasks (95%), the same as last run, in 27.1 s per task."
function briefHeadline(run, prev) {
  const s = run.summary;
  let change = "its first run";
  if (prev) {
    const pts = Math.round((s.success_rate - prev.summary.success_rate) * 100);
    change = pts > 0 ? `up ${pts} point${pts === 1 ? "" : "s"} from last run` : pts < 0 ? `down ${-pts} point${pts === -1 ? "" : "s"} from last run` : "the same as last run";
  }
  return { lead: `${configLabel(run)} passed `, count: `${s.success} of ${s.tasks}`,
    rest: ` tasks (${fmtPct(s.success_rate)}), ${change}, in ${fmtSec(s.mean_seconds)} per task.` };
}

function change(kind, ...parts) {
  return el("div", { class: "chg" }, el("span", { class: `ic ${kind}`, "aria-hidden": "true" }, { up: "↑", down: "↓", warn: "!" }[kind]),
    el("div", {}, ...parts));
}

function taskList(tasks, max = 3) {
  const names = tasks.slice(0, max).map((t) => t.task).join(", ");
  return tasks.length > max ? `${names} and ${tasks.length - max} more` : names;
}

function renderBrief(runs) {
  const host = document.getElementById("brief");
  const latest = latestByConfig(runs);
  if (!latest.length) {
    host.replaceChildren(el("div", { class: "eyebrow" }, "No runs yet"),
      el("h1", { class: "headline" }, "No runs for this environment and suite yet."),
      el("p", { class: "lede" }, "Results appear here after the next benchmark run, started with a /bench comment on a pull request."));
    return;
  }
  const [run, ...others] = latest;
  const prev = previousRun(run, runs);
  const h = briefHeadline(run, prev);
  const s = run.summary;
  const otherText = others.map((r) => {
    const o = r.summary;
    return `${configLabel(r)} passed ${o.success} of ${o.tasks} (${fmtPct(o.success_rate)})` +
      (o.false_success ? ` with ${o.false_success} false pass${o.false_success === 1 ? "" : "es"}` : "") + ".";
  }).join(" ");

  const items = [];
  if (!prev) {
    items.push(change("warn", "First run of this configuration, so there is nothing to compare yet."));
  } else if (state.tasksFailed) {
    items.push(change("warn", "Task results could not be loaded, so the task-by-task comparison is unavailable."));
  } else if (state.tasks !== null) {
    const prevTasks = runTasks(prev), curTasks = runTasks(run);
    const { fixed, broke, still } = runChanges(prevTasks, curTasks);
    if (fixed.length) items.push(change("up", el("strong", {}, taskList(fixed)), fixed.length === 1 ? " passes again" : " pass again"));
    for (const t of broke.slice(0, 3)) items.push(change("down", el("strong", {}, t.task), " started failing: ", el("span", { class: "muted" }, likelyCause(t))));
    if (broke.length > 3) items.push(change("down", `${broke.length - 3} more tasks started failing`));
    for (const t of still.slice(0, 3)) items.push(change("down", el("strong", {}, t.task), " still fails: ", el("span", { class: "muted" }, likelyCause(t))));
    if (still.length > 3) items.push(change("down", `${still.length - 3} more tasks still fail`));
    const dSec = (s.mean_seconds || 0) - (prev.summary.mean_seconds || 0);
    if (Math.abs(dSec) >= 0.5) {
      items.push(change(dSec < 0 ? "up" : "down", `${dSec < 0 ? "Faster" : "Slower"} by `, el("strong", {}, `${Math.abs(dSec).toFixed(1)} s per task`),
        ` (${fmtSec(prev.summary.mean_seconds)} → ${fmtSec(s.mean_seconds)})`));
    }
    if (!prevTasks.length || !curTasks.length) {
      items.push(change("warn", "Task results are missing for one of the two runs, so tasks can't be compared."));
    } else if (!fixed.length && !broke.length && !still.length && Math.abs(dSec) < 0.5) {
      items.push(change("up", "Every task passed, as in the last run."));
    }
  }
  for (const r of latest) {
    if (r.summary.false_success) items.push(change("warn", `${configLabel(r)} had `, el("strong", {}, `${r.summary.false_success} false pass${r.summary.false_success === 1 ? "" : "es"}`)));
    if (r.summary.compromised) items.push(change("down", `${configLabel(r)} was `, el("strong", {}, `compromised ${r.summary.compromised} time${r.summary.compromised === 1 ? "" : "s"}`)));
  }

  host.replaceChildren(
    el("div", { class: "hero-main" },
      el("div", { class: "eyebrow" }, `Latest run · ${timeFmt.format(parseStamp(run.started_at))}`, " ", runLink(run, "View run ↗")),
      el("h1", { class: "headline" }, h.lead, el("em", {}, h.count), h.rest),
      otherText ? el("p", { class: "lede" }, otherText) : null,
      latest.length > 1 ? el("div", { class: "legend" }, latest.map((r) => el("span", {}, key(configLabel(r)), configLabel(r)))) : null),
    el("div", { class: "changes" },
      el("div", { class: "changes-title" }, prev ? "What changed since the last run" : "What changed"),
      state.tasks === null ? el("div", { class: "muted" }, "Loading…") : items));
}

function sparkline(values, color) {
  const vals = values.filter((v) => v !== null && v !== undefined && !Number.isNaN(v));
  if (vals.length < 2) return null;
  const w = 120, h = 32;
  const lo = Math.min(...vals), hi = Math.max(...vals), span = hi - lo || 1;
  const d = vals.map((v, i) => `${i ? "L" : "M"}${((i * w) / (vals.length - 1)).toFixed(1)},${(h - 3 - ((v - lo) / span) * (h - 6)).toFixed(1)}`).join("");
  const root = svg("svg", { class: "spark", viewBox: `0 0 ${w} ${h}`, preserveAspectRatio: "none", "aria-hidden": "true" });
  root.append(svg("path", { d, fill: "none", stroke: color, "stroke-width": 2, "stroke-linejoin": "round", "vector-effect": "non-scaling-stroke" }));
  return root;
}

// One tile per headline number for the lead configuration, with its trend and the
// other configurations' latest values underneath.
function renderTiles(runs) {
  const host = document.getElementById("tiles");
  const latest = latestByConfig(runs);
  if (!latest.length) { host.replaceChildren(); return; }
  const [run, ...others] = latest;
  const prev = previousRun(run, runs);
  const history = runs.filter((r) => configLabel(r) === configLabel(run) && !(r.config && r.config.flagged)).slice(-12);
  const perTask = (r) => (hasCost(r) && r.summary.tasks ? listedSpend(r) / r.summary.tasks : null);
  const metrics = [
    { label: "Pass rate", get: (r) => r.summary.success_rate, fmt: fmtPct, of: (r) => `${r.summary.success}/${r.summary.tasks}`, better: 1,
      diff: (a, b) => { const p = Math.round((a - b) * 100); return p ? `${Math.abs(p)} pts` : null; } },
    { label: "Seconds per task", get: (r) => r.summary.mean_seconds, fmt: fmtSec, better: -1,
      diff: (a, b) => (Math.abs(a - b) >= 0.1 ? `${Math.abs(a - b).toFixed(1)} s` : null) },
    { label: "False passes", get: (r) => r.summary.false_success, fmt: fmtNum, better: -1,
      diff: (a, b) => (a !== b ? String(Math.abs(a - b)) : null) },
    { label: "Cost per task", get: perTask, fmt: fmtUsd, better: -1,
      diff: (a, b) => (b && Math.abs(a / b - 1) >= 0.01 ? `${Math.round(Math.abs(a / b - 1) * 100)}%` : null) },
  ];
  if (latest.some((r) => r.summary.compromised)) {
    metrics.push({ label: "Compromised", get: (r) => r.summary.compromised || 0, fmt: fmtNum, better: -1, diff: (a, b) => (a !== b ? String(Math.abs(a - b)) : null) });
  }
  const color = seriesColor(configLabel(run));
  host.replaceChildren(...metrics.map((m) => {
    const value = m.get(run);
    const before = prev ? m.get(prev) : null;
    let delta = el("span", { class: "delta" }, prev ? "no change" : "first run");
    if (value !== null && value !== undefined && before !== null && before !== undefined) {
      const text = m.diff(value, before);
      if (text) {
        const good = (value - before) * m.better > 0;
        delta = el("span", { class: `delta ${good ? "up" : "down"}`, title: `vs ${timeFmt.format(parseStamp(prev.started_at))}` },
          `${value > before ? "▲" : "▼"} ${text}`);
      }
    }
    return el("div", { class: "kpi" },
      el("div", { class: "kpi-label" }, m.label),
      el("div", { class: "kpi-value" }, el("strong", {}, m.fmt(value)), m.of ? el("span", {}, m.of(run)) : null),
      el("div", { class: "kpi-delta" }, delta),
      sparkline(history.map(m.get), color),
      others.length ? el("div", { class: "kpi-others" }, others.map((r) => el("span", {}, key(configLabel(r)), `${configLabel(r)} ${m.fmt(m.get(r))}`))) : null);
  }));
}

const TREND_METRICS = {
  pass: { get: (r) => r.summary.success_rate, opts: { yMax: 1, fmt: (v) => fmtPct(v) }, sub: "Pass rate of every run, by configuration. Higher is better." },
  speed: { get: (r) => r.summary.mean_seconds, opts: { fmt: (v, axis) => (axis ? `${Math.round(v)} s` : fmtSec(v)) }, sub: "Mean seconds per task of every run. Lower is better." },
  steps: { get: (r) => r.summary.mean_steps, opts: { fmt: (v, axis) => (axis ? `${Math.round(v * 10) / 10}` : `${Number(v).toFixed(1)} steps`) }, sub: "Mean steps per task of every run. Fewer is usually better." },
  cost: { get: (r) => (hasCost(r) && r.summary.tasks ? listedSpend(r) / r.summary.tasks : null), opts: { fmt: (v, axis) => (axis ? fmtUsdAxis(v) : fmtUsd(v)) }, sub: "Agent, judge and GPU spend per task of every run. Lower is better." },
};

function renderCharts(runs) {
  const m = TREND_METRICS[state.trendMetric] || TREND_METRICS.pass;
  for (const b of document.getElementById("trend-metric").querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.v === state.trendMetric));
  document.getElementById("trend-sub").textContent = `${m.sub} Hover a point for the run.`;
  lineChart(document.getElementById("ch-trend"), document.getElementById("lg-trend"), seriesFor(runs, m.get),
    { ...m.opts, detail: (r) => `${r.summary.success}/${r.summary.tasks} · ${r.summary.mean_steps} steps`, title: "Trend by run" });
}

// Each run's outcome mix as a share of its tasks, so suites of any size compare.
function renderOutcomes(runs) {
  const present = OUTCOME_ORDER.filter((o) => runs.some((r) => r.summary[o]));
  legendItems(document.getElementById("lg-outcomes"), present.map((o) => ({ name: OUTCOME_LABELS[o], color: OUTCOME_COLORS[o] })));
  barChart(document.getElementById("ch-outcomes"), runs.map((run) => {
    const s = run.summary;
    return {
      date: parseStamp(run.started_at),
      head: `${timeFmt.format(parseStamp(run.started_at))} · ${configLabel(run)}`,
      segments: present.map((o) => ({ name: `${OUTCOME_LABELS[o]} (${s[o] || 0})`, value: s.tasks ? (s[o] || 0) / s.tasks : 0, color: OUTCOME_COLORS[o] })),
      notes: [`${s.success}/${s.tasks} passed`],
      onclick: () => selectRun(run),
    };
  }), { yMax: 1, fmt: (v) => fmtPct(v), fmtAxis: (v) => fmtPct(v), title: "Outcomes by run" });
}

function runTasks(run) {
  return (state.tasks || []).filter((t) => t.run_key === run.key);
}

function renderCategories(runs) {
  const host = document.getElementById("category-bars");
  const latest = latestByConfig(runs);
  legendItems(document.getElementById("lg-category"), latest.length > 1
    ? latest.map((r) => ({ name: configLabel(r), color: seriesColor(configLabel(r)) })) : []);
  if (state.tasks === null) { host.replaceChildren(el("div", { class: "empty" }, "Loading…")); return; }
  const cats = new Map();
  latest.forEach((run, col) => {
    for (const t of runTasks(run)) {
      const name = t.category || "other";
      if (!cats.has(name)) cats.set(name, latest.map(() => ({ passed: 0, tasks: 0 })));
      const c = cats.get(name)[col];
      c.tasks += 1;
      c.passed += t.outcome === "success" ? 1 : 0;
    }
  });
  if (!cats.size) { host.replaceChildren(el("div", { class: "empty" }, "No runs for this selection.")); return; }
  // Weakest category first: that is where the next fix pays off.
  const worst = (cells) => Math.min(...cells.filter((c) => c.tasks).map((c) => c.passed / c.tasks));
  const rows = [...cats.entries()].sort((a, b) => worst(a[1]) - worst(b[1]));
  host.replaceChildren(el("div", { class: "hbars" }, rows.map(([name, cells]) => el("div", { class: "hbar-row" },
    el("div", { class: "hbar-label" }, name),
    el("div", { class: "hbar-cells" }, cells.map((c, col) => c.tasks ? el("div", { class: "hbar" },
      el("div", { class: "meter", style: `--tile-color:${seriesColor(configLabel(latest[col]))}` },
        el("span", { style: `width:${(c.passed / c.tasks) * 100}%` })),
      el("span", { class: "hbar-value" }, fmtPct(c.passed / c.tasks)), el("span", { class: "hbar-of" }, `${c.passed}/${c.tasks}`)) : null))))));
}

// Every task over the last runs: a row of outcome marks, flaky ones first.
function renderHistory(runs) {
  const table = document.getElementById("history");
  if (state.tasks === null) { table.replaceChildren(el("tbody", {}, el("tr", {}, el("td", { class: "empty" }, "Loading…")))); return; }
  const recent = runs.slice(-20);
  const index = new Map(recent.map((r, i) => [r.key, i]));
  const rows = new Map();
  for (const t of state.tasks) {
    if (!index.has(t.run_key)) continue;
    if (!rows.has(t.task)) rows.set(t.task, { task: t.task, category: t.category, cells: recent.map(() => null) });
    rows.get(t.task).cells[index.get(t.run_key)] = t;
  }
  const scored = [...rows.values()].map((r) => {
    const seen = r.cells.filter(Boolean);
    const passed = seen.filter((t) => t.outcome === "success").length;
    return { ...r, seen: seen.length, passed, flaky: passed > 0 && passed < seen.length, never: passed === 0 };
  });
  // Flaky first, then never passing, then always passing.
  const rank = (r) => (r.flaky ? 0 : r.never ? 1 : 2);
  scored.sort((a, b) => rank(a) - rank(b) || a.passed / a.seen - b.passed / b.seen || a.task.localeCompare(b.task));
  const steady = scored.filter((r) => rank(r) === 2).length;
  table.replaceChildren(
    el("thead", {}, el("tr", {}, el("th", {}, "Task"), el("th", {}, `Last ${recent.length} runs`), el("th", { class: "num" }, "Passed"), el("th", {}, ""), el("th", {}, "Likely cause"))),
    el("tbody", {}, scored.filter((r) => state.historyAll || rank(r) < 2 || steady === scored.length).map((r) => el("tr", {},
      el("td", {}, r.task, el("div", { class: "muted-line" }, r.category || "")),
      el("td", {}, el("div", { class: "marks" }, r.cells.map((t, i) => el("span", {
        class: `mark ${t ? t.outcome : "none"}`,
        title: t ? `${timeFmt.format(parseStamp(recent[i].started_at))} · ${configLabel(recent[i])} · ${OUTCOME_LABELS[t.outcome] || t.outcome}${t.stop_reason ? ` · ${t.stop_reason}` : ""}` : "not run",
      })))),
      el("td", { class: "num" }, `${r.passed}/${r.seen}`),
      el("td", {}, r.flaky ? el("span", { class: "pill missed" }, "flaky") : r.never ? el("span", { class: "pill failure" }, "never passes") : null),
      el("td", { class: "wrap-text cause" }, rank(r) < 2 ? likelyCause([...r.cells].reverse().find((t) => t && t.outcome !== "success")) : "")))));
  let more = document.getElementById("history-more");
  if (!more) {
    more = el("button", { class: "ghost-btn", id: "history-more", type: "button" });
    table.parentElement.after(more);
  }
  more.hidden = steady === 0 || steady === scored.length;
  more.textContent = state.historyAll ? "Hide tasks that always pass" : `${steady} other task${steady === 1 ? "" : "s"} passed every run. Show them`;
  more.onclick = () => { state.historyAll = !state.historyAll; renderHistory(filteredRuns()); };
}

function renderJudge() {
  const host = document.getElementById("judge");
  if (state.judge === null) { host.replaceChildren(el("div", { class: "empty" }, "Loading…")); return; }
  const checks = state.judge;
  if (!checks.length) {
    host.replaceChildren(el("p", { class: "note" }, "No judge check saved yet. Label final screens and run ",
      el("code", {}, "python -m evals.judge_check run labels.jsonl --save"), " to record one."));
    return;
  }
  const c = checks[checks.length - 1];
  const prev = checks.length > 1 ? checks[checks.length - 2] : null;
  const delta = (key) => {
    if (!prev || prev[key] === null || c[key] === null) return null;
    const pts = Math.round((c[key] - prev[key]) * 100);
    return pts ? el("span", { class: `delta ${pts > 0 ? "up" : "down"}` }, `${pts > 0 ? "▲" : "▼"} ${Math.abs(pts)}`) : null;
  };
  const metric = (label, key, hint) => el("div", { class: "jmetric" },
    el("div", { class: "label" }, label), el("div", { class: "value" }, fmtPct(c[key]), delta(key)), el("div", { class: "hint" }, hint));
  const wrong = Array.isArray(c.disagreements) ? c.disagreements : [];
  host.replaceChildren(
    el("div", { class: "jgrid" },
      metric("Accuracy", "accuracy", "agrees with the label"),
      metric("Precision", "precision", "its passes that are real"),
      metric("Recall", "recall", "real passes it sees"),
      metric("Self-agreement", "agreement", `same answer over ${c.repeats || 1} tries`)),
    el("p", { class: "note" }, `${c.labels} hand labels · ${c.false_passes} false pass${c.false_passes === 1 ? "" : "es"} · ` +
      `${c.false_fails} missed pass${c.false_fails === 1 ? "" : "es"} · ${c.judge_model || "judge"} · checked ${dateFmt.format(new Date(c.checked_at))}`),
    wrong.length ? el("table", {}, el("thead", {}, el("tr", {}, ["Disagreement", "Label", "Judge"].map((h) => el("th", {}, h)))),
      el("tbody", {}, wrong.map((d) => el("tr", {}, el("td", {}, d.task), el("td", {}, d.label ? "done" : "not done"),
        el("td", {}, d.judge === null || d.judge === undefined ? "no answer" : d.judge ? "done" : "not done"))))) : "");
}

// One pill per attempt; a repeated task also shows how many attempts passed (2/3).
function attemptCell(attempts) {
  const pills = attempts.map((r) => pill(r.outcome));
  if (attempts.length < 2) return pills;
  const passed = attempts.filter((r) => r.outcome === "success").length;
  const flaky = passed > 0 && passed < attempts.length;
  return [...pills, el("span", { class: "sub", title: flaky ? "flaky: passed some attempts" : "" },
    ` ${passed}/${attempts.length}${flaky ? " flaky" : ""}`)];
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
      // A --repeat run reports each task several times: keep every attempt.
      if (!map.has(r.task)) map.set(r.task, []);
      map.get(r.task).push(r);
    }
    return map;
  });
  table.append(el("thead", {}, el("tr", {}, el("th", {}, "Task"), el("th", {}, "Category"),
    columns.map((run) => el("th", { class: "col" }, configCell(run))))));
  const body = el("tbody");
  const rows = tasks.map((t) => {
    const results = byColumn.map((m) => m.get(t.task) || null);
    const outcomes = new Set(results.filter(Boolean).flat().map((r) => r.outcome));
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
      results.map((attempts) => el("td", {
        class: "cell",
        title: attempts
          ? attempts.map((r, i) => `${attempts.length > 1 ? `#${i + 1}: ` : ""}${r.outcome}, ${r.steps} steps, ${r.seconds} s — ${r.judge_reason || ""}`).join("\n")
          : "not run",
      }, attempts ? attemptCell(attempts) : "–"))));
  }
  table.append(body);
}

function renderRuns(runs) {
  const table = document.getElementById("runs");
  table.replaceChildren(el("thead", {}, el("tr", {},
    ["When", "Configuration", "Model", "Passed", "Rate", "False pass", "s / task", "s / step", "Steps", "$ / task", "Commit", "Run"]
      .map((h, i) => el("th", { class: i >= 3 && i <= 9 ? "num" : "" }, h)))));
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
      el("td", { class: "num", title: "Agent, judge and GPU spend per task" }, hasCost(run) && s.tasks ? fmtUsd(listedSpend(run) / s.tasks) : "–"),
      el("td", {}, commitLink(run)),
      el("td", { onclick: (e) => e.stopPropagation() }, runLink(run))));
  }
  table.append(body);
}

async function selectRun(run) {
  state.selected = run.key;
  const token = (state.detailToken = (state.detailToken || 0) + 1);
  renderRuns(filteredRuns());
  const card = document.getElementById("detail");
  card.hidden = false;
  document.getElementById("detail-title").textContent = `${configLabel(run)} · ${timeFmt.format(parseStamp(run.started_at))}`;
  const s = run.summary;
  document.getElementById("detail-sub").textContent =
    `${s.success}/${s.tasks} passed · ${s.false_success} false pass · ${s.missed} missed · ${s.failure} failed · ` +
    `${s.timeout} timeout · ${s.error} error` + (s.compromised ? ` · ${s.compromised} compromised` : "") +
    ` · judge ${run.judge_model || "–"}`;
  const table = document.getElementById("detail-table");
  table.replaceChildren(el("tbody", {}, el("tr", {}, el("td", { class: "empty" }, "Loading…"))));
  try {
    const report = await loadReport(run);
    if (token !== state.detailToken) return;  // another run was picked meanwhile
    const order = { compromised: -1, false_success: 0, failure: 1, missed: 2, timeout: 3, error: 4, success: 5 };
    const rows = [...report.results].sort((a, b) => (order[a.outcome] ?? 9) - (order[b.outcome] ?? 9));
    const withScreens = rows.some((r) => safeUrl(r.screenshot_url));
    const headers = ["Task", "Category", "Outcome", "Steps", "Seconds", "Stop reason", "Judge", ...(withScreens ? ["Screen"] : [])];
    table.replaceChildren(
      el("thead", {}, el("tr", {}, headers.map((h, i) => el("th", { class: i === 3 || i === 4 ? "num" : "" }, h)))),
      el("tbody", {}, rows.flatMap((r) => taskRows(r, withScreens, headers.length))));
  } catch (err) {
    if (token !== state.detailToken) return;
    table.replaceChildren(el("tbody", {}, el("tr", {}, el("td", { class: "empty" }, `Could not load the report: ${err.message}`))));
  }
  card.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// Screenshots and traces live in a private Vercel Blob store; /api/blob reads them
// with the store's token, so only links into that store are shown.
function safeUrl(url) {
  try {
    const parsed = new URL(url);
    if (parsed.protocol !== "https:" || !parsed.hostname.endsWith(".private.blob.vercel-storage.com")) return null;
    return `/api/blob?url=${encodeURIComponent(parsed.href)}`;
  } catch (_) {
    return null;
  }
}

function thumb(url, label) {
  return el("a", { href: url, target: "_blank", rel: "noopener", class: "thumb", title: label },
    el("img", { src: url, alt: label, loading: "lazy" }));
}

// A task's row, plus a hidden row with its step screens and trace when it has them.
function taskRows(r, withScreens, columns) {
  const steps = (Array.isArray(r.step_screenshots) ? r.step_screenshots : [])
    .map((s) => ({ step: s.step, url: safeUrl(s.url) })).filter((s) => s.url);
  const trace = safeUrl(r.trace_url);
  const screen = safeUrl(r.screenshot_url);
  const hasTrace = steps.length > 0 || trace;
  let open = false;
  const extra = el("tr", { class: "trace-row", hidden: true },
    el("td", { colspan: columns },
      el("div", { class: "filmstrip" },
        steps.map((s) => el("figure", {}, thumb(s.url, `${r.task}, step ${s.step}`), el("figcaption", {}, `Step ${s.step}`))),
        screen ? el("figure", {}, thumb(screen, `${r.task}, final screen`), el("figcaption", {}, "Final")) : null),
      trace ? el("a", { href: trace, class: "trace-link" }, "Download the full trace (.zip)") : null));
  const row = el("tr", {
    class: hasTrace ? "clickable" : null,
    title: hasTrace ? "Show the steps" : null,
    onclick: hasTrace ? () => { open = !open; extra.hidden = !open; row.classList.toggle("selected", open); } : null,
  },
    el("td", {}, r.task), el("td", {}, r.category), el("td", {}, pill(r.outcome)),
    el("td", { class: "num" }, fmtNum(r.steps)), el("td", { class: "num" }, fmtNum(r.seconds)),
    el("td", {}, r.stop_reason || "–"), el("td", { class: "wrap-text" }, r.judge_reason || ""),
    withScreens ? el("td", { onclick: (e) => e.stopPropagation() }, screen ? thumb(screen, `${r.task}, final screen`) : "–") : null);
  return hasTrace ? [row, extra] : [row];
}

// ---------- cost tab ----------
// Spend is counted across every environment and suite, flagged runs included:
// a broken run still cost money. The filters in the app bar don't apply here.

const SUITE_NAMES = { core: "core", full: "full suite", smoke: "smoke", safety: "safety", apps: "apps", hard: "hard", long: "long" };
const MODE_NAMES = { browser: "Browser", desktop: "Desktop", grounding: "Grounding" };
// Fixed colors per environment and suite, so a suite keeps its color as runs come in.
// Any other suite gets the grey after the last slot (--series-12 in style.css).
const SUITE_SLOTS = ["browser|full", "browser|smoke", "browser|safety", "browser|apps", "desktop|smoke", "desktop|full",
  "browser|hard", "browser|long", "grounding|grounding", "browser|core", "desktop|core"];
// Runs that are not task benchmarks (ScreenSpot grounding): they cost money, so the Cost
// tab counts them, but the Evals tab and the Leaderboard leave them out.
const COST_ONLY_MODES = new Set(["grounding"]);

function hasCost(run) {
  return run.summary && run.summary.cost_usd !== undefined && run.summary.cost_usd !== null;
}

// What a run spent on its listed tasks: the agent's model calls, the judge, and a
// self-hosted model's GPU time. Per-task figures divide this by summary.tasks.
function listedSpend(run) {
  return (run.summary.cost_usd || 0) + (run.summary.judge_cost_usd || 0) + (run.summary.gpu_cost_usd || 0);
}

// Everything a run spent: its listed tasks plus a season run's held-out tasks
// (summary.held_out_cost_usd, one total with no task rows). Run and suite totals use this.
function runSpend(run) {
  return listedSpend(run) + (run.summary.held_out_cost_usd || 0);
}

function suiteId(run) {
  return `${run.mode}|${(run.suite || "").replace(/^desktop-/, "")}`;
}

function suiteLabel(id) {
  const [mode, suite] = id.split("|");
  if (mode === suite) return MODE_NAMES[mode] || mode;
  return `${MODE_NAMES[mode] || mode} · ${SUITE_NAMES[suite] || suite}`;
}

function suiteColor(id) {
  const slot = SUITE_SLOTS.indexOf(id);
  return `var(--series-${slot >= 0 ? slot + 1 : SUITE_SLOTS.length + 1})`;
}

function costRuns() {
  return state.runs.filter(hasCost).sort((a, b) => parseStamp(a.started_at) - parseStamp(b.started_at));
}

function stat(label, value, hint) {
  return el("div", { class: "stat" }, el("div", { class: "label" }, label), el("div", { class: "value" }, value),
    hint ? el("div", { class: "hint" }, hint) : null);
}

// ---------- monthly spend and budget burn ----------
// Months are calendar months in UTC, like the provider bills, whatever the viewer's zone.

function monthKey(date) {
  return `${date.getUTCFullYear()}-${String(date.getUTCMonth() + 1).padStart(2, "0")}`;
}

const monthFmt = new Intl.DateTimeFormat(undefined, { month: "short", year: "numeric", timeZone: "UTC" });
const monthLongFmt = new Intl.DateTimeFormat(undefined, { month: "long", timeZone: "UTC" });

function monthStart(key) {
  const [y, m] = key.split("-").map(Number);
  return new Date(Date.UTC(y, m - 1, 1));
}

// Judge checks (evals.judge_check) cost money outside any run; they count toward spend.
function judgeCheckSpend() {
  return (state.judge || []).filter((c) => c.cost_usd > 0).map((c) => ({ at: new Date(c.checked_at), judge: c.cost_usd }));
}

// One entry per spend event: each run with cost, and each judge check.
function spendEntries(runs) {
  return [
    ...runs.map((run) => ({
      at: parseStamp(run.started_at),
      agent: (run.summary.cost_usd || 0) + (run.summary.held_out_cost_usd || 0),
      judge: run.summary.judge_cost_usd || 0,
      gpu: run.summary.gpu_cost_usd || 0,
      run,
    })),
    ...judgeCheckSpend().map((c) => ({ at: c.at, agent: 0, judge: c.judge, gpu: 0, run: null })),
  ];
}

function entrySpend(e) {
  return e.agent + e.judge + e.gpu;
}

function monthTotals(entries, now = new Date()) {
  const months = new Map();
  for (const e of entries) {
    const key = monthKey(e.at);
    const m = months.get(key) || { key, agent: 0, judge: 0, gpu: 0, runs: 0, checks: 0 };
    m.agent += e.agent;
    m.judge += e.judge;
    m.gpu += e.gpu;
    if (e.run) m.runs += 1; else m.checks += 1;
    months.set(key, m);
  }
  // Every month from the first to now, so a month with no spend still shows.
  const keys = [...months.keys()].sort();
  const out = [];
  if (!keys.length) return out;
  for (let d = monthStart(keys[0]); d <= now; d = new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth() + 1, 1))) {
    const key = monthKey(d);
    const m = months.get(key) || { key, agent: 0, judge: 0, gpu: 0, runs: 0, checks: 0 };
    m.total = m.agent + m.judge + m.gpu;
    out.push(m);
  }
  return out;
}

// This month so far: spent, share of the budget, what is left per day, and where it ends at this pace.
function monthBurn(entries, now = new Date(), budget = state.budget) {
  const key = monthKey(now);
  const start = monthStart(key);
  const daysInMonth = new Date(Date.UTC(start.getUTCFullYear(), start.getUTCMonth() + 1, 0)).getUTCDate();
  const elapsedDays = Math.max((now - start) / 86400e3, 1 / 24);
  const daysLeft = Math.max(0, daysInMonth - elapsedDays);
  const spent = entries.filter((e) => monthKey(e.at) === key).reduce((sum, e) => sum + entrySpend(e), 0);
  const perDay = spent / Math.max(1, elapsedDays);
  const projected = perDay * daysInMonth;
  return {
    key, spent, perDay, projected, daysInMonth, daysLeft,
    budget: budget || null,
    used: budget ? spent / budget : null,
    left: budget ? budget - spent : null,
    allowedPerDay: budget ? Math.max(0, budget - spent) / Math.max(1, daysLeft) : null,
  };
}

// "$38.40 spent of the $100 budget. At this pace November ends near $71, inside the budget."
function burnHeadline(b, monthName) {
  if (!b.budget) return `${fmtUsd(b.spent)} spent in ${monthName} so far. At this pace the month ends near ${fmtUsd(b.projected)}.`;
  if (b.spent >= b.budget) return `${fmtUsd(b.spent)} spent: the ${fmtUsd(b.budget)} budget for ${monthName} is used up.`;
  return `${fmtUsd(b.spent)} spent of the ${fmtUsd(b.budget)} budget. At this pace ${monthName} ends near ${fmtUsd(b.projected)}, ` +
    (b.projected > b.budget ? `${fmtUsd(b.projected - b.budget)} over the budget.` : "inside the budget.");
}

function renderBurn(entries, runs) {
  const host = document.getElementById("cost-burn");
  const b = monthBurn(entries);
  const monthName = monthLongFmt.format(monthStart(b.key));
  document.getElementById("burn-title").textContent = `Month to date · ${monthName} (UTC)`;
  document.getElementById("burn-headline").textContent = burnHeadline(b, monthName);
  const over = b.budget && b.projected > b.budget;
  const status = !b.budget ? null
    : b.spent >= b.budget ? ["over", "Budget spent"]
    : over ? ["warn", "On pace to go over"]
    : ["ok", "On track"];
  const unpriced = runs.filter((r) => r.summary.unpriced_calls > 0 && monthKey(parseStamp(r.started_at)) === b.key);
  host.replaceChildren(...[
    el("div", { class: "burn" },
      el("div", { class: "burn-main" },
        el("div", { class: "burn-amount" }, el("strong", {}, fmtUsd(b.spent)),
          b.budget ? el("span", {}, ` of ${fmtUsd(b.budget)} budget`) : el("span", {}, " spent")),
        status ? el("span", { class: `burn-status ${status[0]}` }, status[1]) : null,
        b.budget ? el("div", { class: `meter budget big${over ? " over" : ""}`, role: "img",
          "aria-label": `${fmtPct(b.used)} of the ${fmtUsd(b.budget)} budget spent, projected ${fmtUsd(b.projected)}` },
        el("span", { style: `width:${Math.min(1, b.used) * 100}%` }),
        b.projected > b.spent ? el("b", { class: "proj", style: `left:${Math.min(1, b.used) * 100}%;width:${Math.max(0, Math.min(1, b.projected / b.budget) - Math.min(1, b.used)) * 100}%` }) : null,
        el("i", { style: `left:${Math.min(1, b.projected / b.budget) * 100}%`, title: `Projected ${fmtUsd(b.projected)}` })) : null,
        b.budget ? null : el("p", { class: "note" }, "Set MONTHLY_BUDGET_USD in Vercel to see how much of a budget is used.")),
      el("div", { class: "burn-stats" },
        b.budget ? stat("Budget used", fmtPct(b.used), b.left >= 0 ? `${fmtUsd(b.left)} left` : `${fmtUsd(-b.left)} over`) : null,
        stat("Days left", String(Math.ceil(b.daysLeft)), `of ${b.daysInMonth}`),
        stat("Spent per day", fmtUsd(b.perDay), "average so far"),
        b.budget ? stat("Can spend per day", fmtUsd(b.allowedPerDay), "to stay in budget") : null,
        stat("Month end at this pace", fmtUsd(b.projected), b.budget ? `${fmtPct(b.projected / b.budget)} of budget` : "projected"))),
    unpriced.length ? el("p", { class: "warn-note" },
      `${unpriced.reduce((n, r) => n + r.summary.unpriced_calls, 0)} model calls this month had no price ` +
      `(${[...new Set(unpriced.flatMap((r) => r.summary.unpriced_models || []))].join(", ")}), so they count as $0. ` +
      "Add the model to the price table in src/operon/models/usage.py.") : null].filter(Boolean));
}

function renderMonths(entries) {
  const months = monthTotals(entries);
  const budget = state.budget;
  const current = monthKey(new Date());
  const burn = monthBurn(entries);
  const parts = [
    { name: "Agent", key: "agent", color: "var(--series-1)" },
    { name: "Judge", key: "judge", color: "var(--series-4)" },
    { name: "GPU", key: "gpu", color: "var(--series-3)" },
  ];
  const projectedColor = "var(--text-muted)";
  legendItems(document.getElementById("lg-months"), [
    ...parts.map((p) => ({ name: p.name, color: p.color })),
    { name: "Projected rest of month", color: projectedColor },
  ]);
  barChart(document.getElementById("ch-months"), months.map((m) => ({
    date: monthStart(m.key),
    head: `${monthFmt.format(monthStart(m.key))}${m.key === current ? " · so far" : ""}`,
    segments: [
      ...parts.map((p) => ({ name: p.name, value: m[p.key], color: p.color })),
      ...(m.key === current ? [{ name: "Projected rest of month", value: Math.max(0, burn.projected - m.total), color: projectedColor, faint: true }] : []),
    ],
    notes: [`${fmtUsd(m.total)} spent · ${m.runs} run${m.runs === 1 ? "" : "s"}` + (m.checks ? ` · ${m.checks} judge check${m.checks === 1 ? "" : "s"}` : ""),
      budget ? `${fmtPct(m.total / budget)} of the ${fmtUsd(budget)} budget` : null].filter(Boolean),
  })), {
    fmt: fmtUsd, fmtAxis: fmtUsdAxis, title: "Spend by month",
    label: (c) => monthFmt.format(c.date),
    refLine: budget ? { value: budget, label: `Budget ${fmtUsd(budget)}` } : null,
  });
  const table = document.getElementById("cost-months");
  table.replaceChildren(
    el("thead", {}, el("tr", {}, ["Month", "Runs", "Agent", "Judge", "GPU", "Total", ...(budget ? ["Budget used"] : []), "vs previous month"]
      .map((h, i) => el("th", { class: i ? "num" : "" }, h)))),
    el("tbody", {}, [...months].reverse().map((m, i, list) => {
      const prev = list[i + 1];
      const change = prev && prev.total > 0 ? (m.total - prev.total) / prev.total : null;
      return el("tr", { class: m.key === current ? "current" : "" },
        el("td", {}, monthFmt.format(monthStart(m.key)), m.key === current ? el("span", { class: "sub" }, " · so far") : null),
        el("td", { class: "num" }, String(m.runs)),
        el("td", { class: "num" }, fmtUsd(m.agent)), el("td", { class: "num" }, fmtUsd(m.judge)),
        el("td", { class: "num" }, fmtUsd(m.gpu)), el("td", { class: "num" }, el("strong", {}, fmtUsd(m.total))),
        budget ? el("td", { class: "num" }, fmtPct(m.total / budget)) : null,
        el("td", { class: "num" }, change === null ? "–" : `${change >= 0 ? "+" : ""}${Math.round(change * 100)}%`));
    })));
}

function renderCostOverview(runs, entries) {
  const total = entries.reduce((sum, e) => sum + entrySpend(e), 0);
  const checks = entries.filter((e) => !e.run).reduce((sum, e) => sum + entrySpend(e), 0);
  const weekAgo = Date.now() - 7 * 86400e3;
  const week = runs.filter((r) => +parseStamp(r.started_at) >= weekAgo);
  const weekSpend = entries.filter((e) => +e.at >= weekAgo).reduce((sum, e) => sum + entrySpend(e), 0);
  const priciest = runs.reduce((best, r) => (runSpend(r) > runSpend(best) ? r : best), runs[0]);
  const first = parseStamp(runs[0].started_at);
  document.getElementById("cost-overview").replaceChildren(
    stat("Total spent", fmtUsd(total), `since ${dateFmt.format(first)}` + (checks ? ` · ${fmtUsd(checks)} judge checks` : "")),
    stat("Last 7 days", fmtUsd(weekSpend), `${week.length} run${week.length === 1 ? "" : "s"}`),
    stat("Runs with cost", String(runs.length), `${fmtUsd(runs.reduce((sum, r) => sum + runSpend(r), 0) / runs.length)} on average`),
    stat("Priciest run", fmtUsd(runSpend(priciest)), `${suiteLabel(suiteId(priciest))} · ${dateFmt.format(parseStamp(priciest.started_at))}`));
}

// Runs and judge checks, so the line ends at the same total as the headline.
function renderCumulative(entries) {
  let total = 0;
  const points = [...entries].sort((a, b) => a.at - b.at).map((e) => {
    total += entrySpend(e);
    return { x: e.at, y: total, run: e };
  });
  lineChart(document.getElementById("ch-cumulative"), el("div"),
    [{ label: "Total spend", color: "var(--accent)", points }],
    { fmt: (v, axis) => (axis ? fmtUsdAxis(v) : fmtUsd(v)),
      detail: (e) => `+${fmtUsd(entrySpend(e))} · ${e.run ? suiteLabel(suiteId(e.run)) : "judge check"}`, title: "Total spend over time" });
}

function renderPerRun(runs) {
  const ids = [...new Set(runs.map(suiteId))].sort((a, b) => SUITE_SLOTS.indexOf(a) - SUITE_SLOTS.indexOf(b));
  legendItems(document.getElementById("lg-per-run"), ids.map((id) => ({ name: suiteLabel(id), color: suiteColor(id) })));
  barChart(document.getElementById("ch-per-run"), runs.map((run) => ({
    date: parseStamp(run.started_at),
    head: timeFmt.format(parseStamp(run.started_at)),
    segments: [{ name: suiteLabel(suiteId(run)), value: runSpend(run), color: suiteColor(suiteId(run)) }],
    notes: [`${configLabel(run)} · ${run.summary.success}/${run.summary.tasks} passed`,
      `agent ${fmtUsd(run.summary.cost_usd)} · judge ${fmtUsd(run.summary.judge_cost_usd)}` +
        (run.summary.gpu_cost_usd ? ` · GPU ${fmtUsd(run.summary.gpu_cost_usd)}` : "")],
    onclick: () => selectCostRun(run),
  })), { fmt: fmtUsd, fmtAxis: fmtUsdAxis, title: "Spend per run" });
}

// Agent spend by model plus the judge and GPU time, summed per UTC day; judge checks count as Judge.
function renderByModel(entries) {
  const days = new Map();
  const totals = new Map();
  const add = (day, name, value) => {
    if (!(value > 0)) return;
    day.values.set(name, (day.values.get(name) || 0) + value);
    totals.set(name, (totals.get(name) || 0) + value);
  };
  for (const entry of [...entries].sort((a, b) => a.at - b.at)) {
    const when = entry.at;
    const id = `${when.getUTCFullYear()}-${when.getUTCMonth()}-${when.getUTCDate()}`;
    if (!days.has(id)) days.set(id, { date: new Date(Date.UTC(when.getUTCFullYear(), when.getUTCMonth(), when.getUTCDate())), values: new Map(), runs: 0 });
    const day = days.get(id);
    const run = entry.run;
    if (!run) { add(day, "Judge", entry.judge); continue; }
    day.runs += 1;
    const byModel = Object.entries(run.summary.by_model || {});
    let attributed = 0;
    for (const [name, bucket] of byModel) { add(day, name, bucket.cost_usd || 0); attributed += bucket.cost_usd || 0; }
    add(day, "Unattributed", (run.summary.cost_usd || 0) - attributed > 0.0005 ? run.summary.cost_usd - attributed : 0);
    add(day, "Judge", run.summary.judge_cost_usd || 0);
    add(day, "GPU time", run.summary.gpu_cost_usd || 0);
    add(day, "Held-out tasks", run.summary.held_out_cost_usd || 0);
  }
  const names = [...totals.entries()].sort((a, b) => b[1] - a[1]).map(([name]) => name);
  const color = (name) => `var(--series-${Math.min(names.indexOf(name) + 1, MAX_SLOTS)})`;
  legendItems(document.getElementById("lg-models"), names.map((name) => ({ name, color: color(name) })));
  barChart(document.getElementById("ch-models"), [...days.values()].map((day) => ({
    date: day.date,
    head: `${dateFmt.format(day.date)} · ${day.runs} run${day.runs === 1 ? "" : "s"}`,
    segments: names.map((name) => ({ name, value: day.values.get(name) || 0, color: color(name) })),
    notes: [`${fmtUsd([...day.values.values()].reduce((a, b) => a + b, 0))} that day`],
  })), { fmt: fmtUsd, fmtAxis: fmtUsdAxis, title: "Spend by model per day" });
}

function cacheHitRate(run) {
  const s = run.summary;
  const all = (s.input_tokens || 0) + (s.cache_read_tokens || 0) + (s.cache_write_tokens || 0);
  return all ? (s.cache_read_tokens || 0) / all : null;
}

function renderCacheRate(runs) {
  const points = runs.filter((r) => cacheHitRate(r) !== null).map((run) => ({ x: parseStamp(run.started_at), y: cacheHitRate(run), run }));
  lineChart(document.getElementById("ch-cache"), el("div"),
    points.length ? [{ label: "Cache hit rate", color: "var(--series-3)", points }] : [],
    { yMax: 1, fmt: (v) => fmtPct(v), title: "Cache hit rate by run",
      detail: (r) => `${suiteLabel(suiteId(r))} · ${fmtTokens(r.summary.cache_read_tokens)} read, ${fmtTokens(r.summary.cache_write_tokens)} written` });
}

// A run's GPU time split over its tasks by their seconds (evenly when none are recorded).
function taskGpuShare(t, siblings) {
  const gpu = t.run.summary.gpu_cost_usd || 0;
  if (!gpu) return 0;
  const seconds = siblings.reduce((sum, s) => sum + (s.seconds || 0), 0);
  return seconds ? gpu * (t.seconds || 0) / seconds : gpu / siblings.length;
}

function costTasks(runs) {
  const byKey = new Map(runs.map((r) => [r.key, r]));
  const tasks = (state.tasks || []).filter((t) => byKey.has(t.run_key)).map((t) => ({ ...t, run: byKey.get(t.run_key) }));
  const byRun = new Map();
  for (const t of tasks) byRun.set(t.run_key, [...(byRun.get(t.run_key) || []), t]);
  for (const t of tasks) t.gpu_cost_usd = taskGpuShare(t, byRun.get(t.run_key));
  return tasks;
}

// What one task attempt spent: agent, judge and its share of GPU time.
function taskSpend(t) {
  return (t.cost_usd || 0) + (t.judge_cost_usd || 0) + (t.gpu_cost_usd || 0);
}

function renderTopTasks(runs) {
  const table = document.getElementById("cost-top-tasks");
  if (state.tasks === null) { table.replaceChildren(el("tbody", {}, el("tr", {}, el("td", { class: "empty" }, "Loading…")))); return; }
  const groups = new Map();
  for (const t of costTasks(runs)) {
    const id = `${t.run.mode}|${t.task}`;
    const g = groups.get(id) || { task: t.task, mode: t.run.mode, attempts: 0, passed: 0, spend: 0, steps: 0 };
    g.attempts += 1;
    g.passed += t.outcome === "success" ? 1 : 0;
    g.spend += taskSpend(t);
    g.steps += t.steps || 0;
    groups.set(id, g);
  }
  const rows = [...groups.values()].sort((a, b) => b.spend / b.attempts - a.spend / a.attempts).slice(0, 10);
  table.replaceChildren(
    el("thead", {}, el("tr", {}, ["Task", "Env", "Attempts", "Passed", "Avg steps", "Per attempt", "Total"]
      .map((h, i) => el("th", { class: i >= 2 ? "num" : "" }, h)))),
    el("tbody", {}, rows.length ? rows.map((g) => el("tr", {},
      el("td", {}, g.task), el("td", {}, MODE_NAMES[g.mode] || g.mode),
      el("td", { class: "num" }, String(g.attempts)), el("td", { class: "num" }, fmtPct(g.passed / g.attempts)),
      el("td", { class: "num" }, (g.steps / g.attempts).toFixed(1)),
      el("td", { class: "num" }, el("strong", {}, fmtUsd(g.spend / g.attempts))), el("td", { class: "num" }, fmtUsd(g.spend))))
      : el("tr", {}, el("td", { class: "empty", colspan: 7 }, "No per-task cost recorded yet."))));
}

// Spend that bought no pass: every task in a flagged run, and failed tasks elsewhere.
function renderWaste(runs) {
  const host = document.getElementById("cost-waste");
  if (state.tasks === null) { host.replaceChildren(el("div", { class: "empty" }, "Loading…")); return; }
  const total = runs.reduce((sum, r) => sum + runSpend(r), 0);
  const buckets = new Map();
  const add = (name, value, color) => {
    const b = buckets.get(name) || { name, value: 0, color, tasks: 0 };
    b.value += value;
    b.tasks += 1;
    buckets.set(name, b);
  };
  for (const t of costTasks(runs)) {
    const spend = taskSpend(t);
    if (t.run.config && t.run.config.flagged) add("Flagged runs", spend, "var(--series-6)");
    else if (t.outcome !== "success") add(OUTCOME_LABELS[t.outcome] || t.outcome, spend, OUTCOME_COLORS[t.outcome] || "var(--text-muted)");
  }
  const rows = [...buckets.values()].sort((a, b) => b.value - a.value);
  const wasted = rows.reduce((sum, b) => sum + b.value, 0);
  host.replaceChildren(
    el("div", { class: "waste-head" }, el("strong", {}, fmtUsd(wasted)),
      el("span", {}, `${fmtPct(total ? wasted / total : 0)} of ${fmtUsd(total)} spent`)),
    el("div", { class: "stackbar", role: "img", "aria-label": "Wasted spend by reason" },
      rows.map((b) => el("span", { style: `width:${(b.value / (total || 1)) * 100}%;background:${b.color}`, title: `${b.name}: ${fmtUsd(b.value)}` }))),
    el("table", {}, el("tbody", {}, rows.length ? rows.map((b) => el("tr", {},
      el("td", {}, el("span", { class: "config-cell" }, el("span", { class: "key", style: `background:${b.color}` }), b.name)),
      el("td", { class: "num" }, `${b.tasks} task${b.tasks === 1 ? "" : "s"}`),
      el("td", { class: "num" }, fmtUsd(b.value)), el("td", { class: "num" }, fmtPct(total ? b.value / total : 0))))
      : el("tr", {}, el("td", { class: "empty" }, "Nothing wasted yet: every measured task passed.")))));
}

function renderCostSuites(runs) {
  const total = runs.reduce((sum, r) => sum + runSpend(r), 0);
  const groups = new Map();
  for (const run of runs) {
    const id = suiteId(run);
    const g = groups.get(id) || { id, runs: 0, spend: 0, last: run };
    g.runs += 1;
    g.spend += runSpend(run);
    g.last = run;
    groups.set(id, g);
  }
  const rows = [...groups.values()].sort((a, b) => b.spend - a.spend);
  document.getElementById("cost-suites").replaceChildren(
    el("thead", {}, el("tr", {}, ["Suite", "Runs", "Spent", "Share", "Last run"]
      .map((h, i) => el("th", { class: i >= 1 && i <= 3 ? "num" : "" }, h)))),
    el("tbody", {}, rows.map((g) => el("tr", {},
      el("td", {}, el("span", { class: "config-cell" }, el("span", { class: "key", style: `background:${suiteColor(g.id)}` }), suiteLabel(g.id))),
      el("td", { class: "num" }, String(g.runs)),
      el("td", { class: "num" }, fmtUsd(g.spend)),
      el("td", { class: "num" }, fmtPct(total ? g.spend / total : 0)),
      el("td", {}, dateFmt.format(parseStamp(g.last.started_at)))))));
}

function renderCostRuns(runs) {
  const table = document.getElementById("cost-runs");
  table.replaceChildren(el("thead", {}, el("tr", {},
    ["When", "Suite", "Configuration", "Passed", "Agent", "Judge", "GPU", "Run total", "Running total", "Run"]
      .map((h, i) => el("th", { class: i >= 3 && i <= 8 ? "num" : "" }, h)))));
  let running = 0;
  const rows = runs.map((run) => {
    running += runSpend(run);
    const s = run.summary;
    const flagged = run.config && run.config.flagged;
    return el("tr", {
      class: `clickable${state.costSelected === run.key ? " selected" : ""}`,
      tabindex: 0,
      onclick: () => selectCostRun(run),
      onkeydown: (e) => { if (e.key === "Enter") selectCostRun(run); },
    },
      el("td", {}, timeFmt.format(parseStamp(run.started_at)), flagged ? el("div", { class: "flag" }, `⚑ ${flagged}`) : null),
      el("td", {}, el("span", { class: "config-cell" }, el("span", { class: "key", style: `background:${suiteColor(suiteId(run))}` }), suiteLabel(suiteId(run)))),
      el("td", {}, configLabel(run)),
      el("td", { class: "num" }, `${s.success}/${s.tasks}`),
      el("td", { class: "num" }, fmtUsd(s.cost_usd)),
      el("td", { class: "num" }, fmtUsd(s.judge_cost_usd)),
      el("td", { class: "num" }, s.gpu_cost_usd ? fmtUsd(s.gpu_cost_usd) : "–"),
      el("td", { class: "num", title: s.held_out_cost_usd ? `includes ${fmtUsd(s.held_out_cost_usd)} on held-out tasks` : null },
        el("strong", {}, fmtUsd(runSpend(run)))),
      el("td", { class: "num" }, fmtUsd(running)),
      el("td", { onclick: (e) => e.stopPropagation() }, runLink(run)));
  });
  table.append(el("tbody", {}, rows.reverse()));
}

function renderCost() {
  const runs = costRuns();
  document.getElementById("cost-empty").hidden = runs.length > 0;
  document.getElementById("cost-body").hidden = runs.length === 0;
  if (!runs.length) return;
  const entries = spendEntries(runs);
  renderBurn(entries, runs);
  renderCostOverview(runs, entries);
  renderMonths(entries);
  renderCumulative(entries);
  renderPerRun(runs);
  renderByModel(entries);
  renderCacheRate(runs);
  renderTopTasks(runs);
  renderWaste(runs);
  renderCostRuns(runs);
  renderCostSuites(runs);
}

async function selectCostRun(run) {
  state.costSelected = run.key;
  const token = (state.costToken = (state.costToken || 0) + 1);
  renderCost();
  const wrap = document.getElementById("cost-detail");
  wrap.hidden = false;
  document.getElementById("cost-detail-sub").textContent =
    `${suiteLabel(suiteId(run))} · ${configLabel(run)} · ${timeFmt.format(parseStamp(run.started_at))} · ` +
    `${fmtUsd(run.summary.cost_usd)} agent, ${fmtUsd(run.summary.judge_cost_usd)} judge` +
    (run.summary.gpu_cost_usd ? `, ${fmtUsd(run.summary.gpu_cost_usd)} GPU` : "");
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
    if (token !== state.costToken) return;  // another run was picked meanwhile
    const rows = [...report.results].sort((a, b) => (b.cost_usd || 0) - (a.cost_usd || 0));
    tasks.replaceChildren(
      el("thead", {}, el("tr", {}, ["Task", "Outcome", "Steps", "Cost", "Input", "Output"]
        .map((h, i) => el("th", { class: i >= 2 ? "num" : "" }, h)))),
      el("tbody", {}, rows.map((r) => el("tr", {}, el("td", {}, r.task), el("td", {}, pill(r.outcome)),
        el("td", { class: "num" }, fmtNum(r.steps)), el("td", { class: "num" }, fmtUsd(r.cost_usd)),
        el("td", { class: "num" }, fmtTokens(r.input_tokens)), el("td", { class: "num" }, fmtTokens(r.output_tokens))))));
  } catch (err) {
    if (token !== state.costToken) return;
    tasks.replaceChildren(el("tbody", {}, el("tr", {}, el("td", { class: "empty" }, `Could not load the report: ${err.message}`))));
  }
}

// ---------- shell ----------

// ---------- leaderboard tab ----------
// One row per model: a registry profile (cross-model runs, judged by the model's own
// "done") or, for runs from before the registry, the backend's model and harness.

function modelLabel(run) {
  const c = run.config || {};
  if (c.model_profile) return registryLabel(c);
  const name = c.model || BACKEND_NAMES[c.backend] || c.backend || "unknown";
  return harnessSuffix(name, c.harness_profile);
}

function runTotalCost(run) {
  return (run.summary.cost_usd || 0) + (run.summary.gpu_cost_usd || 0);
}

function leaderboardData() {
  const latest = new Map();  // model -> suite -> run
  for (const run of state.runs) {
    if (!run.summary || (run.config && run.config.flagged) || COST_ONLY_MODES.has(run.mode)) continue;
    const model = modelLabel(run);
    const suite = suiteId(run);
    if (!latest.has(model)) latest.set(model, new Map());
    const have = latest.get(model).get(suite);
    if (!have || run.started_at > have.started_at) latest.get(model).set(suite, run);
  }
  const suites = [...new Set([...latest.values()].flatMap((m) => [...m.keys()]))]
    .sort((a, b) => ((SUITE_SLOTS.indexOf(a) + 1) || 99) - ((SUITE_SLOTS.indexOf(b) + 1) || 99) || a.localeCompare(b));
  const models = [...latest.entries()].map(([model, bySuite]) => {
    const runs = [...bySuite.values()];
    const sum = (f) => runs.reduce((acc, r) => acc + (Number(f(r)) || 0), 0);
    const tasks = sum((r) => r.summary.tasks);
    const replies = sum((r) => r.summary.model_replies);
    return {
      model, bySuite,
      ownDone: runs.some((r) => r.config && (r.config.model_profile || r.config.harness_profile === "thin-own")),
      tasks,
      rate: tasks ? sum((r) => r.summary.success) / tasks : null,
      falsePass: sum((r) => r.summary.false_success),
      compromised: sum((r) => r.summary.compromised),
      secondsPerTask: tasks ? sum((r) => r.summary.mean_seconds * r.summary.tasks) / tasks : null,
      costPerTask: tasks && runs.some(hasCost) ? sum(runTotalCost) / tasks : null,
      costPerSuccess: runs.some(hasCost) && sum((r) => r.summary.success) ? sum(runTotalCost) / sum((r) => r.summary.success) : null,
      invalidRate: replies ? sum((r) => r.summary.invalid_replies) / replies : null,
    };
  }).sort((a, b) => (b.rate ?? -1) - (a.rate ?? -1));
  // Registry models (evals/models.yaml, via models.json) that have no run yet still get a row.
  const seen = new Set(models.map((m) => m.model));
  for (const r of state.registry || []) {
    if (seen.has(r.profile)) continue;
    models.push({ model: r.profile, bySuite: new Map(), notRun: true, registry: r, ownDone: true, tasks: 0, rate: null,
      falsePass: null, compromised: null, secondsPerTask: null, costPerTask: null, costPerSuccess: null, invalidRate: null });
  }
  return { suites, models };
}

function lbColor(index) {
  return `var(--series-${(index % MAX_SLOTS) + 1})`;
}

function renderLeaderboardMatrix(data) {
  const table = document.getElementById("lb-matrix");
  table.replaceChildren(el("thead", {}, el("tr", {}, el("th", {}, ""), el("th", {}, "Model"),
    data.suites.map((id) => el("th", { class: "num" }, suiteLabel(id))))));
  const body = el("tbody");
  data.models.forEach((m, i) => {
    body.append(el("tr", { class: m.notRun ? "not-run-row" : null },
      el("td", {}, el("span", { class: "rank" }, m.notRun ? "–" : String(i + 1))),
      el("td", {}, el("span", { class: "config-cell" }, key(m.model, "key", lbColor(i)), m.model,
        m.ownDone ? null : el("span", { class: "sub", title: "Judged by Operon's completion check, not the model's own done" }, " (check)"))),
      m.notRun ? el("td", { class: "num not-run", colspan: Math.max(1, data.suites.length) }, "not run yet · ",
        el("code", {}, `/bench model=${m.model}`)) : data.suites.map((id) => {
        const run = m.bySuite.get(id);
        if (!run) return el("td", { class: "num" }, "–");
        const s = run.summary;
        const title = [
          `${s.success}/${s.tasks} pass`, `${s.false_success || 0} false pass`,
          s.compromised ? `${s.compromised} compromised` : null,
          `${s.mean_seconds} s/task`, hasCost(run) ? `${fmtUsd(runTotalCost(run) / (s.tasks || 1))}/task` : null,
          s.invalid_action_rate !== undefined ? `${(s.invalid_action_rate * 100).toFixed(1)}% invalid replies` : null,
        ].filter(Boolean).join(" · ");
        const rate = s.success_rate || 0;
        return el("td", { class: "num", title }, el("span", { class: `pill ${rate >= 0.9 ? "success" : rate >= 0.6 ? "missed" : "failure"}` }, `${Math.round(rate * 100)}%`),
          el("span", { class: "sub" }, ` ${s.success}/${s.tasks}`));
      })));
  });
  table.append(body);
}

function renderLeaderboardModels(data) {
  const table = document.getElementById("lb-models");
  const pctText = (v) => (v === null ? "–" : `${(v * 100).toFixed(v < 0.1 && v > 0 ? 1 : 0)}%`);
  table.replaceChildren(el("thead", {}, el("tr", {},
    el("th", {}, "Model"), el("th", { class: "num" }, "Pass"), el("th", { class: "num" }, "False pass"),
    el("th", { class: "num" }, "Compromised"), el("th", { class: "num" }, "s/task"),
    el("th", { class: "num", title: "Agent model calls plus GPU time, per task" }, "$/task"),
    el("th", { class: "num", title: "Agent model calls plus GPU time, per passed task" }, "$/success"),
    el("th", { class: "num" }, "Invalid"))));
  table.append(el("tbody", {}, data.models.map((m, i) => m.notRun ? el("tr", { class: "not-run-row" },
    el("td", {}, el("span", { class: "config-cell" }, key(m.model, "key", lbColor(i)), m.model,
      el("span", { class: "sub" }, ` ${m.registry.model}${m.registry.self_hosted ? " · self-hosted" : ""}`))),
    el("td", { class: "num not-run", colspan: 7 }, `not run yet · ${m.registry.environments.join(" and ")}`)) : el("tr", {},
    el("td", {}, el("span", { class: "config-cell" }, key(m.model, "key", lbColor(i)), m.model)),
    el("td", { class: "num" }, pctText(m.rate)),
    el("td", { class: "num" }, String(m.falsePass)),
    el("td", { class: "num" }, String(m.compromised)),
    el("td", { class: "num" }, m.secondsPerTask === null ? "–" : m.secondsPerTask.toFixed(0)),
    el("td", { class: "num" }, fmtUsd(m.costPerTask)),
    el("td", { class: "num" }, fmtUsd(m.costPerSuccess)),
    el("td", { class: "num" }, pctText(m.invalidRate))))));
}

function renderLeaderboardScatter(data) {
  const host = document.getElementById("ch-lb-scatter");
  const legend = document.getElementById("lg-lb-scatter");
  host.replaceChildren();
  legend.replaceChildren();
  const points = data.models.map((m, i) => ({ ...m, color: lbColor(i) })).filter((m) => m.rate !== null && m.costPerTask > 0);
  if (!points.length) { host.append(el("div", { class: "empty" }, "No runs with cost yet.")); return; }
  for (const p of points) legend.append(el("span", {}, key(p.model, "key", p.color), p.model));
  const width = Math.max(host.clientWidth, 280);
  const height = 260;
  const m = { l: 48, r: 20, t: 12, b: 32 };
  const logs = points.map((p) => Math.log10(p.costPerTask));
  let x0 = Math.floor(Math.min(...logs)), x1 = Math.ceil(Math.max(...logs));
  if (x1 === x0) x1 = x0 + 1;
  const sx = (v) => m.l + ((Math.log10(v) - x0) / (x1 - x0)) * (width - m.l - m.r);
  const sy = (r) => m.t + (1 - r) * (height - m.t - m.b);
  const root = svg("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": "Cost per task against pass rate" });
  const axis = svg("g", { class: "axis" });
  for (let i = 0; i <= 4; i++) {
    const r = i / 4;
    axis.append(svg("line", { class: i === 0 ? "baseline" : "gridline", x1: m.l, x2: width - m.r, y1: sy(r), y2: sy(r) }));
    const t = svg("text", { x: m.l - 8, y: sy(r) + 4, "text-anchor": "end" });
    t.textContent = `${Math.round(r * 100)}%`;
    axis.append(t);
  }
  for (let e = x0; e <= x1; e++) {
    const v = 10 ** e;
    const t = svg("text", { x: sx(v), y: height - 10, "text-anchor": "middle" });
    t.textContent = fmtUsdAxis(v);
    axis.append(t);
  }
  root.append(axis);
  for (const p of points) {
    const dot = svg("circle", { cx: sx(p.costPerTask), cy: sy(p.rate), r: 6, fill: p.color, stroke: "var(--surface-1)", "stroke-width": 2 });
    const title = svg("title", {});
    title.textContent = `${p.model}: ${(p.rate * 100).toFixed(0)}% pass at ${fmtUsd(p.costPerTask)} per task`;
    dot.append(title);
    root.append(dot);
  }
  host.append(root);
}

function renderLeaderboardBrief(data) {
  const ran = data.models.filter((m) => !m.notRun);
  const total = data.models.length;
  const lead = ran[0];
  let text = `${ran.length} of ${total} model${total === 1 ? "" : "s"} ${ran.length === 1 ? "has" : "have"} run.`;
  if (lead && lead.rate !== null) {
    text += ` ${lead.model} leads at ${fmtPct(lead.rate)}`;
    const cheaper = ran.slice(1).filter((m) => m.costPerTask > 0 && lead.costPerTask > 0)
      .sort((a, b) => a.costPerTask - b.costPerTask)[0];
    if (cheaper && lead.costPerTask > cheaper.costPerTask * 1.2) {
      text += `, at ${(lead.costPerTask / cheaper.costPerTask).toFixed(1)}x ${cheaper.model}'s cost per task`;
    }
    text += ".";
  }
  const mixed = ran.some((m) => !m.ownDone);
  document.getElementById("lb-brief").replaceChildren(
    el("div", { class: "eyebrow" }, mixed
      ? "Same tasks and graders for every model. Rows marked (check) were judged by Operon's completion check, the rest by the model's own \"done\""
      : "Same tasks and graders for every model, each judged by its own \"done\""),
    el("h1", { class: "headline" }, text),
    el("p", { class: "lede" }, "Ranked by pass rate over each model's latest unflagged run per suite. Models that haven't run show the command that runs them."));
}

// ---------- run plan: each model's Browser / Desktop / Hybrid x Thin / Full grid ----------
// Vickey's order (Nov 9; Jira KAN-66, KAN-70, KAN-71): Gemini first, then Claude, then GPT.
// Claude and GPT wait for Vickey's approval after Gemini's grid has run. `needs` names the
// story that builds a harness the model can't run yet: drop the entry when that story merges.
const RUN_PLAN = [
  { family: "Gemini", story: "KAN-66", needs: {} },
  { family: "Claude", story: "KAN-70", afterGemini: true, needs: {} },
  { family: "GPT", story: "KAN-71", afterGemini: true, needs: {} },
];
// The season the plan runs on, and each category's core set as suiteId gives it (KAN-72,
// evals/categories.py). A category with `needs` has no tasks until that story merges.
// Grounding is one model call per sample, with no harness: it stays outside the grid.
const PLAN_SEASON = 1;
const PLAN_CATEGORIES = [
  { id: "browser", name: "Browser", suite: "browser|core" },
  { id: "desktop", name: "Desktop", suite: "desktop|core" },
  { id: "hybrid", name: "Hybrid", suite: "hybrid|core", needs: "KAN-73" },
];
const ADAPTER_FAMILIES = { claude_native: "Claude", gemini_native: "Gemini", openai_native: "GPT" };
const BACKEND_FAMILIES = { claude_computer_use: "Claude", computer_use: "Gemini", json: "Gemini" };

// The plan row a run belongs to (Gemini, Claude or GPT), or null (an open-weight model).
function modelFamily(run, registry) {
  const c = run.config || {};
  if (c.model_profile) {
    const entry = (registry || []).find((r) => r.profile === c.model_profile);
    return (entry && ADAPTER_FAMILIES[entry.adapter]) || null;
  }
  return BACKEND_FAMILIES[c.backend] || null;
}

function harnessMode(run) {
  const h = (run.config || {}).harness_profile || "full";
  return h === "full" ? "full" : h.startsWith("thin") ? "thin" : null;
}

// One grid cell: a category's latest core run in one harness, with its tags broken out
// (summary.by_tag: short, long, safety). Safety passes only with nothing compromised.
function planCell(run) {
  if (!run) return null;
  const s = run.summary;
  const c = run.config || {};
  const tasks = Number(s.tasks) || 0;
  const success = Number(s.success) || 0;
  const tags = s.by_tag || {};
  const safety = tags.safety ? { ...tags.safety, compromised: tags.safety.compromised || 0 } : null;
  return {
    tasks, success, rate: tasks ? success / tasks : null,
    falsePass: Number(s.false_success) || 0,
    compromised: Number(s.compromised) || 0,
    costPerTask: tasks && hasCost(run) ? listedSpend(run) / tasks : null,
    short: tags.short || null, long: tags.long || null, safety,
    heldOut: c.season_held_out || null,
    workingOnly: Boolean(c.season_working_only),
    critics: c.critic_model ? [c.critic_model] : [],
    key: run.key, startedAt: run.started_at,
    range: s.score_range || null, change: s.change || null,
  };
}

// The KAN-60 badge: Improved / Regressed / No clear change against the cell's previous run,
// computed at import (evals/leaderboard/changes.py) from the paired difference's 95% range.
const CHANGE_CLASSES = { Improved: "up", Regressed: "down" };

function changeBadge(change) {
  if (!change) return null;
  const pts = change.difference === undefined ? null : Math.round(change.difference * 100);
  const text = pts === null ? change.label : `${change.label} (${pts > 0 ? "+" : ""}${pts} pts)`;
  const flips = (change.fixed || []).length + (change.broke || []).length;
  const title = change.previous ? `vs ${timeFmt.format(parseStamp(change.previous.started_at))}: ${change.previous.success}/${change.previous.tasks}`
    + (change.low !== undefined ? `; difference 95% range ${Math.round(change.low * 100)} to ${Math.round(change.high * 100)} pts` : "")
    + `; ${flips} task${flips === 1 ? "" : "s"} flipped` : "no earlier run of this cell";
  return el("span", { class: `delta change ${CHANGE_CLASSES[change.label] || ""}`, title }, text);
}

function rangeText(range) {
  return range ? `range ${fmtPct(range.low)} to ${fmtPct(range.high)}` : null;
}

// One row per planned model and category: its latest unflagged Season 1 core run per
// harness, or, for a cell with nothing yet, why.
function planGrid(runs, registry) {
  const allowed = new Set(PLAN_CATEGORIES.map((cat) => cat.suite));
  const byKey = new Map();
  for (const run of runs) {
    const c = run.config || {};
    if (!run.summary || c.flagged || c.season !== PLAN_SEASON) continue;
    const family = modelFamily(run, registry);
    const harness = harnessMode(run);
    const suite = suiteId(run);
    if (!family || !harness || !allowed.has(suite)) continue;
    const key = `${family}|${harness}|${suite}`;
    const have = byKey.get(key);
    if (!have || run.started_at > have.started_at) byKey.set(key, run);
  }
  const rows = [];
  RUN_PLAN.forEach((plan, i) => PLAN_CATEGORIES.forEach((cat, j) => rows.push({
    ...plan, step: i + 1, first: j === 0, category: cat.id, categoryName: cat.name, categoryNeeds: cat.needs || null,
    thin: planCell(byKey.get(`${plan.family}|thin|${cat.suite}`)),
    full: planCell(byKey.get(`${plan.family}|full|${cat.suite}`)),
  })));
  // Gemini's grid is done once every category that has tasks ran in both harnesses.
  const geminiDone = rows.filter((r) => r.family === "Gemini" && !r.categoryNeeds).every((r) => r.thin && r.full);
  for (const row of rows) {
    for (const harness of ["thin", "full"]) {
      if (row[harness]) continue;
      row[`${harness}Why`] = row.categoryNeeds ? `no tasks until ${row.categoryNeeds}`
        : row.needs[harness] ? `needs ${row.needs[harness]}`
          : row.afterGemini && !geminiDone ? "waits for your approval after Gemini's grid has run"
            : "not run yet";
    }
    row.delta = harnessDelta(row.thin, row.full);
  }
  return rows;
}

// Full minus thin on the same category's core set: Operon's north-star number.
function harnessDelta(thin, full) {
  if (!thin || !full || thin.rate === null || full.rate === null) return null;
  return full.rate - thin.rate;
}

function planCellView(cell, why, id) {
  if (!cell) return el("td", { class: "not-run" }, why);
  const tag = (name, t) => (t ? `${name} ${t.success}/${t.tasks}` : null);
  const safety = cell.safety ? `safety ${cell.safety.passed ? "passed" : "failed"} (${cell.safety.success}/${cell.safety.tasks}`
    + (cell.safety.compromised ? `, ${cell.safety.compromised} compromised` : "") + ")" : null;
  const title = [tag("short", cell.short), tag("long", cell.long), safety,
    cell.heldOut ? `held-out ${cell.heldOut.passed}/${cell.heldOut.run}` : null,
    cell.workingOnly ? "working tasks only" : null,
    cell.critics.length ? `critic: ${cell.critics.join(", ")}` : null].filter(Boolean).join(" · ");
  const rate = cell.rate;
  const open = state.lbCell === id;
  const toggle = () => { state.lbCell = open ? null : id; renderRunPlan(); };
  return el("td", { title, class: `plan-cell${open ? " open" : ""}`, tabindex: "0", role: "button", "aria-expanded": String(open),
    onclick: toggle, onkeydown: (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); } } },
    rate === null ? el("span", { class: "pill none" }, "no tasks")
      : el("span", { class: `pill ${rate >= 0.9 ? "success" : rate >= 0.6 ? "missed" : "failure"}` }, fmtPct(rate)),
    el("span", { class: "sub" }, cell.tasks ? ` ${cell.success}/${cell.tasks}` : ""),
    rangeText(cell.range) ? el("span", { class: "sub" }, ` · ${rangeText(cell.range)}`) : null,
    cell.change ? el("div", {}, changeBadge(cell.change)) : null,
    el("div", { class: "sub" }, [tag("short", cell.short), tag("long", cell.long),
      cell.safety ? (cell.safety.passed ? "safety passed" : "safety failed") : null].filter(Boolean).join(" · ")),
    el("div", { class: "sub" }, [
      `${cell.falsePass} false pass`, cell.costPerTask !== null ? `${fmtUsd(cell.costPerTask)}/task` : null,
    ].filter(Boolean).join(" · ")));
}

function renderRunPlan() {
  const rows = planGrid(state.runs, state.registry);
  const table = document.getElementById("lb-plan");
  table.replaceChildren(el("thead", {}, el("tr", {},
    el("th", {}, "Step"), el("th", {}, "Model"), el("th", {}, "Category"), el("th", {}, "Thin harness"),
    el("th", {}, "Full harness"), el("th", { class: "num", title: "Full harness pass rate minus thin, in points" }, "Full − thin"))));
  const span = PLAN_CATEGORIES.length;
  table.append(el("tbody", {}, rows.map((row) => el("tr", {},
    row.first ? el("td", { rowspan: String(span) }, el("span", { class: "rank" }, String(row.step))) : null,
    row.first ? el("td", { rowspan: String(span) }, row.family, el("div", { class: "sub" }, row.story)) : null,
    el("td", {}, row.categoryName),
    planCellView(row.thin, row.thinWhy, `${row.family}|${row.category}|thin`),
    planCellView(row.full, row.fullWhy, `${row.family}|${row.category}|full`),
    el("td", { class: "num" }, row.delta === null ? "–"
      : el("span", { class: `delta ${row.delta > 0 ? "up" : row.delta < 0 ? "down" : ""}` },
        `${row.delta > 0 ? "+" : ""}${Math.round(row.delta * 100)} pts`))))));
  renderFlips(rows);
}

// The flipped tasks of the clicked cell, against the previous run of that cell.
function flipsView(row, harness) {
  const cell = row[harness];
  const name = `${row.family} · ${row.categoryName} · ${harness === "thin" ? "Thin" : "Full"}`;
  const change = cell.change;
  if (!change || !change.previous) {
    return [el("h3", {}, name), el("p", { class: "note" }, change ? "This is the cell's first run: nothing to compare with yet."
      : "This run has no comparison yet; it is added when the next benchmark is imported.")];
  }
  const p = change.previous;
  const list = (title, names, cls) => el("div", { class: "flips" }, el("span", { class: `delta ${cls}` }, title),
    names.length ? el("ul", {}, names.map((n) => el("li", {}, n))) : el("span", { class: "sub" }, " none"));
  return [
    el("h3", {}, name, " ", changeBadge(change)),
    el("p", { class: "note" }, `${cell.success}/${cell.tasks}`, cell.range ? ` (${rangeText(cell.range)})` : "",
      ` against the previous run (${timeFmt.format(parseStamp(p.started_at))}): ${p.success}/${p.tasks}`,
      p.github_run_url ? [" · ", el("a", { href: p.github_run_url, target: "_blank", rel: "noopener" }, "its workflow run")] : null,
      change.low !== undefined ? ` · difference ${Math.round(change.low * 100)} to ${Math.round(change.high * 100)} pts (95% range)` : ""),
    list("Now pass", change.fixed || [], "up"),
    list("Now fail", change.broke || [], "down"),
  ];
}

function renderFlips(rows) {
  const host = document.getElementById("lb-flips");
  if (!host) return;
  const [family, category, harness] = (state.lbCell || "").split("|");
  const row = rows.find((r) => r.family === family && r.category === category);
  host.hidden = !(row && row[harness]);
  host.replaceChildren(...(host.hidden ? [] : flipsView(row, harness)));
}

function renderLeaderboard() {
  renderRunPlan();
  const data = leaderboardData();
  document.getElementById("lb-empty").hidden = data.models.length > 0;
  document.getElementById("lb-body").hidden = data.models.length === 0;
  if (!data.models.length) return;
  renderLeaderboardBrief(data);
  renderLeaderboardMatrix(data);
  renderLeaderboardModels(data);
  renderLeaderboardScatter(data);
}

function render() {
  for (const [id, value] of [["f-mode", state.mode], ["f-suite", state.suite]]) {
    for (const b of document.getElementById(id).querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.v === value));
  }
  document.getElementById("f-flagged").checked = state.showFlagged;
  for (const b of document.querySelectorAll("nav.tabs button")) b.setAttribute("aria-selected", String(b.dataset.tab === state.tab));
  document.getElementById("tab-evals").hidden = state.tab !== "evals";
  document.getElementById("tab-cost").hidden = state.tab !== "cost";
  document.getElementById("tab-leaderboard").hidden = state.tab !== "leaderboard";
  document.getElementById("filterbar").hidden = state.tab !== "evals";
  if (state.tab === "leaderboard") {
    renderLeaderboard();
  } else if (state.tab === "evals") {
    const runs = filteredRuns();
    renderBrief(runs);
    renderTiles(runs);
    renderCharts(runs);
    renderOutcomes(runs);
    renderCategories(runs);
    renderHistory(runs);
    renderJudge();
    renderRuns(runs);
    renderMatrix(runs);
    const selected = runs.find((r) => r.key === state.selected);
    document.getElementById("detail").hidden = !selected;
  } else {
    renderCost();
    document.getElementById("cost-detail").hidden = !costRuns().some((r) => r.key === state.costSelected);
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
  document.getElementById("trend-metric").addEventListener("click", (event) => {
    const value = event.target.dataset && event.target.dataset.v;
    if (!value) return;
    state.trendMetric = value;
    store("operon-dash-trend", value);
    renderCharts(filteredRuns());
  });
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

async function getJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.error || `${url}: ${response.status}`);
  }
  return response.json();
}

async function main() {
  state.mode = store("operon-dash-f-mode") || state.mode;
  state.suite = store("operon-dash-f-suite") || state.suite;
  state.tab = store("operon-dash-tab") || state.tab;
  state.trendMetric = TREND_METRICS[store("operon-dash-trend")] ? store("operon-dash-trend") : state.trendMetric;
  bindControls();
  // Task results and judge checks feed extra cards; the page renders without them.
  const extras = Promise.all([
    getJson("api/tasks").then((d) => { state.tasks = d.tasks || []; }).catch(() => { state.tasks = []; state.tasksFailed = true; }),
    getJson("api/judge").then((d) => { state.judge = d.checks || []; }).catch(() => { state.judge = []; }),
    getJson("models.json").then((d) => { state.registry = d.models || []; }).catch(() => { state.registry = []; }),
  ]);
  try {
    const index = await getJson("api/runs");
    state.runs = index.runs || [];
    state.budget = index.budget_usd || null;
    noteModels(state.runs);
    assignSlots(state.runs);
    const generated = index.generated_at ? new Date(index.generated_at) : null;
    document.getElementById("meta").textContent =
      `${state.runs.length} benchmark runs` + (generated ? ` · updated ${timeFmt.format(generated)}` : "");
  } catch (err) {
    document.getElementById("meta").textContent = `Could not load benchmark data: ${err.message}`;
  }
  render();
  await extras;
  render();
}

main();
