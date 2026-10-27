// Operon eval dashboard: renders /api/runs (one entry per benchmark run) and loads
// /api/report?key=<key> for a run's per-task results; both read the benchmark
// database. No build step; all text from the data goes into the DOM through textContent.
"use strict";

const SVG_NS = "http://www.w3.org/2000/svg";
const BACKEND_NAMES = { claude_computer_use: "Claude", computer_use: "Gemini", json: "Gemini" };
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
  reports: {},
  tasks: null,
  judge: null,
  budget: null,
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
  const yMax = opts.yMax || niceMax(Math.max(...totals) * 1.1);
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
    const day = dateFmt.format(c.date);
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
      g.append(svg("rect", { x: cx(i) - barW / 2, y: top, width: barW, height: Math.max(1, sy(base) - top), fill: seg.color }));
      base += seg.value;
    }
    root.append(g);
    return g;
  });
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
        ...(s.compromised ? [["Compromised", fmtNum(s.compromised), "color:var(--critical)"]] : []),
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
    el("thead", {}, el("tr", {}, el("th", {}, "Task"), el("th", {}, `Last ${recent.length} runs`), el("th", { class: "num" }, "Passed"), el("th", {}, ""))),
    el("tbody", {}, scored.filter((r) => state.historyAll || rank(r) < 2 || steady === scored.length).map((r) => el("tr", {},
      el("td", {}, r.task, el("div", { class: "muted-line" }, r.category || "")),
      el("td", {}, el("div", { class: "marks" }, r.cells.map((t, i) => el("span", {
        class: `mark ${t ? t.outcome : "none"}`,
        title: t ? `${timeFmt.format(parseStamp(recent[i].started_at))} · ${configLabel(recent[i])} · ${OUTCOME_LABELS[t.outcome] || t.outcome}${t.stop_reason ? ` · ${t.stop_reason}` : ""}` : "not run",
      })))),
      el("td", { class: "num" }, `${r.passed}/${r.seen}`),
      el("td", {}, r.flaky ? el("span", { class: "pill missed" }, "flaky") : r.never ? el("span", { class: "pill failure" }, "never passes") : null)))));
  let more = document.getElementById("history-more");
  if (!more) {
    more = el("button", { class: "ghost-btn", id: "history-more", type: "button" });
    table.parentElement.after(more);
  }
  more.hidden = steady === 0 || steady === scored.length;
  more.textContent = state.historyAll ? "Hide tasks that always pass" : `Show ${steady} tasks that always pass`;
  more.onclick = () => { state.historyAll = !state.historyAll; renderHistory(filteredRuns()); };
}

function renderJudge() {
  const host = document.getElementById("judge");
  if (state.judge === null) { host.replaceChildren(el("div", { class: "empty" }, "Loading…")); return; }
  const checks = state.judge;
  if (!checks.length) {
    host.replaceChildren(el("p", { class: "note" }, "No judge check saved yet. Label final screens and run ",
      el("code", {}, "python -m tests.eval.judge_check run labels.jsonl --save"), " to record one."));
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
        el("td", {}, d.judge === null || d.judge === undefined ? "no answer" : d.judge ? "done" : "not done"))))) : null);
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
    `${s.timeout} timeout · ${s.error} error` + (s.compromised ? ` · ${s.compromised} compromised` : "") +
    ` · judge ${run.judge_model || "–"}`;
  const table = document.getElementById("detail-table");
  table.replaceChildren(el("tbody", {}, el("tr", {}, el("td", { class: "empty" }, "Loading…"))));
  try {
    const report = await loadReport(run);
    const order = { compromised: -1, false_success: 0, failure: 1, missed: 2, timeout: 3, error: 4, success: 5 };
    const rows = [...report.results].sort((a, b) => (order[a.outcome] ?? 9) - (order[b.outcome] ?? 9));
    const withScreens = rows.some((r) => safeUrl(r.screenshot_url));
    const headers = ["Task", "Category", "Outcome", "Steps", "Seconds", "Stop reason", "Judge", ...(withScreens ? ["Screen"] : [])];
    table.replaceChildren(
      el("thead", {}, el("tr", {}, headers.map((h, i) => el("th", { class: i === 3 || i === 4 ? "num" : "" }, h)))),
      el("tbody", {}, rows.flatMap((r) => taskRows(r, withScreens, headers.length))));
  } catch (err) {
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

const SUITE_NAMES = { full: "full suite", smoke: "smoke", safety: "safety", apps: "apps" };
const MODE_NAMES = { browser: "Browser", desktop: "Desktop" };
// Fixed colors per environment and suite, so a suite keeps its color as runs come in.
const SUITE_SLOTS = ["browser|full", "browser|smoke", "browser|safety", "browser|apps", "desktop|smoke", "desktop|full"];

function hasCost(run) {
  return run.summary && run.summary.cost_usd !== undefined && run.summary.cost_usd !== null;
}

function runSpend(run) {
  return (run.summary.cost_usd || 0) + (run.summary.judge_cost_usd || 0);
}

function suiteId(run) {
  return `${run.mode}|${(run.suite || "").replace(/^desktop-/, "")}`;
}

function suiteLabel(id) {
  const [mode, suite] = id.split("|");
  return `${MODE_NAMES[mode] || mode} · ${SUITE_NAMES[suite] || suite}`;
}

function suiteColor(id) {
  const slot = SUITE_SLOTS.indexOf(id);
  return `var(--series-${slot >= 0 ? slot + 1 : MAX_SLOTS})`;
}

function costRuns() {
  return state.runs.filter(hasCost).sort((a, b) => parseStamp(a.started_at) - parseStamp(b.started_at));
}

function stat(label, value, hint) {
  return el("div", { class: "stat" }, el("div", { class: "label" }, label), el("div", { class: "value" }, value),
    hint ? el("div", { class: "hint" }, hint) : null);
}

function renderCostOverview(runs) {
  const total = runs.reduce((sum, r) => sum + runSpend(r), 0);
  const weekAgo = Date.now() - 7 * 86400e3;
  const week = runs.filter((r) => +parseStamp(r.started_at) >= weekAgo);
  const weekSpend = week.reduce((sum, r) => sum + runSpend(r), 0);
  const priciest = runs.reduce((best, r) => (runSpend(r) > runSpend(best) ? r : best), runs[0]);
  const first = parseStamp(runs[0].started_at);
  // This calendar month so far, and where it ends at the same daily pace.
  const now = new Date();
  const monthStart = new Date(now.getFullYear(), now.getMonth(), 1);
  const monthDays = new Date(now.getFullYear(), now.getMonth() + 1, 0).getDate();
  const month = runs.filter((r) => parseStamp(r.started_at) >= monthStart).reduce((sum, r) => sum + runSpend(r), 0);
  const elapsed = Math.max(1, (now - monthStart) / 86400e3);
  const projected = (month / elapsed) * monthDays;
  const monthName = new Intl.DateTimeFormat(undefined, { month: "long" }).format(now);
  const budget = state.budget;
  const pace = el("div", { class: "hint" }, `on pace for ${fmtUsd(projected)}` + (budget ? ` of ${fmtUsd(budget)}` : ""));
  const monthStat = stat(`${monthName} so far`, fmtUsd(month));
  monthStat.append(pace);
  if (budget) {
    const over = projected > budget;
    monthStat.append(el("div", { class: `meter budget${over ? " over" : ""}`, role: "img", "aria-label": `${fmtPct(month / budget)} of budget` },
      el("span", { style: `width:${Math.min(1, month / budget) * 100}%` }),
      el("i", { style: `left:${Math.min(1, projected / budget) * 100}%`, title: `Projected ${fmtUsd(projected)}` })));
  }
  document.getElementById("cost-overview").replaceChildren(
    stat("Total spent", fmtUsd(total), `since ${dateFmt.format(first)}`),
    monthStat,
    stat("Last 7 days", fmtUsd(weekSpend), `${week.length} run${week.length === 1 ? "" : "s"}`),
    stat("Runs with cost", String(runs.length), `${fmtUsd(total / runs.length)} on average`),
    stat("Priciest run", fmtUsd(runSpend(priciest)), `${suiteLabel(suiteId(priciest))} · ${dateFmt.format(parseStamp(priciest.started_at))}`));
}

function renderCumulative(runs) {
  let total = 0;
  const points = runs.map((run) => {
    total += runSpend(run);
    return { x: parseStamp(run.started_at), y: total, run };
  });
  lineChart(document.getElementById("ch-cumulative"), el("div"),
    [{ label: "Total spend", color: "var(--accent)", points }],
    { fmt: (v, axis) => (axis ? fmtUsdAxis(v) : fmtUsd(v)), detail: (r) => `+${fmtUsd(runSpend(r))} · ${suiteLabel(suiteId(r))}`, title: "Total spend over time" });
}

function renderPerRun(runs) {
  const ids = [...new Set(runs.map(suiteId))].sort((a, b) => SUITE_SLOTS.indexOf(a) - SUITE_SLOTS.indexOf(b));
  legendItems(document.getElementById("lg-per-run"), ids.map((id) => ({ name: suiteLabel(id), color: suiteColor(id) })));
  barChart(document.getElementById("ch-per-run"), runs.map((run) => ({
    date: parseStamp(run.started_at),
    head: timeFmt.format(parseStamp(run.started_at)),
    segments: [{ name: suiteLabel(suiteId(run)), value: runSpend(run), color: suiteColor(suiteId(run)) }],
    notes: [`${configLabel(run)} · ${run.summary.success}/${run.summary.tasks} passed`,
      `agent ${fmtUsd(run.summary.cost_usd)} · judge ${fmtUsd(run.summary.judge_cost_usd)}`],
    onclick: () => selectCostRun(run),
  })), { fmt: fmtUsd, fmtAxis: fmtUsdAxis, title: "Spend per run" });
}

// Agent spend by model plus the judge, summed per local day.
function renderByModel(runs) {
  const days = new Map();
  const totals = new Map();
  const add = (day, name, value) => {
    if (!(value > 0)) return;
    day.values.set(name, (day.values.get(name) || 0) + value);
    totals.set(name, (totals.get(name) || 0) + value);
  };
  for (const run of runs) {
    const when = parseStamp(run.started_at);
    const id = `${when.getFullYear()}-${when.getMonth()}-${when.getDate()}`;
    if (!days.has(id)) days.set(id, { date: new Date(when.getFullYear(), when.getMonth(), when.getDate()), values: new Map(), runs: 0 });
    const day = days.get(id);
    day.runs += 1;
    const byModel = Object.entries(run.summary.by_model || {});
    let attributed = 0;
    for (const [name, bucket] of byModel) { add(day, name, bucket.cost_usd || 0); attributed += bucket.cost_usd || 0; }
    add(day, "Unattributed", (run.summary.cost_usd || 0) - attributed > 0.0005 ? run.summary.cost_usd - attributed : 0);
    add(day, "Judge", run.summary.judge_cost_usd || 0);
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

function costTasks(runs) {
  const byKey = new Map(runs.map((r) => [r.key, r]));
  return (state.tasks || []).filter((t) => byKey.has(t.run_key)).map((t) => ({ ...t, run: byKey.get(t.run_key) }));
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
    g.spend += (t.cost_usd || 0) + (t.judge_cost_usd || 0);
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
    const spend = (t.cost_usd || 0) + (t.judge_cost_usd || 0);
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
    ["When", "Suite", "Configuration", "Passed", "Agent", "Judge", "Run total", "Running total", "Run"]
      .map((h, i) => el("th", { class: i >= 3 && i <= 7 ? "num" : "" }, h)))));
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
      el("td", { class: "num" }, el("strong", {}, fmtUsd(runSpend(run)))),
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
  renderCostOverview(runs);
  renderCumulative(runs);
  renderPerRun(runs);
  renderByModel(runs);
  renderCacheRate(runs);
  renderTopTasks(runs);
  renderWaste(runs);
  renderCostRuns(runs);
  renderCostSuites(runs);
}

async function selectCostRun(run) {
  state.costSelected = run.key;
  renderCost();
  const wrap = document.getElementById("cost-detail");
  wrap.hidden = false;
  document.getElementById("cost-detail-sub").textContent =
    `${suiteLabel(suiteId(run))} · ${configLabel(run)} · ${timeFmt.format(parseStamp(run.started_at))} · ` +
    `${fmtUsd(run.summary.cost_usd)} agent, ${fmtUsd(run.summary.judge_cost_usd)} judge`;
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
  for (const id of ["f-mode", "f-suite", "f-flagged-wrap"]) document.getElementById(id).hidden = state.tab === "cost";
  if (state.tab === "evals") {
    const runs = filteredRuns();
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
  bindControls();
  // Task results and judge checks feed extra cards; the page renders without them.
  const extras = Promise.all([
    getJson("api/tasks").then((d) => { state.tasks = d.tasks || []; }).catch(() => { state.tasks = []; }),
    getJson("api/judge").then((d) => { state.judge = d.checks || []; }).catch(() => { state.judge = []; }),
  ]);
  try {
    const index = await getJson("api/runs");
    state.runs = index.runs || [];
    state.budget = index.budget_usd || null;
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
