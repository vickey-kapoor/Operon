# Operon eval dashboard

A static site (no build step) that shows the live benchmark results: pass rate,
seconds per task and per step, false passes, per-task outcomes by configuration
(Claude or Gemini backend, full or thin harness), and, for runs that record it,
cost per task, per success, by model and by task.

- `index.html`, `app.js`, `style.css`: the page. Plain JS and inline SVG charts.
- `data/index.json`: every run with its summary and config.
- `data/reports/<key>.json`: one benchmark report per run, with per-task results.

## Updating the data

The **Dashboard data** workflow (`.github/workflows/dashboard-data.yml`) runs after
every "Live benchmark", "Live desktop benchmark" or "Bench command" run, imports
the reports from its artifacts, and pushes them to `main`; Vercel redeploys on
that push. To add reports by hand:

```bash
python -m tests.eval.dashboard import .var/bench/full-20261024T012945Z.json
python -m tests.eval.dashboard import path/to/artifacts --backend json --flag "why it is not comparable"
```

Flagged runs (a broken runner, a harness bug) stay in the data but are hidden
unless "Show flagged runs" is on.

## Run it locally

```bash
python -m http.server 8000 --directory dashboard   # then open http://localhost:8000
```

## Deploy on Vercel

1. vercel.com → **Add New… → Project** → import `vickey-kapoor/Operon`.
2. Set **Root Directory** to `dashboard`, **Framework Preset** to *Other*, and leave
   the build and output settings empty.
3. Deploy. Every push to `main` that touches `dashboard/` redeploys it.

Or from a terminal: `cd dashboard && npx vercel --prod`.
