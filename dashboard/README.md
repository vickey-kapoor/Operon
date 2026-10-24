# Operon eval dashboard

Shows the live benchmark results: pass rate, seconds per task and per step, false
passes, per-task outcomes by configuration (Claude or Gemini backend, full or thin
harness), and, for runs that record it, cost per task, per success, by model and
by task.

- `index.html`, `app.js`, `style.css`: the page. Plain JS and inline SVG charts, no build step.
- `api/runs.js` (`GET /api/runs`): every run with its summary and config.
- `api/report.js` (`GET /api/report?key=<run key>`): one run with its per-task results.
- `server/db.js`: the Postgres pool the routes share, from `DATABASE_URL`.

## Data

Results live in a Neon Postgres database connected to the Vercel project. Two
tables, created by the importer if missing:

- `bench_runs`: one row per benchmark run (mode, suite, backend, harness, model,
  start time, commit, workflow run, flag reason, `config` and `summary` as JSON).
- `bench_task_results`: one row per task in a run (outcome, steps, seconds, cost
  and tokens as columns, the full result as JSON). Deleting a run deletes its tasks.

The **Dashboard data** workflow (`.github/workflows/dashboard-data.yml`) runs after
every "Live benchmark", "Live desktop benchmark" or "Bench command" run and writes
its reports to the database, using the `DATABASE_URL` repository secret. Importing a
report again replaces it. Run the workflow by hand to import older workflow runs
(`run_ids`) or reports committed at a git ref (`backfill_ref`).

From a terminal:

```bash
pip install "psycopg[binary]"
export DATABASE_URL=postgres://...
python -m tests.eval.dashboard import .var/bench/full-20261024T012945Z.json
python -m tests.eval.dashboard import path/to/artifacts --backend json --flag "why it is not comparable"
```

Flagged runs (a broken runner, a harness bug) stay in the data but are hidden
unless "Flagged runs" is on.

## Run it locally

```bash
cd dashboard
npm install
DATABASE_URL=postgres://... npm run dev   # page and API on http://localhost:3000
```

## Deploy on Vercel

The project is `operon-evals` (Root Directory `dashboard`, Framework Preset *Other*,
no build settings). Every push to `main` that touches `dashboard/` redeploys it.

To connect the database: in the Vercel project, **Storage → Create Database →
Neon**, connect it to the project (this sets `DATABASE_URL`), then copy the same
connection string into the GitHub repository secret `DATABASE_URL`.
