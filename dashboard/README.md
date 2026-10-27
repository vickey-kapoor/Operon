# Operon eval dashboard

Shows the live benchmark results: pass rate, seconds per task and per step, false
passes, per-task outcomes by configuration (Claude or Gemini backend, full or thin
harness), and, on the Cost tab, total spend over time: a running total and the
spend of each run across every environment and suite (flagged runs included),
spend by suite, and each run's spend by model and by task.

- `index.html`, `app.js`, `style.css`: the page. Plain JS and inline SVG charts, no build step.
- `api/runs.js` (`GET /api/runs`): every run with its summary and config.
- `api/report.js` (`GET /api/report?key=<run key>`): one run with its per-task results.
- `api/blob.js` (`GET /api/blob?url=<blob url>`): one screenshot or trace from the private Blob store.
- `server/db.js`: the Postgres pool the routes share, from `DATABASE_URL`.
- `scripts/blob-upload.mjs`: uploads screenshots and traces to Vercel Blob for the importer.

## Access

The dashboard is protected by Vercel's own login (Settings > Deployment Protection >
Vercel Authentication, scope "All Deployments", which covers the production URL and
is available on the Hobby plan). Only signed-in Vercel users with access to the
project, or someone holding a shareable link, can open the page or the API. With
protection turned off everything is public, screenshots included, so keep it on.
The local `npm run dev` server listens on 127.0.0.1 only.

The page only reads the database, so give Vercel a read-only role rather than the
owner connection string the importer uses. In the Neon SQL editor:

```sql
CREATE ROLE dashboard_reader LOGIN PASSWORD '<a new password>';
GRANT USAGE ON SCHEMA public TO dashboard_reader;
GRANT SELECT ON bench_runs, bench_task_results TO dashboard_reader;
```

Then set the Vercel project's `DATABASE_URL` to that role's connection string; the
GitHub `DATABASE_URL` secret keeps the owner role, since the importer writes.

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

Flagged runs (a broken runner, a harness bug, a run stopped at its cost cap) stay
in the data but are hidden unless "Flagged runs" is on.

## Screenshots and traces

The benchmark workflows run `live_bench --keep-screens`, which saves each task's
graded screen and, for tasks that did not pass, the whole run folder. The **Dashboard
data** workflow uploads them to Vercel Blob (`--blob`, `BLOB_READ_WRITE_TOKEN`
repository secret): the final screen of every task, and every step's screen plus a
zip of the run folder for the others. The links are stored in each task's result
(`screenshot_url`, `step_screenshots`, `trace_url`). In a run's detail, the Screen
column shows the final screen, and clicking a task that did not pass shows its
steps and the trace download. The Blob store is **private**: its files can't be read
without credentials, and the page loads them through `/api/blob`, which reads them as
the Vercel project (`BLOB_STORE_ID`, set when the store is connected to the project, or
`BLOB_READ_WRITE_TOKEN`). The benchmark workflows upload with the `BLOB_READ_WRITE_TOKEN`
GitHub secret. So the screenshots are
visible to whoever has the dashboard password, and to no one else.

Without the secret, results are imported without screenshots.

## Alerts

After importing a finished benchmark run, the workflow (`--alert`) posts to a Slack
incoming webhook (`SLACK_WEBHOOK_URL` repository secret) when:

- its pass rate is 5 or more points below the previous comparable run (same mode,
  suite, backend, harness and model, not flagged),
- its agent cost per task is 25% or more above that run's, or
- it stopped at its cost cap (`MAX_RUN_COST_USD`, the workflows' `max_cost_usd` input, $5 by default).

Without the secret, no alerts are sent.

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
