# Operon eval dashboard

Shows the live benchmark results. Each tab opens with a one-sentence summary built
from the stored runs (a fixed template, no model call, so it updates after every
imported run): on Evals, the latest run's result and what changed since the
configuration's previous run (tasks that started or stopped failing, speed, false
passes), with a "likely cause" for each miss: the judge's or grader's reason, or when a
run has none (model comparisons run without a judge), how the task ended and why the
agent stopped. Below it: number tiles with trends, one trend chart that switches
between pass rate, speed, steps and cost, false
passes, per-task outcomes by configuration (Claude or Gemini backend, full or thin
harness), why tasks fail, pass rate by category, each task's history across runs
(flaky ones first) and the judge's agreement with hand labels. The Cost tab shows
spend across every environment and suite (flagged, incomplete and partial runs
included, plus grounding evals and judge checks): this month's spend against the
monthly budget (used, left, per-day pace, month-end projection), spend by month,
a running total, spend per run, spend by model per day, cache hit rate, the most
expensive tasks, wasted spend, spend by suite, and each run's spend by model and by
task. Spend is the agent's model calls, the judge and a self-hosted model's GPU
time; months are calendar months in UTC. The Leaderboard lists every model in the
registry (`models.json`, written from `evals/models.yaml` by
`python -m evals.dashboard models`), with "not run yet" until a model has a run.
Its thin-against-full grid shows each cell's score with a 95% range and an Improved /
Regressed / No clear change badge against the previous run of that cell; clicking a cell
lists the tasks that flipped (`summary.score_range` and `summary.change`, written by the
importer; KAN-60, docs/evaluation.md).

- `index.html`, `app.js`, `style.css`: the page. Plain JS and inline SVG charts, no build step.
- `api/runs.js` (`GET /api/runs`): every run with its summary and config.
- `api/report.js` (`GET /api/report?key=<run key>`): one run with its per-task results.
- `api/tasks.js` (`GET /api/tasks`): every task result of every run, slimmed for the history and cost cards.
- `api/judge.js` (`GET /api/judge`): judge checks saved by `python -m evals.judge_check run labels.jsonl --save`, with what their judge calls cost.
- `models.json`: the model registry for the Leaderboard; a test keeps it in step with `evals/models.yaml`.
- `api/blob.js` (`GET /api/blob?url=<blob url>`): one screenshot or trace from the private Blob store.
- `server/db.js`: the Postgres pool the routes share, from `DASHBOARD_DATABASE_URL` (read-only role), else `DATABASE_URL`.
- `scripts/blob-upload.mjs`: uploads screenshots and traces to Vercel Blob for the importer.

Set `MONTHLY_BUDGET_USD` in the Vercel project's environment variables to see how
much of the monthly budget is used, the budget line on the month chart and what can
still be spent per day. Without it the tab shows spend and pace only.

The Cost tab counts what benchmark runs, grounding evals and judge checks spent. It
leaves out spend outside them: local runs, the Azure desktop VM, and GPU VM hours
while idle. Each provider's billing page has the full bill.

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
CREATE ROLE dashboard_ro LOGIN PASSWORD '<a new password>';
GRANT CONNECT ON DATABASE neondb TO dashboard_ro;
GRANT USAGE ON SCHEMA public TO dashboard_ro;
GRANT SELECT ON bench_runs, bench_task_results, judge_checks TO dashboard_ro;
```

Then add a Vercel environment variable `DASHBOARD_DATABASE_URL` with that role's
connection string and redeploy. The API reads it before `DATABASE_URL`, which the
Neon integration manages (it holds the owner role and Vercel won't let you edit it).
The GitHub `DATABASE_URL` secret keeps the owner role, since the importer writes.

## Data

Results live in a Neon Postgres database connected to the Vercel project. Two
tables, created by the importer if missing:

- `bench_runs`: one row per benchmark run (mode, suite, backend, harness, model,
  start time, commit, workflow run, flag reason, `config` and `summary` as JSON).
- `bench_task_results`: one row per task in a run, and per attempt in a `--repeat`
  run (outcome, steps, seconds, cost and tokens as columns, the full result as JSON).
  Deleting a run deletes its tasks.

Grounding evals (`evals.grounding`) are stored as runs with mode and suite
`grounding` and no task rows: the Cost tab counts them, the Evals tab and the
Leaderboard leave them out.

The **Dashboard data** workflow (`.github/workflows/dashboard-data.yml`) runs after
every "Live benchmark", "Live desktop benchmark", "Bench command" or "Grounding eval"
run, cancelled ones included, and writes its reports to the database, using the
`DATABASE_URL` repository secret. Importing a report again replaces it. Run the workflow by hand to import older workflow runs
(`run_ids`) or reports committed at a git ref (`backfill_ref`).

From a terminal:

```bash
pip install "psycopg[binary]"
export DATABASE_URL=postgres://...
python -m evals.dashboard import .var/bench/full-20261024T012945Z.json
python -m evals.dashboard import path/to/artifacts --backend json --flag "why it is not comparable"
```

Flagged runs stay in the data but are hidden unless "Flagged runs" is on. A run is
flagged on its own when it stopped at its cost cap, ended early (`live_bench` saves
its report after every task, marked incomplete until the run finishes, so a crash,
timeout or cancelled workflow keeps what it spent), ran only some tasks
(`--only`/`--category`, saved as `partial-*.json`; `--skip-partial` leaves those out),
or made model calls with no price (a model missing from the price table, which would
count as $0).

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
