# Benchmarks and eval tools

Benchmarks cost real credits: run only when asked. Paid runs are public benchmarks only (KAN-184): **OSWorld**
(`evals/osworld.py`, workflow `osworld-bench.yml`) and **ScreenSpot grounding** (`evals/grounding.py`, `grounding.yml`),
both on Gemini (gemini-3.5-flash); OSWorld can also run Claude Sonnet 5.5 and GPT computer use (`--model`). Operon's own suites (browser, desktop, hybrid) no longer run on paid runs. The free
checks ($0: offline evals, replay, browser/desktop/windows e2e, unit tests) are unchanged.

```powershell
# Name runs by cell when asking Vickey: "Gemini · OSWorld · 25 tasks · once, about $X" or "Gemini · Grounding · zoom · about $X".
# Stops once a run has spent its cost cap ($MAX_RUN_COST_USD / --max-cost-usd; 0 = no cap). The cap is checked between
# tasks, so a run can overshoot it by one task's spend (Claude/GPT: about $0.12-0.24 a task; use cap=6 for 25 tasks).
# OSWorld-Verified subset (25 fixed tasks, scored by OSWorld; needs the Linux VM, docs/osworld-runner.md):
.venv\Scripts\python -m evals.osworld [--limit 5] [--only <task id prefix>] [--max-cost-usd 3] [--model claude-sonnet-5-5|openai-computer-use] [--list]
# A task that errors is recorded (status/stop_reason/error per result) and the run goes on; --job-minutes prints the CI job timeout for the chosen tasks (30 + 35 min a task).
# --model (empty = Gemini) runs Claude or GPT through the KAN-168 model adapter (OPERON_DESKTOP_BACKEND=model); they spend Anthropic/OpenAI credits: need Vickey's cost OK (about $3-6 per 25 tasks per model). Batch cell: osworld model=<profile>

# Grounding eval: ScreenSpot-v2/Pro samples, one model call each through the real backend, click scored vs the box
.venv\Scripts\python -m evals.grounding [--per-set 100] [--set v2-web] [--list] [--zoom off|dense|always]

# Local only, not paid runs (KAN-184): Operon's own suites below no longer start from any workflow (a PR comment,
# a batch cell or by hand in Actions). The modules and task files stay until the cleanup PR; running them yourself
# on your own keys still costs real credits, so only do it when asked.
# Live safety suite: prompt-injection and overreach tasks on a local site, graded from its request log
.venv\Scripts\python -m evals.live_bench --suite safety

# Live apps suite: 11 tasks on local fake shop, mail, account and forms, graded from their state (/state)
.venv\Scripts\python -m evals.live_bench --suite apps
# Long suite: 21 multi-app tasks from the /apps/ workspace page (email -> act -> reply), 20-50 steps each
.venv\Scripts\python -m evals.live_bench --suite long
.venv\Scripts\python -m evals.local_apps   # serve the local site on :8765 to look at it
# Perception experiments (KAN-115): the settle, note, planner, expect cell words (batch cells refuse them now) turned on
# OPERON_SETTLE_DETECT / OPERON_CHANGE_NOTE / OPERON_PLANNER / OPERON_EXPECT_CHECK for that cell, e.g. "browser thin dynamic x3 settle note".
# Dynamic suite (KAN-114): 7 tasks on screens that move (toasts, spinners, errors that vanish, late results),
# graded from /state (evals/dynamic_apps.py, dynamic_tasks.py). Measures OPERON_SETTLE_DETECT / OPERON_CHANGE_NOTE (A/B is KAN-115).
# In no season. Browser only; run it by hand:
.venv\Scripts\python -m evals.live_bench --suite dynamic
.venv\Scripts\python -m evals.dynamic_apps   # serve the local site with the dynamic app (/apps/dynamic/)

# Seasons (frozen task sets for tracking Operon; evals/seasons/, KAN-58): freeze, check, retire; --season on the benches
.venv\Scripts\python -m evals.leaderboard.season check 1
.venv\Scripts\python -m evals.live_bench --season 1               # Browser core; refuses if any task changed
.venv\Scripts\python -m evals.live_bench --suite long --season 1   # held-out from $OPERON_SEASON_PRIVATE
# Season 2 (KAN-78): harder core sets (evals/season2/, CORE2; new Finance/People/Rooms apps), run as core
.venv\Scripts\python -m evals.live_bench --season 2               # also live_desktop_bench / live_hybrid_bench

# Judge check: compare the benchmark judge with hand labels (judge calls only, no agent runs)
.venv\Scripts\python -m evals.judge_check template .var\bench\full-<stamp>.json > labels.jsonl
.venv\Scripts\python -m evals.judge_check run labels.jsonl [--repeat 3]

# Live desktop benchmark: Desktop core (10 tasks) by default, --suite full for all 20 (safety|long for Desktop Safety/Long); drives the real mouse and keyboard
# (no longer a paid CI run, KAN-184; docs/azure-desktop-runner.md)
.venv\Scripts\python -m evals.live_desktop_bench [--suite smoke|full|safety|long] [--keep-screens]
# Live Hybrid benchmark (KAN-73): 10 tasks that each use a desktop app and Edge on the local site, which it serves
# on 127.0.0.1 on the same machine; graded from the site's /state and files in Documents\OperonFiles (evals/hybrid_tasks.py)
.venv\Scripts\python -m evals.live_hybrid_bench [--suite core|smoke|full] [--keep-screens]

# Paid runs from a PR comment (owners/collaborators; runs on the PR head, never on push):
#   /bench grounding [zoom|zoom=always]   (zoom: OPERON_ZOOM=dense, KAN-82)
#   /bench batch  (then one cell per line): the day's approved queue in one run, caps summed up front
#     (KAN-105, benchmark.yml, evals/bench_batch.py). Cells:
#       grounding [zoom|zoom=always] [per_set=<1..500>] [pr=<n>|ref=<ref>] [cap=<USD>]
#       osworld [limit=<0..25>] [only=<task id prefix>] [steps=<1..200>] [model=claude-sonnet-5-5|openai-computer-use] [cap=<USD>]   (default branch, own VM; no pr=/ref=)
#   Own-suite words (browser, desktop, hybrid, core, smoke, ...) are refused, with a KAN-184 message. model= is for osworld cells only.
#   OSWorld by hand: Actions > Run OSWorld test tasks (own VM) > Run workflow (osworld-bench.yml).
# Record and replay (KAN-167, evals/replay.py): --keep-screens also writes <report>/recordings/<task>.json for each
# thin-harness run (Gemini's answers + the actions Operon ran). Copy one into evals/recordings/ and every PR
# replays it through the real loop for $0 (tests/evals/test_replay.py); a changed action fails with the step.
#   python -m evals.replay check [evals/recordings/*.json]
# Repo secrets: GEMINI_API_KEY, DATABASE_URL, BLOB_READ_WRITE_TOKEN, SLACK_WEBHOOK_URL,
# AZURE_CREDENTIALS (fallback); variables AZURE_CLIENT_ID, AZURE_TENANT_ID, AZURE_SUBSCRIPTION_ID (OIDC),
# optional AZURE_VM_RESOURCE_GROUP, AZURE_VM_NAME. Keep Vercel Authentication (All Deployments) on for the dashboard.

# Eval dashboard (dashboard/, Vercel + Neon Postgres): write reports to the DB, then serve locally
# (the import needs `pip install -e ".[dashboard]"` or `pip install "psycopg[binary]"`; the "Save run results to dashboard" workflow does it after each benchmark)
$env:DATABASE_URL = "postgres://..."; .venv\Scripts\python -m evals.dashboard import .var\bench\full-<stamp>.json
cd dashboard; npm install; npm run dev   # page + API on :3000, reads $env:DATABASE_URL
```

## Run videos (KAN-185)

Every benchmark workflow run makes one short captioned MP4 per task (`--videos` on
`evals.live_bench` and `evals.osworld`): the screenshots a run already saved, stitched by
`evals/run_video.py` with the task and "Step N/M: action" as captions and the verdict on the held
last frame. $0: no model calls. Held-out season tasks never get a video. Try one by hand:
`python -m evals.run_video --run .var/runs/<run_id> --out run.mp4` (or `--all .var/runs --out-dir videos`).
With `ffmpeg` on PATH the file is H.264 (plays in browsers, Windows, Loom); without it, mp4v.

- Workflow artifact `run-videos-<cell suffix>` (`run-videos-osworld`, or `run-videos-osworld-cell2` in a batch), kept 14 days.
- GitHub prerelease `videos-<bench|osworld>-<run id>-<attempt>` (job "Keep the run videos"), kept
  forever, asset names prefixed by the artifact name.
- Loom, by hand: ask Claude to download the release asset and upload it with the Atlassian
  connector (`createLoomObjectUpload`, then `publishLoomObject`) into the Loom folder "Operon
  benchmarks". Loom Starter holds 25 videos, so put up the failures and one success, and remove
  Loom copies older than about two weeks.
- Captions include the text Operon typed, and release assets are permanent and as visible as the
  repo, so benchmark tasks must only ever use fake data.
