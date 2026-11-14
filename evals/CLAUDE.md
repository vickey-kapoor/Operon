# Benchmarks and eval tools

Benchmarks cost real credits: run only when asked. Every run uses Gemini (gemini-3.5-flash),
Operon's only model for now; other models come back later.

```powershell
# Benchmarks are grouped by category (Browser, Desktop, Hybrid) x harness (thin, full): a 3x2 grid per
# model (KAN-72, evals/categories.py). Default suite is the category's core set (~10 state-graded tasks:
# short, long, safety); the report names its cell and breaks the score down in summary.by_tag.
# Name runs by cell when asking Vickey: "Gemini · Browser · Thin · once, about $X". Saves to .var/bench/
# Stops once a run has spent --max-cost-usd / $MAX_RUN_COST_USD (default 3; 0 = no cap)
.venv\Scripts\python -m evals.live_bench                       # Browser core
.venv\Scripts\python -m evals.live_bench --suite full [--category forms] [--only NAME]   # 44 live web tasks, judge-graded

# Hard subset repeated 3 times: pass@1 with a CI, pass^3 and flaky tasks (--repeat works on any suite)
.venv\Scripts\python -m evals.live_bench --suite hard --repeat 3

# Grounding eval: ScreenSpot-v2/Pro samples, one model call each through the real backend, click scored vs the box
.venv\Scripts\python -m evals.grounding [--per-set 100] [--set v2-web] [--list]

# Live safety suite: prompt-injection and overreach tasks on a local site, graded from its request log
.venv\Scripts\python -m evals.live_bench --suite safety

# Live apps suite: 11 tasks on local fake shop, mail, account and forms, graded from their state (/state)
.venv\Scripts\python -m evals.live_bench --suite apps
# Long suite: 21 multi-app tasks from the /apps/ workspace page (email -> act -> reply), 20-50 steps each
.venv\Scripts\python -m evals.live_bench --suite long
.venv\Scripts\python -m evals.local_apps   # serve the local site on :8765 to look at it

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
# (in CI: hosted windows-latest, or our Azure VM via `/bench desktop azure`; docs/azure-desktop-runner.md)
.venv\Scripts\python -m evals.live_desktop_bench [--suite smoke|full|safety|long] [--keep-screens]
# Live Hybrid benchmark (KAN-73): 10 tasks that each use a desktop app and Edge on the local site, which it serves
# on 127.0.0.1 on the same machine; graded from the site's /state and files in Documents\OperonFiles (evals/hybrid_tasks.py)
.venv\Scripts\python -m evals.live_hybrid_bench [--suite core|smoke|full] [--keep-screens]

# Live benchmarks from a PR comment (owners/collaborators; runs on the PR head, never on push):
#   /bench [browser|desktop|hybrid] [thin|full] [core|smoke|all|safety|apps|hard|long] [x2|x3|x5] [same-critic] [azure] [season=<n>]
#   (category, then harness: one grid cell; core and full are the defaults; hybrid runs on the desktop workflow)
#   /bench grounding
#   /bench batch  (then one cell per line, e.g. "desktop thin season=2 pr=242 cap=1.5"): the day's approved
#   queue in one run, one Azure VM start, caps summed up front (KAN-105, bench-batch.yml, evals/bench_batch.py)
# Repo secrets: GEMINI_API_KEY, DATABASE_URL, BLOB_READ_WRITE_TOKEN, SLACK_WEBHOOK_URL,
# AZURE_CREDENTIALS (fallback); variables AZURE_CLIENT_ID, AZURE_TENANT_ID, AZURE_SUBSCRIPTION_ID (OIDC),
# optional AZURE_VM_RESOURCE_GROUP, AZURE_VM_NAME. Keep Vercel Authentication (All Deployments) on for the dashboard.

# Eval dashboard (dashboard/, Vercel + Neon Postgres): write reports to the DB, then serve locally
# (the import needs `pip install "psycopg[binary]"`; the "Send results to dashboard" workflow does it after each benchmark)
$env:DATABASE_URL = "postgres://..."; .venv\Scripts\python -m evals.dashboard import .var\bench\full-<stamp>.json
cd dashboard; npm install; npm run dev   # page + API on :3000, reads $env:DATABASE_URL
```
