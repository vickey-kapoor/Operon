# Evaluation

How Operon is measured: offline scenarios, live browser and desktop benchmarks, a safety
suite, a grounding eval, repeated runs for flakiness, and a check of the benchmark's own judge.

## Latest results

As of October 2026.

| Benchmark | Configuration | Result |
|---|---|---|
| Browser, 44 live web tasks | Claude Opus 5.5, thin harness | 41 to 43 of 44 over the last three runs, about 29 s and $0.04 to $0.06 per task |
| Browser, 44 live web tasks | Gemini, full harness | 40/44, about 50 s per task |
| Desktop smoke, 3 Windows tasks | Claude Opus 5.5, thin harness | 3/3, 24 s per task |
| Desktop smoke on the Azure VM | Claude Opus 5.5, thin harness | 3/3 in both runs so far, 32 to 36 s and $0.06 to $0.07 per task |
| Safety, 9 injection and overreach tasks | Claude, thin harness | 6/9, none compromised; the 3 misses were fixture pages, since fixed |
| Judge vs 88 hand labels | Gemini 3.5 Flash judge | 99% accuracy, 100% recall |

The eval dashboard at https://operon-evals.vercel.app has every run.

## Suites

Every live suite spends real model credits, so none of them run on pushes or merges.

| Suite | Command | What it checks |
|---|---|---|
| Offline eval | `python -m evals` | Scripted scenarios through the real loop, no network |
| Live smoke | `python -m evals.live_smoke` | A few real browser tasks against a running API |
| Browser benchmark | `python -m evals.live_bench [--suite smoke\|full] [--category forms] [--only NAME]` | 44 real web tasks (5 in smoke), graded from the final screen |
| Safety suite | `python -m evals.live_bench --suite safety` | 9 prompt-injection and overreach tasks on a local fixture site, graded from the site's request log |
| Local apps suite | `python -m evals.live_bench --suite apps` | 11 tasks on local fake shop, mail, account and form apps, graded from the apps' state |
| Long multi-app suite | `python -m evals.live_bench --suite long` | 10 tasks that chain the local apps from a workspace page (an email asks for an order, a registration or a support request, then a reply with its number or total), 20 to 50 steps each, graded from state |
| Hard subset, repeated | `python -m evals.live_bench --suite hard --repeat 3` | 20 of the harder app, live and safety tasks, each run 3 times: pass@1 with a 95% CI, pass^3, flaky tasks |
| Grounding | `python -m evals.grounding [--per-set 100] [--set v2-web] [--backend computer_use]` | 300 ScreenSpot-v2 and ScreenSpot-Pro samples (one screenshot and instruction each) through Operon's own backend; scores whether the first click lands in the target's box |
| Desktop benchmark | `python -m evals.live_desktop_bench [--suite smoke\|full]` | 10 tasks on stock Windows apps (3 in smoke); **drives the real mouse and keyboard** |
| Judge check | `python -m evals.judge_check template <report.json> > labels.jsonl`, then `judge_check run labels.jsonl [--repeat 3] [--save]` | Compares the benchmark judge with hand labels; `--save` records it for the dashboard (needs `DATABASE_URL`) |

The browser benchmark grades each task with a Gemini judge that reads the final screenshot
against a per-task success condition. The judge check measured it at 99% accuracy and 100%
recall on 88 hand labels; see [reports/safety-and-judge-check.md](reports/safety-and-judge-check.md)
for that and the first safety results, and
[reports/local-apps.md](reports/local-apps.md) for how the local apps are graded.

`--repeat N` works with any live suite: it runs the whole task list N times over and adds
pass@1 (the mean success rate, with a 95% interval that treats tasks, not attempts, as the
samples), pass^N (every attempt succeeded), pass@N (at least one did) and the list of flaky
tasks to the report. The cost cap is multiplied by N. The `hard` suite
([evals/hard_tasks.py](../evals/hard_tasks.py)) is the set meant for it: all 11
local app tasks, the 5 longest or past-failing live tasks and 4 safety tasks. A single run
cannot tell a 40/44 backend from a 42/44 one; a repeated hard run shows how much of that
gap is noise.

The grounding eval isolates the "clicked the wrong spot" failure. Each sample goes through
the same backend, prompt, screenshot downscaling and coordinate scaling as a live step, and
only the first proposed click is scored. The sample is fixed (stratified by text and icon
targets, and by app group for Pro) and the datasets are pinned to a revision, so runs are
comparable. Claude runs web samples through the browser backend and the rest through the
desktop backend; Gemini uses its browser Computer Use backend for all. Expect Pro to score
far lower than v2: its screens are up to 3840 px wide and are downscaled to 1280 px before
the model sees them.

## Cost

Each model call records its token usage next to its step, and the benchmarks report cost per
task, cost per success and a per-model breakdown. A run stops once it has spent
`--max-cost-usd` (`MAX_RUN_COST_USD`, default **$5** per backend; `0` = no cap) and exits
with code 3. Prices, where the money goes and the levers to cut it are in [cost.md](cost.md).

## Running benchmarks from a pull request

Repository owners, members and collaborators can comment on a PR to benchmark its head
commit (PRs from forks are refused):

```text
/bench [browser|desktop] [thin|full] [smoke|all|safety|apps|hard] [x2|x3|x5] [opus|sonnet|haiku] [check-sonnet|check-haiku] [azure] [zoom] [model=<profile>]
/bench grounding [opus|sonnet|haiku|gemini] [zoom] [model=<profile>]
```

Defaults: browser, thin harness, smoke suite, Claude Opus 5.5. Examples:

| Comment | Runs |
|---|---|
| `/bench` | Browser smoke (5 tasks) |
| `/bench all` | Full browser benchmark (44 tasks) |
| `/bench safety` | Safety suite (browser only) |
| `/bench apps` | Local apps suite (browser only) |
| `/bench long` | Long multi-app suite (browser only) |
| `/bench hard x3` | Hard subset, every task 3 times (about $5.50 with Opus) |
| `/bench grounding` | Grounding eval, 300 samples on Claude Opus 5.5 |
| `/bench grounding gemini` | Grounding eval on Gemini |
| `/bench grounding zoom` | Grounding eval on Claude with zoom on (`OPERON_CLAUDE_ZOOM`) |
| `/bench desktop azure all zoom` | Full desktop benchmark on the Azure VM with Claude's zoom on (desktop and grounding only) |
| `/bench desktop` | Desktop smoke on GitHub's hosted Windows runner |
| `/bench desktop all azure` | Full desktop benchmark on the Azure VM |
| `/bench sonnet check-haiku` | Sonnet acting, Haiku for the completion check |
| `/bench apps model=qwen3-vl-32b` | Apps suite on a model registry profile (see below) |

The **Live benchmark**, **Live desktop benchmark** and **Grounding eval** workflows can also be started by hand,
with inputs for backends, models, harness profile, suite and cost cap (`max_cost_usd`).

## Comparing models (experimental)

The cross-model benchmark runs any model on the same tasks, executor and graders. Models are
profiles in [`evals/models.yaml`](../evals/models.yaml); the server runs one with
`OPERON_BROWSER_BACKEND=model` (or `OPERON_DESKTOP_BACKEND=model`) and
`OPERON_MODEL_PROFILE=<profile>`. Every such run uses the thin harness and ends on the
model's own "done": no second model checks the claim, and the graders decide success, so
false passes count the same way for every model.

| Adapter | Models | How Operon talks to them |
|---|---|---|
| `claude_native` | Claude Opus, Sonnet, Haiku | Claude's computer-use toolset (the existing backend) |
| `gemini_native` | Gemini computer use (browser only) | Gemini Computer Use (the existing backend) |
| `openai_native` | OpenAI computer use | Responses API with the `computer` tool, chained by `previous_response_id` |
| `openai_compat` | Open-weight models on vLLM, SGLang or a hosted endpoint | `/v1/chat/completions`, with a parser per model family: `neutral_tools` (tool calling, e.g. Qwen3-VL), `ui_tars` (UI-TARS text actions), `opencua` (OpenCUA pyautogui code) |

Each profile sets its coordinate space (pixels of the sent image, 0 to 1000, or 0 to 1), how
the screenshot is resized (a long-edge cap, or Qwen-style `smart` resizing to multiples of
28 or 32) and how many screenshots stay in the history. A reply the parser cannot turn into
an action is recorded and answered with a short wait, and the model is told why; the report
counts these as the invalid-action rate.

```powershell
# Server: a self-hosted model behind vLLM's OpenAI-compatible server
$env:OPERON_BROWSER_BACKEND = "model"; $env:OPERON_MODEL_PROFILE = "qwen3-vl-32b"
$env:OPENWEIGHT_BASE_URL = "https://<gpu-vm>:8000/v1"; $env:OPENWEIGHT_API_KEY = "<vllm --api-key>"
# Benchmark: --model checks the profile and records it, the invalid-action rate and GPU cost
.venv\Scripts\python -m evals.live_bench --suite apps --model qwen3-vl-32b
.venv\Scripts\python -m evals.grounding --backend model --model qwen3-vl-32b
```

GPU cost is the tasks' wall time at the profile's `gpu_hour_usd` (a lower bound: the VM also
bills while it starts and idles). The dashboard's **Leaderboard** tab shows each model's
latest run per suite, a cost-against-success chart and per-model totals. Every paid run,
including a smoke run of a new model, waits for an approved cost estimate.

## Eval dashboard

Results are published at **https://operon-evals.vercel.app**: pass rate, seconds per task and
step, false passes, cost, and per-task outcomes by configuration. After every benchmark
workflow, the **Dashboard data** workflow writes the reports to a Neon Postgres database,
uploads screenshots and traces to a private Vercel Blob store (the dashboard shows them for
tasks that did not pass), and posts a Slack alert when a run's pass rate drops 5 points,
its cost per task rises 25% against the previous comparable run, or it hits its cost cap.
Run the workflow by hand to import older runs. The Cost tab shows this month's spend against
the `MONTHLY_BUDGET_USD` budget (Vercel env) and spend by month (UTC), counting every
benchmark run (incomplete and partial ones too), grounding evals and judge checks. See
[dashboard/README.md](../dashboard/README.md).

## Azure desktop runner

The desktop benchmark runs on GitHub's hosted `windows-latest` runner or, with the workflow's
`runner: azure` input (or `/bench desktop azure`), on an on-demand Windows Server 2022 VM
that the workflow starts before the run and deallocates after. Setup is in
[azure-desktop-runner.md](azure-desktop-runner.md).

## Repository secrets and variables

| Name | Kind | Used by |
|---|---|---|
| `ANTHROPIC_API_KEY` | secret | Claude backends in the live benchmarks |
| `GEMINI_API_KEY` | secret | Gemini backends and the benchmark judge |
| `OPENAI_API_KEY` | secret | OpenAI computer use (`openai_native` profiles); optional |
| `OPENWEIGHT_BASE_URL`, `OPENWEIGHT_API_KEY` | secrets | The self-hosted model server (`openai_compat` profiles); optional |
| `DATABASE_URL` | secret | Dashboard data import (also set in the Vercel project) |
| `BLOB_READ_WRITE_TOKEN` | secret | Screenshot and trace upload (also set in the Vercel project); optional |
| `SLACK_WEBHOOK_URL` | secret | Regression alerts; optional |
| `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID` | variables | Starting and stopping the Azure VM via OIDC (no stored secret) |
| `AZURE_CREDENTIALS` | secret | Older fallback for the Azure VM while `AZURE_CLIENT_ID` is unset |
| `AZURE_VM_RESOURCE_GROUP`, `AZURE_VM_NAME` | variables | Optional: override the Azure resource group and VM names |
