# Operon cross-model computer-use benchmark: design proposal

Nov 1, 2026 · @Vickey Kapoor

## Summary

Operon already has the hard parts of a benchmark: one control loop for browser and desktop, 89 state- or judge-graded tasks, a validated judge, cost tracking, and a dashboard. What it lacks is a way to plug in any model. The proposal is one new adapter layer that turns any model's output into Operon's existing `PolicyDecision`, so every model runs the same tasks through the same executor and the same graders.

Three adapter kinds cover the field: native computer-use APIs (Claude and Gemini today, OpenAI next), a generic tool-calling adapter for any OpenAI-compatible endpoint, and small text parsers for open models trained on their own action format (UI-TARS, OpenCUA). Open-weight models are served with vLLM on a GPU VM that the benchmark workflow starts and stops, the same way the Azure desktop runner works.

Nothing gets built or run until Vickey picks the open decisions at the end. A first sweep of 7 models over the core suites should cost roughly $40 to $60 including GPU time (estimate, not measured).

## Goals and non-goals

Goals:

- Run any closed or open-weight model on the same Operon tasks, browser and desktop, with one config line per model.
- Score every model the same way: same executor, same screen size, same step and time limits, same graders.
- Report success, reliability, speed, cost and safety side by side, on a leaderboard tab in the existing dashboard.
- Keep each model's own best prompt and action format, so a model is not penalised for Operon's Claude-shaped defaults.

Non-goals for now:

- Fine-tuning or training any model.
- Porting OSWorld or Online-Mind2Web in the first build (listed as a later phase).
- Changing the default production backend. Claude stays the default; new adapters ship behind flags, off by default.

## Model lineup

The recommended first sweep is 7 models: 4 closed and 3 open-weight, chosen to span sizes and action formats. Exact model versions get confirmed against each vendor's docs when the adapter is built; the names below are from memory.

| Model | Weights | How Operon talks to it | Adapter | Status in Operon |
| --- | --- | --- | --- | --- |
| Claude Opus 5.5 | Closed | Anthropic computer-use tool | Native (exists) | Default, measured |
| Claude Sonnet 5.5 or Haiku 5.5 | Closed | Same tool, cheaper model | Native (exists) | `/bench sonnet`, `/bench haiku` work today |
| Gemini computer use | Closed | Gemini Computer Use API | Native (exists) | Measured, browser only |
| OpenAI computer-use model | Closed | Responses API computer tool | Native (new) | Needs an OPENAI\_API\_KEY secret |
| Qwen-VL, 32B class (Qwen3-VL) | Open | vLLM, OpenAI-compatible chat with tools | Generic tool-calling (new) | Needs a GPU endpoint |
| UI-TARS 7B class | Open | vLLM, text actions with 0 to 1000 coordinates | Text parser (new) | Needs a GPU endpoint |
| OpenCUA 32B class | Open | vLLM, pyautogui-style code actions | Text parser (new) | Needs a GPU endpoint |

Later candidates, each a config line once the generic adapter exists: GLM-4.xV, Holo, larger Qwen MoE models, and plain chat models with no computer-use training (GPT, Gemini Flash) as a "generalist" baseline through the generic tool-calling adapter.

## Model adapter interface

Every adapter is an `AgentBackend` (the interface Claude and Gemini already implement in `src/operon/agent/backends/base.py`), so the loop, executors, graders and cost reporting need no changes. The new code sits in one place: a `ModelAdapter` that splits a backend into three swappable parts.

1. **Prompt and history builder.** Builds the request from the task, the latest screenshot and the last N screenshots and actions. N is per model, because small open models lose track with long histories and long histories cost closed models money.
2. **Transport.** One of: Anthropic, Gemini, OpenAI Responses, or a generic OpenAI-compatible `/v1/chat/completions` client pointed at any base URL (vLLM, SGLang, a hosted provider).
3. **Action parser.** Turns the reply into Operon actions: native tool calls, a neutral tool schema (`click`, `double_click`, `type`, `key`, `scroll`, `drag`, `navigate`, `wait`, `done`), UI-TARS text actions, or OpenCUA pyautogui lines. Anything it can't parse becomes a recorded invalid action, not a crash.

Coordinates are the main trap between models. Each model profile declares its coordinate space (pixels of the sent image, 0 to 1000 relative, or 0 to 1 normalised) and its image rule (long edge cap, or Qwen-style resize to multiples of 28). The adapter converts back to screen pixels, the same way the Claude backend scales from its 1280 px image today.

Models are declared in one registry file, `evals/models.yaml`, for example:

```yaml
qwen3-vl-32b:
  adapter: openai_compat
  parser: neutral_tools
  base_url_env: OPENWEIGHT_BASE_URL
  model: Qwen/Qwen3-VL-32B-Instruct
  coords: relative_1000
  image: {resize: multiple_of_28, max_pixels: 1310720}
  history_screens: 3
  pricing: {gpu_hour_usd: 2.50}
```

The API selects it with `OPERON_BROWSER_BACKEND=model` plus `OPERON_MODEL_PROFILE=qwen3-vl-32b` (same for desktop), and the eval tools get `--model qwen3-vl-32b`. Both are new flags, off by default.

Harness rule for fairness: benchmark runs use the thin harness for every model, so the loop never rewrites a model's action. The thin harness's completion check currently calls Claude, which would favour Claude. In benchmark mode it is replaced by the model's own `done`, and the independent graders (state checks and the Gemini judge) decide success. That also makes false successes a clean metric.

## Task suites

The leaderboard should rest on the suites that grade from state, because live websites change, bot-block and rate-limit, which adds noise unrelated to the model. Live web stays in as a secondary score.

| Suite | Tasks | Graded by | Role in the leaderboard |
| --- | --- | --- | --- |
| Apps (local shop, mail, account, forms) | 11 | App state via `/state` | Core, deterministic |
| Safety (prompt injection, overreach) | 9 | Local request log | Core, reported separately as compromised count |
| Desktop (stock Windows apps, Azure VM) | 10 | Judge on final screen | Core |
| Grounding (ScreenSpot v2 and Pro) | 100 per set | Click inside target box | Core, one call per sample, cheapest signal |
| Live web | 44 | Gemini judge on final screen | Secondary, noisier |
| Hard subset x3 | 20 | Mixed | Reliability (pass^3, flaky tasks) |

Two additions are worth it before the first sweep:

- **Longer app tasks.** The apps suite averages a few steps, and Claude already scores 11/11, so it can't separate strong models. Add about 10 multi-page workflows on the same local apps (for example, find an order in mail, refund it in the shop, reply with the refund id), 10 to 25 steps each.
- **Harder desktop tasks.** Claude went 10/10 on the current 10 tasks, so add about 5 harder ones (multi-app, file dialogs, settings) so the suite separates models.

A later phase can add OSWorld-Verified and Online-Mind2Web so results compare with published numbers, as the expansion plan suggested.

## Metrics and reporting

Every run reports the same columns, most of which `live_bench` already writes:

| Metric | What it measures | Exists today? |
| --- | --- | --- |
| Success rate with 95% CI | Pass@1 per suite | Yes (CI added with `--repeat`) |
| pass^3 and flaky tasks | Reliability over 3 repeats | Yes |
| False successes | Model said done, grader says no | Yes |
| Compromised (safety) | Followed an injected instruction | Yes |
| Steps, s/task, s/step | Efficiency and latency | Yes |
| Cost per task and per success | API tokens, or GPU-hours for self-hosted | Tokens yes, GPU-hours new |
| Invalid action rate | Replies the parser could not turn into an action | New |
| Grounding accuracy | Click inside target box | Yes |

Fairness controls fixed for every model: same 1280x800 browser viewport and same desktop VM image, same max steps and timeout per task, same judge model, temperature 0 where the API allows it, and each model's own recommended prompt. The report records model id, adapter, prompt hash and serving settings (quantisation, vLLM version) so a run can be reproduced.

Dashboard: a new Leaderboard tab with a models by suites matrix, a cost against success scatter that shows the cheapest model at each success level, and per-model drill-down into the existing run and screenshot views. Rows go into the existing Neon tables, with a `model_profile` column added.

## Hosting open-weight models

The recommendation is one GPU VM running vLLM's OpenAI-compatible server, started by the benchmark workflow and deallocated when it ends, like the Azure desktop runner. The browser runner (GitHub-hosted Ubuntu) and the desktop runner (Azure VM) both call it over HTTPS.

| Model size | GPU needed (bf16, or FP8 where noted) | Rough on-demand price | Time per core sweep |
| --- | --- | --- | --- |
| 7B to 8B (UI-TARS 7B class) | 1x 24 GB (L4 or A10G) | $0.70 to $1.20 per hour | About 1 to 1.5 hours |
| 32B (Qwen3-VL 32B, OpenCUA 32B) | 1x 80 GB (A100 or H100), FP8 on one H100 | $2.50 to $4 per hour | About 1.5 to 2 hours |
| 72B dense | 2x 80 GB | $6 to $8 per hour | About 2 hours |
| 200B+ MoE | 4 to 8x 80 GB | $15 to $30 per hour | Later, if needed |

Prices are typical public cloud rates from memory, not quotes; Azure GPU quota usually needs a request that takes a day or more.

What Vickey sets up once:

- A GPU VM (or one per size) with vLLM, the model weights cached on its disk, and `--api-key` set so the endpoint isn't open.
- Repo secrets `OPENWEIGHT_BASE_URL` and `OPENWEIGHT_API_KEY`, plus the VM name as a variable for start and stop.
- I'd write the setup script and a doc, as with `docs/azure-desktop-runner.md`.

Alternative: a hosted inference provider (Together, Fireworks, OpenRouter and similar) for models it already serves. It is pay per token with no GPU ops, but it doesn't offer every computer-use model (UI-TARS and OpenCUA are often missing) and serving settings vary. A mix works too: hosted for Qwen, self-hosted for the rest.

## Cost estimate

One core sweep of 7 models, each run once, should cost about $40 to $60. A "core sweep" per model is: apps (11 now plus about 10 longer tasks), safety (9), desktop (about 15), grounding (300 samples) and live web (44). Claude figures scale from measured runs; the rest are guesses until the first run.

| Model | Basis | Estimated cost per core sweep |
| --- | --- | --- |
| Claude Opus 5.5 | Measured: $0.063 per live web task, $1.06 to $1.44 per apps run, $0.29 safety, $0.055 per desktop task, about $3 grounding; longer tasks guessed at $0.40 each | About $12 |
| Claude Sonnet or Haiku | Opus usage at its lower token price | About $1 to $6 |
| Gemini computer use | Earlier runs used about 2x Claude's steps at lower token prices | About $5 (guess) |
| OpenAI computer use | Not measured | About $5 to $10 (guess) |
| 3 open-weight models | 1.5 to 2 GPU-hours each at $1 to $4 per hour | About $5 to $15 total |
| Gemini judge | About $0.004 per graded task | About $2 to $3 total |

The hard subset repeated 3 times adds about $10 per Opus-priced model (the last one cost $10.13), so I'd run repeats only on the top 3 models after the first sweep. Every run keeps the existing $5 per-leg cap (`MAX_RUN_COST_USD`) and Slack alert.

## Build plan

Six PRs, each small enough to review on its own. PRs 1 to 4 cost nothing to run; every paid run waits for Vickey's OK.

1. **Adapter layer.** `ModelAdapter`, the model registry, the generic OpenAI-compatible transport, the neutral tool parser, coordinate and image conversion, and benchmark-mode completion (model's own `done`). Offline tests use recorded replies from each model family, so CI needs no GPU or key.
2. **Open-model parsers.** UI-TARS and OpenCUA action parsers, with tests on real reply samples from their model cards.
3. **OpenAI computer-use backend.** Native adapter for the Responses API computer tool, behind a flag.
4. **Eval plumbing.** `--model <profile>` in `live_bench`, `live_desktop_bench` and `grounding`; GPU-hour cost; invalid-action rate; `/bench ... model=<profile>` in the comment trigger; `model_profile` in Neon; Leaderboard tab.
5. **Hosting.** vLLM setup script, start and stop jobs in the workflows, and a setup doc. Then a $1 to $2 smoke run per open model to check parsing and coordinates.
6. **Longer tasks and first sweep.** About 10 long app tasks and 5 harder desktop tasks, then the core sweep and a results write-up in `docs/reports/`.

Phase 2, later and optional: OSWorld-Verified and Online-Mind2Web runners for comparison with published numbers.

## Open decisions for Vickey

| # | Decision | Recommendation |
| --- | --- | --- |
| 1 | Which models go in the first sweep | The 7 in the lineup table |
| 2 | Where open models run: your own GPU VM, a hosted provider, or both | One GPU VM with vLLM on whichever cloud has GPU quota for you; tell me which cloud |
| 3 | Harness for the benchmark: thin harness with each model's own `done` for everyone, or each model's best setup | Thin plus own `done` for all, so the comparison is fair |
| 4 | Budget for the first sweep | Up to $60, repeats only on the top 3 afterwards |
| 5 | Add an OPENAI\_API\_KEY repo secret | Yes, if OpenAI is in the sweep |
| 6 | Results private or public | Private dashboard first, public page once the numbers are stable |
| 7 | OSWorld-Verified and Online-Mind2Web now or later | Later, after the first sweep |

Once these are picked, I start with PR 1, which needs no secrets and no paid runs.
