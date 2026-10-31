# Run and benchmark cost

How much an Operon run costs, where it goes, and the levers to cut it.

## Measuring

Every model call writes a `*_usage.json` next to its step in `.var/runs/<run_id>/step_N/`
(`computer_use_usage.json`, `verification_usage.json`, `final_check_usage.json`, ...).
`estimate_usage_cost` (`src/operon/models/usage.py`) prices each call from list prices:

- Anthropic: uncached input, cache writes (1.25x input), cache reads (the model's
  read rate: $0.20/MTok on Opus 5.5) and output (thinking included) each at their own rate.
- Gemini: cached prompt tokens at the cached rate, thinking tokens billed as output.

The live benchmark (`evals.live_bench`, desktop too) sums those files per task and
reports `cost_usd`, token counts and a per-model breakdown for each task, plus
`cost_per_task_usd`, `cost_per_success_usd` (failed tasks' spend charged to the
successes) and the judge's cost in the summary. The eval dashboard stores the same
fields in its database.

Limit: a step that calls the same backend twice (a mid-step re-look) overwrites its
usage file, so a run's total can be slightly low.

## Where the money goes

Measured on the browser smoke suite, 2026-10-24 (Opus 5.5, effort low, full harness,
5/5 success, 1.4 steps per task):

| Item | Per task |
|---|---|
| Opus 5.5 (the acting model) | $0.033 |
| Gemini 3.5 Flash per-step verifier | $0.006 |
| **Agent total** | **$0.040** |
| Benchmark judge (Gemini 3.5 Flash) | $0.004 |

About three quarters of the Opus spend went to cache writes (29.5k tokens written,
15.2k read, 13.3k uncached, 2.0k output). Short runs write the whole prompt and end
before reusing much of it. A full 44-task run comes to roughly $2 (more steps per task
than the smoke suite).

## Levers, cheapest to try first

1. **Run fewer benchmarks.** Runs happen only on a `/bench` comment. Use smoke for a
   quick check and `all` only before a decision.
2. **Cheaper acting model.** `OPERON_BROWSER_CLAUDE_MODEL` / `OPERON_DESKTOP_CLAUDE_MODEL`,
   or `/bench sonnet` / `/bench haiku`. Sonnet 5.5 is half Opus 5.5's price; Haiku 5.5
   about a fortieth. Judge it by cost per success: a model that needs more steps or
   fails more can cost more per finished task.
3. **Cheaper completion check (thin harness).** `OPERON_FINAL_CHECK_MODEL`, or
   `/bench thin check-sonnet`. The check reads one screenshot per claimed finish, so a
   cheaper model saves a little without touching the actions; watch false successes.
4. **Fewer steps.** Each extra turn adds a screenshot and output. Steps per task is
   already on the report; the thin harness and Claude's batched actions keep it low.

Already in place: prompt caching on every Claude turn (top-level `cache_control`; check
`cache_read_tokens` on the report is non-zero), plus a breakpoint on the system prompt
so the tools and system prompt every run shares are written once and read by later
runs within five minutes; effort `low`; screenshots downscaled to a 1280 px long edge.

Not recommended:

- **Trimming old screenshots from the conversation.** History is a cache read at
  $0.20/MTok, and editing it breaks the cache and Opus 5.5's preserved thinking.
- **A 1-hour cache TTL.** Steps are seconds apart, so the 5-minute cache never expires
  mid-run; the 1-hour write costs 2x.
- **Smaller screenshots than 1280 px.** Saves under a cent per task and risks
  targeting accuracy.
