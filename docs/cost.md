# Run and benchmark cost

How much an Operon run costs, where it goes, and the levers to cut it.

## Measuring

Every model call writes a `*_usage.json` next to its step in `.var/runs/<run_id>/step_N/`
(`computer_use_usage.json`, `verification_usage.json`, `final_check_usage.json`, ...).
`estimate_usage_cost` (`src/operon/models/usage.py`) prices each call from list prices:

- Anthropic: uncached input, cache writes (1.25x input), cache reads (the model's
  read rate: $0.20/MTok on Opus 5.5) and output (thinking included) each at their own rate.
- Gemini: cached prompt tokens at the cached rate, thinking tokens billed as output.

The live benchmark (`tests.eval.live_bench`, desktop too) sums those files per task and
reports `cost_usd`, token counts and a per-model breakdown for each task, plus
`cost_per_task_usd`, `cost_per_success_usd` (failed tasks' spend charged to the
successes) and the judge's cost in the summary. The eval dashboard reads the same
report JSON.

Limit: a step that calls the same backend twice (a mid-step re-look) overwrites its
usage file, so a run's total can be slightly low.

## Where the money goes (estimate, to be confirmed by the next benchmark)

For the Claude browser backend (Opus 5.5, effort low, ~1.7 actions per task):

| Item | Approx. per task |
|---|---|
| First turn: prompt, toolset and first screenshot written to cache | ~$0.02 |
| Each later turn: new screenshot written, history read from cache | ~$0.008 |
| Output, thinking included, ~300 tokens per turn at $20/MTok | ~$0.006 per turn |
| Completion check (thin) or Gemini verifier (full) | ~$0.003-0.01 |
| **Total** | **~$0.05-0.07** |

So a full 44-task browser benchmark is roughly $2-3, a smoke run about $0.30. These
numbers come from token arithmetic, not a bill; the first benchmark after this change
replaces them with measured ones.

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
`cache_read_tokens` on the report is non-zero), effort `low`, and screenshots
downscaled to a 1280 px long edge.

Not recommended:

- **Trimming old screenshots from the conversation.** History is a cache read at
  $0.20/MTok, and editing it breaks the cache and Opus 5.5's preserved thinking.
- **A 1-hour cache TTL.** Steps are seconds apart, so the 5-minute cache never expires
  mid-run; the 1-hour write costs 2x.
- **Smaller screenshots than 1280 px.** Saves under a cent per task and risks
  targeting accuracy.
