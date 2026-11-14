# Run and benchmark cost

How much an Operon run costs, where it goes, and the levers to cut it. Operon runs
Gemini only, so every figure here is Gemini's.

## Measuring

Every model call writes a `*_usage.json` next to its step in `.var/runs/<run_id>/step_N/`
(`~/.operon/runs/<run_id>/` when installed with pip)
(`computer_use_usage.json`, `verification_usage.json`, ...). `estimate_usage_cost`
(`src/operon/models/usage.py`) prices each call from Gemini's list prices: cached prompt
tokens at the cached rate, the rest of the prompt at the input rate, and output with the
thinking tokens (which Gemini bills as output).

The live benchmark (`evals.live_bench`, desktop and hybrid too) sums those files per task and
reports `cost_usd`, token counts and a per-model breakdown for each task, plus
`cost_per_task_usd`, `cost_per_success_usd` (failed tasks' spend charged to the
successes) and the judge's cost in the summary. The eval dashboard stores the same
fields in its database.

Limit: a step that calls the same backend twice (a mid-step re-look) overwrites its
usage file, so a run's total can be slightly low.

## Where the money goes

On the Season 1 core sets, Gemini 3.5 Flash costs about $0.03 to $0.14 per task: one
10-task cell ran $0.30 to $1.40, depending on the category and the harness
([Season 1 grid](reports/season-1-grid.md)). Hybrid tasks are the longest and cost the
most. The full harness adds a per-step Gemini critic call; the thin harness has none.

## Levers, cheapest to try first

1. **Run fewer benchmarks.** Runs happen only on a `/bench` comment. Use smoke for a
   quick check and `all` only before a decision.
2. **Thin harness.** No per-step critic call. Compare both harnesses on the grid before
   choosing: the critic can save failed tasks, which lowers cost per success.
3. **Fewer steps.** Each extra turn adds a screenshot and output. Steps per task is
   already on the report.
4. **Thinking level.** `OPERON_*_THINKING` (`low` by default); `minimal` cuts output
   tokens. Benchmark before switching, since it can cost accuracy.

Already in place: thinking `low` for the agent and the critic; screenshots downscaled before
they are sent.
