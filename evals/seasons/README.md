# Seasons

Each `season-<n>.json` here is a frozen season: the tasks Operon's yardstick runs
are scored on, with a SHA-256 of every task's definition (its data, its grader code
and the fixture files it runs on). See `evals/leaderboard/season.py`.

- Run a season: `--season <n>` on `evals.live_bench`, `evals.live_desktop_bench`
  or `evals.grounding`. The run refuses to start when any task changed, appeared
  or went missing.
- Held-out tasks are not in this repo. The manifest only holds an opaque id and
  a hash for each. Their file is read from `$OPERON_SEASON_PRIVATE`, and reports
  give their results as counts only.
- Never edit a manifest by hand (its `season_hash` would stop matching), and
  never edit a task in place once a season includes it. Retire it instead:
  `python -m evals.leaderboard.season retire <n> <suite> <task> "<reason>"`.

Season 1 gets frozen once its harder tasks exist (KAN-65):
`python -m evals.leaderboard.season freeze 1 --suite long --suite desktop --suite grounding --suite apps --suite safety`.
