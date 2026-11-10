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

Season 1 (KAN-65, core sets added in KAN-72, Hybrid in KAN-73): browser-core,
desktop-core and hybrid-core (the category core sets in `evals/categories.py`, 10 working
tasks each; these are scored), long (21 working + held-out), desktop (14 working +
held-out), hybrid (10 working + 4 held-out), grounding, apps and safety. Frozen with
`python -m evals.leaderboard.season freeze 1 --suite browser-core --suite desktop-core --suite hybrid-core --suite long --suite desktop --suite hybrid --suite grounding --suite apps --suite safety`
and `$OPERON_SEASON_PRIVATE` set. In CI the held-out file comes from the
`OPERON_SEASON_PRIVATE_PY` secret (base64).
