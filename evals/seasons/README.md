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

Season 1 (KAN-65, core sets added in KAN-72, Desktop Long and Safety in KAN-74, Hybrid in
KAN-73): browser-core, desktop-core and hybrid-core (the category core sets in
`evals/categories.py`, 10 working tasks each; these are scored), long (21 working +
held-out), desktop (20 working + held-out), hybrid (10 working + 4 held-out), grounding,
apps and safety. Frozen with
`python -m evals.leaderboard.season freeze 1 --suite browser-core --suite desktop-core --suite hybrid-core --suite long --suite desktop --suite hybrid --suite grounding --suite apps --suite safety`
and `$OPERON_SEASON_PRIVATE` set. In CI the held-out file comes from the
`OPERON_SEASON_PRIVATE_PY` secret (base64).

Season 2 (KAN-78): harder core sets for the three categories, built to land at 40 to 70% for
the reference model (Gemini 3.5 Flash) where Season 1's score 9 or 10 out of 10. Suites
`browser-core2`, `desktop-core2` and `hybrid-core2` (scored, 10 working tasks each: 4 short,
3 long, 3 safety; `CORE2` in `evals/categories.py`, tasks in `evals/season2/`) and grounding.
No held-out tasks. The browser and hybrid tasks run on the Season 2 site
(`evals/season2/apps.py`: Northwind Finance, People and Rooms beside the Season 1 apps), so
their fingerprints hash that file too. Run them as the core suite, so reports keep their grid
cell: `python -m evals.live_bench --season 2` (or `/bench browser thin gemini season=2`).
Frozen with
`python -m evals.leaderboard.season freeze 2 --suite browser-core2 --suite desktop-core2 --suite hybrid-core2 --suite grounding`.
It is frozen ahead of its calibration run (KAN-87): if calibration shows a task outside the
40 to 70% band for the wrong reason (a broken task, not a hard one), refreeze before any scored run.
KAN-104 refroze it (hash 5126b757b7bc, was 29d7c48c0f23) after calibration: Gemini passed all ten
Desktop tasks (KAN-87), so `desktop-core2` has ten harder tasks; the browser and hybrid sets are unchanged.
