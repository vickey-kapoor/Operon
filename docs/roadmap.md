# Roadmap

Where Operon has been, where it is now and what comes next. Last updated 2026-11-14.

Plans are written and approved in the project's Confluence space (OPERON) and tracked as Jira
stories (project KAN); this page summarises them for the repo. The plan of record is "Roadmap:
scaling Operon in six stages", approved on 2026-11-14. A story starts only after it is approved,
and every paid benchmark run needs its own cost approval.

## Past: what has been built

### October 2026: the agent and the first benchmarks

- **Thin harness** (#157, #162): the model makes every decision. The full harness (rules,
  post-model guards, per-step verifier, recovery) remains the default.
- **Desktop mode** (#154 to #156): stock Windows apps, driven with the real mouse and keyboard.
- **Live web benchmark**: 44 real web tasks graded by a screenshot judge.
- **Measurement around it**: cost per task and per success (#159), a per-run cost cap, Slack
  alerts and screenshots on the [eval dashboard](https://operon-evals.vercel.app) (#167); a
  local apps suite graded from app state; a prompt-injection and overreach safety suite graded
  from a request log; a judge check (99% accuracy on 88 hand labels); the ScreenSpot grounding
  eval and `--repeat` runs on a hard subset (#186). See [reports](reports/).
- **Security audit fixes** (#177, #185): the API requires a key by default and the batch
  browser has no debug port.
- **Repo restructure** (#191): `evals/` at the top level, prompts inside the package.

### November 2026: one yardstick

- **Gemini on the desktop** (#207), and Gemini in both harnesses (#224, #226).
- **Graded from state, not a judge**: the desktop benchmark reads the machine's state (#209,
  #212), and model comparisons use no screenshot judge.
- **Categories and the grid** (#227, KAN-72): every benchmark belongs to Browser, Desktop or
  Hybrid, and each runs in the thin and full harness, a 3 x 2 grid per model with a 10-task
  core set per category. Desktop Long and Desktop Safety tasks (#230, KAN-74) and the Hybrid
  benchmark, where each task uses a Windows app and the browser (#231, KAN-73), completed it.
- **Leaderboard machinery** (#221, #222, #229): frozen task hashes, paired bootstrap ranges,
  run-over-run change labels per grid cell on the dashboard and on `/bench` PR comments.
- **Season 1** (#225, KAN-65): frozen on 2026-11-10 with held-out tasks kept out of the repo.
  See [evaluation](evaluation.md#seasons).
- **Full harness fixes** (#228, KAN-75).
- **Front door** (#240, KAN-89): `operon serve` and `operon run "task"`, with a local task page to
  type a task, watch it, approve risky steps and see the result, later the first app screens (#245,
  KAN-106).
- **Season 2** (#237, KAN-78): harder 10-task core sets per category on new local apps (Finance,
  People, Rooms), frozen on 2026-11-11. Calibration runs are under way (KAN-87).
- **Tool action channel** (#254, KAN-79, [RFC 0002](rfcs/0002-tool-action-channel.md)): typed
  `read_file`, `list_dir`, `write_file`, `http_get` and, opt-in, `run_shell` beside the GUI, behind
  `OPERON_TOOLS` (off by default until the on/off comparison holds the score).
- **Trust gate on by default** (#232, KAN-15): risky actions pause for a human's approval.
  Benchmarks turn it off.
- **Whole-repo review fixes** (#212 to #220, KAN-46 to KAN-51), a monthly budget on the dashboard's
  Cost tab (#208), a redesigned dashboard (#210) that reads through a read-only database role
  (#234, KAN-28), and the Azure desktop runner logging in with OIDC (KAN-27).
- **Stage 1, installable** (#255, #265, #267, #269; KAN-140, KAN-141, KAN-147, KAN-36):
  `pip install operon-agent` from PyPI (0.1.0, published from a tag by `release.yml`), `operon
  setup` (saves the Gemini key to `~/.operon`, installs Chromium, runs a self-check), a per-user
  data folder for runs installed with pip, a 5-minute README quick start with a demo GIF, and a
  [step-by-step first run](first-run.md) for new users.
- **Removed**: the Command Center UI (#206), which was unused, and, on 2026-11-11, every model
  but Gemini (Claude, GPT and the cross-model adapter), so that model onboarding can be rebuilt
  clean and simple. Their earlier benchmark data stays on the dashboard.

## Present: where Operon stands

### Season 1 grid

Core sets, 10 tasks each, every task graded from app or machine state. Details and triage are
in [reports/season-1-grid.md](reports/season-1-grid.md).

| Model | Browser Thin | Browser Full | Desktop Thin | Desktop Full | Hybrid Thin | Hybrid Full |
|---|---|---|---|---|---|---|
| Gemini 3.5 Flash | 10/10 | 10/10 | 9/10 | 10/10 | 9/10 | 7/10 |

What this tells us:

- **The core sets are saturated.** Most cells score 9 or 10, so they can no longer show whether
  a change to Operon helps. Harder tasks (Season 2) are the first item under Next.
- **Hybrid is the hardest category**, and most of its misses come from writing files or
  commands by typing into a shell, which the tool channel (A1 below) addresses.
- **Grounding**: Gemini scores 48% on ScreenSpot-Pro.

### In progress

- **Gemini only.** Operon runs Gemini 3.5 Flash for every run; other models come back in stage 4.
- **Stranger test** (KAN-142), the stage 1 finish test: Vickey's own fresh installs from PyPI
  pass in about 5 minutes; a run by someone new to Operon is next.
- **Season 2 calibration** (KAN-87): Gemini on the Season 2 core sets, then the tools on/off
  comparison that decides whether `OPERON_TOOLS=on` becomes the default.

## Next: scaling Operon in six stages

**Get real users first, then build what they need.** Each stage ends with a test you can check,
and the engine stays the same throughout: what changes is the form Operon ships in and who can
use it. Today Operon installs from PyPI, and stage 1 waits only on the stranger test.

| Stage | What it includes | Done when | Operon becomes |
|---|---|---|---|
| 1. Installable | PyPI alpha (`pip install operon-agent`), a one-step setup command, 5-minute quickstart and demo GIF | A stranger finishes a task in 5 minutes | PyPI package (alpha) |
| 2. First 10 users | Soft launch, an opt-in way to see how people use it (nothing sent by default) | 10 people ran it and 3 came back | Tool with real users |
| 3. Reliable | Season 2 calibration, the tool channel on by default, safety (KAN-103), verify before done (KAN-102), perception work (epic KAN-113) | Season 2 scores climb | Beta |
| 4. Builders | Model onboarding beyond Gemini, local MCP server and Claude plugin (KAN-96), Docker (KAN-40), task files and replay cache (KAN-86) | An outside project builds on Operon | Platform |
| 5. Public launch | Show HN, demo video, docs (KAN-37), launch post (KAN-39), a decision on publishing the benchmark grid | Stars become weekly installs | 1.0 project |
| 6. Scale | Python SDK (KAN-35), the Windows desktop app (epics KAN-122 and KAN-94, including the bookkeeping pilot), parallel runs (KAN-80, KAN-81), hosted runs (KAN-107 to KAN-109) | Teams run it on a schedule | SDK + app + cloud |

Notes on the order:

- **The repo goes public with the PyPI alpha** in stage 1, because a wheel ships the source. The
  repo is hardened first (KAN-34), and the big announcement waits for stage 5.
- **The first users are developers** installing from PyPI. The bookkeeping design partner pilot
  (KAN-100) moves to stage 6, once the Windows app exists.
- **Architecture work from epic KAN-77 not named above** (zoom KAN-82/KAN-88, a decision on a swappable grounder slot KAN-148,
  best-of-N KAN-84, step efficiency KAN-85, Windows risks KAN-90 to KAN-93) belongs to stage 3 when
  it improves reliability on what users try, and to stage 6 when it is about scale.
- **Every workstream ships behind a flag that defaults to off** and is compared on the grid before
  it is turned on. **Pixels stay Operon's only way of seeing and targeting the GUI**; typed tools
  sit beside it.

Small items alongside: triage the `hybrid_it_inventory` failure from KAN-76, and split `loop.py`
along its seams before the async work.

## Not planned

- **Public benchmarks** such as OSWorld or WebArena, for budget reasons. Operon's own suites are
  the yardstick, and public-style tasks are rebuilt as our own when needed.
- **A screenshot judge in model comparisons.** The grid compares raw model capability on tasks
  graded from state.
- **DOM, selectors or the accessibility tree** for perception or targeting (the vision-only
  invariant).
- **A browser extension**, which would lose the desktop and invite DOM use.

## Related documents

- [PRD](PRD.md): market context, users and requirements.
- [RFC 0001](rfcs/0001-cua-v2.md): the first v2 design proposal, and which parts of it are built.
- [Evaluation](evaluation.md): every suite, the grid, seasons and the dashboard.
- [Reports](reports/): benchmark write-ups.
