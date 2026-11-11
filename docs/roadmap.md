# Roadmap

Where Operon has been, where it is now and what comes next. Last updated 2026-11-11.

Plans are written and approved in the project's Confluence space (OPERON) and tracked as Jira
stories (project KAN); this page summarises them for the repo. A story starts only after it is
approved, and every paid benchmark run needs its own cost approval. Nothing under
[Next](#next) or [Later](#later) is built yet.

## Past: what has been built

### October 2026: the agent and the first benchmarks

- **Claude computer use backend** for the browser and the desktop, the default when an
  Anthropic key is set. Gemini computer use stays a supported backend.
- **Thin harness** (#157, #162): Claude makes every decision and one completion check judges its
  "done". The full harness (rules, post-model guards, per-step verifier, recovery) remains for
  other backends and on request.
- **Claude on the desktop** (#154 to #156): stock Windows apps, driven with the real mouse and keyboard.
- **Live web benchmark**: 44 real web tasks graded by a screenshot judge. First comparison
  (25 Oct): Claude 96% at 30 s per task, Gemini 91% at 48 s.
- **Measurement around it**: cost per task and per success (#159), a per-run cost cap, Slack
  alerts and screenshots on the [eval dashboard](https://operon-evals.vercel.app) (#167); a
  local apps suite graded from app state; a prompt-injection and overreach safety suite graded
  from a request log; a judge check (99% accuracy on 88 hand labels); the ScreenSpot grounding
  eval and `--repeat` runs on a hard subset (#186). See [reports](reports/).
- **Security audit fixes** (#177, #185): the API requires a key by default and the batch
  browser has no debug port.
- **Repo restructure** (#191): `evals/` at the top level, prompts inside the package.
- **Claude zoom** behind `OPERON_CLAUDE_ZOOM` (#190), off by default: ScreenSpot-Pro went from
  70% to 83%, while desktop tasks scored 10/10 either way.

### November 2026: one yardstick for every model

- **Cross-model adapter** (#204, KAN-20): any model in [`evals/models.yaml`](../evals/models.yaml)
  runs on the same tasks, executor and graders. Native adapters for Claude, Gemini and OpenAI
  computer use, and an OpenAI-compatible adapter for open-weight models. Gemini got a desktop
  backend (#207), and Gemini and registry models (GPT) run in both harnesses (#224, #226).
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
- **Full harness fixes** (#228, KAN-75) and a working Claude step critic (#233, KAN-70).
- **Trust gate on by default** (#232, KAN-15): risky actions pause for a human's approval.
  Benchmarks turn it off.
- **Whole-repo review fixes** (#212 to #220, KAN-46 to KAN-51), a monthly budget on the dashboard's
  Cost tab (#208), a redesigned dashboard (#210) that reads through a read-only database role
  (#234, KAN-28), and the Azure desktop runner logging in with OIDC (KAN-27).
- **Removed**: the Command Center UI (#206), which was unused.

## Present: where Operon stands

### Season 1 grid

Core sets, 10 tasks each, every task graded from app or machine state. Details and triage are
in [reports/season-1-grid.md](reports/season-1-grid.md).

| Model | Browser Thin | Browser Full | Desktop Thin | Desktop Full | Hybrid Thin | Hybrid Full |
|---|---|---|---|---|---|---|
| Gemini 3.5 Flash | 10/10 | 10/10 | 9/10 | 10/10 | 9/10 | 7/10 |
| Claude Sonnet 5.5 | 9/10 | 10/10 | 10/10\* | 10/10\* | not run | not run |
| GPT (OpenAI computer use) | 10/10 | in progress | 10/10 | in progress | not run | not run |

\* On the Desktop core before the long and safety tasks were added (KAN-74).

What this tells us:

- **The core sets are saturated.** Most cells score 9 or 10, so they can no longer show whether
  a change to Operon helps. Harder tasks (Season 2) are the first item under Next.
- **Hybrid is the hardest category**, and most of its misses come from writing files or
  commands by typing into a shell, which the tool channel (A1 below) addresses.
- **Grounding**: on ScreenSpot-Pro, Claude scores 70% (83% with zoom) and Gemini 48%.

### In progress

- **GPT grid (KAN-71).** Browser Thin and Desktop Thin are done (10/10 each). The Full cells
  wait on more OpenAI API credit and on PR #235, which keeps a GPT critic verdict whose reason
  is free text instead of dropping it.
- **Plans awaiting decisions**: the consumer product plan and the CUA market research, both
  under [Later](#later).

## Next: architecture in line with the CUA market (epic KAN-77)

Approved order from the Confluence plan "Plan: Operon next phase, architecture in line with the
CUA market". The leading computer-use agents of 2026 share a hybrid action space, separate
planning and grounding, parallel attempts, isolated environments and batched actions. Operon is
ahead on verification, safety and state-graded benchmarks, and behind on the rest.

Decided: **pixels stay Operon's only way of seeing and targeting the GUI** (no accessibility
tree); typed tools are added beside it. Every workstream ships behind a flag that defaults to
off, per the repo's rules, and is compared on the grid before it is turned on.

| Order | Story | What it adds |
|---|---|---|
| 1 | KAN-78 Season 2 core sets | Harder tasks that today's models pass 40% to 70% of the time, plus step counts, so the work below shows up in the grid |
| 2 | KAN-89 Front door (A0) | A local task page (type a task, watch it, approve risky steps, see the result) and an `operon run "task"` command |
| 3 | KAN-79 Hybrid action channel (A1) | RFC 0002, then `run_command`, `read_file`, `write_file` and HTTP or MCP tools beside the GUI, all through the trust gate, behind `OPERON_TOOLS` |
| 4 | KAN-80 Run environments (A2) | Environment providers (local, Windows VM from a snapshot, Linux container), checkpoints after each verified step, resume |
| 5 | KAN-81 Async loop (A2) | One API process drives several runs in parallel |
| 6 | KAN-82 Grounder slot (A3) | Operon-side zoom for any model and a slot for a dedicated grounding model |
| 7 | KAN-83 Planner (A3) | An optional planner that splits long tasks into subgoals |
| 8 | KAN-84 Task-level best-of-N (A4) | Several attempts in parallel environments, the best picked by Operon's verifier |
| 9 | KAN-85 Step efficiency (A5) | Action batching for every backend, loop detection, a step-efficiency score |
| 10 | KAN-86 Task files (A6) | Saved, parameterised YAML tasks with expected-state checks and JUnit output |
| 11 | KAN-87 Run: Season 2 calibration | Gemini, tools on and off (paid, about $5 to $8) |
| 12 | KAN-88 Run: ScreenSpot-Pro with zoom | Grounding with Operon-side zoom (paid, about $1.50 per model) |

Proposed changes, not yet decided: the CUA research page ("Research: CUA market 2026 and
recommended changes to Operon's plan") recommends dropping the planner (KAN-83) for a
task-state note and a check before "done", running the zoom and screen-resolution
measurement (KAN-88) earlier, growing task files (KAN-86) into a replay cache that re-checks
each step by vision, keeping best-of-N (KAN-84) for benchmarks only, adding a safety story
(confirmations by action category, secrets never shown to the model, a redacted audit bundle)
and bringing the MCP server forward to right after the tool channel.

Small items alongside: finish KAN-71, triage the `hybrid_it_inventory` failure from KAN-76, and
split `loop.py` along its seams before the async work.

## Later

- **Operon for Windows** (Confluence: "Plan: Operon as a consumer product", awaiting Vickey's
  decisions). The recommendation is a signed Windows app for specific repetitive back-office jobs
  (for example bookkeepers in desktop accounting software), not a general consumer assistant,
  built on the front door (KAN-89). A local MCP server would ship the same day, with a no-key
  first run, design partners and, later, phone approvals and a hosted browser-only MCP server.
- **Open-weight models** through the `openai_compat` adapter, once a GPU is available (KAN-21).
- **SDK release last** (epic KAN-41, KAN-34 to KAN-40): `pip install operon-agent` (the PyPI name
  is reserved), a Python API and CLI, an MCP server and Docker, with the repo going public on the
  same day. Safe defaults: desktop opt-in, trust gate on.
- **macOS or Linux desktop executor**, only if the SDK's users need it.

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
