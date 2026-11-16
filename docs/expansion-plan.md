# Operon expansion plan: competing with the AI labs

Oct 27, 2026 · @Vickey Kapoor

## Summary

Operon should not try to out-model the labs; it should be the open, model-neutral harness that makes their models reliable, cheap and safe on long real-world workflows, across browser and desktop, with proof. The labs win on raw model capability, and the short-task benchmarks they quote are now saturated: OSWorld-Verified tops out at 85 to 90% against a 72% human baseline, and Online-Mind2Web at 90 to 98%. The open ground is long workflows (OSWorld 2.0's best binary score is under 50%), cost per successful task, safety under attack, and Windows desktop work that browser agents cannot reach.

The three moves, in order:

1. **Get on the public scoreboards (Nov to Dec 2026).** Run OSWorld-Verified, OSWorld 2.0 and Online-Mind2Web through the same thin harness, and publish success, cost per task and safety side by side for Claude, Gemini and one open-weights model.
2. **Win long workflows (Q1 2027).** Add the harness features that matter only past 20 steps: checkpoints and resume, replay of a verified past run, and approval gates. Target OSWorld 2.0 binary completion above the best published lab number.
3. **Ship it where people build (Q2 2027).** Open-source the repo, add an MCP server and Python SDK, and pick one vertical with a design partner: Windows back-office automation or UI testing.

## Where Operon stands today

Operon already has what most small agent projects lack: a measured, cheap, safe agent on both browser and desktop, and the eval machinery to prove it. What it lacks is any number on a benchmark the field recognises.

| Area | Result today | Source |
| --- | --- | --- |
| Browser, 44 live web tasks | 41 to 43 of 44 over the last three runs (Claude Opus 5.5, thin harness), about 29 s, 3 steps and $0.04 to $0.06 per task, 0 false successes | `docs/evaluation.md` |
| Browser, Gemini backend | 40/44, about 50 s per task; kept as a supported backend and fallback | `docs/evaluation.md` |
| Desktop smoke, Azure Windows VM | 3/3 in both runs, 32 to 36 s and $0.06 to $0.07 per task | `docs/evaluation.md` |
| Safety, 9 injection and overreach tasks | 9/9 after fixture fixes, 0 compromised, $0.03 per task | `docs/evals/safety-and-judge-check.md` |
| Judge vs 88 hand labels | 99% accuracy, 100% recall, 100% self-agreement over 3 repeats | same |
| Local apps suite | 11 state-graded tasks (shop, mail, account, forms) | `docs/evals/local-apps.md` |
| Infra | `/bench` from a PR comment, $5 run cap, Slack alerts, eval and cost dashboard on Vercel + Neon | `docs/evaluation.md` |

Three assets matter most for the plan. The **one loop for browser and desktop**, driven by pixels only. The **thin harness**, which gets out of the frontier model's way and so rides each new model release for free. And the **eval stack**, which lets Operon make claims a reader can check.

The gaps are just as clear. The 44 tasks are short (3 steps on average) and mostly read-only, the desktop suite has 10 tasks with only 3 run regularly, there is no OSWorld or WebArena score, no users outside this project, and the repo is private.

## The landscape

Every lab now sells computer use, but as either a model API you must wrap yourself or a consumer assistant tied to its own browser; none ships an open, cross-model harness for browser plus desktop. Scores below are mostly self-reported and use different judges and step limits, so treat them as directional.

**Who sells what**

| Player | Form | Surface | Public numbers |
| --- | --- | --- | --- |
| Anthropic | Computer-use API tool; Claude in Chrome from $20/month | Desktop and browser; you run the sandbox | Opus 5.5: 81.8% partial, 48.7% binary on OSWorld 2.1; Claude models hold 6 of the top 10 OSWorld-Verified rows ([Presenc](https://presenc.ai/research/osworld-computer-use-leaderboard-2026), [Steel](https://leaderboard.steel.dev/leaderboards/osworld/)) |
| OpenAI | GPT-5.x native computer use in the API; cloud browser in ChatGPT; Operator and ChatGPT agent retired | Browser first, macOS for Codex | GPT-5.4: 93% Online-Mind2Web, self-reported ([Steel](https://leaderboard.steel.dev/leaderboards/online-mind2web/)) |
| Google | Gemini 2.5 Computer Use API, still preview; Gemini in Chrome auto browse | Browser only | 69% Online-Mind2Web, no OSWorld-Verified score ([Steel](https://leaderboard.steel.dev/leaderboards/online-mind2web/)) |
| Microsoft | Copilot Studio computer use, GA May 2026, built on OpenAI and Claude models | Windows machines you register | Inherits the model's score; sells governance: DLP, Key Vault, audit logs ([Digital Applied](https://www.digitalapplied.com/blog/computer-use-agents-microsoft-anthropic-google)) |
| Open-weights labs | Qwen3.8-Max, Xiaomi MiMo-V2.6, ByteDance UI-TARS-2, Microsoft Fara | Self-hosted | MiMo-V2.6-Pro 82.0% OSWorld-Verified; UI-TARS-2 88.2% Online-Mind2Web ([Steel](https://leaderboard.steel.dev/leaderboards/osworld/)) |
| Harness startups | Browser Use Cloud, Skyvern, Stagehand/Browserbase, Coasty, Pointer | Mostly browser | Browser Use Cloud 97 to 98% Online-Mind2Web, self-judged; Coasty 82.8% and Pointer + Opus 4.7 83.6% OSWorld-Verified ([Browser Use](https://browser-use.com/best-web-agents)) |

**What the benchmarks say**

| Benchmark | What it tests | Best today | Human | Read |
| --- | --- | --- | --- | --- |
| OSWorld-Verified | 369 desktop tasks, Ubuntu apps, 100 steps | 85 to 90% | 72% | Saturated; table stakes, not a differentiator |
| OSWorld 2.0 (June 2026) | 108 long workflows, about 1.6 hours each for a person, 500 steps | under 50% binary (Opus 5.5 48.7%) | not published | The open frontier, and where a harness matters |
| Online-Mind2Web | 300 live-web tasks | 90 to 98%, judges differ | n/a | Saturated for top agents |
| WebArena-Verified | Self-hosted web apps | GPT-5.4 82.3%, single source | n/a | Close to saturated |

Two patterns stand out. Harness startups beat the labs' own numbers on the same models (Pointer, Coasty, Browser Use), so the harness is worth points, not just packaging. And the lab products all stop at a border: Google at the browser, OpenAI at macOS, Microsoft at its own cloud, Anthropic at "run your own sandbox".

## Where Operon can win

Operon wins where a lab's incentives stop: a lab sells its own model inside its own surface, so it will not build an open harness that switches models, runs on any Windows box, and publishes cost and safety next to success.

**Compete here**

- **Model neutrality.** Operon already runs Claude and Gemini through one loop and measures both. Adding an open-weights backend (MiMo or Qwen) makes it the only place a team can compare success, cost and speed across vendors on their own tasks.
- **Long workflows.** OSWorld 2.0 is under 50% for everyone. Verification, checkpoints and recovery are harness work, and Operon's verifier and recovery ladder already exist in the full harness.
- **Cost per successful task.** Operon measures it today ($0.04 to $0.06 per browser task). No lab publishes it. A replay of verified past runs could cut repeat-task cost by an order of magnitude, which a per-token vendor has no reason to build.
- **Windows desktop and pixel-only surfaces.** Citrix, RDP and legacy Windows apps have no DOM. Google stops at the browser and OpenAI starts on macOS; only Microsoft is here, and only inside Power Platform.
- **Trust you can audit.** Self-hosted, every screenshot and decision on disk, a safety suite with 0 compromised runs, a judge checked against hand labels. That is what regulated buyers ask for.

**Do not compete here**

- **Training a base model.** The labs spend more on one run than this project ever will, and the thin harness gets each new model for free.
- **Consumer browser assistants.** Claude in Chrome, Gemini in Chrome and ChatGPT's browser own that with distribution Operon cannot match.
- **Chasing saturated scores.** A 1-point gain on OSWorld-Verified or Online-Mind2Web says little. Run them once for credibility, then spend effort on OSWorld 2.0 and cost.

## The plan, by phase

Three phases, each closed by a gate that decides whether the next is worth starting.

&#91;embedded content: Expansion roadmap · 3 phases, 2 gates\]

**Phase 1: credibility on public benchmarks (Nov to Dec 2026)**

- Run **OSWorld-Verified** (369 tasks, Ubuntu VM) and **Online-Mind2Web** (300 tasks) once each with Claude Opus 5.5 on the thin harness. Goal: within 3 points of the best published Claude number, with cost per task reported.
- Run **OSWorld 2.0** (108 tasks) as the baseline for Phase 2.
- Add a **model matrix**: Opus, Sonnet, Haiku, Gemini, and one open-weights model behind `OPERON_*_BACKEND`. Publish success against cost per task as one chart on the dashboard.
- Grow internal suites toward long tasks: 20 tasks of 15+ steps on the local apps, desktop suite from 10 to 30 Windows tasks, 3 repeats per task as the eval plan proposes.
- Budget: roughly $40 to $80 for one pass of OSWorld-Verified and Online-Mind2Web at today's cost per task. OSWorld 2.0 tasks run up to 500 steps, so estimate $100 to $500 there, a guess until measured. Each run needs Vickey's go.

**Phase 2: win long workflows (Q1 2027)**

- **Checkpoints and resume** across hours, so a 200-step task survives a crash or a human pause.
- **Verified replay**: store successful trajectories, replay them on a matching screen, and call the model only where the screen differs. Behind an `OPERON_REPLAY` flag, measured on repeated tasks.
- **Progress memory** for the model: a running summary of what is done, so context does not run out on long tasks.
- **Approval gates** on payments, sends and deletes, extending the safety suite to grade them.
- **Desktop depth**: RDP and Citrix sessions on the Azure runner, Office and a line-of-business app.

**Phase 3: distribution (Q2 2027)**

- **Open-source the repo** under Apache-2.0, once the Azure runner is safe from fork PRs.
- **MCP server and Python SDK**, so Claude Code, other agents and scripts can call Operon as a tool.
- **One-command install**: a Docker image for the browser, an installer for the Windows desktop agent.
- **Public results page**: a read-only view of the dashboard, refreshed by each benchmark run.
- **One design partner** in the vertical Vickey picks (below), with their tasks added as a private suite.

## Metrics and gates

Each phase ends on numbers that are public and repeatable, not on features shipped.

| Phase | Metric | Target | If missed |
| --- | --- | --- | --- |
| 1 | OSWorld-Verified, Opus 5.5 thin harness | within 3 points of best published Claude score | Fix the harness before Phase 2; the harness is losing points |
| 1 | Online-Mind2Web, same setup | 90% or more, WebJudge-graded | Same |
| 1 | Cost per successful task, all suites | published for every model in the matrix | Not optional; this is the differentiator |
| 1 | Safety suite | 0 compromised, grown to 30 tasks | Block release of any backend that fails |
| 2 | OSWorld 2.0 binary completion | above 48.7% (best published, Opus 5.5) | Keep Phase 2 running; skip Phase 3 marketing claims |
| 2 | Repeat-task cost with verified replay | 5x cheaper than a fresh run, same success | Drop replay, keep checkpoints |
| 2 | Desktop suite, 30 Windows tasks, 3 repeats | 80% or more | Hold desktop claims |
| 3 | Outside users | 1 design partner running weekly; first external contributors | Revisit the vertical choice |

## Risks and open decisions

The biggest risk is that a lab ships the harness itself; the hedge is to be the neutral layer across labs, which no single lab will build.

**Risks**

- **Labs close the harness gap.** Anthropic or Microsoft could ship checkpoints, replay and governance. Model neutrality and self-hosting stay out of their reach.
- **Benchmarks move.** OSWorld 2.0 may saturate within a year the way Verified did. Keep the private long-task suites as the ground truth and treat public boards as marketing.
- **Benchmark cost.** OSWorld 2.0 runs are long and the cost is unmeasured. Start with a 20-task slice to price it.
- **Bandwidth.** This is a small team against funded startups (Browser Use, Coasty, Pointer). Phase 3 only makes sense with Phase 2 numbers in hand.
- **Bot blocks.** 2 of the last browser misses were site CAPTCHAs and error pages. Public web tasks will keep producing noise the agent cannot fix.

**Decisions for Vickey**

- [ ] **Vertical for Phase 3:** Windows back-office automation (replacing RPA scripts, closest to the desktop strength) or UI testing for web apps (closest to the eval stack). Recommendation: back-office, because that is where no lab product reaches except Microsoft's.
- [ ] **Open-source timing:** at the start of Phase 3, once the Azure runner no longer runs on PRs from forks.
- [ ] **Monthly benchmark budget:** a cap for public-benchmark runs, separate from the $5 per-run cap.
- [ ] **Open-weights backend:** add one (MiMo-V2.6 or Qwen) in Phase 1, or wait for demand.

## Sources

Leaderboards and comparisons, read 2026-10-27; most scores are vendor self-reported.

- [Steel OSWorld leaderboard](https://leaderboard.steel.dev/leaderboards/osworld/), updated 2026-10-11
- [Steel Online-Mind2Web leaderboard](https://leaderboard.steel.dev/leaderboards/online-mind2web/), updated 2026-10-11
- [Presenc OSWorld and OSWorld 2.0 leaderboard](https://presenc.ai/research/osworld-computer-use-leaderboard-2026), as of 2026-10-01
- [Browser Use: best web agents](https://browser-use.com/best-web-agents), pricing checked 2026-10-15 (vendor page)
- [Digital Applied: Microsoft, Anthropic, Google computer use](https://www.digitalapplied.com/blog/computer-use-agents-microsoft-anthropic-google), 2026-05-22
- Operon results: `docs/evaluation.md`, `docs/evals/safety-and-judge-check.md` in the repo
