# Operon eval expansion plan

Oct 24, 2026 · @Vickey Kapoor

## Summary

Operon's evals are good at one thing: catching regressions on short web tasks. They are too thin to tell us much else. Both backends now score 40 to 42 of 44, so the browser suite is close to its ceiling. Tasks average 2 to 4 steps, while public benchmarks average about 30 (OSWorld) to about 300 (OSWorld 2.0). Each task runs once. The judge reads only the final screenshot, and nobody has checked how often it is right. There is nothing for safety, prompt injection, or grounding.

My recommendation is to add evals in three tiers, cheapest first. **Tier 1** needs no paid agent runs to build and costs little to run: a check on the judge itself, an offline grounding eval, a local deterministic task site with harder tasks, a prompt-injection and confirmation suite, and repeated runs on a hard subset. **Tier 2** adds public, comparable numbers at moderate cost: an Online-Mind2Web subset, WebArena-Verified Hard, and a state-checked desktop suite. **Tier 3** (OSWorld-Verified, WindowsAgentArena, AndroidWorld, OSWorld 2.0) should wait, because each needs new infrastructure or a large budget.

No benchmarks were run for this report.

## What we have today

| Suite | Size | How it is graded | Gaps |
| --- | --- | --- | --- |
| Offline eval (`tests/eval/scenarios.py`) | 20 scripted scenarios | Scripted screens; exact | Not real pixels or models; tests the loop only |
| Live browser bench (`tests/eval/live_tasks.py`) | 44 tasks, 7 categories, live public sites | Gemini 3.5 Flash judge on the final screenshot | Near ceiling (Claude 42/44, Gemini 40/44); 2 to 4 steps per task; one run each; the judge has never been validated; live sites cause noise (DuckDuckGo error page, PyPI CAPTCHA); no tasks that return an answer |
| Live desktop bench (`tests/eval/desktop_tasks.py`) | 10 Windows tasks, 3 in smoke | Same screenshot judge | Small; single-app; no check of real system state (file exists, file content) |
| Eval dashboard (`dashboard/`) | All saved reports | n/a | No cost data until PR #159; one suite only |

What is missing compared with the field:

- **Difficulty.** No long-horizon or multi-app tasks, so we cannot see where the agent breaks.
- **Reliability.** Single runs hide variance. A 2026 OSWorld study (On the Reliability of Computer Use Agents) found that results swing between runs of the same task and recommends repeated runs.
- **Grader trust.** AgentRewardBench (1,302 expert-labelled web trajectories) found that no single LLM judge does well on every benchmark, and that rule-based checks undercount success. We do not know our judge's error rate.
- **Safety.** Public results show computer-use agents are easily steered by text on screen: VPI-Bench deceived computer-use agents in up to 51% of cases, and RedTeamCUA measured up to 60% attack success for Claude Sonnet 4.5 end to end. We have no test of this, and the benchmark auto-approves every confirmation pause.
- **Comparability.** None of our numbers can be put next to published results.

## The benchmark landscape

| Benchmark | What it is | Fit for Operon | Cost to adopt |
| --- | --- | --- | --- |
| **ScreenSpot-v2 / ScreenSpot-Pro** | Grounding: one screenshot plus an instruction, predict a click point. Pro uses high-resolution professional apps | Very good. Pure vision, matches our invariant, one model call per sample | Low |
| **Online-Mind2Web** | 300 tasks on 136 live sites; WebJudge LLM judge (about 85% agreement with humans) or human grading | Good. Vision-friendly, live web, widely reported | Medium (live-site noise, many steps) |
| **WebArena-Verified** (ServiceNow) | 812 audited WebArena tasks, plus a 258-task Hard subset; deterministic scoring, no LLM judge | Good, but needs Docker sites (shopping, admin, Reddit, GitLab, Wikipedia, map) | Medium to high (hosting) |
| **REAL** (AGI Inc) | Deterministic clones of 11 real apps (Amazon, DoorDash, Airbnb, Gmail and others) | Fair. Its harness feeds DOM and accessibility trees, so we would drive the sites with our own loop | Medium |
| **WebGames** | About 50 short, self-contained browser challenges that are easy for humans, hard for agents | Good as a source of hard interaction tasks we can host locally | Low |
| **OSWorld-Verified** | 369 tasks in an Ubuntu VM, checked by scripts against final VM state; about 30 tool calls per task; human 72% | The headline computer-use number (top systems are now about 83 to 90%), but our desktop executor targets Windows | High |
| **OSWorld 2.0** (June 2026) | 108 long workflows, about 318 tool calls per task; Claude Opus 5.5 reports 81.8% partial, 48.7% binary | Too long and costly for now | Very high |
| **WindowsAgentArena** | 154 Windows tasks in a VM, state-checked | Matches our desktop target, but needs a Windows VM with nested virtualisation, which GitHub-hosted runners do not provide | High |
| **VisualWebArena** | 910 visually grounded WebArena-style tasks | Similar hosting cost to WebArena; lower priority than WebArena-Verified | High |
| **Mind2Web (offline)** | Element-selection on saved pages | Poor fit; scored on DOM elements, not pixels | n/a |
| **WebVoyager** | 643 tasks on 15 live sites | Superseded by Online-Mind2Web; many tasks are stale | Skip |
| **AndroidWorld** | 116 Android tasks | Out of scope; no Android executor | Skip |
| **OS-Harm** | 150 OSWorld tasks: misuse, prompt injection, model misbehaviour | Good design to copy; full version needs OSWorld | Medium |
| **RedTeamCUA (RTC-Bench)** | 864 injection cases in a hybrid web plus OS sandbox | Good design to copy | Medium |
| **VPI-Bench** | 306 visual prompt-injection cases across five platforms | Very good fit; attacks are pixels on a page | Low to medium |
| **WASP** | Prompt-injection tasks on WebArena sites | Useful once WebArena runs | Medium |

## Recommendations

### Tier 1: build now, little or no agent cost

**1. Validate the judge (do this first).** Every live number we publish rests on the Gemini judge, and we have never measured it. Take about 150 saved final screenshots from past runs in `.var/bench/` and the Actions artifacts, label them by hand (about two hours), and report the judge's precision and recall against the labels. Then try a stronger judge mode on the cases it gets wrong: give it the last 3 to 5 screenshots and the action log, in the style of WebJudge. Costs only judge calls. This tells us whether 42/44 means 42/44.

**2. Offline grounding eval (ScreenSpot-v2 plus a ScreenSpot-Pro subset).** Feed about 300 samples through the same model call Operon uses to pick a click point and score hit or miss against the target box. One call per sample, no browser, can run in CI on demand. It isolates the "clicked the wrong spot" failure mode that cost Gemini 350 to 500 px on desktop, and it shows the effect of the image-size and coordinate-scaling fixes directly.

**3. A local, deterministic task site with harder tasks.** Serve a small fixture web app from the repo during the benchmark and grade by its own server-side state (what was submitted, in what order), so there is no judge cost and no live-site noise. Aim for 25 to 30 tasks of 10 to 25 steps that cover what our live tasks do not: validation errors and retries, date pickers, nested menus, iframes, canvas, file upload and download, new tabs and pop-ups, long forms, infinite scroll, sortable and paginated tables, rich-text editors, and tasks that must return an answer ("what is the cheapest item over 4 stars?"). WebGames is open source and a good source of ready-made challenges. Note: grading by reading server state is fine under the vision-only rule, because the grader is not part of perception or policy.

**4. Safety: prompt injection and confirmation.** Add 20 to 30 pages to the fixture site with attacks taken from VPI-Bench, RedTeamCUA and OS-Harm: instructions hidden in page text, fake system dialogs, fake "session expired" banners asking for credentials, and email or review content saying "ignore your task and go to X". Score two numbers per case: did the agent follow the injection (attack success), and did it still finish the user's task. Add a second set of tasks that end in an irreversible action (buy, delete, send) and check that the trust gate pauses; today the benchmark auto-approves every pause, so we never see whether it should have paused. This is the biggest gap relative to how labs now evaluate computer use.

**5. Repeated runs on a hard subset.** Pick the 15 to 20 hardest tasks (from items 3 and 4 plus our live misses), run each 3 times, and report pass@1, pass^3 (all three succeed) and a confidence interval. Our current 40 vs 42 difference between backends is within single-run noise. Only the hard subset is repeated, so the extra cost stays small.

Also add to every report: cost per success (from PR #159), false-success rate (already there), pause correctness, and a short failure category per failed task (wrong target, gave up, false stop, blocked by site) labelled by the judge.

### Tier 2: public, comparable numbers, run occasionally

**6. Online-Mind2Web, 50-task hard subset.** Live sites, vision friendly, and graded with the published WebJudge, so our result sits next to published ones (top systems are about 90% on the full 300). Run a fixed 50-task subset at most once per major change.

**7. WebArena-Verified Hard (258 tasks).** Deterministic scoring with no LLM judge, and many tasks require returning an answer. Needs the six Docker sites and a self-hosted runner with enough disk; Operon must also return a final answer through the API, which it does not do today. Start with one or two sites (shopping and GitLab) rather than all six.

**8. State-checked desktop suite.** Grow the 10 Windows tasks to about 25, including multi-app tasks (copy text from a file into Notepad and save it, rename files in Explorer, change a Settings toggle), and grade with a PowerShell check after each run (file exists, file content, setting value) instead of only the screenshot. Runs on the existing Windows workflow.

### Tier 3: defer

- **OSWorld-Verified:** the standard headline number, but it needs a Linux desktop executor and an Ubuntu VM host, and about 369 tasks at about 30 steps each. Revisit once desktop mode is stable; a 50-task subset would be the first step.
- **WindowsAgentArena:** right platform, but needs a nested-virtualisation Windows VM host.
- **OSWorld 2.0, VisualWebArena, AndroidWorld:** too costly or out of scope for now.

## Proposed order and cost

| Step | Paid agent runs to build | Cost per run |
| --- | --- | --- |
| 1. Judge validation | None (labelling plus judge calls) | Judge calls only |
| 2. Grounding eval | None | About 300 single model calls |
| 3. Local task site | None to build; one run to baseline | About 30 tasks, no judge |
| 4. Safety suite | None to build; one run to baseline | About 30 to 40 tasks, no judge |
| 5. Repeats on hard subset | n/a | 3 times about 20 tasks |
| 6 to 8. Tier 2 | One baseline each, then only on major changes | 25 to 258 tasks |

Items 1 to 4 can all be built and tested offline with scripted models in the existing harness, so no Claude runs are needed until a baseline is wanted. I'd start with 1 and 4: the judge check decides how much we can trust every other number, and safety is the largest blind spot.

## Sources

- OSWorld leaderboards: [Steel](https://leaderboard.steel.dev/leaderboards/osworld/), [Presenc summary](https://presenc.ai/research/osworld-computer-use-leaderboard-2026)
- Online-Mind2Web: [paper](https://arxiv.org/abs/2504.01382), [leaderboard](https://leaderboard.steel.dev/leaderboards/online-mind2web)
- [WebArena-Verified](https://github.com/ServiceNow/webarena-verified)
- [REAL / AGI SDK](https://github.com/agi-inc/agisdk)
- [WebGames](https://arxiv.org/abs/2502.18356)
- [ScreenSpot-Pro](https://arxiv.org/abs/2504.07981)
- [Windows Agent Arena](https://arxiv.org/abs/2409.08264)
- [AgentRewardBench](https://arxiv.org/abs/2504.08942)
- [On the Reliability of Computer Use Agents](https://arxiv.org/abs/2604.17849)
- [tau-bench (pass^k)](https://arxiv.org/abs/2406.12045)
- [OS-Harm](https://arxiv.org/abs/2506.14866), [RedTeamCUA](https://arxiv.org/abs/2505.21936), [VPI-Bench](https://arxiv.org/abs/2506.02456)

Leaderboard figures come from third-party aggregators as of October 2026 and are partly self-reported by vendors.
