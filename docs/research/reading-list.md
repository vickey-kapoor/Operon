# Research: reading list and paper notes

The git copy of the Confluence page "Research: reading list and paper notes" (space OPERON).
A weekly GitHub job (`.github/workflows/research-weekly.yml`) adds new arXiv papers under
"Weekly scans" at the bottom. Weekly entries are collected automatically from arXiv and are
abstract excerpts, not reviewed notes; promote a paper into a section above once it has been read.

**Purpose.** A running reading list of research papers that matter for Operon, each with a plain-language note on what it actually shows and what it means for us. Every paper in sections 1 to 5 was checked on arXiv (title and date) on 12 Nov 2026; notes are drawn from the abstracts unless marked otherwise, so numbers are the authors' own claims on their own setups.

**Start here (top 3 to read first)**

1. **Agent-Computer Observation Interfaces (AOI)**: turning frames between steps into text helped; extra images hurt Gemini Flash.
2. **Scaling Agents for Computer Use (Agent S3 / BJudge)**: run several attempts, compare them as text narratives, pick the best. Same idea as our best-of-N flag, at whole-run level.
3. **Attacking Vision-Language Computer Agents via Pop-ups**: simple fake pop-ups got agents to click them 86% of the time on average. Direct test case for our safety suite.

## 1. GUI grounding

Grounding = turning "click the Save button" into the right pixel. Operon already scores this with `evals.grounding` on ScreenSpot-v2/Pro.

| Paper | What it shows | Why it matters for Operon |
| --- | --- | --- |
| [SeeClick](https://arxiv.org/abs/2401.10935) (Jan 2024) | Screenshot-only agent; finds grounding is the key bottleneck; introduces **ScreenSpot**, the first realistic grounding benchmark. | Origin of the benchmark our grounding eval uses. |
| [ScreenSpot-Pro](https://arxiv.org/abs/2504.07981) (Apr 2025) | High-resolution professional apps (23 apps, 3 OSes). Best model at release scored only 18.9%. | Closest match to our Desktop category: small targets on big screens. |
| [UGround](https://arxiv.org/abs/2410.05243) (Oct 2024) | Argues agents should perceive purely visually, like humans; trains a universal grounding model. | Academic support for Operon's vision-only invariant. |
| [OS-Atlas](https://arxiv.org/abs/2410.23218) (Oct 2024) | Open cross-platform grounding model plus the largest open grounding dataset at the time. | Data source if we ever host our own grounder (`OPERON_GROUNDER`). |
| [GUI-Actor](https://arxiv.org/abs/2506.03143) (Jun 2025) | Grounding without generating coordinates as text: an attention head points at screen patches and can propose several regions. | Model-level idea; relevant only if we host a grounder. |
| [OSWorld-G / Jedi](https://arxiv.org/abs/2505.13227) (May 2025) | Benchmark of 564 harder grounding cases (layout, fine manipulation) and a 4M-example dataset. | Candidate extra set for `evals.grounding` beyond short labels. |
| [GTA1](https://arxiv.org/abs/2507.05791) (Jul 2025) | Samples several candidate actions per step and lets a judge model pick; also RL-trained grounding. | Same idea as our `best_of_n.py` (flag off today). |

## 2. Computer-use agent designs

| Paper | What it shows | Why it matters for Operon |
| --- | --- | --- |
| [UI-TARS](https://arxiv.org/abs/2501.12326) (Jan 2025) | One end-to-end model from screenshots to mouse/keyboard; reported 24.6 on OSWorld (50 steps) vs Claude's 22.0 at the time. | Reference for "native agent model" vs harness; open weights could return when models are re-added. |
| [UI-TARS-2](https://arxiv.org/abs/2509.02544) (Sep 2025) | Multi-turn RL, data flywheel, and a hybrid environment with files and terminals. | Their hybrid GUI+terminal setup mirrors our Hybrid category. |
| [Agent S2](https://arxiv.org/abs/2504.00906) (Apr 2025) | Splits work across generalist and specialist models; "Mixture-of-Grounding" and hierarchical planning. | The modular design we chose not to copy (single Gemini). |
| [Scaling Agents for Computer Use](https://arxiv.org/abs/2510.02250) (Oct 2025, Agent S3) | Runs several full attempts and compares them as behaviour narratives (BJudge); reports 72.6% on OSWorld. | Already cited in the KAN-83 decision; run-level best-of-N is a future option once cost allows. |
| [OpenCUA](https://arxiv.org/abs/2508.09123) (Aug 2025) | Open framework: demo-capture tool, AgentNet dataset (3 OSes, 200+ apps), open models. | Shows how others record human demos; relevant to replay cache (KAN-86/99). |
| [OmniParser](https://arxiv.org/abs/2408.00203) (Aug 2024) | Parses a screenshot into labelled interactive elements so a general model can act on them. | The "local UI detector" in the later stage of KAN-113. |

## 3. World models and continuous perception (feeds KAN-113)

| Paper | What it shows | Why it matters for Operon |
| --- | --- | --- |
| [WebDreamer](https://arxiv.org/abs/2411.06559) (Nov 2024) | Uses an LLM to imagine the result of each candidate action before committing, because the web has irreversible actions. | Prediction stage without training; ties to the expected-change check (KAN-102). |
| [Web Agents with World Models (WMA)](https://arxiv.org/abs/2410.13232) (Oct 2024) | Finds current LLMs lack a world model; trains one that predicts outcomes as text descriptions of changes. | Supports predicting *changes as text*, not whole screens. |
| [Computer-Using World Model](https://arxiv.org/abs/2602.17365) (Feb 2026) | Predicts the next screen for a candidate action in Office apps. No numbers in the abstract. | KAN-113 research spike (world-model lookahead). |
| [AOI](https://arxiv.org/abs/2606.29472) (Jun 2026) | Keyframes + audio + narration between steps; big gains on the authors' own dynamic benchmark. Frame images hurt Gemini 3 Flash; text narration helped. | Basis for KAN-110/111. Full notes on Confluence page 3112961. |
| [CUA-Suite](https://arxiv.org/abs/2603.24440) (Mar 2026) | Dataset: ~10k desktop tasks, ~55 h of 30 fps video. | Training data, not a runtime design. |
| [Watch and Learn](https://arxiv.org/abs/2510.04673) (Oct 2025) | Turns web videos into 53k training trajectories via an inverse-dynamics model. | Training-data method. |

## 4. Benchmarks and how to grade agents

| Paper | What it shows | Why it matters for Operon |
| --- | --- | --- |
| [OSWorld](https://arxiv.org/abs/2404.07972) (Apr 2024) | Real-computer benchmark graded by checking machine state after the run. | Our Desktop and apps suites use the same "grade from state" approach. |
| [WebArena](https://arxiv.org/abs/2307.13854) (Jul 2023) | Self-hosted realistic websites (shop, forum, GitLab, CMS) graded on functional correctness. | Model for our local fake shop/mail/account apps. |
| [VisualWebArena](https://arxiv.org/abs/2401.13649) (Jan 2024) | Web tasks that need images to solve. | Vision-only agents should shine here; idea for tasks. |
| [Windows Agent Arena](https://arxiv.org/abs/2409.08264) (Sep 2024) | Windows version of OSWorld-style tasks, run in parallel on Azure. | Same platform as our Azure Desktop runner. |
| [AndroidWorld](https://arxiv.org/abs/2405.14573) (May 2024) | 116 tasks with randomised parameters, so tasks vary each run. | Parameterised tasks would make our seasons harder to overfit. |
| [An Illusion of Progress?](https://arxiv.org/abs/2504.01382) (Apr 2025) | Re-tests web agents on 300 live tasks (Online-Mind2Web); finds earlier results were over-optimistic. | Caution for our own scores; supports frozen seasons and state grading. |
| [AgentRewardBench](https://arxiv.org/abs/2504.08942) (Apr 2025) | Measures how well LLM judges grade web-agent runs against expert labels. | Same question as our `judge_check` (99% on 88 hand labels). |

## 5. Safety and prompt injection

| Paper | What it shows | Why it matters for Operon |
| --- | --- | --- |
| [Attacking VLM agents via Pop-ups](https://arxiv.org/abs/2411.02391) (Nov 2024) | Adversarial pop-ups got clicked 86% of the time on average in OSWorld and VisualWebArena. | Add pop-up tasks to our safety suite. |
| [EIA](https://arxiv.org/abs/2409.11295) (Sep 2024) | Hidden injected page content steals users' personal data from web agents. | Threat model for the consumer app (KAN-94/122). |
| [OS-Harm](https://arxiv.org/abs/2506.14866) (Jun 2025) | 150 OSWorld-based tasks across misuse, prompt injection and model misbehaviour. | Template for a Desktop Safety set. |
| [RedTeamCUA](https://arxiv.org/abs/2505.21936) (May 2025) | Hybrid web + OS sandbox; 864 injection tests. | Matches our Hybrid category; idea for Hybrid safety tasks. |

## 6. Candidate findings for Operon (Vickey decides)

Proposals only, not stories:

- **Adversarial pop-up tasks** in the Browser and Desktop safety suites (from the pop-ups paper). Cheap, offline to build.
- **Harder grounding samples** (OSWorld-G style) in `evals.grounding`.
- **Parameterised tasks** (AndroidWorld style) so season tasks vary per run.

## 7. Weekly scans (newest first)

