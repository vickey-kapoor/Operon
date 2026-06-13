# Operon — Product Requirements Document

| | |
|---|---|
| **Document** | Product Requirements Document (PRD) |
| **Product** | Operon — vision-first computer-use agent |
| **Version** | 1.0 (Draft for review) |
| **Status** | Draft |
| **Last updated** | 2026-06-13 |
| **Owner** | Operon maintainers |
| **Related docs** | [`docs/product_requirements.md`](./product_requirements.md) (engineering implementation-truth spec), [`docs/architecture.md`](./architecture.md), [`docs/codebase_overview.md`](./codebase_overview.md), [`docs/recovery-ladder.md`](./recovery-ladder.md), [`README.md`](../README.md), [`AGENTS.md`](../AGENTS.md) |

> **How this document relates to the existing spec.** `docs/product_requirements.md` is the engineering "implementation truth" — what the code does today, endpoint by endpoint. *This* document is the broader product PRD: market context, users, positioning, requirements (current **and** roadmap), success metrics, and risks. Where the two disagree about current behavior, `product_requirements.md` and the code win.

> **Note on external figures.** Market-size dollar figures in §2 come from commercial market-research firms whose estimates diverge 3–8× by definition; they are cited as ranges, not point values. Benchmark scores (§5, §10) are fast-moving, frequently vendor-self-reported, and sometimes rely on multiple attempts (pass@k) rather than single-attempt (pass@1); treat them as dated snapshots. Sources are listed in §15.

---

## 1. Executive summary

Operon is a **vision-first computer-use agent**: it operates software the way a person does — by looking at the screen, choosing a coordinate-based action, executing it, **verifying the visual result, and recovering when progress stalls** — with **no DOM, CSS selectors, or XPath**. A single control loop (`capture → perceive → decide → execute → verify → recover`) drives **both browser and desktop** automation through the same verifier, recovery manager, and persistence model.

The computer-use agent category exploded in 2024–2025: Anthropic shipped Claude Computer Use (Oct 2024), OpenAI shipped Operator/CUA (Jan 2025), Google shipped Gemini 2.5 Computer Use (Oct 2025), and Microsoft brought computer use to Copilot Studio (GA, 2025) [1][5][8][11]. Yet the category's defining problem is **reliability**: on OSWorld (real desktop tasks) humans score ~72% while agents only crossed ~60% single-attempt in late 2025 [16][30], and Gartner predicts **over 40% of agentic-AI projects will be cancelled by end of 2027** on cost, unclear value, and inadequate risk controls [24].

Operon's thesis is that **reliability and trust — not raw capability — are the wedge**. It pairs LLM-driven perception/policy with **deterministic guardrails that run before the model**, a **verify-after-every-action** loop, an explicit **recovery ladder**, **human-in-the-loop** pause/resume, and **full run-artifact observability**. It is open (Apache-2.0), self-hostable, and **unifies browser + desktop** — deliberately covering the desktop-OS gap that the lowest-latency browser model (Gemini 2.5 Computer Use) explicitly does **not** yet address [8].

---

## 2. Market context & opportunity

**The category is real and growing, but credibility is the gating factor.**

- **Agentic AI adoption is inflecting.** Gartner projects **40% of enterprise applications will feature task-specific AI agents by end of 2026, up from <5% in 2025** [22], and that by 2028 a third of enterprise software will embed agentic AI [23].
- **…but reliability and value are unproven at scale.** Gartner also predicts **>40% of agentic-AI projects will be cancelled by end of 2027** [24], and warns of "agent washing" — rebranding legacy RPA/chatbots as agents [24]. This is the credibility gap Operon targets.
- **Automation markets are large (figures vary widely by definition).** RPA market estimates for 2025 span **~$7B (Gartner, software-only) to ~$28B** (broader market-research scopes) [18][19][21]; the AI-agents market is estimated at **~$7.8B in 2025 growing to ~$53B by 2030 (~46% CAGR, MarketsandMarkets)** [25]; the **software test-automation** market is put at **~$25–29B in 2025 → ~$60B by 2029 (~19.6% CAGR, Research and Markets)** [27].
- **Traditional RPA's core weakness is exactly what vision-first solves.** Selector-based bots break when the UI changes — minor relabeling or reordering silently breaks scripts — and environments like **Citrix/RDP/VDI stream pixels with no DOM at all** [28][29]. Vision-based automation is the analyst-and-practitioner-recognized answer to brittle selectors and DOM-less surfaces [29].

**Opportunity statement.** There is a durable wedge for an **open, self-hostable, vision-first agent that is trustworthy by construction** — deterministic guardrails, verification, recovery, HITL, and a complete audit trail — and that **spans browser and desktop in one loop**, rather than forcing teams to stitch a browser-only tool to a separate desktop one.

---

## 3. Problem statement

Teams that want to automate real UI workflows face three bad options:

1. **Traditional RPA (UiPath, Automation Anywhere):** powerful but brittle. Selector/coordinate scripts break on UI change, carry heavy maintenance cost, and cannot operate DOM-less surfaces (Citrix/RDP) without bolt-on computer vision [28][29].
2. **Browser-only AI agents (browser-use, Skyvern, Operator-style products):** capable on the web but **cannot touch the desktop**, and DOM-based ones still inherit selector fragility [3][4][10].
3. **Frontier closed CUAs (Anthropic, OpenAI, Google, Microsoft):** strong models, but **closed, token/credit-metered, and reliability-limited** — they "sometimes assume outcomes of actions without explicitly checking results" [12], the single most-cited failure mode, and they are exposed to prompt injection as the #1 LLM risk [13].

**No widely available option is simultaneously:** vision-first (selector-free), **browser *and* desktop**, **open/self-hostable**, and **engineered for reliability and auditability** (deterministic guardrails + verification + recovery + HITL + full artifacts). That is the gap Operon fills.

---

## 4. Vision & strategy

**Vision.** *Operate any interface with a vision-driven computer-use engine* — so any UI a human can use, an agent can reliably and observably automate, without integration work.

**Strategy (how Operon wins):**

1. **Reliability as the product, not a feature.** Deterministic rules before LLM fallback, verify-after-every-action, an explicit recovery ladder, and HITL turn "impressive demo" into "trustworthy run."
2. **One loop, two surfaces.** Browser and desktop share the loop, verifier, recovery, and persistence — covering the desktop gap browser-only and Gemini-CUA leave open [8].
3. **Open and inspectable.** Apache-2.0, self-hostable, with every run producing screenshots, model I/O, policy decisions, execution traces, and logs — an auditable record that closed metered APIs do not surface.
4. **Backend-agnostic.** Pluggable planner/verifier providers (Gemini, Anthropic) and a cheaper JSON perception/policy fallback give cost and vendor flexibility.

---

## 5. Competitive landscape & positioning

**Approach axis (the key differentiator).** The market splits into DOM/accessibility-based vs. pure-vision (pixel+coordinate) vs. hybrid. The leading *commercial* CUAs are all primarily vision-based; pure-DOM is now mainly the open-source browser-only tier [11].

| Player | Approach | Browser | Desktop | Open? | Pricing | Notes |
|---|---|:--:|:--:|:--:|---|---|
| **Operon** | **Pure vision (pixel+coord), no DOM** | ✅ | ✅ | ✅ Apache-2.0 | Self-host + provider tokens | Deterministic guardrails + verify + recovery + HITL + full artifacts; unified loop |
| Anthropic Claude Computer Use | Pure vision | ✅ | ✅ | ❌ | ~$3/$15 per 1M tok [2] | OSWorld 61.4% (Sonnet 4.5, Sep 2025) [2]; strong OS coverage |
| OpenAI Operator/CUA | Vision-first hybrid | ✅ (product) | ✅ (model) | ❌ | $3/$12 per 1M tok (API) [6][7] | Standalone Operator shut 2025-08-31, folded into ChatGPT Agent [5] |
| Google Gemini 2.5 Computer Use | Vision (screenshots+coords) | ✅ | **❌ "not yet optimized for desktop OS"** [8] | ❌ | Gemini API token pricing | Lowest-latency browser control; **Operon's default browser backend** |
| Microsoft Copilot Studio (CUA) | Vision | ✅ | ✅ (Windows) | ❌ | 5/15 Copilot Credits per step [11] | No-code + enterprise governance; pluggable OpenAI/Anthropic models; Citrix/Electron may be unsupported [11] |
| browser-use | **DOM/HTML** | ✅ | ❌ | ✅ MIT (~98.5k★) | Library | Browser-only; selector-based extraction [3] |
| Skyvern | Hybrid (vision+Playwright) | ✅ | ❌ | ✅ AGPL (~21.9k★) | OSS + cloud | YC-backed; anti-brittle-selector positioning [4] |
| UI-TARS (ByteDance) | Pure vision native VLM | ✅ | ✅ | ✅ Apache (~10.9k★) | Model+runtime | Cross-env end-to-end model [9] |
| Agent-S (Simular) | Pure vision (screenshots only) | ✅ | ✅ | ✅ (~11.8k★) | Framework | OS-level; mixture-of-grounding [10] |

**Operon's positioning statement:** *the open, vision-first agent that automates browser **and** desktop through one verification-driven loop — built for reliability and auditability, not just capability.*

**Where Operon is differentiated:** (a) unified browser+desktop vision loop in an open project; (b) reliability architecture (deterministic-rules-first + verify + recovery ladder + HITL); (c) full self-hostable observability/audit trail.
**Where competitors lead today:** frontier models hold higher raw benchmark scores (Anthropic/OpenAI on OSWorld; Google on web/mobile) [11][30]; Microsoft leads on no-code + enterprise governance [11]; browser-use leads on web-ecosystem mindshare [3]. Operon consumes the best models rather than competing on model training.

---

## 6. Target users & personas

1. **QA / Test Automation Engineer ("Maya").** Maintains end-to-end UI tests that break whenever the front-end changes. Wants selector-free tests that survive redesigns, run on web and desktop apps, and produce screenshots + traces for triage. *Success = fewer flaky tests, less maintenance, visual evidence on every failure.*

2. **RPA / Automation Developer ("Devin").** Owns back-office bots (onboarding, data entry, multi-system approvals), some on **Citrix/RDP where there is no DOM** [29]. Wants a vision agent that handles DOM-less surfaces and degrades to human handoff instead of silently failing. *Success = automate previously un-automatable surfaces; auditable runs.*

3. **AI/Platform Engineer ("Priya").** Embeds agentic automation into an internal product. Needs an API, pluggable model backends, cost controls, observability, and self-hosting for data-residency. *Success = reliable API, token-cost visibility, no per-seat lock-in.*

4. **Operations / Knowledge Worker ("Sam") (roadmap-facing).** Non-technical; wants to delegate a repetitive multi-app task in natural language and watch/intervene live. *Success = describe a task, supervise via live view, take over when needed.*

---

## 7. Use cases & user stories

- **UC-1 Selector-free E2E web testing.** *As Maya, I run a regression suite against a redesigned web app without updating a single selector, and get before/after screenshots and an execution trace for every step.*
- **UC-2 Desktop / DOM-less automation.** *As Devin, I automate a Win32/Citrix workflow that has no DOM, and the agent pauses for me when it is uncertain rather than acting blindly.*
- **UC-3 Cross-surface workflow.** *As Devin, one task spans a desktop app and a browser; the same loop handles both with shared verification and recovery.*
- **UC-4 Embedded automation API.** *As Priya, I POST a task, stream live frames over WebSocket, and read token-usage and artifacts via the Observer API.*
- **UC-5 Supervised live run.** *As Sam, I watch the agent operate via the observable browser stream and take over for a login.*
- **UC-6 Audit & replay.** *As a compliance reviewer, I export a completed run and replay exactly what the agent saw and did.*

---

## 8. Product principles

1. **Vision only.** Perceive pixels; act on coordinates. No DOM/selectors/XPath. (Removes the #1 RPA fragility [28].)
2. **Deterministic before probabilistic.** Rule engine runs before LLM policy; predictable behavior wherever possible.
3. **Trust, but verify — every action.** Always screenshot-and-evaluate after acting; never assume an outcome [12].
4. **Recover, don't crash.** Escalate through an explicit ladder; hard-stop on no-progress loops; never claim unverified success.
5. **Human-in-the-loop on uncertainty.** Pause for confirmation/input instead of risking a wrong, irreversible action.
6. **Everything is observable.** Every run is a complete, inspectable, exportable artifact bundle.
7. **One loop, every surface.** Browser and desktop share loop, verifier, recovery, persistence.

---

## 9. Functional requirements

Legend: **[Current]** implemented today (verified in code); **[Roadmap]** proposed/aspirational.

### 9.1 Perception & action
- **FR-1 [Current]** Operate purely from visual perception (`UIElement` coordinates); no DOM/selectors.
- **FR-2 [Current]** Single action vocabulary end-to-end (perception → policy → executor → verifier): pointer, keyboard, navigation, clipboard, screenshot, upload, read-text, batch, stop, wait, and HITL actions (`src/operon/models/policy.py`).
- **FR-3 [Current]** TYPE actions are atomic from the policy's perspective.
- **FR-4 [Current]** Visual click-servo stability check before browser/desktop clicks.

### 9.2 Backends & models
- **FR-5 [Current]** Browser backend defaults to **Gemini 2.5 Computer Use** (`gemini-2.5-computer-use-preview-10-2025`) with an optional **JSON** perception/policy fallback.
- **FR-6 [Current]** Desktop backend uses the **JSON combined** backend.
- **FR-7 [Current]** Pluggable planner/verifier provider: **Gemini or Anthropic** (`OPERON_*_PLANNER_PROVIDER`).
- **FR-8 [Roadmap]** Pluggable grounding/vision models (e.g., OmniParser-style screen parsing [11], UI-TARS-style grounding [9]) to improve coordinate accuracy.

### 9.3 Policy & memory
- **FR-9 [Current]** Deterministic `PolicyRuleEngine` runs **before** LLM policy fallback (`PolicyCoordinator`).
- **FR-10 [Current]** Rolling spatial memory of recent elements; ghost/stale element handling.
- **FR-11 [Current]** Episodic memory and guardrails (`FileBackedMemoryStore`).
- **FR-12 [Roadmap]** User-authored allow/deny action policies (e.g., never click "Delete account") surfaced as first-class config — aligning with OWASP least-privilege/human-approval guidance [13].

### 9.4 Verification & recovery
- **FR-13 [Current]** Deterministic verifier classifies each step as `SUCCESS / FAILURE / UNCERTAIN / PENDING / PROGRESSING_STABLE / STABLE_WAIT`, with defined loop behavior (advance, backoff-and-re-verify, recover).
- **FR-14 [Current]** `RuleBasedRecoveryManager` ladder: retry → different tactic → context reset → session reset → stop; hard-stop on repeated no-progress; block unverified terminal success claims.

### 9.5 Human-in-the-loop
- **FR-15 [Current]** Pause/resume on uncertain or blocked states; user override hints; configurable confidence threshold that pauses when policy confidence is low.
- **FR-16 [Roadmap]** Explicit "high-risk action" confirmation gating (purchases, deletes, sends), mirroring Operator/Claude confirmation patterns [12][14].

### 9.6 Execution surfaces
- **FR-17 [Current]** **Browser executor** (`NativeBrowserExecutor`, Playwright/Chromium) incl. **observable mode** via CDP screencast over `/ws/stream`.
- **FR-18 [Current]** **Desktop executor** (`DesktopExecutor`, pyautogui/mss) for native app control (Windows-focused: DPI awareness, display-baseline checks, protected-process guards).
- **FR-19 [Current]** Headed-only native file upload via OS picker macro (`upload_file_native`).
- **FR-20 [Roadmap]** First-class macOS/Linux desktop parity (current desktop path is Windows-centric).

### 9.7 API & integration
- **FR-21 [Current]** FastAPI surface for browser + desktop run lifecycle (run-task, step, resume, stop, pause, get, cleanup, health), Observer/telemetry endpoints, and a `/ws/stream` WebSocket for live frames/events/controls (see §16).
- **FR-22 [Current]** Server-shutdown teardown releases executor resources (browser Chromium/Playwright and desktop recorders/child processes) via a uniform `aclose()`.

### 9.8 Observability & persistence
- **FR-23 [Current]** Every run persists state, step-log JSONL, before/after screenshots, perception/policy/execution/verification artifacts, and logs under `.var/runs/<run_id>/` (override via `OPERON_RUNS_ROOT`).
- **FR-24 [Current]** Observer API for runs, run snapshots, artifacts, **token-usage summaries**, run export, and live-browser frame.
- **FR-25 [Current]** Run replay from persisted artifacts; bounded retention sweep at server startup.
- **FR-26 [Current]** Per-provider token-usage/cost estimation (`estimate_usage_cost`).

### 9.9 Desktop application shell
- **FR-27 [Current]** Tauri desktop shell (`src-tauri`) for a packaged local app.

---

## 10. System architecture (overview)

```
FastAPI routes
  → AgentLoop                     (capture → perceive → decide → execute → verify → recover)
  → ScreenCaptureService
  → Gemini Computer Use  | JSON perception/policy backend   (browser)
    JSON combined backend                                   (desktop)
  → PolicyCoordinator + PolicyRuleEngine   (deterministic rules before LLM)
  → NativeBrowserExecutor | DesktopExecutor (selected by Environment enum)
  → DeterministicVerifierService           (6 outcome states)
  → RuleBasedRecoveryManager               (recovery ladder)
  → FileBackedRunStore + FileBackedMemoryStore + BackgroundWriter
Live view: BrowserManager (CDP screencast) → /ws/stream
```

Design invariants: browser and desktop share the loop, verifier, recovery, and persistence; the `Environment` enum selects the executor path; untrusted model output is constrained by the deterministic layer and re-checked by verification.

---

## 11. Non-functional requirements

- **NFR-1 Reliability.** The loop must verify every action and never advance on unverified success. Target: measurable task-success and recovery rates per release (see §12). *(Directly answers the category's top failure mode [12] and the reliability-driven project-cancellation risk [24].)*
- **NFR-2 Safety & security.** Treat all on-screen/web content as untrusted (prompt injection is OWASP LLM01 #1 [13]). Requirements: HITL gating on uncertainty **[Current]**; high-risk-action confirmation **[Roadmap, FR-16]**; allow/deny policies **[Roadmap, FR-12]**; sandbox/VM deployment guidance **[Roadmap]**; full audit trail **[Current]**. Adopt Anthropic/OWASP best practices (untrusted content isolation, least privilege, screening) [13][14].
- **NFR-3 Performance/latency.** Browser web tasks should stay competitive with low-latency CUAs (Gemini CUA reports ~70%+ web accuracy at ~225s/task class [8]); track latency-per-step and end-to-end task time per release.
- **NFR-4 Cost transparency.** Token usage and estimated cost must be visible per run; JSON fallback offers a cheaper path than frontier CUA tokens.
- **NFR-5 Portability.** Browser path cross-platform via Playwright; desktop path Windows-first today with macOS/Linux parity on the roadmap (FR-20). Python 3.11+ (3.14 for dev).
- **NFR-6 Observability.** 100% of runs produce a complete, exportable, replayable artifact bundle.
- **NFR-7 Privacy/data residency.** Self-hostable; runtime artifacts stay local under `.var/` by default.

---

## 12. Success metrics & KPIs

**Product/quality KPIs (internal, per release):**
- **Task success rate** (single-attempt) on a curated internal suite, segmented browser vs desktop.
- **Human-intervention rate** (fraction of runs needing HITL) — trending down for routine tasks, available for risky ones.
- **Recovery effectiveness** (share of `FAILURE/UNCERTAIN` steps resolved without human help).
- **Steps-to-completion** and **end-to-end task latency**.
- **Cost-per-successful-task** (tokens × price).
- **Unverified-success incidents** (target: zero — the loop must not claim success it didn't verify).
- **False-action rate** on high-risk actions (target: zero un-gated irreversible actions).

**External yardsticks (benchmark against, snapshots — see caveats):** OSWorld (desktop; human ~72% [16]), Online-Mind2Web and WebVoyager (web), AndroidWorld (mobile) [30]. These move fast and are often pass@k/self-reported — use as directional targets, not guarantees.

**Adoption KPIs (roadmap):** self-host deployments, API task volume, retention of automation definitions, community stars/contributions (peer OSS tier: browser-use ~98.5k★, Skyvern ~21.9k★, Agent-S ~11.8k★ [3][4][10]).

---

## 13. Release plan & roadmap

- **v0.1 (Current).** Vision-first browser + desktop loop; Gemini CU + JSON backends; deterministic rules; 6-state verifier; recovery ladder; HITL pause/resume + confidence threshold; click servo; observable browser via `/ws/stream`; Observer API + replay + usage; run artifacts + retention; uniform shutdown teardown; Tauri shell. Apache-2.0.
- **v0.2 (Near-term).** High-risk-action confirmation gating (FR-16); user-authored allow/deny policies (FR-12); hardened safety defaults (untrusted-content handling per [13][14]); expanded automated eval suite + published internal success metrics.
- **v0.3 (Mid-term).** macOS/Linux desktop parity (FR-20); pluggable grounding/vision models (FR-8); cost/latency dashboards.
- **v1.0 (Vision).** Non-technical supervised-run UX (persona Sam); curated reliability benchmarks published; enterprise governance (credential vaulting, RBAC) competitive with Copilot Studio's posture [11].

*(Roadmap is directional, not a commitment; sequencing subject to change.)*

---

## 14. Dependencies, assumptions, risks

### Dependencies
- Model providers: Google Gemini (Computer Use + flash), optionally Anthropic; Playwright/Chromium; pyautogui/mss (desktop); FastAPI/uvicorn; Tauri (shell).
- Network access to provider APIs (unless a future local-model path is added).

### Assumptions
- Frontier CUA models keep improving and remain API-accessible; Operon's value is the reliability/observability/unification layer **around** them, not the base model.
- Demand shifts from "most capable demo" to "most trustworthy run" as agentic projects hit the reliability wall [24].

### Risks & mitigations
| Risk | Severity | Mitigation |
|---|---|---|
| **Prompt injection via on-screen/web content** (OWASP LLM01 #1; multimodal injection hard to detect [13]) | High | Deterministic guardrails; treat content as untrusted; HITL gating; high-risk confirmation (FR-16); audit trail; adopt provider/OWASP defenses [13][14] |
| **Unverified/hallucinated actions** ("assumes outcomes without checking" [12]) | High | Verify-after-every-action; block unverified success; click servo; recovery ladder |
| **Reliability below human on hard tasks** (OSWorld agents < human [16][30]) | High | Scope to high-value reliable tasks first; HITL fallback; publish honest success metrics |
| **Frontier vendors commoditize CUA** (Anthropic/OpenAI/Google/Microsoft [11]) | Med | Compete on openness, unification, reliability, auditability, cost control — not model training |
| **Benchmark/marketing arms race** (pass@k, self-reported [30]) | Med | Measure single-attempt internal success; avoid overclaiming |
| **Desktop = Windows-centric today** | Med | Roadmap macOS/Linux parity (FR-20); be explicit about current support |
| **Provider cost/latency** | Med | JSON fallback; usage/cost visibility (FR-26); caching where possible |
| **Agentic-project cancellation wave** [24] | Med | Lead with measurable ROI on narrow, verifiable workflows; auditability for risk teams |

---

## 15. References

*Sourced via deep web research (2026-06-13). Benchmark and pricing figures are dated snapshots; several vendor first-party pages returned 403 to automated fetch and were corroborated via secondary reporting (noted inline during research). Market dollar figures are firm-dependent ranges.*

**Commercial CUAs**
1. Anthropic — Claude 3.5 models & computer use (Oct 2024). https://www.anthropic.com/news/3-5-models-and-computer-use
2. Anthropic — Claude Sonnet 4.5 (OSWorld 61.4%, $3/$15) (Sep 2025). https://www.anthropic.com/news/claude-sonnet-4-5
3. browser-use — GitHub. https://github.com/browser-use/browser-use
4. Skyvern — GitHub. https://github.com/Skyvern-AI/skyvern
5. OpenAI Operator (shutdown 2025-08-31; folded into ChatGPT Agent) — Wikipedia. https://en.wikipedia.org/wiki/OpenAI_Operator
6. OpenAI — Computer-Using Agent. https://openai.com/index/computer-using-agent/
7. OpenAI — `computer-use-preview` pricing ($3/$12 per 1M). https://platform.openai.com/docs/models/computer-use-preview
8. Google DeepMind — Gemini 2.5 Computer Use (browser/mobile; "not yet optimized for desktop OS") (Oct 2025). https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-computer-use-model/
9. UI-TARS (ByteDance) — GitHub. https://github.com/bytedance/UI-TARS
10. Agent-S (Simular) — GitHub. https://github.com/simular-ai/Agent-S
11. Microsoft Copilot Studio — computer use (GA; per-step credits; pluggable models). https://learn.microsoft.com/en-us/microsoft-copilot-studio/computer-use

**Safety / reliability**
12. Anthropic — computer use tool (verification-gap guidance; confirmation classifiers). https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool
13. OWASP — LLM01:2025 Prompt Injection. https://genai.owasp.org/llmrisk/llm01-prompt-injection/
14. Anthropic — prompt-injection defenses (23.6%→11.2%/1.4%). https://www.anthropic.com/news/prompt-injection-defenses
15. Anthropic — Claude Opus 4.5 System Card. https://assets.anthropic.com/m/64823ba7485345a7/Claude-Opus-4-5-System-Card.pdf

**Benchmarks**
16. OSWorld — project site (human ~72.36%). https://os-world.github.io/
17. OSWorld-Verified — XLANG. https://xlang.ai/blog/osworld-verified
30. "Illusion of Progress?" / Online-Mind2Web (WebVoyager inflation). https://arxiv.org/pdf/2504.01382

**Market**
18. Gartner — RPA software (~$7.01B by 2025, software-only). https://www.accio.com/business/gartner-rpa-predictions-trend
19. Grand View Research — RPA market. https://www.grandviewresearch.com/press-release/global-robotic-process-automation-rpa-market
21. Precedence Research — RPA market. https://www.precedenceresearch.com/robotic-process-automation-market
22. Gartner — 40% of enterprise apps with task-specific AI agents by 2026. https://www.gartner.com/en/newsroom/press-releases/2025-08-26-gartner-predicts-40-percent-of-enterprise-apps-will-feature-task-specific-ai-agents-by-2026-up-from-less-than-5-percent-in-2025
23. Gartner — agentic AI in a third of enterprise software by 2028. https://medium.com/@dappier/at-least-a-third-of-enterprise-software-will-be-agentic-by-2028-according-to-gartner-dont-wait-9070982ac6a7
24. Gartner — >40% of agentic AI projects cancelled by 2027. https://www.gartner.com/en/newsroom/press-releases/2025-06-25-gartner-predicts-over-40-percent-of-agentic-ai-projects-will-be-canceled-by-end-of-2027
25. MarketsandMarkets — AI Agents market (~$7.84B→$52.62B, 46.3% CAGR). https://www.marketsandmarkets.com/PressReleases/ai-agents.asp
27. Research and Markets — Automation Testing market (~$29.29B 2025 → $59.91B 2029). https://www.globenewswire.com/news-release/2025/01/29/3017343/28124/en/Automation-Testing-Market-Forecast-Report-2025.html
28. UiPath — healing agent / brittle-selector maintenance. https://www.uipath.com/blog/product-and-updates/technical-tuesday-how-healing-agent-solves-ui-automation-challenges
29. AIMultiple — RPA + computer vision (DOM-less Citrix/RDP). https://research.aimultiple.com/rpa-computer-vision/

---

## 16. Appendix A — Current API surface

```
# Browser run lifecycle
POST /run-task   POST /step   POST /resume   POST /stop
POST /run/{run_id}/stop   POST /run/{run_id}/pause
GET  /run/{run_id}   POST /cleanup   GET /health

# Desktop run lifecycle
POST /desktop/run-task   POST /desktop/step   POST /desktop/resume
POST /desktop/cleanup    GET  /desktop/run/{run_id}

# Observer / telemetry
GET /observer/api/runs   GET /observer/api/run/{run_id}
GET /observer/api/artifact   GET /observer/api/usage
GET /observer/api/export/{run_id}   GET /observer/api/live-browser/{run_id}

# Live stream
WS  /ws/stream
```

## 17. Appendix B — Action vocabulary

Pointer · keyboard · navigation · clipboard · screenshot · upload (`upload_file_native`, headed-only) · read-text · batch · stop · wait · HITL. Single vocabulary shared by perception, policy, executor, and verifier (`src/operon/models/policy.py`).

## 18. Appendix C — Glossary

- **CUA** — Computer-Using Agent.
- **DOM-less surface** — UI with no inspectable document model (e.g., Citrix/RDP/VDI, canvas apps), where vision is the only option.
- **Click servo** — visual-stability check performed before a click to avoid acting on a moving/loading UI.
- **Recovery ladder** — escalation sequence: retry → different tactic → context reset → session reset → stop.
- **pass@1 / pass@k** — success on a single attempt vs. best of k attempts (k inflates scores).
- **Observable mode** — live browser view streamed via CDP screencast over `/ws/stream`.
