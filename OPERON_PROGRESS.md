# Operon progress

Local build on branch `dev/local-build` (from `main` at b095d85), 2026-11-17. 50 commits,
107 files, about 11,000 lines added. Nothing is pushed or merged to `main`: every item below is
waiting for Vickey's review. The work order and history are in `docs/local-build-plan.md`; the
implemented design is in `OPERON_ARCHITECTURE.md`; test evidence is in `OPERON_TEST_REPORT.md`.

## How the work was done

Every item went through the repo's dev workflow (`CLAUDE.md`): a read-only scan, one coder per
fix in its own git worktree, and an independent verifier that did not write the change. Items
that failed verification went back to the coder until a fresh verifier passed them. Verifiers
returned 15 FAIL verdicts across the items; each was a real defect, fixed before landing.

All testing was local and free: no paid model calls, no benchmarks, no cloud runs.

## Completed capabilities

| Area | What landed | Flag (default) |
|---|---|---|
| Python SDK (KAN-35) | `from operon import Agent`; `run()`/`arun()`; approvals denied unless a callback approves; a policy deny is reported as `verdict="deny"`; one run per process | none |
| MCP server (KAN-96) | `operon mcp` over stdio: start/get/wait/approve/deny/stop/screenshot/list; never auto-approves; optional `[mcp]` extra | none |
| Task files (KAN-86) | `operon run --tasks file.yaml --junit report.xml`; state checks per task; policy denies never auto-approved; XML-safe report | none |
| Checkpoints (KAN-80) | Resume runs interrupted by a restart; owner-process liveness (pid + create time) so live runs are never taken over | `OPERON_CHECKPOINTS` (off) |
| Parallel runs (KAN-81) | Batch-mode browser runs each get their own Chromium; concurrency cap; 429 when full; desktop and live-view stay serial | `OPERON_PARALLEL_RUNS` (1) |
| Task-state note (KAN-102) | Running progress note in the model's prompt; typed text never echoed | `OPERON_TASK_NOTE` (off) |
| Trust categories (KAN-103) | Payment, send, delete, account, install confirmations; deny vs confirm verdicts | `OPERON_TRUST_CATEGORIES` (off) |
| Secrets vault (KAN-103) | Model types `{{secret:NAME}}` and never sees the value; DPAPI on Windows; typed via SendInput, never the clipboard; always needs approval; leak test over every run file | `OPERON_SECRETS` (off) |
| Focus guard (KAN-92) | Skip the first action after a screenshot if another process took focus | `OPERON_FOCUS_GUARD` (off) |
| DPI (KAN-90) | Per-monitor-v2 awareness set before pyautogui loads; drift warning only when real | `OPERON_DPI_AWARE` (off) |
| Efficiency score (KAN-85) | Steps, time, wasted steps incl. a screen-revisit signal; shown by the API and evals | none |
| Windows Sandbox (KAN-93) | Direct launch, logon marker, early-close detection with one retry, clear timeout diagnostics | none |
| Windows CI (KAN-160) | Offline suite on `windows-latest` (job defined, not run here) | none |
| Repo hygiene (KAN-34) | Code of conduct, CODEOWNERS, least-privilege CI permissions | none |

## Safety bugs found and fixed (default behaviour, no flag)

These were found by the local end-to-end work and code scans, and are bug fixes, so they apply
by default:

1. **Blind recovery clicks.** The recovery ladder's context reset clicked a fixed screen point
   with no trust gate, servo or focus check, so it could hit a Pay or Delete button. It now clicks
   only a verified plain-background spot inside the work window, or sends keys only. Desktop
   session reset no longer alt+tabs; browser session reset navigates only to a gate-allowed
   start URL (reload otherwise).
2. **Unchanged screen verified as success.** With the critic unavailable, a dead click counted as
   SUCCESS, so recovery never escalated. Now a full-resolution pixel compare marks a truly
   unchanged screen UNCERTAIN; small real changes (a checkbox tick) stay SUCCESS.
3. **Stop without verified work.** When the critic gives no verdict, a stop now needs at least
   one verified-success step and no unrecovered recent failure.
4. **Re-grounded retries skipped the trust gate.** A retry onto a different element is re-gated.
5. **Config drift.** `.env.example` disagreed with code defaults (zoom, headless); fixed, with a
   test that fails on future drift.

## Current status

- Offline suite, scripted evals, lint, type check and the real-Chromium end-to-end suite all pass
  (numbers in `OPERON_TEST_REPORT.md`).
- Windows Sandbox end-to-end: 3 of 3 pass on a quiet machine, twice in a row.
- Fixed after the sandbox run: `launch_app` reported success even when no window appeared
  (Notepad is not installed in the sandbox image). It now fails up front when the program is on
  neither PATH nor App Paths, and otherwise waits for a new window. It also launches
  Chrome, Edge and Office through their App Paths registration; before, `cmd /c` could never find
  them.
- The final combined regression after the launch fixes was stopped by Claude Code for low
  memory (8 GB PC). The launch commits each passed the full offline suite in their verifier's
  worktree (2794 passed). Rerun the commands in `OPERON_TEST_REPORT.md` once to confirm the
  merged tip.

## Known limitations and open items

- **No real model was run.** There is no local model runtime on this PC (no Ollama, LM Studio,
  llama.cpp or vLLM) and no NVIDIA GPU. All end-to-end runs use a scripted model at the client
  boundary, so they prove the pipeline (capture, decide, gate, execute, verify, recover), not the
  model's reasoning. Model quality needs either a paid run (Vickey's approval) or a local
  OpenAI-compatible server for the `openai_compat` profiles.
- **Critic is the last line of defence for wrong "done" claims.** Operon's own guards reject a
  stop with no work, but a stop after a wrong click is accepted when the critic approves it. A
  strict xfail in `tests/browser_e2e` documents this.
- **Vision-only by design.** The spec asked for DOM and accessibility-tree perception; this repo's
  invariant forbids them in perception and policy, so they were not added.
- **Reset click on a blank drawing canvas** could still draw (a plain canvas looks like
  background). Low risk; follow-up.
- **Windows Sandbox needs a quiet machine** and `OPERON_SANDBOX_MEMORY_MB=2048` on 8 GB RAM.
- Parallel runs cover batch browser mode only; the live view (`/ws/stream`) shows no frames for
  a parallel run.
- `PROGRESSING_STABLE` is defined but no verifier emits it.
- The MCP stdio smoke test against a real client was blocked by this session's permission
  classifier; run `python mcp_smoke.py` style checks by hand (in-memory client tests pass).
- New flags default off; turning any on by default needs a paid grid run.

## Next priorities

1. Vickey reviews `dev/local-build` and decides what to PR (each item is a separate commit).
2. Run one real-model smoke task (Gemini, about $0.30) to confirm nothing regressed under a
   real model, especially the verifier and recovery changes.
3. Close the open items above, starting with the critic-only stop check.
