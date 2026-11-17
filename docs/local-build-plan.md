# Local build plan: finishing Operon with $0 testing

Started 2026-11-17 on branch `dev/local-build`. This is the work order for building Operon
toward a complete computer-use agent with **local testing only**: no paid model calls, no
benchmarks, no GPU VMs. Every item is proven with the offline suite, scripted models, replays,
and the free real desktops on this PC (Windows Sandbox, the Linux desktop container).

Rules for this plan:

- Each item follows the dev workflow in `CLAUDE.md`: plan, scan, code in a worktree, independent
  verify, then it lands on `dev/local-build`. Nothing is pushed or merged without Vickey.
- New behaviour ships behind an `OPERON_*` flag that defaults to off, and is added to
  `Settings` and `.env.example`.
- The invariants in `CLAUDE.md` hold: vision-only, rules before LLM (full harness only), thin
  harness untouched by rules, no scenario hardcoding, atomic TYPE, visual servo never bypassed.
- Turning a flag on by default needs a paid grid run, so that decision is left to Vickey.

## Status found on 2026-11-17

Checked against the code by read-only scans. "Partial" lists what is missing.

| Story | Status | Missing |
|---|---|---|
| KAN-93 Windows Sandbox | Built | No end-to-end test against a real sandbox |
| KAN-92 focus steal | Partial | No foreground check before click or type; only restore after a human pause |
| KAN-90 DPI-safe clicks | Partial | Baseline reads system DPI only; no per-monitor DPI on the click's monitor; no mixed-DPI test |
| KAN-91 GPU capture | Not started | Capture is `mss` (GDI) everywhere |
| KAN-80 checkpoints and resume | Built behind `OPERON_CHECKPOINTS` | Providers built; interrupted runs are marked at startup and resumed on request |
| KAN-81 parallel runs | Partial | Runs are separate tasks but share one serialized executor; no concurrency cap |
| KAN-85 efficiency | Partial | Batching and loop detection built; no step-efficiency score; no batching on the Gemini desktop backend |
| KAN-102 task-state note | Partial | Full-harness verify-before-done built; no running task-state note carried between steps |
| KAN-103 safety | Partial | Confirm phrases built; no named categories (pay, send, delete); no secrets vault; no redacted audit bundle |
| KAN-35 Python SDK | Not started | No public `operon.Agent` |
| KAN-96 / KAN-40 MCP server, Docker | Not started | No MCP server; Docker image is only the Linux guest desktop |
| KAN-86 task files, replay | Partial | Replay checks the action sequence only; no YAML task files; no JUnit report; no `operon replay` |
| KAN-160 Windows CI | Not started | CI runs on Ubuntu only |
| KAN-34 public repo hygiene | Partial | No CODE_OF_CONDUCT, no CODEOWNERS, `ci.yml` has no `permissions:` block |
| `loop.py` split | Not started | 2514 lines in one file |

## Progress (updated 2026-11-17)

Landed on `dev/local-build`, each verified by an independent reviewer. The offline suite is
2643 passed; `python -m evals` is 22/23 (the miss is the known RFC 0003 scenario).

| Story | What landed | Flag |
|---|---|---|
| KAN-34 | Code of conduct, CODEOWNERS, least-privilege `ci.yml` permissions | none |
| KAN-160 | Windows CI job for the offline suite | none |
| KAN-92 | Focus-steal guard: the first action after each screenshot is skipped if another process took focus | `OPERON_FOCUS_GUARD` |
| KAN-35 | Python SDK: `from operon import Agent`; approvals denied by default; one run per process | none |
| KAN-103 (part 1) | Named risk categories in the trust gate (payment, send, delete, account, install) | `OPERON_TRUST_CATEGORIES` |
| KAN-102 | Running task-state note in the model's prompt; typed text never echoed | `OPERON_TASK_NOTE` |
| KAN-90 | Per-monitor-v2 DPI awareness set before pyautogui loads; drift warning only when real | `OPERON_DPI_AWARE` |
| KAN-85 | Step-efficiency score per run, including a screen-revisit waste signal | none (measurement) |
| KAN-86 (part 1) | `operon run --tasks file.yaml --junit report.xml`; policy denies are never auto-approved | none |
| KAN-96 | `operon mcp` stdio MCP server (optional `[mcp]` extra) | none |
| KAN-80 | Resume interrupted runs after a restart, with owner-process liveness | `OPERON_CHECKPOINTS` |

In progress: secrets vault (KAN-103 part 2), parallel browser runs (KAN-81), and the Windows
Sandbox end-to-end test (KAN-93), whose guest stopped starting on this PC during the session.

## Work order

Grouped into batches. Items in a batch touch different files, so their coders run in parallel.

### Batch 1: test bed and public surface

1. **Windows Sandbox end-to-end test** (KAN-93). The real loop drives a real Windows desktop in
   the sandbox with a scripted model, like `tests/desktop_e2e/test_linux_desktop.py`. Marker
   `windows_sandbox`, deselected by default. This is the $0 proof for the desktop items after it.
2. **Python SDK** (KAN-35). `from operon import Agent`; `Agent(...).run("task")` returns a typed
   result. Safe defaults: trust gate on, `ask` mode, batch browser. Tested with a scripted model.
3. **Repo hygiene** (KAN-34). CODE_OF_CONDUCT, CODEOWNERS, least-privilege `permissions:` in `ci.yml`.

### Batch 2: desktop reliability

4. **Focus-steal guard** (KAN-92). Before a click or type, check the foreground window is the
   one the model saw; if another window stole focus, skip the action and re-observe. Flag
   `OPERON_FOCUS_GUARD`.
5. **Step-efficiency score** (KAN-85). Steps, seconds per step, and wasted steps (repeats,
   no-ops) per run, saved in the run summary and shown by the evals report.

### Batch 3: long and safe runs

6. **Checkpoint and resume** (KAN-80). Save the run state each step; `operon serve` lists
   interrupted runs and resumes one after a restart. Flag `OPERON_CHECKPOINTS`.
7. **Secrets vault** (KAN-103). The model types a placeholder such as `{{secret:bank_password}}`;
   the executor swaps in the value from a local vault, so the model and the logs never see it.
8. **Redacted audit bundle** (KAN-103). `operon audit <run_id>` writes a zip of a run with
   secrets scrubbed from text and typed-secret regions noted.

### Batch 4: developer platform

9. **MCP server** (KAN-96). `operon mcp` serves Operon as tools over stdio (`run_task`,
   `get_run`, `approve`, `stop`), so Claude Code and other agents can drive a computer.
10. **YAML task files and JUnit** (KAN-86). `operon run --tasks file.yaml --junit out.xml`, each
    task with an optional state check.
11. **DPI per monitor** (KAN-90). Use the DPI of the monitor the click lands on; tests with
    faked mixed-DPI monitors.

### Later, if budget remains

- Task-state note for the model (KAN-102), behind a flag.
- Concurrency cap and one executor per run for parallel runs (KAN-81).
- GPU capture with Windows Graphics Capture (KAN-91), behind a flag.
- Split `loop.py` along its stage seams.
- Windows CI job (KAN-160).

## Not in this plan

Anything that needs paid model calls or Vickey's decisions: benchmarks and the grid, turning a
flag on by default, the desktop app shell (KAN-122 decide), launch, pricing, publishing.
