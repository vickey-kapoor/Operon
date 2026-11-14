# RFC 0002 — A tool action channel next to pixel actions

| | |
|---|---|
| **Status** | Built behind `OPERON_TOOLS` (default off); measured in KAN-87 part 2 |
| **Created** | 2026-11-13 |
| **Story** | KAN-79 (epic KAN-77, workstream A1) |
| **Related** | [`../architecture.md`](../architecture.md), [`../configuration.md`](../configuration.md) |

> RFCs are design proposals, not implementation truth. The code and
> [`../architecture.md`](../architecture.md) stay authoritative.

## 1. Summary

Let the model call a small set of tools (read a file, list a folder, write a file, run a
shell command, fetch a URL) in the same turn as its clicks and key presses, and get the
tool's output back as the answer to that call. Perception does not change: the model
still sees the screen only as pixels, and every turn still carries a fresh screenshot.

Each tool comes with a short **skill** note (`src/operon/prompts/tools/<tool>.md`) that tells
the model when the tool is the better choice and when to stay on the screen.

The channel ships behind `OPERON_TOOLS`, off by default, so nothing changes for a run that
doesn't ask for it.

## 2. Motivation

- Other computer-use agents mix GUI actions with tool calls (a shell, a file API, HTTP,
  MCP servers), because a lot of desktop work is mechanical: reading a CSV, renaming forty
  files, checking an API. Doing it through the GUI costs steps, tokens and misclicks.
- Operon's own calibration shows the cost: on Season 2 Desktop (KAN-87, KAN-104), Gemini
  already opens a terminal and types PowerShell to do file work, spending 5 to 30 steps of
  screenshots on what one tool call does.
- The decision on the architecture page (Confluence 1933314) stays: **pixels are the only
  perception.** Tools are an *action* channel. A tool's output is data the model asked
  for, never a description of the screen (no DOM, no accessibility tree).

## 3. Goals / Non-goals

**Goals**
- Tool calls and pixel actions in one conversation, one loop, one run record.
- Every tool call goes through the same gates as a GUI action: trust gate, permission
  mode, file guard, step budget.
- A clean A/B: the same tasks with `OPERON_TOOLS` off and on (KAN-87 part 2).

**Non-goals (this RFC)**
- MCP servers. The tool registry is built so an MCP client can register tools later
  (§8), but no MCP client ships here.
- Reading the screen through tools (window text, DOM, UI Automation). Vision-only stays.
- Changing the full harness's rules for tools. Tools pass through it like any action.

## 4. Design

### 4.1 Tools in v1

| Tool | Arguments | Returns | Notes |
|---|---|---|---|
| `read_file` | `path` | the file's text (first 20,000 characters) | file guard applies |
| `list_dir` | `path` | one line per entry: name, `dir` or size in bytes | file guard applies |
| `write_file` | `path`, `content` | bytes written | file guard applies; creates parent folders |
| `run_shell` | `command`, optional `timeout_seconds` (max 120) | exit code, stdout, stderr (each cut at 8,000 characters) | PowerShell on Windows, bash on Linux; the trust gate asks for approval |
| `http_get` | `url` | status code, content type, body text (first 20,000 characters) | the trust gate's domain allowlist applies |

Relative paths and the shell's working folder are the user's home folder.

### 4.2 Turning it on

`OPERON_TOOLS`:

- `off` (default): no tools are declared; Operon behaves exactly as before.
- `on`: `read_file`, `list_dir`, `write_file`, `http_get`. No shell.
- `all`: the four above plus `run_shell`.
- a comma list of tool names, e.g. `read_file,list_dir`.

### 4.3 How a call flows

1. **Declare.** The Gemini computer-use client sends the enabled tools as
   `functionDeclarations` next to the `computerUse` tool (as `launch_app` already is on the
   desktop). The backend appends the skill notes to the prompt.
2. **Translate.** A call to a tool becomes `AgentAction(action_type=tool, tool=<name>,
   tool_args={...})`. Its `text` is a short summary (`run_shell: <command>`,
   `read_file: <path>`) and an `http_get` also sets `url`, so the trust gate and the
   domain allowlist judge it like any other action. Several calls in one turn become a
   batch, as today, and may mix tools with clicks.
3. **Gate.** The loop's gates already judge every leaf of a batch. Additions:
   - the file guard checks a file tool's `path` (run folder, `OPERON_ALLOWED_FILE_ROOTS`, or
     a path named in the task);
   - with the trust gate on, `run_shell` always needs approval (`auto` mode skips it,
     deny rules still block);
   - `ask` mode approves every tool call, as it does every GUI action.
4. **Run.** `ToolExecutor` wraps the environment's executor. It runs tool actions itself
   and hands everything else to the wrapped executor, so the GUI executors are unchanged.
   In a mixed batch it runs the children in order and stops at the first failure, like the
   GUI batch.
5. **Answer.** The executed action carries `tool_outputs`. When the backend answers the
   model's calls, each tool call's `function_response` holds its `output` (or `error`), and
   the turn still ends with the new screenshot.

A tool that ran is a success even when the command inside it failed (a non-zero exit code
is output for the model to read). A tool call fails only when it could not run: bad
arguments, a blocked path, a timeout, an unreachable URL.

### 4.4 Skill notes

One short Markdown note per tool, in `src/operon/prompts/tools/`. Each says what the tool
is for, when the screen is still the right way (anything the user has to see, anything
only shown on screen) and one example. They are package data, loaded through
`core.paths.prompts_dir()` like the other prompts.

## 5. Safety

| Risk | Control |
|---|---|
| Model reads or writes files it shouldn't | File guard on every file tool's path; it pauses the run, and resuming doesn't override it |
| Shell command does damage | `run_shell` is only in `all` or an explicit list; with the trust gate on it always needs approval (unless the run chose `auto`); deny phrases match the command |
| Prompt injection asks for exfiltration over HTTP | `http_get` only; the domain allowlist applies; no request bodies or custom headers |
| Large outputs blow up the conversation | Outputs are cut (8,000 or 20,000 characters) and marked as cut |
| A tool hangs | Shell timeout (default 30 s, max 120 s); HTTP timeout 20 s |

Benchmarks turn the trust gate off (as today), so tools-on runs only rely on the file
guard and the step budget; the benchmark workflow sets `OPERON_ALLOWED_FILE_ROOTS` to the
task folders' parent.

## 6. Measurement (KAN-87 part 2)

Run Season 2 Desktop and Hybrid, Thin harness, with tools off and on (`/bench ... tools`),
and compare score, steps, cost and time per task. Expected: file-heavy tasks get faster and
cheaper; tasks whose data is only on screen (the KAN-104 pictures) don't change; safety
tasks must not get worse. Browser is measured once with `http_get` only.

## 7. Alternatives considered

- **Tools only, no pixels for file work.** Rejected: the product is a computer-use agent,
  and many tasks only exist on screen.
- **Run tools in a separate sub-agent.** More moving parts and a second conversation to
  keep in sync; one conversation is simpler and is how other CUA agents do it.
- **Put the tools in the GUI executors.** Each executor (Windows, Linux, browser) would
  need the same code; the wrapper keeps it in one place.

## 8. Later

- MCP: an MCP client registers each server tool in the same registry, with its own skill
  note, and its calls go through the same gates.
- Per-tool categories and confirmations, secret redaction in outputs and audit (KAN-103).
- Promote to on by default once the A/B shows it helps and safety tasks hold.
