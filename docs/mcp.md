# Operon as an MCP server

`operon mcp` serves Operon over stdio, so Claude Code, Claude Desktop and other
[MCP](https://modelcontextprotocol.io) clients can hand it computer-use tasks and follow them.
Tasks run in the `operon mcp` process, through the same front door as `operon run`.

## Install

The MCP SDK is an optional extra:

```
pip install "operon-agent[mcp]"
operon setup          # saves your Gemini key and installs Chromium
```

Without the extra, `operon mcp` prints this hint and exits with code 2. Nothing else in Operon needs it.

## Connect a client

Claude Code:

```
claude mcp add operon -- operon mcp
```

Claude Desktop (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "operon": { "command": "operon", "args": ["mcp"] }
  }
}
```

If `operon` is not on the client's PATH, give the full path to the executable. The server reads your
key from `~/.operon` (saved by `operon setup`) or the environment.

## Tools

Runs take minutes, so no tool waits for a run to end. Start it, then wait or poll.

| Tool | What it does |
|---|---|
| `start_task(task, url?, desktop=false, env?)` | Starts a run and returns `{run_id, status}` at once |
| `get_task(run_id)` | Status, step count, last message, what it waits on (`waiting.reason`, `waiting.category`), and once done the `final_answer`, `summary` and `cost_usd` |
| `wait_for_task(run_id, timeout_s=60)` | Polls until the run is done, waits for you, or the timeout passes (at most 300), then returns `get_task`'s result |
| `approve(run_id)` / `deny(run_id)` | Answer an action the run paused on |
| `stop_task(run_id)` | Cancels the run |
| `get_screenshot(run_id)` | The latest step screenshot as an image, at most 1280 px on the long edge |
| `list_environments()` | The desktop environments and whether each is ready (like `operon envs`) |

Workflow: `start_task`, then `wait_for_task`. When it returns `waiting_for_user`, read
`waiting.reason`, ask the person, then `approve` or `deny`, and wait again. When `done` is true,
`final_answer` holds the result.

## Safety

- The permission mode is always `ask` and the trust gate stays on. A paused action waits until the
  client calls `approve`; Operon never approves for it. Clients should approve only when the person
  agrees.
- `desktop=true` on the default `live` environment drives this computer's real mouse and keyboard.
  Browser runs are the default. Use `env` (for example `linux-container`) to keep a desktop run off
  your own screen; see [environments](environments.md).
- Screenshots capture the whole screen, so `get_screenshot` can show anything open there.
- One run at a time: `start_task` is refused while another run is active. Stop it or wait for it.
- stdout carries the protocol, so Operon's messages go to stderr.
