# Python SDK

Run one Operon task from Python. It uses the same in-process path as `operon run`.

```python
from operon import Agent

result = Agent().run("find the weather in Paris", url="https://example.com")          # browser
result = Agent(desktop=True, env="windows-sandbox").run("open Notepad and type hi")  # desktop
print(result.status, result.final_answer, result.steps, result.cost_usd, result.run_id)
```

`GEMINI_API_KEY` must be set (or saved with `operon setup`). A runnable example is in
`examples/sdk_quickstart.py`.

## `Agent(...)`

| Argument | Default | Meaning |
|---|---|---|
| `desktop` | `False` | Drive the desktop instead of a browser |
| `env` | `None` | Desktop environment (see [environments](environments.md)); needs `desktop=True`. Sets `OPERON_ENVIRONMENT` for the run |
| `mode` | `"ask"` | Permission mode: `ask`, `risky` or `auto` |
| `max_steps` | `None` | Step budget; `None` uses the front door default (25) |
| `on_approval` | `None` | `callable(ApprovalRequest) -> bool` (sync or async) |

`run(task, url=None)` blocks and returns a `RunResult`. Inside a running event loop (a
notebook, FastAPI, ...) call `await agent.arun(task, url=None)` instead; `run()` raises a
clear error there.

`RunResult` is frozen: `status`, `final_answer`, `steps`, `cost_usd` (None when unknown),
`run_id`, `artifacts_dir`, `summary`, `denied_reason`.

## Safe defaults and approvals

The trust gate is on, the mode is `ask`, and a browser run uses an isolated batch Chromium.
When a step needs approval, the SDK calls `on_approval(ApprovalRequest(run_id, reason, kind, gate))`.
Return `True` to approve. Without a callback, or when it returns `False`, the run is stopped and
the result has `status == "denied"` and `denied_reason` set. The SDK never approves on its own.
A run handed to a person (takeover) is stopped the same way.

Screenshots in `artifacts_dir` capture the whole screen: review them before sharing.
