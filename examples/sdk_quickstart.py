"""Operon Python SDK quickstart (KAN-35). Needs GEMINI_API_KEY; see docs/sdk.md."""

from operon import Agent

# Browser task in an isolated Chromium. Steps that need approval are denied by default.
result = Agent().run("find the weather in Paris", url="https://example.com")
print(result.status, result.final_answer, result.steps, result.cost_usd, result.run_id)


# To approve some steps yourself, pass a callback.
def on_approval(request) -> bool:
    print("Approval needed:", request.reason)
    return input("approve? [y/N] ").strip().lower() == "y"


# Desktop task (Windows Sandbox environment):
# result = Agent(desktop=True, env="windows-sandbox", on_approval=on_approval).run("open Notepad and type hi")
