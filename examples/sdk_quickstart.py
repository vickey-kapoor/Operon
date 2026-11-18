"""Operon Python SDK quickstart (KAN-35). Needs GEMINI_API_KEY; see docs/sdk.md."""

from operon import Agent, ApprovalRequest


def on_approval(request: ApprovalRequest) -> bool:
    print("Approval needed:", request.reason)
    return input("approve? [y/N] ").strip().lower() == "y"


# Browser task in an isolated Chromium. Without on_approval, steps that need approval are denied.
result = Agent(on_approval=on_approval).run("find the weather in Paris", url="https://example.com")
print(result.status, result.final_answer, result.steps, result.cost_usd, result.run_id)

# Desktop task (Windows Sandbox): the default "live" env would drive your real mouse and keyboard.
# result = Agent(desktop=True, env="windows-sandbox", on_approval=on_approval).run("open Notepad and type hi")
