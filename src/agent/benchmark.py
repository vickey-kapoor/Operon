"""Minimal local entry point for the Gmail draft benchmark loop."""

from __future__ import annotations

import asyncio

from dotenv import find_dotenv, load_dotenv

from src.agent.capture import BrowserCaptureService
from src.agent.loop import AgentLoop
from src.agent.perception import GeminiPerceptionService
from src.agent.policy import GeminiPolicyService
from src.agent.recovery import RuleBasedRecoveryManager
from src.agent.verifier import DeterministicVerifierService
from src.clients.gemini import GeminiHttpClient
from src.executor.browser import PlaywrightBrowserExecutor
from src.store.run_store import FileBackedRunStore


async def run_gmail_draft_benchmark(max_steps: int = 12):
    """Run the browser-only Gmail draft benchmark until a terminal condition is reached."""
    load_dotenv(find_dotenv(usecwd=True), override=False)

    gemini_client = GeminiHttpClient()
    browser_executor = PlaywrightBrowserExecutor()
    run_store = FileBackedRunStore()
    loop = AgentLoop(
        capture_service=BrowserCaptureService(browser_executor=browser_executor),
        perception_service=GeminiPerceptionService(gemini_client=gemini_client),
        run_store=run_store,
        policy_service=GeminiPolicyService(gemini_client=gemini_client),
        browser_executor=browser_executor,
        verifier_service=DeterministicVerifierService(gemini_client=gemini_client),
        recovery_manager=RuleBasedRecoveryManager(),
    )
    try:
        return await loop.run_live_benchmark(max_steps=max_steps)
    finally:
        await browser_executor.close()


if __name__ == "__main__":
    result = asyncio.run(run_gmail_draft_benchmark())
    print(result.model_dump_json(indent=2))
