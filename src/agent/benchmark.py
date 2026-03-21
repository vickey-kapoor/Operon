"""Minimal local entry points for the active form benchmark and optional Gmail benchmark."""

from __future__ import annotations

import asyncio
import os

from dotenv import find_dotenv, load_dotenv

from src.agent.capture import BrowserCaptureService
from src.agent.loop import AgentLoop
from src.agent.policy_coordinator import PolicyCoordinator
from src.agent.perception import GeminiPerceptionService
from src.agent.policy import GeminiPolicyService
from src.agent.recovery import RuleBasedRecoveryManager
from src.agent.verifier import DeterministicVerifierService
from src.clients.gemini import GeminiHttpClient
from src.executor.browser import PlaywrightBrowserExecutor
from src.store.memory import FileBackedMemoryStore
from src.store.run_store import FileBackedRunStore

DEFAULT_FORM_BENCHMARK_INTENT = "Complete the auth-free form and submit it successfully."
DEFAULT_FORM_BENCHMARK_URL = "https://practice-automation.com/form-fields/"
DEFAULT_GMAIL_BENCHMARK_INTENT = "Create a Gmail draft and stop before send."
DEFAULT_GMAIL_BENCHMARK_URL = "https://mail.google.com/"

async def run_form_benchmark(max_steps: int = 12):
    """Run the browser-only auth-free form benchmark until a terminal condition is reached."""
    load_dotenv(find_dotenv(usecwd=True), override=False)

    gemini_client = GeminiHttpClient()
    browser_executor = PlaywrightBrowserExecutor()
    run_store = FileBackedRunStore()
    memory_store = FileBackedMemoryStore()
    policy_service = PolicyCoordinator(
        delegate=GeminiPolicyService(gemini_client=gemini_client),
        memory_store=memory_store,
    )
    loop = AgentLoop(
        capture_service=BrowserCaptureService(browser_executor=browser_executor),
        perception_service=GeminiPerceptionService(gemini_client=gemini_client),
        run_store=run_store,
        policy_service=policy_service,
        browser_executor=browser_executor,
        verifier_service=DeterministicVerifierService(gemini_client=gemini_client),
        recovery_manager=RuleBasedRecoveryManager(),
        memory_store=memory_store,
    )
    try:
        return await loop.run_live_benchmark(
            intent=DEFAULT_FORM_BENCHMARK_INTENT,
            benchmark_url=os.getenv("FORM_BENCHMARK_URL", DEFAULT_FORM_BENCHMARK_URL),
            max_steps=max_steps,
        )
    finally:
        await browser_executor.close()


async def run_gmail_draft_benchmark(max_steps: int = 12):
    """Run the optional Gmail draft benchmark until a terminal condition is reached."""
    load_dotenv(find_dotenv(usecwd=True), override=False)

    gemini_client = GeminiHttpClient()
    browser_executor = PlaywrightBrowserExecutor()
    run_store = FileBackedRunStore()
    memory_store = FileBackedMemoryStore()
    policy_service = PolicyCoordinator(
        delegate=GeminiPolicyService(gemini_client=gemini_client),
        memory_store=memory_store,
    )
    loop = AgentLoop(
        capture_service=BrowserCaptureService(browser_executor=browser_executor),
        perception_service=GeminiPerceptionService(gemini_client=gemini_client),
        run_store=run_store,
        policy_service=policy_service,
        browser_executor=browser_executor,
        verifier_service=DeterministicVerifierService(gemini_client=gemini_client),
        recovery_manager=RuleBasedRecoveryManager(),
        memory_store=memory_store,
    )
    try:
        return await loop.run_live_benchmark(
            intent=DEFAULT_GMAIL_BENCHMARK_INTENT,
            benchmark_url=os.getenv("GMAIL_BENCHMARK_URL", DEFAULT_GMAIL_BENCHMARK_URL),
            max_steps=max_steps,
        )
    finally:
        await browser_executor.close()


if __name__ == "__main__":
    result = asyncio.run(run_form_benchmark())
    print(result.model_dump_json(indent=2))
