"""Task clarification — asks Gemini to identify ambiguous inputs before execution."""

import asyncio
import json
import logging
import os
from typing import List, Optional

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

_CLARIFY_PROMPT = """\
You are a task-intake assistant for an AI browser agent.

Given a user's task description, identify any information that is MISSING or AMBIGUOUS \
and that the agent would genuinely need in order to execute the task successfully.

Return a JSON object with a single key "questions" containing a list of short, specific \
clarifying questions (plain strings). Return at most 4 questions.

Rules:
- Only ask if the missing info would actually block or significantly change execution.
- Do NOT ask about things that can be reasonably inferred or looked up (e.g. do not ask \
  "what website should I use?" — the agent can decide that).
- Do NOT ask unnecessary questions for tasks that are already clear and actionable.
- Keep each question concise (under 15 words).
- If the task is already clear enough to execute, return {"questions": []}.

Examples:
  Task: "search for python tutorials"
  → {"questions": []}          # clear enough

  Task: "book flight tickets to Paris"
  → {"questions": ["From which city?", "What travel dates?", "How many passengers?", "One-way or round trip?"]}

  Task: "find a hotel"
  → {"questions": ["Which city or destination?", "Check-in and check-out dates?", "What is your budget per night?"]}

  Task: "order pizza"
  → {"questions": ["What toppings or type of pizza?", "Delivery address?"]}

  Task: "go to example.com and return the page title"
  → {"questions": []}          # fully specified

Respond ONLY with the JSON object — no markdown, no prose.
"""


class TaskClarifier:
    """Uses Gemini (text-only) to surface clarifying questions before task execution."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None) -> None:
        resolved_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not resolved_key:
            raise ValueError("GOOGLE_API_KEY is required for TaskClarifier")

        self._client = genai.Client(api_key=resolved_key)
        self._model = model or "gemini-2.5-flash"
        self._config = types.GenerateContentConfig(
            system_instruction=_CLARIFY_PROMPT,
            temperature=0.1,
            max_output_tokens=512,
            response_mime_type="application/json",
        )

    async def get_questions(self, task: str) -> List[str]:
        """
        Return a list of clarifying questions for the given task.
        Returns an empty list if the task is already clear enough to execute.
        """
        try:
            raw = await asyncio.get_running_loop().run_in_executor(
                None, self._call, task
            )
            data = json.loads(raw)
            questions = data.get("questions", [])
            if not isinstance(questions, list):
                return []
            return [str(q).strip() for q in questions if str(q).strip()]
        except Exception as exc:
            logger.warning("Clarifier failed (skipping): %s", exc)
            return []

    def _call(self, task: str) -> str:
        response = self._client.models.generate_content(
            model=self._model,
            contents=[types.Content(
                role="user",
                parts=[types.Part.from_text(text=f'Task: "{task}"')],
            )],
            config=self._config,
        )
        return response.text or '{"questions": []}'
