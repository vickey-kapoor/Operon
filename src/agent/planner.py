"""Action planning logic — converts Gemini responses into structured ActionPlans."""

import json
import logging
import re
from typing import List, Optional

from google.genai import types as genai_types
from pydantic import BaseModel, Field, ValidationError, field_validator

from src.executor.actions import Action, ActionType

logger = logging.getLogger(__name__)


class ActionPlan(BaseModel):
    """A complete action plan produced by the vision agent for one step."""

    observation: str = Field(
        ...,
        description="Detailed description of what the agent sees on screen",
    )
    reasoning: str = Field(
        ...,
        description="Step-by-step thinking about what actions to take",
    )
    actions: List[Action] = Field(
        default_factory=list,
        description="List of actions to execute in order",
    )
    done: bool = Field(
        False,
        description="True only when the user's task is fully complete",
    )
    result: Optional[str] = Field(
        None,
        description="Summary of the completed task result when done=True",
    )

    @field_validator("result", mode="before")
    @classmethod
    def coerce_result_to_str(cls, v):
        """Gemini sometimes returns result as a dict — serialize it to a string."""
        if v is None or isinstance(v, str):
            return v
        return json.dumps(v)


def _extract_json_from_text(text: str) -> str:
    """
    Attempt to extract a JSON object from arbitrary text.

    Gemini sometimes wraps its response in markdown fences or other prose.
    This helper tries several strategies to locate the actual JSON payload.
    """
    # 1. Try raw parse first (ideal case).
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    # 2. Look for ```json ... ``` blocks.
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    # 3. Look for the first `{` to the last `}`.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    raise ValueError("No JSON object found in model response")


class ActionPlanner:
    """
    Calls the GeminiVisionClient and parses the response into an ActionPlan.

    Includes retry logic to handle malformed JSON responses from the model.
    """

    MAX_PARSE_RETRIES = 2

    def __init__(self, vision_client) -> None:
        """
        Parameters
        ----------
        vision_client:
            An instance of ``GeminiVisionClient`` (or compatible duck-typed object).
        """
        self.vision_client = vision_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def plan(
        self,
        image,
        task: str,
        history: Optional[List[dict]] = None,
    ) -> ActionPlan:
        """
        Analyse a screenshot and produce an ActionPlan for the given task.

        Parameters
        ----------
        image:
            PIL Image of the current browser viewport.
        task:
            The user's high-level goal, e.g. "Search for 'Python' on Wikipedia".
        history:
            Optional list of previous conversation turns to maintain context.

        Returns
        -------
        ActionPlan
            A validated ActionPlan.  On parse failure after retries a safe
            fallback plan (take a screenshot) is returned.
        """
        last_error: Optional[Exception] = None
        raw_response: str = ""

        for attempt in range(1, self.MAX_PARSE_RETRIES + 2):
            try:
                raw_response = await self.vision_client.analyze_screen(
                    image=image,
                    task=task,
                    history=history,
                )
                plan = self._parse_response(raw_response)
                return plan

            except (json.JSONDecodeError, ValueError, ValidationError) as exc:
                last_error = exc
                logger.warning(
                    "ActionPlanner attempt %d/%d failed to parse response: %s",
                    attempt,
                    self.MAX_PARSE_RETRIES + 1,
                    exc,
                )
                if attempt <= self.MAX_PARSE_RETRIES:
                    # Ask the model to fix its output on retry.
                    history = (history or []) + [
                        genai_types.Content(
                            role="model",
                            parts=[genai_types.Part.from_text(text=raw_response)],
                        ),
                        genai_types.Content(
                            role="user",
                            parts=[
                                genai_types.Part.from_text(
                                    text=(
                                        "Your previous response was not valid JSON. "
                                        "Please respond ONLY with a valid JSON object "
                                        "matching the required schema — no markdown, no prose."
                                    )
                                )
                            ],
                        ),
                    ]

        logger.error(
            "ActionPlanner could not parse model response after %d attempts. "
            "Returning fallback plan. Last error: %s",
            self.MAX_PARSE_RETRIES + 1,
            last_error,
        )
        return self._fallback_plan(raw_response)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_response(self, raw: str) -> ActionPlan:
        """Parse a raw model response string into an ActionPlan."""
        json_str = _extract_json_from_text(raw)
        data = json.loads(json_str)

        # Normalise the actions list — the model may emit action dicts that
        # use extra/missing fields; Pydantic will coerce them.
        raw_actions = data.get("actions", [])
        validated_actions: List[Action] = []
        for raw_action in raw_actions:
            if isinstance(raw_action, dict):
                try:
                    validated_actions.append(Action(**raw_action))
                except ValidationError as exc:
                    logger.warning("Skipping invalid action dict: %s — %s", raw_action, exc)
            else:
                logger.warning("Skipping non-dict action: %r", raw_action)

        data["actions"] = validated_actions
        return ActionPlan(**data)

    @staticmethod
    def _fallback_plan(raw_response: str) -> ActionPlan:
        """Return a safe plan that takes a screenshot to re-observe the screen."""
        return ActionPlan(
            observation=(
                "Unable to parse model response. Raw response: "
                + raw_response[:200]
                + ("..." if len(raw_response) > 200 else "")
            ),
            reasoning=(
                "The model returned a response that could not be parsed as JSON. "
                "Taking a screenshot to re-observe the current state."
            ),
            actions=[
                Action(
                    type=ActionType.SCREENSHOT,
                    description="Re-observe screen after parse failure",
                )
            ],
            done=False,
            result=None,
        )
