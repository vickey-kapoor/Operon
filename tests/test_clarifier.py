"""Tests for the TaskClarifier."""

import json
from unittest.mock import MagicMock, patch

import pytest


class TestTaskClarifier:

    def _make_clarifier(self, response_text: str):
        """Create a TaskClarifier with a mocked Gemini _call."""
        from src.agent.clarifier import TaskClarifier

        with patch("src.agent.clarifier.genai.Client"):
            clarifier = TaskClarifier(api_key="fake-key")

        clarifier._call = MagicMock(return_value=response_text)
        return clarifier

    async def test_returns_questions_for_ambiguous_task(self):
        """Clarifier surfaces questions when the model returns them."""
        payload = json.dumps({"questions": ["From which city?", "What dates?"]})
        clarifier = self._make_clarifier(payload)
        questions = await clarifier.get_questions("book a flight")
        assert questions == ["From which city?", "What dates?"]

    async def test_returns_empty_list_for_clear_task(self):
        """Clarifier returns [] when model says no questions needed."""
        payload = json.dumps({"questions": []})
        clarifier = self._make_clarifier(payload)
        questions = await clarifier.get_questions("search for python tutorials")
        assert questions == []

    async def test_returns_empty_list_on_gemini_failure(self):
        """Clarifier swallows errors and returns [] rather than raising."""
        from src.agent.clarifier import TaskClarifier

        with patch("src.agent.clarifier.genai.Client"):
            clarifier = TaskClarifier(api_key="fake-key")

        clarifier._call = MagicMock(side_effect=RuntimeError("API down"))
        questions = await clarifier.get_questions("book a flight")
        assert questions == []

    async def test_returns_empty_list_on_invalid_json(self):
        """Clarifier handles non-JSON responses gracefully."""
        clarifier = self._make_clarifier("not valid json at all")
        questions = await clarifier.get_questions("do something")
        assert questions == []

    async def test_filters_blank_questions(self):
        """Clarifier strips whitespace-only question strings."""
        payload = json.dumps({"questions": ["Real question?", "   ", ""]})
        clarifier = self._make_clarifier(payload)
        questions = await clarifier.get_questions("some task")
        assert questions == ["Real question?"]

    async def test_handles_non_list_questions_field(self):
        """Clarifier handles malformed responses where 'questions' is not a list."""
        payload = json.dumps({"questions": "not a list"})
        clarifier = self._make_clarifier(payload)
        questions = await clarifier.get_questions("some task")
        assert questions == []

    def test_requires_api_key(self):
        """TaskClarifier raises ValueError when no API key is available."""
        import os
        from src.agent.clarifier import TaskClarifier

        # Ensure GOOGLE_API_KEY is absent/empty in the patched env.
        with patch.dict(os.environ, {"GOOGLE_API_KEY": ""}, clear=False):
            with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
                TaskClarifier(api_key=None)
