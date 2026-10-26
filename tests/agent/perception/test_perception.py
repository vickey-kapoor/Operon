"""Focused tests for Gemini perception parsing."""

from __future__ import annotations

import pytest

from operon.agent.actions.selector import DeterministicTargetSelector
from operon.agent.perception import (
    PerceptionError,
    parse_perception_output,
)
from operon.models.perception import PageHint, UIElementNameSource, UIElementType
from operon.models.selector import TargetIntent, TargetIntentAction


def test_form_page_classification() -> None:
    raw_output = """
    {
      "summary": "Contact form is visible with Name, Email, Message, and Submit fields.",
      "page_hint": "form_page",
      "focused_element_id": "name-input",
      "confidence": 0.96,
      "visible_elements": []
    }
    """

    perception = parse_perception_output(raw_output, "runs/run-1/step_1/before.png")

    assert perception.page_hint is PageHint.FORM_PAGE


def test_form_success_classification() -> None:
    raw_output = """
    {
      "summary": "Thank you, the form was submitted successfully.",
      "page_hint": "form_success",
      "focused_element_id": null,
      "confidence": 0.97,
      "visible_elements": []
    }
    """

    perception = parse_perception_output(raw_output, "runs/run-1/step_2/before.png")

    assert perception.page_hint is PageHint.FORM_SUCCESS


def test_missing_page_hint_uses_small_summary_fallback() -> None:
    raw_output = """
    {
      "summary": "Google sign in page asking for email or phone.",
      "focused_element_id": null,
      "confidence": 0.88,
      "visible_elements": []
    }
    """

    perception = parse_perception_output(raw_output, "runs/run-1/step_1/before.png")

    # Fallback classifier no longer detects app-specific pages; LLM provides page_hint.
    assert perception.page_hint is PageHint.UNKNOWN


def test_missing_page_hint_uses_form_summary_fallback() -> None:
    raw_output = """
    {
      "summary": "Thank you for your message, the form was submitted successfully.",
      "focused_element_id": null,
      "confidence": 0.88,
      "visible_elements": []
    }
    """

    perception = parse_perception_output(raw_output, "runs/run-1/step_1/before.png")

    assert perception.page_hint is PageHint.FORM_SUCCESS


def test_null_page_hint_is_rejected() -> None:
    raw_output = """
    {
      "summary": "Google sign in page asking for email or phone.",
      "page_hint": null,
      "focused_element_id": null,
      "confidence": 0.88,
      "visible_elements": []
    }
    """

    with pytest.raises(PerceptionError, match="strict schema"):
        parse_perception_output(raw_output, "runs/run-1/step_1/before.png")


def test_empty_visible_element_label_is_normalized_and_accepted() -> None:
    raw_output = """
    {
      "summary": "Form page is visible.",
      "page_hint": "form_page",
      "focused_element_id": null,
      "confidence": 0.88,
      "visible_elements": [
        {
          "element_id": "email-input",
          "element_type": "input",
          "label": "   ",
          "x": 320,
          "y": 180,
          "width": 300,
          "height": 28,
          "is_interactable": true,
          "confidence": 0.94
        }
      ]
    }
    """

    perception = parse_perception_output(raw_output, "runs/run-1/step_1/before.png")

    assert len(perception.visible_elements) == 1
    assert perception.visible_elements[0].label is None
    assert perception.visible_elements[0].primary_name == "unlabeled_input"
    assert perception.visible_elements[0].name_source is UIElementNameSource.SYNTHETIC
    assert perception.visible_elements[0].is_unlabeled is True
    assert perception.visible_elements[0].usable_for_targeting is False


def test_missing_visible_element_label_is_normalized_from_element_type() -> None:
    raw_output = """
    {
      "summary": "Form page is visible.",
      "page_hint": "form_page",
      "focused_element_id": null,
      "confidence": 0.88,
      "visible_elements": [
        {
          "element_id": "email-input",
          "element_type": "input",
          "x": 320,
          "y": 180,
          "width": 300,
          "height": 28,
          "is_interactable": true,
          "confidence": 0.94
        }
      ]
    }
    """

    perception = parse_perception_output(raw_output, "runs/run-1/step_1/before.png")

    assert len(perception.visible_elements) == 1
    assert perception.visible_elements[0].label is None
    assert perception.visible_elements[0].primary_name == "unlabeled_input"
    assert perception.visible_elements[0].name_source is UIElementNameSource.SYNTHETIC


def test_whitespace_weak_semantic_fields_are_normalized_to_none() -> None:
    raw_output = """
    {
      "summary": "Form page is visible.",
      "page_hint": "form_page",
      "focused_element_id": null,
      "confidence": 0.88,
      "visible_elements": [
        {
          "element_id": "name-input",
          "element_type": "input",
          "label": "  ",
          "text": " ",
          "placeholder": "  ",
          "name": "\\n",
          "x": 320,
          "y": 180,
          "width": 300,
          "height": 28,
          "is_interactable": true,
          "confidence": 0.94
        }
      ]
    }
    """

    perception = parse_perception_output(raw_output, "runs/run-1/step_1/before.png")

    assert len(perception.visible_elements) == 1
    element = perception.visible_elements[0]
    assert element.label is None
    assert element.text is None
    assert element.placeholder is None
    assert element.name is None
    assert element.primary_name == "unlabeled_input"


def test_structurally_invalid_visible_element_still_fails_cleanly() -> None:
    raw_output = """
    {
      "summary": "Form page is visible.",
      "page_hint": "form_page",
      "focused_element_id": null,
      "confidence": 0.88,
      "visible_elements": [
        {
          "element_id": "name-input",
          "element_type": "input",
          "label": "",
          "x": 100,
          "y": 200,
          "width": 0,
          "height": 28,
          "is_interactable": true,
          "confidence": 0.5
        }
      ]
    }
    """

    with pytest.raises(PerceptionError, match="strict schema"):
        parse_perception_output(raw_output, "runs/run-1/step_1/before.png")


def test_unlabeled_heavy_perception_is_salvaged_into_selector_candidates() -> None:
    raw_output = """
    {
      "summary": "Form page with visible labels and fields.",
      "page_hint": "form_page",
      "focused_element_id": null,
      "confidence": 0.88,
      "visible_elements": [
        {
          "element_id": "name-label",
          "element_type": "text",
          "text": "Name",
          "x": 320,
          "y": 140,
          "width": 80,
          "height": 24,
          "is_interactable": false,
          "confidence": 0.94
        },
        {
          "element_id": "name-input",
          "element_type": "input",
          "label": " ",
          "x": 320,
          "y": 180,
          "width": 300,
          "height": 28,
          "is_interactable": true,
          "confidence": 0.94
        },
        {
          "element_id": "submit-label",
          "element_type": "text",
          "text": "Submit",
          "x": 320,
          "y": 260,
          "width": 90,
          "height": 24,
          "is_interactable": false,
          "confidence": 0.94
        },
        {
          "element_id": "submit-button",
          "element_type": "button",
          "label": "",
          "x": 320,
          "y": 300,
          "width": 120,
          "height": 32,
          "is_interactable": true,
          "confidence": 0.94
        }
      ]
    }
    """

    perception = parse_perception_output(raw_output, "runs/run-1/step_1/before.png")
    selector = DeterministicTargetSelector()
    result = selector.select(
        perception,
        TargetIntent(
            action=TargetIntentAction.CLICK,
            target_text="submit",
            expected_element_types=[UIElementType.BUTTON],
            expected_section="form",
        ),
    )

    assert any(element.usable_for_targeting for element in perception.visible_elements)
    assert result.trace.candidate_count >= 1
    assert result.selected is not None
    assert result.selected.element_id == "submit-button"


def test_form_page_with_nearby_text_and_unlabeled_inputs_is_minimally_viable() -> None:
    raw_output = """
    {
      "summary": "Practice form page is visible.",
      "page_hint": "form_page",
      "focused_element_id": null,
      "confidence": 0.84,
      "visible_elements": [
        {
          "element_id": "name-label",
          "element_type": "text",
          "text": "Name",
          "x": 280,
          "y": 150,
          "width": 80,
          "height": 20,
          "is_interactable": false,
          "confidence": 0.92
        },
        {
          "element_id": "name-input",
          "element_type": "input",
          "label": " ",
          "x": 280,
          "y": 180,
          "width": 160,
          "height": 28,
          "is_interactable": true,
          "confidence": 0.9
        },
        {
          "element_id": "email-label",
          "element_type": "text",
          "text": "Email",
          "x": 280,
          "y": 220,
          "width": 80,
          "height": 20,
          "is_interactable": false,
          "confidence": 0.92
        },
        {
          "element_id": "email-input",
          "element_type": "input",
          "label": "",
          "x": 280,
          "y": 250,
          "width": 160,
          "height": 28,
          "is_interactable": true,
          "confidence": 0.9
        }
      ]
    }
    """

    perception = parse_perception_output(raw_output, "runs/run-1/step_1/before.png")

    assert perception.page_hint is PageHint.FORM_PAGE
    assert len([element for element in perception.visible_elements if element.is_interactable]) == 2
    assert any(element.primary_name == "Name" for element in perception.visible_elements if element.element_id == "name-input")
    assert any(element.usable_for_targeting for element in perception.visible_elements if element.element_id == "email-input")
