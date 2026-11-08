"""UI-TARS and OpenCUA parsers on replies in each model's published output format."""

from __future__ import annotations

import pytest

from operon.agent.backends.model_adapter.actions import (
    CoordinateMapper,
    NeutralAction,
    to_payload,
)
from operon.agent.backends.model_adapter.formats import PromptContext, get_format
from operon.agent.backends.model_adapter.formats._calls import (
    CallSyntaxError,
    parse_calls,
)

MAPPER = CoordinateMapper(space="pixels", sent_width=1288, sent_height=812, screen_width=1288, screen_height=812)


def _ctx(environment: str = "browser") -> PromptContext:
    return PromptContext(intent="Find a blue mug", environment=environment, mapper=MAPPER, task_prompt="Task: Find a blue mug")


def _parse(fmt: str, text: str, environment: str = "browser"):
    return get_format(fmt).parse({"role": "assistant", "content": text}, _ctx(environment))


# ── UI-TARS ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    ("action", "expected"),
    [
        ("click(start_box='<|box_start|>(235,512)<|box_end|>')", NeutralAction("click", {"x": 235.0, "y": 512.0})),
        ("click(start_box='(235,512)')", NeutralAction("click", {"x": 235.0, "y": 512.0})),
        ("click(point='<point>235 512</point>')", NeutralAction("click", {"x": 235.0, "y": 512.0})),
        ("click(start_box='<|box_start|>(100,200,300,400)<|box_end|>')", NeutralAction("click", {"x": 200.0, "y": 300.0})),
        ("left_double(start_box='(10,20)')", NeutralAction("double_click", {"x": 10.0, "y": 20.0})),
        ("right_single(start_box='(10,20)')", NeutralAction("right_click", {"x": 10.0, "y": 20.0})),
        ("hotkey(key='ctrl c')", NeutralAction("key", {"keys": "ctrl+c"})),
        ("hotkey(key='enter')", NeutralAction("key", {"keys": "enter"})),
        ("type(content='blue mug\\n')", NeutralAction("type", {"text": "blue mug", "press_enter": True})),
        ("type(content='it\\'s here')", NeutralAction("type", {"text": "it's here", "press_enter": False})),
        (
            "scroll(start_box='(640,400)', direction='down')",
            NeutralAction("scroll", {"direction": "down", "amount": 5, "x": 640.0, "y": 400.0}),
        ),
        (
            "drag(start_box='(1,2)', end_box='(3,4)')",
            NeutralAction("drag", {"start_x": 1.0, "start_y": 2.0, "end_x": 3.0, "end_y": 4.0}),
        ),
        ("wait()", NeutralAction("wait", {"seconds": 5})),
        ("navigate(url='https://example.com')", NeutralAction("navigate", {"url": "https://example.com"})),
    ],
)
def test_ui_tars_actions(action, expected) -> None:
    reply = _parse("ui_tars", f"Thought: The search box is at the top. I'll use it.\nAction: {action}")
    assert reply.invalid is None
    assert reply.actions == [expected]
    assert reply.thought == "The search box is at the top. I'll use it."


def test_ui_tars_finished_and_call_user_are_terminal() -> None:
    done = _parse("ui_tars", "Thought: The mug is in the cart.\nAction: finished(content='Added the blue mug.')")
    assert done.invalid is None and done.terminal == NeutralAction("done", {"answer": "Added the blue mug."})
    stuck = _parse("ui_tars", "Thought: A login is needed.\nAction: call_user()")
    assert stuck.terminal.name == "fail"


def test_ui_tars_several_actions_in_one_reply() -> None:
    reply = _parse(
        "ui_tars",
        "Thought: Fill the box and search.\nAction: click(start_box='(100,50)')\n\ntype(content='mug\\n')",
    )
    assert [a.name for a in reply.actions] == ["click", "type"]
    payload = to_payload(reply.actions, MAPPER)
    assert payload == {"action_type": "type", "text": "mug", "clear_before_typing": False, "press_enter": True, "x": 100, "y": 50}


def test_ui_tars_reflection_prefix_and_desktop_open_app() -> None:
    reply = _parse("ui_tars", "Reflection: Notepad is not open.\nAction: open_app(app_name='Notepad')", "desktop")
    assert reply.actions == [NeutralAction("launch_app", {"app": "notepad"})]


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("Thought: hmm, I am not sure.", "no 'Action:' line"),
        ("Thought: x\nAction: click(start_box='(1)')", "expected a point"),
        ("Thought: x\nAction: teleport(start_box='(1,2)')", "unknown action 'teleport'"),
        ("Thought: x\nAction: click(start_box=__import__('os'))", "literal values"),
        ("Thought: x\nAction: import os", "only function calls"),
        ("Thought: x\nAction: click(start_box='(1,2)')\nfinished(content='ok')", "only action"),
    ],
)
def test_ui_tars_bad_replies_are_invalid(text, reason) -> None:
    reply = _parse("ui_tars", text)
    assert reply.invalid is not None and reason in reply.invalid


def test_ui_tars_prompt_carries_the_task_as_a_user_message() -> None:
    fmt = get_format("ui_tars")
    prompt = fmt.system_prompt(_ctx())
    assert fmt.system_as_user is True
    assert prompt.endswith("## User Instruction\nTask: Find a blue mug")
    assert "navigate(url='')" in prompt and "open_app" not in prompt


# ── OpenCUA ──────────────────────────────────────────────────────────────────

_OPENCUA = """## Thought
The search field is empty and sits at the top of the page.

## Action
Click the search field and type the query.

## Code
```python
{code}
```"""


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("pyautogui.click(x=640, y=60)", [NeutralAction("click", {"x": 640.0, "y": 60.0})]),
        ("pyautogui.click(640, 60)", [NeutralAction("click", {"x": 640.0, "y": 60.0})]),
        ("pyautogui.click(x=1, y=2, clicks=2)", [NeutralAction("double_click", {"x": 1.0, "y": 2.0})]),
        ("pyautogui.click(x=1, y=2, button='right')", [NeutralAction("right_click", {"x": 1.0, "y": 2.0})]),
        ("pyautogui.doubleClick(x=1, y=2)", [NeutralAction("double_click", {"x": 1.0, "y": 2.0})]),
        ("pyautogui.tripleClick(x=1, y=2)", [NeutralAction("triple_click", {"x": 1.0, "y": 2.0})]),
        ("pyautogui.write(message='blue mug')", [NeutralAction("type", {"text": "blue mug", "press_enter": False})]),
        ("pyautogui.press('enter')", [NeutralAction("key", {"keys": "enter"})]),
        ("pyautogui.hotkey('ctrl', 'a')", [NeutralAction("key", {"keys": "ctrl+a"})]),
        ("pyautogui.scroll(-5)", [NeutralAction("scroll", {"direction": "down", "amount": 5})]),
        ("pyautogui.scroll(3, x=10, y=20)", [NeutralAction("scroll", {"direction": "up", "amount": 3, "x": 10.0, "y": 20.0})]),
        (
            "pyautogui.moveTo(x=1, y=2)\npyautogui.dragTo(x=3, y=4)",
            [
                NeutralAction("hover", {"x": 1.0, "y": 2.0}),
                NeutralAction("drag", {"start_x": 1.0, "start_y": 2.0, "end_x": 3.0, "end_y": 4.0}),
            ],
        ),
        ("computer.wait()", [NeutralAction("wait", {"seconds": 3.0})]),
    ],
)
def test_opencua_actions(code, expected) -> None:
    reply = _parse("opencua", _OPENCUA.format(code=code))
    assert reply.invalid is None, reply.invalid
    assert reply.actions == expected
    assert reply.thought.startswith("The search field is empty")


def test_opencua_click_then_write_is_one_type() -> None:
    reply = _parse("opencua", _OPENCUA.format(code="pyautogui.click(x=640, y=60)\npyautogui.write(message='mug\\n')"))
    assert to_payload(reply.actions, MAPPER) == {
        "action_type": "type", "text": "mug", "clear_before_typing": False, "press_enter": True, "x": 640, "y": 60,
    }


def test_opencua_terminate() -> None:
    done = _parse("opencua", _OPENCUA.format(code="computer.terminate(status='success', answer='$12')"))
    assert done.terminal == NeutralAction("done", {"answer": "$12"})
    failed = _parse("opencua", _OPENCUA.format(code="computer.terminate(status='failure')"))
    assert failed.terminal.name == "fail"


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("I clicked it.", "no code block"),
        (_OPENCUA.format(code="pyautogui.dragTo(x=3, y=4)"), "dragTo needs"),
        (_OPENCUA.format(code="import os\nos.system('rm -rf /')"), "only function calls"),
        (_OPENCUA.format(code="os.system('ls')"), "unsupported call os.system"),
        (_OPENCUA.format(code="pyautogui.click(x=1, y=2)\ncomputer.terminate(status='success')"), "only call"),
    ],
)
def test_opencua_bad_replies_are_invalid(text, reason) -> None:
    reply = _parse("opencua", text)
    assert reply.invalid is not None and reason in reply.invalid


def test_calls_are_parsed_never_run() -> None:
    with pytest.raises(CallSyntaxError):
        parse_calls("pyautogui.click(x=__import__('os').getpid(), y=1)")
