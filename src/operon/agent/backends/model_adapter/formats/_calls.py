"""Safe parsing of Python-style call text (``click(start_box='(1,2)')``,
``pyautogui.click(x=1, y=2)``): ``ast`` plus literal values only, never ``eval``."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any

_NUMBERS = re.compile(r"-?\d+(?:\.\d+)?")


class CallSyntaxError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Call:
    name: str  # dotted, e.g. "pyautogui.click" or "click"
    args: list[Any] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_dotted(node.value)}.{node.attr}"
    raise CallSyntaxError("call target must be a plain name")


def parse_calls(code: str) -> list[Call]:
    """Every top-level call statement in ``code``, in order. Anything else (an
    assignment, an import, a non-literal argument) is a syntax error."""
    try:
        tree = ast.parse(code.strip())
    except SyntaxError as exc:
        raise CallSyntaxError(f"not valid call syntax: {exc.msg}") from exc
    calls = []
    for statement in tree.body:
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
            raise CallSyntaxError("only function calls are allowed")
        node = statement.value
        try:
            args = [ast.literal_eval(arg) for arg in node.args]
            kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords if kw.arg}
        except (ValueError, SyntaxError) as exc:
            raise CallSyntaxError("call arguments must be literal values") from exc
        calls.append(Call(_dotted(node.func), args, kwargs))
    return calls


def numbers(text: Any) -> list[float]:
    """The numbers in a box or point string such as ``'<|box_start|>(12,34)<|box_end|>'``
    or ``'<point>12 34</point>'``."""
    return [float(n) for n in _NUMBERS.findall(str(text))]


def point_of(text: Any) -> tuple[float, float]:
    """A point from a point string, or the centre of a two-corner box."""
    values = numbers(text)
    if len(values) == 2:
        return values[0], values[1]
    if len(values) == 4:
        return (values[0] + values[2]) / 2, (values[1] + values[3]) / 2
    raise CallSyntaxError(f"expected a point or a box, got {text!r}")
