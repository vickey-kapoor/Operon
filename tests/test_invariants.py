"""Static guards for load-bearing CLAUDE.md invariants (no live server, no screen)."""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
PERCEPTION_POLICY_DIRS = [
    SRC / "operon" / "agent" / "perception",
    SRC / "operon" / "agent" / "policy",
    SRC / "operon" / "agent" / "backends",
]

# Attribute / name / import identifiers that mean DOM or accessibility-tree access.
FORBIDDEN_IDENTIFIERS = {
    "locator",
    "frame_locator",
    "query_selector",
    "query_selector_all",
    "wait_for_selector",
    "get_by_role",
    "get_by_text",
    "get_by_label",
    "get_by_placeholder",
    "get_by_test_id",
    "elementFromPoint",
    "elementsFromPoint",
    "accessibility",
    "aria_snapshot",
    "xpath",
    "XPath",
}
# String literals that embed DOM-access code (e.g. a page.evaluate JS payload).
# Docstrings and comments are not scanned for identifiers, only real string constants here.
FORBIDDEN_STRING_RE = re.compile(
    r"elementFromPoint|document\.(querySelector|getElementBy|evaluate)|"
    r"\bxpath\b|aria-snapshot|accessibility\.snapshot|\.locator\(",
    re.IGNORECASE,
)
# (relative posix path, token) -> reason. Currently empty: the scan is clean.
# A real violation must be listed with a reason starting with "KNOWN VIOLATION".
ALLOWLIST: dict[tuple[str, str], str] = {}


def _scan(path: Path) -> list[tuple[str, int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    docstring_nodes: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(
            node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        ) and (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
        ):
            docstring_nodes.add(id(node.body[0].value))
    hits: list[tuple[str, int, str]] = []
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Attribute):
            names.append(node.attr)
        elif isinstance(node, ast.Name):
            names.append(node.id)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.extend(node.module.split("."))
            names.extend(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            for a in node.names:
                names.extend(a.name.split("."))
        elif isinstance(node, ast.keyword) and node.arg:
            names.append(node.arg)
        for name in names:
            if name in FORBIDDEN_IDENTIFIERS:
                hits.append((name, getattr(node, "lineno", 0), "identifier"))
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstring_nodes
        ):
            m = FORBIDDEN_STRING_RE.search(node.value)
            if m:
                hits.append((m.group(0), node.lineno, "string"))
    return hits


def test_perception_policy_backends_are_vision_only() -> None:
    violations: list[str] = []
    scanned = 0
    for root in PERCEPTION_POLICY_DIRS:
        assert root.is_dir(), root
        for path in sorted(root.rglob("*.py")):
            scanned += 1
            rel = path.relative_to(SRC).as_posix()
            for token, line, kind in _scan(path):
                if (rel, token) in ALLOWLIST:
                    continue
                violations.append(f"{rel}:{line} {kind} {token!r}")
    assert scanned > 20, "scan found suspiciously few files"
    assert not violations, "DOM/selector/a11y use in vision-only path:\n" + "\n".join(violations)


def test_scanner_detects_violations(tmp_path: Path) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text(
        '"""locator( in a docstring is fine"""\n'
        "def f(page):\n"
        "    page.locator('#x').click()\n"
        "    page.evaluate('document.elementFromPoint(1, 2)')\n",
        encoding="utf-8",
    )
    tokens = {t for t, _, _ in _scan(bad)}
    assert "locator" in tokens
    assert "elementFromPoint" in tokens


def test_browser_manager_does_not_import_ws_stream_at_load() -> None:
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(SRC)!r})\n"
        "import operon.browser.manager\n"
        "assert 'operon.api.ws_stream' not in sys.modules, 'ws_stream imported at load'\n"
    )
    env = {**os.environ, "GEMINI_API_KEY": "fake-test-key", "OPERON_TEST_SAFE_MODE": "true"}
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
