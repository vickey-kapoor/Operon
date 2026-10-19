"""The agent layer must not depend on the API layer (api -> agent only)."""

from __future__ import annotations

import ast
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parents[2] / "src" / "operon" / "agent"


def _imported_modules(path: Path) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_agent_package_never_imports_operon_api() -> None:
    offenders = {
        str(path.relative_to(AGENT_DIR)): sorted(m for m in _imported_modules(path) if m.startswith("operon.api"))
        for path in AGENT_DIR.rglob("*.py")
    }
    offenders = {name: mods for name, mods in offenders.items() if mods}
    assert not offenders, f"agent/ must not import operon.api: {offenders}"
