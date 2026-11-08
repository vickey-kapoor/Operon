"""`.env.example` is a contract (CLAUDE.md): it must list every setting `src/` reads,
and every variable it lists must actually be read somewhere in `src/`."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_READ = re.compile(r"""(?:getenv|environ\.get|environ\[|_split_env)\(?\s*["']([A-Z][A-Z0-9_]+)["']""")
_LISTED = re.compile(r"^#?\s*([A-Z][A-Z0-9_]+)=", re.MULTILINE)


def _src_text() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "src").rglob("*.py"))


def _listed() -> set[str]:
    return set(_LISTED.findall((ROOT / ".env.example").read_text(encoding="utf-8")))


def test_every_setting_read_by_src_is_listed() -> None:
    read = set(_READ.findall(_src_text()))
    missing = sorted(read - _listed())
    assert not missing, f"add these to .env.example: {missing}"


def test_every_listed_variable_is_read_by_src() -> None:
    src = _src_text()
    unread = sorted(name for name in _listed() if name not in src)
    assert not unread, f"remove these from .env.example or read them in src/: {unread}"
