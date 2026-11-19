"""Every relative link in README.md points at a file that exists."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_LINK = re.compile(r"(!?)\[([^\]]*)\]\(([^)\s]+)\)")
_EXTERNAL = re.compile(r"^(?:[a-z][a-z0-9+.-]*:|#|/)", re.IGNORECASE)


def test_every_readme_link_target_exists() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for _, _, target in _LINK.findall(readme):
        if _EXTERNAL.match(target):
            continue
        assert (ROOT / target.split("#")[0]).exists(), target
