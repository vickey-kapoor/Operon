"""Point the README's relative links at GitHub, for the copy shown on PyPI.

PyPI renders the README without the repository around it, so ``docs/api.md`` and
``docs/assets/demo.gif`` would be dead links there. The release workflow runs this
on the README before building; images go to raw.githubusercontent.com, other links to
the blob view, both at the release tag so they never drift.

    python scripts/pypi_readme.py README.md --ref v0.1.0a1
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = "vickey-kapoor/Operon"
_LINK = re.compile(r"(!?)\[([^\]]*)\]\(([^)\s]+)\)")
_ABSOLUTE = re.compile(r"^(?:[a-z][a-z0-9+.-]*:|#|/)", re.IGNORECASE)


def absolutize(text: str, ref: str, repo: str = REPO) -> str:
    def fix(match: re.Match) -> str:
        bang, label, target = match.groups()
        if _ABSOLUTE.match(target):
            return match.group(0)
        path = target.removeprefix("./")
        if bang:
            url = f"https://raw.githubusercontent.com/{repo}/{ref}/{path}"
        else:
            kind = "tree" if path.endswith("/") else "blob"
            url = f"https://github.com/{repo}/{kind}/{ref}/{path}"
        return f"{bang}[{label}]({url})"

    return _LINK.sub(fix, text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("readme", type=Path)
    parser.add_argument("--ref", required=True, help="git tag or branch the links should point at")
    args = parser.parse_args(argv)
    args.readme.write_text(absolutize(args.readme.read_text(encoding="utf-8"), args.ref), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
