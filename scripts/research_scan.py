"""Weekly research scan: find new arXiv papers on computer-use agents and add them to the reading list.

Looks back a number of days across five themes (grounding, agent designs, world models,
benchmarks, safety), drops papers already in ``docs/research/reading-list.md``, and writes a
dated section under "Weekly scans". Entries are abstract excerpts straight from arXiv, not
reviewed notes. Stdlib only, no model call, free to run.

    python scripts/research_scan.py [--days 7] [--doc docs/research/reading-list.md] [--dry-run]

If CONFLUENCE_EMAIL and CONFLUENCE_API_TOKEN are set, the same section is also added to the
Confluence page (CONFLUENCE_PAGE_ID, default 3145739). Without them that step is skipped.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta
from pathlib import Path

ARXIV_API = "https://export.arxiv.org/api/query"
DEFAULT_DOC = Path("docs/research/reading-list.md")
HEADING = "## 7. Weekly scans (newest first)"
CONFLUENCE_HEADING = "7. Weekly scans (newest first)"
DEFAULT_CONFLUENCE_BASE = "https://vickeykapoor11.atlassian.net/wiki"
DEFAULT_PAGE_ID = "3145739"
PER_THEME = 8
_ATOM = {"a": "http://www.w3.org/2005/Atom"}
_ID = re.compile(r"\b(\d{4}\.\d{4,5})(?:v\d+)?\b")

AGENT = '(all:"computer use agent" OR all:"computer-use agent" OR all:"GUI agent" OR all:"web agent")'
THEMES: dict[str, str] = {
    "GUI grounding": '(all:"GUI grounding" OR (all:"visual grounding" AND all:"GUI"))',
    "Agent designs": '(all:"computer use agent" OR all:"computer-use agent" OR all:"GUI agent")',
    "World models and continuous perception": f'{AGENT} AND (all:"world model" OR all:"video" OR all:"continuous")',
    "Benchmarks and grading": f"{AGENT} AND (all:benchmark OR all:evaluation)",
    "Safety and prompt injection": f'{AGENT} AND (all:"prompt injection" OR all:adversarial OR all:safety)',
}


def known_ids(text: str) -> set[str]:
    """arXiv ids already mentioned anywhere in the reading list."""
    return set(_ID.findall(text))


def build_url(search: str, start: datetime, end: datetime, limit: int = PER_THEME) -> str:
    window = f"submittedDate:[{start:%Y%m%d}0000 TO {end:%Y%m%d}2359]"
    params = {
        "search_query": f"({search}) AND {window}",
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": str(limit),
    }
    return f"{ARXIV_API}?{urllib.parse.urlencode(params)}"


def parse_feed(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    papers = []
    for entry in root.findall("a:entry", _ATOM):
        raw_id = (entry.findtext("a:id", "", _ATOM) or "").rsplit("/", 1)[-1]
        match = _ID.search(raw_id)
        if not match:
            continue
        authors = [a.findtext("a:name", "", _ATOM) for a in entry.findall("a:author", _ATOM)]
        papers.append(
            {
                "id": match.group(1),
                "title": " ".join((entry.findtext("a:title", "", _ATOM) or "").split()),
                "published": (entry.findtext("a:published", "", _ATOM) or "")[:10],
                "authors": authors,
                "summary": " ".join((entry.findtext("a:summary", "", _ATOM) or "").split()),
            }
        )
    return papers


def excerpt(summary: str, limit: int = 320) -> str:
    """First sentences of the abstract, cut at a sentence end under the limit."""
    if len(summary) <= limit:
        return summary
    cut = summary[:limit]
    end = max(cut.rfind(". "), cut.rfind("? "))
    return (cut[: end + 1] if end > 80 else cut.rsplit(" ", 1)[0] + "...").strip()


def short_authors(authors: list[str]) -> str:
    if not authors:
        return "unknown authors"
    return authors[0] + (" et al." if len(authors) > 1 else "")


def collect(fetch, start: datetime, end: datetime, seen: set[str], sleep: float = 3.0) -> dict[str, list[dict]]:
    """Run each theme query, skipping ids already seen (in the doc or an earlier theme)."""
    seen = set(seen)
    found: dict[str, list[dict]] = {}
    for i, (theme, search) in enumerate(THEMES.items()):
        if i and sleep:
            time.sleep(sleep)  # arXiv asks for one request every 3 seconds
        fresh = []
        for paper in parse_feed(fetch(build_url(search, start, end))):
            if paper["id"] in seen:
                continue
            seen.add(paper["id"])
            fresh.append(paper)
        if fresh:
            found[theme] = fresh
    return found


def render_markdown(day: str, found: dict[str, list[dict]]) -> str:
    lines = [
        f"### Weekly scan {day}",
        "",
        "Collected from arXiv automatically. Abstract excerpts, not yet reviewed; numbers are the authors' own claims.",
        "",
    ]
    for theme, papers in found.items():
        lines += [f"**{theme}**", ""]
        for p in papers:
            lines.append(
                f"- [{p['title']}](https://arxiv.org/abs/{p['id']}) ({short_authors(p['authors'])}, {p['published']}): "
                f"{excerpt(p['summary'])}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_html(day: str, found: dict[str, list[dict]]) -> str:
    parts = [f"<h3>Weekly scan {day}</h3>", "<p>Collected from arXiv automatically. Abstract excerpts, not yet reviewed; numbers are the authors' own claims.</p>"]
    for theme, papers in found.items():
        parts.append(f"<p><strong>{html.escape(theme)}</strong></p><ul>")
        for p in papers:
            parts.append(
                f'<li><p><a href="https://arxiv.org/abs/{p["id"]}">{html.escape(p["title"])}</a> '
                f"({html.escape(short_authors(p['authors']))}, {p['published']}): {html.escape(excerpt(p['summary']))}</p></li>"
            )
        parts.append("</ul>")
    return "".join(parts)


def insert_section(doc: str, section: str) -> str:
    """Put a new weekly section right under the Weekly scans heading (newest first)."""
    if HEADING not in doc:
        return doc.rstrip() + f"\n\n{HEADING}\n\n{section}"
    head, tail = doc.split(HEADING, 1)
    return f"{head}{HEADING}\n\n{section}\n{tail.lstrip()}"


def _http(url: str, *, data: bytes | None = None, headers: dict | None = None, method: str | None = None) -> str:
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read().decode("utf-8")
        except Exception:
            if attempt == 2:
                raise
            time.sleep(10 * (attempt + 1))
    raise RuntimeError("unreachable")


def fetch_arxiv(url: str) -> str:
    return _http(url, headers={"User-Agent": "operon-research-scan"})


def sync_confluence(day: str, section_html: str, base: str, page_id: str, email: str, token: str) -> bool:
    """Add the section to the Confluence page, unless that day is already there."""
    auth = base64.b64encode(f"{email}:{token}".encode()).decode()
    headers = {"Authorization": f"Basic {auth}", "Content-Type": "application/json"}
    url = f"{base}/rest/api/content/{page_id}"
    page = json.loads(_http(f"{url}?expand=body.storage,version", headers=headers))
    body = page["body"]["storage"]["value"]
    if f"Weekly scan {day}" in body:
        return False
    marker = re.search(rf"<h2[^>]*>{re.escape(CONFLUENCE_HEADING)}</h2>", body)
    body = body[: marker.end()] + section_html + body[marker.end() :] if marker else body + section_html
    payload = {
        "version": {"number": page["version"]["number"] + 1, "message": f"Weekly scan {day}"},
        "title": page["title"],
        "type": "page",
        "body": {"storage": {"value": body, "representation": "storage"}},
    }
    _http(url, data=json.dumps(payload).encode(), headers=headers, method="PUT")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--days", type=int, default=7, help="look back this many days (default 7)")
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC, help="reading list to update")
    parser.add_argument("--dry-run", action="store_true", help="print the section, change nothing")
    args = parser.parse_args(argv)

    end = datetime.now(UTC)
    start = end - timedelta(days=args.days)
    day = f"{end:%Y-%m-%d}"
    doc = args.doc.read_text(encoding="utf-8")

    found = collect(fetch_arxiv, start, end, known_ids(doc))
    if not found:
        print(f"No new papers since {start:%Y-%m-%d}.")
        return 0
    total = sum(len(v) for v in found.values())
    section = render_markdown(day, found)
    print(f"{total} new papers.")
    if args.dry_run:
        print(section)
        return 0
    args.doc.write_text(insert_section(doc, section), encoding="utf-8")

    email, token = os.environ.get("CONFLUENCE_EMAIL"), os.environ.get("CONFLUENCE_API_TOKEN")
    if email and token:
        base = os.environ.get("CONFLUENCE_BASE", DEFAULT_CONFLUENCE_BASE)
        page_id = os.environ.get("CONFLUENCE_PAGE_ID", DEFAULT_PAGE_ID)
        added = sync_confluence(day, render_html(day, found), base, page_id, email, token)
        print("Confluence page updated." if added else "Confluence page already had this week.")
    else:
        print("CONFLUENCE_EMAIL / CONFLUENCE_API_TOKEN not set: Confluence not updated.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
