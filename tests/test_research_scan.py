import importlib.util
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("research_scan", ROOT / "scripts" / "research_scan.py")
scan = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scan)

FEED = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry><id>http://arxiv.org/abs/2611.01234v2</id><title>New
    GUI Agent Paper</title><published>2026-11-10T01:00:00Z</published>
    <summary>It does a thing. It does another thing.</summary>
    <author><name>Ada A</name></author><author><name>Bo B</name></author></entry>
  <entry><id>http://arxiv.org/abs/2404.07972v1</id><title>OSWorld</title>
    <published>2024-04-11T00:00:00Z</published><summary>Old.</summary><author><name>X Y</name></author></entry>
</feed>"""


def test_known_ids_ignores_versions_and_finds_links():
    text = "see [x](https://arxiv.org/abs/2404.07972) and 2611.01234v3"
    assert scan.known_ids(text) == {"2404.07972", "2611.01234"}


def test_collect_skips_known_and_duplicate_papers_across_themes():
    start, end = datetime(2026, 11, 7, tzinfo=UTC), datetime(2026, 11, 14, tzinfo=UTC)
    found = scan.collect(lambda url: FEED, start, end, {"2404.07972"}, sleep=0)
    assert [p["id"] for papers in found.values() for p in papers] == ["2611.01234"]
    assert len(found) == 1  # later themes return the same paper, which is dropped


def test_build_url_has_date_window_and_newest_first():
    url = scan.build_url("all:x", datetime(2026, 11, 7, tzinfo=UTC), datetime(2026, 11, 14, tzinfo=UTC))
    assert "20261107" in url and "20261114" in url and "sortOrder=descending" in url


def test_render_and_insert_put_new_week_on_top():
    found = {"Agent designs": scan.parse_feed(FEED)[:1]}
    section = scan.render_markdown("2026-11-14", found)
    assert "Ada A et al." in section and "https://arxiv.org/abs/2611.01234" in section
    doc = f"intro\n\n{scan.HEADING}\n\n### Weekly scan 2026-11-07\n- old\n"
    out = scan.insert_section(doc, section)
    assert out.index("2026-11-14") < out.index("2026-11-07")


def test_insert_adds_heading_when_missing():
    assert scan.HEADING in scan.insert_section("intro\n", "### Weekly scan 2026-11-14\n")


def test_excerpt_cuts_at_sentence_end():
    text = "A" * 100 + ". " + "B" * 300
    assert scan.excerpt(text).endswith(".")
