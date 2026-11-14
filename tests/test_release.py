"""PyPI release prep (KAN-36): package metadata, version flag, README links for PyPI."""

from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path

import pytest

from operon import __version__, cli

ROOT = Path(__file__).resolve().parents[1]


def _pypi_readme():
    spec = importlib.util.spec_from_file_location("pypi_readme", ROOT / "scripts" / "pypi_readme.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_distribution_is_operon_agent_and_version_matches() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["name"] == "operon-agent"
    assert __version__ == project["version"]


def test_version_flag(capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.build_parser().parse_args(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"operon {__version__}"


def test_readme_links_point_at_github_for_pypi() -> None:
    text = (
        "![Demo](docs/assets/demo.gif)\n"
        "See [the API](docs/api.md#task-page) and [reports](docs/reports/) and [License](./LICENSE).\n"
        "Keep [site](https://example.com), [anchor](#quick-start) and [abs](/x)."
    )
    out = _pypi_readme().absolutize(text, "v0.1.0a1", repo="o/r")
    assert "![Demo](https://raw.githubusercontent.com/o/r/v0.1.0a1/docs/assets/demo.gif)" in out
    assert "[the API](https://github.com/o/r/blob/v0.1.0a1/docs/api.md#task-page)" in out
    assert "[reports](https://github.com/o/r/tree/v0.1.0a1/docs/reports/)" in out
    assert "[License](https://github.com/o/r/blob/v0.1.0a1/LICENSE)" in out
    assert "[site](https://example.com)" in out and "[anchor](#quick-start)" in out and "[abs](/x)" in out


def test_every_readme_link_target_exists() -> None:
    module = _pypi_readme()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for _, _, target in module._LINK.findall(readme):
        if module._ABSOLUTE.match(target):
            continue
        assert (ROOT / target.split("#")[0]).exists(), target
