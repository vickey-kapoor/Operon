"""Prompts ship inside the ``operon`` package so a wheel install can load them."""

from __future__ import annotations

import tomllib

from operon.core.paths import project_root, prompts_dir


def test_prompts_dir_is_inside_the_package() -> None:
    assert prompts_dir().parent.name == "operon"
    assert sorted(p.name for p in prompts_dir().glob("*.txt"))


def test_prompts_are_declared_as_package_data() -> None:
    config = tomllib.loads((project_root() / "pyproject.toml").read_text(encoding="utf-8"))
    assert "prompts/*.txt" in config["tool"]["setuptools"]["package-data"]["operon"]
