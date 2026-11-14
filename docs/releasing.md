# Releasing to PyPI

Operon is published as `operon-agent` (the import and the command stay `operon`).
The "Release to PyPI" workflow (`.github/workflows/release.yml`) does the work; nobody
needs a PyPI token, because PyPI trusts that workflow (trusted publisher: this repo,
`release.yml`, environment `pypi`).

1. Set the version in `pyproject.toml` (`0.1.0a1`, `0.1.0a2`, ... for alphas) and the
   matching `version` of `operon-agent` in `uv.lock`; merge that to `main`.
2. Optional dry run: run the workflow by hand from the Actions tab. It builds, checks the
   metadata and installs the wheel in a clean venv, and uploads nothing.
3. Release: push the tag from `main`, e.g. `git tag v0.1.0a1 && git push origin v0.1.0a1`.
   The workflow checks the tag matches the version, points the README's relative links at
   GitHub for the PyPI page, builds, smoke-tests, then uploads.

An upload cannot be undone, and PyPI never accepts the same version twice (even after a
delete), so a broken release is fixed with the next version, not a re-upload.
