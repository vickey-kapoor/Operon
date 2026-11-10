"""Frozen seasons: which tasks a yardstick run used, proven by hash (KAN-58).

A season is a frozen set of tasks that Operon's yardstick runs are scored on. Its
manifest (``evals/seasons/season-<n>.json``) records a SHA-256 of every task's
definition, so a later run can prove it ran the same tasks and refuses to run when
any of them changed:

- the task's data (intent, start page, check, step limit, seeded app state, ...);
- its grader: the source of the function that grades it (``goal``/``harm`` for local
  app and safety tasks, the desktop state check, the judge for public web tasks),
  plus every helper that function calls in ``evals``;
- the fixture files it runs on (the local apps site, the desktop probe script).

A season splits its tasks in two:

- **working** tasks live in this repo. Their failures are triaged and fixed in Operon.
- **held-out** tasks live outside the repo, in a private Python file whose path is
  ``$OPERON_SEASON_PRIVATE`` (fetched only by the benchmark workflow). The manifest
  records only an opaque id and the hash of each, so they can't change silently, and
  reports give their results as counts only: no names, intents, screenshots or logs,
  so nobody tunes Operon on them.

A broken task is never edited in place: it is retired (``python -m
evals.leaderboard.season retire``), for every model, with the date and reason logged
in the manifest, and the season is rescored without it.

Grounding is part of a season as a configuration (sets, samples per set, the pinned
dataset revisions, the sampling seed and the scoring code), since its samples come
from public datasets and are picked deterministically.

    .venv/bin/python -m evals.leaderboard.season freeze 1 --suite long --suite desktop --suite grounding
    .venv/bin/python -m evals.leaderboard.season check 1
    .venv/bin/python -m evals.leaderboard.season retire 1 long long_berlin "grader too strict on the street line"
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import linecache
import os
import subprocess
import sys
import textwrap
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePath
from types import FunctionType, ModuleType
from typing import Any

from evals.leaderboard.suite import SuiteMismatchError, hash_bytes

EVALS_DIR = Path(__file__).resolve().parents[1]
SEASONS_DIR = EVALS_DIR / "seasons"
PRIVATE_ENV = "OPERON_SEASON_PRIVATE"
FORMAT = 1

WORKING, HELD_OUT = "working", "held_out"

# Season suites and how each one runs. ``scored`` suites count toward the season's
# score: the core set of each category (KAN-72, evals/categories.py), one grid cell
# per harness, and grounding as a component score. The others are the larger pools,
# run on request, and pass/fail gates (saturated suites every model passes).
RUNNERS: dict[str, dict[str, Any]] = {
    "browser-core": {"runner": "live_bench", "mode": "browser", "bench_suite": "core", "scored": True},
    "desktop-core": {"runner": "live_desktop_bench", "mode": "desktop", "bench_suite": "core", "scored": True},
    "long": {"runner": "live_bench", "mode": "browser", "bench_suite": "long", "scored": False},
    "desktop": {"runner": "live_desktop_bench", "mode": "desktop", "bench_suite": "full", "scored": False},
    "grounding": {"runner": "grounding", "scored": True},
    "apps": {"runner": "live_bench", "mode": "browser", "bench_suite": "apps", "scored": False},
    "safety": {"runner": "live_bench", "mode": "browser", "bench_suite": "safety", "scored": False},
}

# Settings whose names contain these are secrets: never copied into a report.
_SECRET_PARTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "DATABASE_URL", "WEBHOOK")


class SeasonError(RuntimeError):
    """A season can't be used as asked (missing, unknown suite, private tasks absent)."""


# ---------- fingerprints ----------


def _plain(value: Any, found: list[Callable] | None = None) -> Any:
    """``value`` as JSON-able data, in a stable order. Functions become a placeholder
    and are appended to ``found``, so their code gets hashed (a safety task's pages,
    a grader)."""
    if isinstance(value, FunctionType) or callable(value) and not isinstance(value, type):
        if found is not None:
            found.append(value)
        return "<function>"
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name), found) for f in fields(value)}
    if isinstance(value, dict):
        return {str(k): _plain(v, found) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (set, frozenset)):
        return sorted((_plain(v, found) for v in value), key=repr)
    if isinstance(value, (list, tuple)):
        return [_plain(v, found) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, PurePath):
        # A path the same on every checkout and OS: relative to the repo, else its name.
        if isinstance(value, Path) and value.is_absolute():
            try:
                return value.resolve().relative_to(EVALS_DIR.parent).as_posix()
            except (ValueError, OSError):
                pass
        return value.name
    return repr(value)


def _value_text(value: Any) -> str:
    """A non-function value a grader depends on, as stable text."""
    try:
        return json.dumps(_plain(value), sort_keys=True)
    except (TypeError, ValueError):
        return repr(value)


def _is_evals_function(obj: Any) -> bool:
    return isinstance(obj, FunctionType) and (obj.__module__ or "").split(".")[0] in {"evals", "_operon_season_private"}


def _function_source(fn: FunctionType) -> str:
    try:
        return textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError):
        return fn.__code__.co_code.hex()


# Per-task registries: a grader that looks its check up by name (``desktop_state.grade``)
# is hashed, but the registry itself is not followed, because each task hashes its own
# entry as a root. Otherwise every desktop task would change when any one check did.
# The probe's per-task settings are in each task's fingerprint (desktop_probe_settings),
# and held-out tasks add entries to all four, so none of them is hashed whole.
_REGISTRIES = {("evals.desktop_state", key) for key in ("CHECKS", "OUTPUTS", "INSPECT", "CLOSE_FIRST")} | {
    ("evals.desktop_workspace", "WORKSPACES"),
}


def code_closure(roots: Iterable[Callable]) -> str:
    """The source of ``roots`` and of every ``evals`` function they reach (helpers they
    name, functions captured in their closures or defaults, functions nested in their
    code), with the values each one depends on: captured variables, default arguments
    and the module-level constants it names.

    So editing a grader's helper (say ``_one_order``) changes the hash of every task
    whose grader calls it, and so does ``_overreach(3)`` in place of ``_overreach(2)``."""
    seen: dict[str, str] = {}
    stack = [fn for fn in roots if _is_evals_function(fn)]
    done: set[int] = set()
    while stack:
        fn = stack.pop()
        # Each function once: a name it uses can be its own (``handle.read`` inside ``read``),
        # which would otherwise push it again forever.
        if id(fn) in done:
            continue
        done.add(id(fn))
        values: list[str] = []
        captured = []
        for cell in fn.__closure__ or ():
            try:
                captured.append(cell.cell_contents)
            except ValueError:
                captured.append("<empty>")
        defaults = [*(fn.__defaults__ or ()), *sorted((fn.__kwdefaults__ or {}).items())]
        for value in [*captured, *defaults]:
            found: list[Callable] = []
            values.append(_value_text(_plain(value, found)))
            stack.extend(f for f in found if _is_evals_function(f))
        codes = [fn.__code__]
        while codes:
            code = codes.pop()
            for name in sorted(set(code.co_names)):
                if name not in fn.__globals__ or (fn.__module__, name) in _REGISTRIES:
                    continue
                target = fn.__globals__[name]
                if _is_evals_function(target):
                    stack.append(target)
                elif not (inspect.ismodule(target) or inspect.isclass(target) or callable(target)):
                    values.append(f"{name} = {_value_text(target)}")
            codes.extend(c for c in code.co_consts if inspect.iscode(c))
        text = _function_source(fn) + "".join(f"\n# uses {v}" for v in values)
        # No line number in the key: an edit elsewhere in the file must not change it.
        key = f"{fn.__module__}:{fn.__qualname__}:{hashlib.sha256(text.encode()).hexdigest()[:12]}"
        if key in seen:
            continue
        seen[key] = text
    return "\n".join(f"# {key}\n{seen[key]}" for key in sorted(seen))


def file_hashes(paths: Iterable[Path]) -> dict[str, str]:
    """SHA-256 of each file's bytes (CRLF read as LF), keyed by its path under ``evals``."""
    result = {}
    for path in paths:
        rel = path.resolve().relative_to(EVALS_DIR.parent).as_posix()
        result[rel] = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    return dict(sorted(result.items()))


def grader_of(task: Any) -> tuple[list[Callable], list[Path]]:
    """The functions that grade ``task`` and the fixture files it runs on."""
    from evals.app_tasks import AppTask
    from evals.safety_tasks import SafetyTask

    if isinstance(task, (AppTask, SafetyTask)):
        return [task.goal, task.harm], [EVALS_DIR / "local_apps.py", EVALS_DIR / "fixture_site.py"]
    if getattr(task, "start_url", "") is None:  # desktop: graded from the machine's state
        from evals import desktop_state

        check = desktop_state.CHECKS.get(task.name)
        # prepare and probe too: how the reset clears a task's files and how the probe
        # reads them (decoding, size limit) decide its score as much as its check does.
        roots = [desktop_state.grade, desktop_state.prepare, desktop_state.probe,
                 *([check] if check is not None else [])]
        from evals import desktop_workspace

        if task.name in desktop_workspace.WORKSPACES:  # a seeded folder: its reset, read and harm check
            roots += [desktop_workspace.prepare, desktop_workspace.attach, desktop_workspace.harm]
        return roots, [EVALS_DIR / "desktop_probe.ps1"]
    from evals import live_bench

    return [live_bench.gemini_judge, live_bench.parse_verdict, live_bench.classify], []


def task_fingerprint(task: Any) -> str:
    """SHA-256 of everything that decides what ``task`` asks and how it is graded."""
    roots, fixtures = grader_of(task)
    in_data: list[Callable] = []
    data = json.dumps(_plain(task, in_data), sort_keys=True)
    roots = [*roots, *in_data]  # e.g. a safety task's pages built by a function
    parts = [
        ("data", data.encode()),
        ("grader", code_closure(roots).encode()),
        ("fixtures", json.dumps(file_hashes(fixtures), sort_keys=True).encode()),
    ]
    if getattr(task, "start_url", "") is None:
        parts.append(("probe", json.dumps(desktop_probe_settings(task.name), sort_keys=True).encode()))
    return hash_bytes(parts)


def desktop_probe_settings(name: str) -> dict:
    """What the probe does for one desktop task: the files it reads (and the reset
    removes), the windows it reads and the apps it stops first."""
    from evals import desktop_state, desktop_workspace

    settings = {key: list(getattr(desktop_state, key).get(name, ())) for key in ("OUTPUTS", "INSPECT", "CLOSE_FIRST")}
    if name in desktop_workspace.WORKSPACES:  # only then, so the other tasks' hashes stay the same
        settings["WORKSPACE"] = _plain(desktop_workspace.WORKSPACES[name])
    return settings


def held_out_id(season: int, name: str) -> str:
    """The opaque id a held-out task goes by in the manifest and in reports."""
    return "h-" + hashlib.sha256(f"season-{season}:{name}".encode()).hexdigest()[:10]


def grounding_config(sets: list[str] | None = None, per_set: int | None = None) -> dict:
    """What decides a grounding run's samples and scores, from the code as it is now."""
    from evals import grounding

    return {
        "sets": list(sets or grounding.SETS),
        "per_set": per_set or grounding.DEFAULT_PER_SET,
        "datasets": {grounding.V2_REPO: grounding.V2_REVISION, grounding.PRO_REPO: grounding.PRO_REVISION},
        "seed": grounding.SEED,
        "intent": grounding.INTENT,
        # Dataset picks the annotation files and builds each set's pool; its methods are
        # reached through an instance, which code_closure can't follow, so hash the class.
        "code": hashlib.sha256((code_closure([
            grounding.select_samples, grounding.stratified, grounding.v2_samples, grounding.pro_samples,
            grounding.first_point, grounding.score,
        ]) + textwrap.dedent(inspect.getsource(grounding.Dataset))).encode()).hexdigest(),
    }


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


# ---------- tasks ----------


def suite_tasks(suite: str) -> list[Any]:
    """The repo's tasks for a season suite, as the benchmark would select them."""
    from evals.live_bench import select_tasks

    spec = RUNNERS.get(suite)
    if spec is None or spec["runner"] == "grounding":
        raise SeasonError(f"{suite!r} has no tasks to fingerprint (suites: {', '.join(RUNNERS)})")
    return select_tasks(spec["bench_suite"], None, None, spec["mode"])


def load_private(path: str | os.PathLike | None = None) -> dict[str, list[Any]]:
    """The held-out tasks, by suite, from the private file ``$OPERON_SEASON_PRIVATE``.

    The file is Python that defines ``HELD_OUT = {"long": [AppTask(...), ...], ...}``
    with the same task classes as the repo, and for held-out desktop tasks
    ``CHECKS = {name: check}`` like ``desktop_state.CHECKS`` plus, where a task needs
    them, ``OUTPUTS``, ``INSPECT`` and ``CLOSE_FIRST`` like the ones there (the files
    it saves, the windows to read, the apps to stop first) and ``WORKSPACES`` like
    ``desktop_workspace.WORKSPACES`` (its seeded folder). {} when no file is configured."""
    path = path or os.getenv(PRIVATE_ENV, "").strip()
    if not path:
        return {}
    file = Path(path)
    if not file.is_file():
        raise SeasonError(f"${PRIVATE_ENV} points at {file}, which is not a file")
    # Compiled from the source each time: a cached .pyc can be stale (same size and
    # mtime second) and must not be left next to the private file either.
    module = ModuleType("_operon_season_private")
    module.__file__ = str(file)
    sys.modules[module.__name__] = module
    linecache.checkcache(str(file))
    exec(compile(file.read_text(encoding="utf-8"), str(file), "exec"), module.__dict__)
    held = getattr(module, "HELD_OUT", None)
    if not isinstance(held, dict):
        raise SeasonError(f"{file} must define HELD_OUT = {{suite: [task, ...]}}")
    from evals import desktop_state, desktop_workspace

    # Held-out desktop tasks' state checks, probe settings and seeded folders, by task name.
    for name in ("CHECKS", "OUTPUTS", "INSPECT", "CLOSE_FIRST"):
        extra = getattr(module, name, None)
        if extra:
            getattr(desktop_state, name).update(extra)
    desktop_workspace.WORKSPACES.update(getattr(module, "WORKSPACES", None) or {})
    return {suite: list(tasks) for suite, tasks in held.items()}


# ---------- manifest ----------


def manifest_path(season: int) -> Path:
    return SEASONS_DIR / f"season-{season}.json"


def season_hash(manifest: dict) -> str:
    """One hash over every suite's tasks (working and held-out) and the retire log."""
    body = {key: manifest.get(key) for key in ("season", "suites", "retired")}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def operon_commit() -> str:
    sha = os.getenv("GITHUB_SHA", "").strip()
    if sha:
        return sha
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=EVALS_DIR.parent, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def operon_settings(environ: dict[str, str] | None = None) -> dict[str, str]:
    """Operon's ``OPERON_*`` settings for a report, leaving out anything secret."""
    env = os.environ if environ is None else environ
    return {
        name: value for name, value in sorted(env.items())
        if name.startswith("OPERON_") and not any(part in name for part in _SECRET_PARTS)
    }


def freeze(season: int, suites: list[str], *, held_out: dict[str, list[Any]] | None = None,
           grounding: dict | None = None, now: datetime | None = None) -> dict:
    """A new season manifest over the repo's tasks for ``suites`` as they are now."""
    unknown = [s for s in suites if s not in RUNNERS]
    if unknown:
        raise SeasonError(f"unknown suite(s): {', '.join(unknown)} (suites: {', '.join(RUNNERS)})")
    held_out = held_out if held_out is not None else load_private()
    built: dict[str, dict] = {}
    for suite in suites:
        spec = RUNNERS[suite]
        entry: dict[str, Any] = {"runner": spec["runner"], "scored": spec["scored"]}
        if spec["runner"] == "grounding":
            config = grounding or grounding_config()
            entry.update(config=config, hash=config_hash(config))
        else:
            entry.update(mode=spec["mode"], bench_suite=spec["bench_suite"])
            entry[WORKING] = {t.name: task_fingerprint(t) for t in suite_tasks(suite)}
            private = held_out.get(suite, [])
            clash = sorted({t.name for t in private} & set(entry[WORKING]))
            if clash:
                raise SeasonError(f"held-out task(s) also in the repo: {', '.join(clash)}")
            entry[HELD_OUT] = {held_out_id(season, t.name): task_fingerprint(t) for t in private}
        built[suite] = entry
    stray = sorted(set(held_out) - set(suites))
    if stray:
        raise SeasonError(f"held-out tasks for suites not in this season: {', '.join(stray)}")
    manifest = {
        "format": FORMAT,
        "season": season,
        "frozen_at": (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "operon_commit": operon_commit(),
        "suites": built,
        "retired": [],
    }
    manifest["season_hash"] = season_hash(manifest)
    return manifest


def load(season: int, path: Path | None = None) -> dict:
    file = path or manifest_path(season)
    if not file.is_file():
        raise SeasonError(f"season {season} has no manifest at {file}")
    manifest = json.loads(file.read_text(encoding="utf-8"))
    if manifest.get("season_hash") != season_hash(manifest):
        raise SuiteMismatchError(f"{file} was edited by hand: its season_hash no longer matches its contents")
    return manifest


def save(manifest: dict, path: Path | None = None) -> Path:
    file = path or manifest_path(manifest["season"])
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return file


def retired_names(manifest: dict, suite: str) -> set[str]:
    return {r["task"] for r in manifest.get("retired", []) if r["suite"] == suite}


def retire(manifest: dict, suite: str, task: str, reason: str, *, now: datetime | None = None) -> dict:
    """``manifest`` with ``task`` retired from ``suite`` for every model (``task`` is a
    working task's name or a held-out id). Its hash stays in the manifest; runs skip it."""
    entry = manifest["suites"].get(suite)
    if entry is None or entry["runner"] == "grounding":
        raise SeasonError(f"season {manifest['season']} has no task suite {suite!r}")
    if task not in entry.get(WORKING, {}) and task not in entry.get(HELD_OUT, {}):
        raise SeasonError(f"{task!r} is not a task of season {manifest['season']} {suite}")
    if task in retired_names(manifest, suite):
        raise SeasonError(f"{task!r} is already retired")
    if not reason.strip():
        raise SeasonError("say why the task is retired")
    updated = json.loads(json.dumps(manifest))
    updated["retired"].append({
        "suite": suite, "task": task, "reason": reason.strip(),
        "at": (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ"),
    })
    updated["season_hash"] = season_hash(updated)
    return updated


# ---------- checking a run against its season ----------


@dataclass
class SeasonRun:
    """The tasks a season run executes, after every hash was checked."""

    season: int
    suite: str
    season_hash: str
    working: list[Any] = field(default_factory=list)
    held_out: list[tuple[str, Any]] = field(default_factory=list)  # (opaque id, task)
    retired: list[str] = field(default_factory=list)
    held_out_skipped: bool = False

    def is_held_out(self, task: Any) -> bool:
        return any(t is task for _, t in self.held_out)

    def held_out_id(self, task: Any) -> str | None:
        return next((hid for hid, t in self.held_out if t is task), None)

    def tasks(self) -> list[Any]:
        return [*self.working, *(t for _, t in self.held_out)]


def _diff(recorded: dict[str, str], current: dict[str, str], what: str) -> list[str]:
    problems = []
    for name in sorted(set(recorded) | set(current)):
        if name not in current:
            problems.append(f"{what} {name}: in the season but gone from the code")
        elif name not in recorded:
            problems.append(f"{what} {name}: new since the season was frozen")
        elif recorded[name] != current[name]:
            problems.append(f"{what} {name}: changed since the season was frozen")
    return problems


def check(manifest: dict, suite: str, *, held_out: dict[str, list[Any]] | None = None,
          working_only: bool = False) -> SeasonRun:
    """Check ``suite``'s tasks against the season and return what a run executes.

    Raises ``SuiteMismatchError`` naming every task that changed, appeared or went
    missing (retired tasks are skipped), and ``SeasonError`` when the season has
    held-out tasks but ``$OPERON_SEASON_PRIVATE`` isn't set, unless ``working_only``."""
    season = manifest["season"]
    entry = manifest["suites"].get(suite)
    if entry is None:
        raise SeasonError(f"season {season} has no suite {suite!r} (it has: {', '.join(manifest['suites'])})")
    run = SeasonRun(season=season, suite=suite, season_hash=manifest["season_hash"])
    retired = retired_names(manifest, suite)
    run.retired = sorted(retired)
    if entry["runner"] == "grounding":
        config = grounding_config(entry["config"]["sets"], entry["config"]["per_set"])
        if config_hash(config) != entry["hash"]:
            changed = sorted(k for k in config if config[k] != entry["config"].get(k))
            raise SuiteMismatchError(f"season {season} grounding changed since it was frozen: {', '.join(changed)}")
        return run

    problems: list[str] = []
    by_name = {t.name: t for t in suite_tasks(suite)}
    recorded = {n: h for n, h in entry[WORKING].items() if n not in retired}
    current = {n: task_fingerprint(t) for n, t in by_name.items() if n not in retired}
    problems += _diff(recorded, current, "task")
    run.working = [by_name[n] for n in entry[WORKING] if n not in retired and n in by_name]

    recorded_held = {h: v for h, v in entry.get(HELD_OUT, {}).items() if h not in retired}
    if recorded_held:
        private = (held_out if held_out is not None else load_private()).get(suite)
        if private is None:
            if not working_only:
                raise SeasonError(
                    f"season {season} {suite} has {len(recorded_held)} held-out tasks, but ${PRIVATE_ENV} "
                    "is not set; run with --working-only to run the working tasks alone"
                )
            run.held_out_skipped = True
        else:
            by_id = {held_out_id(season, t.name): t for t in private}
            current_held = {h: task_fingerprint(t) for h, t in by_id.items() if h not in retired}
            problems += _diff(recorded_held, current_held, "held-out task")
            run.held_out = [(h, by_id[h]) for h in recorded_held if h in by_id]
    if problems:
        raise SuiteMismatchError(
            f"season {season} {suite} doesn't match the code, so its scores wouldn't be comparable:\n  "
            + "\n  ".join(problems)
            + "\nRetire a broken task instead of editing it, or freeze a new season."
        )
    return run


def report_block(run: SeasonRun, results: list[dict] | None = None, *, is_held_out=None) -> dict:
    """The ``season`` block of a report: which season, its hash, the Operon commit and
    settings, and the held-out results as counts only."""
    block: dict[str, Any] = {
        "season": run.season,
        "suite": run.suite,
        "season_hash": run.season_hash,
        "operon_commit": operon_commit(),
        "operon_settings": operon_settings(),
        "working_tasks": len(run.working),
        "retired": run.retired,
        "held_out_tasks": len(run.held_out),
        "held_out_skipped": run.held_out_skipped,
    }
    if results is not None and is_held_out is not None:
        held = [r for r in results if is_held_out(r)]
        outcomes: dict[str, int] = {}
        for r in held:
            outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
        block["held_out"] = {"run": len(held), "passed": outcomes.get("success", 0), "outcomes": dict(sorted(outcomes.items()))}
    return block


# ---------- command line ----------


def _cmd_freeze(args: argparse.Namespace) -> int:
    path = manifest_path(args.season)
    if path.exists() and not args.force:
        print(f"{path} exists; a frozen season never changes (pass --force only before any run used it)", file=sys.stderr)
        return 2
    manifest = freeze(args.season, args.suite)
    save(manifest, path)
    for suite, entry in manifest["suites"].items():
        if entry["runner"] == "grounding":
            print(f"{suite:10} {entry['config']['per_set']} samples per set, {len(entry['config']['sets'])} sets")
        else:
            print(f"{suite:10} {len(entry[WORKING])} working, {len(entry[HELD_OUT])} held-out")
    print(f"saved {path} (season hash {manifest['season_hash'][:12]})")
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    manifest = load(args.season)
    failed = False
    for suite in manifest["suites"]:
        try:
            run = check(manifest, suite, working_only=args.working_only)
        except (SeasonError, SuiteMismatchError) as exc:
            print(f"{suite}: {exc}")
            failed = True
            continue
        held = "skipped" if run.held_out_skipped else str(len(run.held_out))
        print(f"{suite:10} ok: {len(run.working)} working, {held} held-out, {len(run.retired)} retired")
    return 1 if failed else 0


def _cmd_retire(args: argparse.Namespace) -> int:
    manifest = retire(load(args.season), args.suite, args.task, args.reason)
    save(manifest)
    print(f"retired {args.task} from season {args.season} {args.suite}; rescore the season without it")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Freeze, check and retire tasks of a season.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("freeze", help="write evals/seasons/season-<n>.json from the tasks as they are now")
    p.add_argument("season", type=int)
    p.add_argument("--suite", action="append", required=True, choices=list(RUNNERS))
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("check", help="check every suite of a season against the code")
    p.add_argument("season", type=int)
    p.add_argument("--working-only", action="store_true", help="don't need the held-out tasks")
    p = sub.add_parser("retire", help="retire a broken task for every model, with a reason")
    p.add_argument("season", type=int)
    p.add_argument("suite")
    p.add_argument("task", help="the task's name, or a held-out task's id")
    p.add_argument("reason")
    args = parser.parse_args(argv)
    try:
        return {"freeze": _cmd_freeze, "check": _cmd_check, "retire": _cmd_retire}[args.command](args)
    except (SeasonError, SuiteMismatchError) as exc:
        print(exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

