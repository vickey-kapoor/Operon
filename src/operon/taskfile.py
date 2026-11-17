"""Task files (KAN-86): a suite of Operon tasks in a YAML file, run like tests in CI.

    operon run --tasks tasks.yaml --junit report.xml

Tasks run one after another through the SDK (``operon.sdk.Agent``), one Agent per task.
After a run the ``expect`` checks are applied (graders may read the final answer and the
host machine's files; nothing here touches perception or policy). With no human in a
batch run, a step that needs approval fails the task unless ``approve_all`` is true.
"""

from __future__ import annotations

import asyncio
import re
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

__all__ = ["TaskFile", "TaskFileError", "TaskOutcome", "load_task_file", "run_task_file", "write_junit"]


class TaskFileError(ValueError):
    """The task file is unreadable or invalid; the message says where."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FileContains(_Strict):
    path: str
    text: str


class Expect(_Strict):
    status: str = "succeeded"
    answer_contains: list[str] = Field(default_factory=list)
    file_exists: str | None = None
    file_contains: FileContains | None = None
    max_steps: int | None = Field(default=None, ge=1)


class Defaults(_Strict):
    desktop: bool = False
    env: str | None = None
    # risky, not ask: nobody is there to approve, and `ask` pauses at the first click.
    mode: Literal["ask", "risky", "auto"] = "risky"
    max_steps: int | None = Field(default=25, ge=1)
    url: str | None = None
    approve_all: bool = False


class TaskSpec(_Strict):
    name: str = Field(min_length=1)
    task: str = Field(min_length=1)
    desktop: bool | None = None
    env: str | None = None
    mode: Literal["ask", "risky", "auto"] | None = None
    max_steps: int | None = Field(default=None, ge=1)
    url: str | None = None
    approve_all: bool | None = None
    expect: Expect = Field(default_factory=Expect)


class TaskFile(_Strict):
    defaults: Defaults = Field(default_factory=Defaults)
    tasks: list[TaskSpec] = Field(min_length=1)

    def resolved(self, spec: TaskSpec) -> dict[str, Any]:
        """The task's settings with the file defaults filled in."""
        merged = self.defaults.model_dump()
        merged.update({k: v for k, v in spec.model_dump(exclude={"name", "task", "expect"}).items() if v is not None})
        return merged


def load_task_file(path: str | Path) -> TaskFile:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise TaskFileError(f"{path}: cannot read: {exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f" (line {mark.line + 1}, column {mark.column + 1})" if mark else ""
        raise TaskFileError(f"{path}: invalid YAML{where}: {getattr(exc, 'problem', None) or exc}") from exc
    if not isinstance(data, dict):
        raise TaskFileError(f"{path}: expected a mapping with a `tasks:` list at the top level")
    try:
        parsed = TaskFile.model_validate(data)
    except ValidationError as exc:
        lines = []
        for err in exc.errors():
            loc = "".join(f"[{p}]" if isinstance(p, int) else f".{p}" for p in err["loc"]).lstrip(".")
            lines.append(f"{loc or '<file>'}: {err['msg']}")
        raise TaskFileError(f"{path}: invalid task file:\n  - " + "\n  - ".join(lines)) from exc
    names = [t.name for t in parsed.tasks]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise TaskFileError(f"{path}: duplicate task names: {', '.join(dupes)}")
    return parsed


@dataclass
class TaskOutcome:
    name: str
    status: str
    steps: int
    cost_usd: float | None
    seconds: float
    passed: bool
    reason: str = ""
    run_id: str = ""
    artifacts_dir: str | None = None


def check_expectations(expect: Expect, result: Any, env: str | None) -> list[str]:
    """Return the failed checks (empty when all pass)."""
    problems: list[str] = []
    if result.status != expect.status:
        why = result.denied_reason or result.message or ""
        suffix = f" ({why})" if why else ""
        problems.append(f"status was {result.status!r}, expected {expect.status!r}{suffix}")
    if expect.answer_contains:
        answer = (result.final_answer or result.message or "").lower()
        for needle in expect.answer_contains:
            if needle.lower() not in answer:
                problems.append(f"answer does not contain {needle!r}")
    if expect.max_steps is not None and result.steps > expect.max_steps:
        problems.append(f"used {result.steps} steps, limit {expect.max_steps}")
    file_checks = expect.file_exists is not None or expect.file_contains is not None
    if file_checks and env not in (None, "live"):
        problems.append(f"file checks are unsupported for env {env!r} (they read this host's files, not the guest's)")
    elif file_checks:
        if expect.file_exists is not None and not Path(expect.file_exists).exists():
            problems.append(f"file does not exist: {expect.file_exists}")
        if expect.file_contains is not None:
            target = Path(expect.file_contains.path)
            try:
                body = target.read_text(encoding="utf-8", errors="replace")
            except OSError:
                problems.append(f"file cannot be read: {target}")
            else:
                if expect.file_contains.text not in body:
                    problems.append(f"file {target} does not contain {expect.file_contains.text!r}")
    return problems


def run_task_file(
    task_file: TaskFile, *, only: list[str] | None = None, fail_fast: bool = False
) -> list[TaskOutcome]:
    """Run the tasks in order. ``only`` keeps tasks with those names; unknown names raise."""
    specs = task_file.tasks
    if only:
        unknown = [n for n in only if n not in {t.name for t in specs}]
        if unknown:
            raise TaskFileError(f"--only: no such task: {', '.join(unknown)}")
        specs = [t for t in specs if t.name in only]
    outcomes: list[TaskOutcome] = []
    for spec in specs:
        outcome = asyncio.run(_run_one(task_file, spec))
        outcomes.append(outcome)
        if fail_fast and not outcome.passed:
            break
    return outcomes


async def _run_one(task_file: TaskFile, spec: TaskSpec) -> TaskOutcome:
    from operon import sdk

    cfg = task_file.resolved(spec)
    started = time.monotonic()
    try:
        def on_approval(request: Any) -> bool:
            # approve_all never overrides a hard policy block (verdict "deny").
            return request.verdict != "deny"

        agent = sdk.Agent(
            desktop=cfg["desktop"] or bool(cfg["env"]),
            env=cfg["env"],
            mode=cfg["mode"],
            max_steps=cfg["max_steps"],
            on_approval=on_approval if cfg["approve_all"] else None,
        )
        result = await agent.arun(spec.task, url=cfg["url"])
    except Exception as exc:  # noqa: BLE001 - one broken task must not stop the suite
        return TaskOutcome(spec.name, "error", 0, None, time.monotonic() - started, False, f"{type(exc).__name__}: {exc}")
    problems = check_expectations(spec.expect, result, cfg["env"])
    if result.status == sdk.STATUS_DENIED and result.denied_reason:
        problems = [p for p in problems if not p.startswith("status was")]
        kind = result.denied_kind
        if kind == "takeover":
            lead = f"run was handed to a person and cannot continue unattended: {result.denied_reason}"
        elif kind == "policy":
            reason = result.denied_reason
            if reason.lower().startswith("blocked by policy"):
                lead = reason[0].lower() + reason[1:]
            else:
                lead = f"blocked by policy: {reason}"
        else:
            lead = f"approval needed and not granted (set approve_all: true to allow): {result.denied_reason}"
        problems.insert(0, lead)
    return TaskOutcome(
        spec.name,
        result.status,
        result.steps,
        result.cost_usd,
        time.monotonic() - started,
        not problems,
        "; ".join(problems),
        result.run_id,
        result.artifacts_dir,
    )


def format_table(outcomes: list[TaskOutcome]) -> str:
    rows = [("name", "status", "steps", "cost", "secs", "result", "reason")]
    for o in outcomes:
        cost = "-" if o.cost_usd is None else f"${o.cost_usd:.3f}"
        rows.append((o.name, o.status, str(o.steps), cost, f"{o.seconds:.1f}", "PASS" if o.passed else "FAIL", o.reason))
    widths = [max(len(r[i]) for r in rows) for i in range(6)]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(r[:6], widths)) + "  " + r[6] for r in rows).rstrip()


_XML_BAD = re.compile("[^\u0009\u000a\u000d\u0020-\ud7ff\ue000-\ufffd\U00010000-\U0010ffff]")


def _xml_safe(text: str) -> str:
    """Replace characters XML 1.0 cannot hold (NUL, ESC, lone surrogates, ...) with U+FFFD."""
    return _XML_BAD.sub("\ufffd", text)


def write_junit(outcomes: list[TaskOutcome], path: str | Path, *, classname: str) -> None:
    failures = sum(1 for o in outcomes if not o.passed)
    root = ET.Element("testsuites")
    suite = ET.SubElement(
        root,
        "testsuite",
        name="operon",
        tests=str(len(outcomes)),
        failures=str(failures),
        errors="0",
        skipped="0",
        time=f"{sum(o.seconds for o in outcomes):.3f}",
    )
    for o in outcomes:
        case = ET.SubElement(suite, "testcase", classname=_xml_safe(classname), name=_xml_safe(o.name), time=f"{o.seconds:.3f}")
        if not o.passed:
            failure = ET.SubElement(case, "failure", message=_xml_safe(o.reason or o.status))
            failure.text = _xml_safe(
                f"run_id: {o.run_id or '-'}\nartifacts: {o.artifacts_dir or '-'}\nstatus: {o.status}\n{o.reason}"
            )
    tree = ET.ElementTree(root)
    ET.indent(tree)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="utf-8", xml_declaration=True)
    ET.parse(path)  # fail loudly if what we wrote does not parse


def tasks_flag_problem(args: Any) -> str | None:
    """Why this `operon run` flag mix is wrong, or None. --tasks excludes the single-run options."""
    if args.tasks:
        single = [
            flag
            for flag, used in (
                ("a task argument", args.task),
                ("--request", args.request),
                ("--api", args.api),
                ("--api-key", args.api_key),
                ("--desktop", args.desktop),
                ("--env", args.env),
                ("--url", args.url),
                ("--mode", args.mode),
                ("--max-steps", args.max_steps is not None),
                ("--headless", args.headless),
            )
            if used
        ]
        if single:
            return "--tasks cannot be combined with " + ", ".join(single) + " (set them in the task file)"
        return None
    stray = [f for f, used in (("--junit", args.junit), ("--only", args.only), ("--fail-fast", args.fail_fast)) if used]
    if stray:
        return ", ".join(stray) + (" only works" if len(stray) == 1 else " only work") + " with --tasks"
    return None


def run_tasks_command(args: Any) -> int:
    """CLI handler for ``operon run --tasks``."""
    path = Path(args.tasks)
    try:
        outcomes = run_task_file(load_task_file(path), only=args.only or None, fail_fast=args.fail_fast)
    except TaskFileError as exc:
        print(f"operon run: {exc}", file=sys.stderr)
        return 2
    print(format_table(outcomes))
    junit_failed = False
    if args.junit:
        try:
            write_junit(outcomes, args.junit, classname=path.stem)
        except Exception as exc:  # noqa: BLE001 - the suite ran; a bad report must not crash the command
            print(f"operon run: could not write the JUnit report {args.junit}: {exc}", file=sys.stderr)
            junit_failed = True
    failed = sum(1 for o in outcomes if not o.passed)
    print(f"{len(outcomes) - failed}/{len(outcomes)} passed")
    return 1 if failed or junit_failed else 0
