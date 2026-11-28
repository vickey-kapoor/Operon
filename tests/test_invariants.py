"""Static guards for load-bearing CLAUDE.md invariants and lessons from past caught mistakes (no live server, no screen)."""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
PERCEPTION_POLICY_DIRS = [
    SRC / "operon" / "agent" / "perception",
    SRC / "operon" / "agent" / "policy",
    SRC / "operon" / "agent" / "backends",
]

# Attribute / name / import identifiers that mean DOM or accessibility-tree access.
FORBIDDEN_IDENTIFIERS = {
    "locator",
    "frame_locator",
    "query_selector",
    "query_selector_all",
    "wait_for_selector",
    "get_by_role",
    "get_by_text",
    "get_by_label",
    "get_by_placeholder",
    "get_by_test_id",
    "elementFromPoint",
    "elementsFromPoint",
    "accessibility",
    "aria_snapshot",
    "xpath",
    "XPath",
}
# String literals that embed DOM-access code (e.g. a page.evaluate JS payload).
# Docstrings and comments are not scanned for identifiers, only real string constants here.
FORBIDDEN_STRING_RE = re.compile(
    r"elementFromPoint|document\.(querySelector|getElementBy|evaluate)|"
    r"\bxpath\b|aria-snapshot|accessibility\.snapshot|\.locator\(",
    re.IGNORECASE,
)
# (relative posix path, token) -> reason. Currently empty: the scan is clean.
# A real violation must be listed with a reason starting with "KNOWN VIOLATION".
ALLOWLIST: dict[tuple[str, str], str] = {}


def _scan(path: Path) -> list[tuple[str, int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    docstring_nodes: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(
            node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        ) and (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
        ):
            docstring_nodes.add(id(node.body[0].value))
    hits: list[tuple[str, int, str]] = []
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Attribute):
            names.append(node.attr)
        elif isinstance(node, ast.Name):
            names.append(node.id)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.extend(node.module.split("."))
            names.extend(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            for a in node.names:
                names.extend(a.name.split("."))
        elif isinstance(node, ast.keyword) and node.arg:
            names.append(node.arg)
        for name in names:
            if name in FORBIDDEN_IDENTIFIERS:
                hits.append((name, getattr(node, "lineno", 0), "identifier"))
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstring_nodes
        ):
            m = FORBIDDEN_STRING_RE.search(node.value)
            if m:
                hits.append((m.group(0), node.lineno, "string"))
    return hits


def test_perception_policy_backends_are_vision_only() -> None:
    violations: list[str] = []
    scanned = 0
    for root in PERCEPTION_POLICY_DIRS:
        assert root.is_dir(), root
        for path in sorted(root.rglob("*.py")):
            scanned += 1
            rel = path.relative_to(SRC).as_posix()
            for token, line, kind in _scan(path):
                if (rel, token) in ALLOWLIST:
                    continue
                violations.append(f"{rel}:{line} {kind} {token!r}")
    assert scanned > 20, "scan found suspiciously few files"
    assert not violations, "DOM/selector/a11y use in vision-only path:\n" + "\n".join(violations)


def test_scanner_detects_violations(tmp_path: Path) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text(
        '"""locator( in a docstring is fine"""\n'
        "def f(page):\n"
        "    page.locator('#x').click()\n"
        "    page.evaluate('document.elementFromPoint(1, 2)')\n",
        encoding="utf-8",
    )
    tokens = {t for t, _, _ in _scan(bad)}
    assert "locator" in tokens
    assert "elementFromPoint" in tokens


def test_browser_manager_does_not_import_ws_stream_at_load() -> None:
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(SRC)!r})\n"
        "import operon.browser.manager\n"
        "assert 'operon.api.ws_stream' not in sys.modules, 'ws_stream imported at load'\n"
    )
    env = {**os.environ, "GEMINI_API_KEY": "fake-test-key", "OPERON_TEST_SAFE_MODE": "true"}
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


# ---------------------------------------------------------------------------
# Lessons from past caught mistakes (KAN-214). Each guard is a pure helper over
# source text/AST, one test over the real tree, and one synthetic-violation test.
# ---------------------------------------------------------------------------


def _read(path: Path) -> str:
    # utf-8-sig: src/operon/clients/gemini.py starts with a BOM.
    return path.read_text(encoding="utf-8-sig")


def _functions(tree: ast.AST) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)]


def _own_nodes(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.AST]:
    """Nodes of fn's body, not descending into nested function definitions."""
    out: list[ast.AST] = []
    stack: list[ast.AST] = list(fn.body)
    while stack:
        node = stack.pop()
        out.append(node)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        stack.extend(ast.iter_child_nodes(node))
    return out


def _call_name(call: ast.Call) -> str:
    """Last name of the callee: `a.b.c(...)` -> "c", `f(...)` -> "f"."""
    f = call.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return ""


def _dotted(node: ast.AST) -> str:
    """Dotted name of a Name/Attribute chain ("subprocess.Popen"), else ""."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


# A. NAVIGATE is guarded (KAN-191, #359): a model-supplied URL must pass is_web_url before goto.
def _goto_sites_and_violations(source: str, rel: str) -> tuple[list[str], list[str]]:
    sites: list[str] = []
    violations: list[str] = []
    for fn in _functions(ast.parse(source)):
        calls = [n for n in _own_nodes(fn) if isinstance(n, ast.Call)]
        gotos = [c for c in calls if isinstance(c.func, ast.Attribute) and c.func.attr == "goto"]
        if not gotos:
            continue
        guarded = any(_call_name(c) == "is_web_url" for c in calls)
        for c in gotos:
            sites.append(f"{rel}:{c.lineno} {fn.name}")
            if not guarded:
                violations.append(f"{rel}:{c.lineno} {fn.name}() calls .goto() without is_web_url")
    return sites, violations


def test_goto_is_guarded_by_is_web_url() -> None:
    sites: list[str] = []
    violations: list[str] = []
    for path in sorted((SRC / "operon").rglob("*.py")):
        s, v = _goto_sites_and_violations(_read(path), path.relative_to(SRC).as_posix())
        sites += s
        violations += v
    assert len(sites) >= 2, f"expected browser_native execute + session_reset, found {sites}"
    assert not violations, "unguarded page.goto:\n" + "\n".join(violations)


def test_goto_guard_detects_violations() -> None:
    bad = "async def f(page, u):\n    await page.goto(u)\n"
    ok = "async def f(page, u):\n    if not is_web_url(u):\n        return\n    await page.goto(u)\n"
    assert len(_goto_sites_and_violations(bad, "bad.py")[1]) == 1
    assert _goto_sites_and_violations(ok, "ok.py")[1] == []


# A2. The start URL is judged (KAN-197): start_run runs the trust gate's verdict through the
# run's permission mode before it navigates, so `ask` mode and a CONFIRM pause the run.
def _start_url_violations(source: str, rel: str) -> list[str]:
    out: list[str] = []
    found = False
    for fn in _functions(ast.parse(source)):
        if fn.name != "start_run":
            continue
        found = True
        calls = [n for n in _own_nodes(fn) if isinstance(n, ast.Call)]
        judged = [c.lineno for c in calls if _call_name(c) == "apply_permission_mode"]
        if not judged:
            out.append(f"{rel}:{fn.lineno} start_run() never calls apply_permission_mode")
            continue
        navs = [c.lineno for c in calls if _call_name(c) == "execute" and c.lineno < min(judged)]
        if navs:
            out.append(f"{rel}:{navs[0]} start_run() executes before apply_permission_mode")
    if not found:
        out.append(f"{rel}: start_run not found")
    return out


def test_start_url_is_judged_with_permission_mode() -> None:
    rel = "operon/agent/loop.py"
    violations = _start_url_violations(_read(SRC / rel), rel)
    assert not violations, "start URL not judged:\n" + "\n".join(violations)


def test_start_url_guard_detects_violations() -> None:
    bad = "class A:\n    async def start_run(self, r):\n        await self.executor.execute(r)\n"
    late = (
        "class A:\n    async def start_run(self, r):\n        await self.executor.execute(r)\n"
        "        v = apply_permission_mode(r, r, None)\n"
    )
    ok = (
        "class A:\n    async def start_run(self, r):\n        v = apply_permission_mode(r, r, None)\n"
        "        await self.executor.execute(r)\n"
    )
    assert len(_start_url_violations(bad, "bad.py")) == 1
    assert len(_start_url_violations(late, "late.py")) == 1
    assert _start_url_violations(ok, "ok.py") == []


# B. Spawned processes use launch_env (KAN-181, #356): the server's secrets in its own
# environment must not leak into the apps and tools Operon launches.
SPAWN_FILES = [
    SRC / "operon" / "executor" / "desktop.py",
    SRC / "operon" / "executor" / "linux_desktop.py",
    SRC / "operon" / "agent" / "tools" / "runner.py",
]
SPAWN_NAMES = {
    "subprocess.Popen",
    "asyncio.create_subprocess_exec",
    "asyncio.create_subprocess_shell",
}


def _spawn_violations(source: str, rel: str) -> list[str]:
    out: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        if _dotted(node.func) in SPAWN_NAMES:
            first = node.args[0] if node.args else None
        elif (
            _call_name(node) == "to_thread"
            and node.args
            and _dotted(node.args[0]) == "subprocess.Popen"
        ):
            first = node.args[1] if len(node.args) > 1 else None
        else:
            continue
        # Allowlist: `taskkill` only kills a pid and runs no user code, so it needs no scrubbed env.
        if isinstance(first, ast.Constant) and first.value == "taskkill":
            continue
        env = next((k.value for k in node.keywords if k.arg == "env"), None)
        if not (isinstance(env, ast.Call) and _call_name(env).endswith("launch_env")):
            out.append(f"{rel}:{node.lineno} spawn without env=launch_env()")
    return out


def test_spawned_processes_use_launch_env() -> None:
    violations: list[str] = []
    for path in SPAWN_FILES:
        violations += _spawn_violations(_read(path), path.relative_to(SRC).as_posix())
    assert not violations, "\n".join(violations)


def test_spawn_guard_detects_violations() -> None:
    assert len(_spawn_violations("subprocess.Popen(cmd)\n", "a.py")) == 1
    assert len(_spawn_violations("subprocess.Popen(cmd, env=os.environ)\n", "a.py")) == 1
    assert len(_spawn_violations("await asyncio.to_thread(subprocess.Popen, cmd)\n", "a.py")) == 1
    assert len(_spawn_violations("await asyncio.create_subprocess_exec('x')\n", "a.py")) == 1
    good = (
        "subprocess.Popen(cmd, env=launch_env())\n"
        "await asyncio.to_thread(subprocess.Popen, cmd, env=_desktop._launch_env())\n"
        "await asyncio.create_subprocess_exec('taskkill', '/F')\n"
    )
    assert _spawn_violations(good, "a.py") == []


# C. Scroll goes through wheel_clicks (KAN-171, #346): pixels are not wheel clicks.
SCROLL_DIRS = [SRC / "operon" / "executor", SRC / "operon" / "environments"]
# (relative posix path, function) -> reason.
SCROLL_ALLOWLIST: dict[tuple[str, str], str] = {
    # The live view receives browser wheel deltas (100 px per notch) and converts them itself.
    ("operon/executor/live_view.py", "user_input"): "converts browser wheel deltas itself",
}


def _scroll_violations(source: str, rel: str) -> tuple[int, list[str]]:
    """Return (functions that scroll, violations)."""
    sites = 0
    out: list[str] = []
    for fn in _functions(ast.parse(source)):
        nodes = _own_nodes(fn)
        scrolls = [
            n.lineno
            for n in nodes
            if isinstance(n, ast.Attribute) and n.attr in {"scroll", "hscroll"}
        ]
        scrolls += [
            n.lineno
            for n in nodes
            if isinstance(n, ast.Constant)
            and isinstance(n.value, str)
            and ("pyautogui.scroll" in n.value or "hscroll" in n.value)
        ]
        if not scrolls:
            continue
        sites += 1
        uses = any(
            (isinstance(n, ast.Name) and n.id == "wheel_clicks")
            or (isinstance(n, ast.Attribute) and n.attr == "wheel_clicks")
            for n in nodes
        )
        if not uses and (rel, fn.name) not in SCROLL_ALLOWLIST:
            out.append(f"{rel}:{min(scrolls)} {fn.name}() scrolls without wheel_clicks")
    return sites, out


def test_scroll_goes_through_wheel_clicks() -> None:
    sites = 0
    violations: list[str] = []
    for root in SCROLL_DIRS:
        for path in sorted(root.rglob("*.py")):
            n, v = _scroll_violations(_read(path), path.relative_to(SRC).as_posix())
            sites += n
            violations += v
    assert sites >= 3, "scan found suspiciously few scroll sites"
    assert not violations, "\n".join(violations)


def test_scroll_guard_detects_violations() -> None:
    bad = "def f(a):\n    pyautogui.scroll(a.scroll_amount)\n"
    assert len(_scroll_violations(bad, "bad.py")[1]) == 1
    bad_str = "def f(a):\n    return f'pyautogui.scroll({a.scroll_amount})'\n"
    assert len(_scroll_violations(bad_str, "bad.py")[1]) == 1
    ok = "def f(a):\n    pyautogui.scroll(wheel_clicks(a.scroll_amount))\n"
    assert _scroll_violations(ok, "ok.py")[1] == []


# D. Tool HTTP client is locked down (KAN-181, #356): no env/proxy trust, no redirects, no bare httpx.get.
TOOLS_DIR = SRC / "operon" / "agent" / "tools"
BARE_HTTPX = {"get", "post", "put", "patch", "delete", "head", "options", "request", "stream"}


def _http_client_violations(source: str, rel: str) -> tuple[int, list[str]]:
    """Return (clients seen, violations)."""
    clients = 0
    out: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        name = _dotted(node.func)
        if name in {"httpx.AsyncClient", "httpx.Client"}:
            clients += 1
            kw = {k.arg: k.value for k in node.keywords}
            for flag in ("trust_env", "follow_redirects"):
                v = kw.get(flag)
                if not (isinstance(v, ast.Constant) and v.value is False):
                    out.append(f"{rel}:{node.lineno} {name}() needs {flag}=False")
        elif name.startswith("httpx.") and name.split(".", 1)[1] in BARE_HTTPX:
            out.append(f"{rel}:{node.lineno} module-level {name}() bypasses the locked-down client")
    return clients, out


def test_tool_http_client_is_locked_down() -> None:
    clients = 0
    violations: list[str] = []
    for path in sorted(TOOLS_DIR.glob("*.py")):
        n, v = _http_client_violations(_read(path), path.relative_to(SRC).as_posix())
        clients += n
        violations += v
    assert clients >= 1, "no httpx client found in agent/tools"
    assert not violations, "\n".join(violations)


def test_http_client_guard_detects_violations() -> None:
    assert len(_http_client_violations("httpx.AsyncClient(timeout=5)\n", "a.py")[1]) == 2
    assert len(_http_client_violations("httpx.Client(trust_env=True, follow_redirects=False)\n", "a.py")[1]) == 1
    assert len(_http_client_violations("httpx.get(url)\n", "a.py")[1]) == 1
    good = "httpx.AsyncClient(timeout=5, trust_env=False, follow_redirects=False)\n"
    assert _http_client_violations(good, "a.py")[1] == []
