"""Desktop tasks graded from the machine's state: each check on a pass and its near misses,
stale windows ignored, and live_bench using the probe instead of the judge."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from evals import desktop_state, live_bench
from evals.desktop_tasks import DESKTOP_TASKS

T0 = datetime(2026, 11, 8, 15, 0, tzinfo=timezone.utc)
AFTER = (T0 + timedelta(seconds=5)).isoformat()
BEFORE = (T0 - timedelta(minutes=10)).isoformat()


def _window(process: str, title: str = "", *, started: str = AFTER, names=(), selected=(), texts=(), hwnd=1) -> dict:
    return {"title": title or process, "class": "", "pid": 1, "hwnd": hwnd, "process": process, "started": started,
            "rect": [0, 0, 800, 600], "names": list(names), "selected": list(selected), "texts": list(texts)}


def _snap(*wins: dict, explorer=(), folder=False, canvas=None) -> dict:
    return {"windows": list(wins), "explorer": list(explorer), "documents": r"C:\Users\bench\Documents",
            "since": T0.isoformat(), "bench_folder_exists": folder, "paint_canvas": canvas}


def _grade(name: str, snapshot: dict) -> bool:
    return desktop_state.grade(name, snapshot)[0]


def test_every_desktop_task_has_a_state_check() -> None:
    assert {task.name for task in DESKTOP_TASKS} == set(desktop_state.CHECKS)


def test_notepad_checks_read_the_document_text() -> None:
    assert _grade("notepad_type", _snap(_window("notepad", texts=["Hello from Operon"])))
    assert not _grade("notepad_type", _snap(_window("notepad", texts=["Hello from"])))
    assert not _grade("notepad_type", _snap())
    two = _window("Notepad", texts=["first line\r\nsecond line"])
    assert _grade("notepad_two_lines", _snap(two))
    assert not _grade("notepad_two_lines", _snap(_window("notepad", texts=["first line second line"])))
    assert not _grade("notepad_two_lines", _snap(_window("notepad", texts=["second line\nfirst line"])))


def test_a_window_from_before_the_task_does_not_count() -> None:
    stale = _window("notepad", texts=["Hello from Operon"], started=BEFORE)
    ok, reason = desktop_state.grade("notepad_type", _snap(stale))
    assert not ok and "no Notepad window" in reason


def test_explorer_checks_read_the_folder_shown() -> None:
    assert _grade("explorer_windows", _snap(explorer=["C:\\Windows\\"]))
    assert not _grade("explorer_windows", _snap(explorer=[r"C:\Windows\System32"]))
    docs = r"C:\Users\bench\Documents"
    assert _grade("explorer_new_folder", _snap(explorer=[docs], folder=True))
    assert not _grade("explorer_new_folder", _snap(explorer=[docs], folder=False))
    assert not _grade("explorer_new_folder", _snap(explorer=[r"C:\Users\bench"], folder=True))


def test_console_checks_read_the_buffer() -> None:
    cmd = _window("cmd", "C:\\Windows\\system32\\cmd.exe",
                  texts=["Microsoft Windows\nC:\\Users\\bench>echo operon-bench-ok\noperon-bench-ok\n\nC:\\Users\\bench>"])
    assert _grade("cmd_echo", _snap(cmd))
    typed_only = _window("cmd", "Command Prompt", texts=["C:\\Users\\bench>echo operon-bench-ok"])
    assert not _grade("cmd_echo", _snap(typed_only))
    ps = _window("powershell", "Windows PowerShell",
                 texts=["PS C:\\Users\\bench> Get-Date\n\nSunday, November 8, 2026 3:01:12 PM\n\nPS C:\\Users\\bench>"])
    assert _grade("powershell_date", _snap(ps))
    no_output = _window("powershell", "Windows PowerShell", texts=["PS C:\\Users\\bench> Get-Date"])
    assert not _grade("powershell_date", _snap(no_output))


def test_uia_checks_for_settings_task_manager_and_calculator() -> None:
    settings = _window("ApplicationFrameHost", "Settings", names=["Settings", "Display", "Scale and layout"])
    assert _grade("settings_display", _snap(settings))
    assert not _grade("settings_display", _snap(_window("ApplicationFrameHost", "Settings", names=["Settings", "Home"])))
    assert _grade("taskmgr_performance", _snap(_window("Taskmgr", "Task Manager", selected=["Performance"])))
    assert not _grade("taskmgr_performance", _snap(_window("Taskmgr", "Task Manager", selected=["Processes"])))
    calc = _window("ApplicationFrameHost", "Calculator", names=["Calculator", "Display is 444"])
    assert _grade("calculator_multiply", _snap(calc))
    assert not _grade("calculator_multiply", _snap(_window("ApplicationFrameHost", "Calculator", names=["Display is 4,440"])))


def test_paint_check_needs_a_drawn_shape() -> None:
    paint = _window("mspaint", "Untitled - Paint")
    assert _grade("paint_rectangle", _snap(paint, canvas={"found": True, "dark_pixels": 400, "bbox": [120, 80]}))
    assert not _grade("paint_rectangle", _snap(paint, canvas={"found": True, "dark_pixels": 0, "bbox": [0, 0]}))
    assert not _grade("paint_rectangle", _snap(paint, canvas={"found": True, "dark_pixels": 50, "bbox": [50, 1]}))


def test_an_unreadable_paint_canvas_is_a_grader_error_not_a_miss() -> None:
    paint = _window("mspaint", "Untitled - Paint")
    with pytest.raises(desktop_state.ProbeError, match="capture failed"):
        desktop_state.grade("paint_rectangle", _snap(paint, canvas={"found": False, "error": "capture failed"}))
    with pytest.raises(desktop_state.ProbeError):
        desktop_state.grade("paint_rectangle", _snap(paint, canvas=None))


def test_canvas_stats_counts_dark_pixels_and_their_box() -> None:
    width, height = 20, 10
    rgb = bytearray(b"\xff" * width * height * 3)
    for x in range(2, 12):  # a 10 px dark line on row 3
        i = (3 * width + x) * 3
        rgb[i:i + 3] = b"\x00\x00\x00"
    assert desktop_state.canvas_stats(bytes(rgb), width, height) == {"dark_pixels": 10, "bbox": [10, 1]}
    assert desktop_state.canvas_stats(b"\xff" * 30, 5, 2)["dark_pixels"] == 0


def test_probe_refuses_on_other_platforms(monkeypatch) -> None:
    monkeypatch.setattr(desktop_state.sys, "platform", "linux")
    with pytest.raises(desktop_state.ProbeError, match="Windows"):
        desktop_state.probe(T0)


class _Done:
    def __init__(self, stdout: str) -> None:
        self.stdout = stdout


def test_reset_closes_explorer_and_clears_the_folder_the_probe_reads(monkeypatch, tmp_path) -> None:
    documents = tmp_path / "OneDrive" / "Documents"  # a redirected Documents folder
    (documents / desktop_state.BENCH_FOLDER).mkdir(parents=True)
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd[-1])
        return _Done(f"{documents}\r\n")

    monkeypatch.setattr(desktop_state.sys, "platform", "win32")
    monkeypatch.setattr(desktop_state.subprocess, "run", run)
    desktop_state.prepare("explorer_new_folder")
    assert not (documents / desktop_state.BENCH_FOLDER).exists()
    assert "Quit()" in calls[0] and "MyDocuments" in calls[0]
    desktop_state.prepare("explorer_windows")  # closes old windows too
    assert len(calls) == 2
    desktop_state.prepare("notepad_type")  # nothing to reset
    assert len(calls) == 2


def test_reset_refuses_a_folder_with_files_in_it(monkeypatch, tmp_path) -> None:
    folder = tmp_path / desktop_state.BENCH_FOLDER
    folder.mkdir()
    (folder / "keep.txt").write_text("x")
    monkeypatch.setattr(desktop_state.sys, "platform", "win32")
    monkeypatch.setattr(desktop_state.subprocess, "run", lambda cmd, **kw: _Done(str(tmp_path)))
    with pytest.raises(desktop_state.ProbeError, match="not empty"):
        desktop_state.prepare("explorer_new_folder")
    assert (folder / "keep.txt").exists()


class _Response:
    def __init__(self, data: dict) -> None:
        self._data = data

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._data


class _Client:
    async def post(self, path: str, json: dict) -> _Response:
        if path.endswith("/run-task"):
            return _Response({"run_id": "run-1", "status": "running"})
        return _Response({"status": "succeeded", "step_count": 3})


async def test_desktop_tasks_are_graded_by_state_not_by_the_judge(monkeypatch) -> None:
    async def judge(task, screenshot):
        raise AssertionError("the judge must not be called for desktop tasks")

    seen = {}

    def probe(since):
        seen["since"] = since
        return _snap(_window("notepad", texts=["Hello from Operon"]))

    monkeypatch.setattr(desktop_state, "probe", probe)
    monkeypatch.setattr(live_bench, "_final_screenshot", lambda run_id: None)
    task = next(t for t in DESKTOP_TASKS if t.name == "notepad_type")
    result = await live_bench.run_task(_Client(), task, judge, timeout_s=30, approve_pauses=0, mode="desktop")
    assert result.outcome == "success" and "Notepad" in result.judge_reason
    assert result.judge_cost_usd == 0.0
    assert seen["since"].tzinfo is not None


async def test_an_unreadable_desktop_is_an_error_not_a_pass(monkeypatch) -> None:
    def probe(since):
        raise desktop_state.ProbeError("desktop probe failed: boom")

    async def judge(task, screenshot):
        raise AssertionError("no judge fallback")

    monkeypatch.setattr(desktop_state, "probe", probe)
    monkeypatch.setattr(live_bench, "_final_screenshot", lambda run_id: None)
    task = next(t for t in DESKTOP_TASKS if t.name == "cmd_echo")
    result = await live_bench.run_task(_Client(), task, judge, timeout_s=30, approve_pauses=0, mode="desktop")
    assert result.outcome == "error" and "boom" in result.judge_reason


async def test_a_failed_reset_skips_the_run_and_is_an_error(monkeypatch) -> None:
    def prepare(name):
        raise desktop_state.ProbeError("desktop reset failed: boom")

    class _NoRunClient:
        async def post(self, path, json):
            raise AssertionError("no run should start when the desktop could not be reset")

    async def judge(task, screenshot):
        raise AssertionError("no judge fallback")

    monkeypatch.setattr(desktop_state, "prepare", prepare)
    task = next(t for t in DESKTOP_TASKS if t.name == "explorer_windows")
    result = await live_bench.run_task(_NoRunClient(), task, judge, timeout_s=30, approve_pauses=0, mode="desktop")
    assert result.outcome == "error" and "boom" in result.judge_reason and result.run_id is None
