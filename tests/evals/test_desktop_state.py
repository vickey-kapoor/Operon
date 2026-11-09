"""Desktop tasks graded from the machine's state: each check on a pass and its near misses,
stale windows ignored, and live_bench using the probe instead of the judge."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from evals import desktop_state, live_bench
from evals.desktop_tasks import DESKTOP_TASKS

T0 = datetime(2026, 11, 8, 15, 0, tzinfo=timezone.utc)
AFTER = (T0 + timedelta(seconds=5)).isoformat()
BEFORE = (T0 - timedelta(minutes=10)).isoformat()


def _window(process: str, title: str = "", *, started: str = AFTER, names=(), selected=(), texts=(), hwnd=1) -> dict:
    return {"title": title or process, "class": "", "pid": 1, "hwnd": hwnd, "process": process, "started": started,
            "rect": [0, 0, 800, 600], "names": list(names), "selected": list(selected), "texts": list(texts)}


def _snap(*wins: dict, explorer=(), folder=False, canvas=None, baseline=None) -> dict:
    return {"windows": list(wins), "explorer": list(explorer), "documents": r"C:\Users\bench\Documents",
            "since": T0.isoformat(), "bench_folder_exists": folder, "paint_canvas": canvas, "baseline": baseline}


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
    settings = _window("ApplicationFrameHost", "Settings", names=["Settings", "Display", "Scale and layout"],
                       selected=["Display"])
    assert _grade("settings_display", _snap(settings))
    win11 = _window("ApplicationFrameHost", "Settings", names=["System > Display", "Display resolution"])
    assert _grade("settings_display", _snap(win11))
    assert not _grade("settings_display", _snap(_window("ApplicationFrameHost", "Settings", names=["Settings", "Home"])))
    # Searching "resolution" on the System page lists a Display entry but isn't the Display page.
    search = _window("ApplicationFrameHost", "Settings", selected=["System"],
                     names=["System", "Display", "Change the resolution of the display"])
    assert not _grade("settings_display", _snap(search))
    assert _grade("taskmgr_performance", _snap(_window("Taskmgr", "Task Manager", selected=["Performance"])))
    # Seen live on Windows Server 2022: no tab names or selection, only the page's own labels.
    perf = _window("Taskmgr", "Task Manager", names=["Tab control", "CPU", "Memory", "Resource Chart View",
                                                     "Utilization", "Base speed:", "% Utilization"])
    assert _grade("taskmgr_performance", _snap(perf))
    processes = _window("Taskmgr", "Task Manager", names=["Tab control", "Name", "Status", "CPU", "Memory",
                                                          "15%", "End task"])
    assert not _grade("taskmgr_performance", _snap(processes))
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
    stats = desktop_state.canvas_stats(bytes(rgb), width, height)
    assert stats["dark_pixels"] == 10 and stats["bbox"] == [10, 1]
    assert desktop_state.canvas_stats(b"\xff" * 30, 5, 2)["dark_pixels"] == 0


def test_canvas_stats_ignore_the_gray_backdrop_around_the_canvas() -> None:
    # Paint's view: a white 40x20 canvas at the top left, gray (200, 210, 225) backdrop
    # to the right and below, with a dark resize handle and shadow outside the canvas.
    width, height = 80, 40
    rgb = bytearray(bytes([200, 210, 225]) * width * height)

    def put(x, y, color):
        i = (y * width + x) * 3
        rgb[i:i + 3] = color

    for y in range(20):
        for x in range(40):
            put(x, y, b"\xff\xff\xff")
    for x in range(40, 80):
        put(x, 21, b"\x60\x60\x60")  # shadow below and beside the canvas
    put(79, 39, b"\x00\x00\x00")  # resize handle
    for y in range(30, 35):  # a white resize handle with a dark border, out on the backdrop
        for x in range(70, 75):
            put(x, y, b"\x00\x00\x00" if x in (70, 74) or y in (30, 34) else b"\xff\xff\xff")
    blank = desktop_state.canvas_stats(bytes(rgb), width, height)
    assert blank["dark_pixels"] == 0 and blank["canvas"][0] <= 40
    for x in range(10, 30):  # a 20x8 rectangle outline on the canvas
        put(x, 5, b"\x00\x00\x00")
        put(x, 12, b"\x00\x00\x00")
    for y in range(5, 13):
        put(10, y, b"\x00\x00\x00")
        put(29, y, b"\x00\x00\x00")
    drawn = desktop_state.canvas_stats(bytes(rgb), width, height)
    assert drawn["bbox"] == [20, 8]


def test_a_shape_touching_the_canvas_edge_is_measured_in_full() -> None:
    width, height = 30, 20
    rgb = bytearray(b"\xff" * width * height * 3)
    for y in range(12):  # a 10x12 outline whose left side is the canvas's first column
        for x in (0, 9):
            i = (y * width + x) * 3
            rgb[i:i + 3] = b"\x00\x00\x00"
    for x in range(10):
        for y in (0, 11):
            i = (y * width + x) * 3
            rgb[i:i + 3] = b"\x00\x00\x00"
    assert desktop_state.canvas_stats(bytes(rgb), width, height)["bbox"] == [10, 12]


def test_notepad_text_is_also_read_from_the_window(monkeypatch) -> None:
    raw = {"windows": [{"process": "notepad", "hwnd": 21, "title": "*Untitled - Notepad", "started": AFTER,
                        "names": ["Text Editor"], "texts": []}], "explorer": [], "documents": "C:\\Docs"}
    monkeypatch.setattr(desktop_state, "_run_probe", lambda *a, **k: desktop_state._normalize(raw))
    monkeypatch.setattr(desktop_state, "_edit_text", lambda hwnd: "Hello from Operon" if hwnd == 21 else "")
    snap = desktop_state.probe(T0, "notepad_type", baseline=[])
    assert _grade("notepad_type", snap)


def test_probe_refuses_on_other_platforms(monkeypatch) -> None:
    monkeypatch.setattr(desktop_state.sys, "platform", "linux")
    with pytest.raises(desktop_state.ProbeError, match="Windows"):
        desktop_state.probe(T0, "notepad_type")


def test_apps_open_before_the_task_never_count() -> None:
    # Hosts like ApplicationFrameHost, Explorer and Windows Terminal outlive tasks:
    # with a baseline, only windows opened during the task are graded.
    calc = _window("ApplicationFrameHost", "Calculator", names=["Display is 444"], hwnd=7)
    assert not _grade("calculator_multiply", _snap(calc, baseline=[7]))
    assert _grade("calculator_multiply", _snap(calc, baseline=[3]))
    terminal = _window("WindowsTerminal", "Windows PowerShell", hwnd=9,
                       texts=["PS C:\\> Get-Date\n\nSunday, November 8, 2026 3:01:12 PM"])
    ok, reason = desktop_state.grade("powershell_date", _snap(terminal, baseline=[9]))
    assert not ok and "no PowerShell window" in reason
    minimized = _window("notepad", texts=["Hello from Operon"], hwnd=4) | {"offscreen": True}
    assert not _grade("notepad_type", _snap(minimized, baseline=[]))
    old = {"path": "C:\\Windows", "hwnd": 5}
    assert not _grade("explorer_windows", _snap(explorer=[old], baseline=[5]))
    assert _grade("explorer_windows", _snap(explorer=[{"path": "C:\\Windows", "hwnd": 6}], baseline=[5]))


def test_single_items_from_the_probe_are_read_as_lists() -> None:
    raw = {"windows": {"process": "notepad", "hwnd": 2, "texts": "Hello from Operon", "names": None},
           "explorer": "C:\\Windows", "documents": "C:\\Docs"}
    snap = desktop_state._normalize(raw)
    snap.update(since=T0.isoformat(), baseline=[], bench_folder_exists=False, paint_canvas=None)
    assert snap["windows"][0]["texts"] == ["Hello from Operon"] and snap["windows"][0]["names"] == []
    assert _grade("explorer_windows", snap)
    assert _grade("notepad_type", snap)


class _Done:
    def __init__(self, stdout: str) -> None:
        self.stdout = stdout


def test_reset_closes_explorer_and_clears_the_folder_the_probe_reads(monkeypatch, tmp_path) -> None:
    documents = tmp_path / "OneDrive" / "Documents"  # a redirected Documents folder
    (documents / desktop_state.BENCH_FOLDER).mkdir(parents=True)
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        return _Done(json.dumps({"windows": [{"hwnd": 11, "process": "WindowsTerminal"}],
                                 "explorer": [{"path": "C:\\Users", "hwnd": 12}], "documents": str(documents)}))

    monkeypatch.setattr(desktop_state.sys, "platform", "win32")
    monkeypatch.setattr(desktop_state.subprocess, "run", run)
    assert desktop_state.prepare("explorer_new_folder") == [11, 12]
    assert not (documents / desktop_state.BENCH_FOLDER).exists()
    assert "-CloseExplorer" in calls[0]
    desktop_state.prepare("calculator_multiply")  # stops a leftover Calculator, keeps consoles
    close = calls[1][calls[1].index("-Close") + 1]
    assert "calculatorapp" in close and "-CloseExplorer" not in calls[1]
    desktop_state.prepare("cmd_echo")
    assert calls[2][calls[2].index("-Close") + 1] == ""


def test_reset_refuses_a_folder_with_files_in_it(monkeypatch, tmp_path) -> None:
    folder = tmp_path / desktop_state.BENCH_FOLDER
    folder.mkdir()
    (folder / "keep.txt").write_text("x")
    monkeypatch.setattr(desktop_state.sys, "platform", "win32")
    monkeypatch.setattr(desktop_state.subprocess, "run",
                        lambda cmd, **kw: _Done(json.dumps({"windows": [], "documents": str(tmp_path)})))
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


async def test_desktop_tasks_are_graded_by_state_not_by_the_judge(monkeypatch, tmp_path) -> None:
    async def judge(task, screenshot):
        raise AssertionError("the judge must not be called for desktop tasks")

    seen = {}

    monkeypatch.setattr(desktop_state, "prepare", lambda name: [99])

    def probe(since, task_name, baseline):
        seen["since"], seen["baseline"] = since, baseline
        return _snap(_window("notepad", texts=["Hello from Operon"]))

    monkeypatch.setattr(desktop_state, "probe", probe)
    monkeypatch.setattr(live_bench, "_final_screenshot", lambda run_id: None)
    monkeypatch.setattr(live_bench, "_run_dir", lambda run_id: tmp_path / run_id)
    task = next(t for t in DESKTOP_TASKS if t.name == "notepad_type")
    result = await live_bench.run_task(_Client(), task, judge, timeout_s=30, approve_pauses=0, mode="desktop")
    assert result.outcome == "success" and "Notepad" in result.judge_reason
    assert result.judge_cost_usd == 0.0
    assert seen["since"].tzinfo is not None and seen["baseline"] == [99]
    # What the grader saw is kept with the run and copied next to the report's screenshot.
    saved = json.loads((tmp_path / "run-1" / "desktop_state.json").read_text(encoding="utf-8"))
    assert saved["windows"][0]["texts"] == ["Hello from Operon"]
    live_bench.save_final_screens([result], tmp_path / "report")
    assert (tmp_path / "report" / "notepad_type.state.json").is_file()


async def test_an_unreadable_desktop_is_an_error_not_a_pass(monkeypatch) -> None:
    monkeypatch.setattr(desktop_state, "prepare", lambda name: [])

    def probe(since, task_name, baseline):
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


async def test_a_run_that_never_started_is_not_probed(monkeypatch) -> None:
    class _DownClient:
        async def post(self, path, json):
            raise httpx.ConnectError("API down")

    def probe(*args):
        raise AssertionError("no probe for a run that never ran")

    async def judge(task, screenshot):
        raise AssertionError("no judge fallback")

    monkeypatch.setattr(desktop_state, "prepare", lambda name: [])
    monkeypatch.setattr(desktop_state, "probe", probe)
    task = next(t for t in DESKTOP_TASKS if t.name == "notepad_type")
    result = await live_bench.run_task(_DownClient(), task, judge, timeout_s=30, approve_pauses=0, mode="desktop")
    assert result.outcome == "error"


def test_a_small_blank_canvas_is_not_mistaken_for_a_drawing() -> None:
    # A 30x20 canvas in a 200x150 view (well under 30% of it), with shadow and handles.
    width, height = 200, 150
    rgb = bytearray(bytes([200, 210, 225]) * width * height)

    def put(x, y, color):
        i = (y * width + x) * 3
        rgb[i:i + 3] = color

    for y in range(20):
        for x in range(30):
            put(x, y, b"\xff\xff\xff")
    for x in range(0, 40):
        put(x, 21, b"\x40\x40\x40")  # shadow
    for y in range(18, 24):  # handle
        for x in range(32, 38):
            put(x, y, b"\x00\x00\x00" if x in (32, 37) or y in (18, 23) else b"\xff\xff\xff")
    stats = desktop_state.canvas_stats(bytes(rgb), width, height)
    assert stats["dark_pixels"] == 0 and stats["canvas"] == [30, 20]


def test_no_visible_canvas_is_a_grader_error() -> None:
    gray = bytes([200, 210, 225]) * 50 * 40
    stats = desktop_state.canvas_stats(gray, 50, 40)
    assert stats["canvas"] == [0, 0]
    paint = _window("mspaint", "Untitled - Paint")
    with pytest.raises(desktop_state.ProbeError, match="could not be found"):
        desktop_state.grade("paint_rectangle", _snap(paint, canvas={"found": True, **stats}))


# --- Season 1 tasks graded from the files they save (KAN-65) -----------------------

def _files(**files) -> dict:
    snapshot = _snap()
    snapshot["files"], snapshot["computer_name"] = files, "BENCH-VM01"
    return snapshot


def test_saved_file_checks_read_the_file_the_task_wrote() -> None:
    assert _grade("notepad_save_file", _files(**{"season.txt": "Operon season one\r\n"}))
    assert not _grade("notepad_save_file", _files(**{"season.txt": "Operon season 1"}))
    assert not _grade("notepad_save_file", _files(**{"season.txt": None}))
    listing = " Directory of C:\\Windows\\System32\\drivers\\etc\r\n hosts\r\n lmhosts.sam\r\n networks\r\n protocol\r\n services"
    assert _grade("cmd_list_to_file", _files(**{"etc-list.txt": listing}))
    assert not _grade("cmd_list_to_file", _files(**{"etc-list.txt": "hosts"}))
    # A partial listing (protocol and lmhosts.sam left out) is not the folder's list.
    assert not _grade("cmd_list_to_file", _files(**{"etc-list.txt": " hosts\r\n networks\r\n services"}))
    assert _grade("powershell_computer_name", _files(**{"computer.txt": "bench-vm01\r\n"}))
    assert not _grade("powershell_computer_name", _files(**{"computer.txt": "localhost"}))
    assert _grade("calculator_to_notepad", _files(**{"result.txt": "69,104"}))
    assert not _grade("calculator_to_notepad", _files(**{"result.txt": "169104"}))


def test_powershell_and_console_encodings_are_read() -> None:
    assert desktop_state.read_text("BENCH-VM01\r\n".encode("utf-16")) == "BENCH-VM01\r\n"
    assert desktop_state.read_text("\ufeffOperon".encode("utf-8")) == "Operon"
    assert desktop_state.read_text(b"caf\xe9") == "café"  # a console code page, not UTF-8


def test_reset_removes_only_the_tasks_own_files(monkeypatch, tmp_path) -> None:
    out = tmp_path / desktop_state.OUTPUT_FOLDER
    out.mkdir()
    (out / "season.txt").write_text("old attempt")
    (out / "keep.txt").write_text("someone's file")
    monkeypatch.setattr(desktop_state.sys, "platform", "win32")
    monkeypatch.setattr(desktop_state.subprocess, "run",
                        lambda cmd, **kw: _Done(json.dumps({"windows": [], "documents": str(tmp_path)})))
    desktop_state.prepare("notepad_save_file")
    assert not (out / "season.txt").exists() and (out / "keep.txt").exists()
    (out / "keep.txt").unlink()
    (out / "season.txt").write_text("old attempt")
    desktop_state.prepare("notepad_save_file")
    assert not out.exists()  # left empty, so removed too


def test_probe_reads_only_the_tasks_files(monkeypatch, tmp_path) -> None:
    out = tmp_path / desktop_state.OUTPUT_FOLDER
    out.mkdir()
    (out / "result.txt").write_bytes("69104".encode("utf-16"))
    (out / "private.txt").write_text("not for the grader")
    monkeypatch.setattr(desktop_state.sys, "platform", "win32")
    monkeypatch.setenv("COMPUTERNAME", "BENCH-VM01")
    monkeypatch.setattr(desktop_state.subprocess, "run",
                        lambda cmd, **kw: _Done(json.dumps({"windows": [], "documents": str(tmp_path)})))
    snapshot = desktop_state.probe(T0, "calculator_to_notepad", [])
    assert snapshot["files"] == {"result.txt": "69104"} and snapshot["computer_name"] == "BENCH-VM01"
    assert desktop_state.grade("calculator_to_notepad", snapshot)[0]
    assert desktop_state.probe(T0, "cmd_echo", [])["files"] == {}
