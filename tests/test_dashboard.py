"""
Tests for the Test Dashboard system.

Covers:
  - agent_runner.py: pytest output parsing, report generation, scenario dispatch,
    TimeoutExpired handling (BUG-1), failure message extraction from correct stream (BUG-4)
  - dashboard_server.py: all endpoints, duplicate-run protection, runner/judge race guard
    (BUG-2), exit-code fields in /status (BUG-3), file handle leak (BUG-5),
    _last_run timestamp set by fix-loop (BUG-6)
  - judge_runner.py: missing API key, missing report file, strip_fences helper

All BUG-1 through BUG-6 fixes have been applied — xfail markers removed.

All subprocess calls are mocked — no real test suite or server is started.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure both test helper modules are importable
# ---------------------------------------------------------------------------
_TESTS_DIR = Path(__file__).parent
_REPO_ROOT = _TESTS_DIR.parent

sys.path.insert(0, str(_TESTS_DIR))
sys.path.insert(0, str(_REPO_ROOT))

import agent_runner as ar
import judge_runner as jr
import dashboard_server as ds


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

PYTEST_PASSED_OUTPUT = textwrap.dedent("""\
    tests/test_webpilot_e2e.py::test_session_lifecycle PASSED          [  4%]
    tests/test_webpilot_e2e.py::test_task_navigate_and_done PASSED     [  8%]
""")

PYTEST_FAILED_OUTPUT = textwrap.dedent("""\
    tests/test_webpilot_e2e.py::test_confirm_flow PASSED               [ 12%]
    tests/test_webpilot_e2e.py::test_confirm_denied FAILED             [ 16%]
    FAILED tests/test_webpilot_e2e.py::test_confirm_denied - AssertionError: expected stopped
""")

PYTEST_ERROR_OUTPUT = textwrap.dedent("""\
    tests/test_webpilot_e2e.py::test_interrupt_redirect ERROR          [ 20%]
    ERROR tests/test_webpilot_e2e.py::test_interrupt_redirect - RuntimeError: server died
""")

PYTEST_MIXED_OUTPUT = textwrap.dedent("""\
    tests/test_webpilot_e2e.py::test_session_lifecycle PASSED          [  4%]
    tests/test_webpilot_e2e.py::test_task_navigate_and_done PASSED     [  8%]
    tests/test_webpilot_e2e.py::test_confirm_denied FAILED             [ 12%]
    tests/test_webpilot_e2e.py::test_interrupt_redirect ERROR          [ 16%]
    FAILED tests/test_webpilot_e2e.py::test_confirm_denied - AssertionError: status mismatch
    ERROR tests/test_webpilot_e2e.py::test_interrupt_redirect - TimeoutError
""")


def _make_completed_process(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


def _make_mock_popen(returncode=0, stdout="", stderr=""):
    """Create a MagicMock that behaves like subprocess.Popen for agent_runner."""
    mock_proc = MagicMock()
    mock_proc.communicate.return_value = (stdout, stderr)
    mock_proc.returncode = returncode
    mock_proc.pid = 12345
    mock_proc.kill.return_value = None
    return mock_proc


def _reset_ds_state():
    """Reset all module-level state in dashboard_server between tests."""
    ds._runner_proc = None
    ds._judge_proc = None
    ds._fix_loop_proc = None
    ds._last_run = None
    # These may not exist yet (BUG-5: file handle tracking not implemented)
    if hasattr(ds, "_runner_log_fh"):
        ds._runner_log_fh = None
    if hasattr(ds, "_fix_loop_log_fh"):
        ds._fix_loop_log_fh = None


def _make_handler(path: str = "/status", method: str = "GET"):
    """Construct a DashboardHandler without a real socket for unit testing."""
    _reset_ds_state()

    handler = ds.DashboardHandler.__new__(ds.DashboardHandler)
    handler._output_buf = BytesIO()
    handler.wfile = handler._output_buf
    handler.path = path
    handler.command = method
    handler._headers_sent = []
    handler._response_code = None

    handler.send_response = lambda code: setattr(handler, "_response_code", code)
    handler.send_header = lambda k, v: handler._headers_sent.append((k, v))
    handler.end_headers = lambda: None
    return handler


def _read_json_response(handler) -> dict:
    handler._output_buf.seek(0)
    return json.loads(handler._output_buf.read())


# ===========================================================================
# agent_runner — pytest output parsing
# ===========================================================================

class TestParsePytestOutput:
    def test_all_passed_returns_passed_entries(self):
        results = ar._parse_pytest_output(PYTEST_PASSED_OUTPUT, "")
        assert len(results) == 2
        for r in results:
            assert r["status"] == "passed"
            assert r["message"] == ""

    def test_extracts_test_name_from_double_colon_notation(self):
        results = ar._parse_pytest_output(PYTEST_PASSED_OUTPUT, "")
        names = [r["name"] for r in results]
        assert "test_session_lifecycle" in names
        assert "test_task_navigate_and_done" in names

    def test_failed_status_detected(self):
        results = ar._parse_pytest_output(PYTEST_FAILED_OUTPUT, "")
        statuses = {r["name"]: r["status"] for r in results}
        assert statuses["test_confirm_flow"] == "passed"
        assert statuses["test_confirm_denied"] == "failed"

    def test_error_status_detected(self):
        results = ar._parse_pytest_output(PYTEST_ERROR_OUTPUT, "")
        assert results[0]["status"] == "error"
        assert results[0]["name"] == "test_interrupt_redirect"

    def test_mixed_statuses_all_captured(self):
        results = ar._parse_pytest_output(PYTEST_MIXED_OUTPUT, "")
        statuses = {r["name"]: r["status"] for r in results}
        assert statuses["test_session_lifecycle"] == "passed"
        assert statuses["test_task_navigate_and_done"] == "passed"
        assert statuses["test_confirm_denied"] == "failed"
        assert statuses["test_interrupt_redirect"] == "error"

    def test_empty_stdout_returns_empty_list(self):
        assert ar._parse_pytest_output("", "") == []

    def test_no_double_colon_skipped_as_non_test_line(self):
        # Lines without "::" are not real pytest test results — skip them (BUG-8 fix)
        output = "some_test_module PASSED"
        results = ar._parse_pytest_output(output, "")
        assert results == []

    def test_percentage_progress_lines_not_treated_as_tests(self):
        # Summary line: "24 passed" uses lowercase — PASSED token check is uppercase
        output = "24 passed, 0 failed in 5.32s\n"
        results = ar._parse_pytest_output(output, "")
        assert results == []


# ===========================================================================
# agent_runner — failure message extraction (BUG-4 fix verification)
# ===========================================================================

class TestExtractFailureMessage:
    def test_finds_message_in_combined_text(self):
        combined = textwrap.dedent("""\
            FAILED tests/test_webpilot_e2e.py::test_confirm_denied - AssertionError
            AssertionError: expected 'stopped' but got 'running'
            some extra detail
        """)
        msg = ar._extract_failure_message(combined, "test_confirm_denied")
        assert "AssertionError" in msg

    def test_returns_empty_when_test_name_not_present(self):
        msg = ar._extract_failure_message("some unrelated output", "test_nonexistent")
        assert msg == ""

    def test_caps_output_at_500_chars(self):
        long_line = "x" * 600
        combined = f"FAILED tests/test_foo.py::test_bar\n{long_line}"
        msg = ar._extract_failure_message(combined, "test_bar")
        assert len(msg) <= 500


    def test_parse_pytest_output_passes_stdout_to_extract(self):
        """BUG-4: failure message must be extracted from stdout, not only stderr."""
        stdout = textwrap.dedent("""\
            tests/test_webpilot_e2e.py::test_confirm_denied FAILED     [  4%]
            FAILED tests/test_webpilot_e2e.py::test_confirm_denied - AssertionError: sentinel_error_text
        """)
        stderr = ""  # empty — would give empty message before the fix
        results = ar._parse_pytest_output(stdout, stderr)
        failed = next(r for r in results if r["name"] == "test_confirm_denied")
        assert "sentinel_error_text" in failed["message"], (
            "BUG-4: failure message must come from stdout, not only stderr"
        )


# ===========================================================================
# agent_runner — run_scenario with subprocess mocking
# ===========================================================================

class TestRunScenario:
    def test_unknown_scenario_returns_error_report(self):
        report = ar.run_scenario("nonexistent_scenario")
        assert report["passed"] is False
        assert report["scenario"] == "nonexistent_scenario"
        assert report["tests"][0]["status"] == "error"
        assert "Unknown scenario" in report["tests"][0]["message"]

    def test_unknown_scenario_lists_available_scenarios(self):
        report = ar.run_scenario("bad")
        msg = report["tests"][0]["message"]
        for key in ar._SCENARIO_FILTERS:
            assert key in msg

    @patch("agent_runner.subprocess.Popen")
    def test_passing_run_sets_passed_true(self, mock_popen):
        mock_popen.return_value = _make_mock_popen(
            returncode=0, stdout=PYTEST_PASSED_OUTPUT
        )
        with patch("agent_runner._read_ws_message_log", return_value=[]):
            with patch("agent_runner._clear_ws_message_log"):
                report = ar.run_scenario("session_lifecycle")
        assert report["passed"] is True
        assert report["scenario"] == "session_lifecycle"

    @patch("agent_runner.subprocess.Popen")
    def test_failing_run_sets_passed_false(self, mock_popen):
        mock_popen.return_value = _make_mock_popen(
            returncode=1, stdout=PYTEST_FAILED_OUTPUT
        )
        with patch("agent_runner._read_ws_message_log", return_value=[]):
            with patch("agent_runner._clear_ws_message_log"):
                report = ar.run_scenario("confirm_flow")
        assert report["passed"] is False

    @patch("agent_runner.subprocess.Popen")
    def test_report_includes_ws_message_log(self, mock_popen):
        mock_popen.return_value = _make_mock_popen(returncode=0, stdout="")
        ws_log = [{"type": "action", "direction": "recv"}]
        with patch("agent_runner._read_ws_message_log", return_value=ws_log):
            with patch("agent_runner._clear_ws_message_log"):
                report = ar.run_scenario("task_flow")
        assert report["ws_message_log"] == ws_log

    @patch("agent_runner.subprocess.Popen")
    def test_report_always_has_suggestions_key(self, mock_popen):
        mock_popen.return_value = _make_mock_popen(returncode=0, stdout="")
        with patch("agent_runner._read_ws_message_log", return_value=[]):
            with patch("agent_runner._clear_ws_message_log"):
                report = ar.run_scenario("task_flow")
        assert "suggestions" in report

    @patch("agent_runner.subprocess.Popen")
    def test_env_sets_google_api_key_stub(self, mock_popen):
        mock_popen.return_value = _make_mock_popen(returncode=0, stdout="")
        with patch("agent_runner._read_ws_message_log", return_value=[]):
            with patch("agent_runner._clear_ws_message_log"):
                ar.run_scenario("task_flow")
        call_kwargs = mock_popen.call_args[1]
        env_used = call_kwargs["env"]
        assert "GOOGLE_API_KEY" in env_used

    @patch("agent_runner.subprocess.Popen")
    def test_scenario_all_maps_to_test_webpilot_e2e(self, mock_popen):
        mock_popen.return_value = _make_mock_popen(returncode=0, stdout="")
        with patch("agent_runner._read_ws_message_log", return_value=[]):
            with patch("agent_runner._clear_ws_message_log"):
                ar.run_scenario("all")
        cmd = mock_popen.call_args[0][0]
        assert "tests/test_webpilot_e2e.py" in cmd
        # -k filter for 'all' should be the module-level marker
        assert "test_webpilot_e2e" in cmd


# ===========================================================================
# agent_runner — BUG-1: TimeoutExpired handling
# ===========================================================================

class TestRunScenarioTimeout:

    @patch("agent_runner.subprocess.Popen")
    def test_timeout_returns_error_report(self, mock_popen):
        """BUG-1: TimeoutExpired must not propagate; must return a structured error report."""
        mock_proc = _make_mock_popen(returncode=1, stdout="", stderr="")
        mock_proc.communicate.side_effect = [
            subprocess.TimeoutExpired(cmd=["pytest"], timeout=120),
            ("", ""),  # second call after kill
        ]
        mock_proc.pid = 99999
        mock_popen.return_value = mock_proc

        with patch("agent_runner._read_ws_message_log", return_value=[]):
            with patch("agent_runner._clear_ws_message_log"):
                with patch("agent_runner.subprocess.run"):  # mock taskkill
                    report = ar.run_scenario("session_lifecycle")

        assert report["passed"] is False
        assert report["scenario"] == "session_lifecycle"
        # Should have at least the timeout error entry
        timeout_entries = [t for t in report["tests"] if "timed out" in t.get("message", "")]
        assert len(timeout_entries) >= 1


    @patch("agent_runner.subprocess.Popen")
    def test_timeout_kills_stuck_process(self, mock_popen):
        """The stuck child process must be killed on timeout (via taskkill on Windows or proc.kill)."""
        mock_proc = _make_mock_popen(returncode=1, stdout="", stderr="")
        mock_proc.communicate.side_effect = [
            subprocess.TimeoutExpired(cmd=["pytest"], timeout=120),
            ("", ""),
        ]
        mock_proc.pid = 99999
        mock_popen.return_value = mock_proc

        with patch("agent_runner._read_ws_message_log", return_value=[]):
            with patch("agent_runner._clear_ws_message_log"):
                with patch("agent_runner.subprocess.run") as mock_taskkill:
                    with patch("agent_runner.sys") as mock_sys:
                        mock_sys.platform = "linux"
                        mock_sys.executable = sys.executable
                        ar.run_scenario("session_lifecycle")

        mock_proc.kill.assert_called_once()


    @patch("agent_runner.subprocess.Popen")
    def test_timeout_still_includes_ws_message_log(self, mock_popen):
        """Even on timeout, partial ws_message_log from file should be returned."""
        mock_proc = _make_mock_popen(returncode=1, stdout="", stderr="")
        mock_proc.communicate.side_effect = [
            subprocess.TimeoutExpired(cmd=["pytest"], timeout=120),
            ("", ""),
        ]
        mock_proc.pid = 99999
        mock_popen.return_value = mock_proc

        partial_log = [{"type": "thinking", "direction": "recv"}]
        with patch("agent_runner._read_ws_message_log", return_value=partial_log):
            with patch("agent_runner._clear_ws_message_log"):
                with patch("agent_runner.subprocess.run"):
                    report = ar.run_scenario("task_flow")

        assert report["ws_message_log"] == partial_log


    @patch("agent_runner.subprocess.Popen")
    def test_timeout_with_unkillable_process_does_not_raise(self, mock_popen):
        """kill() may fail on already-dead processes; that must not surface to the caller."""
        mock_proc = _make_mock_popen(returncode=1, stdout="", stderr="")
        mock_proc.communicate.side_effect = [
            subprocess.TimeoutExpired(cmd=["pytest"], timeout=120),
            ("", ""),
        ]
        mock_proc.pid = 99999
        mock_proc.kill.side_effect = OSError("already dead")
        mock_popen.return_value = mock_proc

        with patch("agent_runner._read_ws_message_log", return_value=[]):
            with patch("agent_runner._clear_ws_message_log"):
                with patch("agent_runner.subprocess.run", side_effect=OSError("no taskkill")):
                    report = ar.run_scenario("session_lifecycle")  # must not raise

        assert report["passed"] is False


# ===========================================================================
# agent_runner — write_report
# ===========================================================================

class TestWriteReport:
    def test_writes_json_to_tmp_file(self, tmp_path):
        report = {
            "scenario": "all", "passed": True,
            "tests": [], "ws_message_log": [], "suggestions": [],
        }
        import tempfile as _tf
        orig = _tf.gettempdir
        _tf.gettempdir = lambda: str(tmp_path)
        # Also patch the reference held inside agent_runner's own module namespace
        with patch("agent_runner.tempfile.gettempdir", return_value=str(tmp_path)):
            try:
                ar.write_report(report)
            finally:
                _tf.gettempdir = orig
        target = tmp_path / "wp_test_report.json"
        written = json.loads(target.read_text())
        assert written["passed"] is True

    def test_handles_os_error_gracefully(self):
        report = {
            "scenario": "all", "passed": True,
            "tests": [], "ws_message_log": [], "suggestions": [],
        }
        with patch("pathlib.Path.write_text", side_effect=OSError("disk full")):
            ar.write_report(report)  # must not raise


# ===========================================================================
# agent_runner — _read_ws_message_log
# ===========================================================================

class TestReadWsMessageLog:
    def test_returns_empty_list_when_file_absent(self, tmp_path):
        missing = tmp_path / "no_such_file.json"
        with patch("agent_runner._MSG_LOG_PATH", missing):
            assert ar._read_ws_message_log() == []

    def test_returns_parsed_json_when_file_present(self, tmp_path):
        log_data = [{"type": "action", "direction": "recv"}]
        log_file = tmp_path / "wp_test_messages.json"
        log_file.write_text(json.dumps(log_data))
        with patch("agent_runner._MSG_LOG_PATH", log_file):
            result = ar._read_ws_message_log()
        assert result == log_data

    def test_returns_empty_list_on_invalid_json(self, tmp_path):
        log_file = tmp_path / "wp_test_messages.json"
        log_file.write_text("{invalid json{{")
        with patch("agent_runner._MSG_LOG_PATH", log_file):
            result = ar._read_ws_message_log()
        assert result == []


# ===========================================================================
# dashboard_server — _is_running helper
# ===========================================================================

class TestIsRunning:
    def test_none_proc_returns_false(self):
        assert ds._is_running(None) is False

    def test_running_proc_returns_true(self):
        proc = MagicMock()
        proc.poll.return_value = None
        assert ds._is_running(proc) is True

    def test_finished_proc_returns_false(self):
        proc = MagicMock()
        proc.poll.return_value = 0
        assert ds._is_running(proc) is False

    def test_proc_with_nonzero_exit_returns_false(self):
        proc = MagicMock()
        proc.poll.return_value = 1
        assert ds._is_running(proc) is False


# ===========================================================================
# dashboard_server — /status endpoint
# ===========================================================================

class TestGetStatus:
    def test_all_fields_present_when_no_processes(self):
        h = _make_handler("/status")
        h._get_status()
        data = _read_json_response(h)
        assert "runner_running" in data
        assert "judge_running" in data
        assert "fix_loop_running" in data
        assert "last_run" in data


    def test_exit_code_fields_present(self):
        """BUG-3: /status must expose runner_exit_code, judge_exit_code, fix_loop_exit_code."""
        h = _make_handler("/status")
        h._get_status()
        data = _read_json_response(h)
        assert "runner_exit_code" in data, "BUG-3: runner_exit_code missing from /status"
        assert "judge_exit_code" in data, "BUG-3: judge_exit_code missing from /status"
        assert "fix_loop_exit_code" in data, "BUG-3: fix_loop_exit_code missing from /status"


    def test_exit_codes_none_when_no_processes_started(self):
        h = _make_handler("/status")
        h._get_status()
        data = _read_json_response(h)
        assert data["runner_exit_code"] is None
        assert data["judge_exit_code"] is None
        assert data["fix_loop_exit_code"] is None


    def test_exit_code_populated_after_process_finishes(self):
        """BUG-3: exit code must reflect actual process return code once done."""
        h = _make_handler("/status")
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0      # finished
        mock_proc.returncode = 0
        ds._runner_proc = mock_proc          # inject after _make_handler reset
        h._get_status()
        data = _read_json_response(h)
        assert data["runner_exit_code"] == 0
        assert data["runner_running"] is False


    def test_nonzero_exit_code_reported(self):
        h = _make_handler("/status")
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1
        mock_proc.returncode = 1
        ds._runner_proc = mock_proc
        h._get_status()
        data = _read_json_response(h)
        assert data["runner_exit_code"] == 1


    def test_exit_code_none_while_process_still_running(self):
        h = _make_handler("/status")
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None   # still running
        ds._runner_proc = mock_proc
        h._get_status()
        data = _read_json_response(h)
        assert data["runner_running"] is True
        assert data["runner_exit_code"] is None


    def test_closes_runner_log_fh_when_process_done(self):
        """BUG-5: log file handle must be closed when process finishes."""
        h = _make_handler("/status")
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1      # finished with failure
        mock_proc.returncode = 1
        mock_fh = MagicMock()
        ds._runner_proc = mock_proc
        ds._runner_log_fh = mock_fh  # attribute may not exist before fix
        h._get_status()
        mock_fh.close.assert_called_once()
        assert ds._runner_log_fh is None

    def test_does_not_close_runner_log_fh_while_running(self):
        """BUG-5: handle must stay open while process is still writing."""
        h = _make_handler("/status")
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None   # still running
        mock_fh = MagicMock()
        ds._runner_proc = mock_proc
        ds._runner_log_fh = mock_fh  # attribute may not exist before fix
        h._get_status()
        mock_fh.close.assert_not_called()


    def test_closes_fix_loop_log_fh_when_process_done(self):
        """BUG-5: fix loop log handle must also be closed when done."""
        h = _make_handler("/status")
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        mock_proc.returncode = 0
        mock_fh = MagicMock()
        ds._fix_loop_proc = mock_proc
        ds._fix_loop_log_fh = mock_fh  # attribute may not exist before fix
        h._get_status()
        mock_fh.close.assert_called_once()
        assert ds._fix_loop_log_fh is None

    def test_status_reflects_multiple_running_processes(self):
        h = _make_handler("/status")
        mock_runner = MagicMock()
        mock_runner.poll.return_value = None
        mock_judge = MagicMock()
        mock_judge.poll.return_value = None
        ds._runner_proc = mock_runner
        ds._judge_proc = mock_judge
        h._get_status()
        data = _read_json_response(h)
        assert data["runner_running"] is True
        assert data["judge_running"] is True


# ===========================================================================
# dashboard_server — /run endpoint
# ===========================================================================

class TestRunTests:
    @patch("dashboard_server.subprocess.Popen")
    @patch("builtins.open", return_value=MagicMock())
    def test_starts_process_and_returns_started(self, mock_open, mock_popen):
        mock_popen.return_value = MagicMock(poll=MagicMock(return_value=None))
        h = _make_handler("/run")
        h._run_tests()
        data = _read_json_response(h)
        assert data["status"] == "started"
        mock_popen.assert_called_once()

    @patch("dashboard_server.subprocess.Popen")
    @patch("builtins.open", return_value=MagicMock())
    def test_sets_last_run_timestamp(self, mock_open, mock_popen):
        mock_popen.return_value = MagicMock(poll=MagicMock(return_value=None))
        h = _make_handler("/run")
        h._run_tests()
        assert ds._last_run is not None

    @patch("dashboard_server.subprocess.Popen")
    @patch("builtins.open", return_value=MagicMock())
    def test_duplicate_run_returns_already_running(self, mock_open, mock_popen):
        h = _make_handler("/run")
        running_proc = MagicMock()
        running_proc.poll.return_value = None   # still running
        ds._runner_proc = running_proc          # inject after reset
        h._run_tests()
        data = _read_json_response(h)
        assert data["status"] == "already_running"
        mock_popen.assert_not_called()

    @patch("dashboard_server.subprocess.Popen")
    @patch("builtins.open", return_value=MagicMock())
    def test_restarts_after_previous_run_completed(self, mock_open, mock_popen):
        finished_proc = MagicMock()
        finished_proc.poll.return_value = 0    # finished
        new_proc = MagicMock(poll=MagicMock(return_value=None))
        mock_popen.return_value = new_proc
        h = _make_handler("/run")
        ds._runner_proc = finished_proc
        h._run_tests()
        data = _read_json_response(h)
        assert data["status"] == "started"

    @patch("dashboard_server.subprocess.Popen")
    @patch("builtins.open", return_value=MagicMock())
    def test_popen_called_with_agent_runner_script(self, mock_open, mock_popen):
        mock_popen.return_value = MagicMock(poll=MagicMock(return_value=None))
        h = _make_handler("/run")
        h._run_tests()
        cmd = mock_popen.call_args[0][0]
        assert "agent_runner.py" in str(cmd)
        assert "--scenario" in cmd
        assert "all" in cmd


# ===========================================================================
# dashboard_server — /run_judge endpoint (BUG-2: runner race guard)
# ===========================================================================

class TestRunJudge:
    def test_no_report_file_returns_no_report_to_judge(self, tmp_path):
        missing = tmp_path / "no_report.json"   # does not exist
        h = _make_handler("/run_judge")
        with patch("dashboard_server._REPORT_PATH", missing):
            h._run_judge()
        data = _read_json_response(h)
        assert data["status"] == "no_report_to_judge"


    @patch("dashboard_server.subprocess.Popen")
    def test_runner_in_progress_blocks_judge(self, mock_popen):
        """BUG-2: judge must not start while runner is still running."""
        h = _make_handler("/run_judge")
        running_proc = MagicMock()
        running_proc.poll.return_value = None   # runner still running
        ds._runner_proc = running_proc
        h._run_judge()
        data = _read_json_response(h)
        assert data["status"] == "runner_in_progress", (
            "BUG-2: /run_judge should return runner_in_progress when runner is active"
        )
        mock_popen.assert_not_called()

    @patch("dashboard_server.subprocess.Popen")
    def test_judge_already_running_returns_already_running(self, mock_popen, tmp_path):
        report_file = tmp_path / "report.json"
        report_file.write_text('{}')
        h = _make_handler("/run_judge")
        running_judge = MagicMock()
        running_judge.poll.return_value = None
        ds._judge_proc = running_judge
        with patch("dashboard_server._REPORT_PATH", report_file):
            h._run_judge()
        data = _read_json_response(h)
        assert data["status"] == "already_running"
        mock_popen.assert_not_called()

    @patch("dashboard_server.subprocess.Popen")
    def test_starts_judge_when_runner_finished_and_report_exists(self, mock_popen, tmp_path):
        report_file = tmp_path / "report.json"
        report_file.write_text('{}')
        finished_runner = MagicMock()
        finished_runner.poll.return_value = 0   # runner finished
        mock_popen.return_value = MagicMock(poll=MagicMock(return_value=None))
        h = _make_handler("/run_judge")
        ds._runner_proc = finished_runner
        with patch("dashboard_server._REPORT_PATH", report_file):
            h._run_judge()
        data = _read_json_response(h)
        assert data["status"] == "started"
        mock_popen.assert_called_once()

    @patch("dashboard_server.subprocess.Popen")
    def test_starts_judge_when_no_runner_ever_ran(self, mock_popen, tmp_path):
        """If runner_proc is None (never ran), judge should start normally."""
        report_file = tmp_path / "report.json"
        report_file.write_text('{}')
        mock_popen.return_value = MagicMock(poll=MagicMock(return_value=None))
        h = _make_handler("/run_judge")
        # _runner_proc is None after _make_handler reset
        with patch("dashboard_server._REPORT_PATH", report_file):
            h._run_judge()
        data = _read_json_response(h)
        assert data["status"] == "started"


# ===========================================================================
# dashboard_server — /run_fix_loop endpoint (BUG-6: sets _last_run)
# ===========================================================================

class TestRunFixLoop:
    @patch("dashboard_server.subprocess.Popen")
    @patch("builtins.open", return_value=MagicMock())
    def test_starts_fix_loop_and_returns_started(self, mock_open, mock_popen):
        mock_popen.return_value = MagicMock(poll=MagicMock(return_value=None))
        h = _make_handler("/run_fix_loop")
        h._run_fix_loop()
        data = _read_json_response(h)
        assert data["status"] == "started"


    @patch("dashboard_server.subprocess.Popen")
    @patch("builtins.open", return_value=MagicMock())
    def test_sets_last_run_timestamp(self, mock_open, mock_popen):
        """BUG-6: _run_fix_loop must set _last_run like _run_tests does."""
        mock_popen.return_value = MagicMock(poll=MagicMock(return_value=None))
        h = _make_handler("/run_fix_loop")
        assert ds._last_run is None
        h._run_fix_loop()
        assert ds._last_run is not None, "BUG-6: _last_run not set by _run_fix_loop"

    @patch("dashboard_server.subprocess.Popen")
    @patch("builtins.open", return_value=MagicMock())
    def test_duplicate_fix_loop_returns_already_running(self, mock_open, mock_popen):
        h = _make_handler("/run_fix_loop")
        running_proc = MagicMock()
        running_proc.poll.return_value = None
        ds._fix_loop_proc = running_proc
        h._run_fix_loop()
        data = _read_json_response(h)
        assert data["status"] == "already_running"
        mock_popen.assert_not_called()

    @patch("dashboard_server.subprocess.Popen")
    @patch("builtins.open", return_value=MagicMock())
    def test_uses_shell_true(self, mock_open, mock_popen):
        mock_popen.return_value = MagicMock(poll=MagicMock(return_value=None))
        h = _make_handler("/run_fix_loop")
        h._run_fix_loop()
        _, kwargs = mock_popen.call_args
        assert kwargs.get("shell") is True


    @patch("dashboard_server.subprocess.Popen")
    @patch("builtins.open", return_value=MagicMock())
    def test_last_run_is_iso_format(self, mock_open, mock_popen):
        """Timestamp written to _last_run must be parseable as ISO 8601."""
        from datetime import datetime
        mock_popen.return_value = MagicMock(poll=MagicMock(return_value=None))
        h = _make_handler("/run_fix_loop")
        h._run_fix_loop()
        # Should parse without raising
        parsed = datetime.fromisoformat(ds._last_run)
        assert parsed is not None


# ===========================================================================
# dashboard_server — /report endpoint
# ===========================================================================

class TestGetReport:
    def test_returns_no_report_yet_when_file_absent(self, tmp_path):
        missing = tmp_path / "no_report.json"
        h = _make_handler("/report")
        with patch("dashboard_server._REPORT_PATH", missing):
            h._get_report()
        data = _read_json_response(h)
        assert data == {"status": "no_report_yet"}

    def test_returns_report_data_when_file_present(self, tmp_path):
        report_data = {"scenario": "all", "passed": True, "tests": []}
        report_file = tmp_path / "report.json"
        report_file.write_text(json.dumps(report_data))
        h = _make_handler("/report")
        with patch("dashboard_server._REPORT_PATH", report_file):
            h._get_report()
        data = _read_json_response(h)
        assert data["passed"] is True

    def test_returns_500_on_invalid_json(self, tmp_path):
        report_file = tmp_path / "report.json"
        report_file.write_text("{not valid json")
        h = _make_handler("/report")
        with patch("dashboard_server._REPORT_PATH", report_file):
            h._get_report()
        assert h._response_code == 500
        data = _read_json_response(h)
        assert "error" in data


# ===========================================================================
# dashboard_server — /judge endpoint
# ===========================================================================

class TestGetJudge:
    def test_returns_no_judge_yet_when_file_absent(self, tmp_path):
        missing = tmp_path / "no_judge.json"
        h = _make_handler("/judge")
        with patch("dashboard_server._JUDGE_PATH", missing):
            h._get_judge()
        data = _read_json_response(h)
        assert data == {"status": "no_judge_yet"}

    def test_returns_judge_data_when_file_present(self, tmp_path):
        judge_data = {"validity_issues": [], "coverage_gaps": ["gap A"]}
        judge_file = tmp_path / "judge.json"
        judge_file.write_text(json.dumps(judge_data))
        h = _make_handler("/judge")
        with patch("dashboard_server._JUDGE_PATH", judge_file):
            h._get_judge()
        data = _read_json_response(h)
        assert "coverage_gaps" in data

    def test_returns_500_on_corrupt_judge_file(self, tmp_path):
        judge_file = tmp_path / "judge.json"
        judge_file.write_text("{{broken")
        h = _make_handler("/judge")
        with patch("dashboard_server._JUDGE_PATH", judge_file):
            h._get_judge()
        assert h._response_code == 500
        data = _read_json_response(h)
        assert "error" in data


# ===========================================================================
# dashboard_server — do_GET routing
# ===========================================================================

class TestDoGetRouting:
    def test_unknown_path_returns_404(self):
        h = _make_handler("/nonexistent")
        h.do_GET()
        assert h._response_code == 404
        data = _read_json_response(h)
        assert "error" in data

    def test_status_path_routes_to_get_status(self):
        h = _make_handler("/status")
        h.do_GET()
        data = _read_json_response(h)
        assert "runner_running" in data  # confirms it hit _get_status

    def test_options_returns_204(self):
        h = _make_handler("/status", method="OPTIONS")
        h.do_OPTIONS()
        assert h._response_code == 204

    def test_report_with_query_string_still_routes(self, tmp_path):
        """do_GET strips query params before routing."""
        missing = tmp_path / "no_report.json"
        h = _make_handler("/report?foo=bar")
        with patch("dashboard_server._REPORT_PATH", missing):
            h.do_GET()
        data = _read_json_response(h)
        assert data == {"status": "no_report_yet"}


# ===========================================================================
# judge_runner — _strip_fences
# ===========================================================================

class TestStripFences:
    def test_removes_json_fence(self):
        raw = "```json\n{\"key\": \"value\"}\n```"
        assert jr._strip_fences(raw) == '{"key": "value"}'

    def test_removes_plain_fence(self):
        raw = "```\n{\"key\": \"value\"}\n```"
        assert jr._strip_fences(raw) == '{"key": "value"}'

    def test_leaves_unfenced_text_unchanged(self):
        raw = '{"key": "value"}'
        assert jr._strip_fences(raw) == raw

    def test_strips_outer_whitespace_before_matching(self):
        raw = "  \n```json\n{}\n```\n  "
        assert jr._strip_fences(raw) == "{}"

    def test_multiline_json_body_preserved(self):
        body = '{\n  "a": 1,\n  "b": 2\n}'
        raw = f"```json\n{body}\n```"
        result = jr._strip_fences(raw)
        assert result == body.strip()

    def test_empty_string_unchanged(self):
        assert jr._strip_fences("") == ""

    def test_fence_without_closing_not_stripped(self):
        raw = "```json\n{}"
        # no closing fence — regex should not match; text returned as-is (stripped)
        result = jr._strip_fences(raw)
        assert "{}" in result


# ===========================================================================
# judge_runner — run() with missing ANTHROPIC_API_KEY
# ===========================================================================

class TestJudgeRunMissingApiKey:
    def test_writes_error_output_when_api_key_not_set(self, tmp_path):
        report_file = tmp_path / "report.json"
        report_file.write_text('{"scenario": "all", "passed": true}')
        with patch.dict("os.environ", {}, clear=True):
            with patch("judge_runner._write_output") as mock_write:
                with pytest.raises(SystemExit) as exc_info:
                    jr.run(report_file)
        assert exc_info.value.code == 1
        written = mock_write.call_args[0][0]
        assert "ANTHROPIC_API_KEY" in written.get("error", "")

    def test_exits_with_code_1_when_api_key_missing(self, tmp_path):
        report_file = tmp_path / "report.json"
        report_file.write_text('{"scenario": "all", "passed": true}')
        with patch.dict("os.environ", {}, clear=True):
            with patch("judge_runner._write_output"):
                with pytest.raises(SystemExit) as exc_info:
                    jr.run(report_file)
        assert exc_info.value.code == 1


# ===========================================================================
# judge_runner — run() with missing report file
# ===========================================================================

class TestJudgeRunMissingReport:
    def test_writes_error_when_report_not_found(self, tmp_path):
        missing = tmp_path / "nonexistent_report.json"
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "fake-key"}):
            with patch("judge_runner._write_output") as mock_write:
                with pytest.raises(SystemExit) as exc_info:
                    jr.run(missing)
        assert exc_info.value.code == 1
        written = mock_write.call_args[0][0]
        assert "not found" in written.get("error", "").lower() or str(missing) in written.get("error", "")

    def test_exits_with_code_1_for_missing_report(self, tmp_path):
        missing = tmp_path / "no_such_file.json"
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "fake-key"}):
            with patch("judge_runner._write_output"):
                with pytest.raises(SystemExit) as exc_info:
                    jr.run(missing)
        assert exc_info.value.code == 1


# ===========================================================================
# judge_runner — run() with corrupt JSON report
# ===========================================================================

class TestJudgeRunInvalidReport:
    def test_writes_error_for_corrupt_report(self, tmp_path):
        bad_report = tmp_path / "bad.json"
        bad_report.write_text("{not json}")
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "fake-key"}):
            with patch("judge_runner._write_output") as mock_write:
                with pytest.raises(SystemExit) as exc_info:
                    jr.run(bad_report)
        assert exc_info.value.code == 1
        written = mock_write.call_args[0][0]
        assert "error" in written


# ===========================================================================
# judge_runner — run() when judge prompt file is missing
# ===========================================================================

class TestJudgeRunMissingPrompt:
    def test_writes_error_when_prompt_not_found(self, tmp_path):
        report = tmp_path / "report.json"
        report.write_text('{"scenario": "all", "passed": true}')
        missing_prompt = tmp_path / "judge_prompt_missing.md"   # does not exist
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "fake-key"}):
            with patch("judge_runner._JUDGE_PROMPT_PATH", missing_prompt):
                with patch("judge_runner._write_output") as mock_write:
                    with pytest.raises(SystemExit) as exc_info:
                        jr.run(report)
        assert exc_info.value.code == 1
        written = mock_write.call_args[0][0]
        assert "error" in written


# ===========================================================================
# judge_runner — _write_output
# ===========================================================================

class TestWriteOutput:
    def test_writes_json_file(self, tmp_path):
        output_file = tmp_path / "output.json"
        with patch("judge_runner._JUDGE_OUTPUT", output_file):
            jr._write_output({"status": "ok", "score": 10})
        written = json.loads(output_file.read_text())
        assert written["score"] == 10

    def test_output_is_pretty_printed(self, tmp_path):
        output_file = tmp_path / "output.json"
        with patch("judge_runner._JUDGE_OUTPUT", output_file):
            jr._write_output({"a": 1})
        raw = output_file.read_text()
        assert "\n" in raw  # indent=2 produces newlines
