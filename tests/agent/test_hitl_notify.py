"""Desktop notifications must never splice LLM-written text into a script.

The HITL message is generated from on-screen element names, so it is
attacker-influenced; each platform helper must hand it to the subprocess as
data (env var / argv), never as script source.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

from operon.agent import hitl

PAYLOAD = 'Done" $(Remove-Item C:\\x) `whoami` & do shell script "id'


def _popen_call(fn) -> tuple[list[str], dict]:
    with patch.object(hitl.subprocess, "Popen") as popen:
        fn("Operon needs you", PAYLOAD)
    assert popen.call_count == 1
    args, kwargs = popen.call_args
    return list(args[0]), kwargs


def test_powershell_fallback_passes_message_via_env_not_script() -> None:
    # win11toast unavailable → the PowerShell balloon-tip fallback runs.
    with patch.dict(sys.modules, {"win11toast": None}):
        argv, kwargs = _popen_call(hitl._notify_windows)
    script = argv[argv.index("-Command") + 1]
    assert PAYLOAD not in script
    assert "$(" not in script
    assert "$env:OPERON_NOTIFY_MESSAGE" in script
    assert kwargs["env"]["OPERON_NOTIFY_MESSAGE"] == PAYLOAD
    assert kwargs["env"]["OPERON_NOTIFY_TITLE"] == "Operon needs you"


def test_macos_passes_message_as_argv_not_applescript() -> None:
    argv, _ = _popen_call(hitl._notify_macos)
    script_parts = [argv[i + 1] for i, a in enumerate(argv) if a == "-e"]
    assert all(PAYLOAD not in part for part in script_parts)
    assert argv[-2:] == ["Operon needs you", PAYLOAD]


def test_notify_send_ends_options_before_title_and_message() -> None:
    argv, _ = _popen_call(hitl._notify_linux)
    assert argv == ["notify-send", "--", "Operon needs you", PAYLOAD]
