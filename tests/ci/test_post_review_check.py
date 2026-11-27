"""Offline tests for scripts/post_review_check.py (KAN-260). No network: requests are built, never sent."""

from __future__ import annotations

import base64
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


prc = _load("post_review_check")
gate = _load("pr_gate")

SHA = "a" * 40
needs_openssl = pytest.mark.skipif(
    shutil.which("openssl") is None or sys.platform == "win32", reason="needs the openssl CLI (not on Windows)"
)


def _unb64(part: str) -> bytes:
    return base64.urlsafe_b64decode(part + "=" * (-len(part) % 4))


def test_check_names_match_the_gate() -> None:
    assert prc.VERIFIER_CHECK == gate.VERIFIER_CHECK
    assert prc.TOP_REVIEW_CHECK == gate.TOP_REVIEW_CHECK
    assert prc.KIND_TO_NAME == {"verifier": gate.VERIFIER_CHECK, "top-review": gate.TOP_REVIEW_CHECK}


def test_b64url_has_no_padding_or_plus_slash() -> None:
    out = prc.b64url(b"\xfb\xff\xfe\x01")
    assert "=" not in out and "+" not in out and "/" not in out
    assert _unb64(out) == b"\xfb\xff\xfe\x01"


def test_jwt_header_and_payload() -> None:
    assert prc.jwt_header() == {"alg": "RS256", "typ": "JWT"}
    assert prc.jwt_payload("123", 1000) == {"iat": 940, "exp": 1540, "iss": "123"}
    head, payload = prc.signing_input("123", 1000).split(".")
    assert json.loads(_unb64(head)) == {"alg": "RS256", "typ": "JWT"}
    assert json.loads(_unb64(payload)) == {"iat": 940, "exp": 1540, "iss": "123"}


def test_token_request_asks_for_checks_write_on_one_repo() -> None:
    assert prc.token_request_body("o/Operon") == {"permissions": {"checks": "write"}, "repositories": ["Operon"]}


def test_check_run_body_maps_kind_to_name() -> None:
    body = prc.check_run_body("verifier", SHA.upper(), "success", "ok")
    assert body == {
        "name": "Operon verifier",
        "head_sha": SHA,
        "status": "completed",
        "conclusion": "success",
        "output": {"title": "Operon verifier: success", "summary": "ok"},
    }
    assert prc.check_run_body("top-review", SHA, "failure", "no")["name"] == "Operon top review"


def test_summary_is_truncated() -> None:
    body = prc.check_run_body("verifier", SHA, "success", "x" * 70000)
    assert len(body["output"]["summary"]) == 65535


@pytest.mark.parametrize("sha", ["", "abc", "g" * 40, "a" * 39, "a" * 41, SHA + "\n", "../" + "a" * 37])
def test_bad_sha_is_refused(sha: str) -> None:
    with pytest.raises(ValueError):
        prc.validate_sha(sha)


@pytest.mark.parametrize("repo", ["", "noslash", "a/b/c", "a b/c", "a/b c", "a/", "/b", "a/b\n", "../..", "./.", "../r", "o/..", "o/.", "\u00e9/r", "o/r\u00e9", "\u0663/r"])
def test_bad_repo_is_refused(repo: str) -> None:
    with pytest.raises(ValueError):
        prc.validate_repo(repo)


def test_bad_kind_and_conclusion_are_refused() -> None:
    with pytest.raises(ValueError):
        prc.check_run_body("other", SHA, "success", "")
    with pytest.raises(ValueError):
        prc.check_run_body("verifier", SHA, "neutral", "")


def test_request_builder() -> None:
    req = prc.build_request("POST", "/repos/o/r/check-runs", "TOKEN", {"a": 1})
    assert req.full_url == "https://api.github.com/repos/o/r/check-runs"
    assert req.get_method() == "POST"
    assert json.loads(req.data) == {"a": 1}
    assert req.get_header("Authorization") == "Bearer TOKEN"
    get = prc.build_request("GET", "/repos/o/r/installation", "JWT")
    assert get.data is None and get.get_method() == "GET"


def test_read_key_prefers_env_text_then_file(tmp_path: Path) -> None:
    f = tmp_path / "k.pem"
    f.write_text("FILEKEY")
    assert prc.read_key({"OPERON_VERIFIER_APP_KEY": "TEXT", "OPERON_VERIFIER_APP_KEY_FILE": str(f)}) == "TEXT"
    assert prc.read_key({"OPERON_VERIFIER_APP_KEY_FILE": str(f)}) == "FILEKEY"
    with pytest.raises(ValueError):
        prc.read_key({})


def test_read_key_turns_literal_backslash_n_into_newlines() -> None:
    one_line = "-----BEGIN TEST DATA-----\\nAAAA\\n-----END TEST DATA-----\\n"
    assert prc.read_key({"OPERON_VERIFIER_APP_KEY": one_line}) == "-----BEGIN TEST DATA-----\nAAAA\n-----END TEST DATA-----\n"
    real = "-----BEGIN TEST DATA-----\nAAAA\n-----END TEST DATA-----\n"
    assert prc.read_key({"OPERON_VERIFIER_APP_KEY": real}) == real


def test_post_check_sends_the_three_calls_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[Any] = []
    answers = iter([{"id": 77}, {"token": "INSTTOKEN"}, {"html_url": "https://x/run/1"}])
    monkeypatch.setattr(prc, "build_jwt", lambda app_id, pem, now=None: "THEJWT")
    monkeypatch.setattr(prc, "_send", lambda req: (sent.append(req), next(answers))[1])
    url = prc.post_check("o/r", SHA, "verifier", "success", "fine", "9", "PEM")
    assert url == "https://x/run/1"
    g, t, c = sent
    assert (g.get_method(), g.selector, g.get_header("Authorization")) == ("GET", "/repos/o/r/installation", "Bearer THEJWT")
    assert (t.get_method(), t.selector) == ("POST", "/app/installations/77/access_tokens")
    assert json.loads(t.data) == {"permissions": {"checks": "write"}, "repositories": ["r"]}
    assert t.get_header("Authorization") == "Bearer THEJWT"
    assert c.selector == "/repos/o/r/check-runs" and c.get_header("Authorization") == "Bearer INSTTOKEN"
    assert json.loads(c.data)["name"] == "Operon verifier"


def test_bad_input_makes_no_network_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prc, "_send", lambda req: pytest.fail("network used"))
    with pytest.raises(ValueError):
        prc.post_check("o/r", "short", "verifier", "success", "", "9", "PEM")


@needs_openssl
def test_signature_verifies_and_the_key_is_not_left_behind(tmp_path: Path) -> None:
    key = tmp_path / "key.pem"
    pub = tmp_path / "pub.pem"
    subprocess.run(["openssl", "genrsa", "-out", str(key), "2048"], check=True, capture_output=True)
    subprocess.run(["openssl", "rsa", "-in", str(key), "-pubout", "-out", str(pub)], check=True, capture_output=True)
    jwt = prc.build_jwt("42", key.read_text(), now=2000)
    head, payload, sig = jwt.split(".")
    assert json.loads(_unb64(payload)) == {"iat": 1940, "exp": 2540, "iss": "42"}
    sigfile = tmp_path / "sig"
    sigfile.write_bytes(_unb64(sig))
    res = subprocess.run(
        ["openssl", "dgst", "-sha256", "-verify", str(pub), "-signature", str(sigfile)],
        input=f"{head}.{payload}".encode(),
        capture_output=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr


@needs_openssl
def test_bad_key_fails_without_printing_it(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    secret = "NOT-A-REAL-KEY-SECRET-12345"
    summary = tmp_path / "s.md"
    summary.write_text("ok")
    monkeypatch.setenv("OPERON_VERIFIER_APP_ID", "1")
    monkeypatch.setenv("OPERON_VERIFIER_APP_KEY", secret)
    monkeypatch.setattr(prc, "_send", lambda req: pytest.fail("network used"))
    argv = ["--repo", "o/r", "--sha", SHA, "--kind", "verifier", "--conclusion", "success", "--summary-file", str(summary)]
    assert prc.main(argv) == 1
    out = capsys.readouterr()
    assert secret not in out.out + out.err
    assert "error:" in out.err


def test_main_reports_a_missing_app_id(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    summary = tmp_path / "s.md"
    summary.write_text("ok")
    monkeypatch.delenv("OPERON_VERIFIER_APP_ID", raising=False)
    argv = ["--repo", "o/r", "--sha", SHA, "--kind", "verifier", "--conclusion", "success", "--summary-file", str(summary)]
    assert prc.main(argv) == 1
    assert "OPERON_VERIFIER_APP_ID" in capsys.readouterr().err
