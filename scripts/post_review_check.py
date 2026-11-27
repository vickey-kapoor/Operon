"""Post the verifier's verdict as a check run from the operon-verifier GitHub App (KAN-260).

Run by the verifier-only cloud session, the only place that holds the app's key. The merge gate
(``scripts/pr_gate.py``) trusts only check runs from that app, so the coding account cannot post a pass itself.
Stdlib only; the RS256 signature comes from the ``openssl`` CLI. Nothing secret is printed.

    python scripts/post_review_check.py --repo owner/name --sha <40-hex> --kind verifier|top-review \
        --conclusion success|failure --summary-file FILE

Env: ``OPERON_VERIFIER_APP_ID`` and ``OPERON_VERIFIER_APP_KEY`` (PEM text) or ``OPERON_VERIFIER_APP_KEY_FILE``.
See docs/verifier-app.md.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

API = "https://api.github.com"
VERIFIER_CHECK = "Operon verifier"  # must match scripts/pr_gate.py (a test checks)
TOP_REVIEW_CHECK = "Operon top review"
KIND_TO_NAME = {"verifier": VERIFIER_CHECK, "top-review": TOP_REVIEW_CHECK}
CONCLUSIONS = ("success", "failure")
MAX_SUMMARY = 65535
_REPO_RE = re.compile(r"(?!\.+/)[\w.-]+/(?!\.+$)[\w.-]+", re.ASCII)
_SHA_RE = re.compile(r"[0-9a-fA-F]{40}")


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def jwt_header() -> dict[str, str]:
    return {"alg": "RS256", "typ": "JWT"}


def jwt_payload(app_id: str, now: int) -> dict[str, Any]:
    return {"iat": now - 60, "exp": now + 540, "iss": app_id}


def signing_input(app_id: str, now: int) -> str:
    parts = (jwt_header(), jwt_payload(app_id, now))
    return ".".join(b64url(json.dumps(p, separators=(",", ":")).encode()) for p in parts)


def sign_rs256(message: str, pem: str) -> str:
    """Sign ``message`` with the PEM key through ``openssl``; the key sits in an owner-only temp file, deleted after."""
    fd, path = tempfile.mkstemp(prefix="appkey-")
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(pem)
        proc = subprocess.run(  # noqa: S603  # nosec B603 B607: fixed argv
            ["openssl", "dgst", "-sha256", "-sign", path], input=message.encode(), capture_output=True, check=False
        )
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError("openssl could not sign the token (check the app key)")
    return b64url(proc.stdout)


def build_jwt(app_id: str, pem: str, now: int | None = None) -> str:
    message = signing_input(app_id, int(time.time()) if now is None else now)
    return message + "." + sign_rs256(message, pem)


def validate_repo(repo: str) -> str:
    if not _REPO_RE.fullmatch(repo):
        raise ValueError("--repo must look like owner/name")
    return repo


def validate_sha(sha: str) -> str:
    if not _SHA_RE.fullmatch(sha):
        raise ValueError("--sha must be 40 hex characters")
    return sha.lower()


def token_request_body(repo: str) -> dict[str, Any]:
    """Ask for the least: checks write, on this one repository."""
    return {"permissions": {"checks": "write"}, "repositories": [validate_repo(repo).split("/", 1)[1]]}


def check_run_body(kind: str, sha: str, conclusion: str, summary: str) -> dict[str, Any]:
    if kind not in KIND_TO_NAME:
        raise ValueError("--kind must be verifier or top-review")
    if conclusion not in CONCLUSIONS:
        raise ValueError("--conclusion must be success or failure")
    name = KIND_TO_NAME[kind]
    return {
        "name": name,
        "head_sha": validate_sha(sha),
        "status": "completed",
        "conclusion": conclusion,
        "output": {"title": f"{name}: {conclusion}", "summary": summary[:MAX_SUMMARY]},
    }


def build_request(method: str, path: str, token: str, body: dict[str, Any] | None = None) -> urllib.request.Request:
    req = urllib.request.Request(  # noqa: S310  # nosec B310: https base
        API + path,
        data=None if body is None else json.dumps(body).encode(),
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "operon-review-check",
            "Content-Type": "application/json",
        },
    )
    return req


def _send(req: urllib.request.Request) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310  # nosec B310: https base
            data = json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"GitHub answered {e.code} for {req.get_method()} {req.selector}") from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"could not reach GitHub: {e.reason}") from None
    if not isinstance(data, dict):
        raise RuntimeError("unexpected answer from GitHub")
    return data


def post_check(repo: str, sha: str, kind: str, conclusion: str, summary: str, app_id: str, pem: str) -> str:
    """Post the check run and return its html url."""
    validate_repo(repo)
    run_body = check_run_body(kind, sha, conclusion, summary)  # validates before any network call
    jwt = build_jwt(app_id, pem)
    inst = _send(build_request("GET", f"/repos/{repo}/installation", jwt))
    inst_id = inst.get("id")
    if not isinstance(inst_id, int):
        raise RuntimeError("the app is not installed on this repository")
    tok = _send(build_request("POST", f"/app/installations/{inst_id}/access_tokens", jwt, token_request_body(repo)))
    token = tok.get("token")
    if not isinstance(token, str):
        raise RuntimeError("GitHub gave no installation token")
    run = _send(build_request("POST", f"/repos/{repo}/check-runs", token, run_body))
    return str(run.get("html_url") or run.get("id") or "posted")


def read_key(env: dict[str, str]) -> str:
    key = env.get("OPERON_VERIFIER_APP_KEY")
    if key:
        # Some secret stores keep a PEM on one line with literal \n escapes.
        return key.replace("\\n", "\n") if "\n" not in key else key
    path = env.get("OPERON_VERIFIER_APP_KEY_FILE")
    if path:
        return Path(path).read_text()
    raise ValueError("set OPERON_VERIFIER_APP_KEY or OPERON_VERIFIER_APP_KEY_FILE")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--sha", required=True)
    ap.add_argument("--kind", required=True, choices=sorted(KIND_TO_NAME))
    ap.add_argument("--conclusion", required=True, choices=CONCLUSIONS)
    ap.add_argument("--summary-file", required=True)
    args = ap.parse_args(argv)
    try:
        app_id = os.environ.get("OPERON_VERIFIER_APP_ID", "")
        if not app_id:
            raise ValueError("set OPERON_VERIFIER_APP_ID")
        summary = Path(args.summary_file).read_text()
        url = post_check(args.repo, args.sha, args.kind, args.conclusion, summary, app_id, read_key(dict(os.environ)))
    except (ValueError, RuntimeError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"posted {KIND_TO_NAME[args.kind]}: {args.conclusion} ({url})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
