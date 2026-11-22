# Security Policy

## Reporting a vulnerability

Please **do not** open a public issue for security problems.

Report privately via GitHub's [private vulnerability
reporting](https://github.com/vickey-kapoor/Operon/security/advisories/new).

Please include: what the issue is, how to reproduce it, and what an attacker
could achieve. A working proof of concept helps but is not required.

This is a personal open-source project, not a funded product — expect a first
response within about a week. There is no bug bounty.

## Threat model

Operon is a computer-use agent. By design it launches browsers, moves the mouse,
types on the keyboard, and opens native applications on the host it runs on.
**The agent's capabilities are the operator's capabilities.** Treat an Operon
instance as equivalent to giving someone an interactive session on that machine.

The security boundary is the host, not the agent. Operon does not sandbox itself.

### In scope

- Authentication or authorization bypass on the HTTP or WebSocket surface
- Command, path, or argument injection reachable from a task instruction
- Trust-gate bypass — actions executing that the deny list, confirm list, or
  domain allowlist should have blocked
- Escapes from the configured artifact roots (arbitrary file read or write)
- Secret leakage into run artifacts, logs, or the observer API

### Out of scope

- **Running with `OPERON_ALLOW_UNAUTHENTICATED=true`.** This disables
  authentication and is documented as such. Exposing an unauthenticated instance
  to a network is an operator error, not a vulnerability.
- The agent taking a wrong or destructive action from an ambiguous instruction.
  That is a correctness and alignment problem — file it as a normal issue.
- Prompt injection from web page content steering a browser run. This is a real
  and unsolved risk that the trust gate only partially mitigates; concrete
  bypasses of the gate itself *are* in scope, but "a malicious page influenced
  the model" is a known limitation.
- Vulnerabilities in Chromium, Playwright, or upstream model providers. Report
  those upstream.

## Operating safely

- **Keep authentication on.** With `API_KEYS` unset the server generates a key in
  `.var/api-key`; set your own `API_KEYS` before the port is reachable by anything
  but localhost, and never set `OPERON_ALLOW_UNAUTHENTICATED` there.
- **Set `CORS_ORIGINS` explicitly.** Empty means no origins are allowed, which is
  the safe default — do not widen it to `*`.
- **Keep the trust gate on** (the default; `OPERON_TRUST_GATE=off` disables it),
  and for anything unattended add `OPERON_TRUST_ALLOW_DOMAINS` to fail closed on navigation.
  `OPERON_TRUST_CATEGORIES=on` (experimental) also asks before buttons that name a payment,
  send, delete, account or install action (see `docs/configuration.md`).
- **Use the secrets vault for passwords** (`OPERON_SECRETS=on`, `docs/secrets.md`) instead of
  putting them in a task: the model types `{{secret:NAME}}` and never sees the value. Screenshots
  still show a plain (unmasked) field, and on Linux/macOS the vault is not encrypted at rest.
- **Prefer a VM or dedicated machine** for desktop runs. There is no undo.
- **Run artifacts contain full screenshots** of whatever was on screen, written
  under `.var/runs/<run_id>/`. Treat them as sensitive and scrub before sharing.

## Automated checks

The `Security` workflow (`.github/workflows/security.yml`) runs on every pull request, on every push
to `main`, and weekly. Each job is a separate check:

- **Secrets (gitleaks):** scans the commits of a PR, the whole history reachable from `HEAD` on a push
  to `main`, and every branch and tag on a weekly or manual run, for committed secrets. Fails on any finding not
  allowlisted in `.gitleaks.toml`. Git mode is the authoritative check; run `gitleaks git --redact .`
  locally. Avoid `gitleaks dir`: it skips allowlisted paths entirely (so it misses real secrets there)
  and also scans `.claude/worktrees` and `__pycache__` copies.
- **Python dependencies (pip-audit):** audits the versions pinned in `uv.lock` against known
  advisories. Fails when any pinned package has one.
- **Dashboard dependencies (npm audit):** audits `dashboard/package-lock.json`. Fails on a high or
  critical advisory.
- **Python code (bandit):** static scan of `src`. Fails on a Medium or higher finding that is not in
  `.bandit-baseline.json`.
- **Workflows (zizmor):** audits `.github/workflows` for injection, token and trigger mistakes.
  Fails on a medium or higher finding.
- **CodeQL (python, javascript-typescript, actions):** code scanning; results appear under the
  repository's Security tab.

Updating the checks:

- **bandit baseline.** It holds the findings that were reviewed when the check was added. Bandit
  matches a finding on its file, test id, issue message, severity, confidence and CWE (not the line
  number), so on its own it would let a new finding of an already-baselined kind in the same file
  through. The job therefore also fails when any (file, test id) pair has more findings than the
  baseline lists. The baseline must be generated on Linux or WSL (forward-slash paths). Regenerate it
  with the command below, and explain every new entry in the PR:

  ```bash
  uvx --python 3.14 bandit==1.9.4 -r src -ll -q -f json \
    | python3 -c "import json,sys; json.dump({'results': json.load(sys.stdin)['results']}, sys.stdout, indent=2)" \
    > .bandit-baseline.json
  ```

- **gitleaks allowlist.** Add an `[[allowlists]]` entry to `.gitleaks.toml` with the exact path, the
  exact value (an anchored regex) and the reason. Do not allowlist a whole directory or rule.
- **pip-audit can turn red without a code change** when a new advisory lands. Fix it with
  `uv lock --upgrade-package <name>`. If no fixed version exists, add a commented
  `--ignore-vuln <ID>` to the job.
