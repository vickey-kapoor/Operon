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

- **Running with `API_KEYS` unset.** This disables authentication and is
  documented as such. Exposing an unauthenticated instance to a network is an
  operator error, not a vulnerability.
- The agent taking a wrong or destructive action from an ambiguous instruction.
  That is a correctness and alignment problem — file it as a normal issue.
- Prompt injection from web page content steering a browser run. This is a real
  and unsolved risk that the trust gate only partially mitigates; concrete
  bypasses of the gate itself *are* in scope, but "a malicious page influenced
  the model" is a known limitation.
- Vulnerabilities in Chromium, Playwright, or upstream model providers. Report
  those upstream.

## Operating safely

- **Set `API_KEYS`** before the port is reachable by anything but localhost.
- **Set `CORS_ORIGINS` explicitly.** Empty means no origins are allowed, which is
  the safe default — do not widen it to `*`.
- **Turn the trust gate on** for anything unattended: `OPERON_TRUST_GATE=on`,
  plus `OPERON_TRUST_ALLOW_DOMAINS` to fail closed on navigation.
- **Prefer a VM or dedicated machine** for desktop runs. There is no undo.
- **Run artifacts contain full screenshots** of whatever was on screen, written
  under `.var/runs/<run_id>/`. Treat them as sensitive and scrub before sharing.
