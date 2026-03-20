# UI Navigator MVP

This repository contains the initial scaffold for a browser-only, reliability-first computer-use agent.

Scope is intentionally narrow:

- Gmail draft creation only
- Stop before send
- Local-only persistence
- No dashboard
- No desktop mode
- No extension runtime
- No cloud storage

The control loop is the architectural center:

`capture -> perceive -> update state -> choose action -> execute -> verify -> recover`

## Runtime

Use Python `3.11` for this repo.

Confirmed local interpreter path:

- [`python`](python)

Confirmed local virtual environment:

- [`.venv311sys`](./.venv311sys)

## Windows Playwright Setup

Use repo-local temp and browser cache paths when running Playwright:

```powershell
$env:TEMP=".\.tmp"
$env:TMP=".\.tmp"
$env:PLAYWRIGHT_BROWSERS_PATH=".\.ms-playwright"
```

Install Chromium for the repo-local Playwright environment:

```powershell
.\.venv311sys\Scripts\python.exe -m playwright install chromium
```

Quick browser launch check:

```powershell
.\.venv311sys\Scripts\python.exe -c "from playwright.sync_api import sync_playwright; p=sync_playwright().start(); b=p.chromium.launch(); page=b.new_page(); page.goto('https://example.com'); print(page.title()); b.close(); p.stop()"
```

## Known Windows Host Limitation

This machine currently blocks the Windows named-pipe and subprocess behavior Playwright depends on.

Observed failures:

- `PermissionError: [WinError 5] Access is denied`
- `spawn EPERM`

The repo is prepared for Playwright, but browser launch will still fail until the host allows:

- Python-created named pipes
- Python `asyncio` subprocess transport
- Node child-process spawn used by Playwright browser download

Likely places to fix:

- Windows Defender Controlled Folder Access
- enterprise endpoint security / EDR
- AppLocker / WDAC style process restrictions

Current status: the MVP architecture, schemas, logging, verifier, recovery, and Playwright-backed browser boundary are implemented, but successful local browser execution still depends on removing the host restriction above.
