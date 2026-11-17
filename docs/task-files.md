# Task files (KAN-86)

Keep a suite of Operon tasks in your repo and run them in CI like tests.

```
operon run --tasks tasks.yaml [--junit report.xml] [--only NAME] [--fail-fast]
```

Tasks run one after another through the [Python SDK](sdk.md), one `Agent` per task. A compact table
(name, status, steps, cost, seconds, PASS/FAIL, reason) is printed, and the exit code is non-zero if
any task fails (2 if the file itself is invalid).

## Format

```yaml
defaults:                 # optional; every field can be overridden per task
  desktop: false
  env: null               # e.g. windows-sandbox (needs desktop; see `operon envs`)
  mode: risky             # ask | risky | auto (default risky, see below)
  max_steps: 25
  url: https://example.com
  approve_all: false      # see "Approvals" below

tasks:
  - name: search-weather            # unique
    task: "Find the weather in Paris"
    url: https://example.com        # overrides the default
    expect:                         # optional checks after the run
      status: succeeded             # default succeeded
      answer_contains: ["Paris"]    # case-insensitive substrings of the final answer (or, if the run gave
                                    # no answer, of the model's last message)
      max_steps: 20                 # fail if more steps were used
  - name: save-report
    task: "Save the report to C:/tmp/out.txt"
    desktop: true
    expect:
      file_exists: C:/tmp/out.txt
      file_contains: {path: C:/tmp/out.txt, text: "Total"}
```

Unknown keys are rejected, with the path of the bad field (`tasks[2].mode: ...`).

`file_exists` and `file_contains` read this host's files, so they work for the `live` environment
only. With any other `env` (a sandbox or VM) the task fails with "unsupported for env X" rather than
checking the wrong machine.

## Mode

Task files default to `mode: risky`: only risky actions pause. Under `ask`, every action that changes
something pauses for approval, and with nobody there to approve, an unattended suite fails at its
first click. You can still set `mode: ask` (in `defaults` or per task) to see which tasks would stop.

## Approvals

There is no human in a task-file run. If a step needs approval (a trust gate, or the model asks to
confirm), the task **fails** with the reason. Setting `approve_all: true` grants every approval
automatically, except a hard policy block (the trust gate's "Blocked by policy" deny): `approve_all`
never overrides one, and that task fails with `blocked by policy: <reason>`. **`approve_all` lets the agent make purchases, send messages and delete things without anyone
looking. Use it only in a disposable environment, on tasks you wrote.**

## JUnit report

`--junit report.xml` writes one `testsuite` named `operon` with one `testcase` per task
(`classname` is the file's name without extension, `time` is the seconds the task took). A failing
task has a `failure` whose message is the reason and whose body has the `run_id` and the artifacts
directory. Tasks excluded by `--only` are left out.

## GitHub Actions

This needs a real model key (`GEMINI_API_KEY`) and **costs money** per run, so keep it to a manual or
nightly trigger. Use a self-hosted runner with a display and the browsers installed.

```yaml
name: operon-tasks
on:
  workflow_dispatch:
  schedule: [{cron: "0 5 * * *"}]
jobs:
  tasks:
    runs-on: self-hosted
    steps:
      - uses: actions/checkout@v4
      - run: pip install operon-agent && operon setup --check
      - run: operon run --tasks tasks.yaml --junit report.xml
        env:
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
      - uses: actions/upload-artifact@v4
        if: always()
        with: {name: operon-report, path: report.xml}
```
