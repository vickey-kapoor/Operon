# OSWorld benchmark runner (KAN-169)

`Actions > Run OSWorld test tasks (own VM) > Run workflow` runs Operon on a fixed 25-task subset of
OSWorld-Verified (`evals/osworld_subset.json`) and scores it with OSWorld's own checkers.
It is manual only: it spends Gemini credits (about $0.06 a task; the 5-task pilot cost
$0.31), so you pick how many tasks and the cost cap each time.

```
start-vm   az vm start operon-linux            (ubuntu-latest, Azure OIDC sign-in)
bench      self-hosted runner on operon-linux  scripts/osworld-vm/setup.sh, then python -m evals.osworld
stop-vm    az vm deallocate operon-linux       (always, even when the run fails), then delete the run's runner if it is still registered
```

The `bench` job runs on a one-job (just-in-time) runner (KAN-195). `start-vm` mints it through the GitHub API
under a label unique to the run (`operon-osworld-<run id>-<attempt>-<random>`), hands its config to the VM with
`az vm run-command`, and the runner takes that one job and is gone. The label carries a random suffix, so cells of one batch
do not clash and it cannot be guessed from the public run id. No lasting runner stays registered, so another
workflow, such as an approved fork PR's, has no standing runner to target on the VM. The logic is `scripts/bench-runners/jit-runner.sh`
(`start` and `remove`) and `scripts/bench-runners/vm-linux.sh` (runs on the VM).

Inputs: `limit` (first N tasks; 5 = the pilot, 25 or 0 = all), `only` (one task id prefix),
`max_cost_usd` (stop starting tasks past this; use 8 for all 25), `max_steps`. The report
(`osworld-*.json`, per-task pass/fail, cost, seconds), the log and each task's Operon run folder
(screenshots of the task VM, model I/O, traces) are uploaded as the `osworld-report` artifact, so a failed task can be diagnosed from it.
Each task's Operon output (the exit code or a timeout note, then stdout and stderr) is saved in `osworld-logs/<task id>.log` and uploaded too, so a task that fails at startup still shows its error. The log's tail is on the run's summary page.

A task that raises (VM reset, `wait_for_desktop`, Operon, `evaluate`, or a missing task file) does not stop the run: it is
recorded with score 0 and `status: "error"`, a `::warning::` is printed, and the next task starts. Ctrl+C
still stops the run, and the report is saved. Each result has `status` (`passed`, `failed`, `timed_out` or `error`),
`error` (the exception, or null), `run_status` and `stop_reason` (read from the Operon run's `state.json`, null if
it has none), plus `run_id`, `cost_usd`, `seconds`, `note` and `log`. An error after Operon ran keeps its cost and
run id, so it still counts toward the cost cap. The summary adds `errors` and `timed_out` counts.

Timeout: the `bench` job's `timeout-minutes` is sized to the tasks chosen, `30 + 35 * tasks` (30 min for
`setup.sh`; 25 min task timeout plus 10 min for VM reset and boot, desktop wait, settle, evaluate and video
render per task): 905 for 25 tasks, 205 for 5, 65 for 1. The `start-vm` job computes it with
`python -m evals.osworld --job-minutes [--limit N | --only ID]`, which is free and needs no VM.

Scoring: OSWorld's `DesktopEnv.evaluate()` decides pass or fail, never Operon's own done-check.
Runs use the thin harness with `OPERON_AUTO_CONFIRM=on` (Operon answers the model's own
"require confirmation" pause, because nobody can click Resume), `OPERON_TOOLS=off` and the
OSWorld app catalog. Report scores as run with those.

## One-time setup

The VM `operon-linux` (Standard_D2s_v4, `operon-bench` resource group) needs KVM, Docker and a
128 GB disk (the task image is about 11 GB and is downloaded once into `~/osworld-data`).

1. On the VM, install Docker and let the login user use it and KVM, then log in again:

   ```bash
   sudo apt-get install -y docker.io python3-venv python3-pip git
   sudo usermod -aG docker,kvm "$USER"
   ```

2. Unpack the GitHub Actions runner binaries on the VM, as the login user, into `~/actions-runner`
   (Settings > Actions > Runners > New self-hosted runner > Linux, x64 shows the download and unpack
   commands). Do not run `config.sh` or `svc.sh`: the workflow registers a one-job runner for each run.
   A runner registered the old way is retired on the next run (its service is removed and its registration deleted),
   but do this once before the first run: delete the old persistent runners under Settings > Actions > Runners.

   Then create the repo secret `RUNNER_ADMIN_TOKEN`: a fine-grained personal access token for this
   repository only, permission Administration: read and write, nothing else. The `start-vm` and
   `stop-vm` jobs use it to mint and delete the one-job runner; the `bench` job never sees it.

3. Azure sign-in is the one the desktop benchmark already uses (`AZURE_CLIENT_ID`,
   `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`; see `docs/azure-desktop-runner.md`). Its role is
   assigned on the whole `operon-bench` resource group, so it can already start and deallocate
   `operon-linux`. It is trusted for the `main` branch only, so the workflow can only be run
   from `main`. If the VM has another name, set the repo variable `OSWORLD_VM_NAME`. Delivering the
   runner config needs one more permission in the role, `Microsoft.Compute/virtualMachines/runCommand/action`
   (the role JSON is in `docs/azure-desktop-runner.md`; update the role with `az role definition update`).

4. The repo secret `GEMINI_API_KEY` is the one the other benchmarks use. The key reaches the run
   as an environment variable of the job; it is not written to the VM's disk by the workflow.
   A `model` run also uses the repo secret `ANTHROPIC_API_KEY` (Claude) or `OPENAI_API_KEY` (GPT); each reaches
   only a run of that provider's model, never a Gemini or grounding run.

The first run also creates the two Python environments (`~/osw-venv`, Python 3.11 with OSWorld's
requirements, which pin `torch` below Python 3.14; `~/operon-venv`, Python 3.14 with Operon) and
downloads the image, so it takes a few minutes longer.

## Notes

- The runner label is different every run and exists only while its one job does; there is no lasting
  `operon-osworld` runner and no label for another workflow to ask for.
- A self-hosted runner trusts the code it runs: never point a workflow that runs on pull requests at the VM.
- Never approve a fork PR's workflow runs while the "Fork PR workflow check" is red (it lists changes under `.github/`).
- The VM keeps its disk (about $10 a month for 128 GB) while deallocated; compute bills only
  while a run is going.
- Backstop, as for the desktop VM: `az vm auto-shutdown --resource-group operon-bench --name operon-linux --time 0700`
  deallocates it daily in case a run never reaches its `stop-vm` job.
