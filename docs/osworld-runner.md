# OSWorld benchmark runner (KAN-169)

`Actions > Run OSWorld test tasks (own VM) > Run workflow` runs Operon on a fixed 25-task subset of
OSWorld-Verified (`evals/osworld_subset.json`) and scores it with OSWorld's own checkers.
It is manual only: it spends Gemini credits (about $0.06 a task; the 5-task pilot cost
$0.31), so you pick how many tasks and the cost cap each time.

```
start-vm   az vm start operon-linux            (ubuntu-latest, Azure OIDC sign-in)
bench      self-hosted runner on operon-linux  scripts/osworld-vm/setup.sh, then python -m evals.osworld
stop-vm    az vm deallocate operon-linux       (always, even when the run fails)
```

Inputs: `limit` (first N tasks; 5 = the pilot, 25 or 0 = all), `only` (one task id prefix),
`max_cost_usd` (stop starting tasks past this; use 8 for all 25), `max_steps`. The report
(`osworld-*.json`, per-task pass/fail, cost, seconds), the log and each task's Operon run folder
(screenshots of the task VM, model I/O, traces) are uploaded as the `osworld-report` artifact, so a failed task can be diagnosed from it.
Each task's Operon output (exit code, stdout, stderr) is saved in `osworld-logs/<task id>.log` and uploaded too, so a task that fails at startup still shows its error. The log's tail is on the run's summary page.

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

2. Register a GitHub Actions runner on the VM with the label `operon-osworld`. Get a
   registration token in the repo under Settings > Actions > Runners > New self-hosted runner
   (Linux, x64), then follow the commands it shows, adding the label and installing it as a
   service so it comes back after every start of the VM:

   ```bash
   ./config.sh --url https://github.com/vickey-kapoor/Operon --token <token> --labels operon-osworld --unattended
   sudo ./svc.sh install && sudo ./svc.sh start
   ```

3. Azure sign-in is the one the desktop benchmark already uses (`AZURE_CLIENT_ID`,
   `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`; see `docs/azure-desktop-runner.md`). Its role is
   assigned on the whole `operon-bench` resource group, so it can already start and deallocate
   `operon-linux`. It is trusted for the `main` branch only, so the workflow can only be run
   from `main`. If the VM has another name, set the repo variable `OSWORLD_VM_NAME`.

4. The repo secret `GEMINI_API_KEY` is the one the other benchmarks use. The key reaches the run
   as an environment variable of the job; it is not written to the VM's disk by the workflow.

The first run also creates the two Python environments (`~/osw-venv`, Python 3.11 with OSWorld's
requirements, which pin `torch` below Python 3.14; `~/operon-venv`, Python 3.14 with Operon) and
downloads the image, so it takes a few minutes longer.

## Notes

- Only this workflow uses the `operon-osworld` runner label, and it only runs on a manual start.
- A self-hosted runner trusts the code it runs: do not use the label from any workflow that runs
  on pull requests.
- The VM keeps its disk (about $10 a month for 128 GB) while deallocated; compute bills only
  while a run is going.
