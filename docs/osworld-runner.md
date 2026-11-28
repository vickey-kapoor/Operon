# OSWorld benchmark runner (KAN-169)

`Actions > Run OSWorld test tasks (own VM) > Run workflow` runs Operon on a fixed 25-task subset of
OSWorld-Verified (`evals/osworld_subset.json`) and scores it with OSWorld's own checkers.
It is manual only: it spends Gemini credits (about $0.06 a task; the 5-task pilot cost
$0.31), so you pick how many tasks and the cost cap each time.

It runs on the Google Cloud VM `operon-osworld` (KAN-224; project `propane-cubist-479515-c5`, zone
`us-central1-a`), made with Terraform in `infra/gcp` (see `infra/gcp/README.md`). It used to run on the Azure
VM `operon-linux`, which was deleted.

```
start-vm   gcloud compute instances start operon-osworld   (ubuntu-latest, Google Cloud OIDC sign-in)
bench      one-job runner on operon-osworld                 scripts/osworld-vm/setup.sh, then python -m evals.osworld
stop-vm    gcloud compute instances stop operon-osworld    (always, even when the run fails), then delete the run's runner if it is still registered
```

The `bench` job runs on a one-job (just-in-time) runner (KAN-195). `start-vm` mints it through the GitHub API
under a label unique to the run (`operon-osworld-<run id>-<attempt>-<random>`) and puts the VM script that starts
it on the VM's metadata key `operon-jit-script` (`gcloud compute instances add-metadata --metadata-from-file`, so
the config is never in a command line). The VM's `operon-jit-watch` service runs that script once; the runner
takes that one job and is gone. As soon as the runner is online, or when it never comes online, `start-vm` takes
the key off the metadata again (a failed removal fails the job), and `stop-vm` clears it once more. The label
carries a random suffix, so cells of one batch do not clash and it cannot be guessed from the public run id. No
lasting runner stays registered, so another workflow, such as an approved fork PR's, has no standing runner to
target on the VM. The logic is `scripts/bench-runners/jit-runner.sh` (`start` with `VM_CLOUD=gcp`, and `remove`)
and `scripts/bench-runners/vm-linux.sh` (runs on the VM; it hands the config to the runner in the
`ACTIONS_RUNNER_INPUT_JITCONFIG` variable, never on its command line).

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

1. Create the VM and the sign-in with Terraform and set the repo variables `GCP_WORKLOAD_IDENTITY_PROVIDER`
   and `GCP_SERVICE_ACCOUNT` from its outputs: `infra/gcp/README.md` has the steps. Nothing has to be done on
   the VM by hand: its startup script installs Docker, Python, git and the Actions runner binaries (unpacked
   only, never configured) on its first start.

2. The repo secret `RUNNER_ADMIN_TOKEN`: a fine-grained personal access token for this repository only,
   permission Administration: read and write, nothing else. The `start-vm` and `stop-vm` jobs use it to mint
   and delete the one-job runner; the `bench` job never sees it.

3. The sign-in is trusted for this repository's `main` branch and for jobs of `osworld-bench.yml` only (also
   when `benchmark.yml` calls it), so the workflow can only be run from `main`. Its service account
   `operon-osworld-ci` can only get, start, stop, change the metadata of and read the serial console of
   `operon-osworld`. If the VM has another name or zone, set the repo variables `OSWORLD_VM_NAME` and
   `OSWORLD_GCP_ZONE` (and `GCP_PROJECT_ID` for another project).

4. The model keys `GEMINI_API_KEY`, `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` are secrets of the `paid-runs`
   GitHub environment (KAN-282), not repo secrets. `benchmark.yml` passes each key to the called workflow by name (a
   reusable workflow gets an empty value otherwise); the value comes from the environment of the called job. The jobs `start-vm` and `bench` both wait for Vickey's Approve
   on that environment, so a run needs two Approve clicks: machine start, then tasks. After the first click the
   VM is up and its runner is registered, so the 20-minute idle power-off in `infra/gcp/startup.sh` does not fire:
   it bills about $0.20 an hour until the second click (worst case the VM's 17 h `max_run_duration`, about $3.40).
   Press the second Approve right away, or Reject to stop the VM. A key reaches the run
   as an environment variable of the job; it is not written to the VM's disk by the workflow.
   A `model` run uses `ANTHROPIC_API_KEY` (Claude) or `OPENAI_API_KEY` (GPT); each reaches
   only a run of that provider's model, never a Gemini or grounding run.

The first run also installs everything on the VM (`start-vm` waits up to 15 minutes for the runner), creates the
two Python environments (`~/osw-venv`, Python 3.11 with OSWorld's requirements, which pin `torch` below Python 3.14;
`~/operon-venv`, Python 3.14 with Operon) and downloads the image, so it takes a few minutes longer.

The task VM OSWorld starts inside `operon-osworld` keeps OSWorld's default size (4 GB, 4 cores), so scores stay
comparable with earlier runs. `operon-osworld` has 16 GB and an 8 GB swap file.

### Paid-runs environment setup

Done once in GitHub (Settings, Environments), by Vickey (KAN-282). Order: steps 1-6 BEFORE this PR merges (if it
merges first, GitHub creates `paid-runs` on the first run with no reviewer, and the repo keys are still there), then
merge, then step 7.

1. Create the environment `paid-runs`.
2. Required reviewers: Vickey.
3. Untick "Allow administrators to bypass configured protection rules".
4. Leave "Prevent self-review" off: agents start runs as her account, so it would lock her out.
5. Deployment branches and tags: Selected branches and tags, `main` only.
6. Add `GEMINI_API_KEY`, `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` as environment secrets.
7. After the merge, delete the three repo secrets: while they exist, any job without the environment can read them.
8. Check the gate at $0 with `gate-dryrun.yml` ("Approve gate check"). No model call, no machine start, no checkout.
   Only Vickey adds the label and starts the real check; agents do not.
   - Before the merge: create both test environments first, then add the label (if a job names an environment that
     does not exist yet, GitHub creates it without protection). `dryrun-gate`: required reviewer Vickey, no branch
     rule, secret `DRYRUN_KEY` with any dummy value. `dryrun-main-only`: required reviewer Vickey, deployment
     branches `main` only, secret `DRYRUN_KEY`. Then add the label `gate-dryrun` to the PR and Approve `direct`,
     `by-name` and `not-passed`.
   - `main-only` must be refused by its branch rule, so the PR run ends with a red `main-only` job by design. Read
     the `report` job: it is green when everything held, and its summary names who approved and when, with a table of
     every job. If `main-only` asks for your Approve, that is a fail: Reject it.
   - After the merge and step 7: start "Approve gate check" by hand with target `real`, from main, and Approve the
     waiting jobs. It runs the real grounding and OSWorld workflows with `dry_run` true (they check their keys, then
     stop) and checks that a job outside `paid-runs` has none of the three model keys, which shows the repo secrets
     are gone.

   What this does not prove:
   - The main-only rule on `paid-runs` itself (real mode runs on main, so it never meets the rule).
   - The admin-bypass and self-review settings.
   - The `benchmark.yml` and `/bench` paths.
   - OSWorld's second Approve (`bench`, `stop-vm`) and the Google sign-in: a dry run stops before them.
   - The Anthropic and OpenAI keys inside `paid-runs` (a dry run uses the default model).
   - That an agent's token could not approve through the API: the no-approve rule covers that.

The first real approved run (with Vickey's OK for its cost) should go through the paid-run workflow (`benchmark.yml`)
and confirm the "Check the secrets" step passes, which shows the keys arrive. A run left waiting holds its queue for
up to 30 days, so Reject the runs you do not want.

A hand start of `grounding.yml`, `benchmark.yml` or `osworld-bench.yml` from a branch is now blocked (the environment accepts `main` only). To test a branch, use
a batch cell `grounding ref=<branch>` or `/bench grounding` on the PR. A batch needs two Approve clicks per OSWorld cell,
and its cells run one after another.

## When a run fails

- `stop-vm` prints the last 60 lines of the VM's memory log (`free -m` and the five biggest processes, every
  30 s) on the run's summary page when the `bench` job did not succeed: a run that froze or was killed for memory
  shows it there. The same log is on the serial console:
  `gcloud compute instances get-serial-port-output operon-osworld --zone us-central1-a | grep operon-memlog`.
- If a run never reaches `stop-vm`, the VM still stops: it powers itself off after 20 minutes with no runner, and
  Google Cloud stops it after 17 hours running whatever happens.

## Notes

- The runner label is different every run and exists only while its one job does; there is no lasting
  `operon-osworld` runner and no label for another workflow to ask for.
- A self-hosted runner trusts the code it runs: never point a workflow that runs on pull requests at the VM.
- Never approve a fork PR's workflow runs while the "Fork PR workflow check" is red (it lists changes under `.github/`).
- Cost: about $0.20 an hour while a run is going (n2-standard-4 and its address); the 150 GB disk about $15 a
  month while the VM is stopped. A stopped VM bills nothing else.
- The VM has no Google identity of its own and its firewall lets nothing in; it reaches out only (GitHub, Docker
  Hub, PyPI, the model APIs).
