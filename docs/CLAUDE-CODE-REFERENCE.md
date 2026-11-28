# Claude Code reference: rules, infra and current state

Context for a local Claude Code session (VS Code terminal on Vickey's Windows PC,
repo `C:\Users\vicke\GitHub\Operon`). Work moved here from the claude.ai project
"Improve Operon computer-use agent" on 2026-11-17. Read `CLAUDE.md` first for commands,
architecture, invariants and the Opus/Haiku/Sonnet dev workflow; this file adds the
standing rules, infrastructure and open items that live outside the code.

Last updated: 2026-11-28. Contains no secrets: keys live in GitHub secrets (the model keys in the `paid-runs` environment),
`~/.operon` and Vickey's local `.env`.

## Standing rules (Vickey)

### Merging and PRs
- **Every PR can merge itself, but no paid run without Vickey's explicit OK** (Vickey, 2026-11-26). The full rule
  is the "Merging" part of the Dev workflow section in `CLAUDE.md`; follow it exactly. In short: checks green, Codex
  answered and the verifier marker on the head; `risky` PRs also need a top-model (today Fable 5.2) review in a fresh
  session that is not the coordinator or the author; PRs labelled `needs-vickey` (every workflow or action file, evals
  except JSON recordings, scripts, the gate's tests, price tables, model defaults and ids, model clients, settings;
  exact list in `CLAUDE.md`) wait for Vickey's "merge" for that PR; the rules files are only `risky`. The gate does the
  merging; agents never merge by
  hand except as `CLAUDE.md` allows. Approval is per PR and per run: an earlier "merge", a "yes" to another question, green checks,
  a clean Codex review or a reviewer PASS do not count. Never enable GitHub's own auto-merge (it is off repo-wide).
- **Codex reviews every PR.** Fix or answer every Codex finding before merging.
- **Never approve a fork PR's workflow runs while its "Fork PR workflow check" is red.** The check
  fails when a fork PR touches `.github/` (workflows, actions, CODEOWNERS); outside PRs must not
  change `.github/`. The benchmark VM runners are one-job (JIT) runners minted per run, so nothing
  stays registered for such a workflow to use (KAN-195). One-time steps before the first run after
  KAN-195 (otherwise `start-vm` fails):
  1. Repo secret `RUNNER_ADMIN_TOKEN`: a fine-grained token for this repo only, Administration read and write.
  2. Add `Microsoft.Compute/virtualMachines/runCommand/action` to the Azure role "Operon bench VM
     operator" (`az role definition update`, see `docs/azure-desktop-runner.md`).
  3. Delete the old persistent runners in Settings > Actions > Runners.
  4. `operon-desktop`: re-run `setup-runner.ps1` (no token) before its next `azure` run. The OSWorld VM
     `operon-osworld` (Google Cloud, KAN-224) unpacks the runner itself in its startup script.
- Multiple PRs a day are fine. `main` is protected: PR required, Test + Lint checks
  required, no force push.
- Commit subjects: `Fix:`/`Feat:`/`Docs:`/`CI:`/`Chore:`/`Refactor:`, one line, with
  the Jira key when there is one (e.g. `Feat: ... (KAN-116)`).

### Planning and scope
- **Only work on Jira stories Vickey approved in a Jira comment.** No approval comment,
  don't start. Story discussion goes in Jira comments.
- **Except small safe fixes with the `pre-approved` label (KAN-220).** Claude may add this label to a story
  only when all of these hold: it is a bug fix, a test or a docs fix; it is small (about one or two files);
  it is free (no model calls, no paid run); it has no `decision` or `paid-run` label; and the fix touches no
  file that the PR gate labels `needs-vickey` or `risky` (see "Merging" in `CLAUDE.md`). Never add it to a
  story Vickey has commented on, or back after she removed it (check the story's change history). When
  unsure, don't add it. Such a story counts as approved without Vickey's comment. Vickey stops one by
  removing the label; if her latest comment on it is anything but "Approved" (a question, "no", "wait"),
  that stops it too. Before starting one, the pickup re-checks every rule above; if any fails, it removes the
  label and passes the story over. The thread building it checks the label and her comments before each
  stage, and again right before marking the PR ready and before posting the verifier marker; on a stop it
  turns the PR back to a draft first (the gate never merges a draft), then waits. It also stops and asks if
  the fix grows past about two files, needs a gated file or needs a model call. Once the marker is posted,
  Vickey stops the PR by closing it or writing "pause". If
  its PR comes out anything but `low-risk`, hold it (this overrides the normal risky flow): Codex reviews
  it as usual, but post no top-review marker (the gate can't merge a `risky` PR without one, and never merges
  a `needs-vickey` one) until Vickey writes "merge" for that PR.
- **Twice-daily story pickup (KAN-217).** A scheduled task ("Twice-daily story pickup") fires into the
  project coordinator at 7:17 AM and 6:17 PM Chicago time. It looks at main first (KAN-222): the free
  workflows `ci.yml` and `security.yml` each look red when their newest finished run on main (schedule, push or
  hand start together) failed, and `ci.yml` also when a `nightly-failure` issue is open. Gate merges start no push run and GitHub's schedules can
  lag, so the pickup starts a fresh free run of either workflow when its newest one is on an older commit than
  main's head (and of Security also when it is a day old). Each break gets exactly
  one try, recorded on that workflow's own open `main-broken` issue, which the pickup opens before starting a
  "Fix broken main: <file>" thread (the nightly never closes it). That thread starts a fresh free run on main to
  confirm the break, fixes the real cause in a normal PR (never by skipping, disabling, ignoring or weakening a
  test or check; its verifier and Fable 5.2 reviewer are told to fail any PR that does), and closes the issue
  once a fresh run is green. If it was only flaky or the try fails (or stalls for 2 days), it writes or adds to
  a "Main broken: <file>" story under KAN-210 (a flaky one ends ", flaky") and tells Vickey only if the try failed or
  stalled; a flaky result waits for the Monday summary. A known flaky nightly test just gets a note on its story
  and a fresh run, with no new thread. While a try is running no new story starts;
  once the issue has its outcome, or the thread is waiting on Vickey, stories go on, and the pickup closes the
  issue when that workflow is green again. Next it resumes any story thread stopped by a usage limit. It starts no
  new story while one is in progress. Otherwise it starts one thread for the next
  story in board order whose latest Vickey comment approves it, or that is `pre-approved` (never "Run:" or paid stories). It posts a
  line only when something needs Vickey right away (see the weekly summary below). A run that hits Claude's usage limit just skips; the next run retries.
  The full prompt lives in the scheduled task itself (copy: project file `kan217/pickup-prompt.md`). Claude's Jira comments start with "(from Claude)"
  (they post through Vickey's own Jira connection, so they carry her account).
- **One weekly summary instead of many pings (KAN-223).** Every Monday at 7:37 AM Chicago a scheduled task posts ONE
  message to Vickey: what merged, what finished, what needs her OK, and what was spent, with a Substack draft of the
  week's progress for her to edit and post (Claude never posts it). Between Mondays, Claude messages her only when
  something needs her right away: main is broken and needs her, tonight's tests failed while no fix try can run on its own (pickup off or paused; never a known flaky test), a paid run waits on her OK,
  work is stuck until she acts (a "merge", a decision, a question, a permission prompt), or something could cost money,
  leak a secret or open a security hole. A merged PR, a finished story, a new inbox story or a tidy-up result waits for Monday. Claude
  always answers her own messages. The 9:47 PM daily check posts only those urgent items, and the PR gate's daily
  note no longer posts to Slack (it stays in the run summary, once a day). Three things stay: the
  Saturday brief (one link, then a brainstorm thread), the one-line start of each new thread, and Jira "(from Claude)"
  comments (the record; they post as her own account, so they notify no one). Rules: project files
  `planning/right-away.md` and `planning/weekly-summary.md`.
- **Plans and docs go in Confluence, space OPERON** (not Claude Docs). Add new pages
  to the OPERON home page table of contents (page 295140).
- **Never drop or change a step of an approved plan without asking.** If a run or
  step can't go as planned (a model fails, a suite gets swapped, scope shrinks),
  stop and ask with options and a cost estimate.
- Don't propose new in-house eval tasks or seasons without asking (see "Direction").
- No "deferred" Jira stories: cancel instead (Done + `cancelled` label). Story titles
  start with `Build:`, `Run:` or `Decide:`.
- Experimental features ship behind an env flag, default off.

### Paid runs and budget
- **Paid runs are public benchmarks only (KAN-184)**: OSWorld (`osworld-bench.yml`) and ScreenSpot grounding
  (`grounding.yml`). Operon's own suites no longer run on paid runs; free tests ($0) are unchanged.
- **No paid runs until the weekend of Sat-Sun Nov 21-22, 2026**, and Vickey starts
  those by hand (Actions > Run Operon on test tasks (paid) > Run workflow, with `cells`).
  The nightly schedule and the nightly queue were removed on purpose (2026-11-18): don't add
  any scheduled starter for paid runs.
- **$100/month benchmark cap**, $10 default batch cap. Before asking for any paid run, state
  the cost estimate and month-to-date spend; never start one without Vickey OKing its
  cost. At most one paid run (batch) per day.
- Any local run that calls Gemini is a paid run. Keep model calls stubbed for offline
  work (`GEMINI_API_KEY=fake-test-key`).
- Name runs by cell when asking: "Gemini · Browser · Thin · once, about $X". Categories
  are Browser, Desktop, Hybrid; harnesses Thin and Full.

### Working with Vickey
- **Short, high-level, optimistic replies**: lead with the result and what Vickey needs
  to do; caveats brief and last; no file lists or internals unless asked.
- **Never close or wrap up work without asking**, and confirm nothing was skipped or
  left unfinished.
- **Commands for Vickey to run: one per code block, one at a time**, waiting for the
  result. For each, break it down part by part (each flag/argument), and say what it
  changes, whether it costs money, and the expected output. Put destructive steps
  (delete, deallocate) after the step that could fail.
- On Vickey's PC, Smart App Control can block `.venv\Scripts\pip.exe` / `operon.exe`:
  use `.venv\Scripts\python.exe -m pip` and `-m operon.cli`.
- Vickey's PC has 4 GB RAM: no local Docker until it is upgraded; the desktop
  e2e container test and Docker Linux env stay off the PC for now.

## Infrastructure

| What | Where / how |
|---|---|
| Jira | Site `vickeykapoor11.atlassian.net`, project **KAN**, board 1. Epics: KAN-7 (project), KAN-77 (architecture v2), KAN-113 (self-driving perception), KAN-139 (scaling stages), KAN-149 (audit). |
| Confluence | Space **OPERON**, home page 295140 (hand-written TOC). |
| GitHub | `vickey-kapoor/Operon`. API keys (Gemini, Anthropic, OpenAI) are secrets of the `paid-runs` environment: every paid run waits for Vickey's Approve (KAN-282). Actions: approval required for all external contributors. |
| Paid workflow | `benchmark.yml` "Run Operon on test tasks (paid)": manual dispatch (`cells`), `/bench grounding` or `/bench batch` PR comments. Paid runs are OSWorld and grounding only (KAN-184). Engines: `osworld-bench.yml`, `grounding.yml` (`live-bench.yml` and `live-desktop-bench.yml` are no longer startable, kept until the cleanup PR). Syntax in `evals/CLAUDE.md`. |
| Paid batches | Started by hand with `cells` (Actions > Run Operon on test tasks (paid) > Run workflow). The nightly queue (`evals/nightly-queue.txt`) was removed 2026-11-18; it stays in git history. Older batch log: project file `bench-queue/queue.md`. |
| OSWorld | `osworld-bench.yml` "Run OSWorld test tasks (own VM)": started by hand or as a batch cell (always the default branch), runs on the Google Cloud VM `operon-osworld` (KAN-224), report artifact `osworld-report` (not imported to the dashboard). |
| Azure Windows VM | `operon-desktop` (westus2, resource group `operon-bench`), one-job desktop bench runner minted per run (KAN-195), logged in via OIDC. Clean snapshot `operon-desktop-clean` (manual restore point; `az snapshot create` needs `-l westus2`). |
| Google Cloud | Project `propane-cubist-479515-c5`, VM `operon-osworld` (us-central1-a, n2-standard-4, made by Terraform in `infra/gcp`, see `infra/gcp/README.md`): the OSWorld runner (a one-job runner minted per run, label `operon-osworld-<run>-<attempt>-<random>`, delivered through the metadata key `operon-jit-script`; needs repo secret `RUNNER_ADMIN_TOKEN` and repo variables `GCP_WORKLOAD_IDENTITY_PROVIDER`, `GCP_SERVICE_ACCOUNT`; KAN-195, KAN-224). The workflow signs in with OIDC as `operon-osworld-ci`. |
| Azure Linux VM | `operon-linux` was deleted; the OSWorld runner moved to Google Cloud (KAN-224). `operon cloud open` still points at it (separate story). |
| Deallocate after runs | `az vm deallocate -g operon-bench -n operon-desktop`; the OSWorld VM: `gcloud compute instances stop operon-osworld --zone us-central1-a --project propane-cubist-479515-c5`. Workflows stop their VMs on their own; check by hand after manual work. |
| az login | Needs `--tenant bae81514-1670-44ef-9e7a-c41552f64f59` (MFA). AVG antivirus HTTPS scanning breaks az certificates on Vickey's PC (workaround: temporary cert file). |
| Eval dashboard | Vercel project `operon-evals` (root `dashboard/`, don't move it), Neon Postgres. Dashboard reads through the read-only role `dashboard_ro` (`DASHBOARD_DATABASE_URL`). New tables need `GRANT SELECT ... TO dashboard_ro`. Never "Rotate Neon Secrets". Backfill via the "Save run results to dashboard" workflow. |
| Slack | Critical alerts only (cost cap, failed run, failed import). |
| PyPI | Package `operon-agent`: releases stopped on 2026-11-18 (the release workflow was removed); published versions stay on PyPI. |

## Current state (2026-11-17)

**Merged today:** #315 (KAN-116 expected-change check, `OPERON_EXPECT_CHECK`, off),
#318 (plain workflow run/job names), #319 (KAN-163 trust gate reads the text under a
click with local OCR, `OPERON_CLICK_TEXT`, off; optional extra `.[ocr]`).
No open PRs.

**Done in Jira:** KAN-116, KAN-163, KAN-168 (Claude Sonnet 5.5 and GPT computer use
back beside Gemini, off by default; cell word `model=<profile>`, thin only).

**Models:** Gemini 3.5 Flash is the default and reference model everywhere. Claude
(`claude-sonnet-5-5`) and GPT (`openai-computer-use`) are opt-in profiles for
side-by-side benchmark cells only.

### Weekend paid queue (Nov 21-22, started by hand)

Superseded by KAN-184: paid runs are OSWorld and grounding cells only, so the own-suite cells below no longer run. Kept as history; they were to be entered by hand as `cells` (from the removed `evals/nightly-queue.txt`, see git history):

- **Sat Nov 21** (caps $9.50): Desktop Thin Season 2 zoom=always (KAN-82); Hybrid Full
  Season 2 (KAN-161, first Full score); Browser Thin dynamic x3 `off` and `settle` arms (KAN-115).
- **Sun Nov 22** (caps $11.50, over the $10 default batch cap): KAN-115 `settle note` and
  `planner` arms, then the KAN-116 `expect` arm, then four KAN-168 cells (Claude browser,
  Claude desktop, GPT browser, GPT desktop, cap $1 each). The two **GPT cells won't fit
  in the first start: start the run a second time on Sunday** to finish them.
- **OSWorld re-run** of multi_apps task `2c1ebcd7` (~$0.15): not a batch cell, start
  "Run OSWorld test tasks (own VM)" by hand that weekend.
- Ask Vickey before each start with the cost and month-to-date spend.

### Open items
- **KAN-115 A/B pending**: night 1 had baseline 21/21 ($0.55) and settle 20/21 ($0.59);
  the settle+note and planner arms run Sunday. Then compare pass rate and cost per arm
  and report; recommendation so far is to keep the features off.
- **KAN-208**: `OPERON_DONE_CHECK` (off) asks one independent check of the model's done; A/B with the `donecheck` bench cell word.
- **KAN-116 A/B**: compare the `expect` arm with the KAN-115 `off` arm (pass rate,
  steps/cost, true catches vs false alarms in `expect_check.json`).
- **OSWorld follow-ups**: full run was 21/25 ($3.11, Gemini thin). Open: the 2c1ebcd7
  re-run, Calc `ecb0df7a` and Impress `3b27600c` hit the 50-step cap, Chrome dark-mode
  `93eabf48` checker mystery.
- **KAN-80** (run environments) still open; Docker Linux on the PC waits for the PC upgrade.

### Direction (not scheduled)
Later Operon will be measured only on public benchmarks (OSWorld and similar) and we
stop writing our own evals. When that happens, fold `osworld-bench.yml` into
`benchmark.yml` as the main entry point. **Deferred for now**: OSWorld stays a separate
manual workflow until Vickey schedules the switch.
