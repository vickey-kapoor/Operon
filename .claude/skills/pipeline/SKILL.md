---
name: pipeline
description: Operon's coding pipeline for any code change - Opus plans, Haiku scanners find, Sonnet coders fix in isolated worktrees, a fresh Opus verifier checks. Use for every story and every code change, small ones included.
---

# Pipeline: Opus plans, Haiku scans, Sonnet codes, Opus verifies

Use this for every code change in Operon, small ones too. The main session (Opus) plans, hands out the work and
integrates. It does not write the code itself.

## 1. Plan (main session, Opus)
- Read the story and the code it touches. Write a short step-by-step plan: what to change, where, and the exact
  search patterns the scanners should use.
- Post the plan in the thread and carry on. Stop for Vickey only where "Merging" in `CLAUDE.md` says she decides (paid
  runs, `needs-vickey` merges).

## 2. Scan (Haiku)
- Start `scanner` subagents (`.claude/agents/scanner.md`, Haiku) in parallel, one per area of the plan.
- Use a Sonnet helper instead only when the search needs reasoning across files.
- Check the scan covered every area in the plan before moving on.

## 3. Code (Sonnet)
- Start one `coder` subagent (`.claude/agents/coder.md`, Sonnet, its own git worktree) per fix. Fixes that touch the
  same file go to the same coder.
- Coder worktrees branch from `main`. For work on an unmerged branch, tell the coder to check that branch out first.
- Every fix ships with a test that fails without it.
  CI checks this (`Tests fail without the fix`, KAN-280); a PR that changes Python code under `src/` (`src/**/*.py`) with no changed test fails; prompt, yaml and other data-only `src/` changes are skipped. Label `no-red-test` only for a refactor that keeps behaviour the same, or when the changed tests pass on base by design (e.g. a new guard in `tests/test_invariants.py`), and say why in the PR body; the verifier checks that reason. Then re-run the failed job, since the check reads labels when it runs.
- If a coder fails its tests twice, or the fix needs a design decision across modules, hand it to an Opus helper
  subagent and say so on the checklist.

## 4. Integrate (main session)
- Merge the coders' commits onto the story branch. Run lint, type check and the tests (commands in `CLAUDE.md`) on the
  combined result. The main session does not fix code itself: it sends any breakage back to a coder. It may only
  resolve merge conflicts. Never open a PR with failing tests.

## 5. Verify (Opus)
- Start a fresh `verifier` subagent (`.claude/agents/verifier.md`, Opus) that did not write the change. It checks every
  finding and every fix, and reruns lint, types and tests. Send what it finds back to a coder, then verify again on the
  new head.
- After APPROVE on the head commit, and only once step 6's "Real check:" line is in the PR body, the main session posts `<!-- operon-verifier: PASS <full head sha> -->` on the PR.
- A Fable 5.2 review in a fresh session is required for `risky` PRs. It is also done for `needs-vickey` PRs, which
  then still wait for Vickey's own "merge".
- See "Merging" in `CLAUDE.md` for the markers, the no-paid-run rule and the rule that agents never merge by hand, except as Merging allows.

## 6. Real $0 check (main session)
- Run the change for real at $0: a real end-to-end test run, a dry run on GitHub, or a replay of real data. Never a paid run, never a real model key.
- Put a "Real check:" line in the PR body with the run or log link and what it showed. Tests and reviews alone do not count.
- No "Real check:" link, no verifier marker. The PR gate refuses to merge without it and leaves a note (KAN-298).

## Record who did what
- Show each stage on the progress checklist with the model that ran it.
- Add one line to the PR body, for example:
  `Who did what: Plan: Opus · Scan: Haiku x2 · Code: Sonnet x1 · Verify: Opus · Review: Fable 5.2`
- If a stage was handed to a different model, name it and say why.
