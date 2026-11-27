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
- On a large change, show the plan to Vickey and wait for her OK before going on.

## 2. Scan (Haiku)
- Start `scanner` subagents (`.claude/agents/scanner.md`, Haiku) in parallel, one per area of the plan.
- Use a Sonnet helper instead only when the search needs reasoning across files.
- Check the scan covered every area in the plan before moving on.

## 3. Code (Sonnet)
- Start one `coder` subagent (`.claude/agents/coder.md`, Sonnet, its own git worktree) per fix. Fixes that touch the
  same file go to the same coder.
- Coder worktrees branch from `main`. For work on an unmerged branch, tell the coder to check that branch out first.
- Every fix ships with a test that fails without it.
- If a fix fails its tests twice or spans several files, hand it to an Opus helper and say so on the checklist.

## 4. Integrate (main session)
- Merge the coders' commits onto the story branch. Run lint, type check and the tests (commands in `CLAUDE.md`) on the
  combined result, and fix anything that breaks before going on. Never open a PR with failing tests.

## 5. Verify (Opus)
- Start a fresh `verifier` subagent (`.claude/agents/verifier.md`, Opus) that did not write the change. It checks every
  finding and every fix, and reruns lint, types and tests. Fix what it finds and verify again on the new head.
- Risky and `needs-vickey` PRs also get a Fable 5.2 review in a fresh session (see "Merging" in `CLAUDE.md`).

## Record who did what
- Show each stage on the progress checklist with the model that ran it.
- Add one line to the PR body, for example:
  `Who did what: Plan: Opus · Scan: Haiku x2 · Code: Sonnet x1 · Verify: Opus · Review: Fable 5.2`
- If a stage was handed to a different model, name it and say why.
