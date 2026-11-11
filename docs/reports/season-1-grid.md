# Season 1 grid: Gemini, Claude and GPT

Results of the first model grids on Season 1 (frozen 2026-11-10, season hash `423cd257962f`).
Each cell is one model in one category (Browser, Desktop, Hybrid) and one harness (Thin, Full),
run once on that category's 10-task core set. Every task is graded from the apps' or the
machine's state; no screenshot judge. Runs used `/bench` PR comments; Desktop and Hybrid ran on
the Azure VM. How the grid works: [evaluation.md](../evaluation.md#categories-and-the-grid).

## Current grid

| Model | Browser Thin | Browser Full | Desktop Thin | Desktop Full | Hybrid Thin | Hybrid Full |
|---|---|---|---|---|---|---|
| Gemini 3.5 Flash (computer use) | 10/10 | 10/10 | 9/10 | 10/10 | 9/10 | 7/10 |
| Claude Sonnet 5.5 | 9/10 | 10/10 | 10/10\* | 10/10\* | not run | not run |
| GPT (`openai-computer-use` profile) | 10/10 | in progress | 10/10 | in progress | not run | not run |

\* Claude's Desktop cells ran on the Desktop core before the long and safety tasks were added
(KAN-74), so they aren't directly comparable with Gemini's.

Full cells for Claude and GPT use the model as its own step critic (`same-critic`).

## Gemini 3.5 Flash (KAN-66, KAN-75, KAN-76)

| Run | Cells | Cost |
|---|---|---|
| First pass, 2026-11-09 | Browser Thin 10, Browser Full 8, Desktop Thin 10, Desktop Full 9 | $3.50 |
| Full rerun after the KAN-75 fixes | Browser Full 10, Desktop Full 10 | $1.95 |
| New Desktop and Hybrid core sets, 2026-11-10 | Desktop Thin 9, Desktop Full 10, Hybrid Thin 9, Hybrid Full 7 | $3.64 |

The first pass lost three Full tasks, all to Operon full-harness bugs that #228 fixed: Gemini's
key_down and key_up weren't mapped, a critic stop on a non-stop step ended a run, and a STOP was
accepted without a screenshot check.

Misses on the current core sets:

- `desk_csv_archive`, Desktop Thin: model miss. The total file held 188.75 instead of 147.50.
- `hybrid_notes_to_email`, Hybrid Thin: model miss. The email body was "Meeting notes" without
  the notes. In Hybrid Full the same task timed out with nothing sent.
- `hybrid_order_from_file`, Hybrid Full: model miss. The order was right, but the receipt said
  "Total charged: .14" for $63.14: Gemini wrote it with a double-quoted PowerShell
  `Set-Content`, so PowerShell expanded `$63` as an empty variable. Operon typed the text
  exactly as given.
- `hybrid_it_inventory`, Hybrid Full: failed after 9 steps with no state change. Not triaged yet.

## Claude Sonnet 5.5 (KAN-70)

| Cell | Result | Cost |
|---|---|---|
| Browser Thin | 9/10, 0 false successes, nothing compromised | $0.67 |
| Desktop Thin | 10/10 | $0.51 |
| Browser Full, Claude critic | 10/10, 0 false successes, nothing compromised | $0.92 |
| Desktop Full, Claude critic | 10/10 | $0.46 |

Total $3.41, including a first Full pass that didn't count: every Claude critic call returned
HTTP 400 because Operon sent `thinking: disabled`, which Sonnet 5.5 rejects. #233 fixed it.

- `long_team_supplies`, Browser Thin: Operon miss. The app state was correct, but the thin
  completion check rejected Claude's "done" because the item it checked wasn't among the screens
  it was shown.
- In the Full rerun some critic replies still couldn't be parsed and those steps fell back to the
  deterministic check. #233 also reads a verdict with prose around the JSON; that change has not
  been checked on a live run yet.

## GPT (KAN-71, in progress)

| Cell | Result | Cost |
|---|---|---|
| Browser Thin | 10/10, 0 invalid replies, 16.8 steps and 61 s per task | $0.70 |
| Desktop Thin | 10/10 | $0.33 |

The first Browser Full run doesn't count: the OpenAI account ran out of credit partway through,
and every later task failed in 0 to 2 steps. It also showed that GPT's critic writes its stop
reason as free text, which made Operon drop the whole verdict. Draft PR #235 keeps the verdict
and sets an unknown label to empty. The Full cells will rerun on #235 once the OpenAI credit is
topped up.

## What the grid says

- **The core sets are saturated.** Nearly every cell is 9 or 10 of 10, so the grid can't yet show
  whether a change to Operon helps. Season 2 (KAN-78) adds harder core sets aimed at 40% to 70%.
- **Hybrid is the hardest category**, and its misses come from writing files and commands by
  typing into a shell window. The planned tool channel (KAN-79) gives the model typed file and
  command tools instead. See the [roadmap](../roadmap.md).
- **Most remaining misses are model misses**, not Operon bugs, after the KAN-75 and #233 fixes.
