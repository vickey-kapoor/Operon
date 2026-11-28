# The "Real check:" line (KAN-298)

Every PR the gate merges needs a "Real check:" line in its description. It is a link to a real $0 run of the
change, plus a few words on what the run showed.

A real run is one of:

- a real end-to-end test run (desktop e2e, browser e2e, Windows e2e) with a scripted model;
- a GitHub dry run of the workflow or script you changed;
- a replay of real data (a saved run, a recording) through the changed code.

It is never a paid run. Unit tests and reviews alone do not count. The gate checks only that the line and a link
are there; the verifier opens the link and checks that the run really shows the change.

## Where to put it

On its own top-level line: starts at column 0, no indent, no quote mark, at most one list marker or heading mark
before the words. Put it near the top, before any code block, quote or raw HTML in the description, because the
gate stops reading at the first of those. A code span such as `<path>` is fine. An HTML comment before the words
does not count.

## Good examples

- `Real check: https://github.com/vickey-kapoor/Operon/actions/runs/<id> (desktop e2e job) typed the text into xterm and the file matched.`
- `Real check: https://github.com/vickey-kapoor/Operon/actions/runs/<id> (dry run of the workflow) printed the expected labels for the PR.`
- `Real check: https://github.com/vickey-kapoor/Operon/actions/runs/<id> replayed the saved run through the new code and the report matched.`

## Lines that do not count

- `Real check: I ran the unit tests and they pass.` There is no link, so the gate cannot tell a real run from a claim.
- A "Real check:" line inside a code block. The gate treats code as literal text and stops there.
- A "Real check:" line placed after a code block in a Proof section. The gate stops reading at the first code block,
  so a line after it is never seen. Move it above.

## If it is missing

The gate does not merge the PR. On each sweep it leaves one note per head commit saying the line is missing (a
new commit gets a new note). Add the line to the description; the next sweep can then merge the PR once its other
checks are met. Fixing the line does not skip the verifier: it still opens the link.
