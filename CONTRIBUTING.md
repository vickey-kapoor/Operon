# Contributing to Operon

Thanks for taking a look. Operon is early — the architecture is still moving, so
open an issue before starting anything large.

## Setup

See [Quick Start](README.md#quick-start). Then confirm your environment is sane:

```bash
GEMINI_API_KEY=fake-test-key python -m pytest tests -q
ruff check src tests --select E,F,W,I --ignore E501
```

554 tests should pass in about 30 seconds with no network access. If they don't,
that's a bug in the setup instructions — please report it.

## Before you open a PR

- `ruff check src tests --select E,F,W,I --ignore E501` is clean
- The offline suite passes
- New behaviour has a test that runs **without** a live server or a real screen.
  If you genuinely can't test it offline, mark it `@pytest.mark.live_server` and
  say so in the PR — but that code path won't be covered by CI, so keep it thin.

CI runs the offline suite on Python 3.11 and 3.14 plus ruff. It does not run the
`live_server` suite.

## Reporting bugs

Agent failures are hard to debug from a description alone. Operon writes
everything needed to `.var/runs/<run_id>/` — screenshots, model I/O, policy
decisions, execution traces. Please include:

- The instruction you gave and the environment (browser or desktop)
- The backend and model (`OPERON_BROWSER_BACKEND`, `OPERON_BROWSER_MODEL`, etc.)
- The run ID, and the relevant slice of the run artifacts

**Screenshots in run artifacts capture your whole screen.** Check them for
credentials, tokens, and personal information before attaching anything.

"The agent did the wrong thing" is a valid and useful bug report. So is "the
agent was correct but took 70 seconds" — perception latency is a known weak
point and data on where time goes is welcome.

## Things worth knowing

- **Vision-only is the constraint, not an implementation detail.** Proposals that
  reach for the DOM, accessibility tree, or app-specific APIs to make targeting
  easier defeat the point. Improving *perception* is the intended path.
- **Deterministic rules run before the model.** If something can be handled by
  `agent/policy/rules.py` without an LLM call, it should be.
- **Experimental work goes behind an env flag, defaulting to off.** See the
  `OPERON_BESTOFN_*`, `OPERON_GROUNDER`, and `OPERON_TRUST_*` flags for the
  pattern, and `docs/rfcs/` for the shape of a larger proposal.
- **`.env.example` is a contract.** Every variable in it must be read by code in
  `src/`. Don't document configuration that isn't wired up.

## Security

Don't file security issues publicly — see [SECURITY.md](SECURITY.md).
