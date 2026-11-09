"""Leaderboard machinery: frozen task sets, confidence intervals, change labels, drift.

Brought into Operon from the separate model-release-benchmark repo
(vickey-kapoor/model-release-benchmark at ef23db8, MIT, same author) so that the
benchmark is part of Operon rather than a separate harness (KAN-55, KAN-57). Only the
generic machinery came over; that repo's text tasks and direct model-API providers
did not, because every score here comes from Operon itself driving a model.

- ``suite``: hash a task set so a run can prove which tasks it used, and refuse a
  mismatch unless a change is allowed on purpose.
- ``stats``: paired bootstrap 95% intervals over tasks, with Improved / Regressed /
  No clear change labels (Bonferroni-corrected), plus a drift check.
- ``season``: frozen seasons (KAN-58): every task's definition hashed into
  ``evals/seasons/season-<n>.json``, a working / held-out split with the held-out
  tasks kept outside the repo, retiring broken tasks, and the ``--season`` check the
  benchmarks run before they start.

The reports' confidence intervals (KAN-60) build on ``stats``.
"""
