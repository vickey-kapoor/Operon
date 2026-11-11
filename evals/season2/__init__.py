"""Season 2 task sets (KAN-78): harder Browser, Desktop and Hybrid core sets.

Season 1's core sets saturated (9 or 10 out of 10 in most grid cells), so they can no
longer show whether a change to Operon helps. Season 2's core sets are built to land
at about 40 to 70% for the reference model (Gemini 3.5 Flash) through what today's
models still miss, never through trick wording:

- **dense screens**: a 36-row invoice list over four pages, a 33-person directory, a
  week of room bookings, a 60-line log, a 24-row CSV;
- **exact data copying**: phone numbers, invoice numbers, payment references and
  amounts copied character for character, next to near-duplicates (Chen Wei and Chen
  Wen, Acme Supplies and Acme Logistics, Marta Kowalczyk and Marta Kowalski);
- **choices under constraints**: the smallest free room with the right equipment, as
  many items as a budget allows;
- **more apps per task**: three new local apps (``apps.py``: Northwind Finance,
  People and Rooms) beside the Season 1 shop, mail, account and forms;
- **two-sided safety**: each safety task passes only if the asked-for work is done
  *and* the planted instruction or the nearby overreach is not.

The modules:

- ``apps.py``: the three new apps and the Season 2 workspace page (a fixture file:
  every Season 2 browser and hybrid task's fingerprint hashes it);
- ``browser.py``, ``desktop.py``, ``hybrid.py``: the ten tasks of each category's
  Season 2 core set (``evals/categories.py`` ``CORE2``).

Season 1's tasks, apps and fixture files are untouched, so Season 1 still checks
and its scores stay comparable. Season 2 tasks live only here, never in the Season 1
pools (``APP_TASKS``, ``LONG_TASKS``, ``DESKTOP_TASKS``, ``HYBRID_TASKS``).
"""
