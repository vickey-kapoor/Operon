"""Live smoke run: the five ``smoke`` tasks of the live benchmark.

Same runner, grading and options as ``python -m evals.live_bench``; this
only defaults to ``--suite smoke``. See ``evals/live_bench.py``.
"""

from __future__ import annotations

import sys

from evals.live_bench import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:], default_suite="smoke"))
