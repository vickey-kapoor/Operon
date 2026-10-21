"""Live smoke run: the five ``smoke`` tasks of the live benchmark.

Same runner, grading and options as ``python -m tests.eval.live_bench``; this
only defaults to ``--suite smoke``. See ``tests/eval/live_bench.py``.
"""

from __future__ import annotations

import sys

from tests.eval.live_bench import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:], default_suite="smoke"))
