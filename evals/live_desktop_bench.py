"""Live desktop benchmark: the tasks in ``evals/desktop_tasks.py``.

Same runner, grading and options as ``python -m evals.live_bench``; this
only defaults to ``--mode desktop``. It drives the real mouse and keyboard, so
run it on a Windows session you are not using (or the "Live desktop
benchmark" GitHub Actions workflow). See ``evals/live_bench.py``.
"""

from __future__ import annotations

import sys

from evals.live_bench import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:], default_mode="desktop"))
