"""Live desktop benchmark: the tasks in ``tests/eval/desktop_tasks.py``.

Same runner, grading and options as ``python -m tests.eval.live_bench``; this
only defaults to ``--mode desktop``. It drives the real mouse and keyboard, so
run it on a Windows session you are not using (or the "Live desktop
benchmark" GitHub Actions workflow). See ``tests/eval/live_bench.py``.
"""

from __future__ import annotations

import sys

from tests.eval.live_bench import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:], default_mode="desktop"))
