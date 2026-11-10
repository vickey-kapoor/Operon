"""Live Hybrid benchmark: the tasks in ``evals/hybrid_tasks.py``, a desktop app and the browser in one task.

Same runner, grading and options as ``python -m evals.live_bench``; this only defaults
to ``--mode hybrid``. Like the desktop benchmark it drives the real mouse and keyboard
through the ``/desktop`` routes, so run it on a Windows session you are not using (or
the "Live desktop benchmark" workflow with category hybrid, ``/bench hybrid``). It
serves the local site on 127.0.0.1 itself, so it must run on the machine the API drives.
"""

from __future__ import annotations

import sys

from evals.live_bench import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:], default_mode="hybrid"))
