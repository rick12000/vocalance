import os
import sys
import warnings

import PySide6.QtAsyncio as QtAsyncio
from PySide6.QtWidgets import QApplication

from vocalance.qt_main import main

if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.pop("SSLKEYLOGFILE", None)

if __name__ == "__main__":
    # Known PySide6.QtAsyncio bug: it leaks the shutdown_asyncgens coroutine
    # during its own teardown, then closes the loop before asyncio.Runner can
    # drain it. Both produce spurious noise unrelated to application state.
    warnings.filterwarnings("ignore", message=".*shutdown_asyncgens.*", category=RuntimeWarning)

    _app = QApplication(sys.argv)
    _app.setStyle("Fusion")
    try:
        QtAsyncio.run(main(), keep_running=False)
    except RuntimeError as exc:
        if "Event loop is closed" not in str(exc):
            raise

    sys.exit(0)
