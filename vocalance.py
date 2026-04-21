import sys
import warnings

import PySide6.QtAsyncio as QtAsyncio
from PySide6.QtWidgets import QApplication

from vocalance.qt_main import main

if __name__ == "__main__":
    # Suppress the RuntimeWarning emitted by Python when QtAsyncio leaks the
    # shutdown_asyncgens coroutine during its own teardown (known QtAsyncio bug).
    warnings.filterwarnings("ignore", message=".*shutdown_asyncgens.*", category=RuntimeWarning)

    _app = QApplication(sys.argv)
    _app.setStyle("Fusion")
    try:
        QtAsyncio.run(main(), keep_running=False)
    except RuntimeError as e:
        # PySide6.QtAsyncio closes its event loop before Python's asyncio.Runner
        # teardown runs shutdown_asyncgens(), producing a spurious "Event loop is closed"
        # RuntimeError. All application cleanup has already completed at this point.
        if "Event loop is closed" not in str(e):
            raise
    sys.exit(0)
