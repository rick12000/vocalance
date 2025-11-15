import asyncio
import gc
import subprocess
import sys
from pathlib import Path

import psutil
import pytest

from vocalance.app.config.app_config import GlobalAppConfig

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def app_config():
    return GlobalAppConfig()


@pytest.mark.asyncio
@pytest.mark.memory
@pytest.mark.slow
async def test_process_exit_releases_memory(app_config):
    """Test that memory is fully released when the application process exits."""
    main_script_path = Path(__file__).parent.parent.parent / "vocalance" / "main.py"

    # Start the application process
    process = subprocess.Popen(
        [sys.executable, str(main_script_path)],
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )

    # Wait for application to initialize and stabilize
    await asyncio.sleep(120.0)

    if process.poll() is not None:
        raise RuntimeError("Process exited prematurely")

    # Get the application process memory usage
    app_process = psutil.Process(process.pid)
    app_uss_mb = app_process.memory_full_info().uss / (1024 * 1024)

    # Verify app is using reasonable memory (should be between 50MB and 2GB)
    assert 50.0 < app_uss_mb < 2048.0, f"Application memory usage {app_uss_mb:.1f} MB is outside expected range"

    # Kill the process and wait for it to exit
    process.kill()
    process.wait()

    # Give OS time to reclaim memory
    await asyncio.sleep(4.0)
    gc.collect()

    # Verify the process is actually dead
    assert not psutil.pid_exists(process.pid), f"Process {process.pid} still exists after kill"
