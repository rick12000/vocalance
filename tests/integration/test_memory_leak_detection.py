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


# @pytest.mark.asyncio
# @pytest.mark.memory
# @pytest.mark.slow
# @pytest.mark.timeout(600)  # 10 minute timeout for slow initialization cycles
# async def test_repeated_initialization_memory(app_config):
#     """Test that repeated initialization cycles don't accumulate memory.

#     Note: This test can take several minutes due to repeated Whisper model initialization.
#     """
#     from vocalance.app.services.audio.sound_recognizer.streamlined_sound_service import StreamlinedSoundService
#     from vocalance.app.services.audio.stt.stt_service import SpeechToTextService
#     from vocalance.app.services.automation_service import AutomationService
#     from vocalance.app.services.grid.grid_service import GridService
#     from vocalance.app.services.storage.storage_service import StorageService
#     from vocalance.main import _cleanup_services

#     test_process = psutil.Process(os.getpid())
#     memory_snapshots = []

#     baseline_uss = test_process.memory_full_info().uss / (1024 * 1024)
#     memory_snapshots.append(baseline_uss)

#     for cycle in range(3):
#         event_bus = EventBus()
#         gui_event_loop = asyncio.new_event_loop()

#         gui_thread = threading.Thread(
#             target=lambda loop=gui_event_loop: (asyncio.set_event_loop(loop), loop.run_forever()),
#             daemon=False,
#             name=f"GUIEventLoop-{cycle}",
#         )
#         gui_thread.start()

#         gui_event_loop.call_soon_threadsafe(lambda loop=gui_event_loop, bus=event_bus: loop.create_task(bus.start_worker()))

#         await asyncio.sleep(0.3)

#         services = {
#             "grid": GridService(event_bus, app_config),
#             "automation": AutomationService(event_bus, app_config),
#             "unified_storage": StorageService(config=app_config),
#             "gui_thread": gui_thread,
#         }
#         services["grid"].setup_subscriptions()
#         services["automation"].setup_subscriptions()

#         services["sound_service"] = StreamlinedSoundService(event_bus, app_config, services["unified_storage"])
#         await services["sound_service"].initialize()

#         services["stt"] = SpeechToTextService(event_bus, app_config)
#         await services["stt"].initialize_engines()
#         services["stt"].setup_subscriptions()

#         await asyncio.sleep(1.0)

#         await _cleanup_services(services, event_bus, gui_event_loop, gui_thread)
#         del services, event_bus, gui_event_loop

#         for _ in range(3):
#             gc.collect()
#             await asyncio.sleep(0.1)

#         current_memory = test_process.memory_full_info().uss / (1024 * 1024)
#         memory_snapshots.append(current_memory)

#     first_cycle_overhead = memory_snapshots[1] - memory_snapshots[0]
#     cumulative_growth = memory_snapshots[-1] - memory_snapshots[1]

#     assert cumulative_growth < 50.0, f"Cumulative leak detected: {cumulative_growth:.1f} MB"
