import asyncio
import gc
import os
import subprocess
import sys
import threading
from pathlib import Path

import psutil
import pytest

from iris.app.config.app_config import GlobalAppConfig
from iris.app.event_bus import EventBus

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def app_config():
    return GlobalAppConfig()


@pytest.mark.asyncio
@pytest.mark.memory
@pytest.mark.slow
async def test_process_exit_releases_memory(app_config):
    """Test that memory is fully released when the application process exits."""
    main_script_path = Path(__file__).parent.parent.parent / "iris" / "main.py"

    test_process = psutil.Process(os.getpid())
    baseline_uss = test_process.memory_full_info().uss / (1024 * 1024)

    process = subprocess.Popen(
        [sys.executable, str(main_script_path)],
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )

    await asyncio.sleep(120.0)

    if process.poll() is not None:
        raise RuntimeError("Process exited prematurely")

    app_uss = psutil.Process(process.pid).memory_full_info().uss / (1024 * 1024)
    print(f"\nApp memory: {app_uss:.1f} MB")

    process.kill()
    process.wait()

    await asyncio.sleep(4.0)
    gc.collect()

    final_uss = test_process.memory_full_info().uss / (1024 * 1024)
    memory_delta = final_uss - baseline_uss

    print(f"Memory delta after exit: {memory_delta:+.1f} MB")

    assert abs(memory_delta) < 50.0, f"Memory not released: {memory_delta:+.1f} MB retained"


@pytest.mark.asyncio
@pytest.mark.memory
@pytest.mark.slow
async def test_repeated_initialization_memory(app_config):
    """Test that repeated initialization cycles don't accumulate memory."""
    from iris.app.services.audio.sound_recognizer.streamlined_sound_service import StreamlinedSoundService
    from iris.app.services.audio.stt.stt_service import SpeechToTextService
    from iris.app.services.automation_service import AutomationService
    from iris.app.services.grid.grid_service import GridService
    from iris.app.services.storage.storage_service import StorageService
    from iris.main import _cleanup_services

    test_process = psutil.Process(os.getpid())
    memory_snapshots = []

    baseline_uss = test_process.memory_full_info().uss / (1024 * 1024)
    memory_snapshots.append(baseline_uss)

    for cycle in range(3):
        event_bus = EventBus()
        gui_event_loop = asyncio.new_event_loop()

        gui_thread = threading.Thread(
            target=lambda loop=gui_event_loop: (asyncio.set_event_loop(loop), loop.run_forever()),
            daemon=False,
            name=f"GUIEventLoop-{cycle}",
        )
        gui_thread.start()

        gui_event_loop.call_soon_threadsafe(lambda loop=gui_event_loop, bus=event_bus: loop.create_task(bus.start_worker()))

        await asyncio.sleep(0.3)

        services = {
            "grid": GridService(event_bus, app_config),
            "automation": AutomationService(event_bus, app_config),
            "unified_storage": StorageService(config=app_config),
            "gui_thread": gui_thread,
        }
        services["grid"].setup_subscriptions()
        services["automation"].setup_subscriptions()

        services["sound_service"] = StreamlinedSoundService(event_bus, app_config, services["unified_storage"])
        await services["sound_service"].initialize()

        services["stt"] = SpeechToTextService(event_bus, app_config)
        services["stt"].initialize_engines()
        services["stt"].setup_subscriptions()

        await asyncio.sleep(1.0)

        await _cleanup_services(services, event_bus, gui_event_loop, gui_thread)
        del services, event_bus, gui_event_loop

        for _ in range(3):
            gc.collect()
            await asyncio.sleep(0.1)

        memory_snapshots.append(test_process.memory_full_info().uss / (1024 * 1024))

    first_cycle_overhead = memory_snapshots[1] - memory_snapshots[0]
    cumulative_growth = memory_snapshots[-1] - memory_snapshots[1]

    print(f"\nFirst cycle: +{first_cycle_overhead:.1f} MB (ML libraries)")
    print(f"Cumulative growth: {cumulative_growth:+.1f} MB")

    assert cumulative_growth < 50.0, f"Cumulative leak detected: {cumulative_growth:.1f} MB"
