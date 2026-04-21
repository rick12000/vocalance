import asyncio
import importlib.util
import logging
import os
import signal
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType
from typing import List, Optional

import PySide6.QtAsyncio as QtAsyncio
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from vocalance.app.config.app_config import AppInfoConfig, GlobalAppConfig
from vocalance.app.config.logging_config import setup_logging
from vocalance.app.event_bus import EventBus
from vocalance.app.services.audio.dictation_handling.dictation_coordinator import DictationCoordinator
from vocalance.app.services.audio.simple_audio_service import AudioService
from vocalance.app.services.audio.sound_recognizer.streamlined_sound_service import SoundService
from vocalance.app.services.audio.stt.stt_service import SpeechToTextService
from vocalance.app.services.automation_service import AutomationService
from vocalance.app.services.commands.management import CommandManagementService
from vocalance.app.services.commands.parser import CentralizedCommandParser
from vocalance.app.services.grid.click_tracker_service import ClickTrackerService
from vocalance.app.services.grid.grid_service import GridService
from vocalance.app.services.mark_service import MarkService
from vocalance.app.services.pause_state_manager import PauseStateManager
from vocalance.app.services.protected_terms_validator import ProtectedTermsValidator
from vocalance.app.services.storage.llm_model_downloader import LLMModelDownloader
from vocalance.app.services.storage.runtime_configuration import RuntimeConfigurationStore, register_configuration_listeners
from vocalance.app.services.storage.storage_service import StorageService
from vocalance.app.shutdown_coordinator import ShutdownCoordinator
from vocalance.app.ui.qt_main_window import VocalanceMainWindow
from vocalance.app.ui.qt_startup_window import StartupProgressTracker, StartupWindow
from vocalance.app.ui.qt_theme import theme
from vocalance.app.ui.utils.window_icon_manager import WindowIconManager

logger = logging.getLogger(__name__)


@dataclass
class Services:
    """Container for all application services after construction.

    Attributes:
        storage: Persistent storage facade.
        gui_event_loop: Asyncio loop bound to the Qt GUI thread.
        background_tasks: Fire-and-forget tasks owned by startup (e.g. model warmups).
    """

    storage: StorageService
    gui_event_loop: asyncio.AbstractEventLoop
    validator: ProtectedTermsValidator
    runtime_config: RuntimeConfigurationStore
    grid: GridService
    automation: AutomationService
    click_tracker: ClickTrackerService
    mark: MarkService
    command_management: CommandManagementService
    audio: AudioService
    sound_service: SoundService
    stt: SpeechToTextService
    pause_state_manager: PauseStateManager
    centralized_parser: CentralizedCommandParser
    dictation: DictationCoordinator
    background_tasks: List[asyncio.Task] = field(default_factory=list, repr=False)


def construct_services(
    event_bus: EventBus,
    config: GlobalAppConfig,
    gui_loop: asyncio.AbstractEventLoop,
) -> Services:
    """Build the default service graph and wire configuration listeners."""
    storage = StorageService(config=config)
    runtime_config = RuntimeConfigurationStore(event_bus=event_bus, config=config, storage=storage)
    validator = ProtectedTermsValidator(config=config, storage=storage)
    validator.setup_invalidation_subscriptions(event_bus)

    grid = GridService(event_bus=event_bus, config=config)
    automation = AutomationService(event_bus=event_bus, config=config)
    click_tracker = ClickTrackerService(
        event_bus=event_bus,
        storage=storage,
        gui_event_loop=gui_loop,
        ui_refresh_debounce_s=config.grid.click_history_ui_refresh_debounce_s,
        persist_debounce_s=config.grid.click_history_persist_debounce_s,
    )
    mark = MarkService(
        event_bus=event_bus,
        config=config,
        storage=storage,
        protected_terms_validator=validator,
    )
    command_management = CommandManagementService(
        event_bus=event_bus,
        storage=storage,
        protected_terms_validator=validator,
    )

    stt = SpeechToTextService(event_bus=event_bus, config=config)

    dictation = DictationCoordinator(
        event_bus=event_bus,
        config=config,
        storage=storage,
        gui_event_loop=gui_loop,
        stt_service=stt,
    )

    audio = AudioService(
        event_bus=event_bus,
        config=config,
        main_event_loop=gui_loop,
        dictation=dictation,
    )
    sound_service = SoundService(event_bus=event_bus, config=config, storage=storage)

    pause_manager = PauseStateManager(event_bus=event_bus)
    parser = CentralizedCommandParser(
        event_bus=event_bus,
        app_config=config,
        storage=storage,
        pause_state_manager=pause_manager,
    )

    register_configuration_listeners(
        runtime_config,
        sound_service=sound_service,
        audio_service=audio,
        llm_service=dictation.llm_service,
    )

    return Services(
        storage=storage,
        gui_event_loop=gui_loop,
        validator=validator,
        runtime_config=runtime_config,
        grid=grid,
        automation=automation,
        click_tracker=click_tracker,
        mark=mark,
        command_management=command_management,
        audio=audio,
        sound_service=sound_service,
        stt=stt,
        pause_state_manager=pause_manager,
        centralized_parser=parser,
        dictation=dictation,
    )


def moonshine_model_cache_ready() -> bool:
    """Return True if the Moonshine cache directory exists and is non-empty."""
    if importlib.util.find_spec("moonshine_voice.download_file") is None:
        return False
    from moonshine_voice.download_file import get_cache_dir

    cache_dir = get_cache_dir()
    return cache_dir.is_dir() and any(cache_dir.iterdir())


async def initialize_services(
    services: Services,
    config: GlobalAppConfig,
    shutdown_coordinator: ShutdownCoordinator,
    progress_tracker: StartupProgressTracker,
) -> None:
    """Run staged startup: storage, audio stack, STT, dictation, and LLM assets."""
    progress_tracker.start_step(step_name="Initializing storage...")
    progress_tracker.update_status_animated(status="Loading user settings")
    await services.runtime_config.initialize()
    progress_tracker.update_status_animated(status="Initializing click tracking")
    await services.click_tracker.initialize()
    await services.centralized_parser.initialize()
    progress_tracker.complete_step()
    check_initialization_cancelled(shutdown_coordinator)

    progress_tracker.start_step(step_name="Starting audio processing...")
    progress_tracker.update_sub_step(sub_step_name="Loading sound recognition model...")

    yamnet_path = os.path.join(config.storage.sound_model_dir, "yamnet")
    yamnet_ready: bool = (
        os.path.exists(yamnet_path)
        and os.path.exists(os.path.join(yamnet_path, "saved_model.pb"))
        and os.path.exists(os.path.join(yamnet_path, "variables"))
    )
    progress_tracker.update_status_animated(
        status="Initializing sound recognition"
        if yamnet_ready
        else "Loading YAMNet model. This should take 1-2 minutes on first use."
    )
    await services.sound_service.initialize()
    services.background_tasks.append(
        asyncio.create_task(asyncio.to_thread(services.sound_service.recognizer.warm_start_esc50_samples))
    )
    check_initialization_cancelled(shutdown_coordinator)

    model_exists = moonshine_model_cache_ready()
    progress_tracker.update_status_animated(
        status="Initializing speech-to-text"
        if model_exists
        else "Fetching Moonshine STT model. This may take several minutes on first use."
    )
    await services.stt.initialize()
    check_initialization_cancelled(shutdown_coordinator)

    progress_tracker.update_sub_step(sub_step_name="Preparing dictation system...")
    allow = config.local_llm_allowlist
    spec = allow.artifact_for(config.llm.selected_model_id) or allow.artifact_for(allow.default_id)
    if spec:
        downloader = LLMModelDownloader(config)
        if not downloader.model_bundle_complete(spec.gguf_filenames):
            progress_tracker.update_sub_step(
                sub_step_name="Fetching default local LLM (~2–4 GB). First launch may take several minutes.",
                progress=0.35,
            )
            if not await downloader.download_model_bundle(repo_id=spec.repo_id, filenames=list(spec.gguf_filenames)):
                raise RuntimeError("Critical asset download failed: LLM model")

    progress_tracker.update_sub_step(sub_step_name="Initializing dictation", progress=0.55)
    if not await services.dictation.initialize():
        raise RuntimeError("Critical dictation initialization failed")
    check_initialization_cancelled(shutdown_coordinator)

    progress_tracker.complete_step()


def check_initialization_cancelled(coordinator: ShutdownCoordinator) -> None:
    """Raise ``CancelledError`` if shutdown was requested during startup."""
    if coordinator.is_shutdown_requested():
        raise asyncio.CancelledError("Initialization cancelled")


def validate_critical_assets(config: GlobalAppConfig) -> bool:
    """Return False if required on-disk assets (e.g. Vosk) are missing."""
    vosk_path = config.asset_paths.get_vosk_model_path()
    if not os.path.exists(vosk_path):
        logger.critical(
            "Vosk model not found: %s – download from https://alphacephei.com/vosk/models",
            vosk_path,
        )
        return False
    return True


def setup_signal_handlers(shutdown_coordinator: ShutdownCoordinator) -> QTimer:
    """Register SIGINT/SIGTERM handlers and a Qt timer that forwards them to ``ShutdownCoordinator``."""
    shutdown_event = threading.Event()

    def on_os_signal(signum: int, frame: Optional[FrameType]) -> None:
        del frame
        logger.info("Received signal %s – initiating graceful shutdown", signum)
        shutdown_event.set()

    signal.signal(signal.SIGINT, on_os_signal)
    signal.signal(signal.SIGTERM, on_os_signal)

    timer = QTimer()
    timer.timeout.connect(
        lambda: shutdown_event.is_set()
        and shutdown_coordinator.request_shutdown(reason="System signal received", source="signal_handler")
    )
    timer.start(100)
    shutdown_coordinator.attach_signal_poll_timer(timer)
    return timer


async def cancel_background_tasks(tasks: List[asyncio.Task]) -> None:
    """Cancel tracked background tasks and log failures from ``gather``."""
    if not tasks:
        return
    for task in tasks:
        if not task.done():
            task.cancel()
    try:
        results: List[object | BaseException] = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=2.0,
        )
    except asyncio.TimeoutError:
        logger.warning("Background tasks did not settle within 2s after cancellation")
        tasks.clear()
        return
    for result in results:
        if isinstance(result, asyncio.CancelledError):
            continue
        if isinstance(result, BaseException):
            logger.error("Background task failed", exc_info=(type(result), result, result.__traceback__))
    tasks.clear()


async def cleanup_services(services: Services, event_bus: EventBus) -> None:
    """Tear down services in dependency order and shut down the event bus."""
    await cancel_background_tasks(services.background_tasks)

    services.audio.stop_processing()
    await services.audio.wait_for_capture_pipeline_idle()

    shutdown_order = [
        services.mark,
        services.grid,
        services.command_management,
        services.sound_service,
        services.centralized_parser,
        services.pause_state_manager,
        services.automation,
        services.dictation,
        services.stt,
        services.click_tracker,
        services.runtime_config,
        services.audio,
        services.validator,
        services.storage,
    ]
    errors: List[Exception] = []
    for svc in shutdown_order:
        try:
            await svc.shutdown()
        except Exception as e:
            logger.error("Error shutting down %s: %s", type(svc).__name__, e, exc_info=True)
            errors.append(e)

    await event_bus.shutdown()

    from vocalance.app.services.commands.utilities.input_executor import shared_input_executor

    shared_input_executor.shutdown(wait=True)

    logger.info("All services cleaned up")

    if errors:
        raise ExceptionGroup("One or more services failed during shutdown", errors)


async def abort_startup(
    startup_window: StartupWindow,
    message: str,
    delay_s: float,
    services: Optional[Services],
    event_bus: Optional[EventBus],
    qt_app: QApplication,
) -> None:
    """Show ``message``, optionally run cleanup, and quit the application."""
    startup_window.update_progress(0.0, message, animate=False)
    await asyncio.sleep(delay_s)
    startup_window.close()
    try:
        if services is not None and event_bus is not None:
            await cleanup_services(services, event_bus)
    finally:
        qt_app.quit()


async def main() -> None:
    """QtAsyncio entry: configure logging, run startup, then block until shutdown."""
    logging.getLogger("numba").setLevel(logging.WARNING)

    event_bus: Optional[EventBus] = None
    services: Optional[Services] = None
    qt_app = QApplication.instance()
    if qt_app is None:
        raise RuntimeError("QApplication instance is missing; create it before calling main()")

    shutdown_coordinator = ShutdownCoordinator(asyncio.get_running_loop())
    setup_signal_handlers(shutdown_coordinator)

    try:
        AppInfoConfig()
        config = GlobalAppConfig()
        setup_logging(config=config.logging)
        os.makedirs(config.storage.user_data_root, exist_ok=True)

        qt_app.setQuitOnLastWindowClosed(False)

        icon_path = Path(config.asset_paths.icon_path) if config.asset_paths.icon_path else None
        icon_manager = WindowIconManager(icon_path=icon_path)
        if not icon_manager.load_icon():
            logger.warning("Failed to load application icon")
        else:
            icon_manager.apply_to_application(qt_app)

        theme.load_fonts(config.asset_paths.fonts_dir)
        theme.load_stylesheet()
        theme.apply_stylesheet(qt_app)
        qt_app.setFont(theme.get_font(size="medium", weight="regular"))

        startup_window = StartupWindow(
            logger=logging.getLogger("StartupWindow"),
            asset_paths_config=config.asset_paths,
            shutdown_coordinator=shutdown_coordinator,
            icon_manager=icon_manager,
        )
        startup_window.show()

        if not validate_critical_assets(config):
            startup_window.update_progress(0.0, "Critical assets missing. Please check logs.", animate=False)
            await asyncio.sleep(3)
            startup_window.close()
            qt_app.quit()
            return

        event_bus = EventBus()
        gui_loop = asyncio.get_running_loop()
        services = construct_services(event_bus, config, gui_loop)

        event_bus.start(gui_loop)

        progress_tracker = StartupProgressTracker(startup_window=startup_window, total_steps=2)
        init_task = asyncio.create_task(initialize_services(services, config, shutdown_coordinator, progress_tracker))
        shutdown_coordinator.register_initialization_task(init_task)

        try:
            await init_task
            shutdown_coordinator.unregister_initialization_task()
        except asyncio.CancelledError:
            logger.info("Initialization cancelled")
            await abort_startup(
                startup_window,
                "Startup cancelled by user",
                1.0,
                services,
                event_bus,
                qt_app,
            )
            return
        except RuntimeError as e:
            logger.critical("Critical initialization error: %s", e)
            await abort_startup(
                startup_window,
                "Initialization failed. Please check your internet connection and try again.",
                3.0,
                services,
                event_bus,
                qt_app,
            )
            return

        services.audio.start_processing()
        startup_window.update_progress(1.0, "Ready!", animate=False)
        await asyncio.sleep(0.5)

        main_window = VocalanceMainWindow(
            event_bus=event_bus,
            logger=logging.getLogger("MainWindow"),
            config=config,
            icon_manager=icon_manager,
            shutdown_coordinator=shutdown_coordinator,
        )
        await services.click_tracker.publish_click_history_snapshot()
        main_window.show()
        main_window.raise_()
        main_window.activateWindow()
        startup_window.close_after_initialization()

        logger.info("Application running")
        await shutdown_coordinator.wait()
        logger.info("Shutdown signal received, cleaning up...")

        await cleanup_services(services=services, event_bus=event_bus)
        logger.info("Application shutdown complete")

    except ExceptionGroup as eg:
        logger.error("Multiple errors during application lifecycle", exc_info=eg)
        raise
    except Exception:
        logger.exception("Unexpected error during application run")
        raise
    finally:
        app_instance = QApplication.instance()
        if app_instance is not None:
            app_instance.quit()


if __name__ == "__main__":
    qt_application = QApplication(sys.argv)
    qt_application.setStyle("Fusion")
    QtAsyncio.run(main(), keep_running=False)
