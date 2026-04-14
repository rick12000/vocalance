"""Application entry point.

Boot sequence:
    1. Construct all services (subscriptions are registered in __init__)
    2. Call event_bus.start() — flushes any queued events and enables live dispatch
    3. Run async initialization on each service (storage reads, heavy imports)
    4. Start audio capture
    5. Show main window
    6. Await shutdown signal
    7. Tear down in dependency order
"""

import asyncio
import logging
import os
import signal
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import PySide6.QtAsyncio as QtAsyncio
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from vocalance.app.config.app_config import AppInfoConfig, GlobalAppConfig, load_app_config
from vocalance.app.config.logging_config import setup_logging
from vocalance.app.event_bus import EventBus
from vocalance.app.services.audio.dictation_handling.dictation_coordinator import DictationCoordinator
from vocalance.app.services.audio.simple_audio_service import AudioService
from vocalance.app.services.audio.sound_recognizer.streamlined_sound_service import SoundService
from vocalance.app.services.audio.stt.stt_service import SpeechToTextService
from vocalance.app.services.automation_service import AutomationService
from vocalance.app.services.commands.action_map_provider import CommandActionMapProvider
from vocalance.app.services.commands.history import CommandHistoryManager
from vocalance.app.services.commands.management import CommandManagementService
from vocalance.app.services.commands.markov import MarkovCommandService
from vocalance.app.services.commands.parser import CentralizedCommandParser
from vocalance.app.services.deduplication.event_deduplicator import EventDeduplicator
from vocalance.app.services.grid.click_tracker_service import ClickTrackerService
from vocalance.app.services.grid.grid_service import GridService
from vocalance.app.services.mark_service import MarkService
from vocalance.app.services.pause_state_manager import PauseStateManager
from vocalance.app.services.protected_terms_validator import ProtectedTermsValidator
from vocalance.app.services.shutdown_coordinator import ShutdownCoordinator
from vocalance.app.services.storage.llm_model_downloader import LLMModelDownloader
from vocalance.app.services.storage.settings_service import SettingsService
from vocalance.app.services.storage.settings_update_coordinator import SettingsUpdateCoordinator
from vocalance.app.services.storage.storage_service import StorageService
from vocalance.app.ui.qt_main_window import VocalanceMainWindow
from vocalance.app.ui.qt_startup_window import StartupProgressTracker, StartupWindow
from vocalance.app.ui.qt_theme import theme
from vocalance.app.ui.utils.window_icon_manager import WindowIconManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Service container
# ---------------------------------------------------------------------------


@dataclass
class Services:
    """Holds every service once constructed.  All fields are non-Optional because
    the bootstrap guarantees they are set before any consumer runs."""

    storage: StorageService
    validator: ProtectedTermsValidator
    action_map_provider: CommandActionMapProvider
    settings_coordinator: SettingsUpdateCoordinator
    settings: SettingsService
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
    markov_predictor: MarkovCommandService
    # Background tasks spawned during init (warm-starts, etc.)
    _background_tasks: list = field(default_factory=list, repr=False)


# ---------------------------------------------------------------------------
# Construction  (subscriptions registered here, bus not yet started)
# ---------------------------------------------------------------------------


def _construct_services(
    event_bus: EventBus,
    config: GlobalAppConfig,
    gui_loop: asyncio.AbstractEventLoop,
) -> Services:
    """Build every service.  No I/O.  Order matters only for dependency injection."""
    storage = StorageService(config=config)
    validator = ProtectedTermsValidator(config=config, storage=storage)
    validator.setup_invalidation_subscriptions(event_bus)
    action_map_provider = CommandActionMapProvider(storage=storage)

    settings_coordinator = SettingsUpdateCoordinator(event_bus=event_bus, config=config)
    settings = SettingsService(event_bus=event_bus, config=config, storage=storage, coordinator=settings_coordinator)

    grid = GridService(event_bus=event_bus, config=config)
    automation = AutomationService(event_bus=event_bus, config=config)
    click_tracker = ClickTrackerService(event_bus=event_bus, storage=storage)
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
        action_map_provider=action_map_provider,
    )

    audio = AudioService(event_bus=event_bus, config=config, main_event_loop=gui_loop)
    sound_service = SoundService(event_bus=event_bus, config=config, storage=storage)
    stt = SpeechToTextService(event_bus=event_bus, config=config)

    pause_manager = PauseStateManager(event_bus=event_bus)
    history_manager = CommandHistoryManager(storage=storage, protected_terms_validator=validator)
    parser = CentralizedCommandParser(
        event_bus=event_bus,
        app_config=config,
        action_map_provider=action_map_provider,
        history_manager=history_manager,
        deduplicator=EventDeduplicator(window_ms=config.command_parser.duplicate_detection_window_ms),
        pause_state_manager=pause_manager,
    )

    dictation = DictationCoordinator(event_bus=event_bus, config=config, storage=storage, gui_event_loop=gui_loop)
    markov = MarkovCommandService(event_bus=event_bus, config=config, storage=storage)

    _register_settings_callbacks(settings_coordinator, sound_service, audio, markov)

    return Services(
        storage=storage,
        validator=validator,
        action_map_provider=action_map_provider,
        settings_coordinator=settings_coordinator,
        settings=settings,
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
        markov_predictor=markov,
    )


def _register_settings_callbacks(
    coordinator: SettingsUpdateCoordinator,
    sound_service: SoundService,
    audio: AudioService,
    markov: MarkovCommandService,
) -> None:
    reg = coordinator.register_callback
    reg("markov_predictor.enabled", markov.on_enabled_updated)
    reg("markov_predictor.confidence_threshold", markov.on_confidence_threshold_updated)
    reg("sound_recognizer.confidence_threshold", sound_service.on_confidence_threshold_updated)
    reg("sound_recognizer.vote_threshold", sound_service.on_vote_threshold_updated)
    reg("vad.command_silent_chunks_for_end", audio.on_command_silent_chunks_updated)


# ---------------------------------------------------------------------------
# Async initialization  (I/O, heavy imports — bus is live by this point)
# ---------------------------------------------------------------------------


async def _initialize_services(
    services: Services,
    config: GlobalAppConfig,
    shutdown_coordinator: ShutdownCoordinator,
    progress_tracker: StartupProgressTracker,
) -> None:
    """Run async init on every service that needs it, with progress reporting."""

    progress_tracker.start_step(step_name="Initializing storage...")
    progress_tracker.update_status_animated(status="Loading user settings")
    await services.settings.initialize()
    await services.settings.apply_startup_settings_to_config()
    progress_tracker.update_status_animated(status="Initializing click tracking")
    await services.click_tracker.initialize()
    await services.centralized_parser.initialize()
    progress_tracker.complete_step()
    _check_cancellation(shutdown_coordinator)

    progress_tracker.start_step(step_name="Starting audio processing...")
    progress_tracker.update_sub_step(sub_step_name="Loading sound recognition model...")

    yamnet_path = os.path.join(config.storage.sound_model_dir, "yamnet")
    yamnet_ready = (
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
    services._background_tasks.append(
        asyncio.create_task(asyncio.to_thread(services.sound_service.recognizer.warm_start_esc50_samples))
    )
    _check_cancellation(shutdown_coordinator)

    try:
        from moonshine_voice.download_file import get_cache_dir

        model_exists = (lambda c: c.is_dir() and any(c.iterdir()))(get_cache_dir())
    except Exception:
        model_exists = False
    progress_tracker.update_status_animated(
        status="Initializing speech-to-text"
        if model_exists
        else "Fetching Moonshine STT model. This may take several minutes on first use."
    )
    await services.stt.initialize()
    _check_cancellation(shutdown_coordinator)

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
    services.dictation.set_stt_service(services.stt)
    services.audio.set_dictation_chunk_callback(services.dictation.feed_moonshine_audio_chunk)
    _check_cancellation(shutdown_coordinator)

    progress_tracker.update_status_animated(status="Initializing command predictor")
    await services.markov_predictor.initialize()
    progress_tracker.complete_step()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_cancellation(coordinator: ShutdownCoordinator) -> None:
    if coordinator.is_shutdown_requested():
        raise asyncio.CancelledError("Initialization cancelled")


def _validate_critical_assets(config: GlobalAppConfig) -> bool:
    vosk_path = config.asset_paths.get_vosk_model_path()
    if not os.path.exists(vosk_path):
        logger.critical("Vosk model not found: %s – download from https://alphacephei.com/vosk/models", vosk_path)
        return False
    return True


def _setup_signal_handlers(shutdown_coordinator: ShutdownCoordinator) -> QTimer:
    """Bridge OS signals into the Qt/asyncio loop via a polled threading.Event."""
    shutdown_event = threading.Event()

    def _handler(signum, _frame):
        logger.info("Received signal %s – initiating graceful shutdown", signum)
        shutdown_event.set()

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

    timer = QTimer()
    timer.timeout.connect(
        lambda: shutdown_event.is_set()
        and shutdown_coordinator.request_shutdown(reason="System signal received", source="signal_handler")
    )
    timer.start(100)
    return timer


async def _cancel_background_tasks(tasks: list) -> None:
    if not tasks:
        return
    for t in tasks:
        if not t.done():
            t.cancel()
    try:
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=2.0)
    except (asyncio.TimeoutError, Exception) as e:
        logger.warning("Background task cancellation: %s", e)
    finally:
        tasks.clear()


async def _cleanup_services(services: Services, event_bus: EventBus) -> None:
    await _cancel_background_tasks(services._background_tasks)

    services.audio.stop_processing()
    await asyncio.sleep(0.3)

    # Shutdown in dependency order (dependants before dependencies)
    shutdown_order = [
        services.mark,
        services.sound_service,
        services.centralized_parser,
        services.automation,
        services.stt,
        services.dictation,
        services.markov_predictor,
        services.click_tracker,
        services.settings_coordinator,
        services.audio,
        services.storage,
    ]
    for svc in shutdown_order:
        try:
            await svc.shutdown()
        except Exception as e:
            logger.error("Error shutting down %s: %s", type(svc).__name__, e, exc_info=True)

    await event_bus.shutdown()
    logger.info("All services cleaned up")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    logging.getLogger("numba").setLevel(logging.WARNING)

    event_bus: Optional[EventBus] = None
    services: Optional[Services] = None

    try:
        app_info = AppInfoConfig()
        config = load_app_config(app_info=app_info)
        setup_logging(config=config.logging)
        os.makedirs(config.storage.user_data_root, exist_ok=True)

        qt_app = QApplication.instance()
        qt_app.setQuitOnLastWindowClosed(False)

        icon_path = Path(config.asset_paths.icon_path) if config.asset_paths.icon_path else None
        icon_manager = WindowIconManager(icon_path=icon_path)
        if not icon_manager.load_icon():
            logger.warning("Failed to load application icon")
        else:
            icon_manager.apply_to_application(qt_app)

        theme.load_fonts(config.asset_paths.fonts_dir)
        theme._apply_app_palette(qt_app)

        shutdown_coordinator = ShutdownCoordinator()
        _signal_timer = _setup_signal_handlers(shutdown_coordinator)  # noqa: F841

        startup_window = StartupWindow(
            logger=logging.getLogger("StartupWindow"),
            asset_paths_config=config.asset_paths,
            shutdown_coordinator=shutdown_coordinator,
            icon_manager=icon_manager,
        )
        startup_window.show()

        if not _validate_critical_assets(config):
            startup_window.update_progress(0.0, "Critical assets missing. Please check logs.", animate=False)
            await asyncio.sleep(3)
            startup_window.close()
            qt_app.quit()
            return

        # --- Phase 1: construct all services (subscriptions registered, bus still paused)
        event_bus = EventBus()
        gui_loop = asyncio.get_event_loop()
        services = _construct_services(event_bus, config, gui_loop)

        # --- Phase 2: start the bus (flush any queued events, enable live dispatch)
        event_bus.start()

        # --- Phase 3: async initialization (I/O, model loading)
        progress_tracker = StartupProgressTracker(startup_window=startup_window, total_steps=2)
        init_task = asyncio.create_task(_initialize_services(services, config, shutdown_coordinator, progress_tracker))
        shutdown_coordinator.register_initialization_task(init_task)

        try:
            await init_task
            shutdown_coordinator.unregister_initialization_task()
        except asyncio.CancelledError:
            logger.info("Initialization cancelled")
            startup_window.update_progress(0.0, "Startup cancelled by user", animate=False)
            await asyncio.sleep(1)
            startup_window.close()
            if services:
                await _cleanup_services(services, event_bus)
            qt_app.quit()
            return
        except RuntimeError as e:
            logger.critical("Critical initialization error: %s", e)
            startup_window.update_progress(
                0.0, "Initialization failed. Please check your internet connection and try again.", animate=False
            )
            await asyncio.sleep(3)
            startup_window.close()
            if services:
                await _cleanup_services(services, event_bus)
            qt_app.quit()
            return

        # --- Phase 4: start audio capture and show window
        services.audio.start_processing()
        startup_window.update_progress(1.0, "Ready!", animate=False)
        await asyncio.sleep(0.5)

        main_window = VocalanceMainWindow(
            event_bus=event_bus,
            logger=logging.getLogger("MainWindow"),
            config=config,
            services=services,
            icon_manager=icon_manager,
            shutdown_coordinator=shutdown_coordinator,
        )
        main_window.show()
        main_window.raise_()
        main_window.activateWindow()
        startup_window.close_after_initialization()

        logger.info("Application running")
        await shutdown_coordinator.wait()
        logger.info("Shutdown signal received, cleaning up...")

        await _cleanup_services(services=services, event_bus=event_bus)
        logger.info("Application shutdown complete")

    except Exception as e:
        logger.exception("Unexpected error: %s", e)

    qt_app = QApplication.instance()
    if qt_app is not None:
        qt_app.quit()


if __name__ == "__main__":
    _qt_app = QApplication(sys.argv)
    _qt_app.setStyle("Fusion")
    QtAsyncio.run(main(), keep_running=False)
