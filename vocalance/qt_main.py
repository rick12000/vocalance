import asyncio
import gc
import logging
import os
import signal
import sys
import threading
from typing import Any, Dict, Optional

import PySide6.QtAsyncio as QtAsyncio
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from vocalance.app.config.app_config import AppInfoConfig, GlobalAppConfig, load_app_config
from vocalance.app.config.logging_config import setup_logging
from vocalance.app.event_bus import EventBus
from vocalance.app.services.shutdown_coordinator import ShutdownCoordinator
from vocalance.app.services.storage.llm_model_downloader import LLMModelDownloader
from vocalance.app.ui.qt_main_window import VocalanceMainWindow
from vocalance.app.ui.qt_startup_window import StartupProgressTracker, StartupWindow
from vocalance.app.ui.qt_theme import theme
from vocalance.app.ui.utils.window_icon_manager import WindowIconManager

logger = logging.getLogger(__name__)


class FastServiceInitializer:
    """Manages parallel service initialization with thread-safe access and shutdown support.

    Coordinates the staged initialization of application services including storage, audio
    processing, speech-to-text engines, and UI components. Provides thread-safe access to
    the services dictionary during concurrent initialization and supports graceful cancellation
    through the shutdown coordinator.
    """

    def __init__(
        self,
        event_bus: EventBus,
        config: GlobalAppConfig,
        gui_loop: asyncio.AbstractEventLoop,
        shutdown_coordinator: Optional[ShutdownCoordinator] = None,
    ) -> None:
        """Initialize the service initializer.

        Args:
            event_bus: Central event bus for inter-service communication.
            config: Global application configuration containing all settings.
            gui_loop: Asyncio event loop dedicated to GUI operations.
            shutdown_coordinator: Optional coordinator for handling shutdown requests during init.
        """
        self.event_bus: EventBus = event_bus
        self.config: GlobalAppConfig = config
        self.gui_loop: asyncio.AbstractEventLoop = gui_loop
        self.services: Dict[str, Any] = {}
        self.shutdown_coordinator: Optional[ShutdownCoordinator] = shutdown_coordinator
        self._services_lock: threading.RLock = threading.RLock()
        self._background_tasks: list[asyncio.Task] = []

    async def initialize_all(self, progress_tracker: StartupProgressTracker) -> Dict[str, Any]:
        """Initialize all non-UI services in staged parallel batches with progress updates.

        Executes initialization in three sequential stages: core services (grid, automation),
        storage services (settings, commands, marks, click tracking), and audio services
        (capture, sound recognition, STT, dictation, command parsing). Within each stage,
        services are initialized concurrently where possible. Checks for shutdown requests
        between stages to support cancellation.

        Args:
            progress_tracker: Tracks and reports initialization progress to the UI.

        Returns:
            Thread-safe dictionary mapping service names to initialized instances.
        """
        progress_tracker.start_step(step_name="Starting core services...")
        progress_tracker.update_sub_step(sub_step_name="Initializing grid service...")
        self._init_core_services()
        progress_tracker.complete_step()
        self._check_cancellation()

        progress_tracker.start_step(step_name="Initializing storage...")
        progress_tracker.update_sub_step(sub_step_name="Setting up unified storage...")
        await self._init_storage_services(progress_tracker=progress_tracker)
        progress_tracker.complete_step()
        self._check_cancellation()

        progress_tracker.start_step(step_name="Starting audio processing...")
        progress_tracker.update_sub_step(sub_step_name="Loading audio engines...")
        await self._init_audio_services(progress_tracker=progress_tracker)
        progress_tracker.complete_step()

        with self._services_lock:
            return dict(self.services)

    def _check_cancellation(self) -> None:
        """Check for shutdown request and abort initialization if detected.

        Raises:
            asyncio.CancelledError: When shutdown coordinator indicates shutdown was requested.
        """
        if self.shutdown_coordinator and self.shutdown_coordinator.is_shutdown_requested():
            logger.info("Shutdown detected during initialization - cancelling")
            raise asyncio.CancelledError("Initialization cancelled due to shutdown request")

    async def cancel_background_tasks(self) -> None:
        """Cancel all background tasks spawned during initialization.

        Cancels and awaits completion of background tasks with timeout protection
        to prevent hanging on stuck tasks during shutdown.
        """
        if not self._background_tasks:
            logger.debug("No background tasks to cancel")
            return

        logger.info(f"Cancelling {len(self._background_tasks)} background initialization tasks")

        # Cancel all tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()

        # Wait for cancellation to complete with timeout
        try:
            await asyncio.wait_for(asyncio.gather(*self._background_tasks, return_exceptions=True), timeout=2.0)
            logger.debug("All background tasks cancelled successfully")
        except asyncio.TimeoutError:
            logger.warning("Background task cancellation timed out after 2s")
        except Exception as e:
            logger.warning(f"Error during background task cancellation: {e}")
        finally:
            self._background_tasks.clear()

    def _init_core_services(self) -> None:
        """Initialize lightweight core services that have no external dependencies.

        Creates GridService for click grid overlay functionality and AutomationService
        for executing keyboard/mouse automation commands.
        """
        from vocalance.app.services.automation_service import AutomationService
        from vocalance.app.services.grid.grid_service import GridService

        with self._services_lock:
            self.services["grid"] = GridService(event_bus=self.event_bus, config=self.config)
            self.services["automation"] = AutomationService(event_bus=self.event_bus, app_config=self.config)

    async def _init_storage_services(self, progress_tracker: Optional[StartupProgressTracker] = None) -> None:
        """Initialize storage layer and dependent services concurrently.

        Creates the StorageService for file I/O, then initializes settings, command management,
        click tracking, and mark services in parallel. These services all depend on storage
        but are independent of each other, enabling concurrent initialization.

        Args:
            progress_tracker: Optional tracker for reporting progress to the startup UI.
        """
        from vocalance.app.services.commands.management import CommandManagementService
        from vocalance.app.services.grid.click_tracker_service import ClickTrackerService
        from vocalance.app.services.mark_service import MarkService
        from vocalance.app.services.storage.settings_service import SettingsService
        from vocalance.app.services.storage.settings_update_coordinator import SettingsUpdateCoordinator
        from vocalance.app.services.storage.storage_service import StorageService

        with self._services_lock:
            self.services["storage"] = StorageService(config=self.config)
            storage = self.services["storage"]

        async def init_settings() -> None:
            if progress_tracker:
                progress_tracker.update_status_animated(status="Loading user settings")

            settings_coordinator = SettingsUpdateCoordinator(event_bus=self.event_bus, config=self.config)
            settings_coordinator.setup_subscriptions()

            settings = SettingsService(
                event_bus=self.event_bus,
                config=self.config,
                storage=storage,
                coordinator=settings_coordinator,
            )
            await settings.initialize()
            settings.setup_subscriptions()
            await settings.apply_startup_settings_to_config()

            with self._services_lock:
                self.services["settings_coordinator"] = settings_coordinator
                self.services["settings"] = settings

        def init_commands() -> None:
            if progress_tracker:
                progress_tracker.update_status_animated(status="Setting up command storage")

            from vocalance.app.services.commands.action_map_provider import CommandActionMapProvider
            from vocalance.app.services.protected_terms_validator import ProtectedTermsValidator

            protected_terms_validator = ProtectedTermsValidator(config=self.config, storage=storage)
            protected_terms_validator.setup_invalidation_subscriptions(self.event_bus)
            action_map_provider = CommandActionMapProvider(storage=storage)

            command_management = CommandManagementService(
                event_bus=self.event_bus,
                storage=storage,
                protected_terms_validator=protected_terms_validator,
                action_map_provider=action_map_provider,
            )
            command_management.setup_subscriptions()

            with self._services_lock:
                self.services["protected_terms_validator"] = protected_terms_validator
                self.services["action_map_provider"] = action_map_provider
                self.services["command_management"] = command_management

        async def init_click_tracker() -> None:
            if progress_tracker:
                progress_tracker.update_status_animated(status="Initializing click tracking")

            click_tracker = ClickTrackerService(event_bus=self.event_bus, config=self.config, storage=storage)
            await click_tracker.initialize()

            with self._services_lock:
                self.services["click_tracker"] = click_tracker

        def init_marks() -> None:
            if progress_tracker:
                progress_tracker.update_status_animated(status="Configuring mark system")

            with self._services_lock:
                protected_terms_validator = self.services.get("protected_terms_validator")

            mark = MarkService(
                event_bus=self.event_bus,
                config=self.config,
                storage=storage,
                protected_terms_validator=protected_terms_validator,
            )

            with self._services_lock:
                self.services["mark"] = mark

        await asyncio.gather(
            init_settings(), asyncio.to_thread(init_commands), init_click_tracker(), asyncio.to_thread(init_marks)
        )

    async def _init_audio_services(self, progress_tracker: Optional[StartupProgressTracker] = None) -> None:
        """Initialize audio processing pipeline services sequentially with cancellation checks.

        Initializes the audio capture service, sound recognition, STT engines, command parser,
        dictation coordinator with LLM support, and Markov command predictor. Services are
        initialized sequentially due to dependencies, with shutdown checks between each stage.
        Registers initialized services with the settings coordinator for dynamic configuration.

        Args:
            progress_tracker: Optional tracker for reporting progress and estimated times to UI.
        """
        from vocalance.app.services.audio.dictation_handling.dictation_coordinator import DictationCoordinator
        from vocalance.app.services.audio.simple_audio_service import AudioService
        from vocalance.app.services.commands.markov import MarkovCommandService
        from vocalance.app.services.commands.parser import CentralizedCommandParser
        from vocalance.app.services.deduplication.event_deduplicator import EventDeduplicator

        with self._services_lock:
            storage = self.services["storage"]
            action_map_provider = self.services["action_map_provider"]
            protected_terms_validator = self.services["protected_terms_validator"]

        # Create unified deduplicator for all command sources (Vosk, sound, Markov)
        deduplicator = EventDeduplicator(window_ms=self.config.command_parser.duplicate_detection_window_ms)

        def init_audio() -> None:
            if progress_tracker:
                progress_tracker.update_sub_step(sub_step_name="Starting audio capture...")

            audio = AudioService(event_bus=self.event_bus, config=self.config, main_event_loop=self.gui_loop)

            with self._services_lock:
                self.services["audio"] = audio

        async def init_sound() -> None:
            """Initialize sound service with non-blocking TensorFlow import."""
            if progress_tracker:
                # Check if YamNet model already exists in app directory
                yamnet_app_path = os.path.join(self.config.storage.sound_model_dir, "yamnet")
                yamnet_exists = (
                    os.path.exists(yamnet_app_path)
                    and os.path.exists(os.path.join(yamnet_app_path, "saved_model.pb"))
                    and os.path.exists(os.path.join(yamnet_app_path, "variables"))
                )

                status_message = (
                    "Loading YAMNet model. This should take 1-2 minutes on first use."
                    if not yamnet_exists
                    else "Initializing sound recognition"
                )
                progress_tracker.update_status_animated(status=status_message)

            def _import_and_create_sound_service():
                from vocalance.app.services.audio.sound_recognizer.streamlined_sound_service import SoundService

                return SoundService(event_bus=self.event_bus, config=self.config, storage=storage)

            sound_service = await asyncio.to_thread(_import_and_create_sound_service)
            await sound_service.initialize()

            with self._services_lock:
                self.services["sound_service"] = sound_service

        async def init_stt() -> None:
            """Initialize STT service with non-blocking imports."""
            if progress_tracker:
                try:
                    from moonshine_voice.download_file import get_cache_dir

                    _ms_cache = get_cache_dir()
                    model_exists = _ms_cache.is_dir() and any(_ms_cache.iterdir())
                except Exception:
                    model_exists = False

                status_message = (
                    "Fetching Moonshine STT model. This may take several minutes on first use."
                    if not model_exists
                    else "Initializing speech-to-text"
                )
                progress_tracker.update_status_animated(status=status_message)

            def _import_and_create_stt_service():
                from vocalance.app.services.audio.stt.stt_service import SpeechToTextService

                return SpeechToTextService(event_bus=self.event_bus, config=self.config)

            stt = await asyncio.to_thread(_import_and_create_stt_service)
            await stt.initialize_engines(shutdown_coordinator=self.shutdown_coordinator)

            with self._services_lock:
                self.services["stt"] = stt

        async def init_command_parser() -> None:
            if progress_tracker:
                progress_tracker.update_status_animated(status="Setting up command processing")

            from vocalance.app.services.commands.history import CommandHistoryManager
            from vocalance.app.services.pause_state_manager import PauseStateManager

            # Create pause state manager
            pause_state_manager = PauseStateManager(event_bus=self.event_bus)

            history_manager = CommandHistoryManager(storage=storage, protected_terms_validator=protected_terms_validator)
            centralized_parser = CentralizedCommandParser(
                event_bus=self.event_bus,
                app_config=self.config,
                action_map_provider=action_map_provider,
                history_manager=history_manager,
                deduplicator=deduplicator,
                pause_state_manager=pause_state_manager,
            )
            await centralized_parser.initialize()

            with self._services_lock:
                self.services["pause_state_manager"] = pause_state_manager
                self.services["history_manager"] = history_manager
                self.services["centralized_parser"] = centralized_parser

        async def init_dictation() -> None:
            if progress_tracker:
                progress_tracker.update_status_animated(status="Preparing dictation system")

            allow = self.config.local_llm_allowlist
            spec = allow.artifact_for(self.config.llm.selected_model_id) or allow.artifact_for(allow.default_id)
            llm_downloader = LLMModelDownloader(self.config)
            if spec and not llm_downloader.model_bundle_complete(spec.gguf_filenames):
                if progress_tracker:
                    progress_tracker.update_sub_step(
                        sub_step_name="Fetching default local LLM (~2–4 GB). First launch may take several minutes.",
                        progress=0.35,
                    )
                primary = await llm_downloader.download_model_bundle(
                    repo_id=spec.repo_id,
                    filenames=list(spec.gguf_filenames),
                )
                if not primary:
                    logger.error("Startup LLM bundle download failed")
                    raise RuntimeError("Critical asset download failed: LLM model")

            dictation = DictationCoordinator(
                event_bus=self.event_bus, config=self.config, storage=storage, gui_event_loop=self.gui_loop
            )

            if progress_tracker:
                progress_tracker.update_sub_step(sub_step_name="Initializing dictation", progress=0.55)

            initialization_success = await dictation.initialize()
            if not initialization_success:
                logger.error("Failed to initialize dictation service")
                raise RuntimeError("Critical dictation initialization failed")

            with self._services_lock:
                self.services["dictation"] = dictation

        async def init_markov_predictor() -> None:
            if progress_tracker:
                progress_tracker.update_status_animated(status="Initializing command predictor")

            markov_predictor = MarkovCommandService(event_bus=self.event_bus, config=self.config, storage=storage)
            await markov_predictor.initialize()

            with self._services_lock:
                self.services["markov_predictor"] = markov_predictor

        init_audio()
        self._check_cancellation()

        await init_sound()
        self._check_cancellation()

        # Warm-start ESC-50 sample cache in background (non-blocking)
        # This ensures fast training even on first use
        def init_esc50_warmstart() -> None:
            """Warm-start ESC-50 samples in background without blocking other initialization."""
            try:
                with self._services_lock:
                    sound_service = self.services.get("sound_service")
                if sound_service:
                    sound_service.recognizer.warm_start_esc50_samples()
            except Exception as e:
                logger.warning(f"ESC-50 warm-start failed (non-critical): {e}")

        # Track background task for cancellation
        esc50_task = asyncio.create_task(asyncio.to_thread(init_esc50_warmstart))
        self._background_tasks.append(esc50_task)

        await init_stt()
        self._check_cancellation()

        await init_command_parser()
        self._check_cancellation()

        await init_dictation()
        self._check_cancellation()

        # Inject STT service reference into dictation coordinator for streaming
        with self._services_lock:
            stt_service = self.services.get("stt")
            dictation = self.services.get("dictation")
            audio = self.services.get("audio")
            if stt_service and dictation:
                dictation.set_stt_service(stt_service)
                logger.debug("STT service reference injected into dictation coordinator")
            if audio and dictation:
                audio.set_dictation_chunk_callback(dictation.feed_moonshine_audio_chunk)
                logger.debug("Moonshine dictation fed from recorder thread (bypasses event bus for PCM)")

        await init_markov_predictor()

        self._register_services_with_settings_coordinator()

    def _register_services_with_settings_coordinator(self) -> None:
        """Register services with settings coordinator to enable dynamic configuration updates.

        Registers audio, grid, sound recognizer, and Markov predictor services so they
        receive real-time configuration changes when settings are updated through the UI.
        """
        with self._services_lock:
            coordinator = self.services.get("settings_coordinator")
            if not coordinator:
                return

            service_mappings = [
                ("markov_predictor", "markov_predictor"),
                ("sound_service", "sound_recognizer"),
                ("grid", "grid"),
                ("audio", "audio"),
            ]

            for service_key, registration_name in service_mappings:
                if service_key in self.services:
                    coordinator.register_service(service_name=registration_name, service_instance=self.services[service_key])

        logger.debug("Services registered with settings coordinator")

    def activate_all_services(self) -> None:
        """Activate all services by setting up event subscriptions and starting audio processing.

        Iterates through initialized services and calls setup_subscriptions() to register
        event handlers. Separately starts the audio service's processing thread to begin
        capturing audio. Must be called after initialization completes and before the main
        window is shown to ensure services are ready but not processing prematurely.
        """
        logger.debug("Activating all services")

        with self._services_lock:
            services_to_activate = [
                "grid",
                "automation",
                "click_tracker",
                "mark",
                "sound_service",
                "stt",
                "pause_state_manager",
                "centralized_parser",
                "dictation",
                "markov_predictor",
            ]

            for service_name in services_to_activate:
                service = self.services.get(service_name)
                if service and hasattr(service, "setup_subscriptions"):
                    logger.debug(f"Calling setup_subscriptions on {service_name}")
                    service.setup_subscriptions()
                else:
                    if not service:
                        logger.warning(f"Service {service_name} not found in services dict")
                    elif not hasattr(service, "setup_subscriptions"):
                        logger.warning(f"Service {service_name} does not have setup_subscriptions method")

            audio_service = self.services.get("audio")
            if audio_service:
                if hasattr(audio_service, "setup_subscriptions"):
                    audio_service.setup_subscriptions()
                if hasattr(audio_service, "start_processing"):
                    audio_service.start_processing()

        logger.info("All services activated successfully")


def _validate_critical_assets(app_config: GlobalAppConfig) -> bool:
    """Validate that critical assets exist before starting application.

    Checks for the presence of the Vosk STT model directory required for offline
    speech recognition. Logs critical error with download instructions if missing.

    Args:
        app_config: Application configuration containing asset paths.

    Returns:
        True if all critical assets are valid, False otherwise.
    """
    vosk_path = app_config.asset_paths.get_vosk_model_path()
    if not os.path.exists(vosk_path):
        logger.critical(f"Vosk model not found: {vosk_path}")
        logger.critical("Download models from: https://alphacephei.com/vosk/models")
        return False
    return True


async def _stop_audio(services: Dict[str, Any]) -> list[str]:
    """Stop audio processing.

    Args:
        services: Active services dict, checked for an 'audio' entry.

    Returns:
        List of error messages encountered.
    """
    errors = []

    if "audio" in services and hasattr(services["audio"], "stop_processing"):
        try:
            services["audio"].stop_processing()
        except Exception as e:
            error_msg = f"Error stopping audio processing: {e}"
            logger.error(error_msg)
            errors.append(error_msg)

    await asyncio.sleep(0.3)
    return errors


async def _stop_event_bus(event_bus: EventBus) -> list[str]:
    """Drain and stop the event bus.

    Args:
        event_bus: Event bus to stop.

    Returns:
        List of error messages encountered.
    """
    errors = []

    try:
        await event_bus.shutdown()
        logger.debug("Event bus stopped successfully")
    except Exception as e:
        error_msg = f"Error stopping event bus: {e}"
        logger.error(error_msg)
        errors.append(error_msg)

    return errors


async def _stop_mark_service_tasks(services: Dict[str, Any]) -> list[str]:
    """Stop mark service background tasks during shutdown.

    Args:
        services: Active services dict, checked for a 'mark' entry.

    Returns:
        List of error messages encountered.
    """
    errors = []

    if "mark" in services and hasattr(services["mark"], "stop_service_tasks"):
        try:
            await asyncio.wait_for(services["mark"].stop_service_tasks(), timeout=3.0)
        except asyncio.TimeoutError:
            logger.warning("Mark service tasks did not stop within timeout")
        except Exception as e:
            error_msg = f"Error stopping mark service: {e}"
            logger.error(error_msg)
            errors.append(error_msg)

    return errors


async def _shutdown_services_in_order(services: Dict[str, Any]) -> list[str]:
    """Shutdown all services in proper dependency order to prevent cleanup issues.

    Calls the shutdown() method on each service in reverse initialization order to
    ensure dependent services are cleaned up before their dependencies. Continues
    shutdown even if individual services fail, collecting all errors.

    Args:
        services: Dictionary of active services with string keys and service instances.

    Returns:
        List of error messages encountered during service shutdown.
    """
    errors = []
    shutdown_order = [
        "sound_service",
        "centralized_parser",
        "automation",
        "stt",
        "dictation",
        "markov_predictor",
        "click_tracker",
        "audio",
        "storage",
    ]

    for service_name in shutdown_order:
        if service_name in services and hasattr(services[service_name], "shutdown"):
            try:
                logger.debug(f"Shutting down {service_name}...")
                shutdown_func = services[service_name].shutdown
                if asyncio.iscoroutinefunction(shutdown_func):
                    await shutdown_func()
                else:
                    shutdown_func()
                logger.debug(f"{service_name} shutdown completed")
            except Exception as e:
                error_msg = f"Error shutting down {service_name}: {e}"
                logger.error(error_msg, exc_info=True)
                errors.append(error_msg)

    return errors


def _cleanup_memory() -> None:
    """Perform aggressive memory cleanup and return memory to OS if possible.

    Runs multiple garbage collection cycles and attempts to call malloc_trim on
    Linux systems to return freed memory to the operating system immediately rather
    than keeping it in the process heap. Also cleans up TensorFlow/PyTorch resources where applicable.
    """
    # Clean up TensorFlow sessions and models
    try:
        import tensorflow as tf

        tf.keras.backend.clear_session()
        logger.debug("TensorFlow session cleared")
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"Error clearing TensorFlow session: {e}")

    # Clean up PyTorch CUDA cache if available
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("PyTorch CUDA cache cleared")
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"Error clearing PyTorch cache: {e}")

    # Force garbage collection multiple times
    for i in range(3):
        gc.collect()
        logger.debug(f"Garbage collection round {i + 1} performed")

    # Platform-specific memory return
    try:
        if sys.platform == "win32":
            # Windows: Use SetProcessWorkingSetSize to trim working set
            try:
                import ctypes
                from ctypes import wintypes

                kernel32 = ctypes.windll.kernel32
                current_process = kernel32.GetCurrentProcess()

                # Set working set size to -1, -1 to trim as much as possible
                result = kernel32.SetProcessWorkingSetSize(
                    wintypes.HANDLE(current_process), ctypes.c_size_t(-1), ctypes.c_size_t(-1)
                )

                if result:
                    logger.debug("Windows process working set trimmed successfully")
                else:
                    logger.debug("SetProcessWorkingSetSize returned false")
            except Exception as e:
                logger.debug(f"Could not trim Windows process working set: {e}")
        else:
            # Linux/Unix: Use malloc_trim
            if hasattr(ctypes, "pythonapi"):
                try:
                    libc_name = ctypes.util.find_library("c")
                    if libc_name:
                        libc = ctypes.CDLL(libc_name)
                        if hasattr(libc, "malloc_trim"):
                            libc.malloc_trim(0)
                            logger.debug("malloc_trim called to return memory to OS")
                except Exception as e:
                    logger.debug(f"Could not call malloc_trim: {e}")
    except Exception as e:
        logger.debug(f"Could not force memory return: {e}")


async def _cleanup_services(
    services: Dict[str, Any],
    event_bus: EventBus,
    service_initializer: Optional[FastServiceInitializer] = None,
) -> None:
    """Clean up all services during shutdown.

    Stops audio, drains and stops the event bus, shuts down services in dependency
    order, clears service references, and performs memory cleanup.

    Args:
        services: Dictionary of active services to shut down.
        event_bus: Event bus instance to stop.
        service_initializer: Optional initializer for cancelling background tasks.
    """
    cleanup_errors: list[str] = []

    try:
        if service_initializer:
            await service_initializer.cancel_background_tasks()

        cleanup_errors.extend(await _stop_audio(services))
        cleanup_errors.extend(await _stop_mark_service_tasks(services))
        cleanup_errors.extend(await _shutdown_services_in_order(services))
        cleanup_errors.extend(await _stop_event_bus(event_bus))

        for service_name in list(services.keys()):
            try:
                del services[service_name]
            except Exception as e:
                logger.warning(f"Error deleting service {service_name}: {e}")

        _cleanup_memory()

        if cleanup_errors:
            logger.warning(f"Cleanup completed with {len(cleanup_errors)} errors")
            for error in cleanup_errors:
                logger.error(f"Cleanup error: {error}")
        else:
            logger.info("All services cleaned up successfully")

    except Exception as e:
        logger.error(f"Critical error during cleanup: {e}", exc_info=True)


async def initialize_services_with_ui_integration(
    initializer: FastServiceInitializer,
    progress_tracker: StartupProgressTracker,
) -> Dict[str, Any]:
    """Initialize services asynchronously, yielding to the Qt event loop between steps.

    With PySide6.QtAsyncio, Qt event processing is integrated into the asyncio loop.
    Each ``await asyncio.sleep(0)`` inside the initializer yields control back to Qt,
    keeping the startup window responsive without manual processEvents() calls.

    Args:
        initializer: FastServiceInitializer managing service creation.
        progress_tracker: Reports initialization progress to the startup UI.

    Returns:
        Dictionary mapping service names to initialized instances.
    """
    return await initializer.initialize_all(progress_tracker=progress_tracker)


async def _handle_initialization(
    init_task: asyncio.Task,
    service_initializer: FastServiceInitializer,
    startup_window: StartupWindow,
    shutdown_coordinator: Optional[Any],
    event_bus: EventBus,
) -> Optional[Dict[str, Any]]:
    """Await the initialization task and handle cancellation or failure.

    Args:
        init_task: Asyncio task executing service initialization.
        service_initializer: Initializer holding partially initialized services on failure.
        startup_window: Startup progress window for status updates.
        shutdown_coordinator: Optional coordinator to unregister the task on success.
        event_bus: Event bus requiring cleanup on failure.

    Returns:
        Dict of initialized services on success, None on cancellation or failure.
    """
    try:
        services = await init_task
        if shutdown_coordinator:
            shutdown_coordinator.unregister_initialization_task()
        return services

    except asyncio.CancelledError:
        logger.info("Initialization cancelled due to shutdown request")
        startup_window.update_progress(0.0, "Startup cancelled by user", animate=False)
        await asyncio.sleep(1)
        await service_initializer.cancel_background_tasks()
        startup_window.close()
        await _cleanup_services(services=service_initializer.services, event_bus=event_bus)
        return None

    except RuntimeError as e:
        logger.critical(f"Critical initialization error: {e}")
        startup_window.update_progress(
            0.0,
            "Initialization failed. Please check your internet connection and try again.",
            animate=False,
        )
        await asyncio.sleep(3)
        startup_window.close()
        await _cleanup_services(services={}, event_bus=event_bus)
        return None


def _setup_infrastructure() -> EventBus:
    """Create the event bus.

    Returns:
        Initialized EventBus.
    """
    event_bus = EventBus()
    return event_bus


def _setup_signal_handlers(shutdown_coordinator: Any) -> QTimer:
    """Setup signal handlers for graceful shutdown on SIGINT and SIGTERM.

    Qt applications need special handling for signals. A QTimer polls a
    threading.Event flag set by the OS signal handler, since signal handlers
    cannot safely call Qt or asyncio methods directly.

    Args:
        shutdown_coordinator: Object with a request_shutdown(reason, source) method.

    Returns:
        QTimer instance that must be kept alive for signal handling to work.
    """
    shutdown_event = threading.Event()

    def signal_handler(signum, frame):
        signal_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        logger.info(f"Received {signal_name}, initiating graceful shutdown...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    def check_shutdown_signal():
        if shutdown_event.is_set():
            logger.info("Shutdown signal detected, requesting application shutdown...")
            shutdown_coordinator.request_shutdown(reason="System signal received", source="signal_handler")

    signal_timer = QTimer()
    signal_timer.timeout.connect(check_shutdown_signal)
    signal_timer.start(100)

    logger.debug("Signal handlers installed for SIGINT and SIGTERM")

    return signal_timer


async def main() -> None:
    """Main application entry point orchestrating startup, initialization, and cleanup.

    Runs under PySide6.QtAsyncio so the asyncio event loop IS the Qt event loop.
    All async handlers run on the Qt main thread; no cross-thread scheduling is needed
    from UI code.

    Shutdown lifecycle:
        1. Any source (window close, signal, error) calls shutdown_coordinator.request_shutdown()
        2. The coordinator resolves shutdown_future without touching the Qt event loop
        3. main() resumes, runs all async cleanup, then calls qt_app.quit() exactly once
    """
    logging.getLogger("numba").setLevel(logging.WARNING)

    icon_manager: Optional[WindowIconManager] = None
    services: Dict[str, Any] = {}
    event_bus: Optional[EventBus] = None
    service_initializer: Optional[FastServiceInitializer] = None

    try:
        app_info = AppInfoConfig()
        app_config = load_app_config(app_info=app_info)
        if hasattr(app_config, "__post_init__"):
            app_config.__post_init__()

        setup_logging(config=app_config.logging)
        os.makedirs(app_config.storage.user_data_root, exist_ok=True)

        qt_app = QApplication.instance()
        qt_app.setQuitOnLastWindowClosed(False)

        icon_path = None
        if app_config.asset_paths.icon_path:
            from pathlib import Path

            icon_path = Path(app_config.asset_paths.icon_path)

        icon_manager = WindowIconManager(icon_path=icon_path)
        if icon_manager.load_icon():
            icon_manager.apply_to_application(qt_app)
            logger.info("Application-level icon set for taskbar visibility")
        else:
            logger.warning("Failed to load application icon; proceeding without icon")

        theme.load_fonts(app_config.asset_paths.fonts_dir)
        theme._apply_app_palette(qt_app)
        logger.info("Theme palette applied to QApplication to override OS colors")

        event_bus = _setup_infrastructure()
        gui_event_loop = asyncio.get_event_loop()

        # Create the shutdown future before anything else so the coordinator can
        # resolve it from any context without touching the Qt event loop directly.
        shutdown_future: asyncio.Future = gui_event_loop.create_future()

        shutdown_coordinator = ShutdownCoordinator(shutdown_future=shutdown_future)
        signal_timer = _setup_signal_handlers(shutdown_coordinator=shutdown_coordinator)  # noqa: F841

        startup_window = StartupWindow(
            logger=logging.getLogger("StartupWindow"),
            asset_paths_config=app_config.asset_paths,
            shutdown_coordinator=shutdown_coordinator,
            icon_manager=icon_manager,
        )
        startup_window.show()

        if not _validate_critical_assets(app_config=app_config):
            startup_window.update_progress(0.0, "Critical assets missing. Please check logs.", animate=False)
            await asyncio.sleep(3)
            startup_window.close()
            qt_app.quit()
            return

        progress_tracker = StartupProgressTracker(startup_window=startup_window, total_steps=4)
        service_initializer = FastServiceInitializer(
            event_bus=event_bus,
            config=app_config,
            gui_loop=gui_event_loop,
            shutdown_coordinator=shutdown_coordinator,
        )

        init_task = asyncio.create_task(
            initialize_services_with_ui_integration(
                initializer=service_initializer,
                progress_tracker=progress_tracker,
            )
        )

        services = await _handle_initialization(
            init_task=init_task,
            service_initializer=service_initializer,
            startup_window=startup_window,
            shutdown_coordinator=shutdown_coordinator,
            event_bus=event_bus,
        )

        if not services:
            qt_app.quit()
            return

        progress_tracker.update_status_static(status="Ready!")
        startup_window.update_progress(1.0, "Ready!", animate=False)

        await asyncio.sleep(0.5)

        logger.info("Activating services now that initialization is complete")
        service_initializer.activate_all_services()

        main_window = VocalanceMainWindow(
            event_bus=event_bus,
            logger=logging.getLogger("MainWindow"),
            config=app_config,
            storage_service=services.get("storage"),
            icon_manager=icon_manager,
            shutdown_coordinator=shutdown_coordinator,
        )

        if "settings" in services:
            main_window.set_settings_service(services["settings"])
        if "audio" in services:
            main_window.set_audio_service(services["audio"])
        if "mark" in services:
            main_window.set_mark_service(services["mark"])
        if "grid" in services:
            main_window.set_grid_service(services["grid"])
        if "sound_service" in services:
            main_window.set_sound_service(services["sound_service"])
        if "command_management" in services:
            main_window.set_command_management_service(services["command_management"])
        if "dictation" in services:
            main_window.set_dictation_service(services["dictation"])
        if "click_tracker" in services:
            main_window.set_click_tracker_service(services["click_tracker"])

        main_window.initialize_controllers_with_services()

        main_window.show()
        main_window.raise_()
        main_window.activateWindow()

        startup_window.close_after_initialization()

        logger.info("Main window displayed, Qt/asyncio event loop running")

        # Block until any shutdown source resolves the future.
        await shutdown_future

        logger.info("Shutdown signal received, starting cleanup...")

        # Cleanup runs while the event loop is still live.
        await _cleanup_services(
            services=services,
            event_bus=event_bus,
            service_initializer=service_initializer,
        )

        logger.info("Application shutdown complete")

    except Exception as e:
        logger.exception(f"Unexpected error during application execution: {e}")

    # Quit Qt while the event loop is still running so QtAsyncio can tear down
    # cleanly. With keep_running=False, returning from this coroutine stops the loop.
    qt_app = QApplication.instance()
    if qt_app is not None:
        qt_app.quit()


if __name__ == "__main__":
    _qt_app = QApplication(sys.argv)
    _qt_app.setStyle("Fusion")
    QtAsyncio.run(main(), keep_running=False)
