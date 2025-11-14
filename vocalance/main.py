import asyncio
import ctypes
import ctypes.util
import gc
import logging
import os
import signal
import threading
import time
import tkinter as tk
from typing import TYPE_CHECKING, Any, Dict, Optional

import customtkinter as ctk

from vocalance.app.config.app_config import AppInfoConfig, GlobalAppConfig, load_app_config
from vocalance.app.config.logging_config import setup_logging
from vocalance.app.event_bus import EventBus
from vocalance.app.services.shutdown_coordinator import ShutdownCoordinator
from vocalance.app.ui import ui_theme
from vocalance.app.ui.startup_window import StartupProgressTracker, StartupWindow
from vocalance.app.ui.utils.ui_icon_utils import configure_dpi_awareness, initialize_windows_taskbar_icon, set_window_icon_robust
from vocalance.app.ui.utils.ui_thread_utils import initialize_ui_scheduler

if TYPE_CHECKING:
    from vocalance.app.services.audio.dictation_handling.dictation_coordinator import DictationCoordinator

logger = logging.getLogger(__name__)


async def initialize_services_with_ui_integration(
    initializer: "FastServiceInitializer",
    progress_tracker: StartupProgressTracker,
    root_window: tk.Tk,
    startup_window: Optional[StartupWindow] = None,
) -> Dict[str, Any]:
    """Initialize services while maintaining UI responsiveness during startup.

    Executes service initialization asynchronously in parallel with Tkinter's event loop
    to prevent the UI from freezing. Periodically processes GUI events and updates the
    startup window while the background initialization task runs to completion.

    Args:
        initializer: FastServiceInitializer instance managing service creation sequence.
        progress_tracker: Monitors initialization steps and publishes progress updates.
        root_window: Main Tkinter root window requiring periodic event processing.
        startup_window: Optional progress indicator window displaying initialization status.

    Returns:
        Dictionary mapping service names to initialized service instances.
    """
    init_task = asyncio.create_task(initializer.initialize_all(progress_tracker=progress_tracker))
    gui_update_interval = 0.01

    while not init_task.done():
        try:
            root_window.update_idletasks()
            root_window.update()

            if startup_window:
                startup_window.process_queue_now()
        except tk.TclError as e:
            logger.debug(f"GUI update failed (window may be closing): {e}")
            break

        await asyncio.sleep(gui_update_interval)

    return await init_task


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
        root: tk.Tk,
        shutdown_coordinator: Optional[ShutdownCoordinator] = None,
    ) -> None:
        """Initialize the service initializer with core infrastructure dependencies.

        Args:
            event_bus: Central event bus for inter-service communication.
            config: Global application configuration containing all settings.
            gui_loop: Asyncio event loop dedicated to GUI operations.
            root: Tkinter root window instance.
            shutdown_coordinator: Optional coordinator for handling shutdown requests during init.
        """
        self.event_bus: EventBus = event_bus
        self.config: GlobalAppConfig = config
        self.gui_loop: asyncio.AbstractEventLoop = gui_loop
        self.root: tk.Tk = root
        self.services: Dict[str, Any] = {}
        self.shutdown_coordinator: Optional[ShutdownCoordinator] = shutdown_coordinator
        self._services_lock: threading.RLock = threading.RLock()

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
        await self._init_core_services()
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

    async def initialize_ui_components(self, progress_tracker: StartupProgressTracker) -> None:
        """Initialize UI components including fonts, theme, and main control room window.

        Loads custom fonts, configures the UI theme with the font service, and creates
        the AppControlRoom instance that manages all UI controls and views. Must be called
        in the main thread after services are initialized.

        Args:
            progress_tracker: Tracks and reports UI initialization progress.
        """
        progress_tracker.start_step(step_name="Creating interface...")
        progress_tracker.update_status_animated(status="Building main window")
        await self._init_ui_components()
        progress_tracker.complete_step()

    async def _init_core_services(self) -> None:
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
        from vocalance.app.services.command_management_service import CommandManagementService
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

        async def init_commands() -> None:
            if progress_tracker:
                progress_tracker.update_status_animated(status="Setting up command storage")

            from vocalance.app.services.command_action_map_provider import CommandActionMapProvider
            from vocalance.app.services.protected_terms_validator import ProtectedTermsValidator

            protected_terms_validator = ProtectedTermsValidator(config=self.config, storage=storage)
            action_map_provider = CommandActionMapProvider(storage=storage)

            command_management = CommandManagementService(
                event_bus=self.event_bus,
                app_config=self.config,
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

            with self._services_lock:
                self.services["click_tracker"] = click_tracker

        async def init_marks() -> None:
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

        await asyncio.gather(init_settings(), init_commands(), init_click_tracker(), init_marks())

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
        from vocalance.app.services.centralized_command_parser import CentralizedCommandParser
        from vocalance.app.services.deduplication.event_deduplicator import EventDeduplicator
        from vocalance.app.services.markov_command_predictor import MarkovCommandService

        with self._services_lock:
            storage = self.services["storage"]
            action_map_provider = self.services["action_map_provider"]

        # Create unified deduplicator for all command sources (Vosk, sound, Markov)
        deduplicator = EventDeduplicator(window_ms=self.config.command_parser.duplicate_detection_window_ms)

        async def init_audio() -> None:
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
                whisper_models_dir = os.path.join(self.config.storage.user_data_root, "whisper_models")
                whisper_model_name = self.config.stt.whisper_model
                model_exists = (
                    os.path.exists(whisper_models_dir) and any(whisper_model_name in f for f in os.listdir(whisper_models_dir))
                    if os.path.exists(whisper_models_dir)
                    else False
                )

                status_message = (
                    "Fetching STT model. This should take 1-5 minutes on first use."
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

            from vocalance.app.services.command_history_manager import CommandHistoryManager

            history_manager = CommandHistoryManager(storage=storage)
            centralized_parser = CentralizedCommandParser(
                event_bus=self.event_bus,
                app_config=self.config,
                storage=storage,
                action_map_provider=action_map_provider,
                history_manager=history_manager,
                deduplicator=deduplicator,
            )
            await centralized_parser.initialize()

            with self._services_lock:
                self.services["history_manager"] = history_manager
                self.services["centralized_parser"] = centralized_parser

        async def init_dictation() -> None:
            if progress_tracker:
                progress_tracker.update_status_animated(status="Preparing dictation system")

            dictation = DictationCoordinator(
                event_bus=self.event_bus, config=self.config, storage=storage, gui_event_loop=self.gui_loop
            )

            llm_mode = getattr(self.config.llm, "startup_mode", "startup")
            if llm_mode == "startup":
                llm_filename = self.config.llm.get_model_filename()
                llm_models_dir = os.path.join(self.config.storage.user_data_root, "llm_models")
                llm_model_path = os.path.join(llm_models_dir, llm_filename)
                llm_exists = os.path.exists(llm_model_path) and os.path.getsize(llm_model_path) > 0

                if progress_tracker:
                    sub_message = (
                        "Fetching LLM model. This should take 5-15 minutes on first use."
                        if not llm_exists
                        else "Setting up LLM resources"
                    )
                    progress_tracker.update_sub_step(sub_step_name=sub_message)

                initialization_success = await dictation.initialize()
                if not initialization_success:
                    logger.error("Failed to initialize dictation service")
                    raise RuntimeError("Critical asset download failed: LLM model")
            elif llm_mode == "background":
                asyncio.create_task(self._background_llm_init(dictation=dictation))

            with self._services_lock:
                self.services["dictation"] = dictation

        async def init_markov_predictor() -> None:
            if progress_tracker:
                progress_tracker.update_status_animated(status="Initializing command predictor")

            markov_predictor = MarkovCommandService(event_bus=self.event_bus, config=self.config, storage=storage)
            await markov_predictor.initialize()

            with self._services_lock:
                self.services["markov_predictor"] = markov_predictor

        await init_audio()
        self._check_cancellation()

        await init_sound()
        self._check_cancellation()

        # Warm-start ESC-50 sample cache in background (non-blocking)
        # This ensures fast training even on first use
        async def init_esc50_warmstart() -> None:
            """Warm-start ESC-50 samples in background without blocking other initialization."""
            try:
                with self._services_lock:
                    sound_service = self.services.get("sound_service")
                if sound_service:
                    await sound_service.recognizer.warm_start_esc50_samples()
            except Exception as e:
                logger.warning(f"ESC-50 warm-start failed (non-critical): {e}")

        # Don't await - run in background
        asyncio.create_task(init_esc50_warmstart())

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
            if stt_service and dictation:
                dictation.set_stt_service(stt_service)
                logger.debug("STT service reference injected into dictation coordinator")

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

    async def _background_llm_init(self, dictation: "DictationCoordinator") -> None:
        """Initialize LLM model in background after application startup completes.

        Delays initialization by 2 seconds to avoid blocking startup, then initializes
        the LLM model asynchronously. Used when startup_mode is set to 'background'.

        Args:
            dictation: DictationCoordinator instance requiring LLM initialization.
        """
        await asyncio.sleep(2.0)
        await dictation.initialize()
        logger.debug("LLM initialized in background")

    async def activate_all_services(self) -> None:
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

    async def _init_ui_components(self) -> None:
        """Initialize UI components including fonts, theme, and AppControlRoom.

        Loads custom fonts through FontService, configures the UI theme, and creates
        the main AppControlRoom window that hosts all UI controls and views. Links
        the settings service and mark service to the control room.
        """
        # DEFERRED IMPORT: UI components
        from vocalance.app.ui.main_window import AppControlRoom
        from vocalance.app.ui.utils.font_service import FontService

        font_service = FontService(self.config.asset_paths)
        font_service.load_fonts()
        ui_theme.theme.font_family.set_font_service(font_service=font_service)

        with self._services_lock:
            storage = self.services.get("storage")
            settings = self.services.get("settings")
            mark = self.services.get("mark")

        control_room_logger = logging.getLogger("AppControlRoom")
        control_room = AppControlRoom(
            root=self.root,
            event_bus=self.event_bus,
            event_loop=self.gui_loop,
            logger=control_room_logger,
            config=self.config,
            storage_service=storage,
        )

        if settings:
            control_room.set_settings_service(settings_service=settings)

        with self._services_lock:
            self.services["control_room"] = control_room

        if mark:
            self.gui_loop.create_task(mark.start_service_tasks())


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


def _setup_infrastructure(app_config: GlobalAppConfig) -> tuple[EventBus, asyncio.AbstractEventLoop, threading.Thread]:
    """Setup core infrastructure: event bus, GUI event loop, and GUI thread.

    Creates a dedicated asyncio event loop running in a separate daemon thread for
    GUI-related async operations. Starts the event bus worker in this loop to handle
    asynchronous event dispatching without blocking the main thread.

    Args:
        app_config: Application configuration (included for consistency, currently unused).

    Returns:
        Tuple of (event_bus, gui_event_loop, gui_thread).
    """
    event_bus = EventBus()
    gui_event_loop = asyncio.new_event_loop()

    gui_thread = threading.Thread(
        target=lambda: (asyncio.set_event_loop(gui_event_loop), gui_event_loop.run_forever()),
        daemon=False,
        name="GUIEventLoop",
    )
    gui_thread.start()

    gui_event_loop.call_soon_threadsafe(lambda: gui_event_loop.create_task(event_bus.start_worker()))

    return event_bus, gui_event_loop, gui_thread


def _create_main_window(app_config: GlobalAppConfig) -> ctk.CTk:
    """Create and configure the main application window with proper DPI and icon handling.

    Configures DPI awareness for crisp rendering on high-DPI displays, sets dark theme,
    creates the CustomTkinter root window, and applies the application icon. The window
    is initially withdrawn to prevent flashing before initialization completes.

    Args:
        app_config: Application configuration containing theme dimensions.

    Returns:
        Configured CustomTkinter root window ready for content.
    """
    configure_dpi_awareness()

    initialize_windows_taskbar_icon()

    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    # Create window
    app_tk_root = ctk.CTk()
    app_tk_root.title("Vocalance")

    # Withdraw immediately to prevent CustomTkinter from showing with default icon
    app_tk_root.withdraw()

    # Set initial size
    app_tk_root.geometry(f"{ui_theme.theme.dimensions.main_window_width}x{ui_theme.theme.dimensions.main_window_height}")
    app_tk_root.minsize(ui_theme.theme.dimensions.main_window_min_width, ui_theme.theme.dimensions.main_window_min_height)
    app_tk_root.resizable(False, False)

    # Realize the window geometry while still withdrawn for accurate screen metrics
    app_tk_root.update_idletasks()

    initialize_ui_scheduler(root_window=app_tk_root)

    return app_tk_root


def _setup_signal_handlers(shutdown_coordinator: ShutdownCoordinator) -> None:
    """Setup signal handlers for graceful shutdown on SIGINT and SIGTERM.

    Registers handlers that request shutdown through the coordinator and start a
    fallback force-exit thread that terminates the process after 5 seconds if
    graceful shutdown fails to complete.

    Args:
        shutdown_coordinator: Coordinator managing the shutdown sequence.
    """

    def signal_handler(signum: int, frame: Any) -> None:
        logger.info(f"Received signal {signum}")
        shutdown_coordinator.request_shutdown(reason=f"Received system signal {signum}", source="signal_handler")

        def force_exit() -> None:
            time.sleep(5)
            logger.error("Force exiting due to shutdown timeout")
            os._exit(1)

        threading.Thread(target=force_exit, daemon=True).start()

    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, signal_handler)


async def _handle_initialization(
    init_task: asyncio.Task,
    service_initializer: FastServiceInitializer,
    startup_window: StartupWindow,
    shutdown_coordinator: ShutdownCoordinator,
    event_bus: EventBus,
    gui_event_loop: asyncio.AbstractEventLoop,
    gui_thread: threading.Thread,
) -> Optional[Dict[str, Any]]:
    """Handle service initialization with proper error handling and cancellation support.

    Awaits the initialization task and handles three outcomes: successful completion,
    cancellation due to user request (with partial cleanup), or runtime error (with
    full cleanup). Updates the startup window with appropriate status messages and
    unregisters the initialization task from the shutdown coordinator on completion.

    Args:
        init_task: Asyncio task executing the initialization sequence.
        service_initializer: Service initializer containing partially initialized services.
        startup_window: Progress window displaying initialization status to user.
        shutdown_coordinator: Coordinator tracking initialization task for cancellation.
        event_bus: Application event bus requiring cleanup on failure.
        gui_event_loop: GUI event loop requiring cleanup on failure.
        gui_thread: GUI thread requiring cleanup on failure.

    Returns:
        Dictionary of initialized services if successful, None if cancelled or failed.
    """
    try:
        services = await init_task
        shutdown_coordinator.unregister_initialization_task()
        return services

    except asyncio.CancelledError:
        logger.info("Initialization cancelled due to shutdown request")
        startup_window.update_progress(progress=0.0, status="Startup cancelled by user", animate=False)
        await asyncio.sleep(1)

        partial_services = service_initializer.services
        partial_services["gui_thread"] = gui_thread

        logger.debug(f"Cleaning up {len(partial_services)} partially initialized services")
        startup_window.close()
        await _cleanup_services(
            services=partial_services, event_bus=event_bus, gui_event_loop=gui_event_loop, gui_thread=gui_thread
        )
        return None

    except RuntimeError as e:
        logger.critical(f"Critical initialization error: {e}")
        logger.critical("Application will shut down")
        startup_window.update_progress(
            progress=0.0,
            status="Initialization failed. Please check your internet connection and try again.",
            animate=False,
        )
        await asyncio.sleep(3)
        startup_window.close()
        await _cleanup_services(services={}, event_bus=event_bus, gui_event_loop=gui_event_loop, gui_thread=gui_thread)
        return None


async def main() -> None:
    """Main application entry point orchestrating startup, initialization, and cleanup.

    Coordinates the complete application lifecycle: configures logging, validates assets,
    sets up infrastructure (event bus, GUI thread), creates the main window and startup
    window, initializes all services with progress tracking, activates services, displays
    the main window, runs the Tkinter mainloop, and performs cleanup on shutdown.
    """
    logging.getLogger("numba").setLevel(logging.WARNING)

    shutdown_coordinator: Optional[ShutdownCoordinator] = None

    try:
        app_info = AppInfoConfig()
        app_config = load_app_config(app_info=app_info)
        if hasattr(app_config, "__post_init__"):
            app_config.__post_init__()

        setup_logging(config=app_config.logging)
        os.makedirs(app_config.storage.user_data_root, exist_ok=True)

        # Create infrastructure and window
        event_bus, gui_event_loop, gui_thread = _setup_infrastructure(app_config=app_config)
        app_tk_root = _create_main_window(app_config=app_config)

        shutdown_coordinator = ShutdownCoordinator(
            event_bus=event_bus,
            root_window=app_tk_root,
            logger=logging.getLogger("ShutdownCoordinator"),
            gui_event_loop=gui_event_loop,
        )
        _setup_signal_handlers(shutdown_coordinator=shutdown_coordinator)

        # Show startup window
        startup_window = StartupWindow(
            logger=logging.getLogger("StartupWindow"),
            main_root=app_tk_root,
            asset_paths_config=app_config.asset_paths,
            shutdown_coordinator=shutdown_coordinator,
        )
        startup_window.show()

        for _ in range(3):
            app_tk_root.update_idletasks()
            app_tk_root.update()

        # Validate critical assets
        if not _validate_critical_assets(app_config=app_config):
            startup_window.update_progress(0.0, "Critical assets missing. Please check logs.", animate=False)
            await asyncio.sleep(3)
            startup_window.close()
            return

        progress_tracker = StartupProgressTracker(startup_window=startup_window, total_steps=4)
        service_initializer = FastServiceInitializer(
            event_bus=event_bus,
            config=app_config,
            gui_loop=gui_event_loop,
            root=app_tk_root,
            shutdown_coordinator=shutdown_coordinator,
        )

        async def run_initialization() -> Dict[str, Any]:
            services = await initialize_services_with_ui_integration(
                initializer=service_initializer,
                progress_tracker=progress_tracker,
                root_window=app_tk_root,
                startup_window=startup_window,
            )
            await service_initializer.initialize_ui_components(progress_tracker=progress_tracker)
            return services

        init_task = asyncio.create_task(run_initialization())
        shutdown_coordinator.register_initialization_task(task=init_task)

        services = await _handle_initialization(
            init_task=init_task,
            service_initializer=service_initializer,
            startup_window=startup_window,
            shutdown_coordinator=shutdown_coordinator,
            event_bus=event_bus,
            gui_event_loop=gui_event_loop,
            gui_thread=gui_thread,
        )

        if not services:
            return

        services["gui_thread"] = gui_thread

        progress_tracker.update_status_static(status="Ready!")
        startup_window.update_progress(progress=1.0, status="Ready!", animate=False)

        await asyncio.sleep(0.5)
        startup_window.close_after_initialization()

        await asyncio.sleep(0.1)

        logger.info("Activating services now that initialization is complete")
        await service_initializer.activate_all_services()

        def position_main_window():
            try:
                app_tk_root.update_idletasks()
                screen_width = app_tk_root.winfo_screenwidth()
                screen_height = app_tk_root.winfo_screenheight()
                window_width = ui_theme.theme.dimensions.main_window_width
                window_height = ui_theme.theme.dimensions.main_window_height

                x = (screen_width - window_width) // 2
                y = int(screen_height * 0.35 - window_height // 2)

                app_tk_root.geometry(f"+{x}+{y}")
            except Exception as e:
                logger.warning(f"Could not position main window: {e}")

        set_window_icon_robust(window=app_tk_root)

        app_tk_root.deiconify()
        app_tk_root.lift()

        position_main_window()

        app_tk_root.after(1, lambda: set_window_icon_robust(window=app_tk_root))
        app_tk_root.after(10, lambda: set_window_icon_robust(window=app_tk_root))
        app_tk_root.after(50, lambda: set_window_icon_robust(window=app_tk_root))

        def safe_focus() -> None:
            try:
                app_tk_root.focus_force()
            except tk.TclError as e:
                logger.debug(f"Focus force failed (expected during startup transition): {e}")

        def check_shutdown() -> None:
            if shutdown_coordinator.is_shutdown_requested():
                logger.debug("Shutdown detected via coordinator")
                return
            app_tk_root.after(100, check_shutdown)

        app_tk_root.after(200, safe_focus)
        app_tk_root.after(100, check_shutdown)

        try:
            app_tk_root.mainloop()
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received")
        finally:
            logger.info("Starting cleanup...")
            await _cleanup_services(services=services, event_bus=event_bus, gui_event_loop=gui_event_loop, gui_thread=gui_thread)

        logger.info("Application shutdown complete")

    except Exception as e:
        logger.exception(f"Unexpected error during application execution: {e}")


async def _stop_audio_and_event_bus(
    services: Dict[str, Any], event_bus: EventBus, gui_event_loop: asyncio.AbstractEventLoop
) -> list[str]:
    """Stop audio service and event bus during shutdown sequence.

    Stops the audio service's processing thread to halt audio capture, waits briefly
    for pending audio processing to complete, then stops the event bus worker to
    prevent new events from being processed.

    Args:
        services: Dictionary of active services potentially including 'audio'.
        event_bus: Event bus instance to be stopped.
        gui_event_loop: GUI event loop where event bus worker is running.

    Returns:
        List of error messages encountered during shutdown.
    """
    errors = []

    if "audio" in services and hasattr(services["audio"], "stop_processing"):
        services["audio"].stop_processing()

    await asyncio.sleep(0.3)

    if not gui_event_loop.is_closed():
        try:
            stop_future = asyncio.run_coroutine_threadsafe(event_bus.stop_worker(), gui_event_loop)
            stop_future.result(timeout=5.0)
            logger.debug("Event bus stopped successfully")
        except Exception as e:
            error_msg = f"Error stopping event bus: {e}"
            logger.error(error_msg)
            errors.append(error_msg)

    return errors


async def _stop_mark_service_tasks(services: Dict[str, Any], gui_event_loop: asyncio.AbstractEventLoop) -> list[str]:
    """Stop mark service background tasks during shutdown.

    Calls the mark service's stop_service_tasks() method in the GUI event loop
    to cleanly terminate any running background tasks with a timeout.

    Args:
        services: Dictionary of active services potentially including 'mark'.
        gui_event_loop: GUI event loop where mark service tasks are running.

    Returns:
        List of error messages encountered during task cancellation.
    """
    errors = []

    if "mark" in services and hasattr(services["mark"], "stop_service_tasks"):
        try:
            if not gui_event_loop.is_closed():
                stop_future = asyncio.run_coroutine_threadsafe(services["mark"].stop_service_tasks(), gui_event_loop)
                stop_future.result(timeout=3)
        except Exception as e:
            error_msg = f"Error stopping mark service: {e}"
            logger.error(error_msg)
            errors.append(error_msg)

    return errors


async def _cancel_gui_event_loop_tasks(gui_event_loop: asyncio.AbstractEventLoop) -> None:
    """Cancel all pending tasks in GUI event loop during shutdown.

    Retrieves all pending tasks from the GUI event loop, cancels them, and waits
    for cancellation to complete with a timeout to prevent hanging on stuck tasks.

    Args:
        gui_event_loop: GUI event loop potentially containing pending async tasks.
    """
    if gui_event_loop.is_closed():
        return

    pending_tasks = [task for task in asyncio.all_tasks(gui_event_loop) if not task.done()]

    if not pending_tasks:
        return

    logger.debug(f"Cancelling {len(pending_tasks)} pending tasks")

    for task in pending_tasks:
        task.cancel()

    try:
        cancel_future = asyncio.run_coroutine_threadsafe(asyncio.gather(*pending_tasks, return_exceptions=True), gui_event_loop)
        cancel_future.result(timeout=2.0)
        logger.debug("All pending tasks cancelled successfully")
    except asyncio.TimeoutError:
        logger.warning("Timeout waiting for tasks to be cancelled")
    except Exception as e:
        logger.warning(f"Error cancelling pending tasks: {e}")


async def _stop_gui_event_loop(gui_event_loop: asyncio.AbstractEventLoop, gui_thread: threading.Thread) -> list[str]:
    """Stop GUI event loop and wait for GUI thread termination during shutdown.

    Stops the GUI event loop using call_soon_threadsafe, waits for the GUI thread to
    terminate with a timeout, and closes the event loop. Logs warnings if the thread
    doesn't terminate cleanly within the timeout period.

    Args:
        gui_event_loop: GUI event loop to be stopped and closed.
        gui_thread: GUI thread running the event loop.

    Returns:
        List of error messages encountered during shutdown.
    """
    errors = []

    if gui_event_loop.is_closed():
        return errors

    gui_event_loop.call_soon_threadsafe(gui_event_loop.stop)

    try:
        gui_thread.join(timeout=5.0)
        if gui_thread.is_alive():
            logger.warning("GUI thread did not terminate cleanly within timeout")
        else:
            logger.debug("GUI thread terminated successfully")
    except Exception as e:
        error_msg = f"Error joining GUI thread: {e}"
        logger.error(error_msg)
        errors.append(error_msg)

    await asyncio.sleep(0.3)

    if not gui_event_loop.is_closed():
        try:
            gui_event_loop.close()
            logger.debug("GUI event loop closed")
        except Exception as e:
            logger.warning(f"Error closing GUI event loop: {e}")

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
        "audio",
        "storage",
    ]

    for service_name in shutdown_order:
        if service_name in services and hasattr(services[service_name], "shutdown"):
            try:
                logger.debug(f"Shutting down {service_name}...")
                await services[service_name].shutdown()
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
    than keeping it in the process heap.
    """
    for i in range(3):
        gc.collect()
        logger.debug(f"Garbage collection round {i+1} performed")

    try:
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
    services: Dict[str, Any], event_bus: EventBus, gui_event_loop: asyncio.AbstractEventLoop, gui_thread: threading.Thread
) -> None:
    """Clean up all services during shutdown with proper async task cleanup.

    Orchestrates the complete shutdown sequence: stops audio and event bus, cancels
    pending GUI tasks, stops the GUI event loop, shuts down services in order, clears
    service references, performs memory cleanup, and exits the process. Collects all
    errors encountered and logs them for debugging.

    Args:
        services: Dictionary of active services to be shut down.
        event_bus: Event bus instance to be stopped.
        gui_event_loop: GUI event loop to be stopped and closed.
        gui_thread: GUI thread to be joined and terminated.
    """
    cleanup_errors: list[str] = []

    try:
        cleanup_errors.extend(await _stop_audio_and_event_bus(services, event_bus, gui_event_loop))
        cleanup_errors.extend(await _stop_mark_service_tasks(services, gui_event_loop))
        await _cancel_gui_event_loop_tasks(gui_event_loop)
        cleanup_errors.extend(await _stop_gui_event_loop(gui_event_loop, gui_thread))
        cleanup_errors.extend(await _shutdown_services_in_order(services))

        service_names_to_clear = [name for name in services.keys() if name != "gui_thread"]
        for service_name in service_names_to_clear:
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

        await asyncio.sleep(0.1)
        os._exit(0)

    except Exception as e:
        logger.error(f"Critical error during cleanup: {e}", exc_info=True)
        os._exit(1)


if __name__ == "__main__":
    asyncio.run(main())
