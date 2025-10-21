import asyncio

# Additional imports moved from functions
import ctypes
import ctypes.util
import gc
import logging
import os
import signal
import threading
import time
import tkinter as tk
from typing import Any, Dict

import customtkinter as ctk

from iris.app.config.app_config import AppInfoConfig, GlobalAppConfig, load_app_config
from iris.app.config.logging_config import setup_logging
from iris.app.event_bus import EventBus
from iris.app.services.audio.dictation_handling.dictation_coordinator import DictationCoordinator
from iris.app.services.audio.simple_audio_service import SimpleAudioService
from iris.app.services.audio.sound_recognizer.streamlined_sound_service import StreamlinedSoundService
from iris.app.services.audio.stt.stt_service import SpeechToTextService
from iris.app.services.automation_service import AutomationService
from iris.app.services.centralized_command_parser import CentralizedCommandParser
from iris.app.services.command_management_service import CommandManagementService
from iris.app.services.grid.click_tracker_service import ClickTrackerService

# Core services
from iris.app.services.grid.grid_service import GridService
from iris.app.services.mark_service import MarkService
from iris.app.services.markov_command_predictor import MarkovCommandService
from iris.app.services.shutdown_coordinator import ShutdownCoordinator
from iris.app.services.storage.settings_service import SettingsService
from iris.app.services.storage.settings_update_coordinator import SettingsUpdateCoordinator
from iris.app.services.storage.storage_service import StorageService
from iris.app.ui import ui_theme
from iris.app.ui.main_window import AppControlRoom

# UI components
from iris.app.ui.startup_window import StartupProgressTracker
from iris.app.ui.utils.font_service import FontService
from iris.app.ui.utils.ui_icon_utils import set_window_icon_robust
from iris.app.ui.utils.ui_thread_utils import initialize_ui_scheduler


async def initialize_services_with_ui_integration(initializer, progress_tracker, root_window, startup_window=None):
    """
    Initialize services while keeping Tkinter responsive.

    Design:
    - Runs in main thread's async event loop
    - Heavy I/O operations (Whisper/LLM downloads) run in daemon threads
    - Tkinter events processed via periodic update() calls
    - Simple, single-threaded initialization with async/await
    """
    # Create a task to run initialization
    init_task = asyncio.create_task(initializer.initialize_all(progress_tracker))

    # Process Tkinter events while initialization runs
    while not init_task.done():
        try:
            root_window.update_idletasks()
            root_window.update()

            # Process startup window queue to flush pending messages
            if startup_window:
                startup_window.process_queue_now()
        except Exception as e:
            logging.debug(f"GUI update failed: {e}")

        # Small yield to let initialization progress
        await asyncio.sleep(0.01)

    # Return result (or raise exception if initialization failed)
    return await init_task


class FastServiceInitializer:
    """Streamlined service initialization for maximum startup speed"""

    def __init__(
        self,
        event_bus: EventBus,
        config: GlobalAppConfig,
        gui_loop: asyncio.AbstractEventLoop,
        root: tk.Tk,
        shutdown_coordinator=None,
    ):
        self.event_bus = event_bus
        self.config = config
        self.gui_loop = gui_loop
        self.root = root
        self.services = {}
        self.shutdown_coordinator = shutdown_coordinator

    async def initialize_all(self, progress_tracker: StartupProgressTracker) -> Dict[str, Any]:
        """Fast parallel initialization of all services (non-UI only)"""

        # Step 1: Core infrastructure (fastest services first)
        progress_tracker.start_step("Starting core services...")
        progress_tracker.update_sub_step("Initializing grid service...")
        await self._init_core_services()
        progress_tracker.complete_step()

        self._check_cancellation()

        # Step 2: Storage and data services (parallel)
        progress_tracker.start_step("Initializing storage...")
        progress_tracker.update_sub_step("Setting up unified storage...")
        await self._init_storage_services(progress_tracker)
        progress_tracker.complete_step()

        self._check_cancellation()

        # Step 3: Audio and command services (parallel)
        progress_tracker.start_step("Starting audio processing...")
        progress_tracker.update_sub_step("Loading audio engines...")
        await self._init_audio_services(progress_tracker)
        progress_tracker.complete_step()

        return self.services

    def _check_cancellation(self):
        """Check if shutdown was requested and raise CancelledError if so."""
        if self.shutdown_coordinator and self.shutdown_coordinator.is_shutdown_requested():
            logging.info("Shutdown detected during initialization - cancelling")
            raise asyncio.CancelledError("Initialization cancelled due to shutdown request")

    async def initialize_ui_components(self, progress_tracker: StartupProgressTracker) -> None:
        """Initialize UI components - MUST run in main thread"""
        progress_tracker.start_step("Creating interface...")
        progress_tracker.update_status_animated("Building main window")
        await self._init_ui_components()
        progress_tracker.complete_step()

    async def _init_core_services(self):
        """Initialize lightweight core services"""
        # Grid service
        self.services["grid"] = GridService(event_bus=self.event_bus, config=self.config)
        self.services["grid"].setup_subscriptions()

        # Automation service
        self.services["automation"] = AutomationService(event_bus=self.event_bus, app_config=self.config)
        self.services["automation"].setup_subscriptions()

    async def _init_storage_services(self, progress_tracker=None):
        """Initialize storage services in parallel"""
        # Storage service
        self.services["storage"] = StorageService(config=self.config)

        # Individual storage services

        # Initialize storage services in parallel
        tasks = []

        async def init_settings():
            if progress_tracker:
                progress_tracker.update_status_animated("Loading user settings")

            # Initialize settings update coordinator
            self.services["settings_coordinator"] = SettingsUpdateCoordinator(event_bus=self.event_bus, config=self.config)
            self.services["settings_coordinator"].setup_subscriptions()

            # Initialize settings service
            self.services["settings"] = SettingsService(
                event_bus=self.event_bus,
                config=self.config,
                storage=self.services["storage"],
                coordinator=self.services["settings_coordinator"],
            )
            self.services["settings"].setup_subscriptions()
            await self.services["settings"].initialize()

            # Apply user settings at startup (via coordinator for consistency)
            await self.services["settings"].apply_startup_settings_to_config()

        async def init_commands():
            if progress_tracker:
                progress_tracker.update_status_animated("Setting up command storage")

            # Create shared helper services
            from iris.app.services.command_action_map_provider import CommandActionMapProvider
            from iris.app.services.protected_terms_validator import ProtectedTermsValidator

            self.services["protected_terms_validator"] = ProtectedTermsValidator(
                config=self.config, storage=self.services["storage"]
            )

            self.services["action_map_provider"] = CommandActionMapProvider(storage=self.services["storage"])

            # Command management service for event handling
            self.services["command_management"] = CommandManagementService(
                event_bus=self.event_bus,
                app_config=self.config,
                storage=self.services["storage"],
                protected_terms_validator=self.services["protected_terms_validator"],
                action_map_provider=self.services["action_map_provider"],
            )
            self.services["command_management"].setup_subscriptions()

        async def init_click_tracker():
            if progress_tracker:
                progress_tracker.update_status_animated("Initializing click tracking")
            self.services["click_tracker"] = ClickTrackerService(
                event_bus=self.event_bus, config=self.config, storage=self.services["storage"]
            )
            self.services["click_tracker"].setup_subscriptions()

        async def init_marks():
            if progress_tracker:
                progress_tracker.update_status_animated("Configuring mark system")

            self.services["mark"] = MarkService(
                event_bus=self.event_bus,
                config=self.config,
                storage=self.services["storage"],
                protected_terms_validator=self.services["protected_terms_validator"],
            )
            self.services["mark"].setup_subscriptions()

        tasks.extend([init_settings(), init_commands(), init_click_tracker(), init_marks()])
        await asyncio.gather(*tasks)

    async def _init_audio_services(self, progress_tracker=None):
        """Initialize audio services in parallel"""

        async def init_audio():
            if progress_tracker:
                progress_tracker.update_sub_step("Starting audio capture...")
            self.services["audio"] = SimpleAudioService(
                event_bus=self.event_bus, config=self.config, main_event_loop=self.gui_loop
            )
            self.services["audio"].setup_subscriptions()
            self.services["audio"].start_processing()

        async def init_sound():
            if progress_tracker:
                progress_tracker.update_status_animated("Loading sound recognition")
            self.services["sound_service"] = StreamlinedSoundService(
                event_bus=self.event_bus, config=self.config, storage=self.services["storage"]
            )
            await self.services["sound_service"].initialize()

        async def init_stt():
            if progress_tracker:
                whisper_models_dir = os.path.join(self.config.storage.user_data_root, "whisper_models")
                whisper_model_name = self.config.stt.whisper_model
                model_exists = (
                    os.path.exists(whisper_models_dir) and any(whisper_model_name in f for f in os.listdir(whisper_models_dir))
                    if os.path.exists(whisper_models_dir)
                    else False
                )

                if not model_exists:
                    progress_tracker.update_status_animated("Fetching STT model. This may take up to 5 minutes on first use.")
                else:
                    progress_tracker.update_status_animated("Initializing speech-to-text")

            try:
                self.services["stt"] = SpeechToTextService(event_bus=self.event_bus, config=self.config)
                await self.services["stt"].initialize_engines(shutdown_coordinator=self.shutdown_coordinator)
                self.services["stt"].setup_subscriptions()
            except Exception as e:
                logging.error(f"Failed to initialize STT service: {e}", exc_info=True)
                raise RuntimeError("Critical asset download failed: Whisper model")

        async def init_command_parser():
            if progress_tracker:
                progress_tracker.update_status_animated("Setting up command processing")

            # Create helper services for command parser
            from iris.app.services.command_history_manager import CommandHistoryManager

            self.services["history_manager"] = CommandHistoryManager(storage=self.services["storage"])

            self.services["centralized_parser"] = CentralizedCommandParser(
                event_bus=self.event_bus,
                app_config=self.config,
                storage=self.services["storage"],
                action_map_provider=self.services["action_map_provider"],
                history_manager=self.services["history_manager"],
            )
            await self.services["centralized_parser"].initialize()
            self.services["centralized_parser"].setup_subscriptions()

        async def init_dictation():
            if progress_tracker:
                progress_tracker.update_status_animated("Preparing dictation system")

            self.services["dictation"] = DictationCoordinator(
                event_bus=self.event_bus, config=self.config, storage=self.services["storage"], gui_event_loop=self.gui_loop
            )
            self.services["dictation"].setup_subscriptions()

            llm_mode = getattr(self.config.llm, "startup_mode", "startup")
            if llm_mode == "startup":
                llm_filename = self.config.llm.get_model_filename()
                llm_models_dir = os.path.join(self.config.storage.user_data_root, "llm_models")
                llm_model_path = os.path.join(llm_models_dir, llm_filename)
                llm_exists = os.path.exists(llm_model_path) and os.path.getsize(llm_model_path) > 0

                if progress_tracker:
                    if not llm_exists:
                        progress_tracker.update_sub_step("Fetching LLM model. This may take up to 15 minutes on first use.")
                    else:
                        progress_tracker.update_sub_step("Setting up LLM resources")

                initialization_success = await self.services["dictation"].initialize()

                if not initialization_success:
                    logging.error("Failed to initialize dictation service")
                    raise RuntimeError("Critical asset download failed: LLM model")
            elif llm_mode == "background":
                asyncio.create_task(self._background_llm_init())

        async def init_markov_predictor():
            if progress_tracker:
                progress_tracker.update_status_animated("Initializing command predictor")
            self.services["markov_predictor"] = MarkovCommandService(
                event_bus=self.event_bus, config=self.config, storage=self.services["storage"]
            )
            self.services["markov_predictor"].setup_subscriptions()
            await self.services["markov_predictor"].initialize()

        # Run sequentially for clear progress tracking (heavy I/O still offloaded to executors)
        # Check for cancellation between each step
        await init_audio()
        self._check_cancellation()

        await init_sound()
        self._check_cancellation()

        await init_stt()  # Whisper download happens here (in executor, non-blocking)
        self._check_cancellation()

        await init_command_parser()
        self._check_cancellation()

        await init_dictation()  # LLM download happens here (in executor, non-blocking)
        self._check_cancellation()

        await init_markov_predictor()

        # Register services with coordinator for real-time updates (after initialization)
        coordinator = self.services.get("settings_coordinator")
        if coordinator:
            if "markov_predictor" in self.services:
                coordinator.register_service(service_name="markov_predictor", service_instance=self.services["markov_predictor"])
            if "sound_service" in self.services:
                coordinator.register_service(service_name="sound_recognizer", service_instance=self.services["sound_service"])
            if "grid" in self.services:
                coordinator.register_service(service_name="grid", service_instance=self.services["grid"])
            logging.info("Services registered with settings coordinator for real-time updates")

    async def _background_llm_init(self):
        """Initialize LLM in background after startup"""
        await asyncio.sleep(2.0)
        await self.services["dictation"].initialize()
        logging.info("LLM initialized in background")

    async def _init_ui_components(self):
        """Initialize UI components"""
        # Initialize font service early
        font_service = FontService(self.config.asset_paths)
        font_service.load_fonts()

        # Set font service on the global theme
        ui_theme.theme.font_family.set_font_service(font_service)

        control_room_logger = logging.getLogger("AppControlRoom")
        self.services["control_room"] = AppControlRoom(
            root=self.root,
            event_bus=self.event_bus,
            event_loop=self.gui_loop,
            logger=control_room_logger,
            config=self.config,
            storage_service=self.services.get("storage"),
        )

        # Pass settings service to control room so it can be used by settings controller
        if "settings" in self.services:
            self.services["control_room"].set_settings_service(self.services["settings"])

        # Start background tasks
        if "mark" in self.services:
            self.gui_loop.create_task(self.services["mark"].start_service_tasks())


async def main():
    """Main application entry point"""
    logging.getLogger("numba").setLevel(logging.WARNING)

    # Shutdown coordinator will be initialized after root window is created
    shutdown_coordinator = None

    try:
        # Load configuration
        app_info = AppInfoConfig()
        app_config = load_app_config(app_info=app_info)
        if hasattr(app_config, "__post_init__"):
            app_config.__post_init__()

        # Note: User settings will be applied through the SettingsService during initialization

        # Validate critical paths
        vosk_path = app_config.asset_paths.get_vosk_model_path()
        if not os.path.exists(vosk_path):
            logging.critical(f"Vosk model not found: {vosk_path}")
            logging.critical("Download models from: https://alphacephei.com/vosk/models")
            return

        # Setup logging and directories
        setup_logging(config=app_config.logging)
        os.makedirs(app_config.storage.user_data_root, exist_ok=True)

        # Initialize event system
        event_bus = EventBus()
        gui_event_loop = asyncio.new_event_loop()

        # Start GUI event loop in thread (not daemon to ensure proper cleanup)
        gui_thread = threading.Thread(
            target=lambda: (asyncio.set_event_loop(gui_event_loop), gui_event_loop.run_forever()),
            daemon=False,
            name="GUIEventLoop",
        )
        gui_thread.start()

        # Start event bus worker
        gui_event_loop.call_soon_threadsafe(lambda: gui_event_loop.create_task(event_bus.start_worker()))

        # Initialize UI
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        app_tk_root = ctk.CTk()
        app_tk_root.withdraw()
        app_tk_root.title("Iris")

        # Configure window
        app_tk_root.geometry(f"{ui_theme.theme.dimensions.main_window_width}x" f"{ui_theme.theme.dimensions.main_window_height}")
        app_tk_root.minsize(ui_theme.theme.dimensions.main_window_min_width, ui_theme.theme.dimensions.main_window_min_height)
        app_tk_root.resizable(False, False)

        # Setup icons FIRST and force render before creating child windows
        set_window_icon_robust(window=app_tk_root)
        app_tk_root.update_idletasks()
        app_tk_root.update()

        # Setup UI scheduler
        initialize_ui_scheduler(root_window=app_tk_root)

        # Create shutdown coordinator for centralized shutdown management
        shutdown_coordinator = ShutdownCoordinator(
            event_bus=event_bus, root_window=app_tk_root, logger=logging.getLogger("ShutdownCoordinator")
        )

        # Setup signal handlers to use shutdown coordinator
        def signal_handler(signum, frame):
            logging.info(f"Received signal {signum}")
            shutdown_coordinator.request_shutdown(reason=f"Received system signal {signum}", source="signal_handler")

            def force_exit():
                time.sleep(5)
                logging.error("Force exiting due to shutdown timeout")
                os._exit(1)

            threading.Thread(target=force_exit, daemon=True).start()

        signal.signal(signal.SIGINT, signal_handler)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, signal_handler)

        # Create startup window (will inherit icon from parent)
        from iris.app.ui.startup_window import StartupProgressTracker, StartupWindow

        startup_window = StartupWindow(
            logger=logging.getLogger("StartupWindow"),
            main_root=app_tk_root,
            asset_paths_config=app_config.asset_paths,
            shutdown_coordinator=shutdown_coordinator,
        )
        startup_window.show()

        # Force startup window to appear
        app_tk_root.update_idletasks()
        app_tk_root.update()

        # Initialize services with progress tracking
        # Total steps: 3 background (core, storage, audio) + 1 main thread (UI)
        progress_tracker = StartupProgressTracker(startup_window, total_steps=4)
        service_initializer = FastServiceInitializer(
            event_bus=event_bus,
            config=app_config,
            gui_loop=gui_event_loop,
            root=app_tk_root,
            shutdown_coordinator=shutdown_coordinator,
        )

        # Create initialization task and register it for cancellation on shutdown
        async def run_initialization():
            """Wrapped initialization that can be cleanly cancelled"""
            services = await initialize_services_with_ui_integration(
                service_initializer, progress_tracker, app_tk_root, startup_window
            )
            await service_initializer.initialize_ui_components(progress_tracker)
            return services

        init_task = asyncio.create_task(run_initialization())
        shutdown_coordinator.register_initialization_task(task=init_task)

        # Initialize services while keeping UI responsive
        try:
            services = await init_task
            shutdown_coordinator.unregister_initialization_task()
        except asyncio.CancelledError:
            logging.info("Initialization cancelled due to shutdown request")
            startup_window.update_progress(0.0, "Startup cancelled by user", animate=False)
            await asyncio.sleep(1)

            # Get whatever services were initialized before cancellation
            partial_services = service_initializer.services
            partial_services["gui_thread"] = gui_thread

            logging.info(f"Cleaning up {len(partial_services)} partially initialized services")
            startup_window.close()
            await _cleanup_services(
                services=partial_services, event_bus=event_bus, gui_event_loop=gui_event_loop, gui_thread=gui_thread
            )
            return
        except RuntimeError as e:
            logging.critical(f"Critical initialization error: {e}")
            logging.critical("Application will shut down")
            startup_window.update_progress(
                0.0, "Initialization failed. Please check your internet connection and try again.", animate=False
            )
            await asyncio.sleep(3)
            startup_window.close()
            await _cleanup_services(services={}, event_bus=event_bus, gui_event_loop=gui_event_loop, gui_thread=gui_thread)
            return

        # Store GUI thread reference for cleanup
        services["gui_thread"] = gui_thread

        # Show main window
        progress_tracker.update_status_static("Ready!")
        app_tk_root.deiconify()
        app_tk_root.lift()
        app_tk_root.focus_force()

        # Close startup window from main thread
        startup_window.update_progress(1.0, "Ready!", animate=False)
        await asyncio.sleep(0.5)
        startup_window.close()

        # Shutdown check mechanism using coordinator
        def check_shutdown():
            if shutdown_coordinator.is_shutdown_requested():
                logging.info("Shutdown detected via coordinator")
                return
            app_tk_root.after(100, check_shutdown)

        app_tk_root.after(100, check_shutdown)

        # Run main loop
        try:
            app_tk_root.mainloop()
        except KeyboardInterrupt:
            logging.info("KeyboardInterrupt received")
        finally:
            logging.info("Starting cleanup...")
            await _cleanup_services(services=services, event_bus=event_bus, gui_event_loop=gui_event_loop, gui_thread=gui_thread)

        logging.info("Application shutdown complete")

    except Exception as e:
        logging.exception(f"Unexpected error during application execution: {e}")


async def _cleanup_services(
    services: Dict[str, Any], event_bus: EventBus, gui_event_loop: asyncio.AbstractEventLoop, gui_thread: threading.Thread
):
    """Clean up all services during shutdown with proper async task cleanup"""
    cleanup_errors = []

    try:
        # Stop audio service first
        if "audio" in services and hasattr(services["audio"], "stop_processing"):
            services["audio"].stop_processing()

        await asyncio.sleep(0.3)

        # Stop event bus first while GUI loop is still running
        if not gui_event_loop.is_closed():
            try:
                stop_future = asyncio.run_coroutine_threadsafe(event_bus.stop_worker(), gui_event_loop)
                stop_future.result(timeout=5.0)
                logging.info("Event bus stopped successfully")
            except Exception as e:
                error_msg = f"Error stopping event bus: {e}"
                logging.error(error_msg)
                cleanup_errors.append(error_msg)

        # Stop mark service tasks on GUI loop
        if "mark" in services and hasattr(services["mark"], "stop_service_tasks"):
            try:
                if not gui_event_loop.is_closed():
                    stop_future = asyncio.run_coroutine_threadsafe(services["mark"].stop_service_tasks(), gui_event_loop)
                    stop_future.result(timeout=3)
            except Exception as e:
                error_msg = f"Error stopping mark service: {e}"
                logging.error(error_msg)
                cleanup_errors.append(error_msg)

        # Cancel all pending tasks in GUI event loop before closing
        if not gui_event_loop.is_closed():
            pending_tasks = []
            try:
                # Get all tasks from the GUI event loop
                for task in asyncio.all_tasks(gui_event_loop):
                    if not task.done():
                        pending_tasks.append(task)
                        task.cancel()

                # Wait for all tasks to be cancelled
                if pending_tasks:
                    logging.info(f"Cancelling {len(pending_tasks)} pending tasks...")
                    try:
                        cancel_future = asyncio.run_coroutine_threadsafe(
                            asyncio.gather(*pending_tasks, return_exceptions=True), gui_event_loop
                        )
                        cancel_future.result(timeout=2.0)
                        logging.info("All pending tasks cancelled successfully")
                    except asyncio.TimeoutError:
                        logging.warning("Timeout waiting for tasks to be cancelled")
            except Exception as e:
                logging.warning(f"Error cancelling pending tasks: {e}")

        # Now stop GUI event loop after event bus is shutdown
        if not gui_event_loop.is_closed():
            gui_event_loop.call_soon_threadsafe(gui_event_loop.stop)

            # Wait for GUI thread to terminate properly
            try:
                gui_thread.join(timeout=5.0)
                if gui_thread.is_alive():
                    logging.warning("GUI thread did not terminate cleanly within timeout")
                else:
                    logging.info("GUI thread terminated successfully")
            except Exception as e:
                error_msg = f"Error joining GUI thread: {e}"
                logging.error(error_msg)
                cleanup_errors.append(error_msg)

            await asyncio.sleep(0.3)

            # Close the event loop to free its resources
            if not gui_event_loop.is_closed():
                try:
                    gui_event_loop.close()
                    logging.info("GUI event loop closed")
                except Exception as e:
                    logging.warning(f"Error closing GUI event loop: {e}")

        # Shutdown services with shutdown methods in proper order
        shutdown_order = [
            "sound_service",
            "centralized_parser",
            "automation",
            "command_storage",
            "stt",
            "dictation",
            "markov_predictor",
            "audio",
        ]
        for service_name in shutdown_order:
            if service_name in services and hasattr(services[service_name], "shutdown"):
                try:
                    logging.info(f"Shutting down {service_name}...")
                    await services[service_name].shutdown()
                    logging.info(f"{service_name} shutdown completed")
                except Exception as e:
                    error_msg = f"Error shutting down {service_name}: {e}"
                    logging.error(error_msg, exc_info=True)
                    cleanup_errors.append(error_msg)

        # Explicitly delete all service references to free memory
        service_names_to_clear = list(services.keys())
        for service_name in service_names_to_clear:
            if service_name != "gui_thread":
                try:
                    del services[service_name]
                except Exception as e:
                    logging.warning(f"Error deleting service {service_name}: {e}")

        # Force aggressive garbage collection to free memory
        # Multiple rounds to catch cyclic references and C-extension cleanup
        for i in range(3):
            gc.collect()
            logging.info(f"Garbage collection round {i+1} performed")

        # Force Python to return memory to OS (if possible)
        try:
            if hasattr(ctypes, "pythonapi"):
                # Try to trim malloc arenas (glibc specific, may not work on all systems)
                try:
                    libc_name = ctypes.util.find_library("c")
                    if libc_name:
                        libc = ctypes.CDLL(libc_name)
                        if hasattr(libc, "malloc_trim"):
                            libc.malloc_trim(0)
                            logging.info("malloc_trim called to return memory to OS")
                except Exception as trim_error:
                    logging.debug(f"Could not call malloc_trim: {trim_error}")
        except Exception as e:
            logging.debug(f"Could not force memory return: {e}")

        if cleanup_errors:
            logging.warning(f"Cleanup completed with {len(cleanup_errors)} errors")
            for error in cleanup_errors:
                logging.error(f"Cleanup error: {error}")
        else:
            logging.info("All services cleaned up successfully")

        # Force immediate exit to prevent daemon threads from continuing
        # This is necessary because WhisperModel download creates daemon threads
        # that continue logging even after cleanup
        logging.info("Forcing immediate process termination")
        await asyncio.sleep(0.1)  # Brief pause for final log flush
        os._exit(0)

    except Exception as e:
        logging.error(f"Critical error during cleanup: {e}", exc_info=True)
        # Force exit even on error
        os._exit(1)


if __name__ == "__main__":
    asyncio.run(main())
