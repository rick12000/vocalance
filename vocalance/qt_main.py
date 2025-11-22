"""Qt-based main application entry point.

Coordinates application startup, initialization, and shutdown using PySide6.
Integrates Qt event loop with asyncio for service coordination.
"""

import asyncio
import logging
import os
import sys
import threading
from typing import Any, Dict, Optional

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from vocalance.app.config.app_config import AppInfoConfig, GlobalAppConfig, load_app_config
from vocalance.app.config.logging_config import setup_logging
from vocalance.app.event_bus import EventBus
from vocalance.app.services.shutdown_coordinator import ShutdownCoordinator
from vocalance.app.ui.qt_main_window import VocalanceMainWindow
from vocalance.app.ui.qt_startup_window import StartupProgressTracker, StartupWindow
from vocalance.app.ui.qt_theme import theme
from vocalance.main import FastServiceInitializer, _cleanup_services, _validate_critical_assets

logger = logging.getLogger(__name__)


def _apply_base_styles(app: QApplication, fonts_dir: str) -> None:
    """Apply base stylesheet to QApplication with color resets and font defaults.

    This applies minimal global styling to establish base colors and fonts.
    Specific component styling is handled by individual components.
    """
    c = theme.config

    base_stylesheet = f"""
    /* Global Reset */
    * {{
        outline: none;
        border: none;
        background: transparent;
        color: {c.text.lightest};
        selection-background-color: {c.shapes.accent};
        selection-color: {c.text.light_blue_accent};
    }}

    QMainWindow {{
        background-color: {c.shapes.darkest};
    }}

    QWidget {{
        font-family: "{c.font_family_primary}", "{c.font_family_secondary}";
        font-size: {c.fonts.medium}px;
    }}

    QLabel[variant="title"] {{
        font-size: {c.fonts.xxlarge}px;
        font-weight: bold;
    }}

    QLabel[variant="subtitle"] {{
        color: {c.text.light};
        font-size: {c.fonts.large}px;
    }}
    """

    app.setStyleSheet(base_stylesheet)


async def initialize_services_with_ui_integration(
    initializer: FastServiceInitializer,
    progress_tracker: StartupProgressTracker,
    qt_app: QApplication,
    startup_window: Optional[StartupWindow] = None,
) -> Dict[str, Any]:
    """Initialize services while maintaining UI responsiveness during startup.

    Executes service initialization asynchronously while periodically processing
    Qt GUI events to prevent the UI from freezing.

    Args:
        initializer: FastServiceInitializer instance managing service creation sequence.
        progress_tracker: Monitors initialization steps and publishes progress updates.
        qt_app: QApplication instance requiring periodic event processing.
        startup_window: Optional progress indicator window displaying initialization status.

    Returns:
        Dictionary mapping service names to initialized service instances.
    """
    init_task = asyncio.create_task(initializer.initialize_all(progress_tracker=progress_tracker))
    gui_update_interval = 0.01

    while not init_task.done():
        try:
            # Process Qt events
            qt_app.processEvents()

            # Process startup window queue if available
            if startup_window:
                # Qt signals handle updates automatically, no manual queue processing needed
                pass
        except Exception as e:
            logger.debug(f"GUI update failed (window may be closing): {e}")
            break

        await asyncio.sleep(gui_update_interval)

    return await init_task


async def _handle_initialization(
    init_task: asyncio.Task,
    service_initializer: FastServiceInitializer,
    startup_window: StartupWindow,
    shutdown_coordinator: Optional[ShutdownCoordinator],
    event_bus: EventBus,
    gui_event_loop: asyncio.AbstractEventLoop,
    gui_thread: threading.Thread,
) -> Optional[Dict[str, Any]]:
    """Handle service initialization with proper error handling and cancellation support.

    Awaits the initialization task and handles three outcomes: successful completion,
    cancellation due to user request, or runtime error.

    Args:
        init_task: Asyncio task executing the initialization sequence.
        service_initializer: Service initializer containing partially initialized services.
        startup_window: Progress window displaying initialization status to user.
        shutdown_coordinator: Optional coordinator tracking initialization task for cancellation.
        event_bus: Application event bus requiring cleanup on failure.
        gui_event_loop: GUI event loop requiring cleanup on failure.
        gui_thread: GUI thread requiring cleanup on failure.

    Returns:
        Dictionary of initialized services if successful, None if cancelled or failed.
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

        partial_services = service_initializer.services
        partial_services["gui_thread"] = gui_thread

        logger.debug(f"Cleaning up {len(partial_services)} partially initialized services")
        startup_window.close()
        await _cleanup_services(
            services=partial_services,
            event_bus=event_bus,
            gui_event_loop=gui_event_loop,
            gui_thread=gui_thread,
        )
        return None

    except RuntimeError as e:
        logger.critical(f"Critical initialization error: {e}")
        logger.critical("Application will shut down")
        startup_window.update_progress(
            0.0,
            "Initialization failed. Please check your internet connection and try again.",
            animate=False,
        )
        await asyncio.sleep(3)
        startup_window.close()
        await _cleanup_services(
            services={},
            event_bus=event_bus,
            gui_event_loop=gui_event_loop,
            gui_thread=gui_thread,
        )
        return None


def _setup_infrastructure(app_config: GlobalAppConfig) -> tuple[EventBus, asyncio.AbstractEventLoop, threading.Thread]:
    """Setup core infrastructure: event bus, GUI event loop, and GUI thread.

    Creates a dedicated asyncio event loop running in a separate daemon thread for
    GUI-related async operations.

    Args:
        app_config: Application configuration.

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


class QtAsyncioIntegration:
    """Integrates Qt event loop with asyncio event loop.

    Uses QTimer to periodically process asyncio events while Qt runs.
    """

    def __init__(self, gui_event_loop: asyncio.AbstractEventLoop):
        """Initialize integration.

        Args:
            gui_event_loop: Asyncio event loop to integrate with Qt.
        """
        self.gui_event_loop = gui_event_loop
        self.timer = QTimer()
        self.timer.timeout.connect(self._process_asyncio_events)
        self.timer.start(10)  # Process every 10ms

    def _process_asyncio_events(self):
        """Process asyncio events in the GUI event loop."""
        if not self.gui_event_loop.is_closed():
            self.gui_event_loop.call_soon_threadsafe(lambda: None)

    def stop(self):
        """Stop the integration timer."""
        self.timer.stop()


async def main() -> None:
    """Main application entry point orchestrating startup, initialization, and cleanup.

    Coordinates the complete application lifecycle using Qt:
    - Configures logging
    - Validates assets
    - Sets up infrastructure (event bus, GUI thread)
    - Creates Qt application and main window
    - Initializes all services with progress tracking
    - Activates services
    - Displays the main window
    - Runs the Qt event loop
    - Performs cleanup on shutdown
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

        # Create Qt Application
        qt_app = QApplication(sys.argv)
        qt_app.setStyle("Fusion")  # Modern Qt style

        # Load fonts and apply base styles
        theme.load_fonts(app_config.asset_paths.fonts_dir)

        # Apply global base stylesheet to QApplication
        _apply_base_styles(qt_app, app_config.asset_paths.fonts_dir)

        # Create infrastructure
        event_bus, gui_event_loop, gui_thread = _setup_infrastructure(app_config=app_config)

        # Create shutdown coordinator (adapted for Qt)
        # We'll need to adapt ShutdownCoordinator to work with Qt
        shutdown_coordinator = None  # Placeholder - will need Qt adaptation

        # Setup signal handlers
        # _setup_signal_handlers(shutdown_coordinator=shutdown_coordinator)

        # Show startup window
        startup_window = StartupWindow(
            logger=logging.getLogger("StartupWindow"),
            asset_paths_config=app_config.asset_paths,
            shutdown_coordinator=shutdown_coordinator,
        )
        startup_window.show()

        # Process events to show window
        qt_app.processEvents()

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
            root=None,  # No Tkinter root in Qt
            shutdown_coordinator=shutdown_coordinator,
        )

        # Adapt FastServiceInitializer to not use Tkinter root
        # For now, initialize without UI components that depend on Tkinter

        async def run_initialization() -> Dict[str, Any]:
            services = await initialize_services_with_ui_integration(
                initializer=service_initializer,
                progress_tracker=progress_tracker,
                qt_app=qt_app,
                startup_window=startup_window,
            )
            # Skip UI components initialization for now - will need adaptation
            # await service_initializer.initialize_ui_components(progress_tracker=progress_tracker)
            return services

        init_task = asyncio.create_task(run_initialization())

        services = await _handle_initialization(
            init_task=init_task,
            service_initializer=service_initializer,
            startup_window=startup_window,
            shutdown_coordinator=shutdown_coordinator if shutdown_coordinator else None,
            event_bus=event_bus,
            gui_event_loop=gui_event_loop,
            gui_thread=gui_thread,
        )

        if not services:
            return

        services["gui_thread"] = gui_thread

        progress_tracker.update_status_static(status="Ready!")
        startup_window.update_progress(1.0, "Ready!", animate=False)

        await asyncio.sleep(0.5)
        startup_window.close_after_initialization()

        await asyncio.sleep(0.1)

        logger.info("Activating services now that initialization is complete")
        await service_initializer.activate_all_services()

        # Create main window
        main_window = VocalanceMainWindow(
            event_bus=event_bus,
            event_loop=gui_event_loop,
            logger=logging.getLogger("MainWindow"),
            config=app_config,
            storage_service=services.get("storage"),
        )

        # Set all services for controller initialization
        if "settings" in services:
            main_window.set_settings_service(services["settings"])
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

        # Initialize controllers now that services are available
        main_window.initialize_controllers_with_services()

        # Setup Qt-asyncio integration
        asyncio_integration = QtAsyncioIntegration(gui_event_loop)

        # Show main window
        main_window.show()
        main_window.raise_()
        main_window.activateWindow()

        logger.info("Main window displayed, entering Qt event loop")

        # Run Qt event loop
        qt_app.exec()

        # Cleanup
        asyncio_integration.stop()
        logger.info("Qt event loop exited, starting cleanup...")
        await _cleanup_services(
            services=services,
            event_bus=event_bus,
            gui_event_loop=gui_event_loop,
            gui_thread=gui_thread,
        )

        logger.info("Application shutdown complete")

    except Exception as e:
        logger.exception(f"Unexpected error during application execution: {e}")


if __name__ == "__main__":
    # Run main coroutine
    asyncio.run(main())
