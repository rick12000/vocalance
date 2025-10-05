import time
import os
import sys
import logging
import argparse
import tkinter as tk
import asyncio
import threading
import concurrent.futures
from typing import Dict, Any, Optional
import signal

from iris.event_bus import EventBus
from iris.ui.main_window import AppControlRoom
from iris.config.app_config import GlobalAppConfig, load_app_config, AppInfoConfig
from iris.config.logging_config import setup_logging

# Core services
from iris.services.grid.grid_service import GridService
from iris.services.audio.sound_recognizer.streamlined_sound_service import StreamlinedSoundService
from iris.services.audio.simple_audio_service import SimpleAudioService
from iris.services.centralized_command_parser import CentralizedCommandParser
from iris.services.storage.unified_storage_service import UnifiedStorageService
from iris.services.storage.storage_adapters import StorageAdapterFactory
from iris.services.automation_service import AutomationService
from iris.services.audio.dictation_handling.dictation_coordinator import DictationCoordinator

# UI components
from iris.ui.startup_window import StartupWindow, StartupProgressTracker
from iris.ui.utils.ui_thread_utils import initialize_ui_scheduler
from iris.ui.utils.ui_icon_utils import set_window_icon_robust


class FastServiceInitializer:
    """Streamlined service initialization for maximum startup speed"""
    
    def __init__(self, event_bus: EventBus, config: GlobalAppConfig, gui_loop: asyncio.AbstractEventLoop, root: tk.Tk):
        self.event_bus = event_bus
        self.config = config
        self.gui_loop = gui_loop
        self.root = root
        self.services = {}
        
    async def initialize_all(self, progress_tracker: StartupProgressTracker) -> Dict[str, Any]:
        """Fast parallel initialization of all services"""
        
        # Step 1: Core infrastructure (fastest services first)
        progress_tracker.start_step("Starting core services...")
        progress_tracker.update_sub_step("Initializing grid service...")
        await self._init_core_services()
        progress_tracker.complete_step()
        
        # Step 2: Storage and data services (parallel)
        progress_tracker.start_step("Initializing storage...")
        progress_tracker.update_sub_step("Setting up unified storage...")
        await self._init_storage_services(progress_tracker)
        progress_tracker.complete_step()
        
        # Step 3: Audio and command services (parallel)
        progress_tracker.start_step("Starting audio processing...")
        progress_tracker.update_sub_step("Loading audio engines...")
        await self._init_audio_services(progress_tracker)
        progress_tracker.complete_step()
        
        # Step 4: UI components
        progress_tracker.start_step("Creating interface...")
        progress_tracker.update_sub_step("Building main window...")
        await self._init_ui_components()
        progress_tracker.complete_step()
        
        return self.services
    
    async def _init_core_services(self):
        """Initialize lightweight core services"""
        # Grid service
        self.services['grid'] = GridService(self.event_bus, self.config)
        self.services['grid'].setup_subscriptions()
        
        # Automation service
        self.services['automation'] = AutomationService(self.event_bus, self.config)
        self.services['automation'].setup_subscriptions()
    
    async def _init_storage_services(self, progress_tracker=None):
        """Initialize storage services in parallel"""
        if progress_tracker:
            progress_tracker.update_sub_step("Configuring storage backend...")
            

        
        # Unified storage
        self.services['unified_storage'] = UnifiedStorageService(
            self.event_bus, self.config
        )
        
        # Storage adapters
        self.services['storage_adapters'] = StorageAdapterFactory(
            self.services['unified_storage']
        )
        
        if progress_tracker:
            progress_tracker.update_sub_step("Loading data services...")
        
        # Individual storage services
        from iris.services.storage.settings_service import SettingsService
        from iris.services.command_management_service import CommandManagementService
        from iris.services.grid.click_tracker_service import ClickTrackerService
        from iris.services.mark_service import MarkService
        
        # Initialize storage services in parallel
        tasks = []
        
        async def init_settings():
            if progress_tracker:
                progress_tracker.update_sub_step("Loading user settings...")
                await asyncio.sleep(0.05)
            self.services['settings'] = SettingsService(
                self.event_bus, self.config, self.services['storage_adapters']
            )
            self.services['settings'].setup_subscriptions()
            await self.services['settings'].initialize()
            
            # Apply user settings to config
            self.config = await self.services['settings'].apply_settings_to_config(self.config)
        
        async def init_commands():
            if progress_tracker:
                progress_tracker.update_sub_step("Setting up command storage...")
                await asyncio.sleep(0.05)
            # Command storage adapter for direct access
            self.services['command_storage'] = self.services['storage_adapters'].get_command_adapter()
            # Command management service for event handling
            self.services['command_management'] = CommandManagementService(
                self.event_bus, self.config, self.services['storage_adapters']
            )
            self.services['command_management'].setup_subscriptions()
        
        async def init_click_tracker():
            if progress_tracker:
                progress_tracker.update_sub_step("Initializing click tracking...")
                await asyncio.sleep(0.05)
            self.services['click_tracker'] = ClickTrackerService(
                self.event_bus, self.config, self.services['storage_adapters']
            )
            self.services['click_tracker'].setup_subscriptions()
        
        async def init_marks():
            if progress_tracker:
                progress_tracker.update_sub_step("Configuring mark system...")
                await asyncio.sleep(0.05)
            # Collect trigger phrases for mark service
            trigger_phrases = {
                self.config.grid.show_grid_phrase,
                self.config.grid.select_cell_phrase,
                self.config.grid.cancel_grid_phrase,
                self.config.mark.triggers.create_mark,
                self.config.mark.triggers.delete_mark,
                self.config.dictation.start_trigger,
                self.config.dictation.stop_trigger,
                self.config.dictation.type_trigger,
                self.config.dictation.smart_start_trigger
            }
            trigger_phrases.update(self.config.mark.triggers.visualize_marks)
            trigger_phrases.update(self.config.mark.triggers.reset_marks)
            trigger_phrases.update(self.config.mark.triggers.visualization_cancel)
            
            self.services['mark'] = MarkService(
                self.event_bus, self.config, self.services['storage_adapters'],
                reserved_labels=trigger_phrases
            )
            self.services['mark'].setup_subscriptions()
        
        tasks.extend([init_settings(), init_commands(), init_click_tracker(), init_marks()])
        await asyncio.gather(*tasks)
    
    async def _init_audio_services(self, progress_tracker=None):
        """Initialize audio services in parallel"""
        tasks = []
        
        async def init_audio():
            if progress_tracker:
                progress_tracker.update_sub_step("Starting audio capture...")
                await asyncio.sleep(0.1)  # Brief pause to show progress
            self.services['audio'] = SimpleAudioService(
                self.event_bus, self.config, self.gui_loop
            )
            self.services['audio'].setup_subscriptions()
            self.services['audio'].start_processing()
        
        async def init_sound():
            if progress_tracker:
                progress_tracker.update_sub_step("Loading sound recognition...")
                await asyncio.sleep(0.1)
            self.services['sound_service'] = StreamlinedSoundService(
                self.event_bus, self.config, self.services['storage_adapters']
            )
            await self.services['sound_service'].initialize()
        
        async def init_stt():
            if progress_tracker:
                progress_tracker.update_sub_step("Initializing speech-to-text...")
                await asyncio.sleep(0.1)
            from iris.services.audio.stt_service import SpeechToTextService
            self.services['stt'] = SpeechToTextService(self.event_bus, self.config)
            self.services['stt'].initialize_engines()
            self.services['stt'].setup_subscriptions()
        
        async def init_command_parser():
            if progress_tracker:
                progress_tracker.update_sub_step("Setting up command processing...")
                await asyncio.sleep(0.1)
            self.services['centralized_parser'] = CentralizedCommandParser(
                self.event_bus, self.config, self.services['command_storage']
            )
            await self.services['centralized_parser'].initialize()
            self.services['centralized_parser'].setup_subscriptions()
        
        async def init_dictation():
            if progress_tracker:
                progress_tracker.update_sub_step("Preparing dictation system...")
                await asyncio.sleep(0.1)
            self.services['dictation'] = DictationCoordinator(
                self.event_bus, self.config, self.services['storage_adapters'], self.gui_loop
            )
            self.services['dictation'].setup_subscriptions()
            
            # Initialize LLM based on startup mode
            llm_mode = getattr(self.config.llm, 'startup_mode', 'startup')
            if llm_mode == 'startup':
                if progress_tracker:
                    progress_tracker.update_sub_step("Loading AI model...")
                    await asyncio.sleep(0.1)
                await self.services['dictation'].initialize()
            elif llm_mode == 'background':
                asyncio.create_task(self._background_llm_init())
        
        async def init_markov_predictor():
            if progress_tracker:
                progress_tracker.update_sub_step("Initializing command predictor...")
                await asyncio.sleep(0.05)
            from iris.services.markov_command_predictor import MarkovCommandPredictor
            history_adapter = self.services['storage_adapters'].get_command_history_adapter()
            self.services['markov_predictor'] = MarkovCommandPredictor(
                self.event_bus, self.config, history_adapter
            )
            self.services['markov_predictor'].setup_subscriptions()
            await self.services['markov_predictor'].initialize()
        
        tasks.extend([init_audio(), init_sound(), init_stt(), init_command_parser(), init_dictation(), init_markov_predictor()])
        await asyncio.gather(*tasks)
    
    async def _background_llm_init(self):
        """Initialize LLM in background after startup"""
        await asyncio.sleep(2.0)
        await self.services['dictation'].initialize()
        logging.info("LLM initialized in background")
    
    async def _init_ui_components(self):
        """Initialize UI components"""
        # Initialize font service early
        from iris.ui.utils.font_service import get_font_service
        font_service = get_font_service()
        font_service.load_fonts()
        
        control_room_logger = logging.getLogger("AppControlRoom")
        self.services['control_room'] = AppControlRoom(
            self.root, self.event_bus, self.gui_loop, control_room_logger, self.config
        )
        
        # Pass settings service to control room so it can be used by settings controller
        if 'settings' in self.services:
            self.services['control_room'].set_settings_service(self.services['settings'])
        
        # Start background tasks
        if 'mark' in self.services:
            self.gui_loop.create_task(self.services['mark'].start_service_tasks())


async def main():
    """Main application entry point"""
    logger = logging.getLogger(__name__)
    logging.getLogger('numba').setLevel(logging.WARNING)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Voice Controlled Application")
    parser.add_argument("--dev-cache", action="store_true", 
                       help="Enable development cache mode")
    args = parser.parse_args()
    
    # Global shutdown handling
    shutdown_requested = False
    
    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        logging.info(f"Received signal {signum}. Requesting shutdown...")
        shutdown_requested = True
        
        def force_exit():
            import time
            time.sleep(5)
            logging.error("Force exiting due to shutdown timeout")
            os._exit(1)
        threading.Thread(target=force_exit, daemon=True).start()
    
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Load configuration
        app_info = AppInfoConfig()
        app_config = load_app_config(app_info=app_info)
        if hasattr(app_config, '__post_init__'):
            app_config.__post_init__()
        
        # Note: User settings will be applied through the SettingsService during initialization
        
        # Validate critical paths
        if not os.path.exists(app_config.model_paths.vosk_model):
            logging.critical(f"Vosk model not found: {app_config.model_paths.vosk_model}")
            logging.critical("Download models from: https://alphacephei.com/vosk/models")
            return
        
        # Setup logging and directories
        setup_logging(app_config.logging)
        os.makedirs(app_config.storage.user_data_root, exist_ok=True)
        
        # Initialize event system
        event_bus = EventBus()
        gui_event_loop = asyncio.new_event_loop()
        
        # Start GUI event loop in thread (not daemon to ensure proper cleanup)
        gui_thread = threading.Thread(
            target=lambda: (asyncio.set_event_loop(gui_event_loop), gui_event_loop.run_forever()),
            daemon=False,
            name="GUIEventLoop"
        )
        gui_thread.start()
        
        # Start event bus worker
        gui_event_loop.call_soon_threadsafe(
            lambda: gui_event_loop.create_task(event_bus.start_worker())
        )
        
        # Initialize UI
        import customtkinter as ctk
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        app_tk_root = ctk.CTk()
        app_tk_root.withdraw()
        app_tk_root.title("Iris")
        
        # Configure window
        from iris.ui import ui_theme
        app_tk_root.geometry(
            f"{ui_theme.theme.dimensions.main_window_width}x"
            f"{ui_theme.theme.dimensions.main_window_height}"
        )
        app_tk_root.minsize(
            ui_theme.theme.dimensions.main_window_min_width,
            ui_theme.theme.dimensions.main_window_min_height
        )
        
        # Setup icons and UI scheduler
        set_window_icon_robust(app_tk_root)
        initialize_ui_scheduler(app_tk_root)
        
        # Create startup window
        startup_window = StartupWindow(logging.getLogger("StartupWindow"), app_tk_root)
        startup_window.show()
        
        # Force startup window to appear
        app_tk_root.update_idletasks()
        app_tk_root.update()
        
        # Initialize services with progress tracking
        progress_tracker = StartupProgressTracker(startup_window, total_steps=4)
        service_initializer = FastServiceInitializer(event_bus, app_config, gui_event_loop, app_tk_root)
        
        services = await service_initializer.initialize_all(progress_tracker)
        
        # Store GUI thread reference for cleanup
        services['gui_thread'] = gui_thread
        
        # Show main window
        progress_tracker.start_step("Launching interface...")
        app_tk_root.deiconify()
        app_tk_root.lift()
        app_tk_root.focus_force()
        progress_tracker.complete_step()
        progress_tracker.finish()
        
        # Brief delay for startup window to close
        await asyncio.sleep(0.3)
        
        # Shutdown check mechanism
        def check_shutdown():
            if shutdown_requested:
                logging.info("Shutdown requested")
                app_tk_root.quit()
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
            await _cleanup_services(services, event_bus, gui_event_loop, gui_thread)
        
        logging.info("Application shutdown complete")
        
    except Exception:
        logging.exception("Unexpected error during application execution")





async def _cleanup_services(services: Dict[str, Any], event_bus: EventBus, gui_event_loop: asyncio.AbstractEventLoop, gui_thread: threading.Thread):
    """Clean up all services during shutdown"""
    cleanup_errors = []
    
    try:
        # Stop audio service first
        if 'audio' in services and hasattr(services['audio'], 'stop_processing'):
            services['audio'].stop_processing()
        
        await asyncio.sleep(0.3)
        
        # Stop event bus first while GUI loop is still running
        if not gui_event_loop.is_closed():
            try:
                stop_future = asyncio.run_coroutine_threadsafe(
                    event_bus.stop_worker(), gui_event_loop
                )
                stop_future.result(timeout=5.0)
                logging.info("Event bus stopped successfully")
            except Exception as e:
                error_msg = f"Error stopping event bus: {e}"
                logging.error(error_msg)
                cleanup_errors.append(error_msg)
        
        # Stop mark service tasks on GUI loop
        if 'mark' in services and hasattr(services['mark'], 'stop_service_tasks'):
            try:
                if not gui_event_loop.is_closed():
                    stop_future = asyncio.run_coroutine_threadsafe(
                        services['mark'].stop_service_tasks(), gui_event_loop
                    )
                    stop_future.result(timeout=3)
            except Exception as e:
                error_msg = f"Error stopping mark service: {e}"
                logging.error(error_msg)
                cleanup_errors.append(error_msg)
        
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
        
        # Shutdown services with shutdown methods in proper order
        shutdown_order = ['sound_service', 'centralized_parser', 'automation', 'command_storage', 'stt', 'dictation', 'markov_predictor']
        for service_name in shutdown_order:
            if service_name in services and hasattr(services[service_name], 'shutdown'):
                try:
                    logging.info(f"Shutting down {service_name}...")
                    await services[service_name].shutdown()
                    logging.info(f"{service_name} shutdown completed")
                except Exception as e:
                    error_msg = f"Error shutting down {service_name}: {e}"
                    logging.error(error_msg, exc_info=True)
                    cleanup_errors.append(error_msg)
        
        if cleanup_errors:
            logging.warning(f"Cleanup completed with {len(cleanup_errors)} errors")
            for error in cleanup_errors:
                logging.error(f"Cleanup error: {error}")
        else:
            logging.info("All services cleaned up successfully")
            
    except Exception as e:
        logging.error(f"Critical error during cleanup: {e}", exc_info=True)
        # Continue with cleanup even if there are errors


if __name__ == "__main__":
    asyncio.run(main())
