import asyncio
import logging
import threading
import tkinter as tk
from typing import Optional

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.event_bus import EventBus
from vocalance.app.ui import ui_theme
from vocalance.app.ui.controls.commands_control import CommandsController
from vocalance.app.ui.controls.dictation_control import DictationController
from vocalance.app.ui.controls.dictation_popup_control import DictationPopupController
from vocalance.app.ui.controls.grid_control import GridController
from vocalance.app.ui.controls.marks_control import MarksController
from vocalance.app.ui.controls.settings_control import SettingsController
from vocalance.app.ui.controls.sound_control import SoundController
from vocalance.app.ui.controls.system_control import SystemController
from vocalance.app.ui.utils.font_service import FontService
from vocalance.app.ui.utils.logo_service import LogoService
from vocalance.app.ui.utils.ui_assets import AssetCache
from vocalance.app.ui.utils.ui_icon_utils import set_window_icon_robust
from vocalance.app.ui.views.commands_view import CommandsView
from vocalance.app.ui.views.components import themed_dialogs
from vocalance.app.ui.views.components.themed_components import (
    SidebarButtonManager,
    SidebarIconButton,
    ThemedFrame,
    ThemedLabel,
    TransparentFrame,
)
from vocalance.app.ui.views.dictation_popup_view import DictationPopupView
from vocalance.app.ui.views.dictation_view import DictationView
from vocalance.app.ui.views.grid_view import GridView
from vocalance.app.ui.views.mark_view import MarkView
from vocalance.app.ui.views.marks_view import MarksView
from vocalance.app.ui.views.settings_view import SettingsView
from vocalance.app.ui.views.sound_view import SoundView


class AppControlRoom:
    """Main application control room managing UI views and controllers.

    Orchestrates the main window UI with sidebar navigation, lazy-loaded tab views,
    and specialized overlay windows (dictation popup, grid overlay, mark visualization).
    Coordinates between UI views, controllers, and the event bus for a fully integrated
    user experience. Thread-safe view caching and tab switching.

    Attributes:
        asset_cache: AssetCache for icons and images.
        font_service: FontService for loading custom fonts.
        logo_service: LogoService for app logo.
        _view_cache: Dict of lazily-loaded view instances.
        _controllers: Dict of controller instances by tab name.
    """

    def __init__(
        self,
        root: tk.Tk,
        event_bus: EventBus,
        event_loop: asyncio.AbstractEventLoop,
        logger: logging.Logger,
        config: GlobalAppConfig,
        storage_service=None,
    ) -> None:
        """Initialize application control room with UI setup.

        Args:
            root: Tkinter root window.
            event_bus: EventBus for pub/sub messaging.
            event_loop: Asyncio event loop for async operations.
            logger: Logger instance.
            config: Global application configuration.
            storage_service: Optional storage service reference.
        """
        self.root = root
        self.event_bus = event_bus
        self.event_loop = event_loop
        self.logger = logger
        self.config = config
        self.current_tab = "Marks"
        self._settings_service = None
        self._storage_service = storage_service

        self.asset_cache = AssetCache(asset_paths_config=self.config.asset_paths)
        self.font_service = FontService(self.config.asset_paths)
        self.logo_service = LogoService(self.asset_cache)

        ui_theme.theme.font_family.set_font_service(self.font_service)

        self._view_cache_lock = threading.RLock()
        self._view_cache = {}
        self._current_view = None

        self._initialize_controllers()
        self._initialize_specialized_views()
        self._setup_main_window()
        self._build_ui()

        self.logger.debug("Control Room Initialized.")

    def set_settings_service(self, settings_service):
        """Set the settings service reference for controllers to use"""
        self._settings_service = settings_service
        if hasattr(self, "settings_controller"):
            self.settings_controller.set_settings_service(settings_service)

    def _initialize_controllers(self):
        """Initialize all controllers"""
        try:
            self.marks_controller = MarksController(self.event_bus, self.event_loop, self.logger, self.config)
            self.sound_controller = SoundController(self.event_bus, self.event_loop, self.logger)
            self.dictation_controller = DictationController(self.event_bus, self.event_loop, self.logger, self.config)

            settings_service = getattr(self, "_settings_service", None)
            self.settings_controller = SettingsController(
                self.event_bus, self.event_loop, self.logger, self.config, settings_service
            )

            self.commands_controller = CommandsController(self.event_bus, self.event_loop, self.logger)
            self.grid_controller = GridController(self.event_bus, self.event_loop, self.logger)
            self.system_controller = SystemController(self.event_bus, self.root, self.event_loop, self.logger)
            self.dictation_popup_controller = DictationPopupController(self.event_bus, self.event_loop, self.logger)

            self.dictation_controller.set_view_callback(self)
            self.settings_controller.set_view_callback(self)
            self.grid_controller.set_view_callback(self)
            self.system_controller.set_view_callback(self)

            self.logger.debug("Controllers initialized")

        except Exception as e:
            self.logger.error(f"Error initializing controllers: {e}", exc_info=True)
            raise

    def _initialize_specialized_views(self):
        """Initialize specialized views that need direct controller connection"""
        try:
            self.grid_view = GridView(
                root=self.root,
                event_bus=self.event_bus,
                default_num_rects=self.config.grid.default_rect_count,
                event_loop=self.event_loop,
                storage=self._storage_service,
            )
            self.grid_controller.set_grid_view(self.grid_view)

            if self._storage_service:
                asyncio.create_task(self.grid_view.initialize_click_cache())

            self.mark_view = MarkView(root=self.root)
            self.marks_controller.set_mark_view(self.mark_view)

            self.dictation_popup_view = DictationPopupView(parent_root=self.root, controller=self.dictation_popup_controller)

            self.logger.debug("Specialized views initialized")

        except Exception as e:
            self.logger.error(f"Error initializing specialized views: {e}", exc_info=True)
            raise

    def _setup_main_window(self):
        """Set up main window properties"""
        try:
            set_window_icon_robust(self.root)

            self.logger.debug("Main window setup completed")

        except Exception as e:
            self.logger.error(f"Error setting up main window: {e}", exc_info=True)

    def _build_ui(self):
        # Root background frame with no corner radius to extend to borders
        # Configure root grid
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Use shape colors directly for background
        self.bg_frame = ThemedFrame(
            self.root,
            fg_color=ui_theme.theme.shape_colors.darkest,  # Use shape_colors.darkest for content background
            corner_radius=0,
        )
        self.bg_frame.grid(row=0, column=0, sticky="nsew")

        # Configure grid
        self.bg_frame.grid_rowconfigure(0, weight=0)  # Header
        self.bg_frame.grid_rowconfigure(1, weight=1)  # Content
        self.bg_frame.grid_columnconfigure(
            0, weight=0, minsize=ui_theme.theme.sidebar_layout.grid_column_minsize
        )  # Sidebar column matches sidebar width
        self.bg_frame.grid_columnconfigure(1, weight=0, minsize=1)  # Separator column - 1px wide
        self.bg_frame.grid_columnconfigure(2, weight=1, minsize=300)  # Content column keeps minimum

        self.create_sidebar()
        self.create_sidebar_separator()
        self.create_header()
        self.create_content_area()

        self.show_tab("Marks")

    def create_sidebar(self):
        """Create the sidebar with navigation buttons"""
        sidebar_config = ui_theme.theme.sidebar_layout
        self.sidebar = ThemedFrame(
            self.bg_frame,
            width=sidebar_config.width,
            corner_radius=0,
            fg_color=ui_theme.theme.shape_colors.darkest,
            border_width=0,
        )
        self.sidebar.grid(
            row=0,
            column=0,
            rowspan=2,
            sticky="nsew",
            padx=(sidebar_config.container_padding_left, sidebar_config.container_padding_right),
            pady=(sidebar_config.container_padding_top, sidebar_config.container_padding_bottom),
        )
        self.sidebar.grid_propagate(False)

        self.sidebar.grid_rowconfigure(0, weight=0, minsize=sidebar_config.top_spacing)
        self.sidebar.grid_rowconfigure(1, weight=1)
        self.sidebar.grid_rowconfigure(2, weight=0)
        self.sidebar.grid_columnconfigure(0, weight=1)

        self._create_sidebar_buttons()

        self._create_sidebar_logo()

    def create_sidebar_separator(self):
        """Create a separator line between sidebar and main content"""
        import tkinter as tk

        sidebar_config = ui_theme.theme.sidebar_layout
        self.sidebar_separator = tk.Frame(
            self.bg_frame,
            width=sidebar_config.border_width,
            bg=sidebar_config.border_color,
            highlightthickness=0,
            bd=0,
        )
        self.sidebar_separator.grid(
            row=0,
            column=1,
            rowspan=2,
            sticky="ns",
        )

    def _create_sidebar_buttons(self):
        """Create sidebar navigation buttons"""
        buttons_frame = TransparentFrame(self.sidebar)
        buttons_frame.grid(row=1, column=0, sticky="new")
        buttons_frame.grid_columnconfigure(0, weight=1)

        self.sidebar_buttons = {}
        self.sidebar_button_manager = SidebarButtonManager()

        tabs = [
            ("Marks", ui_theme.theme.sidebar_icons.marks),
            ("Sounds", ui_theme.theme.sidebar_icons.sounds),
            ("Commands", ui_theme.theme.sidebar_icons.commands),
            ("Dictation", ui_theme.theme.sidebar_icons.dictation),
            ("Settings", ui_theme.theme.sidebar_icons.settings),
        ]

        for i, (tab_name, icon_filename) in enumerate(tabs):
            buttons_frame.grid_rowconfigure(i, weight=0)
            btn = SidebarIconButton(
                buttons_frame,
                text=tab_name,
                icon_filename=icon_filename,
                command=lambda tab=tab_name: self.show_tab(tab),
                asset_paths_config=self.config.asset_paths,
            )
            btn.grid(row=i, column=0, sticky="ew")
            self.sidebar_buttons[tab_name] = btn
            self.sidebar_button_manager.add_button(btn)

        if tabs:
            first_button = self.sidebar_buttons[tabs[0][0]]
            self.sidebar_button_manager.select_button(first_button)

    def _create_sidebar_logo(self):
        """Create sidebar logo using centralized logo service"""
        sidebar_config = ui_theme.theme.sidebar_layout

        self.sidebar_logo = self.logo_service.create_logo_widget(
            self.sidebar, max_size=sidebar_config.logo_max_size, context="sidebar", text_fallback="Vocalance", logo_type="icon"
        )

        self.sidebar_logo.grid(row=2, column=0, pady=(sidebar_config.logo_padding_top, sidebar_config.logo_padding_bottom))

        self.logger.debug("Sidebar logo created")

    def create_header(self):
        """Create the header section"""
        header_config = ui_theme.theme.header_layout
        self.header = ThemedFrame(
            self.bg_frame,
            height=ui_theme.theme.dimensions.header_height,
            fg_color=ui_theme.theme.shape_colors.darkest,
            corner_radius=ui_theme.theme.border_radius.xlarge,
            border_width=header_config.border_width,
            border_color=header_config.border_color,
        )
        self.header.grid(row=0, column=2, sticky="ew", padx=header_config.frame_padx, pady=header_config.frame_pady)
        self.header.grid_propagate(False)

        self.header.grid_columnconfigure(0, weight=1)
        self.header.grid_rowconfigure(0, weight=0)
        self.header.grid_rowconfigure(1, weight=1)

        self.header_label = ThemedLabel(
            self.header,
            text="Welcome to Vocalance!",
            size=ui_theme.theme.font_sizes.xxlarge,
            color=ui_theme.theme.text_colors.lightest,
            bold=True,
        )
        self.header_label.grid(row=0, column=0, padx=header_config.title_padx, pady=(header_config.title_y_offset, 0), sticky="w")

    def create_content_area(self):
        """Create the main content area"""
        self.content_frame = TransparentFrame(self.bg_frame)
        self.content_frame.grid(
            row=1,
            column=2,
            sticky="nsew",
            padx=ui_theme.theme.layout.content_area_padding_x,
            pady=ui_theme.theme.layout.content_area_padding_y,
        )

        # Configure content frame grid
        self.content_frame.grid_rowconfigure(0, weight=1)
        self.content_frame.grid_columnconfigure(0, weight=1)

    def _set_header_subtitle(self, text):
        """Set or update the header subtitle"""
        if not hasattr(self, "header_subtitle"):
            header_config = ui_theme.theme.header_layout
            self.header_subtitle = ThemedLabel(
                self.header, text=text, size=ui_theme.theme.font_sizes.medium, color=ui_theme.theme.text_colors.light
            )
            self.header_subtitle.grid(
                row=1,
                column=0,
                padx=header_config.subtitle_padx,
                pady=(header_config.subtitle_y_offset - header_config.title_y_offset, 0),
                sticky="nw",
            )
        else:
            self.header_subtitle.configure(text=text)

    def show_tab(self, tab_name):
        """Show the specified tab with view caching for performance. Thread-safe."""
        self.current_tab = tab_name

        # Update header
        subtitles = {
            "Sounds": "Use custom sounds to control your computer",
            "Marks": "Pinpoint important locations on your screen",
            "Commands": "Manage voice commands and their actions",
            "Dictation": "Configure smart dictation with AI prompts",
            "Settings": "Configure default Vocalance settings",
        }

        self.header_label.configure(text=tab_name)
        if tab_name in subtitles:
            self._set_header_subtitle(subtitles[tab_name])

        with self._view_cache_lock:
            if self._current_view is not None:
                try:
                    self._current_view.grid_remove()
                except Exception as e:
                    self.logger.debug(f"Error hiding current view: {e}")

            view_cached = tab_name in self._view_cache

        if not view_cached:
            self.logger.debug(f"Creating new view for tab: {tab_name}")
            tab_creators = {
                "Sounds": self.create_sounds_tab,
                "Marks": self.create_marks_tab,
                "Commands": self.create_commands_tab,
                "Dictation": self.create_dictation_tab,
                "Settings": self.create_settings_tab,
            }

            if tab_name in tab_creators:
                tab_creators[tab_name]()
        else:
            self.logger.debug(f"Reusing cached view for tab: {tab_name}")
            with self._view_cache_lock:
                cached_view = self._view_cache[tab_name]
                self._current_view = cached_view
            cached_view.grid(row=0, column=0, sticky="nsew")

    def create_sounds_tab(self):
        """Create the sounds tab and cache it. Thread-safe."""
        self.sound_view = SoundView(self.content_frame, self.sound_controller, self.root)
        self.sound_view.grid(row=0, column=0, sticky="nsew")
        with self._view_cache_lock:
            self._view_cache["Sounds"] = self.sound_view
            self._current_view = self.sound_view

    def create_marks_tab(self):
        """Create the marks tab and cache it. Thread-safe."""
        self.marks_view = MarksView(self.content_frame, self.marks_controller, self.root)
        self.marks_view.grid(row=0, column=0, sticky="nsew")
        with self._view_cache_lock:
            self._view_cache["Marks"] = self.marks_view
            self._current_view = self.marks_view

    def create_settings_tab(self):
        """Create the settings tab and cache it. Thread-safe."""
        self.settings_view = SettingsView(self.content_frame, self.settings_controller, self.root)
        self.settings_view.grid(row=0, column=0, sticky="nsew")
        with self._view_cache_lock:
            self._view_cache["Settings"] = self.settings_view
            self._current_view = self.settings_view

    def create_commands_tab(self):
        """Create the commands tab and cache it. Thread-safe."""
        self.commands_view = CommandsView(self.content_frame, self.commands_controller, self.root, self.logger)
        self.commands_view.grid(row=0, column=0, sticky="nsew")
        with self._view_cache_lock:
            self._view_cache["Commands"] = self.commands_view
            self._current_view = self.commands_view

    def create_dictation_tab(self):
        """Create the dictation tab and cache it. Thread-safe."""
        self.dictation_view = DictationView(self.content_frame, self.dictation_controller, self.root)
        self.dictation_view.grid(row=0, column=0, sticky="nsew")
        with self._view_cache_lock:
            self._view_cache["Dictation"] = self.dictation_view
            self._current_view = self.dictation_view

    def cleanup_controllers(self):
        """Clean up all controllers when shutting down. Thread-safe."""
        try:
            # Clean up cached views first (thread-safe)
            with self._view_cache_lock:
                view_items = list(self._view_cache.items())
                self._view_cache.clear()
                self._current_view = None

            for view_name, view in view_items:
                try:
                    if hasattr(view, "destroy"):
                        view.destroy()
                except Exception as e:
                    self.logger.debug(f"Error destroying cached view {view_name}: {e}")

            # Clean up controllers
            controllers = [
                "marks_controller",
                "sound_controller",
                "dictation_controller",
                "settings_controller",
                "commands_controller",
                "grid_controller",
                "system_controller",
                "dictation_popup_controller",
            ]

            for controller_name in controllers:
                if hasattr(self, controller_name):
                    controller = getattr(self, controller_name)
                    if hasattr(controller, "cleanup"):
                        controller.cleanup()

            self.logger.debug("Controllers cleaned up")
        except Exception as e:
            self.logger.error(f"Error cleaning up controllers: {e}", exc_info=True)

    # Controller callback methods
    def on_grid_visibility_changed(self, visible: bool, rows: Optional[int], cols: Optional[int], show_numbers: Optional[bool]):
        """Called by grid controller when grid visibility changes"""
        self.logger.debug(f"Grid display updated. Visible: {visible}, Rows: {rows}, Cols: {cols}")

    def on_prompts_updated(self, prompts):
        """Called by dictation controller when prompts are updated"""
        if hasattr(self, "dictation_view"):
            self.dictation_view.display_prompts(prompts)

    def on_current_prompt_updated(self, prompt_id):
        """Called by dictation controller when current prompt is updated"""
        if hasattr(self, "dictation_view"):
            self.dictation_view.update_current_prompt(prompt_id)

    def on_settings_updated(self):
        """Called by settings controller when settings are updated"""
        if hasattr(self, "settings_view"):
            self.settings_view.refresh_settings()

    def on_validation_error(self, title: str, message: str):
        """Called by settings controller for validation errors"""
        themed_dialogs.showerror(message, parent=self.root)

    def on_save_success(self, message: str):
        """Called by settings controller for successful saves"""
        themed_dialogs.showinfo(message, parent=self.root)

    def on_save_error(self, message: str):
        """Called by settings controller for save errors"""
        themed_dialogs.showerror(message, parent=self.root)

    def on_reset_complete(self):
        """Called by settings controller when reset is complete"""
        if hasattr(self, "settings_view"):
            self.settings_view.refresh_settings()

    def update_training_progress(self, sound_name: str, status: str, current_sample: int, total_samples: int):
        """Update training progress - delegate to SoundView if available"""
        if hasattr(self, "sound_view"):
            self.sound_view.update_training_progress(sound_name, status, current_sample, total_samples)
