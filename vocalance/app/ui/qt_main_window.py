"""Qt-based main application window.

Main window with sidebar navigation and stacked content area for tabs.
Replaces CustomTkinter AppControlRoom with Qt-native implementation.
"""

import asyncio
import logging
import threading
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QMainWindow, QStackedWidget, QVBoxLayout, QWidget

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.event_bus import EventBus
from vocalance.app.ui.qt_theme import theme_manager
from vocalance.app.ui.utils.qt_assets import QtAssetCache
from vocalance.app.ui.utils.qt_logo_service import QtLogoService
from vocalance.app.ui.views.components.qt_themed_components import (
    ExpandableSidebar,
    SidebarButton,
    SubtitleLabel,
    ThemedFrame,
    TitleLabel,
    TransparentFrame,
)


class VocalanceMainWindow(QMainWindow):
    """Main application window for Vocalance.

    Orchestrates the main UI with sidebar navigation, lazy-loaded tab views,
    and specialized overlay windows. Thread-safe view caching and tab switching.

    Attributes:
        asset_cache: QtAssetCache for icons and images.
        logo_service: QtLogoService for app logo.
        _view_cache: Dict of lazily-loaded view instances.
        _controllers: Dict of controller instances by tab name.
    """

    def __init__(
        self,
        event_bus: EventBus,
        event_loop: asyncio.AbstractEventLoop,
        logger: logging.Logger,
        config: GlobalAppConfig,
        storage_service=None,
    ):
        """Initialize main window.

        Args:
            event_bus: EventBus for pub/sub messaging.
            event_loop: Asyncio event loop for async operations.
            logger: Logger instance.
            config: Global application configuration.
            storage_service: Optional storage service reference.
        """
        super().__init__()

        self.event_bus = event_bus
        self.event_loop = event_loop
        self.logger = logger
        self.config = config
        self._storage_service = storage_service
        self._settings_service = None

        self.current_tab = "Marks"

        # Asset management
        self.asset_cache = QtAssetCache(asset_paths_config=self.config.asset_paths)
        self.logo_service = QtLogoService(self.asset_cache)

        # View caching
        self._view_cache_lock = threading.RLock()
        self._view_cache = {}
        self._current_view = None

        # Setup window
        self._setup_window()
        self._initialize_controllers()
        self._initialize_specialized_views()
        self._build_ui()

        self.logger.debug("VocalanceMainWindow initialized")

    def _setup_window(self) -> None:
        """Configure main window properties."""
        self.setWindowTitle("Vocalance")

        # Set window size
        self.resize(
            theme_manager.dimensions.main_window_width,
            theme_manager.dimensions.main_window_height,
        )
        self.setMinimumSize(
            theme_manager.dimensions.main_window_min_width,
            theme_manager.dimensions.main_window_min_height,
        )

        # Set window background color to match theme
        self.setStyleSheet(f"QMainWindow {{ background-color: {theme_manager.shape_colors.darkest}; }}")

        # Set window icon if available
        icon_path = self.asset_cache.get_icon_path()
        if icon_path and icon_path.exists():
            from PySide6.QtGui import QIcon

            self.setWindowIcon(QIcon(str(icon_path)))

    def _initialize_controllers(self) -> None:
        """Initialize all controllers.

        Controllers will be created as Qt QObject subclasses in a separate step.
        For now, we'll import and create placeholder references.
        """
        try:
            # Import controllers (these will need to be updated to Qt)
            # Temporarily using None placeholders until controllers are migrated
            self.marks_controller = None
            self.sound_controller = None
            self.dictation_controller = None
            self.settings_controller = None
            self.commands_controller = None
            self.grid_controller = None
            self.system_controller = None
            self.dictation_popup_controller = None

            self.logger.debug("Controller placeholders initialized")

        except Exception as e:
            self.logger.error(f"Error initializing controllers: {e}", exc_info=True)
            raise

    def _initialize_specialized_views(self) -> None:
        """Initialize specialized views (grid, marks, dictation popup).

        These will be created when controllers are available.
        """
        try:
            # Overlay views will be created in initialize_controllers_with_services
            self.grid_view = None
            self.mark_view = None
            self.dictation_popup_view = None

            self.logger.debug("Specialized view placeholders initialized")

        except Exception as e:
            self.logger.error(f"Error initializing specialized views: {e}", exc_info=True)
            raise

    def _build_ui(self) -> None:
        """Build the main UI layout."""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout - horizontal: sidebar | separator | content
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create sidebar
        self._create_sidebar()
        main_layout.addWidget(self.sidebar_frame)

        # Create separator (will expand/contract with sidebar)
        self._create_sidebar_separator()
        main_layout.addWidget(self.sidebar_separator)

        # Create right panel wrapper with outer padding (space between border and window edges)
        right_panel_wrapper = QWidget()
        right_wrapper_layout = QVBoxLayout(right_panel_wrapper)
        right_wrapper_layout.setContentsMargins(
            theme_manager.two_box_layout.outer_padding_left,
            theme_manager.two_box_layout.outer_padding_top,
            theme_manager.two_box_layout.outer_padding_right,
            theme_manager.two_box_layout.outer_padding_bottom,
        )
        right_wrapper_layout.setSpacing(0)

        # Create bordered content frame that wraps header + content
        # Use ThemedFrame with content_border frameType (defined in QSS)
        self.content_border_frame = ThemedFrame(frame_type="content_border")
        content_frame_layout = QVBoxLayout(self.content_border_frame)
        content_frame_layout.setContentsMargins(
            theme_manager.two_box_layout.outer_border_padding,
            theme_manager.two_box_layout.outer_border_padding,
            theme_manager.two_box_layout.outer_border_padding,
            theme_manager.two_box_layout.outer_border_padding,
        )
        content_frame_layout.setSpacing(0)

        # Create header with proper padding
        self._create_header()
        content_frame_layout.addWidget(self.header_frame)

        # Create content area
        self._create_content_area()
        content_frame_layout.addWidget(self.content_widget)

        # Add bordered frame to wrapper
        right_wrapper_layout.addWidget(self.content_border_frame)

        main_layout.addWidget(right_panel_wrapper, stretch=1)

        # Show initial tab
        self.show_tab("Marks")

    def _create_sidebar(self) -> None:
        """Create the expandable sidebar with navigation buttons."""
        self.sidebar_frame = ExpandableSidebar()
        self.sidebar_button_manager = self.sidebar_frame.button_manager

        # Buttons container
        self._create_sidebar_buttons()
        self.sidebar_frame.add_button_widget(self.buttons_widget)

        # Stretch
        self.sidebar_frame.add_stretch()

        # Logo at bottom
        self._create_sidebar_logo()
        self.sidebar_frame.add_logo(self.sidebar_logo_frame)

        # Select first button if available
        if self.sidebar_buttons:
            first_button = list(self.sidebar_buttons.values())[0]
            self.sidebar_button_manager.select_button(first_button)

    def _create_sidebar_buttons(self) -> None:
        """Create sidebar navigation buttons."""
        self.buttons_widget = TransparentFrame()
        buttons_layout = QVBoxLayout(self.buttons_widget)
        buttons_layout.setContentsMargins(
            theme_manager.sidebar_layout.button_padding_left,
            0,
            theme_manager.sidebar_layout.button_padding_right,
            0,
        )
        buttons_layout.setSpacing(theme_manager.sidebar_layout.button_spacing_vertical)

        self.sidebar_buttons = {}

        # Tab definitions with icon filenames
        tabs = [
            ("Marks", "bookmark_flag_500dp_E3E3E3_FILL0_wght400_GRAD0_opsz48.png"),
            ("Sounds", "mic_500dp_E3E3E3_FILL0_wght400_GRAD0_opsz48.png"),
            ("Commands", "voice_selection_500dp_E3E3E3_FILL0_wght400_GRAD0_opsz48.png"),
            ("Dictation", "network_intelligence_500dp_E3E3E3_FILL0_wght400_GRAD0_opsz48.png"),
            ("Settings", "settings_500dp_E3E3E3_FILL0_wght400_GRAD0_opsz48.png"),
        ]

        # Import icon loading utility
        from vocalance.app.ui.utils.qt_icon_utils import load_sidebar_icon

        # Use the larger icon size for collapsed state
        icon_size = theme_manager.sidebar_layout.button_icon_size

        for tab_name, icon_filename in tabs:
            # Load and transform icon
            icon_pixmap = load_sidebar_icon(
                icon_filename=icon_filename,
                icons_dir=self.asset_cache.get_icons_dir(),
                target_color=theme_manager.icon_properties.color,
                icon_size=icon_size,
            )

            # Create button with icon
            btn = SidebarButton(text=tab_name, icon_pixmap=icon_pixmap)
            btn.clicked.connect(lambda checked=False, tab=tab_name: self.show_tab(tab))

            buttons_layout.addWidget(btn)

            self.sidebar_buttons[tab_name] = btn
            # Add button to the sidebar's button manager
            self.sidebar_button_manager.add_button(btn)

    def _create_sidebar_logo(self) -> None:
        """Create sidebar logo with transparent background."""
        # Create transparent frame for logo
        logo_frame = TransparentFrame()
        logo_frame.setStyleSheet("background: transparent; border: none;")
        logo_layout = QVBoxLayout(logo_frame)
        logo_layout.setContentsMargins(
            0,
            theme_manager.sidebar_layout.logo_padding_top,
            0,
            theme_manager.sidebar_layout.logo_padding_bottom,
        )

        self.sidebar_logo = self.logo_service.create_logo_widget(
            max_size=theme_manager.sidebar_layout.logo_max_size,
            context="sidebar",
            text_fallback="Vocalance",
            logo_type="icon",
        )
        self.sidebar_logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sidebar_logo.setStyleSheet("background: transparent; border: none;")
        logo_layout.addWidget(self.sidebar_logo)

        # Store the frame instead of just the logo widget
        self.sidebar_logo_frame = logo_frame

    def _create_sidebar_separator(self) -> None:
        """Create separator line between sidebar and content."""
        self.sidebar_separator = QFrame()
        self.sidebar_separator.setFrameShape(QFrame.Shape.VLine)
        self.sidebar_separator.setFrameShadow(QFrame.Shadow.Plain)
        self.sidebar_separator.setFixedWidth(theme_manager.sidebar_layout.border_width)
        self.sidebar_separator.setStyleSheet("background-color: transparent; border: none;")

    def _create_header(self) -> None:
        """Create the header section with wrapper for proper padding."""
        # Outer wrapper for frame padding
        self.header_frame = QWidget()
        self.header_frame.setStyleSheet("background: transparent; border: none;")
        outer_layout = QVBoxLayout(self.header_frame)
        # No outer padding needed - the content_border_frame handles that
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        # Inner header frame
        header_inner = ThemedFrame(frame_type="header")
        header_inner.setFixedHeight(theme_manager.dimensions.header_height)
        header_inner.setStyleSheet("border: none; background: transparent;")

        header_layout = QVBoxLayout(header_inner)
        header_layout.setContentsMargins(
            theme_manager.header_layout.content_padding_left,
            theme_manager.header_layout.title_y_offset,
            theme_manager.header_layout.content_padding_right,
            theme_manager.spacing.large,
        )
        header_layout.setSpacing(theme_manager.spacing.small)

        # Title
        self.header_label = TitleLabel(text="Welcome to Vocalance!")
        self.header_label.setStyleSheet("border: none; background: transparent;")
        header_layout.addWidget(self.header_label, alignment=Qt.AlignmentFlag.AlignLeft)

        # Subtitle (created on demand)
        self.header_subtitle = None
        self.header_inner = header_inner

        # Stretch
        header_layout.addStretch()

        outer_layout.addWidget(header_inner)

    def _create_content_area(self) -> None:
        """Create the main content area with stacked widget for tabs."""
        self.content_widget = TransparentFrame()
        content_layout = QVBoxLayout(self.content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # Stacked widget for tab content (views handle their own padding)
        self.stacked_widget = QStackedWidget()
        content_layout.addWidget(self.stacked_widget)

    def _set_header_subtitle(self, text: str) -> None:
        """Set or update the header subtitle.

        Args:
            text: Subtitle text.
        """
        if not self.header_subtitle:
            self.header_subtitle = SubtitleLabel(text=text)
            self.header_subtitle.setStyleSheet("border: none; background: transparent;")
            # Insert after title
            header_layout = self.header_inner.layout()
            header_layout.insertWidget(1, self.header_subtitle, alignment=Qt.AlignmentFlag.AlignLeft)
        else:
            self.header_subtitle.setText(text)

    def show_tab(self, tab_name: str) -> None:
        """Show the specified tab with view caching. Thread-safe.

        Args:
            tab_name: Name of tab to show.
        """
        self.current_tab = tab_name

        # Update header
        subtitles = {
            "Sounds": "Use custom sounds to control your computer",
            "Marks": "Pinpoint important locations on your screen",
            "Commands": "Manage voice commands and their actions",
            "Dictation": "Configure smart dictation with AI prompts",
            "Settings": "Configure default Vocalance settings",
        }

        self.header_label.setText(tab_name)
        if tab_name in subtitles:
            self._set_header_subtitle(subtitles[tab_name])

        # Check if view is cached
        with self._view_cache_lock:
            view_cached = tab_name in self._view_cache

        if not view_cached:
            self.logger.debug(f"Creating new view for tab: {tab_name}")

            # Create view based on tab name
            # Placeholders for now - will be implemented with actual view classes
            view = self._create_placeholder_view(tab_name)

            with self._view_cache_lock:
                self._view_cache[tab_name] = view
                self._current_view = view

            self.stacked_widget.addWidget(view)
            self.stacked_widget.setCurrentWidget(view)
        else:
            self.logger.debug(f"Reusing cached view for tab: {tab_name}")
            with self._view_cache_lock:
                cached_view = self._view_cache[tab_name]
                self._current_view = cached_view
            self.stacked_widget.setCurrentWidget(cached_view)

    def _create_placeholder_view(self, tab_name: str) -> QWidget:
        """Create actual view widget for the tab.

        Args:
            tab_name: Name of the tab.

        Returns:
            Actual view widget.
        """
        try:
            # Import and create actual view based on tab name
            if tab_name == "Marks":
                from vocalance.app.ui.views.qt_marks_view import QtMarksView

                view = QtMarksView()
                if self.marks_controller:
                    view.set_controller(self.marks_controller)
                return view

            elif tab_name == "Sounds":
                from vocalance.app.ui.views.qt_sounds_view import QtSoundsView

                view = QtSoundsView()
                if self.sound_controller:
                    view.set_controller(self.sound_controller)
                return view

            elif tab_name == "Commands":
                from vocalance.app.ui.views.qt_commands_view import QtCommandsView

                view = QtCommandsView()
                if self.commands_controller:
                    view.set_controller(self.commands_controller)
                return view

            elif tab_name == "Dictation":
                from vocalance.app.ui.views.qt_dictation_view import QtDictationView

                view = QtDictationView()
                if self.dictation_controller:
                    view.set_controller(self.dictation_controller)
                return view

            elif tab_name == "Settings":
                from vocalance.app.ui.views.qt_settings_view import QtSettingsView

                view = QtSettingsView()
                if self.settings_controller:
                    view.set_controller(self.settings_controller)
                return view

        except Exception as e:
            self.logger.error(f"Error creating view for {tab_name}: {e}", exc_info=True)

        # Fallback to placeholder if view creation fails
        placeholder = QWidget()
        layout = QVBoxLayout(placeholder)
        label = QLabel(f"{tab_name} View\n(Fallback - check logs for errors)")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = theme_manager.get_font(size=theme_manager.font_sizes.large)
        label.setFont(font)
        label.setStyleSheet(f"color: {theme_manager.text_colors.light};")
        layout.addWidget(label)
        return placeholder

    def set_settings_service(self, settings_service) -> None:
        """Set the settings service reference for controllers to use.

        Args:
            settings_service: Settings service instance.
        """
        self._settings_service = settings_service
        if self.settings_controller:
            self.settings_controller.set_settings_service(settings_service)

    def closeEvent(self, event) -> None:
        """Handle window close event for graceful shutdown.

        Args:
            event: Close event.
        """
        self.logger.info("Main window close event triggered")

        # Cleanup controllers
        self.cleanup_controllers()

        event.accept()

    def cleanup_controllers(self) -> None:
        """Clean up all controllers when shutting down. Thread-safe."""
        try:
            # Clean up cached views first
            with self._view_cache_lock:
                view_items = list(self._view_cache.items())
                self._view_cache.clear()
                self._current_view = None

            for view_name, view in view_items:
                try:
                    if hasattr(view, "deleteLater"):
                        view.deleteLater()
                except Exception as e:
                    self.logger.debug(f"Error deleting cached view {view_name}: {e}")

            # Clean up controllers (when implemented)
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
                    if controller and hasattr(controller, "cleanup"):
                        controller.cleanup()

            self.logger.debug("Controllers cleaned up")

        except Exception as e:
            self.logger.error(f"Error cleaning up controllers: {e}", exc_info=True)

    # Controller callback methods (to be implemented when controllers are migrated)

    def on_grid_visibility_changed(
        self,
        visible: bool,
        rows: Optional[int],
        cols: Optional[int],
        show_numbers: Optional[bool],
    ) -> None:
        """Called by grid controller when grid visibility changes."""
        self.logger.debug(f"Grid display updated. Visible: {visible}, Rows: {rows}, Cols: {cols}")

    def on_prompts_updated(self, prompts) -> None:
        """Called by dictation controller when prompts are updated."""

    def on_current_prompt_updated(self, prompt_id) -> None:
        """Called by dictation controller when current prompt is updated."""

    def on_settings_updated(self) -> None:
        """Called by settings controller when settings are updated."""

    def on_validation_error(self, title: str, message: str) -> None:
        """Called by settings controller for validation errors."""

    def on_save_success(self, message: str) -> None:
        """Called by settings controller for successful saves."""

    def on_save_error(self, message: str) -> None:
        """Called by settings controller for save errors."""

    def on_reset_complete(self) -> None:
        """Called by settings controller when reset is complete."""

    def update_training_progress(
        self,
        sound_name: str,
        status: str,
        current_sample: int,
        total_samples: int,
    ) -> None:
        """Update training progress - delegate to SoundView if available."""

    # Service setters
    def set_mark_service(self, mark_service) -> None:
        """Set mark service for controller initialization."""
        self._mark_service = mark_service

    def set_grid_service(self, grid_service) -> None:
        """Set grid service for controller initialization."""
        self._grid_service = grid_service

    def set_sound_service(self, sound_service) -> None:
        """Set sound service for controller initialization."""
        self._sound_service = sound_service

    def set_command_management_service(self, command_service) -> None:
        """Set command management service for controller initialization."""
        self._command_service = command_service

    def set_dictation_service(self, dictation_service) -> None:
        """Set dictation service for controller initialization."""
        self._dictation_service = dictation_service

    def initialize_controllers_with_services(self) -> None:
        """Initialize all controllers now that services are available."""
        try:
            # Import all controller implementations
            from vocalance.app.ui.controls.qt_commands_controller import QtCommandsController
            from vocalance.app.ui.controls.qt_dictation_controller import QtDictationController
            from vocalance.app.ui.controls.qt_grid_controller import QtGridController
            from vocalance.app.ui.controls.qt_marks_controller import QtMarksController
            from vocalance.app.ui.controls.qt_settings_controller import QtSettingsController
            from vocalance.app.ui.controls.qt_sound_controller import QtSoundController

            # Initialize marks controller
            if hasattr(self, "_mark_service") and self._mark_service:
                self.marks_controller = QtMarksController(
                    event_bus=self.event_bus,
                    event_loop=self.event_loop,
                    mark_service=self._mark_service,
                    config=self.config,
                    main_window=self,
                )
                self.logger.debug("Marks controller initialized")

            # Initialize grid controller
            if hasattr(self, "_grid_service") and self._grid_service:
                self.grid_controller = QtGridController(
                    event_bus=self.event_bus,
                    event_loop=self.event_loop,
                    grid_service=self._grid_service,
                    config=self.config,
                    main_window=self,
                )
                self.logger.debug("Grid controller initialized")

            # Initialize sound controller
            if hasattr(self, "_sound_service") and self._sound_service:
                self.sound_controller = QtSoundController(
                    event_bus=self.event_bus,
                    event_loop=self.event_loop,
                    sound_service=self._sound_service,
                    storage_service=self._storage_service,
                    config=self.config,
                    main_window=self,
                )
                self.logger.debug("Sound controller initialized")

            # Initialize commands controller
            if hasattr(self, "_command_service") and self._command_service:
                self.commands_controller = QtCommandsController(
                    event_bus=self.event_bus,
                    event_loop=self.event_loop,
                    command_management_service=self._command_service,
                    config=self.config,
                    main_window=self,
                )
                self.logger.debug("Commands controller initialized")

            # Initialize dictation controller
            if hasattr(self, "_dictation_service") and self._dictation_service:
                self.dictation_controller = QtDictationController(
                    event_bus=self.event_bus,
                    event_loop=self.event_loop,
                    dictation_service=self._dictation_service,
                    config=self.config,
                    main_window=self,
                )
                self.logger.debug("Dictation controller initialized")

            # Initialize settings controller
            if hasattr(self, "_settings_service") and self._settings_service:
                self.settings_controller = QtSettingsController(
                    event_bus=self.event_bus,
                    event_loop=self.event_loop,
                    settings_service=self._settings_service,
                    config=self.config,
                    main_window=self,
                )
                self.logger.debug("Settings controller initialized")

            # Initialize dictation popup controller
            try:
                from vocalance.app.ui.controls.qt_dictation_popup_controller import QtDictationPopupController

                self.dictation_popup_controller = QtDictationPopupController(
                    event_bus=self.event_bus,
                    event_loop=self.event_loop,
                )
                self.logger.debug("Dictation popup controller initialized")
            except Exception as e:
                self.logger.warning(f"Could not initialize dictation popup controller: {e}")
                self.dictation_popup_controller = None

            # Initialize overlay views and connect to controllers
            self._initialize_overlay_views()

            # Connect controllers to views
            self._connect_controllers_to_views()

            self.logger.info("All controllers initialized with services")

        except ImportError as e:
            self.logger.warning(f"Controller import failed (will be created): {e}")
        except Exception as e:
            self.logger.error(f"Error initializing controllers: {e}", exc_info=True)

    def _initialize_overlay_views(self) -> None:
        """Initialize overlay views (mark and grid) and connect to controllers."""
        try:
            from vocalance.app.ui.views.qt_grid_view import QtGridView
            from vocalance.app.ui.views.qt_mark_view import QtMarkView

            # Initialize mark overlay
            if self.marks_controller and hasattr(self, "_mark_service") and self._mark_service:
                self.mark_view = QtMarkView(
                    mark_service=self._mark_service,
                    config=self.config,
                )
                self.mark_view.set_controller_callback(self.marks_controller)
                self.marks_controller.set_mark_view(self.mark_view)
                self.logger.debug("Mark overlay view initialized and connected")

            # Initialize grid overlay
            if self.grid_controller and hasattr(self, "_grid_service") and self._grid_service:
                self.grid_view = QtGridView(
                    event_bus=self.event_bus,
                    event_loop=self.event_loop,
                    storage=self._storage_service if hasattr(self, "_storage_service") else None,
                    config=self.config,
                )
                self.grid_view.set_controller_callback(self.grid_controller)
                self.grid_controller.set_grid_view(self.grid_view)

                # Initialize click cache asynchronously
                asyncio.run_coroutine_threadsafe(self.grid_controller.initialize_click_cache(), self.event_loop)

                self.logger.debug("Grid overlay view initialized and connected")

        except Exception as e:
            self.logger.error(f"Error initializing overlay views: {e}", exc_info=True)

    def _connect_controllers_to_views(self) -> None:
        """Connect initialized controllers to their views."""
        # Marks view
        if self.marks_controller and "Marks" in self._view_cache:
            marks_view = self._view_cache["Marks"]
            if hasattr(marks_view, "set_controller"):
                marks_view.set_controller(self.marks_controller)
                self.logger.debug("Marks controller connected to view")

        # Sounds view
        if self.sound_controller and "Sounds" in self._view_cache:
            sounds_view = self._view_cache["Sounds"]
            if hasattr(sounds_view, "set_controller"):
                sounds_view.set_controller(self.sound_controller)
                self.logger.debug("Sound controller connected to view")
