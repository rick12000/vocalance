import asyncio
import logging
import threading
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QMainWindow, QStackedWidget, QVBoxLayout, QWidget

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.event_bus import EventBus
from vocalance.app.ui.components.complex_components import HeaderIconButton, SidebarButton
from vocalance.app.ui.components.labels import BodyLabel, LargeLabel, TitleLabel
from vocalance.app.ui.components.layouts import BaseContainer, TransparentBox
from vocalance.app.ui.components.specialized import ExpandableSidebar
from vocalance.app.ui.qt_theme import theme
from vocalance.app.ui.utils.qt_assets import QtAssetCache
from vocalance.app.ui.utils.qt_logo_service import QtLogoService
from vocalance.app.ui.utils.window_icon_manager import WindowIconManager


class VocalanceMainWindow(QMainWindow):
    """Main application window for Vocalance.

    Orchestrates the main UI with sidebar navigation, lazy-loaded tab views,
    and specialized overlay windows. Thread-safe view caching and tab switching.
    """

    def __init__(
        self,
        event_bus: EventBus,
        event_loop: asyncio.AbstractEventLoop,
        logger: logging.Logger,
        config: GlobalAppConfig,
        storage_service=None,
        icon_manager: Optional[WindowIconManager] = None,
    ):
        super().__init__()

        self.event_bus = event_bus
        self.event_loop = event_loop
        self.logger = logger
        self.config = config
        self._storage_service = storage_service
        self._settings_service = None
        self.icon_manager = icon_manager

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
            theme.config.components.main_window_width,
            theme.config.components.main_window_height,
        )
        self.setMinimumSize(
            theme.config.components.main_window_min_width,
            theme.config.components.main_window_min_height,
        )

        # Set window background color programmatically
        palette = self.palette()
        from PySide6.QtGui import QColor, QPalette

        palette.setColor(QPalette.ColorRole.Window, QColor(theme.config.shapes.darkest))
        self.setPalette(palette)
        self.setAutoFillBackground(True)

        # Apply icon using icon manager if available
        if self.icon_manager and self.icon_manager.is_icon_loaded():
            self.icon_manager.apply_to_window(self)
            self.logger.debug("Icon applied to main window via icon manager")
        else:
            # Fallback: load icon directly if manager not available
            icon_path = self.asset_cache.get_icon_path()
            if icon_path and icon_path.exists():
                from PySide6.QtGui import QIcon

                self.setWindowIcon(QIcon(str(icon_path)))
                self.logger.debug("Icon applied to main window directly")

    def _initialize_controllers(self) -> None:
        """Initialize all controllers."""
        try:
            self.marks_controller = None
            self.sound_controller = None
            self.dictation_controller = None
            self.dictation_alias_controller = None
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
        """Initialize specialized views."""
        try:
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

        # Create separator
        self._create_sidebar_separator()
        main_layout.addWidget(self.sidebar_separator)

        # Create right panel wrapper with outer padding
        right_panel_wrapper = QWidget()
        right_wrapper_layout = QVBoxLayout(right_panel_wrapper)
        right_wrapper_layout.setContentsMargins(0, 0, 0, 0)
        right_wrapper_layout.setSpacing(0)

        # Create bordered content frame that wraps header + content (transparent border)
        self.content_border_frame = BaseContainer(
            layout="vertical",
            bg_color=theme.config.shapes.darkest,
            border_color=None,  # Transparent border
            border_radius=0,
        )

        content_frame_layout = self.content_border_frame.layout()  # It has a layout from BaseContainer
        content_frame_layout.setContentsMargins(
            theme.config.container.box_padding,
            theme.config.container.box_padding,
            theme.config.container.box_padding,
            theme.config.container.box_padding,
        )
        # Consistent spacing between header and content - reduced for tighter layout
        content_frame_layout.setSpacing(theme.config.spacing.small)

        # Create header with proper padding
        self._create_header()
        self.content_border_frame.add(self.header_frame)

        # Create content area
        self._create_content_area()
        self.content_border_frame.add(self.content_widget, stretch=1)

        # Add bordered frame to wrapper
        right_wrapper_layout.addWidget(self.content_border_frame, stretch=1)

        main_layout.addWidget(right_panel_wrapper, stretch=1)

        # Show initial tab
        self.show_tab("Commands")

    def _create_sidebar(self) -> None:
        """Create the expandable sidebar with navigation buttons."""
        self.sidebar_frame = ExpandableSidebar()
        self.sidebar_button_manager = self.sidebar_frame.manager

        # Buttons container
        self._create_sidebar_buttons()
        self.sidebar_frame.add_widget(self.buttons_widget)

        # Stretch
        self.sidebar_frame.add_stretch()

        # Logo at bottom
        self._create_sidebar_logo()
        self.sidebar_frame.add_widget(self.sidebar_logo_frame)

        # Select first button if available
        if self.sidebar_buttons:
            first_button = list(self.sidebar_buttons.values())[0]
            self.sidebar_button_manager.select(first_button)

    def _create_sidebar_buttons(self) -> None:
        """Create sidebar navigation buttons."""
        self.buttons_widget = TransparentBox()
        buttons_layout = self.buttons_widget.layout()
        buttons_layout.setContentsMargins(
            theme.config.sidebar.button_padding_left,
            0,
            theme.config.sidebar.button_padding_right,
            0,
        )
        buttons_layout.setSpacing(theme.config.sidebar.button_spacing_vertical)

        self.sidebar_buttons = {}

        # Tab definitions
        tabs = [
            ("Commands", "voice_selection_500dp_E3E3E3_FILL0_wght400_GRAD0_opsz48.png"),
            ("Marks", "location_on_500dp_E3E3E3_FILL0_wght400_GRAD0_opsz48.png"),
            ("Dictation", "speech_to_text_500dp_E3E3E3_FILL0_wght400_GRAD0_opsz48.png"),
            ("Sounds", "mic_500dp_E3E3E3_FILL0_wght400_GRAD0_opsz48.png"),
            ("Settings", "settings_500dp_E3E3E3_FILL0_wght400_GRAD0_opsz48.png"),
        ]

        # Import icon loading utility
        from vocalance.app.ui.utils.qt_icon_utils import load_sidebar_icon

        icon_size = theme.config.sidebar.button_icon_size

        for tab_name, icon_filename in tabs:
            # Load and transform icon with shapes.accent color
            icon_pixmap = load_sidebar_icon(
                icon_filename=icon_filename,
                icons_dir=self.asset_cache.get_icons_dir(),
                target_color=theme.config.shapes.accent,
                icon_size=icon_size,
            )

            # Create button with icon
            btn = SidebarButton(text=tab_name, icon_pixmap=icon_pixmap)
            btn.clicked.connect(lambda checked=False, tab=tab_name: self.show_tab(tab))

            self.buttons_widget.add(btn)

            self.sidebar_buttons[tab_name] = btn
            self.sidebar_button_manager.add(btn)

    def _create_sidebar_logo(self) -> None:
        """Create sidebar logo with transparent background.

        Uses fixed-width container matching collapsed sidebar to keep logo
        positioned consistently during animation. Logo is aligned with button icons.
        """
        from PySide6.QtWidgets import QHBoxLayout

        logo_frame = TransparentBox(layout="horizontal")
        logo_layout = logo_frame.layout()
        logo_layout.setContentsMargins(
            0,
            theme.config.sidebar.logo_padding_top,
            0,
            theme.config.sidebar.logo_padding_bottom,
        )
        logo_layout.setSpacing(0)

        # Create fixed-width logo area matching collapsed sidebar width
        logo_area = QWidget()
        logo_area.setFixedWidth(theme.config.sidebar.collapsed_width)
        logo_area.setAutoFillBackground(False)
        logo_area_layout = QHBoxLayout(logo_area)
        logo_area_layout.setContentsMargins(0, 0, 0, 0)
        logo_area_layout.setSpacing(0)

        self.sidebar_logo = self.logo_service.create_logo_widget(
            max_size=theme.config.sidebar.logo_max_size,
            context="sidebar",
            text_fallback="Vocalance",
            logo_type="icon",
        )
        self.sidebar_logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sidebar_logo.setAutoFillBackground(False)

        # Position logo to align with button icons
        # Button icons are offset by container padding, so shift logo right to match
        logo_area_layout.addStretch()
        logo_area_layout.addWidget(self.sidebar_logo, alignment=Qt.AlignmentFlag.AlignCenter)
        logo_area_layout.addStretch()

        # Shift logo positioning to match button icon alignment
        # Button icon center is at 50px from left edge of sidebar
        # Logo currently centers at 40px, so shift right by 10px
        # Left margin needed: solve M + (80-M)/2 = 50 → M = 20px
        logo_area_layout.setContentsMargins(20, 0, 0, 0)

        logo_layout.addWidget(logo_area)
        logo_layout.addStretch()  # Push logo area to left

        self.sidebar_logo_frame = logo_frame

    def _create_sidebar_separator(self) -> None:
        """Create separator line between sidebar and content - transparent."""
        self.sidebar_separator = QFrame()
        self.sidebar_separator.setFrameShape(QFrame.Shape.NoFrame)
        self.sidebar_separator.setFixedWidth(theme.config.sidebar.border_width)
        self.sidebar_separator.setAutoFillBackground(False)

    def _create_header(self) -> None:
        """Create the header section."""
        # Outer wrapper
        self.header_frame = QWidget()
        self.header_frame.setAutoFillBackground(False)
        outer_layout = QVBoxLayout(self.header_frame)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        # Inner header frame - now with horizontal layout to accommodate icon button
        header_inner = BaseContainer(
            layout="horizontal",
            bg_color=theme.config.shapes.darkest,
            border_color=None,
            border_radius=0,
        )

        header_layout = header_inner.layout()
        header_layout.setContentsMargins(
            theme.config.container.box_padding,
            0,
            theme.config.container.box_padding,
            theme.config.header.padding_bottom,
        )
        header_layout.setSpacing(0)

        # Left side: Title and subtitle container
        title_container = TransparentBox()
        title_layout = title_container.layout()
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(theme.config.header.spacing)

        # Title
        self.header_label = TitleLabel(text="Welcome to Vocalance!")
        title_layout.addWidget(self.header_label, alignment=Qt.AlignmentFlag.AlignLeft)

        # Subtitle placeholder
        self.header_subtitle = None

        # Stretch within title container
        title_layout.addStretch()

        header_layout.addWidget(title_container, stretch=1)

        # Right side: Header icon button
        self._create_header_icon_button()
        header_layout.addWidget(self.header_icon_button, alignment=Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignCenter)

        self.header_inner = header_inner

        outer_layout.addWidget(header_inner)

    def _create_header_icon_button(self) -> None:
        """Create the header icon button with book icon."""
        from vocalance.app.ui.utils.qt_icon_utils import load_sidebar_icon

        # Load book icon at theme-defined size
        icon_filename = theme.config.icon_properties.documentation_icon_filename
        icon_pixmap = load_sidebar_icon(
            icon_filename=icon_filename,
            icons_dir=self.asset_cache.get_icons_dir(),
            target_color=theme.config.shapes.accent,
            icon_size=theme.config.header.icon_size,
        )

        # Create header icon button with text "Documentation"
        self.header_icon_button = HeaderIconButton(
            text="User Guide",
            icon_pixmap=icon_pixmap,
            icon_size=theme.config.header.icon_size,
            text_icon_spacing=theme.config.header.text_icon_spacing,
        )

        # Connect click handler (placeholder for now)
        self.header_icon_button.clicked.connect(self._on_documentation_clicked)

    def _on_documentation_clicked(self) -> None:
        """Handle documentation button click - open user guide URL."""
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        documentation_url = "https://www.vocalance.com/instructions.html"
        self.logger.info(f"Opening documentation: {documentation_url}")
        QDesktopServices.openUrl(QUrl(documentation_url))

    def _create_content_area(self) -> None:
        """Create the main content area."""
        self.content_widget = TransparentBox()
        content_layout = self.content_widget.layout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # Stacked widget for tab content
        self.stacked_widget = QStackedWidget()
        content_layout.addWidget(self.stacked_widget)

    def _set_header_subtitle(self, text: str) -> None:
        """Set or update the header subtitle."""
        if not self.header_subtitle:
            self.header_subtitle = BodyLabel(text=text)
            # Get the title container (first widget in header layout)
            title_container_widget = self.header_inner.layout().itemAt(0).widget()
            if title_container_widget:
                # Insert after title (index 1) in the title container's layout
                title_container_widget.layout().insertWidget(1, self.header_subtitle, alignment=Qt.AlignmentFlag.AlignLeft)
        else:
            self.header_subtitle.setText(text)

    def show_tab(self, tab_name: str) -> None:
        """Show the specified tab with view caching."""
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

        # Check cache
        with self._view_cache_lock:
            view_cached = tab_name in self._view_cache

        if not view_cached:
            self.logger.debug(f"Creating new view for tab: {tab_name}")
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
        """Create actual view widget for the tab."""
        try:
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
                if self.dictation_alias_controller:
                    view.set_alias_controller(self.dictation_alias_controller)
                return view

            elif tab_name == "Settings":
                from vocalance.app.ui.views.qt_settings_view import QtSettingsView

                view = QtSettingsView()
                if self.settings_controller:
                    view.set_controller(self.settings_controller)
                return view

        except Exception as e:
            self.logger.error(f"Error creating view for {tab_name}: {e}", exc_info=True)

        # Fallback - create a widget with dummy set_controller method
        class FallbackView(QWidget):
            def set_controller(self, controller):
                pass  # Dummy method to prevent AttributeError

        placeholder = FallbackView()
        layout = QVBoxLayout(placeholder)
        label = LargeLabel(f"{tab_name} View\n(Fallback - check logs for errors)")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        return placeholder

    def set_settings_service(self, settings_service) -> None:
        self._settings_service = settings_service
        if self.settings_controller:
            self.settings_controller.set_settings_service(settings_service)

    def closeEvent(self, event) -> None:
        self.logger.info("Main window close event triggered")
        self.cleanup_controllers()
        event.accept()

    def cleanup_controllers(self) -> None:
        """Clean up all controllers when shutting down."""
        try:
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

            controllers = [
                "marks_controller",
                "sound_controller",
                "dictation_controller",
                "dictation_alias_controller",
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

    # Controller callbacks
    def on_grid_visibility_changed(
        self, visible: bool, rows: Optional[int], cols: Optional[int], show_numbers: Optional[bool]
    ) -> None:
        self.logger.debug(f"Grid display updated. Visible: {visible}")

    def on_prompts_updated(self, prompts) -> None:
        pass

    def on_current_prompt_updated(self, prompt_id) -> None:
        pass

    def on_settings_updated(self) -> None:
        pass

    def on_validation_error(self, title: str, message: str) -> None:
        pass

    def on_save_success(self, message: str) -> None:
        pass

    def on_save_error(self, message: str) -> None:
        pass

    def on_reset_complete(self) -> None:
        pass

    def update_training_progress(self, sound_name: str, status: str, current_sample: int, total_samples: int) -> None:
        pass

    # Service setters
    def set_mark_service(self, mark_service) -> None:
        self._mark_service = mark_service

    def set_grid_service(self, grid_service) -> None:
        self._grid_service = grid_service

    def set_sound_service(self, sound_service) -> None:
        self._sound_service = sound_service

    def set_command_management_service(self, command_service) -> None:
        self._command_service = command_service

    def set_dictation_service(self, dictation_service) -> None:
        self._dictation_service = dictation_service

    def set_click_tracker_service(self, click_tracker_service) -> None:
        self._click_tracker_service = click_tracker_service

    def initialize_controllers_with_services(self) -> None:
        """Initialize all controllers now that services are available."""
        try:
            from vocalance.app.ui.controls.qt_commands_controller import QtCommandsController
            from vocalance.app.ui.controls.qt_dictation_alias_controller import QtDictationAliasController
            from vocalance.app.ui.controls.qt_dictation_controller import QtDictationController
            from vocalance.app.ui.controls.qt_grid_controller import QtGridController
            from vocalance.app.ui.controls.qt_marks_controller import QtMarksController
            from vocalance.app.ui.controls.qt_settings_controller import QtSettingsController
            from vocalance.app.ui.controls.qt_sound_controller import QtSoundController

            if hasattr(self, "_mark_service") and self._mark_service:
                self.marks_controller = QtMarksController(self.event_bus, self.event_loop, self._mark_service, self.config, self)

            if hasattr(self, "_grid_service") and self._grid_service:
                self.grid_controller = QtGridController(self.event_bus, self.event_loop, self._grid_service, self.config, self)

            if hasattr(self, "_sound_service") and self._sound_service:
                self.sound_controller = QtSoundController(
                    self.event_bus, self.event_loop, self._sound_service, self._storage_service, self.config, self
                )

            if hasattr(self, "_command_service") and self._command_service:
                self.commands_controller = QtCommandsController(
                    self.event_bus, self.event_loop, self._command_service, self.config, self
                )

            if hasattr(self, "_dictation_service") and self._dictation_service:
                self.dictation_controller = QtDictationController(
                    self.event_bus, self.event_loop, self._dictation_service, self.config, self
                )

                # Initialize alias controller using the alias service from dictation coordinator
                if hasattr(self._dictation_service, "alias_service"):
                    self.dictation_alias_controller = QtDictationAliasController(
                        self.event_bus, self.event_loop, self._dictation_service.alias_service, self
                    )

            if hasattr(self, "_settings_service") and self._settings_service:
                self.settings_controller = QtSettingsController(
                    self.event_bus, self.event_loop, self._settings_service, self.config, self
                )

            try:
                from vocalance.app.ui.controls.qt_dictation_popup_controller import QtDictationPopupController

                self.dictation_popup_controller = QtDictationPopupController(self.event_bus, self.event_loop)
            except Exception as e:
                self.logger.warning(f"Could not initialize dictation popup controller: {e}")

            self._initialize_overlay_views()
            self._connect_controllers_to_views()

            self.logger.info("All controllers initialized with services")

        except Exception as e:
            self.logger.error(f"Error initializing controllers: {e}", exc_info=True)

    def _initialize_overlay_views(self) -> None:
        try:
            from vocalance.app.ui.views.qt_grid_view import QtGridView
            from vocalance.app.ui.views.qt_mark_view import QtMarkView

            if self.marks_controller and hasattr(self, "_mark_service") and self._mark_service:
                self.mark_view = QtMarkView(mark_service=self._mark_service, config=self.config)
                self.mark_view.set_controller_callback(self.marks_controller)
                self.marks_controller.set_mark_view(self.mark_view)

            if self.grid_controller and hasattr(self, "_grid_service") and self._grid_service:
                click_tracker = getattr(self, "_click_tracker_service", None)
                self.grid_view = QtGridView(self.event_bus, self.event_loop, click_tracker, self.config)
                self.grid_view.set_controller_callback(self.grid_controller)
                self.grid_controller.set_grid_view(self.grid_view)

        except Exception as e:
            self.logger.error(f"Error initializing overlay views: {e}", exc_info=True)

    def _connect_controllers_to_views(self) -> None:
        if self.marks_controller and "Marks" in self._view_cache:
            self._view_cache["Marks"].set_controller(self.marks_controller)
        if self.sound_controller and "Sounds" in self._view_cache:
            self._view_cache["Sounds"].set_controller(self.sound_controller)
        if self.commands_controller and "Commands" in self._view_cache:
            self._view_cache["Commands"].set_controller(self.commands_controller)
        if self.dictation_controller and "Dictation" in self._view_cache:
            self._view_cache["Dictation"].set_controller(self.dictation_controller)
            # Also connect alias controller to the dictation view
            if self.dictation_alias_controller:
                self._view_cache["Dictation"].set_alias_controller(self.dictation_alias_controller)
        if self.settings_controller and "Settings" in self._view_cache:
            self._view_cache["Settings"].set_controller(self.settings_controller)
