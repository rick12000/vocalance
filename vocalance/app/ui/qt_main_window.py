import logging
from typing import TYPE_CHECKING, Optional

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QCloseEvent, QColor, QDesktopServices, QIcon, QPalette
from PySide6.QtWidgets import QFrame, QHBoxLayout, QMainWindow, QStackedWidget, QVBoxLayout, QWidget

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.event_bus import EventBus
from vocalance.app.shutdown_coordinator import ShutdownCoordinator
from vocalance.app.ui.application.ui_registry import UiRegistry
from vocalance.app.ui.components.header_icon_button import HeaderIconButton
from vocalance.app.ui.components.labels import BodyLabel, TitleLabel
from vocalance.app.ui.components.layouts import BaseContainer, TransparentBox
from vocalance.app.ui.components.sidebar_button import SidebarButton
from vocalance.app.ui.components.specialized import ExpandableSidebar
from vocalance.app.ui.qt_theme import theme
from vocalance.app.ui.utils.qt_assets import QtAssetCache
from vocalance.app.ui.utils.qt_logo_service import QtLogoService
from vocalance.app.ui.utils.window_icon_manager import WindowIconManager

if TYPE_CHECKING:
    from vocalance.qt_main import Services


class VocalanceMainWindow(QMainWindow):
    """Main application window.

    Receives the fully-initialised ``Services`` container at construction and
    wires up all controllers immediately.  No deferred service injection.
    """

    def __init__(
        self,
        event_bus: EventBus,
        logger: logging.Logger,
        config: GlobalAppConfig,
        services: "Services",
        icon_manager: Optional[WindowIconManager] = None,
        shutdown_coordinator: Optional[ShutdownCoordinator] = None,
    ) -> None:
        super().__init__()

        self.event_bus = event_bus
        self.logger = logger
        self.config = config
        self._services = services
        self.icon_manager = icon_manager
        self._shutdown_coordinator = shutdown_coordinator

        self.current_tab = "Commands"

        self.asset_cache = QtAssetCache(asset_paths_config=self.config.asset_paths)
        self.logo_service = QtLogoService(self.asset_cache)

        self._view_cache: dict[str, QWidget] = {}
        self._current_view: Optional[QWidget] = None

        self._registry = UiRegistry(
            event_bus=self.event_bus,
            logger=self.logger,
            config=self.config,
            services=self._services,
            main_window=self,
        )
        self._attach_registry_attributes()

        self._setup_window()
        self._build_ui()

        self.logger.debug("VocalanceMainWindow initialized")

    def _attach_registry_attributes(self) -> None:
        r = self._registry
        self.system_controller = r.system_controller
        self.marks_controller = r.marks_controller
        self.grid_controller = r.grid_controller
        self.sound_controller = r.sound_controller
        self.commands_controller = r.commands_controller
        self.dictation_controller = r.dictation_controller
        self.dictation_alias_controller = r.dictation_alias_controller
        self.settings_controller = r.settings_controller
        self.dictation_popup_controller = r.dictation_popup_controller
        self.mark_view = r.mark_view
        self.grid_view = r.grid_view

    def _setup_window(self) -> None:
        self.setWindowTitle("Vocalance")
        self.resize(
            theme.config.components.main_window_width,
            theme.config.components.main_window_height,
        )
        self.setMinimumSize(
            theme.config.components.main_window_min_width,
            theme.config.components.main_window_min_height,
        )
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(theme.config.shapes.darkest))
        self.setPalette(palette)
        self.setAutoFillBackground(True)

        if self.icon_manager and self.icon_manager.is_icon_loaded():
            self.icon_manager.apply_to_window(self)
        else:
            icon_path = self.asset_cache.get_icon_path()
            if icon_path and icon_path.exists():
                self.setWindowIcon(QIcon(str(icon_path)))

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self._create_sidebar()
        main_layout.addWidget(self.sidebar_frame)

        self._create_sidebar_separator()
        main_layout.addWidget(self.sidebar_separator)

        right_panel_wrapper = QWidget()
        right_wrapper_layout = QVBoxLayout(right_panel_wrapper)
        right_wrapper_layout.setContentsMargins(0, 0, 0, 0)
        right_wrapper_layout.setSpacing(0)

        self.content_border_frame = BaseContainer(
            layout="vertical",
            bg_color=theme.config.shapes.darkest,
            border_color=None,
            border_radius=0,
        )
        content_frame_layout = self.content_border_frame.layout()
        content_frame_layout.setContentsMargins(
            theme.config.container.box_padding,
            theme.config.container.box_padding,
            theme.config.container.box_padding,
            theme.config.container.box_padding,
        )
        content_frame_layout.setSpacing(theme.config.spacing.small)

        self._create_header()
        self.content_border_frame.add(self.header_frame)

        self._create_content_area()
        self.content_border_frame.add(self.content_widget, stretch=1)

        right_wrapper_layout.addWidget(self.content_border_frame, stretch=1)
        main_layout.addWidget(right_panel_wrapper, stretch=1)

        self.show_tab("Commands")

    def _create_sidebar(self) -> None:
        self.sidebar_frame = ExpandableSidebar()
        self.sidebar_button_manager = self.sidebar_frame.manager

        self._create_sidebar_buttons()
        self.sidebar_frame.add_widget(self.buttons_widget)
        self.sidebar_frame.add_stretch()
        self._create_sidebar_logo()
        self.sidebar_frame.add_widget(self.sidebar_logo_frame)

        if self.sidebar_buttons:
            first_button = list(self.sidebar_buttons.values())[0]
            self.sidebar_button_manager.select(first_button)

    def _create_sidebar_buttons(self) -> None:
        from vocalance.app.ui.utils.qt_icon_utils import load_sidebar_icon

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
        tabs = [
            ("Commands", "voice_selection_500dp_E3E3E3_FILL0_wght400_GRAD0_opsz48.png"),
            ("Marks", "location_on_500dp_E3E3E3_FILL0_wght400_GRAD0_opsz48.png"),
            ("Dictation", "speech_to_text_500dp_E3E3E3_FILL0_wght400_GRAD0_opsz48.png"),
            ("Sounds", "mic_500dp_E3E3E3_FILL0_wght400_GRAD0_opsz48.png"),
            ("Settings", "settings_500dp_E3E3E3_FILL0_wght400_GRAD0_opsz48.png"),
        ]
        icon_size = theme.config.sidebar.button_icon_size
        for tab_name, icon_filename in tabs:
            icon_pixmap = load_sidebar_icon(
                icon_filename=icon_filename,
                icons_dir=self.asset_cache.get_icons_dir(),
                target_color=theme.config.shapes.accent,
                icon_size=icon_size,
            )
            btn = SidebarButton(text=tab_name, icon_pixmap=icon_pixmap)
            btn.clicked.connect(lambda checked=False, tab=tab_name: self.show_tab(tab))
            self.buttons_widget.add(btn)
            self.sidebar_buttons[tab_name] = btn
            self.sidebar_button_manager.add(btn)

    def _create_sidebar_logo(self) -> None:
        logo_frame = TransparentBox(layout="horizontal")
        logo_layout = logo_frame.layout()
        logo_layout.setContentsMargins(0, theme.config.sidebar.logo_padding_top, 0, theme.config.sidebar.logo_padding_bottom)
        logo_layout.setSpacing(0)

        logo_area = QWidget()
        logo_area.setFixedWidth(theme.config.sidebar.collapsed_width)
        logo_area.setAutoFillBackground(False)
        from PySide6.QtWidgets import QHBoxLayout as _HBox

        logo_area_layout = _HBox(logo_area)
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

        logo_area_layout.addStretch()
        logo_area_layout.addWidget(self.sidebar_logo, alignment=Qt.AlignmentFlag.AlignCenter)
        logo_area_layout.addStretch()

        logo_layout.addWidget(logo_area)
        logo_layout.addStretch()

        self.sidebar_logo_frame = logo_frame

    def _create_sidebar_separator(self) -> None:
        self.sidebar_separator = QFrame()
        self.sidebar_separator.setFrameShape(QFrame.Shape.NoFrame)
        self.sidebar_separator.setFixedWidth(theme.config.sidebar.border_width)
        self.sidebar_separator.setAutoFillBackground(True)
        sep_palette = self.sidebar_separator.palette()
        line = QColor(255, 255, 255)
        line.setAlpha(8)
        sep_palette.setColor(QPalette.ColorRole.Window, line)
        self.sidebar_separator.setPalette(sep_palette)

    def _create_header(self) -> None:
        self.header_frame = QWidget()
        self.header_frame.setAutoFillBackground(False)
        outer_layout = QVBoxLayout(self.header_frame)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

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

        title_container = TransparentBox()
        title_layout = title_container.layout()
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(theme.config.header.spacing)

        self.header_label = TitleLabel(text="Welcome to Vocalance!")
        title_layout.addWidget(self.header_label, alignment=Qt.AlignmentFlag.AlignLeft)
        self.header_subtitle: Optional[BodyLabel] = None
        title_layout.addStretch()

        header_layout.addWidget(title_container, stretch=1)
        self._create_header_icon_button()
        header_layout.addWidget(
            self.header_icon_button,
            alignment=Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignCenter,
        )

        self.header_inner = header_inner
        outer_layout.addWidget(header_inner)

    def _create_header_icon_button(self) -> None:
        from vocalance.app.ui.utils.qt_icon_utils import load_sidebar_icon

        icon_pixmap = load_sidebar_icon(
            icon_filename=theme.config.icon_properties.documentation_icon_filename,
            icons_dir=self.asset_cache.get_icons_dir(),
            target_color=theme.config.shapes.accent,
            icon_size=theme.config.header.icon_size,
        )
        self.header_icon_button = HeaderIconButton(
            text="User Guide",
            icon_pixmap=icon_pixmap,
            icon_size=theme.config.header.icon_size,
            text_icon_spacing=theme.config.header.text_icon_spacing,
        )
        self.header_icon_button.clicked.connect(self._on_documentation_clicked)

    def _on_documentation_clicked(self) -> None:
        QDesktopServices.openUrl(QUrl("https://www.vocalance.com/instructions.html"))

    def _create_content_area(self) -> None:
        self.content_widget = TransparentBox()
        content_layout = self.content_widget.layout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        self.stacked_widget = QStackedWidget()
        content_layout.addWidget(self.stacked_widget)

    def _set_header_subtitle(self, text: str) -> None:
        if not self.header_subtitle:
            self.header_subtitle = BodyLabel(text=text)
            title_container_widget = self.header_inner.layout().itemAt(0).widget()
            if title_container_widget:
                title_container_widget.layout().insertWidget(1, self.header_subtitle, alignment=Qt.AlignmentFlag.AlignLeft)
        else:
            self.header_subtitle.setText(text)

    # ------------------------------------------------------------------
    # Tab management
    # ------------------------------------------------------------------

    _TAB_SUBTITLES = {
        "Sounds": "Use custom sounds to control your computer",
        "Marks": "Pinpoint important locations on your screen",
        "Commands": "Manage voice commands and their actions",
        "Dictation": "Configure smart dictation with AI prompts",
        "Settings": "Configure default Vocalance settings",
    }

    def show_tab(self, tab_name: str) -> None:
        self.current_tab = tab_name
        self.header_label.setText(tab_name)
        subtitle = self._TAB_SUBTITLES.get(tab_name)
        if subtitle:
            self._set_header_subtitle(subtitle)

        cached = self._view_cache.get(tab_name)

        if cached is None:
            view = self._registry.create_tab_widget(tab_name)
            self._view_cache[tab_name] = view
            self._current_view = view
            self.stacked_widget.addWidget(view)
            self.stacked_widget.setCurrentWidget(view)
        else:
            self._current_view = cached
            self.stacked_widget.setCurrentWidget(cached)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, close_event: QCloseEvent) -> None:
        self.logger.info("Main window close event")
        self._cleanup_controllers()
        close_event.accept()
        if self._shutdown_coordinator:
            self._shutdown_coordinator.request_shutdown(reason="User closed main window", source="main_window_close_event")

    def _cleanup_controllers(self) -> None:
        view_items = list(self._view_cache.items())
        self._view_cache.clear()
        self._current_view = None

        for view_name, view in view_items:
            try:
                if hasattr(view, "deleteLater"):
                    view.deleteLater()
            except Exception as e:
                self.logger.debug("Error deleting cached view %s: %s", view_name, e)

        self._registry.cleanup_controllers()
