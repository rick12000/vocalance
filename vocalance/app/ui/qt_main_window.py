import logging
from typing import Optional

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QCloseEvent, QColor, QDesktopServices, QIcon, QPalette
from PySide6.QtWidgets import QFrame, QHBoxLayout, QMainWindow, QStackedWidget, QVBoxLayout, QWidget

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.event_bus import EventBus
from vocalance.app.lifecycle import AppLifecycle
from vocalance.app.services.commands.utilities.input_executor import KeyboardInputService
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


class VocalanceMainWindow(QMainWindow):
    """Application shell: sidebar, stacked tab content, and header."""

    def __init__(
        self,
        event_bus: EventBus,
        logger: logging.Logger,
        config: GlobalAppConfig,
        input_service: KeyboardInputService,
        icon_manager: Optional[WindowIconManager] = None,
        lifecycle: Optional[AppLifecycle] = None,
    ) -> None:
        super().__init__()

        self.event_bus = event_bus
        self.logger = logger
        self.config = config
        self.icon_manager = icon_manager
        self._lifecycle = lifecycle
        self._input_service = input_service

        self.current_tab = "Commands"

        self.asset_cache = QtAssetCache(asset_paths_config=self.config.asset_paths)
        self.logo_service = QtLogoService(self.asset_cache)

        self._tab_views: dict[str, QWidget] = {}
        self._active_tab_view: Optional[QWidget] = None

        self._ui_registry = UiRegistry(
            event_bus=self.event_bus,
            logger=self.logger,
            config=self.config,
            main_window=self,
            input_service=self._input_service,
        )
        self._bind_registry_controllers()

        self._configure_window()
        self._build_main_layout()

    def _bind_registry_controllers(self) -> None:
        r = self._ui_registry
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

    def _configure_window(self) -> None:
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

    def _build_main_layout(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self._build_sidebar()
        main_layout.addWidget(self.sidebar_frame)

        self._build_sidebar_separator()
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

        self._build_header()
        self.content_border_frame.add(self.header_frame)

        self._build_content_stack()
        self.content_border_frame.add(self.content_widget, stretch=1)

        right_wrapper_layout.addWidget(self.content_border_frame, stretch=1)
        main_layout.addWidget(right_panel_wrapper, stretch=1)

        self.show_tab("Commands")

    def _build_sidebar(self) -> None:
        self.sidebar_frame = ExpandableSidebar()
        self.sidebar_button_manager = self.sidebar_frame.manager

        self._build_sidebar_buttons()
        self.sidebar_frame.add_widget(self.buttons_widget)
        self.sidebar_frame.add_stretch()
        self._build_sidebar_logo()
        self.sidebar_frame.add_widget(self.sidebar_logo_frame)

        if self.sidebar_buttons:
            first_button = list(self.sidebar_buttons.values())[0]
            self.sidebar_button_manager.select(first_button)

    def _build_sidebar_buttons(self) -> None:
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

    def _build_sidebar_logo(self) -> None:
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

    def _build_sidebar_separator(self) -> None:
        self.sidebar_separator = QFrame()
        self.sidebar_separator.setFrameShape(QFrame.Shape.NoFrame)
        self.sidebar_separator.setFixedWidth(theme.config.sidebar.border_width)
        self.sidebar_separator.setAutoFillBackground(True)
        sep_palette = self.sidebar_separator.palette()
        line = QColor(255, 255, 255)
        line.setAlpha(8)
        sep_palette.setColor(QPalette.ColorRole.Window, line)
        self.sidebar_separator.setPalette(sep_palette)

    def _build_header(self) -> None:
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
        self._build_header_documentation_button()
        header_layout.addWidget(
            self.header_icon_button,
            alignment=Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignCenter,
        )

        self.header_inner = header_inner
        outer_layout.addWidget(header_inner)

    def _build_header_documentation_button(self) -> None:
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
        self.header_icon_button.clicked.connect(self._open_documentation_url)

    def _open_documentation_url(self) -> None:
        QDesktopServices.openUrl(QUrl("https://www.vocalance.com/instructions.html"))

    def _build_content_stack(self) -> None:
        self.content_widget = TransparentBox()
        content_layout = self.content_widget.layout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        self.stacked_widget = QStackedWidget()
        content_layout.addWidget(self.stacked_widget)

    def _sync_header_subtitle(self, text: str) -> None:
        if not self.header_subtitle:
            self.header_subtitle = BodyLabel(text=text)
            title_container_widget = self.header_inner.layout().itemAt(0).widget()
            if title_container_widget:
                title_container_widget.layout().insertWidget(1, self.header_subtitle, alignment=Qt.AlignmentFlag.AlignLeft)
        else:
            self.header_subtitle.setText(text)

    TAB_SUBTITLES = {
        "Sounds": "Use custom sounds to control your computer",
        "Marks": "Pinpoint important locations on your screen",
        "Commands": "Manage voice commands and their actions",
        "Dictation": "Configure smart dictation with AI prompts",
        "Settings": "Configure default Vocalance settings",
    }

    def show_tab(self, tab_name: str) -> None:
        """Switch the stacked content to ``tab_name`` and refresh the header."""
        self.current_tab = tab_name
        self.header_label.setText(tab_name)
        subtitle = self.TAB_SUBTITLES.get(tab_name)
        if subtitle:
            self._sync_header_subtitle(subtitle)

        cached = self._tab_views.get(tab_name)

        if cached is None:
            view = self._ui_registry.create_tab_widget(tab_name)
            self._tab_views[tab_name] = view
            self._active_tab_view = view
            self.stacked_widget.addWidget(view)
            self.stacked_widget.setCurrentWidget(view)
        else:
            self._active_tab_view = cached
            self.stacked_widget.setCurrentWidget(cached)

    def closeEvent(self, close_event: QCloseEvent) -> None:
        """Accept the close and ask the lifecycle to tear everything down."""
        self.logger.debug("Main window close event")
        close_event.accept()
        if self._lifecycle is not None:
            self._lifecycle.request_shutdown(reason="User closed main window", source="main_window_close_event")

    def shutdown(self) -> None:
        """Dispose tab views and tear down the UI registry."""
        view_items = list(self._tab_views.items())
        self._tab_views.clear()
        self._active_tab_view = None

        for _view_name, view in view_items:
            view.deleteLater()

        self._ui_registry.shutdown()
