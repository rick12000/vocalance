import logging
from typing import TYPE_CHECKING, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QVBoxLayout, QWidget

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.event_bus import EventBus
from vocalance.app.ui.application.system_controller import QtSystemController
from vocalance.app.ui.components.labels import LargeLabel
from vocalance.app.ui.features.commands.controller import QtCommandsController
from vocalance.app.ui.features.commands.view import QtCommandsView
from vocalance.app.ui.features.dictation.alias_controller import QtDictationAliasController
from vocalance.app.ui.features.dictation.controller import QtDictationController
from vocalance.app.ui.features.dictation.popup_controller import QtDictationPopupController
from vocalance.app.ui.features.dictation.view import QtDictationView
from vocalance.app.ui.features.marks.controller import QtMarksController
from vocalance.app.ui.features.marks.view import QtMarksView
from vocalance.app.ui.features.overlays.grid_controller import QtGridController
from vocalance.app.ui.features.overlays.grid_overlay import QtGridView
from vocalance.app.ui.features.overlays.mark_overlay import QtMarkView
from vocalance.app.ui.features.settings.controller import QtSettingsController
from vocalance.app.ui.features.settings.view import QtSettingsView
from vocalance.app.ui.features.sounds.controller import QtSoundController
from vocalance.app.ui.features.sounds.view import QtSoundsView

if TYPE_CHECKING:
    from vocalance.qt_main import Services


class UiRegistry:
    """Constructs UI controllers, overlay views, and tab content widgets."""

    def __init__(
        self,
        event_bus: EventBus,
        logger: logging.Logger,
        config: GlobalAppConfig,
        services: "Services",
        main_window: QWidget,
    ) -> None:
        self.event_bus = event_bus
        self.logger = logger
        self.config = config
        self.services = services
        self._main_window = main_window

        s = services

        self.system_controller = QtSystemController(event_bus, main_window)

        self.marks_controller = QtMarksController(event_bus, config) if s.mark else None
        self.grid_controller = QtGridController(event_bus, config) if s.grid else None
        self.sound_controller = QtSoundController(event_bus, config) if s.sound_service else None
        self.commands_controller = QtCommandsController(event_bus, config) if s.command_management else None
        self.dictation_controller = QtDictationController(event_bus, config) if s.dictation else None
        self.dictation_alias_controller = QtDictationAliasController(event_bus) if s.dictation else None
        self.settings_controller = QtSettingsController(event_bus, config) if s.runtime_config else None

        self.dictation_popup_controller: Optional[QtDictationPopupController] = None
        try:
            self.dictation_popup_controller = QtDictationPopupController(event_bus)
        except Exception as e:
            logger.warning("Could not initialize dictation popup controller: %s", e)

        self.mark_view: Optional[QtMarkView] = None
        self.grid_view: Optional[QtGridView] = None
        self._init_overlays()

    def _init_overlays(self) -> None:
        s = self.services
        try:
            if self.marks_controller and s.mark:
                self.mark_view = QtMarkView(config=self.config)
                self.mark_view.bind_controller(self.marks_controller)
                self.marks_controller.set_view(self.mark_view)
            if self.grid_controller and s.grid:
                self.grid_view = QtGridView(self.event_bus, self.config, s.gui_event_loop)
                self.grid_controller.set_view(self.grid_view)
        except Exception as e:
            self.logger.error("Error initializing overlay views: %s", e, exc_info=True)

    def create_tab_widget(self, tab_name: str) -> QWidget:
        try:
            if tab_name == "Marks":
                view = QtMarksView()
                if self.marks_controller:
                    view.set_controller(self.marks_controller)
                return view
            if tab_name == "Sounds":
                view = QtSoundsView()
                if self.sound_controller:
                    view.set_controller(self.sound_controller)
                return view
            if tab_name == "Commands":
                view = QtCommandsView()
                if self.commands_controller:
                    view.set_controller(self.commands_controller)
                return view
            if tab_name == "Dictation":
                view = QtDictationView()
                if self.dictation_controller:
                    view.set_controller(self.dictation_controller)
                if self.dictation_alias_controller:
                    view.set_alias_controller(self.dictation_alias_controller)
                return view
            if tab_name == "Settings":
                view = QtSettingsView()
                if self.settings_controller:
                    view.set_controller(self.settings_controller)
                return view
        except Exception as e:
            self.logger.error("Error creating view for %s: %s", tab_name, e, exc_info=True)

        placeholder = QWidget()
        layout = QVBoxLayout(placeholder)
        label = LargeLabel(f"{tab_name} View\n(Fallback – check logs)")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        return placeholder

    def cleanup_controllers(self) -> None:
        for attr in (
            "marks_controller",
            "sound_controller",
            "dictation_controller",
            "dictation_alias_controller",
            "settings_controller",
            "commands_controller",
            "grid_controller",
            "system_controller",
            "dictation_popup_controller",
        ):
            controller = getattr(self, attr, None)
            if controller and hasattr(controller, "cleanup"):
                controller.cleanup()
