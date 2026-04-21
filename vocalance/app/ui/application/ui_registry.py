import logging
from typing import Optional

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


class UiRegistry:
    def __init__(
        self,
        event_bus: EventBus,
        logger: logging.Logger,
        config: GlobalAppConfig,
        main_window: QWidget,
    ) -> None:
        self.event_bus = event_bus
        self.logger = logger
        self.config = config
        self._main_window = main_window

        self.system_controller = QtSystemController(event_bus, main_window)

        self.marks_controller = QtMarksController(event_bus, config)
        self.grid_controller = QtGridController(event_bus, config)
        self.sound_controller = QtSoundController(event_bus, config)
        self.commands_controller = QtCommandsController(event_bus, config)
        self.dictation_controller = QtDictationController(event_bus, config)
        self.dictation_alias_controller = QtDictationAliasController(event_bus)
        self.settings_controller = QtSettingsController(event_bus, config)

        self.dictation_popup_controller: Optional[QtDictationPopupController] = QtDictationPopupController(event_bus)

        self.mark_view: Optional[QtMarkView] = None
        self.grid_view: Optional[QtGridView] = None
        self._init_overlays()

    def _init_overlays(self) -> None:
        import asyncio

        self.mark_view = QtMarkView(config=self.config)
        self.mark_view.bind_controller(self.marks_controller)
        self.marks_controller.set_view(self.mark_view)

        self.grid_view = QtGridView(self.event_bus, self.config, asyncio.get_running_loop())
        self.grid_controller.set_view(self.grid_view)

    def create_tab_widget(self, tab_name: str) -> QWidget:
        if tab_name == "Marks":
            view = QtMarksView()
            view.set_controller(self.marks_controller)
            return view
        if tab_name == "Sounds":
            view = QtSoundsView()
            view.set_controller(self.sound_controller)
            return view
        if tab_name == "Commands":
            view = QtCommandsView()
            view.set_controller(self.commands_controller)
            return view
        if tab_name == "Dictation":
            view = QtDictationView()
            view.set_controller(self.dictation_controller)
            view.set_alias_controller(self.dictation_alias_controller)
            return view
        if tab_name == "Settings":
            view = QtSettingsView()
            view.set_controller(self.settings_controller)
            return view

        placeholder = QWidget()
        layout = QVBoxLayout(placeholder)
        label = LargeLabel(f"Unknown tab: {tab_name}")
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
