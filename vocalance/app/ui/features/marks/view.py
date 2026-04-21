import logging
from typing import List, Optional

from PySide6.QtWidgets import QMessageBox, QVBoxLayout, QWidget

from vocalance.app.events.mark_events import MarkData
from vocalance.app.ui.components.buttons import DangerButton, DeleteButton, PrimaryButton
from vocalance.app.ui.components.dialogs import askyesno
from vocalance.app.ui.components.labels import BodyLabel, SmallLabel
from vocalance.app.ui.components.layouts import ScrollableContainer, TransparentBox, TwoColumnLayout
from vocalance.app.ui.components.tile import Tile
from vocalance.app.ui.qt_theme import theme


class QtMarksView(QWidget):
    """Marks management panel: instructions, list, overlay trigger, and reset.

    Updates react to ``QtMarksController`` signals.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.controller = None
        self.marks_list: List[MarkData] = []

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        self.setup_ui()
        self.logger.debug("QtMarksView initialized")

    def set_controller(self, controller) -> None:
        """Wire signals and request the initial marks list."""
        self.controller = controller

        self.controller.marks_loaded.connect(self.on_marks_loaded)
        self.controller.operation_error.connect(self.on_error)

        self.logger.info("Loading marks from controller")
        self.controller.refresh_marks()

    def setup_ui(self) -> None:
        self.layout = TwoColumnLayout("Instructions", "Manage Marks", self)
        self.main_layout.addWidget(self.layout, stretch=1)

        self.setup_instructions_panel()
        self.setup_marks_panel()

    def setup_instructions_panel(self) -> None:
        content = self.layout.left_content

        content.add(Tile("Create Mark", "Say 'Mark [name]' to create a mark\nat the current cursor position"), stretch=1)
        content.add(Tile("Navigate", "Say the mark's [name] to automatically click\nat that position"), stretch=1)
        content.add(
            Tile(
                "Manage Marks",
                "Use the right panel to visualize and delete marks,\nor say 'show marks' to see them on screen",
            ),
            stretch=1,
        )

    def setup_marks_panel(self) -> None:
        content = self.layout.right_content

        self.marks_scroll = ScrollableContainer()
        content.add(self.marks_scroll, stretch=1)

        button_box = TransparentBox(layout="horizontal", spacing=theme.config.spacing.small)

        self.show_overlay_btn = PrimaryButton(text="Show Marks", command=self.on_show_overlay_clicked)
        button_box.add(self.show_overlay_btn)

        self.delete_all_btn = DangerButton(text="Reset", command=self.on_delete_all_clicked)
        button_box.add(self.delete_all_btn)

        content.add(button_box)

    def display_marks(self, marks: List[MarkData]) -> None:
        layout = self.marks_scroll.content_layout

        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not marks:
            empty_label = BodyLabel(
                "No available marks.\nRead left panel to create marks.",
                align="center",
                color=theme.config.text.medium,
            )
            self.marks_scroll.add(empty_label)
        else:
            for mark in marks:
                self.marks_scroll.add(self.create_mark_widget(mark))

        self.marks_scroll.add_stretch()

    def create_mark_widget(self, mark: MarkData) -> QWidget:
        row = TransparentBox(layout="horizontal", spacing=theme.config.spacing.small)

        label = SmallLabel(f"{mark.name}", color=theme.config.text.medium)
        row.add(label, stretch=1)

        delete_btn = DeleteButton(command=lambda: self.on_delete_mark(mark.name))
        row.add(delete_btn)

        return row

    def on_show_overlay_clicked(self) -> None:
        if self.controller:
            self.controller.show_mark_overlay()

    def on_delete_all_clicked(self) -> None:
        if not self.marks_list:
            return

        confirmed = askyesno("Are you sure you want to delete all marks?", parent=self)

        if confirmed and self.controller:
            self.controller.delete_all_marks()

    def on_delete_mark(self, mark_name: str) -> None:
        if self.controller:
            self.controller.delete_mark(mark_name)

    def on_marks_loaded(self, marks_list: List[MarkData]) -> None:
        self.marks_list = marks_list
        self.display_marks(marks_list)
        self.logger.info("Marks loaded: %s total", len(marks_list))

    def on_error(self, error_message: str) -> None:
        self.logger.error("Controller error: %s", error_message)
        self.show_error_message(error_message)

    def show_error_message(self, message: str) -> None:
        QMessageBox.critical(self, "Error", message)
