"""Qt-based marks management view.

Displays marks with management capabilities using new component subclasses.
"""

import logging
from typing import List, Optional

from PySide6.QtWidgets import QMessageBox, QVBoxLayout, QWidget

from vocalance.app.events.mark_events import MarkData
from vocalance.app.ui.components.buttons import DangerButton, DeleteButton, PrimaryButton
from vocalance.app.ui.components.complex_components import Tile
from vocalance.app.ui.components.dialogs import askyesno
from vocalance.app.ui.components.labels import BodyLabel, SmallLabel
from vocalance.app.ui.components.layouts import ScrollableContainer, TransparentBox, TwoColumnLayout
from vocalance.app.ui.qt_theme import theme


class QtMarksView(QWidget):
    """Qt-based marks management view.

    Features:
    - Left panel: Instruction tiles
    - Right panel: Marks list + buttons
    - Display marks with coordinates or descriptions
    - Delete marks with confirmation
    - Real-time updates from controller
    """

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize marks view."""
        super().__init__(parent)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.controller = None
        self.marks_list: List[MarkData] = []

        # Setup main layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        self._setup_ui()
        self.logger.debug("QtMarksView initialized")

    def set_controller(self, controller) -> None:
        """Set the controller and connect signals."""
        self.controller = controller

        # Connect controller signals
        self.controller.marks_loaded.connect(self._on_marks_loaded)
        self.controller.mark_created.connect(self._on_mark_created)
        self.controller.mark_deleted.connect(self._on_mark_deleted)
        self.controller.all_marks_deleted.connect(self._on_all_deleted)
        self.controller.operation_error.connect(self._on_error)

        # Load initial marks
        self.logger.info("Loading marks from controller")
        self.controller.refresh_marks()

    def _setup_ui(self) -> None:
        """Build UI with TwoColumnLayout."""
        self.layout = TwoColumnLayout("Instructions", "Manage Marks", self)
        self.main_layout.addWidget(self.layout, stretch=1)

        self._setup_instructions_panel()
        self._setup_marks_panel()

    def _setup_instructions_panel(self) -> None:
        """Setup instructions panel in left content area."""
        content = self.layout.left_content

        content.add(Tile("Create Mark", "Say 'Mark [name]' to create a mark\nat the current cursor position"), stretch=1)
        content.add(Tile("Navigate", "Say the mark's [name] to automatically click\nat that position"), stretch=1)
        content.add(
            Tile("Manage Marks", "Use the right panel to visualize and delete marks,\nor say 'show marks' to see them on screen"),
            stretch=1,
        )

    def _setup_marks_panel(self) -> None:
        """Setup marks management panel in right content area."""
        content = self.layout.right_content

        # Marks list container with scroll area
        self.marks_scroll = ScrollableContainer()
        content.add(self.marks_scroll, stretch=1)

        # Button row
        button_box = TransparentBox(layout="horizontal", spacing=theme.config.spacing.small)

        self.show_overlay_btn = PrimaryButton(text="Show Marks", command=self._on_show_overlay_clicked)
        button_box.add(self.show_overlay_btn)

        self.delete_all_btn = DangerButton(text="Reset", command=self._on_delete_all_clicked)
        button_box.add(self.delete_all_btn)

        content.add(button_box)

    def _display_marks(self, marks: List[MarkData]) -> None:
        """Display marks in the list."""
        layout = self.marks_scroll.content_layout

        # Clear widgets
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not marks:
            # Display empty state message
            empty_label = BodyLabel(
                "No available marks.\nRead left panel to create marks.",
                align="center",
                color=theme.config.text.medium,
            )
            self.marks_scroll.add(empty_label)
        else:
            # Add marks
            for mark in marks:
                self.marks_scroll.add(self._create_mark_widget(mark))

        self.marks_scroll.add_stretch()

    def _create_mark_widget(self, mark: MarkData) -> QWidget:
        """Create a widget for a single mark."""
        row = TransparentBox(layout="horizontal", spacing=theme.config.spacing.small)

        # Mark label
        label = SmallLabel(f"{mark.name}", color=theme.config.text.medium)
        row.add(label, stretch=1)

        # Delete button
        delete_btn = DeleteButton(command=lambda: self._on_delete_mark(mark.name))
        row.add(delete_btn)

        return row

    def _on_show_overlay_clicked(self) -> None:
        """Handle show overlay button click."""
        if self.controller:
            self.controller.show_marks_overlay()

    def _on_delete_all_clicked(self) -> None:
        """Handle delete all button click."""
        if not self.marks_list:
            return

        confirmed = askyesno("Are you sure you want to delete all marks?", parent=self)

        if confirmed and self.controller:
            self.controller.delete_all_marks()

    def _on_delete_mark(self, mark_name: str) -> None:
        """Handle delete single mark."""
        if self.controller:
            self.controller.delete_mark(mark_name)

    def _on_marks_loaded(self, marks_list: List[MarkData]) -> None:
        """Handle marks loaded from controller."""
        try:
            self.marks_list = marks_list
            self._display_marks(marks_list)
            self.logger.info(f"Marks loaded: {len(marks_list)} total")
        except Exception as e:
            self.logger.error(f"Error loading marks: {e}", exc_info=True)
            self._show_error(f"Error loading marks: {e}")

    def _on_mark_created(self, name: str, x: int, y: int) -> None:
        """Handle mark created event."""
        if self.controller:
            self.controller.refresh_marks()
        self.logger.info(f"Mark created: {name}")

    def _on_mark_deleted(self, name: str) -> None:
        """Handle mark deleted event."""
        if self.controller:
            self.controller.refresh_marks()
        self.logger.info(f"Mark deleted: {name}")

    def _on_all_deleted(self) -> None:
        """Handle all marks deleted event."""
        self.marks_list = []
        self._display_marks([])
        self.logger.info("All marks deleted")

    def _on_error(self, error_message: str) -> None:
        """Handle error from controller."""
        self.logger.error(f"Controller error: {error_message}")
        self._show_error(error_message)

    def _show_error(self, message: str) -> None:
        """Show error message dialog."""
        QMessageBox.critical(self, "Error", message)
