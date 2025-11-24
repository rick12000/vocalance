"""Qt-based marks management view.

Displays marks with management capabilities matching legacy layout using TwoColumnLayout.
"""

import logging
from typing import List, Optional

from PySide6.QtWidgets import QVBoxLayout, QWidget

from vocalance.app.events.mark_events import MarkData
from vocalance.app.ui.components.complex_components import Tile
from vocalance.app.ui.components.layouts import ScrollableContainer, TransparentBox, TwoColumnLayout
from vocalance.app.ui.components.simple_components import Button, Label
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

        # Setup main layout - zero margins/spacing like QtBaseView
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
        # Create two-column layout with titles
        self.layout = TwoColumnLayout("Instructions", "Manage Marks", self)
        self.main_layout.addWidget(self.layout, stretch=1)

        # Setup instruction panels
        self._setup_instructions_panel()
        self._setup_marks_panel()

    def _setup_instructions_panel(self) -> None:
        """Setup instructions panel in left content area."""
        content = self.layout.left_content

        # Instruction tiles - use ContentArea.add() for systematic spacing
        content.add(Tile("Create Mark", "Say 'Mark [name]' to create a mark\nat the current cursor position"))
        content.add(Tile("Navigate", "Say the mark's [name] to automatically click\nat that position"))
        content.add(
            Tile("Manage Marks", "Use the right panel to visualize and delete marks,\nor say 'show marks' to see them on screen")
        )

        content.add_stretch()

    def _setup_marks_panel(self) -> None:
        """Setup marks management panel in right content area."""
        content = self.layout.right_content

        # Marks list container with scroll area - use ContentArea.add() for systematic spacing
        self.marks_scroll = ScrollableContainer()
        content.add(self.marks_scroll, stretch=1)

        # Button row
        button_box = TransparentBox(layout="horizontal", spacing=theme.config.spacing.small)

        self.show_overlay_btn = Button(text="Show Marks", command=self._on_show_overlay_clicked)
        button_box.add(self.show_overlay_btn)

        self.delete_all_btn = Button(text="Reset", variant="danger", command=self._on_delete_all_clicked)
        button_box.add(self.delete_all_btn)

        content.add(button_box)

    def _display_marks(self, marks: List[MarkData]) -> None:
        """Display marks in the list."""
        # Clear existing marks - ScrollableContainer has a content_widget/layout
        layout = self.marks_scroll.content_layout

        # Clear widgets
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Add marks
        for mark in marks:
            self.marks_scroll.add(self._create_mark_widget(mark))

        self.marks_scroll.add_stretch()

    def _create_mark_widget(self, mark: MarkData) -> QWidget:
        """Create a widget for a single mark."""
        # Using a TransparentBox as a row
        row = TransparentBox(layout="horizontal", spacing=theme.config.spacing.small)

        # Mark label - smaller font
        label = Label(f"{mark.name}", variant="small", color=theme.config.text.lightest)
        row.add(label, stretch=1)

        # Delete button
        delete_btn = Button("Delete", variant="danger", command=lambda: self._on_delete_mark(mark.name))
        delete_btn.setFixedWidth(80)
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

        # Confirm deletion
        from vocalance.app.ui.components.dialogs import askyesno

        # NOTE: qt_themed_dialogs needs to be checked if it's still valid or needs refactor
        # For now, assuming it works or using QMessageBox

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
        try:
            if self.controller:
                self.controller.refresh_marks()
            self.logger.info(f"Mark created: {name}")
        except Exception as e:
            self.logger.error(f"Error handling mark created: {e}", exc_info=True)

    def _on_mark_deleted(self, name: str) -> None:
        """Handle mark deleted event."""
        try:
            if self.controller:
                self.controller.refresh_marks()
            self.logger.info(f"Mark deleted: {name}")
        except Exception as e:
            self.logger.error(f"Error handling mark deleted: {e}", exc_info=True)

    def _on_all_deleted(self) -> None:
        """Handle all marks deleted event."""
        try:
            self.marks_list = []
            self._display_marks([])
            self.logger.info("All marks deleted")
        except Exception as e:
            self.logger.error(f"Error handling all marks deleted: {e}", exc_info=True)

    def _on_error(self, error_message: str) -> None:
        """Handle error from controller."""
        self.logger.error(f"Controller error: {error_message}")
        self._show_error(error_message)

    def _show_error(self, message: str) -> None:
        """Show error message dialog."""
        # Using standard MessageBox for now if dialogs not refactored
        from PySide6.QtWidgets import QMessageBox

        QMessageBox.critical(self, "Error", message)
