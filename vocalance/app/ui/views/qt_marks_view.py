"""Qt-based marks management view.

Displays marks with management capabilities matching legacy layout using TwoColumnTabLayout.
"""

import logging
from typing import List, Optional

from PySide6.QtWidgets import QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

from vocalance.app.events.mark_events import MarkData
from vocalance.app.ui.qt_theme import theme_manager
from vocalance.app.ui.views.components.qt_themed_components import DangerButton, InstructionTile, PrimaryButton, TwoColumnTabLayout


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
        """Build UI with TwoColumnTabLayout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create two-column layout with titles
        self.layout = TwoColumnTabLayout(self, "Instructions", "Manage Marks")
        main_layout.addWidget(self.layout)

        # Setup instruction panels
        self._setup_instructions_panel()
        self._setup_marks_panel()

    def _setup_instructions_panel(self) -> None:
        """Setup instructions panel in left content area."""
        container = self.layout.left_content

        # Get existing layout
        container_layout = container.layout()
        if container_layout is None:
            container_layout = QVBoxLayout(container)

        container_layout.setContentsMargins(
            theme_manager.two_box_layout.inner_content_padx,
            0,
            theme_manager.two_box_layout.inner_content_padx,
            0,
        )
        container_layout.setSpacing(theme_manager.spacing.small)

        # Instruction tiles
        tile1 = InstructionTile(
            title="Create Mark",
            content="Say 'Mark [name]' to create a mark\nat the current cursor position",
        )
        container_layout.addWidget(tile1)

        tile2 = InstructionTile(
            title="Navigate",
            content="Say the mark's [name] to automatically click\nat that position",
        )
        container_layout.addWidget(tile2)

        tile3 = InstructionTile(
            title="Manage Marks",
            content="Use the right panel to visualize and delete marks,\nor say 'show marks' to see them on screen",
        )
        container_layout.addWidget(tile3)

        container_layout.addStretch()

    def _setup_marks_panel(self) -> None:
        """Setup marks management panel in right content area."""
        container = self.layout.right_content

        # Get existing layout
        container_layout = container.layout()
        if container_layout is None:
            container_layout = QVBoxLayout(container)

        container_layout.setContentsMargins(
            theme_manager.two_box_layout.inner_content_padx,
            0,
            theme_manager.two_box_layout.inner_content_padx,
            0,
        )
        container_layout.setSpacing(theme_manager.spacing.small)

        # Marks list container with scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll_area.setStyleSheet("background: transparent; border: none;")

        self.marks_container = QWidget()
        self.marks_container.setStyleSheet("background: transparent;")
        self.marks_list_layout = QVBoxLayout(self.marks_container)
        self.marks_list_layout.setContentsMargins(0, 0, 0, 0)
        self.marks_list_layout.setSpacing(theme_manager.spacing.tiny)
        self.marks_list_layout.addStretch()

        scroll_area.setWidget(self.marks_container)
        container_layout.addWidget(scroll_area)

        # Button row
        button_layout = QHBoxLayout()
        button_layout.setSpacing(theme_manager.spacing.small)

        self.show_overlay_btn = PrimaryButton(text="Show Marks")
        self.show_overlay_btn.clicked.connect(self._on_show_overlay_clicked)
        button_layout.addWidget(self.show_overlay_btn)

        self.delete_all_btn = DangerButton(text="Delete All Marks")
        self.delete_all_btn.clicked.connect(self._on_delete_all_clicked)
        button_layout.addWidget(self.delete_all_btn)

        container_layout.addLayout(button_layout)

    def _display_marks(self, marks: List[MarkData]) -> None:
        """Display marks in the list."""
        # Clear existing marks
        while self.marks_list_layout.count() > 1:  # Keep the stretch
            item = self.marks_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Add marks
        for mark in marks:
            mark_widget = self._create_mark_widget(mark)
            self.marks_list_layout.insertWidget(self.marks_list_layout.count() - 1, mark_widget)

    def _create_mark_widget(self, mark: MarkData) -> QWidget:
        """Create a widget for a single mark."""
        widget = QWidget()
        widget.setProperty("itemType", "list_item")
        widget.setStyleSheet("background: transparent; border: none;")
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme_manager.spacing.small)

        # Mark label
        label_text = f"{mark.name}"
        label = QLabel(label_text)
        font = theme_manager.get_font(size=theme_manager.font_sizes.medium)
        label.setFont(font)
        label.setStyleSheet(f"color: {theme_manager.text_colors.light}; background: transparent; border: none;")
        layout.addWidget(label, stretch=1)

        # Delete button (pill-shaped)
        delete_btn = DangerButton(text="Delete")
        delete_btn.setFixedWidth(80)
        delete_btn.clicked.connect(lambda: self._on_delete_mark(mark.name))
        layout.addWidget(delete_btn)

        return widget

    def _on_show_overlay_clicked(self) -> None:
        """Handle show overlay button click."""
        if self.controller:
            self.controller.show_marks_overlay()

    def _on_delete_all_clicked(self) -> None:
        """Handle delete all button click."""
        if not self.marks_list:
            return

        # Confirm deletion
        from vocalance.app.ui.views.components.qt_themed_dialogs import askyesno

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
            # Refresh marks to get updated list
            if self.controller:
                self.controller.refresh_marks()
            self.logger.info(f"Mark created: {name}")
        except Exception as e:
            self.logger.error(f"Error handling mark created: {e}", exc_info=True)

    def _on_mark_deleted(self, name: str) -> None:
        """Handle mark deleted event."""
        try:
            # Refresh marks to get updated list
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
        from vocalance.app.ui.views.components.qt_themed_dialogs import showerror

        showerror(message, parent=self)
