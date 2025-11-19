"""Qt-based marks management view - FULLY INTEGRATED WITH INSTRUCTION TILES.

Displays marks with management capabilities matching legacy layout.
"""

import logging
from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QMessageBox, QScrollArea, QVBoxLayout, QWidget

from vocalance.app.events.mark_events import MarkData
from vocalance.app.ui.qt_theme import theme_manager
from vocalance.app.ui.views.components.qt_themed_components import DangerButton, PrimaryButton, ThemedFrame, TitleLabel


class InstructionTile(QWidget):
    """Instruction tile widget matching legacy design."""

    def __init__(self, title: str, content: str, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)

        # Title label
        title_label = QLabel(title)
        title_font = theme_manager.get_font(size=theme_manager.font_sizes.medium, weight="bold")
        title_label.setFont(title_font)
        title_label.setStyleSheet(f"color: {theme_manager.text_colors.light};")
        title_label.setWordWrap(True)
        layout.addWidget(title_label)

        # Content label
        content_label = QLabel(content)
        content_font = theme_manager.get_font(size=theme_manager.font_sizes.small)
        content_label.setFont(content_font)
        content_label.setStyleSheet(f"color: {theme_manager.text_colors.medium};")
        content_label.setWordWrap(True)
        layout.addWidget(content_label)

        # Style the tile
        self.setStyleSheet(
            f"""
            QWidget {{
                background-color: {theme_manager.shape_colors.dark};
                border: 1px solid {theme_manager.shape_colors.medium};
                border-radius: {theme_manager.border_radius.medium}px;
            }}
        """
        )


class QtMarksView(QWidget):
    """Qt-based marks management view - FULLY INTEGRATED WITH LEGACY LAYOUT.

    Features:
    - Left panel: Instruction tiles
    - Right panel: Marks list + buttons
    - Display marks with coordinates or descriptions
    - Double-click to delete with confirmation
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
        """Build two-box layout with instruction tiles."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(
            theme_manager.two_box_layout.outer_padding_left,
            theme_manager.two_box_layout.outer_padding_top,
            theme_manager.two_box_layout.outer_padding_right,
            theme_manager.two_box_layout.outer_padding_bottom,
        )
        main_layout.setSpacing(theme_manager.two_box_layout.base_spacing)

        # Create two-box layout
        boxes_layout = QHBoxLayout()
        boxes_layout.setSpacing(theme_manager.two_box_layout.base_spacing)

        # LEFT BOX - Instructions
        left_box = ThemedFrame(frame_type="two_box")
        left_layout = QVBoxLayout(left_box)
        left_layout.setContentsMargins(
            theme_manager.two_box_layout.inner_content_padx,
            20,
            theme_manager.two_box_layout.inner_content_padx,
            20,
        )
        left_layout.setSpacing(15)

        # Title
        left_layout.addWidget(TitleLabel(text="Instructions"))

        # Instruction tiles
        tile1 = InstructionTile("Create Mark", "Say 'Mark [name]' to create a mark\nat the current cursor position")
        left_layout.addWidget(tile1)

        tile2 = InstructionTile("Navigate", "Say the mark's [name] to automatically click\nat that position")
        left_layout.addWidget(tile2)

        tile3 = InstructionTile(
            "Manage Marks", "Use the right panel to visualize and delete marks,\nor say 'show marks' to see them on screen"
        )
        left_layout.addWidget(tile3)

        left_layout.addStretch()

        # RIGHT BOX - Marks management
        right_box = ThemedFrame(frame_type="two_box")
        right_layout = QVBoxLayout(right_box)
        right_layout.setContentsMargins(
            theme_manager.two_box_layout.inner_content_padx,
            20,
            theme_manager.two_box_layout.inner_content_padx,
            20,
        )
        right_layout.setSpacing(10)

        # Title
        right_layout.addWidget(TitleLabel(text="Manage Marks"))

        # Marks list container with scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        self.marks_container = QWidget()
        self.marks_list_layout = QVBoxLayout(self.marks_container)
        self.marks_list_layout.setContentsMargins(0, 0, 0, 0)
        self.marks_list_layout.setSpacing(5)

        scroll_area.setWidget(self.marks_container)
        right_layout.addWidget(scroll_area)

        # Button row
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        self.show_overlay_btn = PrimaryButton(text="Show Marks")
        self.show_overlay_btn.clicked.connect(self._on_show_overlay_clicked)
        button_layout.addWidget(self.show_overlay_btn)

        self.delete_all_btn = DangerButton(text="Delete All Marks")
        self.delete_all_btn.clicked.connect(self._on_delete_all_clicked)
        button_layout.addWidget(self.delete_all_btn)

        right_layout.addLayout(button_layout)

        # Add boxes to main layout
        boxes_layout.addWidget(left_box, 0)
        boxes_layout.addWidget(right_box, 1)

        main_layout.addLayout(boxes_layout)

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
            self.marks_list.clear()
            self._display_marks([])
            self.logger.info("All marks deleted")
        except Exception as e:
            self.logger.error(f"Error handling all deleted: {e}", exc_info=True)

    def _on_error(self, error_msg: str) -> None:
        """Handle error from controller."""
        self._show_error(error_msg)

    def _display_marks(self, marks: List[MarkData]) -> None:
        """Display marks in the scrollable layout with delete buttons (matching legacy)."""
        # Clear existing items
        while self.marks_list_layout.count():
            item = self.marks_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not marks:
            # Show empty message
            empty_label = QLabel("No available marks.\nRead the left panel to create a mark.")
            empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty_font = theme_manager.get_font(size=theme_manager.font_sizes.medium)
            empty_label.setFont(empty_font)
            empty_label.setStyleSheet(f"color: {theme_manager.text_colors.medium};")
            self.marks_list_layout.addWidget(empty_label)
        else:
            for mark in marks:
                # Format: "name - description" (NO coordinates shown - legacy behavior)
                mark_info = mark.name
                if mark.description:
                    mark_info += f" - {mark.description}"

                # Create item widget with mark name on left and delete button on right
                item_widget = QWidget()
                item_layout = QHBoxLayout(item_widget)
                item_layout.setContentsMargins(5, 5, 5, 5)
                item_layout.setSpacing(10)

                # Mark name label
                name_label = QLabel(mark_info)
                name_font = theme_manager.get_font(size=theme_manager.font_sizes.medium)
                name_label.setFont(name_font)
                name_label.setStyleSheet(f"color: {theme_manager.text_colors.light};")
                item_layout.addWidget(name_label, 1)

                # Delete button
                delete_btn = DangerButton(text="Delete")
                delete_btn.clicked.connect(lambda checked, m=mark.name: self._on_mark_delete_clicked(m))
                item_layout.addWidget(delete_btn)

                # Style the item
                item_widget.setStyleSheet(
                    f"""
                    QWidget {{
                        background-color: {theme_manager.shape_colors.dark};
                        border: 1px solid {theme_manager.shape_colors.medium};
                        border-radius: {theme_manager.border_radius.small}px;
                    }}
                """
                )

                self.marks_list_layout.addWidget(item_widget)

    def _on_mark_delete_clicked(self, mark_name: str) -> None:
        """Handle delete button clicked for a specific mark."""
        if mark_name and self.controller:
            reply = QMessageBox.question(
                self, "Delete Mark", f"Delete mark '{mark_name}'?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.controller.delete_mark_by_name(mark_name)

    def _on_show_overlay_clicked(self) -> None:
        """Handle show overlay button clicked."""
        if self.controller:
            self.controller.request_show_overlay()
            self.logger.info("Show overlay requested")

    def _on_delete_all_clicked(self) -> None:
        """Handle delete all button clicked."""
        if not self.marks_list:
            QMessageBox.information(self, "No Marks", "There are no marks to delete.")
            return

        if self.controller:
            reply = QMessageBox.question(
                self,
                "Delete All Marks",
                f"Delete all {len(self.marks_list)} marks?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.controller.delete_all_marks()

    def _show_error(self, message: str) -> None:
        """Show error message dialog."""
        QMessageBox.critical(self, "Error", message)
