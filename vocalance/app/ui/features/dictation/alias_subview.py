import logging
from typing import Dict, Optional

from PySide6.QtWidgets import QDialog, QHBoxLayout, QMessageBox, QVBoxLayout, QWidget

from vocalance.app.ui.components.buttons import DeleteButton, PrimaryButton
from vocalance.app.ui.components.dialogs import BaseDialog
from vocalance.app.ui.components.inputs import TextInput
from vocalance.app.ui.components.labels import BodyLabel, SmallLabel
from vocalance.app.ui.components.layouts import Box, ScrollableContainer, TransparentWidget, TwoColumnLayout
from vocalance.app.ui.qt_theme import theme


class AliasEditDialog(BaseDialog):
    """Dialog for editing an existing alias.

    Provides interface to edit the substitution value for an alias.
    """

    def __init__(self, key: str, value: str, parent: Optional[QWidget] = None):
        """Initialize alias edit dialog.

        Args:
            key: Alias activation phrase.
            value: Current substitution text.
            parent: Parent widget.
        """
        super().__init__(
            parent=parent,
            title=f"Edit Alias: {key}",
            min_width=500,
            min_height=250,
        )

        self.key = key
        self.original_value = value
        self.result_saved = False
        self.new_value = None

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Build the dialog UI."""
        from PySide6.QtGui import QColor, QPalette

        # Set dialog background to darkest
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(theme.config.shapes.darkest))
        self.setPalette(palette)
        self.setAutoFillBackground(True)

        # Activation Phrase section (read-only display)
        phrase_frame = Box()

        phrase_title = SmallLabel("Activation Phrase:", color=theme.config.text.medium)
        phrase_frame.add(phrase_title)

        phrase_display = SmallLabel(self.key, color=theme.config.text.light)
        phrase_frame.add(phrase_display)

        self._main_layout.addWidget(phrase_frame)

        # Substitution Text section
        substitution_frame = Box()

        value_label = SmallLabel("Substitution Text:", color=theme.config.text.medium)
        substitution_frame.add(value_label)

        self.value_entry = TextInput()
        self.value_entry.setText(self.original_value)
        self.value_entry.selectAll()
        substitution_frame.add(self.value_entry)

        self._main_layout.addWidget(substitution_frame)

        # Buttons at the bottom
        button_layout = QHBoxLayout()
        button_layout.setSpacing(theme.config.spacing.medium)

        save_btn = PrimaryButton(text="Save Changes")
        save_btn.clicked.connect(self._on_save)
        button_layout.addWidget(save_btn)

        cancel_btn = PrimaryButton(text="Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        self._main_layout.addLayout(button_layout)

        # Focus entry field
        self.value_entry.setFocus()

    def _on_save(self) -> None:
        """Handle save button click."""
        new_value = self.value_entry.text().strip()

        if not new_value:
            QMessageBox.warning(self, "Validation Error", "Please enter a substitution text.")
            return

        self.result_saved = True
        self.new_value = new_value
        self.accept()


class QtDictationAliasSubView(QWidget):
    """Qt-based dictation alias management sub-view.

    Features:
    - Add alias form (activation phrase + substitution text)
    - Aliases list with edit and delete buttons
    - Real-time updates from controller
    """

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize alias sub-view."""
        super().__init__(parent)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.controller = None
        self.aliases: Dict[str, str] = {}

        self._setup_ui()
        self.logger.debug("QtDictationAliasSubView initialized")

    def set_controller(self, controller) -> None:
        """Set the controller and connect signals."""
        self.controller = controller

        # Connect controller signals
        self.controller.aliases_loaded.connect(self._on_aliases_loaded)
        self.controller.operation_error.connect(self._on_error)
        self.controller.status_updated.connect(self._on_status_updated)

        # Load initial aliases
        self.logger.info("Loading aliases from controller")
        self.controller.refresh_aliases()

    def _setup_ui(self) -> None:
        """Build UI with TwoColumnLayout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create two-column layout with titles
        self.layout_widget = TwoColumnLayout("Add Alias", "Manage Aliases", self)
        main_layout.addWidget(self.layout_widget)

        # Setup panels
        self._setup_add_alias_form()
        self._setup_manage_aliases_panel()

    def _setup_add_alias_form(self) -> None:
        """Setup add alias form in left content area."""
        content = self.layout_widget.left_content

        # Activation phrase input
        activation_label = BodyLabel("Activation Text:")
        content.add(activation_label)

        self.key_entry = TextInput(placeholder="e.g. email")
        content.add(self.key_entry)

        # Substitution text input
        substitution_label = BodyLabel("Substitution Text:")
        content.add(substitution_label)

        self.value_entry = TextInput(placeholder="e.g. john@something.com")
        content.add(self.value_entry)

        # Add button
        self.add_alias_btn = PrimaryButton(text="Add Alias")
        self.add_alias_btn.clicked.connect(self._on_add_alias_clicked)
        content.add(self.add_alias_btn)

        content.add_stretch()

    def _setup_manage_aliases_panel(self) -> None:
        """Setup manage aliases panel in right content area."""
        content = self.layout_widget.right_content

        # Aliases list widget
        self.aliases_list_widget = TransparentWidget()
        self.aliases_list_layout = QVBoxLayout(self.aliases_list_widget)
        self.aliases_list_layout.setSpacing(theme.config.container.list_item_spacing)
        self.aliases_list_layout.setContentsMargins(0, 0, theme.config.spacing.small, 0)

        # Scroll area for aliases
        scroll_area = ScrollableContainer()
        scroll_area.content_layout.addWidget(self.aliases_list_widget)
        content.add(scroll_area, stretch=1)

    def _on_aliases_loaded(self, aliases: Dict[str, str]) -> None:
        """Handle aliases loaded from controller."""
        try:
            self.aliases = aliases
            self._display_aliases(aliases)
            self.logger.info(f"Aliases loaded: {len(aliases)} total")
        except Exception as e:
            self.logger.error(f"Error loading aliases: {e}", exc_info=True)
            self._show_error(f"Error loading aliases: {e}")

    def _on_status_updated(self, message: str, is_error: bool) -> None:
        """Handle status updates from controller."""
        if is_error:
            self._show_error(message)

    def _on_error(self, error_msg: str) -> None:
        """Handle error from controller."""
        self._show_error(error_msg)

    def _display_aliases(self, aliases: Dict[str, str]) -> None:
        """Display aliases in the list."""
        # Clear existing items
        while self.aliases_list_layout.count():
            item = self.aliases_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not aliases:
            empty_label = BodyLabel(
                "No available aliases.\nCreate aliases in the left panel.", align="center", color=theme.config.text.medium
            )
            self.aliases_list_layout.addWidget(empty_label)
        else:
            # Sort aliases alphabetically by key
            for key in sorted(aliases.keys()):
                value = aliases[key]
                self._create_alias_item(key, value)

        self.aliases_list_layout.addStretch()

    def _create_alias_item(self, key: str, value: str) -> None:
        """Create an alias list item with key, edit, and delete buttons."""
        # Create item widget
        item_widget = TransparentWidget()
        item_layout = QHBoxLayout(item_widget)
        item_layout.setContentsMargins(
            0,
            theme.config.container.list_item_padding_vertical,
            0,
            theme.config.container.list_item_padding_vertical,
        )
        item_layout.setSpacing(theme.config.spacing.small)

        # Key label (activation phrase) - no quotes, medium color
        key_label = SmallLabel(key, color=theme.config.text.medium)
        item_layout.addWidget(key_label, stretch=1)

        # Edit button
        edit_btn = PrimaryButton(text="Edit")
        edit_btn.setFixedWidth(theme.config.components.button_action_width - 10)
        edit_btn.clicked.connect(lambda checked, k=key, v=value: self._on_edit_alias(k, v))
        item_layout.addWidget(edit_btn)

        # Delete button
        delete_btn = DeleteButton(command=lambda checked, k=key: self._on_delete_alias(k))
        item_layout.addWidget(delete_btn)

        self.aliases_list_layout.insertWidget(self.aliases_list_layout.count() - 1, item_widget)

    def _on_add_alias_clicked(self) -> None:
        """Handle add alias button clicked."""
        key = self.key_entry.text().strip()
        value = self.value_entry.text().strip()

        if not key:
            QMessageBox.warning(self, "Validation Error", "Please enter an activation phrase.")
            return

        if not value:
            QMessageBox.warning(self, "Validation Error", "Please enter a substitution text.")
            return

        if not self.controller:
            QMessageBox.critical(self, "Error", "Controller not initialized.")
            return

        if self.controller.add_alias(key, value):
            self.key_entry.clear()
            self.value_entry.clear()

    def _on_edit_alias(self, key: str, value: str) -> None:
        """Handle edit button clicked - show edit dialog."""
        if not self.controller:
            self._show_error("Controller not initialized.")
            return

        dialog = AliasEditDialog(key, value, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            if dialog.result_saved and dialog.new_value:
                self.controller.update_alias(key, dialog.new_value)
                self.logger.info(f"Edited alias: {key}")

    def _on_delete_alias(self, key: str) -> None:
        """Handle delete button clicked."""
        if self.controller:
            self.controller.delete_alias(key)

    def _show_error(self, message: str) -> None:
        """Show error message dialog."""
        QMessageBox.critical(self, "Error", message)
