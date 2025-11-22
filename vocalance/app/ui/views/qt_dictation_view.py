"""Qt-based dictation view - FULLY INTEGRATED WITH PROMPTS MANAGEMENT.

Displays dictation prompts management with add/edit/delete/select capabilities.
"""

import logging
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QRadioButton, QScrollArea, QVBoxLayout, QWidget

from vocalance.app.ui.components.atoms import Button
from vocalance.app.ui.components.complex import TwoColumnLayout
from vocalance.app.ui.qt_theme import theme


class PromptEditDialog(QDialog):
    """Dialog for editing prompts."""

    def __init__(self, prompt_data: Dict[str, Any], parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.prompt_data = prompt_data
        self.result_saved = False
        self.new_name = None
        self.new_text = None

        self.setWindowTitle(f"Edit: {prompt_data.get('name', 'Unnamed')}")
        self.setModal(True)
        self.setMinimumSize(500, 400)

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Build the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Title input
        layout.addWidget(QLabel("Prompt Title:"))
        self.title_entry = QLineEdit()
        self.title_entry.setText(self.prompt_data.get("name", ""))
        layout.addWidget(self.title_entry)

        # Prompt instructions
        layout.addWidget(QLabel("Prompt Instructions:"))
        self.prompt_textbox = QLineEdit()
        self.prompt_textbox.setText(self.prompt_data.get("text", ""))
        layout.addWidget(self.prompt_textbox)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        save_btn = Button(text="Save Changes", variant="primary")
        save_btn.clicked.connect(self._on_save)
        button_layout.addWidget(save_btn)

        cancel_btn = Button(text="Cancel", variant="danger")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        layout.addLayout(button_layout)

    def _on_save(self) -> None:
        """Handle save button click."""
        new_name = self.title_entry.text().strip()
        new_text = self.prompt_textbox.text().strip()

        if not new_name:
            QMessageBox.warning(self, "Validation Error", "Please enter a title for the prompt.")
            return

        if not new_text:
            QMessageBox.warning(self, "Validation Error", "Please enter instructions for the prompt.")
            return

        self.result_saved = True
        self.new_name = new_name
        self.new_text = new_text
        self.accept()


class QtDictationView(QWidget):
    """Qt-based dictation view - FULLY INTEGRATED WITH PROMPTS MANAGEMENT.

    Features:
    - Add custom prompt form (title + instructions text area)
    - Prompts list with radio buttons to select current
    - Edit button for each prompt (opens edit dialog)
    - Delete button for each prompt (disabled for default)
    - Real-time updates from controller
    """

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize dictation view."""
        super().__init__(parent)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.controller = None
        self.prompts_list: List[Dict[str, Any]] = []
        self.current_prompt_id = None
        self.prompt_radio_buttons = {}

        self._setup_ui()
        self.logger.debug("QtDictationView initialized")

    def set_controller(self, controller) -> None:
        """Set the controller and connect signals."""
        self.controller = controller

        # Connect controller signals
        self.controller.prompts_loaded.connect(self._on_prompts_loaded)
        self.controller.current_prompt_updated.connect(self._on_current_prompt_updated)
        self.controller.operation_error.connect(self._on_error)
        self.controller.status_updated.connect(self._on_status_updated)

        # Load initial prompts
        self.logger.info("Loading prompts from controller")
        self.controller.refresh_prompts()

    def _setup_ui(self) -> None:
        """Build UI with TwoColumnTabLayout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create two-column layout with titles
        self.layout = TwoColumnLayout("Add Custom Prompt", "Manage Prompts", self)
        main_layout.addWidget(self.layout)

        # Setup panels
        self._setup_add_prompt_form()
        self._setup_manage_prompts_panel()

    def _setup_add_prompt_form(self) -> None:
        """Setup add prompt form in left content area."""
        container = self.layout.left_content

        # Create layout if it doesn't exist
        container_layout = container.layout()
        if container_layout is None:
            container_layout = QVBoxLayout(container)

        container_layout.setContentsMargins(
            theme.config.dims.inner_content_padx,
            0,
            theme.config.dims.inner_content_padx,
            0,
        )
        container_layout.setSpacing(theme.config.spacing.medium)

        # Prompt title input
        prompt_title_label = QLabel("Prompt Title:")
        prompt_title_label.setStyleSheet("border: none; background: transparent;")
        container_layout.addWidget(prompt_title_label)
        self.title_entry = QLineEdit()
        self.title_entry.setPlaceholderText("e.g. Email Formatting")
        container_layout.addWidget(self.title_entry)

        # Prompt instructions
        prompt_instructions_label = QLabel("Prompt Instructions:")
        prompt_instructions_label.setStyleSheet("border: none; background: transparent;")
        container_layout.addWidget(prompt_instructions_label)
        self.prompt_textbox = QLineEdit()
        self.prompt_textbox.setPlaceholderText(
            "e.g. Format as an email. Start with 'Dear [Recipient Name],' and end with 'Best, Jim.' "
            "Adopt a professional tone and style."
        )
        container_layout.addWidget(self.prompt_textbox)

        # Add button
        self.add_prompt_btn = Button(text="Add Prompt", variant="primary")
        self.add_prompt_btn.clicked.connect(self._on_add_prompt_clicked)
        container_layout.addWidget(self.add_prompt_btn)

        container_layout.addStretch()

    def _setup_manage_prompts_panel(self) -> None:
        """Setup manage prompts panel in right content area."""
        container = self.layout.right_content

        # Create layout if it doesn't exist
        container_layout = container.layout()
        if container_layout is None:
            container_layout = QVBoxLayout(container)

        container_layout.setContentsMargins(
            theme.config.dims.inner_content_padx,
            0,
            theme.config.dims.inner_content_padx,
            0,
        )
        container_layout.setSpacing(theme.config.spacing.small)

        # Prompts list widget
        self.prompts_list_widget = QWidget()
        self.prompts_list_widget.setStyleSheet("background: transparent;")
        self.prompts_list_layout = QVBoxLayout(self.prompts_list_widget)
        self.prompts_list_layout.setSpacing(theme.config.spacing.tiny)
        self.prompts_list_layout.setContentsMargins(0, 0, 0, 0)
        self.prompts_list_layout.addStretch()

        # Scroll area for prompts
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll_area.setStyleSheet("background: transparent; border: none;")
        scroll_area.setWidget(self.prompts_list_widget)
        container_layout.addWidget(scroll_area)

    def _on_prompts_loaded(self, prompts: List[Dict[str, Any]]) -> None:
        """Handle prompts loaded from controller."""
        try:
            self.prompts_list = prompts
            self.current_prompt_id = self.controller.get_current_prompt_id() if self.controller else None
            self._display_prompts(prompts)
            self.logger.info(f"Prompts loaded: {len(prompts)} total")
        except Exception as e:
            self.logger.error(f"Error loading prompts: {e}", exc_info=True)
            self._show_error(f"Error loading prompts: {e}")

    def _on_current_prompt_updated(self, prompt_id: str) -> None:
        """Handle current prompt updated event."""
        try:
            self.current_prompt_id = prompt_id
            # Update radio button selection
            if prompt_id in self.prompt_radio_buttons:
                self.prompt_radio_buttons[prompt_id].setChecked(True)
            self.logger.info(f"Current prompt updated: {prompt_id}")
        except Exception as e:
            self.logger.error(f"Error handling current prompt updated: {e}", exc_info=True)

    def _on_status_updated(self, message: str, is_error: bool) -> None:
        """Handle status updates from controller."""
        if is_error:
            self._show_error(message)

    def _on_error(self, error_msg: str) -> None:
        """Handle error from controller."""
        self._show_error(error_msg)

    def _display_prompts(self, prompts: List[Dict[str, Any]]) -> None:
        """Display prompts with radio buttons, edit, and delete buttons."""
        # Clear existing items
        while self.prompts_list_layout.count():
            item = self.prompts_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.prompt_radio_buttons.clear()

        if not prompts:
            # Show empty message
            empty_label = QLabel("No prompts available.")
            empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty_font = theme.get_font(size=theme.config.fonts.medium)
            empty_label.setFont(empty_font)
            empty_label.setStyleSheet(f"color: {theme.config.text.medium};")
            self.prompts_list_layout.addWidget(empty_label)
        else:
            for prompt in prompts:
                self._create_prompt_item(prompt)

        self.prompts_list_layout.addStretch()

    def _create_prompt_item(self, prompt_data: Dict[str, Any]) -> None:
        """Create a prompt list item with radio, name, edit, and delete buttons."""
        item_widget = QWidget()
        item_widget.setProperty("itemType", "list_item")
        item_widget.setStyleSheet("background: transparent; border: none;")
        item_layout = QHBoxLayout(item_widget)
        item_layout.setContentsMargins(0, 0, 0, 0)
        item_layout.setSpacing(theme.config.spacing.small)

        # Radio button
        radio = QRadioButton()
        prompt_id = prompt_data.get("id", "")
        is_current = prompt_data.get("is_current", False) or (prompt_id == self.current_prompt_id)
        radio.setChecked(is_current)
        radio.toggled.connect(lambda checked, pid=prompt_id: self._on_radio_selected(pid, checked))
        self.prompt_radio_buttons[prompt_id] = radio
        item_layout.addWidget(radio)

        # Prompt name label
        name_label = QLabel(prompt_data.get("name", "Unnamed"))
        name_font = theme.get_font(size=theme.config.fonts.medium)
        name_label.setFont(name_font)
        name_label.setStyleSheet(f"color: {theme.config.text.medium}; background: transparent; border: none;")
        item_layout.addWidget(name_label, 1)

        # Edit button (pill-shaped)
        edit_btn = Button(text="Edit", variant="primary")
        edit_btn.setFixedWidth(70)
        edit_btn.clicked.connect(lambda checked, p=prompt_data: self._on_edit_prompt(p))
        item_layout.addWidget(edit_btn)

        # Delete button (pill-shaped)
        delete_btn = Button(text="Delete", variant="danger")
        delete_btn.setFixedWidth(80)
        is_default = prompt_data.get("is_default", False)
        delete_btn.setEnabled(not is_default)
        if not is_default:
            delete_btn.clicked.connect(lambda checked, pid=prompt_data.get("id"): self._on_delete_prompt(pid))
        item_layout.addWidget(delete_btn)

        self.prompts_list_layout.insertWidget(self.prompts_list_layout.count() - 1, item_widget)

    def _on_radio_selected(self, prompt_id: str, checked: bool) -> None:
        """Handle radio button selection."""
        if checked and self.controller:
            self.controller.select_prompt(prompt_id)

    def _on_add_prompt_clicked(self) -> None:
        """Handle add prompt button clicked."""
        title = self.title_entry.text().strip()
        prompt_text = self.prompt_textbox.text().strip()

        if not title:
            QMessageBox.warning(self, "Validation Error", "Please enter a title for the prompt.")
            return

        if not prompt_text:
            QMessageBox.warning(self, "Validation Error", "Please enter prompt instructions.")
            return

        if not self.controller:
            QMessageBox.critical(self, "Error", "Controller not initialized.")
            return

        if self.controller.add_prompt(title, prompt_text):
            # Clear form
            self.title_entry.clear()
            self.prompt_textbox.clear()

    def _on_edit_prompt(self, prompt_data: Dict[str, Any]) -> None:
        """Handle edit button clicked - show edit dialog."""
        # Check if this is the default prompt
        if prompt_data.get("is_default", False):
            QMessageBox.information(self, "Cannot Edit Default Prompt", "The default prompt cannot be edited.")
            return

        if not self.controller:
            self._show_error("Controller not initialized.")
            return

        dialog = PromptEditDialog(prompt_data, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            if dialog.result_saved:
                self.controller.edit_prompt(prompt_data["id"], dialog.new_name, dialog.new_text)
                self.logger.info(f"Edited prompt: {prompt_data['id']}")

    def _on_delete_prompt(self, prompt_id: str) -> None:
        """Handle delete button clicked."""
        # Check if this is the default prompt
        if self.controller and self.controller.is_default_prompt(prompt_id):
            QMessageBox.information(self, "Cannot Delete Default Prompt", "The default prompt cannot be deleted.")
            return

        if self.controller:
            self.controller.delete_prompt(prompt_id)

    def _show_error(self, message: str) -> None:
        """Show error message dialog."""
        QMessageBox.critical(self, "Error", message)
