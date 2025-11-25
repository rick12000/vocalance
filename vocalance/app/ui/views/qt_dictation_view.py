"""Qt-based dictation view.

Displays dictation prompts management with add/edit/delete/select capabilities.
Uses new component subclasses and dialogs from components module.
"""

import logging
from typing import Any, Dict, List, Optional

from PySide6.QtWidgets import QDialog, QHBoxLayout, QMessageBox, QRadioButton, QVBoxLayout, QWidget

from vocalance.app.ui.components.buttons import DangerButton, PrimaryButton
from vocalance.app.ui.components.dialogs import PromptEditDialog
from vocalance.app.ui.components.inputs import TextInput
from vocalance.app.ui.components.labels import BodyLabel, SmallLabel
from vocalance.app.ui.components.layouts import ScrollableContainer, TransparentWidget, TwoColumnLayout
from vocalance.app.ui.qt_theme import theme


class QtDictationView(QWidget):
    """Qt-based dictation view.

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
        """Build UI with TwoColumnLayout."""
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
        content = self.layout.left_content

        # Prompt title input
        prompt_title_label = BodyLabel("Prompt Title:")
        content.add(prompt_title_label)
        self.title_entry = TextInput(placeholder="e.g. Email Formatting")
        content.add(self.title_entry)

        # Prompt instructions
        prompt_instructions_label = BodyLabel("Prompt Instructions:")
        content.add(prompt_instructions_label)
        self.prompt_textbox = TextInput(
            placeholder="e.g. Format as an email. Start with 'Dear [Recipient Name],' and end with 'Best, Jim.' Adopt a professional tone and style."
        )
        content.add(self.prompt_textbox)

        # Add button
        self.add_prompt_btn = PrimaryButton(text="Add Prompt")
        self.add_prompt_btn.clicked.connect(self._on_add_prompt_clicked)
        content.add(self.add_prompt_btn)

        content.add_stretch()

    def _setup_manage_prompts_panel(self) -> None:
        """Setup manage prompts panel in right content area."""
        content = self.layout.right_content

        # Prompts list widget
        self.prompts_list_widget = TransparentWidget()
        self.prompts_list_layout = QVBoxLayout(self.prompts_list_widget)
        self.prompts_list_layout.setSpacing(theme.config.container.list_item_spacing)
        self.prompts_list_layout.setContentsMargins(0, 0, theme.config.spacing.small, 0)

        # Scroll area for prompts
        scroll_area = ScrollableContainer()
        scroll_area.content_layout.addWidget(self.prompts_list_widget)
        content.add(scroll_area, stretch=1)

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
            empty_label = BodyLabel("No prompts available.", align="center", color=theme.config.text.medium)
            self.prompts_list_layout.addWidget(empty_label)
        else:
            for prompt in prompts:
                self._create_prompt_item(prompt)

        self.prompts_list_layout.addStretch()

    def _create_prompt_item(self, prompt_data: Dict[str, Any]) -> None:
        """Create a prompt list item with radio, name, edit, and delete buttons."""
        # Create item widget
        item_widget = TransparentWidget()
        item_widget.setProperty("itemType", "list_item")
        item_layout = QHBoxLayout(item_widget)
        item_layout.setContentsMargins(
            0,
            theme.config.container.list_item_padding_vertical,
            0,
            theme.config.container.list_item_padding_vertical,
        )
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
        name_label = SmallLabel(prompt_data.get("name", "Unnamed"), color=theme.config.text.medium)
        item_layout.addWidget(name_label, 1)

        # Edit button
        edit_btn = PrimaryButton(text="Edit")
        edit_btn.setFixedWidth(theme.config.components.button_action_width - 10)
        edit_btn.clicked.connect(lambda checked, p=prompt_data: self._on_edit_prompt(p))
        item_layout.addWidget(edit_btn)

        # Delete button
        delete_btn = DangerButton(text="Delete")
        delete_btn.setFixedWidth(theme.config.components.button_action_width)
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
            self.title_entry.clear()
            self.prompt_textbox.clear()

    def _on_edit_prompt(self, prompt_data: Dict[str, Any]) -> None:
        """Handle edit button clicked - show edit dialog."""
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
        if self.controller and self.controller.is_default_prompt(prompt_id):
            QMessageBox.information(self, "Cannot Delete Default Prompt", "The default prompt cannot be deleted.")
            return

        if self.controller:
            self.controller.delete_prompt(prompt_id)

    def _show_error(self, message: str) -> None:
        """Show error message dialog."""
        QMessageBox.critical(self, "Error", message)
