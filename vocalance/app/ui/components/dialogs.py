"""Themed dialogs and message boxes for Qt-based UI.

Provides BaseDialog class and reusable dialog functions for user interactions.
All dialogs use consistent theming from the centralized QSS and component classes.
"""

from typing import Any, Dict, Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QHBoxLayout, QMessageBox, QVBoxLayout, QWidget

from vocalance.app.ui.components.buttons import DangerButton, PrimaryButton
from vocalance.app.ui.components.inputs import ExpandableTextArea, TextInput
from vocalance.app.ui.components.labels import BodyLabel, SectionTitle, SmallLabel
from vocalance.app.ui.qt_theme import theme


def _get_main_window(widget: Optional[QWidget]) -> Optional[QWidget]:
    """Traverse widget hierarchy to find the main QMainWindow.

    Args:
        widget: Starting widget to traverse from.

    Returns:
        The main window widget, or None if not found.
    """
    from PySide6.QtWidgets import QMainWindow

    current = widget
    while current is not None:
        if isinstance(current, QMainWindow):
            return current
        current = current.parentWidget()
    return None


def _center_on_parent(dialog: QDialog, parent: Optional[QWidget] = None) -> None:
    """Center dialog on parent window (main window if available) or screen.

    Traverses up the widget hierarchy to find the main window for proper centering.
    This ensures dialogs center on the actual application window, not intermediate
    widgets, and move with the main window.

    Args:
        dialog: Dialog to center.
        parent: Optional parent widget (traversed to find main window).
    """
    # Try to find the main window first
    main_window = _get_main_window(parent)
    center_on = main_window if main_window else parent

    if center_on:
        parent_rect = center_on.geometry()
        dialog_size = dialog.size()
        x = parent_rect.x() + (parent_rect.width() - dialog_size.width()) // 2
        y = parent_rect.y() + (parent_rect.height() - dialog_size.height()) // 2
        dialog.move(x, y)
    else:
        # Center on screen if no parent found
        screen = dialog.screen()
        if screen:
            screen_geometry = screen.availableGeometry()
            x = (screen_geometry.width() - dialog.width()) // 2
            y = (screen_geometry.height() - dialog.height()) // 2
            dialog.move(x, y)


class BaseDialog(QDialog):
    """Base dialog class with consistent styling and centering.

    Provides common dialog setup including:
    - Consistent window flags
    - Modal behavior
    - Automatic centering on parent
    - Themed background from QSS
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        title: str = "",
        min_width: int = None,
        min_height: int = None,
    ):
        super().__init__(parent)

        # Set window flags
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowTitleHint | Qt.WindowType.WindowCloseButtonHint)
        self.setModal(True)

        if title:
            self.setWindowTitle(title)

        # Set minimum dimensions
        if min_width is None:
            min_width = theme.config.components.dialog_width
        if min_height is None:
            min_height = theme.config.components.dialog_min_height

        self.setMinimumWidth(min_width)
        self.setMinimumHeight(min_height)

        # Setup main layout
        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(
            theme.config.container.box_padding,
            theme.config.container.box_padding,
            theme.config.container.box_padding,
            theme.config.container.box_padding,
        )
        self._main_layout.setSpacing(theme.config.container.box_spacing_between)

    def exec(self):
        """Override exec to center dialog before showing.

        This ensures the dialog is centered on the parent window before
        it becomes visible and enters its modal event loop.
        """
        # Adjust dialog size to content before centering
        self.adjustSize()
        # Center on parent/main window
        _center_on_parent(self, self.parent())
        # Call parent exec to show and run modal event loop
        return super().exec()


class CommandEditDialog(BaseDialog):
    """Dialog for editing command phrases.

    Provides interface to:
    - View command description
    - Edit command phrase
    - Delete custom commands
    """

    def __init__(self, command, parent: Optional[QWidget] = None):
        """Initialize command edit dialog.

        Args:
            command: AutomationCommand instance to edit.
            parent: Parent widget.
        """
        super().__init__(
            parent=parent,
            title=f"Edit Command: {command.command_key}",
            min_width=500,
        )

        self.command = command
        self.result_action = None
        self.new_phrase_value = None

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Build the dialog UI."""
        # Import Box here to avoid circular imports
        from PySide6.QtGui import QColor, QPalette

        from vocalance.app.ui.components.layouts import Box

        # Set dialog background to darkest
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(theme.config.shapes.darkest))
        self.setPalette(palette)
        self.setAutoFillBackground(True)

        # Description section
        desc_frame = Box()
        desc_title = SectionTitle("Description")
        desc_frame.add(desc_title)

        desc_text = self._get_command_description()
        desc_label = BodyLabel(desc_text)
        desc_label.setWordWrap(True)
        desc_frame.add(desc_label)

        self._main_layout.addWidget(desc_frame)

        # Edit section
        edit_frame = Box()

        # Format "Edit Command Phrase" as a normal label like in commands view
        edit_title = SmallLabel("Edit Command Phrase:", color=theme.config.text.medium)
        edit_frame.add(edit_title)

        self.entry = TextInput()
        self.entry.setText(self.command.command_key)
        self.entry.selectAll()
        edit_frame.add(self.entry)

        save_btn = PrimaryButton(text="Save Changes")
        save_btn.clicked.connect(self._on_save)
        edit_frame.add(save_btn)

        self._main_layout.addWidget(edit_frame)

        # Delete section
        delete_frame = Box()
        delete_title = SectionTitle("Delete Command")
        delete_frame.add(delete_title)

        if self.command.is_custom:
            delete_desc = BodyLabel("This is a custom command and can be safely deleted.")
            delete_desc.setWordWrap(True)
            delete_frame.add(delete_desc)

            delete_btn = DangerButton(text="Delete Command")
            delete_btn.clicked.connect(self._on_delete)
            delete_frame.add(delete_btn)
        else:
            delete_desc = BodyLabel("This is a built-in command and cannot be deleted.")
            delete_desc.setWordWrap(True)
            delete_frame.add(delete_desc)

        self._main_layout.addWidget(delete_frame)

        # Focus entry field
        self.entry.setFocus()

    def _get_command_description(self) -> str:
        """Get a detailed description of what the command does."""
        if self.command.long_description:
            return self.command.long_description

        # Fallback to generating description based on action type
        action_descriptions = {
            "hotkey": f"Triggers hotkey: {self.command.action_value or 'Not set'}",
            "key": f"Simulates pressing the key: {self.command.action_value or 'Not set'}",
            "key_sequence": f"Executes key sequence: {self.command.action_value or 'Not set'}",
            "click": f"Performs a mouse click action: {self.command.action_value or 'Left click'}",
            "scroll": f"Performs a scroll action: {self.command.action_value or 'Scroll'}",
            "type": f"Types the text: {self.command.action_value or 'No text set'}",
        }
        return action_descriptions.get(
            self.command.action_type,
            f"Custom action: {self.command.action_value or 'No action defined'}",
        )

    def _on_save(self) -> None:
        """Handle save button click."""
        new_phrase = self.entry.text().strip()
        if new_phrase and new_phrase != self.command.command_key:
            self.result_action = "save"
            self.new_phrase_value = new_phrase
            self.accept()
        else:
            self.reject()

    def _on_delete(self) -> None:
        """Handle delete button click."""
        if not self.command.is_custom:
            return

        self.result_action = "delete"
        self.new_phrase_value = None
        self.accept()

    def get_result(self) -> Tuple[Optional[str], Optional[str]]:
        """Get the dialog result.

        Returns:
            Tuple of (action, new_phrase) where action is 'save', 'delete', or None.
        """
        return self.result_action, self.new_phrase_value


class PromptEditDialog(BaseDialog):
    """Dialog for editing prompts.

    Provides interface to edit prompt name and instructions.
    """

    def __init__(self, prompt_data: Dict[str, Any], parent: Optional[QWidget] = None):
        """Initialize prompt edit dialog.

        Args:
            prompt_data: Dictionary containing prompt 'name' and 'text'.
            parent: Parent widget.
        """
        super().__init__(
            parent=parent,
            title=f"Edit: {prompt_data.get('name', 'Unnamed')}",
            min_width=500,
            min_height=400,
        )

        self.prompt_data = prompt_data
        self.result_saved = False
        self.new_name = None
        self.new_text = None

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Build the dialog UI."""
        from PySide6.QtGui import QColor, QPalette

        from vocalance.app.ui.components.layouts import BaseContainer

        # Set dialog background to darkest
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(theme.config.shapes.darkest))
        self.setPalette(palette)
        self.setAutoFillBackground(True)

        # Single container for all content
        main_container = BaseContainer(
            layout="vertical",
            bg_color=theme.config.shapes.dark,
            border_radius=theme.config.radius.rounded,
        )
        container_layout = main_container.layout()
        container_layout.setContentsMargins(
            theme.config.container.box_padding,
            theme.config.container.box_padding,
            theme.config.container.box_padding,
            theme.config.container.box_padding,
        )
        container_layout.setSpacing(theme.config.spacing.medium)

        # Prompt Title label and input form at top
        title_label = SmallLabel("Prompt Title:", color=theme.config.text.medium)
        container_layout.addWidget(title_label)

        self.title_entry = TextInput()
        self.title_entry.setText(self.prompt_data.get("name", ""))
        container_layout.addWidget(self.title_entry)

        # Prompt Instructions label (right after title input)
        instructions_label = SmallLabel("Prompt Instructions:", color=theme.config.text.medium)
        container_layout.addWidget(instructions_label)

        # Prompt instructions form - large and stretchable like in the dictation view
        self.prompt_textbox = ExpandableTextArea(placeholder="Enter prompt instructions...")
        self.prompt_textbox.setText(self.prompt_data.get("text", ""))
        container_layout.addWidget(self.prompt_textbox, 1)  # Add stretch to fill available space

        # Buttons at the bottom
        button_layout = QHBoxLayout()
        button_layout.setSpacing(theme.config.spacing.medium)

        save_btn = PrimaryButton(text="Save Changes")
        save_btn.clicked.connect(self._on_save)
        button_layout.addWidget(save_btn)

        cancel_btn = DangerButton(text="Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        container_layout.addLayout(button_layout)

        self._main_layout.addWidget(main_container)

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


# =============================================================================
# Utility Dialog Functions
# =============================================================================


def _create_dialog_base(
    message: str,
    parent: Optional[QWidget] = None,
    button_configs: Optional[list] = None,
) -> Optional[bool]:
    """Base function for creating themed dialogs with configurable buttons.

    Args:
        message: Dialog message.
        parent: Parent widget.
        button_configs: List of (button_text, button_type, callback) tuples.

    Returns:
        Result from button callback if applicable, None otherwise.
    """
    result = [None]

    dialog = BaseDialog(parent=parent)

    # Message label
    message_label = BodyLabel(message, align="center")
    message_label.setWordWrap(True)
    message_label.setMaximumWidth(theme.config.components.dialog_message_max_width)
    dialog._main_layout.addWidget(message_label)

    # Button container
    if button_configs:
        button_layout = QHBoxLayout()
        button_layout.setSpacing(theme.config.spacing.medium)

        for btn_text, btn_type, btn_callback in button_configs:
            if btn_type == "danger":
                btn = DangerButton(text=btn_text)
            else:
                btn = PrimaryButton(text=btn_text)

            def on_click(callback=btn_callback):
                if callable(callback):
                    result[0] = callback()
                else:
                    result[0] = callback
                dialog.accept()

            btn.clicked.connect(on_click)
            button_layout.addWidget(btn)

        dialog._main_layout.addLayout(button_layout)

    dialog.exec()
    return result[0]


def askokcancel(message: str, parent: Optional[QWidget] = None) -> bool:
    """Show a themed OK/Cancel dialog."""
    result = _create_dialog_base(
        message=message,
        parent=parent,
        button_configs=[
            ("OK", "primary", lambda: True),
            ("Cancel", "primary", lambda: False),
        ],
    )
    return result if result is not None else False


def askyesno(message: str, parent: Optional[QWidget] = None) -> bool:
    """Show a themed Yes/No dialog."""
    result = _create_dialog_base(
        message=message,
        parent=parent,
        button_configs=[
            ("Yes", "danger", lambda: True),
            ("No", "primary", lambda: False),
        ],
    )
    return result if result is not None else False


def showinfo(message: str, parent: Optional[QWidget] = None) -> None:
    """Show a themed info dialog."""
    _create_dialog_base(
        message=message,
        parent=parent,
        button_configs=[
            ("OK", "primary", lambda: None),
        ],
    )


def showerror(message: str, parent: Optional[QWidget] = None) -> None:
    """Show a themed error dialog."""
    _create_dialog_base(
        message=message,
        parent=parent,
        button_configs=[
            ("OK", "primary", lambda: None),
        ],
    )


def showwarning(message: str, parent: Optional[QWidget] = None) -> None:
    """Show a themed warning dialog."""
    _create_dialog_base(
        message=message,
        parent=parent,
        button_configs=[
            ("OK", "primary", lambda: None),
        ],
    )


def qt_showinfo(message: str, title: str = "Information", parent: Optional[QWidget] = None) -> None:
    """Show a native Qt info message box."""
    QMessageBox.information(parent, title, message)


def qt_showerror(message: str, title: str = "Error", parent: Optional[QWidget] = None) -> None:
    """Show a native Qt error message box."""
    QMessageBox.critical(parent, title, message)


def qt_showwarning(message: str, title: str = "Warning", parent: Optional[QWidget] = None) -> None:
    """Show a native Qt warning message box."""
    QMessageBox.warning(parent, title, message)


def qt_askyesno(message: str, title: str = "Question", parent: Optional[QWidget] = None) -> bool:
    """Show a native Qt yes/no dialog."""
    reply = QMessageBox.question(
        parent,
        title,
        message,
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    return reply == QMessageBox.StandardButton.Yes


def qt_askokcancel(message: str, title: str = "Confirm", parent: Optional[QWidget] = None) -> bool:
    """Show a native Qt ok/cancel dialog."""
    reply = QMessageBox.question(
        parent,
        title,
        message,
        QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
        QMessageBox.StandardButton.Cancel,
    )
    return reply == QMessageBox.StandardButton.Ok
