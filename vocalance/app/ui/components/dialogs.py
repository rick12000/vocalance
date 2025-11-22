"""Themed dialogs and message boxes for Qt-based UI.

Provides dialog functions for user interactions with consistent theming.
"""

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QMessageBox, QVBoxLayout, QWidget

from vocalance.app.ui.qt_theme import theme

from .atoms import Button


def _center_on_parent(dialog: QDialog, parent: Optional[QWidget] = None) -> None:
    """Center dialog on parent window or screen.

    Args:
        dialog: Dialog to center.
        parent: Optional parent widget.
    """
    if parent:
        parent_rect = parent.geometry()
        dialog_size = dialog.size()
        x = parent_rect.x() + (parent_rect.width() - dialog_size.width()) // 2
        y = parent_rect.y() + (parent_rect.height() - dialog_size.height()) // 2
        dialog.move(x, y)
    else:
        # Center on screen
        screen = dialog.screen()
        if screen:
            screen_geometry = screen.availableGeometry()
            x = (screen_geometry.width() - dialog.width()) // 2
            y = (screen_geometry.height() - dialog.height()) // 2
            dialog.move(x, y)


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

    dialog = QDialog(parent)
    dialog.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowTitleHint | Qt.WindowType.WindowCloseButtonHint)
    dialog.setModal(True)
    dialog.setMinimumWidth(theme.config.dims.dialog_width)
    dialog.setMinimumHeight(150)

    # Main layout
    main_layout = QVBoxLayout(dialog)
    main_layout.setContentsMargins(20, 20, 20, 20)
    main_layout.setSpacing(20)

    # Message label
    message_label = QLabel(message)
    message_label.setWordWrap(True)
    message_label.setMaximumWidth(350)
    font = theme.get_font(size="medium")
    message_label.setFont(font)
    message_label.setStyleSheet(f"color: {theme.config.text.light};")
    message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    main_layout.addWidget(message_label)

    # Button container
    if button_configs:
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        for btn_text, btn_type, btn_callback in button_configs:
            variant = "primary"
            if btn_type == "danger":
                variant = "danger"

            btn = Button(text=btn_text, variant=variant)

            def on_click(callback=btn_callback):
                # Only call callback if it's callable, otherwise use the value directly
                if callable(callback):
                    result[0] = callback()
                else:
                    result[0] = callback
                dialog.accept()

            btn.clicked.connect(on_click)
            button_layout.addWidget(btn)

        main_layout.addLayout(button_layout)

    # Center dialog
    _center_on_parent(dialog, parent)

    # Show dialog modally
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
