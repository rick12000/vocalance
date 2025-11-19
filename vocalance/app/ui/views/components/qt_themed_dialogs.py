"""Qt-based themed dialogs and message boxes.

Provides themed dialog functions replacing CustomTkinter dialogs with Qt equivalents.
"""

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QMessageBox, QVBoxLayout, QWidget

from vocalance.app.ui.qt_theme import theme_manager
from vocalance.app.ui.views.components.qt_themed_components import DangerButton, PrimaryButton


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
    dialog.setMinimumWidth(theme_manager.dimensions.dialog_width)
    dialog.setMinimumHeight(theme_manager.dimensions.dialog_min_height)

    # Main layout
    main_layout = QVBoxLayout(dialog)
    main_layout.setContentsMargins(20, 20, 20, 20)
    main_layout.setSpacing(20)

    # Message label
    message_label = QLabel(message)
    message_label.setWordWrap(True)
    message_label.setMaximumWidth(350)
    font = theme_manager.get_font(size=theme_manager.font_sizes.medium)
    message_label.setFont(font)
    message_label.setStyleSheet(f"color: {theme_manager.text_colors.light};")
    message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    main_layout.addWidget(message_label)

    # Button container
    if button_configs:
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        for btn_text, btn_type, btn_callback in button_configs:
            if btn_type == "primary":
                btn = PrimaryButton(text=btn_text)
            elif btn_type == "danger":
                btn = DangerButton(text=btn_text)
            else:
                btn = PrimaryButton(text=btn_text)

            def on_click(callback=btn_callback):
                result[0] = callback()
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
    """Show a themed OK/Cancel dialog and return True if OK was clicked.

    Args:
        message: Dialog message.
        parent: Parent widget.

    Returns:
        True if OK clicked, False otherwise.
    """
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
    """Show a themed Yes/No dialog and return True if Yes was clicked.

    Args:
        message: Dialog message.
        parent: Parent widget.

    Returns:
        True if Yes clicked, False otherwise.
    """
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
    """Show a themed info dialog.

    Args:
        message: Dialog message.
        parent: Parent widget.
    """
    _create_dialog_base(
        message=message,
        parent=parent,
        button_configs=[
            ("OK", "primary", lambda: None),
        ],
    )


def showerror(message: str, parent: Optional[QWidget] = None) -> None:
    """Show a themed error dialog.

    Args:
        message: Error message.
        parent: Parent widget.
    """
    _create_dialog_base(
        message=message,
        parent=parent,
        button_configs=[
            ("OK", "primary", lambda: None),
        ],
    )


def showwarning(message: str, parent: Optional[QWidget] = None) -> None:
    """Show a themed warning dialog.

    Args:
        message: Warning message.
        parent: Parent widget.
    """
    _create_dialog_base(
        message=message,
        parent=parent,
        button_configs=[
            ("OK", "primary", lambda: None),
        ],
    )


# Using Qt's native message boxes as an alternative (simpler but less customized)


def qt_showinfo(message: str, title: str = "Information", parent: Optional[QWidget] = None) -> None:
    """Show Qt native info message box.

    Args:
        message: Dialog message.
        title: Dialog title.
        parent: Parent widget.
    """
    QMessageBox.information(parent, title, message)


def qt_showerror(message: str, title: str = "Error", parent: Optional[QWidget] = None) -> None:
    """Show Qt native error message box.

    Args:
        message: Error message.
        title: Dialog title.
        parent: Parent widget.
    """
    QMessageBox.critical(parent, title, message)


def qt_showwarning(message: str, title: str = "Warning", parent: Optional[QWidget] = None) -> None:
    """Show Qt native warning message box.

    Args:
        message: Warning message.
        title: Dialog title.
        parent: Parent widget.
    """
    QMessageBox.warning(parent, title, message)


def qt_askyesno(message: str, title: str = "Question", parent: Optional[QWidget] = None) -> bool:
    """Show Qt native yes/no question box.

    Args:
        message: Question message.
        title: Dialog title.
        parent: Parent widget.

    Returns:
        True if Yes clicked, False otherwise.
    """
    reply = QMessageBox.question(
        parent,
        title,
        message,
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    return reply == QMessageBox.StandardButton.Yes


def qt_askokcancel(message: str, title: str = "Confirm", parent: Optional[QWidget] = None) -> bool:
    """Show Qt native OK/Cancel question box.

    Args:
        message: Question message.
        title: Dialog title.
        parent: Parent widget.

    Returns:
        True if OK clicked, False otherwise.
    """
    reply = QMessageBox.question(
        parent,
        title,
        message,
        QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
        QMessageBox.StandardButton.Cancel,
    )
    return reply == QMessageBox.StandardButton.Ok
