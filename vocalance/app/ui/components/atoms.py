from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QLabel, QLineEdit, QPushButton, QWidget

from vocalance.app.ui.qt_theme import theme


def _get_button_stylesheet() -> str:
    """Generate stylesheet for button components."""
    c = theme.config
    return f"""
    QPushButton {{
        background-color: {c.shapes.accent};
        color: {c.shapes.darkest};
        border-radius: {c.dims.button_height // 2}px;
        padding: 4px {c.spacing.medium}px;
        font-weight: bold;
        min-height: {c.dims.button_height}px;
        border: none;
        outline: none;
    }}

    QPushButton:hover {{
        background-color: {c.shapes.lightest};
    }}

    QPushButton:pressed {{
        background-color: {c.shapes.accent_minus};
    }}

    QPushButton:disabled {{
        background-color: {c.shapes.medium};
        color: {c.shapes.light};
    }}

    QPushButton[variant="primary"] {{
        background-color: {c.shapes.accent};
        color: {c.text.light_blue_accent};
    }}

    QPushButton[variant="danger"] {{
        background-color: {c.shapes.medium};
        color: {c.text.lightest};
        border: 1px solid {c.shapes.light};
    }}

    QPushButton[variant="danger"]:hover {{
        background-color: {c.shapes.light};
        border-color: {c.shapes.lightest};
    }}

    QPushButton[variant="ghost"] {{
        background-color: transparent;
        color: {c.text.light};
    }}

    QPushButton[variant="ghost"]:hover {{
        background-color: {c.shapes.medium};
        color: {c.text.lightest};
    }}
    """


def _get_input_stylesheet() -> str:
    """Generate stylesheet for input components."""
    c = theme.config
    return f"""
    QLineEdit, QTextEdit, QPlainTextEdit {{
        background-color: {c.shapes.darkest};
        color: {c.text.light};
        border: 1px solid {c.shapes.medium};
        border-radius: {c.radius.small}px;
        padding: {c.spacing.tiny}px {c.spacing.small}px;
        selection-background-color: {c.shapes.accent};
        outline: none;
    }}

    QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
        border: 1px solid {c.shapes.accent};
        color: {c.text.lightest};
    }}

    QLineEdit:disabled {{
        background-color: {c.shapes.dark};
        color: {c.shapes.light};
    }}
    """


class Label(QLabel):
    """Standard label component."""

    def __init__(
        self, text: str, parent: Optional[QWidget] = None, variant: str = "body", color: str = "text.lightest", align: str = "left"
    ):
        super().__init__(text, parent)

        # alignment map
        align_map = {
            "left": Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            "center": Qt.AlignmentFlag.AlignCenter,
            "right": Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        }
        self.setAlignment(align_map.get(align, Qt.AlignmentFlag.AlignLeft))

        # Set font based on variant
        if variant == "title":
            self.setFont(theme.get_font("xxlarge", "bold"))
            color = "text.lightest"
        elif variant == "subtitle":
            self.setFont(theme.get_font("large", "semibold"))
            color = "text.light"
        elif variant == "body":
            self.setFont(theme.get_font("medium", "regular"))
        elif variant == "small":
            self.setFont(theme.get_font("small", "regular"))

        # Apply color
        hex_color = theme.get_color(color)
        self.setStyleSheet(f"color: {hex_color}; border: none; outline: none;")
        self.setProperty("variant", variant)


class Button(QPushButton):
    """Standard button component."""

    def __init__(self, text: str, parent: Optional[QWidget] = None, variant: str = "primary", icon=None, command=None):
        super().__init__(text, parent)
        self.setProperty("variant", variant)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(_get_button_stylesheet())

        if icon:
            self.setIcon(icon)

        if command:
            self.clicked.connect(command)


class Input(QLineEdit):
    """Standard text input component."""

    def __init__(self, placeholder: str = "", parent: Optional[QWidget] = None, password: bool = False):
        super().__init__(parent)
        self.setPlaceholderText(placeholder)
        self.setStyleSheet(_get_input_stylesheet())
        if password:
            self.setEchoMode(QLineEdit.EchoMode.Password)


class Checkbox(QCheckBox):
    """Standard checkbox component."""

    def __init__(self, text: str, parent: Optional[QWidget] = None, checked: bool = False, command=None):
        super().__init__(text, parent)
        self.setChecked(checked)
        if command:
            self.stateChanged.connect(command)
