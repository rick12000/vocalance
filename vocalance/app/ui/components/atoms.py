"""Atomic UI components with consistent styling.

All components use theme tokens - NO MAGIC NUMBERS.
Styled via stylesheets only - never override programmatically.
"""

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QLabel, QLineEdit, QPushButton, QWidget

from vocalance.app.ui.qt_theme import theme


def _get_input_stylesheet() -> str:
    """Generate stylesheet for input components."""
    c = theme.config
    return f"""
    QLineEdit, QTextEdit, QPlainTextEdit {{
        background-color: {c.shapes.darkest};
        color: {c.text.light};
        border: 1px solid {c.shapes.light};
        border-radius: {c.radius.small}px;
        padding: {c.components.input_padding_vertical}px {c.components.input_padding_horizontal}px;
        selection-background-color: {c.shapes.accent};
        outline: none;
        min-height: {c.components.input_height}px;
    }}

    QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
        border: 1px solid {c.shapes.accent};
        color: {c.text.lightest};
    }}

    QLineEdit:disabled {{
        background-color: {c.shapes.dark};
        color: {c.shapes.light};
        border: 1px solid {c.shapes.medium};
    }}
    """


def _get_checkbox_stylesheet() -> str:
    """Generate stylesheet for checkbox components."""
    c = theme.config
    return f"""
    QCheckBox {{
        color: {c.text.light};
        spacing: {c.spacing.small}px;
    }}

    QCheckBox::indicator {{
        width: 18px;
        height: 18px;
        border-radius: {c.radius.small // 2}px;
        border: 1px solid {c.shapes.light};
        background-color: {c.shapes.darkest};
    }}

    QCheckBox::indicator:checked {{
        background-color: {c.shapes.accent};
        border-color: {c.shapes.accent};
    }}

    QCheckBox::indicator:hover {{
        border-color: {c.shapes.lightest};
    }}
    """


class Label(QLabel):
    """Standard label component with variant-based styling."""

    def __init__(
        self, text: str, parent: Optional[QWidget] = None, variant: str = "body", color: Optional[str] = None, align: str = "left"
    ):
        super().__init__(text, parent)

        # Alignment
        align_map = {
            "left": Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            "center": Qt.AlignmentFlag.AlignCenter,
            "right": Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        }
        self.setAlignment(align_map.get(align, Qt.AlignmentFlag.AlignLeft))

        # Font and color based on variant
        if variant == "title":
            self.setFont(theme.get_font("xxlarge", "bold"))
            default_color = "text.lightest"
        elif variant == "subtitle":
            self.setFont(theme.get_font("large", "semibold"))
            default_color = "text.light"
        elif variant == "body":
            self.setFont(theme.get_font("medium", "regular"))
            default_color = "text.lightest"
        elif variant == "small":
            self.setFont(theme.get_font("small", "regular"))
            default_color = "text.light"
        elif variant == "group_header":
            self.setFont(theme.get_font("medium", "semibold"))
            default_color = "text.medium"
        else:
            self.setFont(theme.get_font("medium", "regular"))
            default_color = "text.lightest"

        # Apply color
        color_key = color if color else default_color
        hex_color = theme.get_color(color_key)
        self.setStyleSheet(f"color: {hex_color}; background: transparent; border: none; padding: 0px; margin: 0px;")
        self.setProperty("variant", variant)


class Button(QPushButton):
    """Standard button component with pill-shaped design."""

    def __init__(self, text: str, parent: Optional[QWidget] = None, variant: str = "primary", icon=None, command=None):
        super().__init__(text, parent)
        self.setProperty("variant", variant)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFont(theme.get_font("medium", "semibold"))

        # Set fixed height to enable pill shape (height determines border-radius)
        self.setFixedHeight(theme.config.components.button_height)

        # Apply stylesheet AFTER setting height
        self._update_stylesheet()

        if icon:
            self.setIcon(icon)

        if command:
            self.clicked.connect(command)

    def _update_stylesheet(self) -> None:
        """Apply button-specific stylesheet with pill-shape calculation."""
        c = theme.config
        h = c.components.button_height
        # Pill shape: border-radius = height / 2 (creates semi-circles on sides)
        radius = h // 2

        stylesheet = f"""
        QPushButton {{
            background-color: {c.shapes.accent};
            color: {c.shapes.darkest};
            border: none;
            border-radius: {radius}px;
            padding: {c.components.button_padding_vertical}px {c.components.button_padding_horizontal}px;
            font-weight: bold;
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
            border-radius: {radius}px;
        }}

        QPushButton[variant="primary"]:hover {{
            background-color: {c.shapes.lightest};
        }}

        QPushButton[variant="danger"] {{
            background-color: {c.shapes.medium};
            color: {c.text.lightest};
            border: 1px solid {c.shapes.light};
            border-radius: {radius}px;
        }}

        QPushButton[variant="danger"]:hover {{
            background-color: {c.shapes.light};
            border-color: {c.shapes.lightest};
        }}

        QPushButton[variant="ghost"] {{
            background-color: transparent;
            color: {c.text.light};
            border-radius: {radius}px;
        }}

        QPushButton[variant="ghost"]:hover {{
            background-color: {c.shapes.medium};
            color: {c.text.lightest};
        }}
        """

        self.setStyleSheet(stylesheet)


class Input(QLineEdit):
    """Standard text input component."""

    def __init__(self, placeholder: str = "", parent: Optional[QWidget] = None, password: bool = False):
        super().__init__(parent)
        self.setPlaceholderText(placeholder)
        self.setStyleSheet(_get_input_stylesheet())
        self.setFont(theme.get_font("medium"))

        if password:
            self.setEchoMode(QLineEdit.EchoMode.Password)


class Checkbox(QCheckBox):
    """Standard checkbox component."""

    def __init__(self, text: str, parent: Optional[QWidget] = None, checked: bool = False, command=None):
        super().__init__(text, parent)
        self.setChecked(checked)
        self.setStyleSheet(_get_checkbox_stylesheet())
        self.setFont(theme.get_font("medium"))

        if command:
            self.stateChanged.connect(command)
