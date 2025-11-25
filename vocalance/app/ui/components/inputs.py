"""Input component subclasses with inheritance-based styling.

Each input class inherits from QLineEdit and applies its own styling.
Styles are applied programmatically using QPalette and text margins.
"""

from typing import Optional

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QLineEdit, QWidget

from vocalance.app.ui.qt_theme import theme


class TextInput(QLineEdit):
    """Standard text input field.

    Inherits from QLineEdit with consistent styling from the base QSS.
    Uses minimal stylesheet for border-radius which can't be set programmatically.
    """

    def __init__(
        self,
        placeholder: str = "",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)

        self.setPlaceholderText(placeholder)

        # Set font
        self.setFont(theme.get_font("small"))

        # Set minimum height
        self.setMinimumHeight(theme.config.components.input_height)

        # Set padding via text margins
        margins = self.textMargins()
        margins.setLeft(theme.config.components.input_padding_horizontal)
        margins.setRight(theme.config.components.input_padding_horizontal)
        margins.setTop(theme.config.components.input_padding_vertical)
        margins.setBottom(theme.config.components.input_padding_vertical)
        self.setTextMargins(margins)

        # Apply colors via palette
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Base, QColor(theme.config.shapes.darkest))
        palette.setColor(QPalette.ColorRole.Text, QColor(theme.config.text.light))
        palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(theme.config.text.medium))
        self.setPalette(palette)
        self.setAutoFillBackground(True)

        # Minimal stylesheet for border-radius only - uses class selector
        self.setStyleSheet(
            f"""
            TextInput {{
                background-color: transparent;
                border: 1px solid {theme.config.shapes.light};
                border-radius: {theme.config.radius.small}px;
                padding: 0px;
            }}
            TextInput:focus {{
                border-color: {theme.config.shapes.lightest};
            }}
        """
        )


class PasswordInput(TextInput):
    """Password input field with hidden text.

    Inherits from TextInput and sets echo mode to Password.
    """

    def __init__(
        self,
        placeholder: str = "",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(placeholder, parent)

        # Set echo mode to hide password
        self.setEchoMode(QLineEdit.EchoMode.Password)

        # Override stylesheet to use PasswordInput selector
        self.setStyleSheet(
            f"""
            PasswordInput {{
                background-color: transparent;
                border: 1px solid {theme.config.shapes.light};
                border-radius: {theme.config.radius.small}px;
                padding: 0px;
            }}
            PasswordInput:focus {{
                border-color: {theme.config.shapes.lightest};
            }}
        """
        )
