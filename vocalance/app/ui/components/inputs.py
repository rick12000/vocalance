"""Input component subclasses with inheritance-based styling.

Each input class inherits from QLineEdit and applies its own styling.
Styles are applied programmatically using QPalette and text margins.
"""

from typing import Optional

from PySide6.QtGui import QColor, QPalette, QTextOption
from PySide6.QtWidgets import QLineEdit, QPlainTextEdit, QWidget

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
        self.setFont(theme.get_font("medium"))

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


class ExpandableTextArea(QPlainTextEdit):
    """Multi-line text area with variable height based on placeholder text.

    Expands downward to fit placeholder text, then adds scrolling when content exceeds.
    Uses QPlainTextEdit for built-in scrollbar support.
    """

    def __init__(
        self,
        placeholder: str = "",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)

        self.placeholder_text = placeholder
        self._min_height_lines = 3  # Default minimum height

        # Set placeholder text
        self.setPlaceholderText(placeholder)

        # Set font
        self.setFont(theme.get_font("medium"))

        # Disable word wrap for consistent height calculation
        self.setWordWrapMode(QTextOption.WrapMode.NoWrap)

        # Calculate minimum height based on placeholder text
        self._calculate_height_from_placeholder()

        # Set padding
        padding = theme.config.components.input_padding_horizontal
        self.setContentsMargins(padding, padding, padding, padding)

        # Apply colors via palette
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Base, QColor(theme.config.shapes.darkest))
        palette.setColor(QPalette.ColorRole.Text, QColor(theme.config.text.light))
        palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(theme.config.text.medium))
        self.setPalette(palette)
        self.setAutoFillBackground(True)

        # Set up stylesheet for border-radius and scrollbar
        self.setStyleSheet(
            f"""
            ExpandableTextArea {{
                background-color: transparent;
                border: 1px solid {theme.config.shapes.light};
                border-radius: {theme.config.radius.small}px;
                padding: 0px;
            }}
            ExpandableTextArea:focus {{
                border-color: {theme.config.shapes.lightest};
            }}
            ExpandableTextArea::-webkit-scrollbar {{
                width: 8px;
            }}
            ExpandableTextArea::-webkit-scrollbar-track {{
                background: transparent;
            }}
            ExpandableTextArea::-webkit-scrollbar-thumb {{
                background: {theme.config.shapes.light};
                border-radius: 4px;
            }}
            ExpandableTextArea::-webkit-scrollbar-thumb:hover {{
                background: {theme.config.shapes.lightest};
            }}
        """
        )

        # Connect text changed signal to adjust height if needed
        self.textChanged.connect(self._on_text_changed)

    def _calculate_height_from_placeholder(self) -> None:
        """Calculate and set minimum height based on placeholder text."""
        fm = self.fontMetrics()
        line_height = fm.lineSpacing()

        # Count lines in placeholder text
        lines = max(self.placeholder_text.count("\n") + 1, self._min_height_lines)

        # Calculate width for text wrapping
        # Account for padding and margins
        padding = theme.config.components.input_padding_horizontal * 2
        margins = self.contentsMargins()
        available_width = self.width() - padding - margins.left() - margins.right()

        if available_width > 0:
            # Calculate actual lines needed for placeholder
            placeholder_width = fm.horizontalAdvance(self.placeholder_text)
            if placeholder_width > available_width:
                # Estimate wrapped lines
                lines = max(lines, (placeholder_width // available_width) + 1)

        # Set minimum height with padding
        min_height = (line_height * lines) + (theme.config.components.input_padding_vertical * 2)
        self.setMinimumHeight(min_height)

    def _on_text_changed(self) -> None:
        """Adjust height and scrollbar visibility based on content."""
        # The scrollbar will automatically appear when content exceeds minimum height
        # No additional adjustment needed - QPlainTextEdit handles this natively

    def resizeEvent(self, event):
        """Recalculate height on resize."""
        super().resizeEvent(event)
        self._calculate_height_from_placeholder()

    def text(self) -> str:
        """Get the plain text content."""
        return self.toPlainText()

    def setText(self, text: str) -> None:
        """Set the plain text content."""
        self.setPlainText(text)

    def clear(self) -> None:
        """Clear the text content."""
        self.setPlainText("")
