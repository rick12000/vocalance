from typing import Optional

from PySide6.QtGui import QColor, QPalette, QResizeEvent, QTextOption
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
    ) -> None:
        super().__init__(parent)

        self.setPlaceholderText(placeholder)

        self.setFont(theme.get_font("medium"))

        self.setMinimumHeight(theme.config.components.input_height)

        margins = self.textMargins()
        margins.setLeft(theme.config.components.input_padding_horizontal)
        margins.setRight(theme.config.components.input_padding_horizontal)
        margins.setTop(theme.config.components.input_padding_vertical)
        margins.setBottom(theme.config.components.input_padding_vertical)
        self.setTextMargins(margins)

        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Base, QColor(theme.config.shapes.darkest))
        palette.setColor(QPalette.ColorRole.Text, QColor(theme.config.text.light))
        palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(theme.config.text.medium))
        self.setPalette(palette)
        self.setAutoFillBackground(True)

        self.setStyleSheet(
            f"""
            TextInput {{
                background-color: transparent;
                border: 1px solid {theme.config.shapes.light};
                border-radius: {theme.config.radius.small}px;
                padding: 0px;
            }}
            TextInput:focus {{
                border-color: {theme.config.blue.blue_2};
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
    ) -> None:
        super().__init__(placeholder, parent)

        self.setEchoMode(QLineEdit.EchoMode.Password)

        self.setStyleSheet(
            f"""
            PasswordInput {{
                background-color: transparent;
                border: 1px solid {theme.config.shapes.light};
                border-radius: {theme.config.radius.small}px;
                padding: 0px;
            }}
            PasswordInput:focus {{
                border-color: {theme.config.blue.blue_2};
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
    ) -> None:
        super().__init__(parent)

        self.placeholder_text = placeholder
        self._min_height_lines = 3

        self.setPlaceholderText(placeholder)

        self.setFont(theme.get_font("medium"))

        self.setWordWrapMode(QTextOption.WrapMode.WordWrap)

        self._calculate_height_from_placeholder()

        left_padding = theme.config.components.input_padding_horizontal * 2
        vertical_padding = theme.config.components.input_padding_vertical
        self.setContentsMargins(left_padding, vertical_padding, theme.config.components.input_padding_horizontal, vertical_padding)

        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Base, QColor(theme.config.shapes.darkest))
        palette.setColor(QPalette.ColorRole.Text, QColor(theme.config.text.light))
        palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(theme.config.text.medium))
        self.setPalette(palette)
        self.setAutoFillBackground(True)

        self.setStyleSheet(
            f"""
            ExpandableTextArea {{
                background-color: transparent;
                border: 1px solid {theme.config.shapes.light};
                border-radius: {theme.config.radius.small}px;
                padding: 0px;
            }}
            ExpandableTextArea:focus {{
                border-color: {theme.config.blue.blue_2};
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

    def _calculate_height_from_placeholder(self) -> None:
        """Calculate and set minimum height based on placeholder text."""
        fm = self.fontMetrics()
        line_height = fm.lineSpacing()

        lines = max(self.placeholder_text.count("\n") + 1, self._min_height_lines)

        padding = theme.config.components.input_padding_horizontal * 2
        margins = self.contentsMargins()
        available_width = self.width() - padding - margins.left() - margins.right()

        if available_width > 0:
            placeholder_width = fm.horizontalAdvance(self.placeholder_text)
            if placeholder_width > available_width:
                lines = max(lines, (placeholder_width // available_width) + 1)

        min_height = (line_height * lines) + (theme.config.components.input_padding_vertical * 2)
        self.setMinimumHeight(min_height)

    def resizeEvent(self, resize_event: QResizeEvent) -> None:
        """Recalculate height on resize."""
        super().resizeEvent(resize_event)
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
