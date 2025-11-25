"""Label component subclasses with inheritance-based styling.

Each label class inherits from QLabel and applies its own styling.
Styles only override what differs from the base QSS QLabel definition.
"""

from typing import Literal, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QLabel, QWidget

from vocalance.app.ui.qt_theme import theme


class TitleLabel(QLabel):
    """Large bold title label. xxlarge size, bold weight, lightest color."""

    def __init__(
        self,
        text: str = "",
        parent: Optional[QWidget] = None,
        align: Literal["left", "center", "right"] = "left",
        color: Optional[str] = None,
    ):
        super().__init__(text, parent)

        # Apply font
        self.setFont(theme.get_font(size="xxlarge", weight="bold"))

        # Apply color via palette
        final_color = color if color else theme.config.text.lightest
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.WindowText, QColor(final_color))
        self.setPalette(palette)

        # Set alignment
        self._apply_alignment(align)

        # Transparent background
        self.setAutoFillBackground(False)

    def _apply_alignment(self, align: Literal["left", "center", "right"]) -> None:
        align_map = {
            "left": Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            "center": Qt.AlignmentFlag.AlignCenter,
            "right": Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        }
        self.setAlignment(align_map.get(align, Qt.AlignmentFlag.AlignLeft))


class SubtitleLabel(QLabel):
    """Subtitle label. large size, semibold weight, light color."""

    def __init__(
        self,
        text: str = "",
        parent: Optional[QWidget] = None,
        align: Literal["left", "center", "right"] = "left",
        color: Optional[str] = None,
    ):
        super().__init__(text, parent)

        # Apply font
        self.setFont(theme.get_font(size="large", weight="semibold"))

        # Apply color via palette
        final_color = color if color else theme.config.text.light
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.WindowText, QColor(final_color))
        self.setPalette(palette)

        # Set alignment
        self._apply_alignment(align)

        # Transparent background
        self.setAutoFillBackground(False)

    def _apply_alignment(self, align: Literal["left", "center", "right"]) -> None:
        align_map = {
            "left": Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            "center": Qt.AlignmentFlag.AlignCenter,
            "right": Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        }
        self.setAlignment(align_map.get(align, Qt.AlignmentFlag.AlignLeft))


class BodyLabel(QLabel):
    """Body text label. medium size, regular weight, light color."""

    def __init__(
        self,
        text: str = "",
        parent: Optional[QWidget] = None,
        align: Literal["left", "center", "right"] = "left",
        color: Optional[str] = None,
    ):
        super().__init__(text, parent)

        # Apply font
        self.setFont(theme.get_font(size="medium", weight="regular"))

        # Apply color via palette
        final_color = color if color else theme.config.text.light
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.WindowText, QColor(final_color))
        self.setPalette(palette)

        # Set alignment
        self._apply_alignment(align)

        # Transparent background
        self.setAutoFillBackground(False)

    def _apply_alignment(self, align: Literal["left", "center", "right"]) -> None:
        align_map = {
            "left": Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            "center": Qt.AlignmentFlag.AlignCenter,
            "right": Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        }
        self.setAlignment(align_map.get(align, Qt.AlignmentFlag.AlignLeft))


class SmallLabel(QLabel):
    """Small text label. small size, regular weight, light color."""

    def __init__(
        self,
        text: str = "",
        parent: Optional[QWidget] = None,
        align: Literal["left", "center", "right"] = "left",
        color: Optional[str] = None,
    ):
        super().__init__(text, parent)

        # Apply font
        self.setFont(theme.get_font(size="small", weight="regular"))

        # Apply color via palette
        final_color = color if color else theme.config.text.light
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.WindowText, QColor(final_color))
        self.setPalette(palette)

        # Set alignment
        self._apply_alignment(align)

        # Transparent background
        self.setAutoFillBackground(False)

    def _apply_alignment(self, align: Literal["left", "center", "right"]) -> None:
        align_map = {
            "left": Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            "center": Qt.AlignmentFlag.AlignCenter,
            "right": Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        }
        self.setAlignment(align_map.get(align, Qt.AlignmentFlag.AlignLeft))


class GroupHeaderLabel(QLabel):
    """Group header label. medium size, semibold weight, light color."""

    def __init__(
        self,
        text: str = "",
        parent: Optional[QWidget] = None,
        align: Literal["left", "center", "right"] = "left",
        color: Optional[str] = None,
    ):
        super().__init__(text, parent)

        # Apply font
        self.setFont(theme.get_font(size="medium", weight="semibold"))

        # Apply color via palette
        final_color = color if color else theme.config.text.light
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.WindowText, QColor(final_color))
        self.setPalette(palette)

        # Set alignment
        self._apply_alignment(align)

        # Transparent background
        self.setAutoFillBackground(False)

    def _apply_alignment(self, align: Literal["left", "center", "right"]) -> None:
        align_map = {
            "left": Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            "center": Qt.AlignmentFlag.AlignCenter,
            "right": Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        }
        self.setAlignment(align_map.get(align, Qt.AlignmentFlag.AlignLeft))


class BoxTitleLabel(QLabel):
    """Box title label. large size, semibold weight, lightest color."""

    def __init__(
        self,
        text: str = "",
        parent: Optional[QWidget] = None,
        align: Literal["left", "center", "right"] = "left",
        color: Optional[str] = None,
    ):
        super().__init__(text, parent)

        # Apply font
        self.setFont(theme.get_font(size="large", weight="semibold"))

        # Apply color via palette
        final_color = color if color else theme.config.text.lightest
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.WindowText, QColor(final_color))
        self.setPalette(palette)

        # Set alignment
        self._apply_alignment(align)

        # Transparent background
        self.setAutoFillBackground(False)

    def _apply_alignment(self, align: Literal["left", "center", "right"]) -> None:
        align_map = {
            "left": Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            "center": Qt.AlignmentFlag.AlignCenter,
            "right": Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        }
        self.setAlignment(align_map.get(align, Qt.AlignmentFlag.AlignLeft))


class LargeLabel(QLabel):
    """Large text label. large size, regular weight, light color."""

    def __init__(
        self,
        text: str = "",
        parent: Optional[QWidget] = None,
        align: Literal["left", "center", "right"] = "left",
        color: Optional[str] = None,
    ):
        super().__init__(text, parent)

        # Apply font
        self.setFont(theme.get_font(size="large", weight="regular"))

        # Apply color via palette
        final_color = color if color else theme.config.text.light
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.WindowText, QColor(final_color))
        self.setPalette(palette)

        # Set alignment
        self._apply_alignment(align)

        # Transparent background
        self.setAutoFillBackground(False)

    def _apply_alignment(self, align: Literal["left", "center", "right"]) -> None:
        align_map = {
            "left": Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            "center": Qt.AlignmentFlag.AlignCenter,
            "right": Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        }
        self.setAlignment(align_map.get(align, Qt.AlignmentFlag.AlignLeft))
