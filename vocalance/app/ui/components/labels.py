from typing import Literal, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QLabel, QWidget

from vocalance.app.ui.qt_theme import theme
from vocalance.app.ui.utils.qt_gradient_text import GradientDirection, GradientTextMixin

_LABEL_ALIGN: dict[Literal["left", "center", "right"], Qt.AlignmentFlag] = {
    "left": Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
    "center": Qt.AlignmentFlag.AlignCenter,
    "right": Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
}


def _label_alignment(align: Literal["left", "center", "right"]) -> Qt.AlignmentFlag:
    return _LABEL_ALIGN.get(align, Qt.AlignmentFlag.AlignLeft)


class TitleLabel(GradientTextMixin, QLabel):
    """Large bold title label with gradient text. xxlarge size, bold weight.

    By default, renders text with a gradient from theme.config.text.title_gradient.
    The gradient can be disabled by passing use_gradient=False, which will revert
    to solid color rendering using theme.config.text.lightest.
    """

    def __init__(
        self,
        text: str = "",
        parent: Optional[QWidget] = None,
        align: Literal["left", "center", "right"] = "left",
        color: Optional[str] = None,
        use_gradient: bool = True,
    ):
        super().__init__(text, parent)

        self.setFont(theme.get_font(size="xxlarge", weight="bold", display=True))

        self.setAlignment(_label_alignment(align))

        self.setAutoFillBackground(False)

        if use_gradient and color is None:
            self.enable_gradient(colors=theme.config.text.title_gradient, direction=GradientDirection.DIAGONAL)
        else:
            final_color = color if color else theme.config.text.lightest
            palette = self.palette()
            palette.setColor(QPalette.ColorRole.WindowText, QColor(final_color))
            self.setPalette(palette)


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

        self.setFont(theme.get_font(size="large", weight="semibold", display=True))

        final_color = color if color else theme.config.text.light
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.WindowText, QColor(final_color))
        self.setPalette(palette)

        self.setAlignment(_label_alignment(align))

        self.setAutoFillBackground(False)


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

        self.setFont(theme.get_font(size="medium", weight="regular"))

        final_color = color if color else theme.config.text.light
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.WindowText, QColor(final_color))
        self.setPalette(palette)

        self.setAlignment(_label_alignment(align))

        self.setAutoFillBackground(False)


class SmallLabel(QLabel):
    """Small text label. medium size, regular weight, light color.

    Note: Despite the name, uses medium font size (12pt) for consistency with UI design.
    For actual small text, use BodyLabel or create a new component.
    """

    def __init__(
        self,
        text: str = "",
        parent: Optional[QWidget] = None,
        align: Literal["left", "center", "right"] = "left",
        color: Optional[str] = None,
    ):
        super().__init__(text, parent)

        self.setFont(theme.get_font(size="medium", weight="regular"))

        final_color = color if color else theme.config.text.light
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.WindowText, QColor(final_color))
        self.setPalette(palette)

        self.setAlignment(_label_alignment(align))

        self.setAutoFillBackground(False)


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

        self.setFont(theme.get_font(size="medium", weight="semibold"))

        final_color = color if color else theme.config.text.light
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.WindowText, QColor(final_color))
        self.setPalette(palette)

        self.setAlignment(_label_alignment(align))

        self.setAutoFillBackground(False)


class BoxTitleLabel(GradientTextMixin, QLabel):
    """Box title label with gradient text. large size, semibold weight.

    By default, renders text with a gradient from theme.config.text.title_gradient.
    The gradient can be disabled by passing use_gradient=False, which will revert
    to solid color rendering using theme.config.text.lightest.
    """

    def __init__(
        self,
        text: str = "",
        parent: Optional[QWidget] = None,
        align: Literal["left", "center", "right"] = "left",
        color: Optional[str] = None,
        use_gradient: bool = True,
    ):
        # Initialize mixin first, then QLabel
        super().__init__(text, parent)

        self.setFont(theme.get_font(size="large", weight="semibold", display=True))

        self.setAlignment(_label_alignment(align))

        self.setAutoFillBackground(False)

        if use_gradient and color is None:
            self.enable_gradient(colors=theme.config.text.title_gradient, direction=GradientDirection.DIAGONAL)
        else:
            final_color = color if color else theme.config.text.lightest
            palette = self.palette()
            palette.setColor(QPalette.ColorRole.WindowText, QColor(final_color))
            self.setPalette(palette)


class SectionTitle(QLabel):
    """Section title label. large size, semibold weight, Alata display font, light color.

    Used for organizational section headers within scrollable content areas.
    Matches SubtitleLabel styling but is a distinct component for semantic clarity.
    """

    def __init__(
        self,
        text: str = "",
        parent: Optional[QWidget] = None,
        align: Literal["left", "center", "right"] = "left",
        color: Optional[str] = None,
    ):
        super().__init__(text, parent)

        self.setFont(theme.get_font(size="moderate", weight="semibold", display=True))

        final_color = color if color else theme.config.text.light
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.WindowText, QColor(final_color))
        self.setPalette(palette)

        self.setAlignment(_label_alignment(align))

        self.setAutoFillBackground(False)


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

        self.setFont(theme.get_font(size="large", weight="regular"))

        final_color = color if color else theme.config.text.light
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.WindowText, QColor(final_color))
        self.setPalette(palette)

        self.setAlignment(_label_alignment(align))

        self.setAutoFillBackground(False)
