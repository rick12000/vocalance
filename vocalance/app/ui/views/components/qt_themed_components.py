"""Qt-based themed components for Vocalance UI.

Provides custom QWidget subclasses styled with the application theme.
These components replace CustomTkinter widgets with Qt equivalents.
"""

from typing import Optional

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from vocalance.app.ui.qt_theme import theme_manager


class ThemedButton(QPushButton):
    """Base themed button with consistent styling."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        text: str = "",
        size: Optional[int] = None,
        compact: bool = False,
    ):
        """Initialize themed button.

        Args:
            parent: Parent widget.
            text: Button text.
            size: Font size (defaults to medium).
            compact: Whether to use compact sizing.
        """
        super().__init__(text, parent)

        if size is None:
            size = theme_manager.font_sizes.medium

        font = theme_manager.get_font(size=size, bold=True)
        self.setFont(font)

        self.setMinimumHeight(theme_manager.dimensions.button_height)

        if compact:
            # Calculate width based on text
            fm = self.fontMetrics()
            text_width = fm.horizontalAdvance(text) if text else theme_manager.dimensions.button_height
            self.setFixedWidth(text_width + theme_manager.dimensions.button_text_padding * 2)


class PrimaryButton(ThemedButton):
    """Primary action button with accent styling."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        text: str = "",
        size: Optional[int] = None,
    ):
        """Initialize primary button.

        Args:
            parent: Parent widget.
            text: Button text.
            size: Font size.
        """
        super().__init__(parent, text, size)
        self.setProperty("buttonType", "primary")


class DangerButton(ThemedButton):
    """Danger/destructive action button with warning styling."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        text: str = "",
        size: Optional[int] = None,
    ):
        """Initialize danger button.

        Args:
            parent: Parent widget.
            text: Button text.
            size: Font size.
        """
        super().__init__(parent, text, size)
        self.setProperty("buttonType", "danger")


class SidebarButton(QPushButton):
    """Sidebar navigation button with icon and text."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        text: str = "",
        icon_pixmap: Optional[QPixmap] = None,
    ):
        """Initialize sidebar button.

        Args:
            parent: Parent widget.
            text: Button text.
            icon_pixmap: Optional icon pixmap.
        """
        super().__init__(text, parent)
        self.setProperty("buttonType", "sidebar")
        self._selected = False

        if icon_pixmap:
            icon = QIcon(icon_pixmap)
            self.setIcon(icon)
            self.setIconSize(QSize(24, 24))

        font = theme_manager.get_font(size=theme_manager.font_sizes.medium)
        self.setFont(font)

        self.setMinimumHeight(40)

    def set_selected(self, selected: bool) -> None:
        """Set button selected state.

        Args:
            selected: Whether button is selected.
        """
        self._selected = selected
        self.setProperty("selected", "true" if selected else "false")
        self.style().unpolish(self)
        self.style().polish(self)


class ThemedLabel(QLabel):
    """Themed label with pre-configured design attributes."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        text: str = "",
        size: Optional[int] = None,
        color: Optional[str] = None,
        bold: bool = False,
        word_wrap: bool = False,
    ):
        """Initialize themed label.

        Args:
            parent: Parent widget.
            text: Label text.
            size: Font size (defaults to medium).
            color: Text color (defaults to lightest).
            bold: Whether to use bold font.
            word_wrap: Whether to enable word wrapping.
        """
        super().__init__(text, parent)

        if size is None:
            size = theme_manager.font_sizes.medium

        font = theme_manager.get_font(size=size, bold=bold)
        self.setFont(font)

        if color:
            self.setStyleSheet(f"color: {color};")

        if word_wrap:
            self.setWordWrap(True)


class TitleLabel(ThemedLabel):
    """Large title label."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        text: str = "",
    ):
        """Initialize title label.

        Args:
            parent: Parent widget.
            text: Title text.
        """
        super().__init__(
            parent,
            text,
            size=theme_manager.font_sizes.xxlarge,
            bold=True,
        )
        self.setProperty("labelType", "title")


class SubtitleLabel(ThemedLabel):
    """Subtitle label with lighter color."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        text: str = "",
    ):
        """Initialize subtitle label.

        Args:
            parent: Parent widget.
            text: Subtitle text.
        """
        super().__init__(
            parent,
            text,
            size=theme_manager.font_sizes.medium,
            color=theme_manager.text_colors.light,
        )
        self.setProperty("labelType", "subtitle")


class ThemedEntry(QLineEdit):
    """Themed text entry field with pre-configured styling."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        placeholder_text: str = "",
    ):
        """Initialize themed entry.

        Args:
            parent: Parent widget.
            placeholder_text: Placeholder text when empty.
        """
        super().__init__(parent)

        if placeholder_text:
            self.setPlaceholderText(placeholder_text)

        font = theme_manager.get_font(size=theme_manager.font_sizes.medium)
        self.setFont(font)

        self.setMinimumHeight(theme_manager.dimensions.entry_height)


class ThemedTextEdit(QTextEdit):
    """Themed multi-line text editor."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
    ):
        """Initialize themed text edit.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)

        font = theme_manager.get_font(size=theme_manager.font_sizes.medium)
        self.setFont(font)


class ThemedPlainTextEdit(QPlainTextEdit):
    """Themed plain text editor for code/monospace content."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
    ):
        """Initialize themed plain text edit.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)

        font = theme_manager.get_monospace_font(size=theme_manager.font_sizes.medium)
        self.setFont(font)


class ThemedComboBox(QComboBox):
    """Themed dropdown/combobox."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
    ):
        """Initialize themed combobox.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)

        font = theme_manager.get_font(size=theme_manager.font_sizes.medium)
        self.setFont(font)

        self.setMinimumHeight(theme_manager.dimensions.entry_height)


class ThemedFrame(QFrame):
    """Base themed frame with surface styling."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        frame_type: str = "default",
    ):
        """Initialize themed frame.

        Args:
            parent: Parent widget.
            frame_type: Frame type for styling (default, box, tile, header, sidebar, transparent).
        """
        super().__init__(parent)

        if frame_type != "default":
            self.setProperty("frameType", frame_type)

        self.setFrameShape(QFrame.Shape.NoFrame)


class TransparentFrame(QFrame):
    """Transparent frame for grouping without visual boundaries."""

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize transparent frame.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)
        self.setProperty("frameType", "transparent")
        self.setFrameShape(QFrame.Shape.NoFrame)


class BoxFrame(ThemedFrame):
    """Box frame with border and background."""

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize box frame.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent, frame_type="box")


class TileFrame(ThemedFrame):
    """Tile frame for instruction/info content."""

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize tile frame.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent, frame_type="tile")


class ThemedScrollArea(QScrollArea):
    """Scrollable area with themed scrollbars."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
    ):
        """Initialize themed scroll area.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)

        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)


class ScrollableFrame(QWidget):
    """Scrollable frame container (replacement for CTkScrollableFrame)."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
    ):
        """Initialize scrollable frame.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)

        # Create layout
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)

        # Create scroll area
        self._scroll_area = ThemedScrollArea(self)
        self._layout.addWidget(self._scroll_area)

        # Create content widget
        self._content_widget = TransparentFrame()
        self._content_layout = QVBoxLayout(self._content_widget)
        self._content_layout.setContentsMargins(5, 5, 5, 5)
        self._content_layout.setSpacing(5)
        self._content_layout.addStretch()

        self._scroll_area.setWidget(self._content_widget)

    def add_widget(self, widget: QWidget) -> None:
        """Add widget to scrollable content.

        Args:
            widget: Widget to add.
        """
        # Insert before the stretch
        count = self._content_layout.count()
        self._content_layout.insertWidget(count - 1, widget)

    def get_content_layout(self) -> QVBoxLayout:
        """Get the content layout for adding widgets.

        Returns:
            Content layout object.
        """
        return self._content_layout


class ThemedProgressBar(QProgressBar):
    """Themed progress bar."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
    ):
        """Initialize themed progress bar.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)

        self.setMinimum(0)
        self.setMaximum(100)
        self.setValue(0)
        self.setTextVisible(True)
        self.setMinimumHeight(theme_manager.dimensions.progress_bar_height)


class ThemedCheckBox(QCheckBox):
    """Themed checkbox/toggle."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        text: str = "",
    ):
        """Initialize themed checkbox.

        Args:
            parent: Parent widget.
            text: Checkbox label text.
        """
        super().__init__(text, parent)

        font = theme_manager.get_font(size=theme_manager.font_sizes.medium)
        self.setFont(font)


class SidebarButtonManager:
    """Manages sidebar button selection state."""

    def __init__(self):
        """Initialize sidebar button manager."""
        self._buttons: list[SidebarButton] = []
        self._selected_button: Optional[SidebarButton] = None

    def add_button(self, button: SidebarButton) -> None:
        """Add button to manager.

        Args:
            button: Sidebar button to manage.
        """
        self._buttons.append(button)

        # Connect click to selection
        def on_click():
            self.select_button(button)

        button.clicked.connect(on_click)

    def select_button(self, button: SidebarButton) -> None:
        """Select a button (deselecting others).

        Args:
            button: Button to select.
        """
        # Deselect previous
        if self._selected_button:
            self._selected_button.set_selected(False)

        # Select new
        button.set_selected(True)
        self._selected_button = button


class SpinnerLabel(QLabel):
    """Animated spinner label for loading indicators."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
    ):
        """Initialize spinner label.

        Args:
            parent: Parent widget.
        """
        super().__init__("", parent)

        font = theme_manager.get_monospace_font(size=theme_manager.font_sizes.small)
        self.setFont(font)

        self.setFixedWidth(15)
        self._frame = 0
        self._frames = ["|", "/", "-", "\\"]

    def update_frame(self) -> None:
        """Update spinner to next frame."""
        self.setText(self._frames[self._frame])
        self._frame = (self._frame + 1) % len(self._frames)
