from typing import Optional

from PySide6.QtWidgets import QFrame, QHBoxLayout, QLayout, QScrollArea, QVBoxLayout, QWidget

from vocalance.app.ui.qt_theme import theme


def _get_container_stylesheet() -> str:
    """Generate stylesheet for container components."""
    c = theme.config
    return f"""
    QFrame[variant="box"] {{
        background-color: {c.shapes.darkest};
        border: 1px solid {c.shapes.medium};
        border-radius: {c.radius.rounded}px;
    }}

    QFrame[variant="panel"] {{
        background-color: {c.shapes.dark};
        border: 1px solid {c.shapes.medium};
        border-radius: {c.radius.medium}px;
    }}

    QFrame[variant="card"] {{
        background-color: {c.shapes.medium};
        border-radius: {c.radius.medium}px;
    }}

    QFrame[frameType="box"] {{
        background-color: {c.shapes.darkest};
        border: 1px solid {c.shapes.medium};
        border-radius: {c.radius.rounded}px;
    }}

    QFrame[frameType="two_box"] {{
        background-color: {c.shapes.dark};
        border: 1px solid {c.shapes.medium};
        border-radius: {c.radius.rounded}px;
    }}

    QFrame[frameType="content_border"] {{
        background-color: {c.shapes.darkest};
        border: 3px solid {c.shapes.accent};
        border-radius: {c.radius.xlarge}px;
    }}

    QFrame[variant="transparent"], QFrame[frameType="sidebar"] {{
        background: transparent;
        border: none;
    }}

    QScrollBar:vertical {{
        background: {c.shapes.dark};
        width: 10px;
        margin: 0;
        border-radius: 5px;
        border: none;
    }}

    QScrollBar::handle:vertical {{
        background: {c.shapes.light};
        min-height: 20px;
        border-radius: 5px;
    }}

    QScrollBar::handle:vertical:hover {{
        background: {c.shapes.lightest};
    }}

    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}

    QListWidget {{
        background: transparent;
        border: none;
        outline: none;
    }}

    QListWidget::item {{
        padding: {c.spacing.small}px;
        border-radius: {c.radius.small}px;
    }}

    QListWidget::item:selected {{
        background: {c.shapes.medium};
    }}

    QListWidget::item:hover {{
        background: {c.shapes.light};
    }}
    """


class BaseContainer(QFrame):
    """Base container with layout support."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        layout: str = "vertical",
        margin: int = 0,
        spacing: int = 0,
        variant: str = "default",
    ):
        super().__init__(parent)
        self.setProperty("variant", variant)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet(_get_container_stylesheet())

        if layout == "vertical":
            self._layout = QVBoxLayout(self)
        elif layout == "horizontal":
            self._layout = QHBoxLayout(self)
        else:
            self._layout = QVBoxLayout(self)

        self._layout.setContentsMargins(margin, margin, margin, margin)
        self._layout.setSpacing(spacing)

    def add(self, widget: QWidget, stretch: int = 0):
        self._layout.addWidget(widget, stretch)

    def add_layout(self, layout: QLayout, stretch: int = 0):
        self._layout.addLayout(layout, stretch)

    def add_stretch(self, stretch: int = 1):
        self._layout.addStretch(stretch)


class Box(BaseContainer):
    """Box container with border and background."""

    def __init__(self, parent: Optional[QWidget] = None, layout: str = "vertical"):
        super().__init__(parent, layout, margin=theme.config.spacing.medium, spacing=theme.config.spacing.small, variant="box")


class Card(BaseContainer):
    """Card container for grouped content."""

    def __init__(self, parent: Optional[QWidget] = None, layout: str = "vertical"):
        super().__init__(parent, layout, margin=theme.config.spacing.medium, spacing=theme.config.spacing.small, variant="card")


class TransparentBox(BaseContainer):
    """Transparent container."""

    def __init__(self, parent: Optional[QWidget] = None, layout: str = "vertical", margin: int = 0, spacing: int = 0):
        super().__init__(parent, layout, margin=margin, spacing=spacing, variant="transparent")


class ScrollableContainer(QFrame):
    """Scrollable container."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setStyleSheet(
            """
            QScrollArea {
                background: transparent;
                border: none;
            }
        """
        )

        self.content_widget = QWidget()
        self.content_widget.setStyleSheet("background: transparent;")
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(theme.config.spacing.small)

        self.scroll_area.setWidget(self.content_widget)
        self.layout().addWidget(self.scroll_area)

    def add(self, widget: QWidget, stretch: int = 0):
        self.content_layout.addWidget(widget, stretch)

    def add_stretch(self, stretch: int = 1):
        self.content_layout.addStretch(stretch)
