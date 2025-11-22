from typing import Optional

from PySide6.QtWidgets import QFrame, QHBoxLayout, QVBoxLayout, QWidget

from vocalance.app.ui.qt_theme import theme

from .atoms import Label
from .containers import Box


class Tile(QFrame):
    """Tile component for instructions or info."""

    def __init__(self, title: str, content: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setProperty("variant", "card")  # Using card variant for background

        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            theme.config.spacing.medium, theme.config.spacing.medium, theme.config.spacing.medium, theme.config.spacing.medium
        )
        layout.setSpacing(theme.config.spacing.small)

        # Title
        title_label = Label(title, variant="subtitle", align="center")
        layout.addWidget(title_label)

        # Content
        content_label = Label(content, variant="small", align="center")
        content_label.setWordWrap(True)
        layout.addWidget(content_label)


class TwoColumnLayout(QWidget):
    """Two column layout with titles and content areas."""

    def __init__(self, left_title: str = "", right_title: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent)

        # Main layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.config.dims.box_spacing)

        # Left Box
        self.left_box = Box(layout="vertical")
        if left_title:
            self.left_box.add(Label(left_title, variant="title"))

        # Right Box
        self.right_box = Box(layout="vertical")
        if right_title:
            self.right_box.add(Label(right_title, variant="title"))

        # Content containers (Transparent widgets inside boxes)
        # Note: Don't set layout here - let views handle layout setup
        self.left_content = QWidget()
        self.left_box.add(self.left_content, stretch=1)

        self.right_content = QWidget()
        self.right_box.add(self.right_content, stretch=1)

        layout.addWidget(self.left_box, stretch=1)
        layout.addWidget(self.right_box, stretch=1)
