import logging
from typing import Optional

from PySide6.QtWidgets import QVBoxLayout, QWidget


class QtBaseView(QWidget):
    """Tab view base: zero-margin column layout and optional controller reference."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.controller = None

        self._install_root_layout()

    def _install_root_layout(self) -> None:
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

    def set_controller(self, controller) -> None:
        """Attach ``controller``; subclasses wire signals in overrides."""
        self.controller = controller
