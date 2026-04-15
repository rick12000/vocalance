import logging
from typing import Optional

from PySide6.QtWidgets import QVBoxLayout, QWidget


class QtBaseView(QWidget):
    """Base class for tab views: zero-margin main layout and controller hook."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.controller = None

        self._setup_main_layout()

        self.logger.debug("%s base layout initialized", self.__class__.__name__)

    def _setup_main_layout(self) -> None:
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

    def set_controller(self, controller) -> None:
        """Set the controller; subclasses connect signals in overrides."""
        self.controller = controller
        self.logger.debug("Controller set for %s", self.__class__.__name__)
