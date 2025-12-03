import logging
from typing import Optional

from PySide6.QtWidgets import QVBoxLayout, QWidget


class QtBaseView(QWidget):
    """Base class for all tab views to ensure consistent layout and behavior.

    All tab views should inherit from this class to maintain:
    - Consistent main layout setup (zero margins/spacing)
    - Standardized controller connection
    - Proper logging setup
    """

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize base view with consistent layout setup."""
        super().__init__(parent)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.controller = None

        # Setup consistent main layout - all tab views should have zero margins/spacing
        # to integrate properly with the main window's stacked widget
        self._setup_main_layout()

        self.logger.debug(f"{self.__class__.__name__} base layout initialized")

    def _setup_main_layout(self) -> None:
        """Setup the main layout with consistent margins/spacing.

        ALL tab views must use zero margins and spacing to ensure
        consistent appearance when displayed in the main window's
        stacked widget. The TwoColumnLayout handles its own spacing.
        """
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

    def set_controller(self, controller) -> None:
        """Set the controller and establish connections.

        Subclasses should override this to connect controller signals
        and perform any controller-specific initialization.
        """
        self.controller = controller
        self.logger.debug(f"Controller set for {self.__class__.__name__}")
