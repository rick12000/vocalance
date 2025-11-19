"""Qt-based logo service for managing application logos.

Provides thread-safe logo loading and widget creation using Qt.
"""

import logging
import threading
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel

from vocalance.app.ui.qt_theme import theme_manager
from vocalance.app.ui.utils.qt_assets import QtAssetCache

logger = logging.getLogger(__name__)


class QtLogoService:
    """Thread-safe centralized service for loading and managing application logos.

    Thread Safety:
    - _cache_lock protects logo cache operations.
    - Safe to create logo widgets from any thread.
    """

    def __init__(self, asset_cache: QtAssetCache):
        """Initialize QtLogoService with an asset cache instance.

        Args:
            asset_cache: Qt asset cache instance.
        """
        self.asset_cache = asset_cache
        self._logo_cache = {}
        self._cache_lock = threading.RLock()

    def get_logo_pixmap(
        self,
        max_size: int,
        context: str = "default",
        logo_type: str = "full",
    ) -> Optional[QPixmap]:
        """Get a logo pixmap with specified maximum size. Thread-safe.

        Args:
            max_size: Maximum dimension (width or height) for the logo.
            context: Context for logging (e.g., "startup", "sidebar").
            logo_type: Type of logo to load ("full" or "icon").

        Returns:
            QPixmap if successful, None if fallback needed.
        """
        cache_key = f"{max_size}_{context}_{logo_type}"

        with self._cache_lock:
            if cache_key in self._logo_cache:
                return self._logo_cache[cache_key]

        try:
            logo_pixmap = self.asset_cache.load_logo_pixmap(
                size=None,
                max_dimension=max_size,
                logo_type=logo_type,
            )

            if logo_pixmap:
                with self._cache_lock:
                    self._logo_cache[cache_key] = logo_pixmap
                logger.debug(f"Logo loaded successfully for {context} (size: {logo_pixmap.width()}x{logo_pixmap.height()})")
                return logo_pixmap
            else:
                logger.debug(f"No logo image available for {context}")
                return None

        except Exception as e:
            logger.warning(f"Error loading logo for {context}: {e}")
            return None

    def create_logo_widget(
        self,
        parent: Optional[QLabel] = None,
        max_size: int = 100,
        context: str = "default",
        text_fallback: str = "Vocalance",
        logo_type: str = "full",
    ) -> QLabel:
        """Create a logo widget with automatic image/text fallback.

        Args:
            parent: Parent widget.
            max_size: Maximum logo size.
            context: Context for logging.
            text_fallback: Fallback text if image fails.
            logo_type: Type of logo to load ("full" or "icon").

        Returns:
            QLabel with logo (image or text).
        """
        logo_pixmap = self.get_logo_pixmap(max_size, context, logo_type)

        label = QLabel(parent)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        if logo_pixmap:
            label.setPixmap(logo_pixmap)
        else:
            # Use text fallback
            logger.info(f"Using text logo for {context}")
            label.setText(text_fallback)

            # Set large font for text fallback
            font = theme_manager.get_font(size=max_size // 3, bold=True)
            label.setFont(font)

            # Set color from theme
            color = theme_manager.shape_colors.medium if hasattr(theme_manager, "shape_colors") else "#515151"
            label.setStyleSheet(f"color: {color};")

        return label
