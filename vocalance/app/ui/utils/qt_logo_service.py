"""Qt-based logo service for managing application logos.

Provides thread-safe logo loading and widget creation using Qt.
Supports high-DPI displays with automatic scaling.
"""

import logging
import threading
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel

from vocalance.app.ui.qt_theme import theme
from vocalance.app.ui.utils.qt_assets import QtAssetCache
from vocalance.app.ui.utils.qt_dpi_utils import get_device_pixel_ratio

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
        """Get a logo pixmap with specified maximum size. Thread-safe. Supports high-DPI.

        Args:
            max_size: Maximum dimension (width or height) for the logo in logical pixels.
            context: Context for logging (e.g., "startup", "sidebar").
            logo_type: Type of logo to load ("full" or "icon").

        Returns:
            QPixmap if successful, None if fallback needed.
        """
        # Include device pixel ratio in cache key for high-DPI support
        device_pixel_ratio = get_device_pixel_ratio()
        cache_key = f"{max_size}_{context}_{logo_type}_{device_pixel_ratio}"

        with self._cache_lock:
            if cache_key in self._logo_cache:
                return self._logo_cache[cache_key]

        try:
            logo_pixmap = self.asset_cache.load_logo_pixmap(
                size=None,
                max_dimension=max_size,
                logo_type=logo_type,
                device_pixel_ratio=device_pixel_ratio,
            )

            if logo_pixmap:
                with self._cache_lock:
                    self._logo_cache[cache_key] = logo_pixmap
                logical_width = int(logo_pixmap.width() / device_pixel_ratio)
                logical_height = int(logo_pixmap.height() / device_pixel_ratio)
                logger.debug(
                    f"Logo loaded successfully for {context} (logical: {logical_width}x{logical_height}, physical: {logo_pixmap.width()}x{logo_pixmap.height()}, DPR: {device_pixel_ratio})"
                )
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
            font = theme.get_font(size=max_size // 3, weight="semibold")
            label.setFont(font)

            # Set color from theme
            color = theme.config.shapes.medium if hasattr(theme, "shape_colors") else "#515151"
            label.setStyleSheet(f"color: {color};")

        return label
