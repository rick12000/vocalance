import logging
import threading
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel, QWidget

from vocalance.app.ui.qt_theme import theme
from vocalance.app.ui.utils.qt_assets import QtAssetCache
from vocalance.app.ui.utils.qt_dpi_utils import get_device_pixel_ratio

logger = logging.getLogger(__name__)


class QtLogoService:
    """Caches logo pixmaps and builds ``QLabel`` branding widgets."""

    def __init__(self, asset_cache: QtAssetCache) -> None:
        self.asset_cache = asset_cache
        self._pixmap_by_cache_key: dict[str, QPixmap] = {}
        self._pixmap_lock = threading.RLock()

    def get_logo_pixmap(
        self,
        max_size: int,
        context: str = "default",
        logo_type: str = "full",
    ) -> Optional[QPixmap]:
        """Return a cached logo pixmap scaled to ``max_size`` (logical pixels)."""
        device_pixel_ratio = get_device_pixel_ratio()
        cache_key = f"{max_size}_{context}_{logo_type}_{device_pixel_ratio}"

        with self._pixmap_lock:
            if cache_key in self._pixmap_by_cache_key:
                return self._pixmap_by_cache_key[cache_key]

        try:
            logo_pixmap = self.asset_cache.load_logo_pixmap(
                size=None,
                max_dimension=max_size,
                logo_type=logo_type,
                device_pixel_ratio=device_pixel_ratio,
            )

            if logo_pixmap:
                with self._pixmap_lock:
                    self._pixmap_by_cache_key[cache_key] = logo_pixmap
                return logo_pixmap
            logger.debug("No logo pixmap for context=%s", context)
            return None

        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning("Logo load failed context=%s: %s", context, exc)
            return None

    def create_logo_widget(
        self,
        parent: Optional[QWidget] = None,
        max_size: int = 100,
        context: str = "default",
        text_fallback: str = "Vocalance",
        logo_type: str = "full",
    ) -> QLabel:
        """Build a centered ``QLabel`` with pixmap or styled ``text_fallback``."""
        logo_pixmap = self.get_logo_pixmap(max_size, context, logo_type)

        label = QLabel(parent)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        if logo_pixmap:
            label.setPixmap(logo_pixmap)
        else:
            logger.debug("Text fallback logo context=%s", context)
            label.setText(text_fallback)

            font = theme.get_font(size=max_size // 3, weight="semibold")
            label.setFont(font)

            color = theme.config.shapes.medium
            label.setStyleSheet(f"color: {color};")

        return label
