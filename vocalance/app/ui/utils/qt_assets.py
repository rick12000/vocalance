"""Qt-based asset cache for images, icons, and resources.

Provides thread-safe asset loading using Qt's resource system.
Replaces CustomTkinter-based AssetCache with QPixmap and QIcon.
"""

import logging
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple

from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QImage, QPixmap

from vocalance.app.config.app_config import AssetPathsConfig
from vocalance.app.ui.qt_theme import theme_manager

logger = logging.getLogger("QtAssets")


def transform_monochrome_icon(
    image_path: str,
    color: str,
    size: tuple[int, int],
    force_all_pixels: bool = False,
    preserve_aspect_ratio: bool = False,
) -> Optional[Image.Image]:
    """Transform a monochrome icon by applying color and resizing.

    Args:
        image_path: Path to the image file
        color: Hex color string (e.g., "#ff0000")
        size: Target (width, height) tuple
        force_all_pixels: Whether to force color on all pixels (including transparent)
        preserve_aspect_ratio: Whether to preserve aspect ratio when resizing

    Returns:
        PIL Image object or None if transformation fails
    """
    try:
        # Open and convert to RGBA
        img = Image.open(image_path).convert("RGBA")
        pixels = img.load()

        # Parse color
        if color.startswith("#"):
            color = color[1:]
        r = int(color[0:2], 16)
        g = int(color[2:4], 16)
        b = int(color[4:6], 16)

        # Transform pixels
        for y in range(img.size[1]):
            for x in range(img.size[0]):
                pixel = pixels[x, y]
                if force_all_pixels:
                    # Apply color to all pixels regardless of original color
                    pixels[x, y] = (r, g, b, pixel[3])
                else:
                    # Only color non-transparent black/white pixels
                    if pixel[3] > 0:  # Not transparent
                        # Check if pixel is dark (monochrome)
                        brightness = (pixel[0] + pixel[1] + pixel[2]) / 3
                        if brightness < 128:  # Dark pixel, apply color
                            pixels[x, y] = (r, g, b, pixel[3])
                        else:  # Light pixel, keep as white or transparent
                            pixels[x, y] = (255, 255, 255, pixel[3])

        # Resize
        if preserve_aspect_ratio:
            img.thumbnail(size, Image.Resampling.LANCZOS)
        else:
            img = img.resize(size, Image.Resampling.LANCZOS)

        return img

    except Exception as e:
        logger.error(f"Failed to transform monochrome icon {image_path}: {e}")
        return None


class QtAssetCache:
    """Thread-safe asset cache for Qt resources.

    Caches QPixmap and QIcon objects for efficient reuse.
    Thread-safe for loading from multiple threads.
    """

    def __init__(self, asset_paths_config: AssetPathsConfig):
        """Initialize Qt asset cache.

        Args:
            asset_paths_config: Asset paths configuration.
        """
        self._pixmap_cache: Dict[str, QPixmap] = {}
        self._icon_cache: Dict[str, QIcon] = {}
        self._assets_path: Optional[Path] = None
        self._asset_paths_config = asset_paths_config
        self._cache_lock = threading.RLock()
        self._setup_assets_path()

    def _setup_assets_path(self) -> None:
        """Set up the path to UI assets."""
        self._assets_path = Path(self._asset_paths_config.logo_dir)

    def get_assets_path(self) -> Optional[Path]:
        """Get the assets directory path.

        Returns:
            Path to assets directory or None.
        """
        return self._assets_path

    def load_pixmap(
        self,
        filename: str,
        size: Optional[Tuple[int, int]] = None,
    ) -> Optional[QPixmap]:
        """Load and cache a pixmap. Thread-safe.

        Args:
            filename: Image filename in assets directory.
            size: Optional (width, height) tuple for resizing.

        Returns:
            QPixmap object or None if loading fails.
        """
        cache_key = f"{filename}_{size}"

        with self._cache_lock:
            if cache_key in self._pixmap_cache:
                return self._pixmap_cache[cache_key]

        if not self._assets_path:
            logger.error("Assets path not available")
            return None

        try:
            image_path = self._assets_path / filename
            if not image_path.exists():
                logger.warning(f"Image file not found: {image_path}")
                return None

            pixmap = QPixmap(str(image_path))
            if pixmap.isNull():
                logger.warning(f"Failed to load pixmap from {image_path}")
                return None

            if size:
                pixmap = pixmap.scaled(
                    size[0],
                    size[1],
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )

            with self._cache_lock:
                self._pixmap_cache[cache_key] = pixmap

            logger.debug(f"Loaded and cached pixmap: {filename}")
            return pixmap

        except Exception as e:
            logger.error(f"Failed to load pixmap {filename}: {e}")
            return None

    def load_icon(
        self,
        filename: str,
        size: Optional[Tuple[int, int]] = None,
    ) -> Optional[QIcon]:
        """Load and cache an icon. Thread-safe.

        Args:
            filename: Icon filename in assets directory.
            size: Optional (width, height) tuple for resizing.

        Returns:
            QIcon object or None if loading fails.
        """
        cache_key = f"{filename}_{size}"

        with self._cache_lock:
            if cache_key in self._icon_cache:
                return self._icon_cache[cache_key]

        pixmap = self.load_pixmap(filename, size)
        if not pixmap:
            return None

        icon = QIcon(pixmap)

        with self._cache_lock:
            self._icon_cache[cache_key] = icon

        logger.debug(f"Loaded and cached icon: {filename}")
        return icon

    def get_icon_path(self) -> Optional[Path]:
        """Get the path to the application icon.

        Returns:
            Path to icon file or None.
        """
        if not self._assets_path:
            return None
        icon_path = self._assets_path / "icon.ico"
        return icon_path if icon_path.exists() else None

    def get_icons_dir(self) -> Path:
        """Get the path to the icons directory.

        Returns:
            Path to icons directory.
        """
        return Path(self._asset_paths_config.icons_dir)

    def load_pixmap_monochrome_colored(
        self,
        filename: str,
        color: str,
        size: Optional[Tuple[int, int]] = None,
    ) -> Optional[QPixmap]:
        """Load a monochrome image, recolor it, and return as QPixmap. Thread-safe.

        Args:
            filename: Image filename in assets directory.
            color: Hex color string (e.g., "#ff0000").
            size: Optional (width, height) tuple for resizing.

        Returns:
            QPixmap object or None if loading fails.
        """
        cache_key = f"{filename}_{color}_{size}"

        with self._cache_lock:
            if cache_key in self._pixmap_cache:
                return self._pixmap_cache[cache_key]

        if not self._assets_path:
            logger.error("Assets path not available")
            return None

        try:
            image_path = self._assets_path / filename
            if not image_path.exists():
                logger.warning(f"Image file not found: {image_path}")
                return None

            # Use PIL transformation then convert to QPixmap
            pil_image = transform_monochrome_icon(str(image_path), color, size if size is not None else (100, 100))

            if pil_image is None:
                logger.error(f"Failed to recolor image {filename}")
                return None

            # Convert PIL image to QPixmap
            pixmap = self._pil_to_pixmap(pil_image)

            with self._cache_lock:
                self._pixmap_cache[cache_key] = pixmap

            logger.info(f"Loaded and cached colored pixmap: {filename} with color {color}")
            return pixmap

        except Exception as e:
            logger.error(f"Failed to load and recolor image {filename}: {e}")
            return None

    def load_logo_pixmap(
        self,
        size: Optional[Tuple[int, int]] = None,
        max_dimension: int = 200,
        logo_type: str = "full",
    ) -> Optional[QPixmap]:
        """Load logo as QPixmap, optionally recolored.

        Args:
            size: Optional exact size tuple (width, height).
            max_dimension: Maximum dimension when size is None.
            logo_type: Type of logo ("full" or "icon").

        Returns:
            QPixmap object or None if loading fails.
        """
        # Get logo properties from theme
        logo_props = theme_manager.logo_properties if hasattr(theme_manager, "logo_properties") else None

        if not self._assets_path:
            logger.error("Assets path not available for logo.")
            return None

        # Determine filename and whether to apply monochrome
        if logo_type == "icon":
            filename = "grey_icon_full_size.png"  # Default icon
            apply_monochrome = False
        else:
            filename = "grey_icon_full_size.png"  # Default full logo
            apply_monochrome = False

        # Check if logo properties exist in theme
        if logo_props:
            if logo_type == "icon":
                filename = logo_props.icon_logo_filename
                apply_monochrome = logo_props.icon_logo_apply_monochrome
            else:
                filename = logo_props.full_logo_filename
                apply_monochrome = logo_props.full_logo_apply_monochrome

        logo_path = self._assets_path / filename
        if not logo_path.exists():
            logger.error(f"Logo file not found: {logo_path}")
            return None

        try:
            if apply_monochrome:
                # Use PIL transform then convert to QPixmap
                color = logo_props.color if logo_props else theme_manager.shape_colors.medium

                if size is not None:
                    pil_image = transform_monochrome_icon(
                        str(logo_path),
                        color,
                        size,
                        force_all_pixels=True,
                        preserve_aspect_ratio=True,
                    )
                else:
                    constraint_size = (max_dimension, max_dimension)
                    pil_image = transform_monochrome_icon(
                        str(logo_path),
                        color,
                        constraint_size,
                        force_all_pixels=True,
                        preserve_aspect_ratio=True,
                    )

                if pil_image is None:
                    logger.error(f"Failed to transform logo {filename}")
                    return None

                return self._pil_to_pixmap(pil_image)
            else:
                # Load directly as pixmap and resize
                pixmap = QPixmap(str(logo_path))

                if size is not None:
                    pixmap = pixmap.scaled(
                        size[0],
                        size[1],
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                elif max_dimension > 0:
                    width = pixmap.width()
                    height = pixmap.height()

                    if width > max_dimension or height > max_dimension:
                        pixmap = pixmap.scaled(
                            max_dimension,
                            max_dimension,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation,
                        )

                return pixmap

        except Exception as e:
            logger.error(f"Failed to load logo {filename}: {e}")
            return None

    def _pil_to_pixmap(self, pil_image: Image.Image) -> QPixmap:
        """Convert PIL Image to QPixmap.

        Args:
            pil_image: PIL Image object.

        Returns:
            QPixmap object.
        """
        # Convert PIL image to bytes
        if pil_image.mode != "RGBA":
            pil_image = pil_image.convert("RGBA")

        data = pil_image.tobytes("raw", "RGBA")
        qimage = QImage(
            data,
            pil_image.width,
            pil_image.height,
            QImage.Format.Format_RGBA8888,
        )

        return QPixmap.fromImage(qimage)
