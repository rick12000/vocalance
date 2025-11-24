"""High-DPI aware image scaling and rendering utilities for PySide6.

Provides DPI-aware image loading, scaling, and pixmap creation to ensure
images render sharply on high-DPI displays (retina, 4K, etc).

Key Concepts:
- Device Pixel Ratio (DPR): The ratio of physical pixels to logical pixels.
  - Standard display: DPR = 1.0
  - Retina/2x display: DPR = 2.0
  - High-DPI Windows: DPR = 1.25, 1.5, 1.75, 2.0, etc.

- To render sharply on high-DPI displays, images must be scaled to:
  display_size * device_pixel_ratio

- QPixmap.setDevicePixelRatio() tells Qt the physical resolution of the image,
  allowing proper rendering at the display's DPI.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication

logger = logging.getLogger(__name__)


def get_device_pixel_ratio() -> float:
    """Get the current device's pixel ratio.

    Returns:
        Device pixel ratio (e.g., 1.0 for standard, 2.0 for Retina).
    """
    try:
        app = QApplication.instance()
        if app and app.primaryScreen():
            return app.primaryScreen().devicePixelRatio()
    except Exception as e:
        logger.warning(f"Could not determine device pixel ratio: {e}")

    return 1.0


def calculate_physical_size(logical_size: int, device_pixel_ratio: Optional[float] = None) -> int:
    """Calculate physical pixel size from logical size and DPI.

    Args:
        logical_size: The desired display size in logical pixels.
        device_pixel_ratio: Device pixel ratio. If None, uses system DPR.

    Returns:
        Physical size in pixels needed to render sharply.
    """
    if device_pixel_ratio is None:
        device_pixel_ratio = get_device_pixel_ratio()

    return int(logical_size * device_pixel_ratio)


def calculate_physical_size_tuple(
    logical_size: Tuple[int, int],
    device_pixel_ratio: Optional[float] = None,
) -> Tuple[int, int]:
    """Calculate physical pixel size tuple from logical size and DPI.

    Args:
        logical_size: Tuple of (width, height) in logical pixels.
        device_pixel_ratio: Device pixel ratio. If None, uses system DPR.

    Returns:
        Tuple of (physical_width, physical_height) in pixels.
    """
    if device_pixel_ratio is None:
        device_pixel_ratio = get_device_pixel_ratio()

    return (
        int(logical_size[0] * device_pixel_ratio),
        int(logical_size[1] * device_pixel_ratio),
    )


def load_pixmap_high_dpi(
    image_path: str | Path,
    logical_size: Optional[Tuple[int, int]] = None,
    max_logical_dimension: Optional[int] = None,
    device_pixel_ratio: Optional[float] = None,
    preserve_aspect_ratio: bool = True,
) -> Optional[QPixmap]:
    """Load and scale an image for high-DPI displays.

    Loads the image at 2x (or device_pixel_ratio) the desired display size
    and sets the device pixel ratio on the QPixmap for proper rendering.

    Args:
        image_path: Path to the image file.
        logical_size: Tuple of (width, height) in logical pixels for display.
                      If provided, image will be scaled to this size considering DPI.
        max_logical_dimension: Maximum dimension (width or height) in logical pixels.
                               Used to constrain image size while preserving aspect.
        device_pixel_ratio: Device pixel ratio (e.g., 2.0 for Retina).
                           If None, uses system DPR.
        preserve_aspect_ratio: If True, preserve aspect ratio when scaling.

    Returns:
        QPixmap with proper device pixel ratio set, or None if loading failed.
    """
    if device_pixel_ratio is None:
        device_pixel_ratio = get_device_pixel_ratio()

    try:
        image_path = Path(image_path)
        if not image_path.exists():
            logger.warning(f"Image file not found: {image_path}")
            return None

        # Load image with PIL
        pil_image = Image.open(image_path).convert("RGBA")

        # Calculate physical size needed for rendering
        if logical_size is not None:
            physical_size = calculate_physical_size_tuple(logical_size, device_pixel_ratio)
            target_width, target_height = physical_size

            if preserve_aspect_ratio:
                orig_w, orig_h = pil_image.size
                scale = min(target_width / orig_w, target_height / orig_h)
                new_w = int(orig_w * scale)
                new_h = int(orig_h * scale)
                pil_image = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            else:
                pil_image = pil_image.resize((target_width, target_height), Image.Resampling.LANCZOS)

        elif max_logical_dimension is not None:
            physical_max = calculate_physical_size(max_logical_dimension, device_pixel_ratio)
            orig_w, orig_h = pil_image.size

            if orig_w > physical_max or orig_h > physical_max:
                scale = min(physical_max / orig_w, physical_max / orig_h)
                new_w = int(orig_w * scale)
                new_h = int(orig_h * scale)
                pil_image = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Convert PIL image to QPixmap
        qpixmap = pil_to_qpixmap(pil_image)

        if qpixmap is None:
            return None

        # Set device pixel ratio on the pixmap
        # This tells Qt that the pixmap's physical resolution is device_pixel_ratio times
        # the logical size, so it renders at the correct DPI
        qpixmap.setDevicePixelRatio(device_pixel_ratio)

        logger.debug(
            f"Loaded high-DPI pixmap: {image_path} "
            f"(physical: {qpixmap.width()}x{qpixmap.height()}, "
            f"logical: {int(qpixmap.width() / device_pixel_ratio)}x{int(qpixmap.height() / device_pixel_ratio)}, "
            f"DPR: {device_pixel_ratio})"
        )

        return qpixmap

    except Exception as e:
        logger.error(f"Failed to load high-DPI pixmap {image_path}: {e}")
        return None


def pil_to_qpixmap(pil_image: Image.Image) -> Optional[QPixmap]:
    """Convert PIL Image to QPixmap.

    Args:
        pil_image: PIL Image object in RGBA mode.

    Returns:
        QPixmap or None if conversion failed.
    """
    try:
        # Ensure image is in RGBA mode
        if pil_image.mode != "RGBA":
            pil_image = pil_image.convert("RGBA")

        # Get image data
        data = pil_image.tobytes("raw", "RGBA")
        width, height = pil_image.size

        # Create QImage from raw data
        qimage = QImage(data, width, height, width * 4, QImage.Format.Format_RGBA8888)

        # Convert to QPixmap
        return QPixmap.fromImage(qimage)

    except Exception as e:
        logger.error(f"Failed to convert PIL image to QPixmap: {e}")
        return None
