import logging
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image
from PySide6.QtGui import QImage, QPixmap

from vocalance.app.ui.utils.qt_dpi_utils import get_device_pixel_ratio

logger = logging.getLogger(__name__)


def pil_image_to_qpixmap(pil_image: Image.Image) -> Optional[QPixmap]:
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


def transform_monochrome_icon(
    icon_path: str | Path,
    target_color: str,
    size: Optional[Tuple[int, int]] = None,
    force_all_pixels: bool = False,
    preserve_aspect_ratio: bool = True,
    device_pixel_ratio: Optional[float] = None,
) -> Optional[QPixmap]:
    """Transform a monochrome icon by recoloring it while preserving transparency.

    Ports the legacy CustomTkinter icon transformation logic to work with Qt/PySide6.
    If force_all_pixels is True, recolor all non-transparent pixels regardless of luminance.
    Supports high-DPI displays via device_pixel_ratio parameter.

    Args:
        icon_path: Path to the icon file.
        target_color: Hex color string (e.g., "#ff0000").
        size: Optional tuple (width, height) to resize the icon in physical pixels.
        force_all_pixels: If True, recolor all non-transparent pixels.
        preserve_aspect_ratio: If True, preserve aspect ratio when resizing.
        device_pixel_ratio: Device pixel ratio for high-DPI rendering.
                           If provided and set on returned QPixmap.

    Returns:
        QPixmap with recolored icon, or None if processing failed.
    """
    if device_pixel_ratio is None:
        device_pixel_ratio = get_device_pixel_ratio()

    try:
        icon_path = Path(icon_path)
        if not icon_path.exists():
            logger.warning(f"Icon file not found: {icon_path}")
            return None

        # Load image with PIL
        img = Image.open(icon_path).convert("RGBA")

        # Resize if requested
        if size:
            if preserve_aspect_ratio and len(size) == 2:
                orig_w, orig_h = img.size
                target_w, target_h = size

                scale = min(target_w / orig_w, target_h / orig_h)
                new_w = int(orig_w * scale)
                new_h = int(orig_h * scale)

                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            else:
                img = img.resize(size, Image.Resampling.LANCZOS)

        # Parse target color
        target_color = target_color.lstrip("#")
        if len(target_color) != 6:
            raise ValueError(f"Invalid hex color: {target_color}")

        r = int(target_color[0:2], 16)
        g = int(target_color[2:4], 16)
        b = int(target_color[4:6], 16)
        target_rgb = (r, g, b)

        # Create new colored image
        colored_img = Image.new("RGBA", img.size, (0, 0, 0, 0))

        pixels = img.load()
        colored_pixels = colored_img.load()

        # Recolor pixels
        for y in range(img.height):
            for x in range(img.width):
                r_orig, g_orig, b_orig, a_orig = pixels[x, y]

                # Skip fully transparent pixels
                if a_orig == 0:
                    continue

                if force_all_pixels:
                    # Recolor all non-transparent pixels
                    colored_pixels[x, y] = (target_rgb[0], target_rgb[1], target_rgb[2], a_orig)
                else:
                    # Apply luminance-based opacity for smoother gradients
                    luminance = 0.299 * r_orig + 0.587 * g_orig + 0.114 * b_orig
                    opacity_factor = (255 - luminance) / 255.0
                    new_alpha = int(opacity_factor * a_orig)
                    colored_pixels[x, y] = (target_rgb[0], target_rgb[1], target_rgb[2], new_alpha)

        # Convert to QPixmap
        pixmap = pil_image_to_qpixmap(colored_img)

        # Set device pixel ratio for proper high-DPI rendering
        if pixmap and device_pixel_ratio:
            pixmap.setDevicePixelRatio(device_pixel_ratio)

        return pixmap

    except Exception as e:
        logger.error(f"Failed to transform icon {icon_path}: {e}")
        return None


def load_sidebar_icon(
    icon_filename: str,
    icons_dir: Path,
    target_color: str,
    icon_size: int,
    high_dpi: bool = True,
) -> Optional[QPixmap]:
    """Load and transform a sidebar icon. Supports high-DPI.

    Args:
        icon_filename: Name of icon file.
        icons_dir: Directory containing icon files.
        target_color: Hex color for icon recoloring.
        icon_size: Target size for icon in logical pixels (will preserve aspect ratio).
        high_dpi: If True, scale to device pixel ratio for sharp rendering.

    Returns:
        QPixmap or None if loading failed. Already scaled to logical size with DPR set.
    """
    try:
        device_pixel_ratio = get_device_pixel_ratio() if high_dpi else 1.0
        icon_path = icons_dir / icon_filename

        # For high-DPI support: scale to physical pixels during transformation
        # This ensures the image is loaded at the correct resolution for crisp rendering
        if high_dpi:
            physical_size = (int(icon_size * device_pixel_ratio), int(icon_size * device_pixel_ratio))
        else:
            physical_size = (icon_size, icon_size)

        pixmap = transform_monochrome_icon(
            icon_path=icon_path,
            target_color=target_color,
            size=physical_size,
            force_all_pixels=True,  # Force all pixels to target color for consistent appearance
            preserve_aspect_ratio=True,
            device_pixel_ratio=device_pixel_ratio if high_dpi else None,
        )

        if pixmap and high_dpi:
            # Set device pixel ratio so Qt renders at correct DPI
            pixmap.setDevicePixelRatio(device_pixel_ratio)

        return pixmap
    except Exception as e:
        logger.error(f"Failed to load sidebar icon {icon_filename}: {e}")
        return None
