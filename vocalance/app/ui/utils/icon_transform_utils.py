"""
Icon Transformation Utilities

Pure image transformation functions with no dependencies on other UI utilities.
Single responsibility: Transform and recolor icon images.
"""

import logging

from PIL import Image

logger = logging.getLogger("IconTransform")


def transform_monochrome_icon(
    icon_path: str, target_color: str, size: tuple = None, force_all_pixels: bool = False, preserve_aspect_ratio: bool = True
) -> Image.Image | None:
    """
    Transform a monochrome icon by recoloring it while preserving transparency.
    If force_all_pixels is True, recolor all non-transparent pixels regardless of luminance.

    Args:
        icon_path: Path to the icon file
        target_color: Hex color string (e.g., "#ff0000")
        size: Optional tuple (width, height) to resize the icon
        preserve_aspect_ratio: If True, preserve aspect ratio when resizing

    Returns:
        PIL Image with recolored icon, or None if processing failed
    """
    try:
        img = Image.open(icon_path).convert("RGBA")

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

        target_color = target_color.lstrip("#")
        if len(target_color) != 6:
            raise ValueError(f"Invalid hex color: {target_color}")

        r = int(target_color[0:2], 16)
        g = int(target_color[2:4], 16)
        b = int(target_color[4:6], 16)
        target_rgb = (r, g, b)

        colored_img = Image.new("RGBA", img.size, (0, 0, 0, 0))

        pixels = img.load()
        colored_pixels = colored_img.load()

        for y in range(img.height):
            for x in range(img.width):
                r_orig, g_orig, b_orig, a_orig = pixels[x, y]

                if a_orig == 0:
                    continue

                if force_all_pixels:
                    colored_pixels[x, y] = (target_rgb[0], target_rgb[1], target_rgb[2], a_orig)
                else:
                    luminance = 0.299 * r_orig + 0.587 * g_orig + 0.114 * b_orig

                    opacity_factor = (255 - luminance) / 255.0
                    new_alpha = int(opacity_factor * a_orig)

                    colored_pixels[x, y] = (target_rgb[0], target_rgb[1], target_rgb[2], new_alpha)

        return colored_img

    except Exception as e:
        logger.error(f"Failed to transform icon {icon_path}: {e}")
        return None
