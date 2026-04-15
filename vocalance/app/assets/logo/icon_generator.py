from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Tuple, Union

from PIL import Image

FillColor = Union[Tuple[int, int, int, int], Tuple[int, int, int]]


def pad_to_square(image: Image.Image, fill_color: FillColor = (255, 255, 255, 0)) -> Image.Image:
    """Pad an image with transparency (or a solid color) so width equals height.

    Args:
        image: Source image (any mode; combined with paste on RGBA canvas).
        fill_color: RGBA or RGB tuple used for the letterboxed regions.

    Returns:
        Square RGBA image at max(width, height) on each side.
    """
    width, height = image.size
    if width == height:
        return image
    size = max(width, height)
    new_image = Image.new("RGBA", (size, size), fill_color)
    new_image.paste(image, ((size - width) // 2, (size - height) // 2))
    return new_image


INPUT_FILE = "blue_icon_full_size.png"
OUTPUT_FILE = "icon.ico"
ICON_SIZES: Sequence[int] = [256, 128, 96, 64, 48, 40, 32, 24, 20, 16]

FILL_COLOR: Tuple[int, int, int, int] = (255, 255, 255, 0)


def generate_icon(
    input_path: str | Path,
    output_path: str | Path,
    sizes: Iterable[int],
    fill_color: FillColor,
) -> None:
    """Build a multi-resolution ``.ico`` from a source PNG.

    Args:
        input_path: Path to the source image file.
        output_path: Path for the written ICO file.
        sizes: Edge lengths in pixels for each embedded resolution.
        fill_color: Letterbox color passed to ``pad_to_square``.
    """
    base = Image.open(input_path).convert("RGBA")
    square = pad_to_square(base, fill_color)
    square.save(output_path, sizes=[(s, s) for s in sizes])


if __name__ == "__main__":
    generate_icon(INPUT_FILE, OUTPUT_FILE, ICON_SIZES, FILL_COLOR)
