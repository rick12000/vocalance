from PIL import Image


def pad_to_square(im, fill_color=(255, 255, 255, 0)):  # Use white with full transparency
    width, height = im.size
    if width == height:
        return im
    size = max(width, height)
    new_im = Image.new("RGBA", (size, size), fill_color)
    new_im.paste(im, ((size - width) // 2, (size - height) // 2))
    return new_im


INPUT_FILE = "blue_icon_full_size.png"
OUTPUT_FILE = "icon.ico"
RESAMPLE_FILTER = Image.Resampling.LANCZOS
# Include all standard Windows icon sizes for maximum compatibility
# This ensures Windows always has the perfect size available at any DPI
ICON_SIZES = [256, 128, 96, 64, 48, 40, 32, 24, 20, 16]

# Function requires explicit fill color to avoid hidden defaults
FILL_COLOR = (255, 255, 255, 0)


def generate_icon(input_path, output_path, sizes, fill_color):
    base = Image.open(input_path).convert("RGBA")
    square = pad_to_square(base, fill_color)
    # Save multi-resolution icon using ICO sizes parameter
    square.save(output_path, sizes=[(s, s) for s in sizes])


# Generate multi-resolution icon using high-quality Lanczos filter
generate_icon(INPUT_FILE, OUTPUT_FILE, ICON_SIZES, FILL_COLOR)
