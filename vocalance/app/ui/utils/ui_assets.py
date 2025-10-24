"""
UI Asset Management Utility

Handles loading, caching, and management of UI assets like images and icons.
Single responsibility: Asset loading and caching.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import customtkinter as ctk
from PIL import Image

from vocalance.app.config.app_config import AssetPathsConfig
from vocalance.app.ui import ui_theme
from vocalance.app.ui.utils.icon_transform_utils import transform_monochrome_icon

logger = logging.getLogger("UIAssets")


class AssetCache:
    """Simple asset cache for images and icons."""

    def __init__(self, asset_paths_config: AssetPathsConfig):
        self._image_cache: Dict[str, ctk.CTkImage] = {}
        self._assets_path: Optional[Path] = None
        self._asset_paths_config = asset_paths_config
        self._setup_assets_path()

    def _setup_assets_path(self) -> None:
        """Set up the path to UI assets."""
        self._assets_path = Path(self._asset_paths_config.logo_dir)

    def get_assets_path(self) -> Optional[Path]:
        """Get the assets directory path."""
        return self._assets_path

    def load_image(self, filename: str, size: Optional[Tuple[int, int]] = None) -> Optional[ctk.CTkImage]:
        """
        Load and cache an image for use in CustomTkinter components.

        Args:
            filename: Image filename in assets directory
            size: Optional (width, height) tuple for resizing

        Returns:
            CTkImage object or None if loading fails
        """
        cache_key = f"{filename}_{size}"

        if cache_key in self._image_cache:
            return self._image_cache[cache_key]

        if not self._assets_path:
            logger.error("Assets path not available")
            return None

        try:
            image_path = self._assets_path / filename
            if not image_path.exists():
                logger.warning(f"Image file not found: {image_path}")
                return None

            # Load with PIL
            pil_image = Image.open(image_path)

            # Resize if requested
            if size:
                pil_image = pil_image.resize(size, Image.Resampling.LANCZOS)

            # Create CTkImage
            ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=size or pil_image.size)

            # Cache the result
            self._image_cache[cache_key] = ctk_image
            logger.info(f"Loaded and cached image: {filename}")
            return ctk_image

        except Exception as e:
            logger.error(f"Failed to load image {filename}: {e}")
            return None

    def get_icon_path(self) -> Optional[Path]:
        """Get the path to the application icon."""
        if not self._assets_path:
            return None
        icon_path = self._assets_path / "icon.ico"
        return icon_path if icon_path.exists() else None

    def load_image_monochrome_colored(
        self, filename: str, color: str, size: Optional[Tuple[int, int]] = None
    ) -> Optional[ctk.CTkImage]:
        """
        Load and cache a monochrome image, recolor it using the provided color, and return a CTkImage.

        Args:
            filename: Image filename in assets directory
            color: Hex color string (e.g., "#ff0000")
            size: Optional (width, height) tuple for resizing

        Returns:
            CTkImage object or None if loading fails
        """
        cache_key = f"{filename}_{color}_{size}"
        if cache_key in self._image_cache:
            return self._image_cache[cache_key]
        if not self._assets_path:
            logger.error("Assets path not available")
            return None
        try:
            image_path = self._assets_path / filename
            if not image_path.exists():
                logger.warning(f"Image file not found: {image_path}")
                return None
            pil_image = transform_monochrome_icon(str(image_path), color, size if size is not None else (100, 100))
            if pil_image is None:
                logger.error(f"Failed to recolor image {filename}")
                return None
            ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=size or pil_image.size)
            self._image_cache[cache_key] = ctk_image
            logger.info(f"Loaded and cached colored image: {filename} with color {color}")
            return ctk_image
        except Exception as e:
            logger.error(f"Failed to load and recolor image {filename}: {e}")
            return None

    def load_logo_image_from_config(
        self, size: Optional[Tuple[int, int]] = None, max_dimension: int = 200, logo_type: str = "full"
    ):
        """
        Load the logo as a PIL image, optionally recolored according to the config, without overwriting the original file.
        Returns a PIL.Image object or None.

        Args:
            size: Optional exact size tuple (width, height). If None, uses max_dimension with aspect ratio preservation.
            max_dimension: Maximum dimension (width or height) when size is None. Aspect ratio is preserved.
            logo_type: Type of logo to load ("full" or "icon")
        """
        logo_props = ui_theme.theme.logo_properties
        if not self._assets_path:
            logger.error("Assets path not available for logo.")
            return None

        # Select the appropriate filename and monochrome setting based on logo_type
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

        # Load the PIL image directly
        try:
            pil_image = Image.open(logo_path)
        except Exception as e:
            logger.error(f"Failed to load image {filename}: {e}")
            return None

        # Apply monochrome conversion if enabled
        if apply_monochrome:
            if size is not None:
                # Use exact size if provided
                return transform_monochrome_icon(
                    str(logo_path), logo_props.color, size, force_all_pixels=True, preserve_aspect_ratio=True
                )
            else:
                # Use max_dimension constraint with aspect ratio preservation
                # Create a square constraint that will be scaled down by aspect ratio preservation
                constraint_size = (max_dimension, max_dimension)
                return transform_monochrome_icon(
                    str(logo_path), logo_props.color, constraint_size, force_all_pixels=True, preserve_aspect_ratio=True
                )
        else:
            # Load image as-is, just resize if needed
            if size is not None:
                # Resize to exact size
                pil_image = pil_image.resize(size, Image.Resampling.LANCZOS)
            elif max_dimension > 0:
                # Resize maintaining aspect ratio with max_dimension constraint
                width, height = pil_image.size
                if width > height:
                    new_width = min(width, max_dimension)
                    new_height = int(height * (new_width / width))
                else:
                    new_height = min(height, max_dimension)
                    new_width = int(width * (new_height / height))
                pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            return pil_image
