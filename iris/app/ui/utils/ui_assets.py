"""
UI Asset Management Utility

Handles loading, caching, and management of UI assets like images and icons.
Single responsibility: Asset loading and caching.
"""

import customtkinter as ctk
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple
from PIL import Image

logger = logging.getLogger("UIAssets")

class AssetCache:
    """Simple asset cache for images and icons."""
    
    def __init__(self):
        self._image_cache: Dict[str, ctk.CTkImage] = {}
        self._assets_path: Optional[Path] = None
        self._setup_assets_path()
    
    def _setup_assets_path(self) -> None:
        """Set up the path to UI assets."""
        # Navigate from iris/app/ui/utils to iris/app/assets/logo
        # This works in both development and installed package mode
        current_dir = Path(__file__).resolve().parent
        app_root = current_dir.parent.parent
        self._assets_path = app_root / "assets" / "logo"

        if not self._assets_path.exists():
            logger.error(f"Assets directory not found: {self._assets_path}")
            logger.error(f"Searched from: {current_dir}")
            logger.error(f"App root calculated as: {app_root}")
            self._assets_path = None
        else:
            logger.debug(f"Assets path found: {self._assets_path}")
    
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
            ctk_image = ctk.CTkImage(
                light_image=pil_image,
                dark_image=pil_image,
                size=size or pil_image.size
            )
            
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
    
    def clear_cache(self) -> None:
        """Clear the image cache."""
        self._image_cache.clear()
        logger.info("Asset cache cleared")

# Global asset cache instance
_asset_cache = AssetCache()

def get_asset_cache() -> AssetCache:
    """Get the global asset cache instance."""
    return _asset_cache

def load_image(filename: str, size: Optional[Tuple[int, int]] = None) -> Optional[ctk.CTkImage]:
    """Convenience function to load an image using the global cache."""
    return _asset_cache.load_image(filename, size)

def get_assets_path() -> Optional[Path]:
    """Convenience function to get the assets path."""
    return _asset_cache.get_assets_path()

def get_icon_path() -> Optional[Path]:
    """Convenience function to get the icon path."""
    return _asset_cache.get_icon_path()

def load_image_monochrome_colored(filename: str, color: str, size: Optional[Tuple[int, int]] = None) -> Optional[ctk.CTkImage]:
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
    if cache_key in _asset_cache._image_cache:
        return _asset_cache._image_cache[cache_key]
    if not _asset_cache._assets_path:
        logger.error("Assets path not available")
        return None
    try:
        image_path = _asset_cache._assets_path / filename
        if not image_path.exists():
            logger.warning(f"Image file not found: {image_path}")
            return None
        from iris.app.ui.utils.ui_icon_utils import transform_monochrome_icon
        pil_image = transform_monochrome_icon(str(image_path), color, size if size is not None else (100, 100))
        if pil_image is None:
            logger.error(f"Failed to recolor image {filename}")
            return None
        ctk_image = ctk.CTkImage(
            light_image=pil_image,
            dark_image=pil_image,
            size=size or pil_image.size
        )
        _asset_cache._image_cache[cache_key] = ctk_image
        logger.info(f"Loaded and cached colored image: {filename} with color {color}")
        return ctk_image
    except Exception as e:
        logger.error(f"Failed to load and recolor image {filename}: {e}")
        return None

def load_logo_image_from_config(size: Optional[Tuple[int, int]] = None, max_dimension: int = 200):
    """
    Load the logo as a PIL image recolored according to the config, without overwriting the original file.
    Returns a PIL.Image object or None.
    
    Args:
        size: Optional exact size tuple (width, height). If None, uses max_dimension with aspect ratio preservation.
        max_dimension: Maximum dimension (width or height) when size is None. Aspect ratio is preserved.
    """
    from iris.app.ui import ui_theme
    logo_props = ui_theme.theme.logo_properties
    assets_path = get_assets_path()
    if not assets_path:
        logger.error("Assets path not available for logo.")
        return None
    logo_path = assets_path / logo_props.filename
    if not logo_path.exists():
        logger.error(f"Logo file not found: {logo_path}")
        return None
    from iris.app.ui.utils.ui_icon_utils import transform_monochrome_icon
    
    if size is not None:
        # Use exact size if provided
        return transform_monochrome_icon(str(logo_path), logo_props.color, size, force_all_pixels=True, preserve_aspect_ratio=True)
    else:
        # Use max_dimension constraint with aspect ratio preservation
        # Create a square constraint that will be scaled down by aspect ratio preservation
        constraint_size = (max_dimension, max_dimension)
        return transform_monochrome_icon(str(logo_path), logo_props.color, constraint_size, force_all_pixels=True, preserve_aspect_ratio=True)