"""
Centralized Logo Service for Iris Application

Provides elegant, streamlined logo loading with automatic fallbacks.
"""

import logging
from typing import Optional

import customtkinter as ctk

from iris.app.ui import ui_theme
from iris.app.ui.utils.ui_assets import AssetCache

logger = logging.getLogger(__name__)


class LogoService:
    """Centralized service for loading and managing application logos"""

    def __init__(self, asset_cache: AssetCache):
        """Initialize LogoService with an asset cache instance."""
        self.asset_cache = asset_cache
        self._logo_cache = {}

    def get_logo_image(self, max_size: int, context: str = "default", logo_type: str = "full") -> Optional[ctk.CTkImage]:
        """
        Get a logo image with specified maximum size

        Args:
            max_size: Maximum dimension (width or height) for the logo
            context: Context for logging (e.g., "startup", "sidebar")
            logo_type: Type of logo to load ("full" or "icon")

        Returns:
            CTkImage if successful, None if fallback needed
        """
        cache_key = f"{max_size}_{context}_{logo_type}"

        if cache_key in self._logo_cache:
            return self._logo_cache[cache_key]

        try:
            pil_logo = self.asset_cache.load_logo_image_from_config(size=None, max_dimension=max_size, logo_type=logo_type)

            if pil_logo:
                logo_image = ctk.CTkImage(light_image=pil_logo, dark_image=pil_logo, size=pil_logo.size)
                self._logo_cache[cache_key] = logo_image
                logger.info(f"Logo loaded successfully for {context} (size: {pil_logo.size})")
                return logo_image
            else:
                logger.debug(f"No logo image available for {context}")
                return None

        except Exception as e:
            logger.warning(f"Error loading logo for {context}: {e}")
            return None

    def create_logo_widget(
        self, parent, max_size: int, context: str = "default", text_fallback: str = "Iris", logo_type: str = "full", **kwargs
    ) -> ctk.CTkLabel:
        """
        Create a logo widget with automatic image/text fallback

        Args:
            parent: Parent widget
            max_size: Maximum logo size
            context: Context for logging
            text_fallback: Fallback text if image fails
            logo_type: Type of logo to load ("full" or "icon")

        Returns:
            CTkLabel with logo (image or text)
        """
        logo_image = self.get_logo_image(max_size, context, logo_type)

        if logo_image:
            return ctk.CTkLabel(parent, text="", image=logo_image, anchor="center")
        else:
            logger.info(f"Using text logo for {context}")
            return ctk.CTkLabel(
                parent,
                text=text_fallback,
                font=ctk.CTkFont(size=max_size // 3, weight="bold"),
                text_color=ui_theme.theme.logo_properties.color,
                anchor="center",
            )
