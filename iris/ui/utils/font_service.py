"""
Font service for loading and managing custom fonts in the application.
This service loads TTF fonts from the assets directory and makes them available
to Tkinter without requiring system-wide font installation.
"""

import os
import tkinter as tk
import tkinter.font as tkFont
from typing import Dict, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class FontService:
    """Service for loading and managing custom fonts"""
    
    def __init__(self):
        self._loaded_fonts: Dict[str, str] = {}
        self._font_cache: Dict[Tuple[str, int, str], tkFont.Font] = {}
        self._fonts_loaded = False
        
    def load_fonts(self) -> bool:
        """
        Load all Manrope fonts from the assets directory.
        Returns True if fonts were successfully loaded, False otherwise.
        """
        if self._fonts_loaded:
            return True
            
        try:
            # Get the path to the fonts directory
            # Navigate from src/ui/utils/ to assets/fonts/Manrope/
            current_dir = Path(__file__).parent
            project_root = current_dir.parent.parent.parent
            fonts_dir = project_root / "assets" / "fonts" / "Manrope"
            
            if not fonts_dir.exists():
                logger.error(f"Fonts directory not found: {fonts_dir}")
                return False
            
            # Load variable font first (preferred)
            variable_font_path = fonts_dir / "Manrope-VariableFont_wght.ttf"
            if variable_font_path.exists():
                self._load_font_file(str(variable_font_path), "Manrope")
                logger.info("Loaded Manrope variable font")
            
            # Load static fonts as fallback
            static_fonts_dir = fonts_dir / "static"
            if static_fonts_dir.exists():
                font_mappings = {
                    "Manrope-ExtraLight.ttf": "Manrope ExtraLight",
                    "Manrope-Light.ttf": "Manrope Light", 
                    "Manrope-Regular.ttf": "Manrope",
                    "Manrope-Medium.ttf": "Manrope Medium",
                    "Manrope-SemiBold.ttf": "Manrope SemiBold",
                    "Manrope-Bold.ttf": "Manrope Bold",
                    "Manrope-ExtraBold.ttf": "Manrope ExtraBold"
                }
                
                for filename, font_name in font_mappings.items():
                    font_path = static_fonts_dir / filename
                    if font_path.exists():
                        self._load_font_file(str(font_path), font_name)
                        logger.debug(f"Loaded font: {font_name}")
            
            self._fonts_loaded = True
            logger.info(f"Successfully loaded {len(self._loaded_fonts)} Manrope fonts")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load fonts: {e}")
            return False
    
    def _load_font_file(self, font_path: str, font_name: str) -> bool:
        """
        Load a single font file using Windows GDI font loading.
        """
        try:
            import ctypes
            from ctypes import wintypes
            
            # Load the font using Windows GDI
            gdi32 = ctypes.windll.gdi32
            
            # AddFontResourceEx function
            result = gdi32.AddFontResourceExW(
                ctypes.c_wchar_p(font_path),
                0x10,  # FR_PRIVATE - font is private to the process
                0
            )
            
            if result > 0:
                # Font loaded successfully
                # For TTF fonts, we need to extract the actual font family name
                # We'll use a simplified approach and assume the font name matches the file
                actual_name = self._extract_font_family_name(font_path, font_name)
                self._loaded_fonts[font_name] = actual_name
                logger.debug(f"Successfully loaded font {font_name} -> {actual_name}")
                return True
            else:
                logger.warning(f"Failed to load font {font_name} from {font_path} (GDI result: {result})")
                return False
                
        except Exception as e:
            logger.warning(f"Failed to load font {font_name} from {font_path}: {e}")
            return False
    
    def _extract_font_family_name(self, font_path: str, fallback_name: str) -> str:
        """
        Extract the actual font family name from a TTF file.
        """
        try:
            # Try to read the font name from the TTF file
            # This is a simplified approach - in a production system you might use
            # a library like fonttools for more robust font name extraction
            
            # For now, we'll use a mapping based on the file names
            filename = Path(font_path).name.lower()
            
            if "regular" in filename or filename.endswith("manrope.ttf"):
                return "Manrope Medium"
            elif "light" in filename:
                return "Manrope Light"
            elif "medium" in filename:
                return "Manrope Medium"
            elif "semibold" in filename:
                return "Manrope SemiBold"
            elif "bold" in filename and "extrabold" not in filename:
                return "Manrope Bold"
            elif "extrabold" in filename:
                return "Manrope ExtraBold"
            elif "extralight" in filename:
                return "Manrope ExtraLight"
            else:
                return fallback_name
                
        except Exception as e:
            logger.warning(f"Failed to extract font family name from {font_path}: {e}")
            return fallback_name
    
    def get_font_family(self, weight: str = "regular") -> str:
        """
        Get the appropriate Manrope font family name for the given weight.
        Falls back to system fonts if Manrope is not available.
        
        Args:
            weight: Font weight (regular, light, medium, semibold, bold, extrabold, extralight)
            
        Returns:
            Font family name that can be used in Tkinter font tuples
        """
        if not self._fonts_loaded:
            self.load_fonts()
        
        # Map weight to font name
        weight_mapping = {
            "extralight": "Manrope ExtraLight",
            "light": "Manrope Light",
            "regular": "Manrope",
            "medium": "Manrope Medium", 
            "semibold": "Manrope SemiBold",
            "bold": "Manrope Bold",
            "extrabold": "Manrope ExtraBold"
        }
        
        requested_font = weight_mapping.get(weight.lower(), "Manrope")
        
        # Check if the requested font is loaded
        if requested_font in self._loaded_fonts:
            return self._loaded_fonts[requested_font]
        
        # Fall back to base Manrope if available
        if "Manrope" in self._loaded_fonts:
            return self._loaded_fonts["Manrope"]
        
        # Final fallback to system fonts
        logger.warning(f"Manrope font not available, falling back to system font")
        return "Segoe UI"  # Windows default
    
    def create_font(self, family: str = None, size: int = 12, weight: str = "normal") -> tkFont.Font:
        """
        Create a Font object with the specified parameters.
        Uses caching to avoid creating duplicate font objects.
        
        Args:
            family: Font family (if None, uses Manrope regular)
            size: Font size in points
            weight: Font weight (normal, bold, or Manrope weight names)
            
        Returns:
            tkFont.Font object
        """
        # If no family specified, use Manrope
        if family is None:
            # Map Tkinter weight to Manrope weight
            if weight == "bold":
                family = self.get_font_family("bold")
                tk_weight = "normal"  # Since we're using the bold variant
            else:
                family = self.get_font_family("regular")
                tk_weight = weight
        else:
            tk_weight = weight
        
        # Create cache key
        cache_key = (family, size, tk_weight)
        
        # Return cached font if available
        if cache_key in self._font_cache:
            return self._font_cache[cache_key]
        
        # Create new font
        font = tkFont.Font(family=family, size=size, weight=tk_weight)
        self._font_cache[cache_key] = font
        
        return font
    
    def get_available_fonts(self) -> Dict[str, str]:
        """Get a dictionary of loaded fonts (display name -> actual name)"""
        if not self._fonts_loaded:
            self.load_fonts()
        return self._loaded_fonts.copy()
    
    def is_manrope_available(self) -> bool:
        """Check if Manrope fonts are available"""
        if not self._fonts_loaded:
            self.load_fonts()
        return "Manrope" in self._loaded_fonts


# Global font service instance
font_service = FontService()


def get_font_service() -> FontService:
    """Get the global font service instance"""
    return font_service 