"""
Font service for loading and managing custom fonts in the application.
This service loads TTF fonts from the assets directory and makes them available
to Tkinter without requiring system-wide font installation.

Thread Safety:
- All font loading and cache access protected by _font_lock
- Safe to call from any thread
- Fonts loaded once on first access
"""

import ctypes
import logging
import threading
import tkinter.font as tkFont
from pathlib import Path
from typing import Dict, Tuple

from vocalance.app.config.app_config import AssetPathsConfig

logger = logging.getLogger(__name__)


class FontService:
    """
    Thread-safe service for loading and managing custom fonts.

    Thread Safety:
    - _font_lock protects all font loading and cache operations
    - Safe to load fonts from multiple threads
    - Fonts are loaded once and cached
    """

    def __init__(self, asset_paths_config: AssetPathsConfig):
        self._loaded_fonts: Dict[str, str] = {}
        self._font_cache: Dict[Tuple[str, int, str], tkFont.Font] = {}
        self._fonts_loaded = False
        self._asset_paths_config = asset_paths_config
        self._font_lock = threading.RLock()

    def load_fonts(self) -> bool:
        """
        Load all Manrope fonts from the assets directory. Thread-safe.
        Returns True if fonts were successfully loaded, False otherwise.
        """
        with self._font_lock:
            if self._fonts_loaded:
                return True

        try:
            fonts_dir = Path(self._asset_paths_config.fonts_dir)

            variable_font_path = fonts_dir / "Manrope-VariableFont_wght.ttf"
            if variable_font_path.exists():
                self._load_font_file(str(variable_font_path), "Manrope")
                logger.debug("Loaded Manrope variable font")

            static_fonts_dir = fonts_dir / "static"
            if static_fonts_dir.exists():
                font_mappings = {
                    "Manrope-ExtraLight.ttf": "Manrope ExtraLight",
                    "Manrope-Light.ttf": "Manrope Light",
                    "Manrope-Regular.ttf": "Manrope",
                    "Manrope-Medium.ttf": "Manrope Medium",
                    "Manrope-SemiBold.ttf": "Manrope SemiBold",
                    "Manrope-Bold.ttf": "Manrope Bold",
                    "Manrope-ExtraBold.ttf": "Manrope ExtraBold",
                }

                for filename, font_name in font_mappings.items():
                    font_path = static_fonts_dir / filename
                    if font_path.exists():
                        self._load_font_file(str(font_path), font_name)
                        logger.debug(f"Loaded font: {font_name}")

            with self._font_lock:
                self._fonts_loaded = True
                font_count = len(self._loaded_fonts)
            logger.info(f"Successfully loaded {font_count} Manrope fonts")
            return True

        except Exception as e:
            logger.error(f"Failed to load fonts: {e}")
            return False

    def _load_font_file(self, font_path: str, font_name: str) -> bool:
        """
        Load a single font file using Windows GDI font loading. Thread-safe.
        """
        try:
            gdi32 = ctypes.windll.gdi32

            result = gdi32.AddFontResourceExW(ctypes.c_wchar_p(font_path), 0x10, 0)

            if result > 0:
                actual_name = self._extract_font_family_name(font_path, font_name)
                with self._font_lock:
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
        Get the appropriate Manrope font family name for the given weight. Thread-safe.
        Falls back to system fonts if Manrope is not available.

        Args:
            weight: Font weight (regular, light, medium, semibold, bold, extrabold, extralight)

        Returns:
            Font family name that can be used in Tkinter font tuples
        """
        with self._font_lock:
            if not self._fonts_loaded:
                self._font_lock.release()
                self.load_fonts()
                self._font_lock.acquire()

        weight_mapping = {
            "extralight": "Manrope ExtraLight",
            "light": "Manrope Light",
            "regular": "Manrope",
            "medium": "Manrope Medium",
            "semibold": "Manrope SemiBold",
            "bold": "Manrope Bold",
            "extrabold": "Manrope ExtraBold",
        }

        requested_font = weight_mapping.get(weight.lower(), "Manrope")

        with self._font_lock:
            if requested_font in self._loaded_fonts:
                return self._loaded_fonts[requested_font]

            if "Manrope" in self._loaded_fonts:
                return self._loaded_fonts["Manrope"]

        logger.warning("Manrope font not available, falling back to system font")
        return "Segoe UI"

    def create_font(self, family: str = None, size: int = 12, weight: str = "normal") -> tkFont.Font:
        """Create or retrieve a cached font. Thread-safe."""
        if not family:
            family = self.get_font_family(weight)

        cache_key = f"{family}_{size}_{weight}"

        with self._font_lock:
            if cache_key in self._font_cache:
                return self._font_cache[cache_key]

        tk_weight = weight
        if weight == "bold":
            tk_weight = "bold"
        else:
            tk_weight = "normal"

        font = tkFont.Font(family=family, size=size, weight=tk_weight)
        with self._font_lock:
            self._font_cache[cache_key] = font
        return font

    def get_available_fonts(self) -> Dict[str, str]:
        """Get a dictionary of loaded fonts (display name -> actual name). Thread-safe."""
        with self._font_lock:
            if not self._fonts_loaded:
                self._font_lock.release()
                self.load_fonts()
                self._font_lock.acquire()
            return self._loaded_fonts.copy()

    def is_manrope_available(self) -> bool:
        """Check if Manrope fonts are available. Thread-safe."""
        with self._font_lock:
            if not self._fonts_loaded:
                self._font_lock.release()
                self.load_fonts()
                self._font_lock.acquire()
            return "Manrope" in self._loaded_fonts
