"""
UI Icon Utilities

Focused utilities for setting window icons with robust cross-platform support.
Single responsibility: Icon management for windows.
"""

import tkinter as tk
import logging
import sys
from pathlib import Path
from typing import Optional, Union, List
from iris.app.ui.utils.ui_assets import get_icon_path
from PIL import Image, ImageOps, ImageTk  # Add ImageTk for iconphoto method

logger = logging.getLogger("IconUtils")



def set_window_icon_robust(window: Union[tk.Tk, tk.Toplevel]) -> bool:
    """
    Set icon for any window using a robust approach with multiple fallback methods.
    
    Args:
        window: The window to set the icon for
        
    Returns:
        True if at least one method succeeded, False otherwise
    """
    try:
        icon_path = get_icon_path()
        if not icon_path:
            logger.warning("No icon file available")
            return False
        
        icon_str = str(icon_path.absolute())
        logger.debug(f"Setting window icon from path: {icon_str}")
        
        success_methods = []
        
        # Method 1: Standard iconbitmap (most reliable)
        try:
            window.iconbitmap(icon_str)
            success_methods.append("iconbitmap")
            logger.debug("Set window icon using iconbitmap")
        except Exception as e:
            logger.debug(f"iconbitmap method failed: {e}")
        
        # Method 2: Force window update then try again
        try:
            window.update_idletasks()
            window.update()
            window.iconbitmap(icon_str)
            success_methods.append("post-update iconbitmap")
            logger.debug("Set window icon after forced update")
        except Exception as e:
            logger.debug(f"Post-update iconbitmap failed: {e}")
        
        # Method 3: wm_iconbitmap for better compatibility
        try:
            window.wm_iconbitmap(icon_str)
            success_methods.append("wm_iconbitmap")
            logger.debug("Set window icon using wm_iconbitmap")
        except Exception as e:
            logger.debug(f"wm_iconbitmap failed: {e}")
        
        # Method 4: Direct tk call for CustomTkinter compatibility
        try:
            if hasattr(window, 'tk') and hasattr(window, '_w'):
                window.tk.call('wm', 'iconbitmap', window._w, icon_str)
                success_methods.append("tk.call")
                logger.debug("Set window icon using tk.call")
        except Exception as e:
            logger.debug(f"tk.call method failed: {e}")
        
        # Method 5: Windows-specific approach
        try:
            if sys.platform == "win32":
                window.wm_iconbitmap(default=icon_str)
                success_methods.append("windows default")
                logger.debug("Set icon using Windows default method")
        except Exception as e:
            logger.debug(f"Windows default icon method failed: {e}")
        
        # Method 6: Force title bar refresh to ensure icon visibility
        try:
            current_title = window.title()
            window.title(current_title + " ")  # Slight change
            window.update()
            window.title(current_title)  # Restore original
            window.update()
            logger.debug("Forced title bar refresh to ensure icon visibility")
        except Exception as e:
            logger.debug(f"Title bar refresh failed: {e}")
        
        # Method 7: Use iconphoto for high-DPI environments
        try:
            pil_img = Image.open(icon_path)
            tk_img = ImageTk.PhotoImage(pil_img)
            window.iconphoto(False, tk_img)
            # Keep reference to prevent garbage collection
            setattr(window, '_iconphoto_image', tk_img)
            success_methods.append("iconphoto")
            logger.debug("Set window icon using iconphoto")
        except Exception as e:
            logger.debug(f"iconphoto method failed: {e}")
        
        if success_methods:
            logger.info(f"Successfully set window icon using methods: {success_methods}")
            return True
        else:
            logger.warning("All icon setting methods failed for window")
            return False
            
    except Exception as e:
        logger.error(f"Could not set window icon: {e}", exc_info=True)
        return False

def ensure_parent_has_icon(parent_window: tk.Tk) -> None:
    """
    Ensure parent window has icon set for proper inheritance.
    
    Args:
        parent_window: The parent window to ensure has an icon
    """
    try:
        icon_path = get_icon_path()
        if not icon_path:
            return
        
        icon_str = str(icon_path.absolute())
        
        # Force parent window update to ensure it's established
        parent_window.update_idletasks()
        parent_window.iconbitmap(icon_str)
        parent_window.update()
        logger.debug("Ensured parent window has icon and is updated")
        
    except Exception as e:
        logger.debug(f"Parent icon setting failed: {e}")

def set_window_icon_with_parent_inheritance(window: Union[tk.Tk, tk.Toplevel], 
                                          parent: Optional[tk.Tk] = None) -> bool:
    """
    Set window icon with proper parent inheritance for better compatibility.
    
    Args:
        window: The window to set the icon for
        parent: Optional parent window to ensure has icon first
        
    Returns:
        True if successful, False otherwise
    """
    # First ensure parent has icon if provided
    if parent and window != parent:
        ensure_parent_has_icon(parent)
    
    # Then set icon on the target window
    return set_window_icon_robust(window)



def transform_monochrome_icon(icon_path: str, target_color: str, size: tuple = None, force_all_pixels: bool = False, preserve_aspect_ratio: bool = True) -> Image.Image | None:
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
        # Load the image
        img = Image.open(icon_path).convert("RGBA")
        
        # Resize if requested
        if size:
            if preserve_aspect_ratio and len(size) == 2:
                # Calculate aspect ratio preserving resize
                orig_w, orig_h = img.size
                target_w, target_h = size
                
                # Use the smaller scale factor to ensure image fits within target size
                scale = min(target_w / orig_w, target_h / orig_h)
                new_w = int(orig_w * scale)
                new_h = int(orig_h * scale)
                
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            else:
                img = img.resize(size, Image.Resampling.LANCZOS)
        
        # Convert hex color to RGB
        target_color = target_color.lstrip('#')
        if len(target_color) != 6:
            raise ValueError(f"Invalid hex color: {target_color}")
        
        r = int(target_color[0:2], 16)
        g = int(target_color[2:4], 16) 
        b = int(target_color[4:6], 16)
        target_rgb = (r, g, b)
        
        # Create a new image with the target color
        colored_img = Image.new("RGBA", img.size, (0, 0, 0, 0))
        
        # Get pixel data
        pixels = img.load()
        colored_pixels = colored_img.load()
        
        # Process each pixel
        for y in range(img.height):
            for x in range(img.width):
                r_orig, g_orig, b_orig, a_orig = pixels[x, y]
                
                # Skip transparent pixels
                if a_orig == 0:
                    continue
                
                if force_all_pixels:
                    # Recolor all non-transparent pixels to the target color
                    colored_pixels[x, y] = (target_rgb[0], target_rgb[1], target_rgb[2], a_orig)
                else:
                    # For monochrome icons, we want to preserve the original alpha
                    # and recolor based on how "dark" the original pixel is
                    # Dark pixels (like black icons) should become the target color
                    # Light pixels should become more transparent versions of the target color
                    
                    # Calculate luminance (brightness) of the original pixel
                    luminance = 0.299 * r_orig + 0.587 * g_orig + 0.114 * b_orig
                    
                    # For dark icons on transparent backgrounds:
                    # - Dark pixels (low luminance) should be fully opaque in target color
                    # - Light pixels should be more transparent
                    # Invert the luminance so dark pixels get high opacity
                    opacity_factor = (255 - luminance) / 255.0
                    new_alpha = int(opacity_factor * a_orig)
                    
                    # Apply the target color with computed alpha
                    colored_pixels[x, y] = (target_rgb[0], target_rgb[1], target_rgb[2], new_alpha)
        
        return colored_img
        
    except Exception as e:
        logger.error(f"Failed to transform icon {icon_path}: {e}")
        return None


def track_window_for_icon_management(window: Union[tk.Tk, tk.Toplevel]) -> None:
    """
    Track window for icon management (placeholder function).
    
    This function is a placeholder for window icon tracking functionality.
    Currently does nothing but prevents import errors.
    
    Args:
        window: The window to track
    """
    # Placeholder - no actual tracking needed for now
    pass