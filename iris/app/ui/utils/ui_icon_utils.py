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
from PIL import Image, ImageOps, ImageTk

from iris.app.ui.utils.ui_assets import AssetCache
from iris.app.ui.utils.icon_transform_utils import transform_monochrome_icon
from iris.app.config.app_config import AssetPathsConfig

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
        icon_path = AssetCache(asset_paths_config=AssetPathsConfig()).get_icon_path()
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
        icon_path =  AssetCache(asset_paths_config=AssetPathsConfig()).get_icon_path()
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



