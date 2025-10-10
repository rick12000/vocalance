import logging
from typing import Optional, Dict, Any, List
import asyncio
from iris.app.ui.controls.base_control import BaseController
from iris.app.config.app_config import GlobalAppConfig
from iris.app.events.core_events import SettingsResponseEvent


class SettingsController(BaseController):
    """Simplified settings controller that works directly with SettingsService."""
    
    def __init__(self, event_bus, event_loop, logger, config: GlobalAppConfig, settings_service=None):
        super().__init__(event_bus, event_loop, logger, "SettingsController")
        self.config = config
        self.settings_service = settings_service
        self._cached_settings = None
        
        self.subscribe_to_events([
            (SettingsResponseEvent, self._handle_settings_response),
        ])

    def set_settings_service(self, settings_service):
        """Set the settings service reference"""
        self.settings_service = settings_service
        # Request initial settings when service is set
        if settings_service:
            self._request_settings_update()

    def _request_settings_update(self):
        """Request settings update from the service"""
        if self.settings_service:
            # Schedule async call on the event loop
            asyncio.run_coroutine_threadsafe(
                self._get_settings_async(), 
                self.event_loop
            )

    async def _get_settings_async(self):
        """Async method to get settings and cache them"""
        try:
            if self.settings_service:
                self._cached_settings = await self.settings_service.get_effective_settings()
                # Notify view of update
                if self.view_callback:
                    self.schedule_ui_update(self.view_callback.on_settings_updated)
        except Exception as e:
            self.logger.error(f"Error getting settings: {e}")

    def _handle_settings_response(self, event):
        """Handle settings response from the service"""
        self._cached_settings = event.settings
        if self.view_callback:
            self.schedule_ui_update(self.view_callback.on_settings_updated)

    def load_current_settings(self) -> Dict[str, Any]:
        """Load current effective settings from cache or return defaults"""
        if self._cached_settings:
            return self._cached_settings
        
        if not self.settings_service:
            # Return default config values when service is not available
            return {
                'llm': {
                    'model_size': self.config.llm.model_size,
                    'context_length': self.config.llm.context_length,
                    'max_tokens': self.config.llm.max_tokens,
                    'n_threads': self.config.llm.n_threads
                },
                'grid': {
                    'default_rect_count': self.config.grid.default_rect_count
                }
            }
        
        # If we have a service but no cached settings, request them
        self._request_settings_update()
        
        # Return defaults for now
        return {
            'llm': {
                'model_size': self.config.llm.model_size,
                'context_length': self.config.llm.context_length,
                'max_tokens': self.config.llm.max_tokens,
                'n_threads': self.config.llm.n_threads
            },
            'grid': {
                'default_rect_count': self.config.grid.default_rect_count
            }
        }

    def validate_llm_settings(self, model_size: str, context_length: str, max_tokens: str, n_threads: str) -> List[str]:
        """Validate LLM input fields and return list of errors"""
        errors = []
        
        try:
            if model_size not in ["XS", "S", "M", "L"]:
                errors.append("LLM Model Size must be XS, S, M, or L")
            
            context_length_int = int(context_length)
            if context_length_int < 128 or context_length_int > 32768:
                errors.append("Context Length must be between 128 and 32768")
            
            max_tokens_int = int(max_tokens)
            if max_tokens_int < 1 or max_tokens_int > 1024:
                errors.append("Max Tokens must be between 1 and 1024")
            
            threads_int = int(n_threads)
            if threads_int < 1 or threads_int > 32:
                errors.append("Processing Threads must be between 1 and 32")
        except ValueError:
            errors.append("LLM settings must be valid numbers")
        
        return errors

    def validate_grid_settings(self, default_rect_count: str) -> List[str]:
        """Validate Grid input fields and return list of errors"""
        errors = []
        
        try:
            default_rect_count_int = int(default_rect_count)
            if default_rect_count_int < 100 or default_rect_count_int > 10000:
                errors.append("Default Cell Count must be between 100 and 10000")
        except ValueError:
            errors.append("Default Cell Count must be a valid number")
        
        return errors

    def save_llm_settings(self, model_size: str, context_length: str, max_tokens: str, n_threads: str) -> bool:
        """Save LLM settings through the settings service"""
        if not self.settings_service:
            if self.view_callback:
                self.schedule_ui_update(self.view_callback.on_save_error, "Settings service not available")
            return False
        
        validation_errors = self.validate_llm_settings(model_size, context_length, max_tokens, n_threads)
        if validation_errors:
            if self.view_callback:
                self.schedule_ui_update(self.view_callback.on_validation_error, "Invalid Input", "\n".join(validation_errors))
            return False
        
        try:
            # Prepare settings updates in the format expected by the service
            settings_updates = {
                'llm.model_size': model_size,
                'llm.context_length': int(context_length),
                'llm.max_tokens': int(max_tokens),
                'llm.n_threads': int(n_threads)
            }
            
            # Schedule async save operation
            future = asyncio.run_coroutine_threadsafe(
                self._save_settings_async(settings_updates), 
                self.event_loop
            )
            
            return True  # Return immediately, success/failure will be handled in callback
                
        except Exception as e:
            self.logger.error(f"Error saving LLM settings: {e}")
            if self.view_callback:
                self.schedule_ui_update(self.view_callback.on_save_error, f"An unexpected error occurred: {e}")
            return False

    async def _save_settings_async(self, settings_updates: Dict[str, Any]):
        """Async method to save settings"""
        try:
            success = await self.settings_service.update_multiple_settings(settings_updates)
            
            if success:
                # Update cache
                self._cached_settings = await self.settings_service.get_effective_settings()
                
                if self.view_callback:
                    self.schedule_ui_update(self.view_callback.on_save_success, 
                        "LLM settings saved successfully!\n\nNote: Restart the application for changes to take effect.")
            else:
                if self.view_callback:
                    self.schedule_ui_update(self.view_callback.on_save_error, "Failed to save LLM settings")
                    
        except Exception as e:
            self.logger.error(f"Error in async save: {e}")
            if self.view_callback:
                self.schedule_ui_update(self.view_callback.on_save_error, f"Save error: {e}")

    def save_grid_settings(self, default_rect_count: str) -> bool:
        """Save Grid settings through the settings service"""
        if not self.settings_service:
            if self.view_callback:
                self.schedule_ui_update(self.view_callback.on_save_error, "Settings service not available")
            return False
        
        validation_errors = self.validate_grid_settings(default_rect_count)
        if validation_errors:
            if self.view_callback:
                self.schedule_ui_update(self.view_callback.on_validation_error, "Invalid Input", "\n".join(validation_errors))
            return False
        
        try:
            # Prepare settings updates in the format expected by the service
            settings_updates = {
                'grid.default_rect_count': int(default_rect_count)
            }
            
            # Schedule async save operation
            future = asyncio.run_coroutine_threadsafe(
                self._save_grid_settings_async(settings_updates), 
                self.event_loop
            )
            
            return True  # Return immediately, success/failure will be handled in callback
                
        except Exception as e:
            self.logger.error(f"Error saving Grid settings: {e}")
            if self.view_callback:
                self.schedule_ui_update(self.view_callback.on_save_error, f"An unexpected error occurred: {e}")
            return False

    async def _save_grid_settings_async(self, settings_updates: Dict[str, Any]):
        """Async method to save grid settings"""
        try:
            success = await self.settings_service.update_multiple_settings(settings_updates)
            
            if success:
                # Update cache
                self._cached_settings = await self.settings_service.get_effective_settings()
                
                if self.view_callback:
                    self.schedule_ui_update(self.view_callback.on_save_success, 
                        "Grid settings saved successfully!")
            else:
                if self.view_callback:
                    self.schedule_ui_update(self.view_callback.on_save_error, "Failed to save Grid settings")
                    
        except Exception as e:
            self.logger.error(f"Error in async grid save: {e}")
            if self.view_callback:
                self.schedule_ui_update(self.view_callback.on_save_error, f"Save error: {e}")

    def reset_llm_to_defaults(self) -> bool:
        """Reset LLM settings to default values"""
        if not self.settings_service:
            if self.view_callback:
                self.schedule_ui_update(self.view_callback.on_save_error, "Settings service not available")
            return False
        
        try:
            # Schedule async reset operation
            future = asyncio.run_coroutine_threadsafe(
                self._reset_settings_async(), 
                self.event_loop
            )
            
            return True  # Return immediately, success/failure will be handled in callback
                
        except Exception as e:
            self.logger.error(f"Error resetting LLM settings: {e}")
            if self.view_callback:
                self.schedule_ui_update(self.view_callback.on_save_error, f"Failed to reset LLM settings: {e}")
            return False

    async def _reset_settings_async(self):
        """Async method to reset settings"""
        try:
            # Reset LLM settings through the service
            llm_settings = ['llm.model_size', 'llm.context_length', 'llm.max_tokens', 'llm.n_threads']
            
            success = True
            for setting in llm_settings:
                if not await self.settings_service.reset_setting(setting):
                    success = False
                    break
            
            if success:
                # Update cache
                self._cached_settings = await self.settings_service.get_effective_settings()
                
                if self.view_callback:
                    self.schedule_ui_update(self.view_callback.on_reset_complete)
            else:
                if self.view_callback:
                    self.schedule_ui_update(self.view_callback.on_save_error, "Failed to reset LLM settings")
                    
        except Exception as e:
            self.logger.error(f"Error in async reset: {e}")
            if self.view_callback:
                self.schedule_ui_update(self.view_callback.on_save_error, f"Reset error: {e}")

    def reset_grid_to_defaults(self) -> bool:
        """Reset Grid settings to default values"""
        if not self.settings_service:
            if self.view_callback:
                self.schedule_ui_update(self.view_callback.on_save_error, "Settings service not available")
            return False
        
        try:
            # Schedule async reset operation
            future = asyncio.run_coroutine_threadsafe(
                self._reset_grid_settings_async(), 
                self.event_loop
            )
            
            return True  # Return immediately, success/failure will be handled in callback
                
        except Exception as e:
            self.logger.error(f"Error resetting Grid settings: {e}")
            if self.view_callback:
                self.schedule_ui_update(self.view_callback.on_save_error, f"Failed to reset Grid settings: {e}")
            return False

    async def _reset_grid_settings_async(self):
        """Async method to reset grid settings"""
        try:
            success = await self.settings_service.reset_setting('grid.default_rect_count')
            
            if success:
                # Update cache
                self._cached_settings = await self.settings_service.get_effective_settings()
                
                if self.view_callback:
                    self.schedule_ui_update(self.view_callback.on_reset_complete)
            else:
                if self.view_callback:
                    self.schedule_ui_update(self.view_callback.on_save_error, "Failed to reset Grid settings")
                    
        except Exception as e:
            self.logger.error(f"Error in async grid reset: {e}")
            if self.view_callback:
                self.schedule_ui_update(self.view_callback.on_save_error, f"Reset error: {e}")

    def get_default_llm_settings(self) -> Dict[str, Any]:
        """Get default LLM settings from config"""
        return {
            'model_size': self.config.llm.model_size,
            'context_length': self.config.llm.context_length,
            'max_tokens': self.config.llm.max_tokens,
            'n_threads': self.config.llm.n_threads
        } 