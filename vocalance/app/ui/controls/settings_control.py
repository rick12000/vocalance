import asyncio
from typing import Any, Dict, List, Optional

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.events.core_events import SettingsResponseEvent
from vocalance.app.ui.controls.base_control import BaseController


class SettingsController(BaseController):
    """
    Simplified settings controller that works directly with SettingsService.

    Thread Safety:
    - _cached_settings protected by inherited _state_lock
    - Event handlers run in GUI event loop thread
    - UI updates marshalled to main thread via schedule_ui_update
    """

    def __init__(self, event_bus, event_loop, logger, config: GlobalAppConfig, settings_service=None):
        super().__init__(event_bus, event_loop, logger, "SettingsController")
        self.config = config
        self.settings_service = settings_service
        self._cached_settings = None  # Protected by _state_lock

        self.subscribe_to_events(
            [
                (SettingsResponseEvent, self._handle_settings_response),
            ]
        )

    def set_settings_service(self, settings_service):
        """Set the settings service reference"""
        self.settings_service = settings_service
        if settings_service:
            self._request_settings_update()

    def _request_settings_update(self):
        """Request settings update from the service"""
        if self.settings_service:
            asyncio.run_coroutine_threadsafe(self._get_settings_async(), self.event_loop)

    async def _get_settings_async(self):
        """Async method to get settings and cache them. Thread-safe."""
        try:
            if self.settings_service:
                settings = await self.settings_service.get_effective_settings()
                with self._state_lock:
                    self._cached_settings = settings
                if self.view_callback:
                    self.schedule_ui_update(self.view_callback.on_settings_updated)
        except Exception as e:
            self.logger.error(f"Error getting settings: {e}")

    def _handle_settings_response(self, event):
        """Handle settings response from the service. Thread-safe."""
        with self._state_lock:
            self._cached_settings = event.settings
        if self.view_callback:
            self.schedule_ui_update(self.view_callback.on_settings_updated)

    def load_current_settings(self) -> Dict[str, Any]:
        """Load current effective settings from cache or return defaults. Thread-safe."""
        with self._state_lock:
            if self._cached_settings:
                return self._cached_settings

        if not self.settings_service:
            return {
                "llm": {"context_length": self.config.llm.context_length, "max_tokens": self.config.llm.max_tokens},
                "grid": {"default_rect_count": self.config.grid.default_rect_count},
            }

        self._request_settings_update()

        return {
            "llm": {"context_length": self.config.llm.context_length, "max_tokens": self.config.llm.max_tokens},
            "grid": {"default_rect_count": self.config.grid.default_rect_count},
            "markov_predictor": {"confidence_threshold": self.config.markov_predictor.confidence_threshold},
            "sound_recognizer": {
                "confidence_threshold": self.config.sound_recognizer.confidence_threshold,
                "vote_threshold": self.config.sound_recognizer.vote_threshold,
            },
            "vad": {"dictation_silent_chunks_for_end": self.config.vad.dictation_silent_chunks_for_end},
        }

    # Validation methods
    def validate_llm_settings(self, context_length: str, max_tokens: str) -> List[str]:
        """Validate LLM input fields and return list of errors"""
        errors = []
        try:
            context_length_int = int(context_length)
            if context_length_int < 128 or context_length_int > 32768:
                errors.append("Context Length must be between 128 and 32768")

            max_tokens_int = int(max_tokens)
            if max_tokens_int < 1 or max_tokens_int > 1024:
                errors.append("Max Tokens must be between 1 and 1024")
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

    def _validate_threshold(self, value_str: str, name: str) -> Optional[str]:
        """Validate a threshold value (0.0-1.0). Returns error message or None."""
        try:
            value = float(value_str)
            if value < 0.0 or value > 1.0:
                return f"{name} must be between 0.0 and 1.0"
        except ValueError:
            return f"{name} must be a valid number"
        return None

    def validate_markov_settings(self, confidence_threshold: str) -> List[str]:
        """Validate Markov input fields and return list of errors"""
        errors = []
        error = self._validate_threshold(confidence_threshold, "Markov Confidence Threshold")
        if error:
            errors.append(error)
        return errors

    def validate_sound_settings(self, confidence_threshold: str, vote_threshold: str) -> List[str]:
        """Validate Sound Recognizer input fields and return list of errors"""
        errors = []
        error = self._validate_threshold(confidence_threshold, "Sound Confidence Threshold")
        if error:
            errors.append(error)
        error = self._validate_threshold(vote_threshold, "Sound Vote Threshold")
        if error:
            errors.append(error)
        return errors

    def validate_dictation_settings(self, silent_chunks: str) -> List[str]:
        """Validate Dictation input fields and return list of errors"""
        errors = []
        try:
            silent_chunks_int = int(silent_chunks)
            if silent_chunks_int < 1 or silent_chunks_int > 1000:
                errors.append("Silent Chunks for End must be between 1 and 1000")
        except ValueError:
            errors.append("Silent Chunks for End must be a valid integer")
        return errors

    # Generic save method
    def _save_settings_generic(self, settings_updates: Dict[str, Any], success_msg: str, category: str) -> bool:
        """Generic settings save method"""
        if not self.settings_service:
            if self.view_callback:
                self.schedule_ui_update(self.view_callback.on_save_error, "Settings service not available")
            return False

        try:
            asyncio.run_coroutine_threadsafe(self._save_settings_async(settings_updates, success_msg, category), self.event_loop)
            return True
        except Exception as e:
            self.logger.error(f"Error saving {category} settings: {e}")
            if self.view_callback:
                self.schedule_ui_update(self.view_callback.on_save_error, f"An unexpected error occurred: {e}")
            return False

    async def _save_settings_async(self, settings_updates: Dict[str, Any], success_msg: str, category: str):
        """Unified async save method. Thread-safe."""
        try:
            success = await self.settings_service.update_multiple_settings(settings_updates)

            if success:
                settings = await self.settings_service.get_effective_settings()
                with self._state_lock:
                    self._cached_settings = settings
                if self.view_callback:
                    self.schedule_ui_update(self.view_callback.on_save_success, success_msg)
            else:
                if self.view_callback:
                    self.schedule_ui_update(self.view_callback.on_save_error, f"Failed to save {category} settings")
        except Exception as e:
            self.logger.error(f"Error in async {category} save: {e}")
            if self.view_callback:
                self.schedule_ui_update(self.view_callback.on_save_error, f"Save error: {e}")

    # Generic reset method
    def _reset_settings_generic(self, settings_list: List[str], category: str) -> bool:
        """Generic settings reset method"""
        if not self.settings_service:
            if self.view_callback:
                self.schedule_ui_update(self.view_callback.on_save_error, "Settings service not available")
            return False

        try:
            asyncio.run_coroutine_threadsafe(self._reset_settings_async(settings_list, category), self.event_loop)
            return True
        except Exception as e:
            self.logger.error(f"Error resetting {category} settings: {e}")
            if self.view_callback:
                self.schedule_ui_update(self.view_callback.on_save_error, f"Failed to reset {category} settings: {e}")
            return False

    async def _reset_settings_async(self, settings_list: List[str], category: str):
        """Unified async reset method. Thread-safe."""
        try:
            success = True
            for setting in settings_list:
                if not await self.settings_service.reset_setting(setting):
                    success = False
                    break

            if success:
                settings = await self.settings_service.get_effective_settings()
                with self._state_lock:
                    self._cached_settings = settings
                if self.view_callback:
                    self.schedule_ui_update(self.view_callback.on_reset_complete)
            else:
                if self.view_callback:
                    self.schedule_ui_update(self.view_callback.on_save_error, f"Failed to reset {category} settings")
        except Exception as e:
            self.logger.error(f"Error in async {category} reset: {e}", exc_info=True)
            if self.view_callback:
                self.schedule_ui_update(self.view_callback.on_save_error, f"Reset error: {e}")

    # Public save methods
    def save_llm_settings(self, context_length: str, max_tokens: str) -> bool:
        """Save LLM settings through the settings service"""
        validation_errors = self.validate_llm_settings(context_length, max_tokens)
        if validation_errors:
            if self.view_callback:
                self.schedule_ui_update(self.view_callback.on_validation_error, "Invalid Input", "\n".join(validation_errors))
            return False

        settings_updates = {"llm.context_length": int(context_length), "llm.max_tokens": int(max_tokens)}
        return self._save_settings_generic(
            settings_updates=settings_updates,
            success_msg="LLM settings saved successfully!\n\nNote: Restart the application for changes to take effect.",
            category="LLM",
        )

    def save_grid_settings(self, default_rect_count: str) -> bool:
        """Save Grid settings through the settings service"""
        validation_errors = self.validate_grid_settings(default_rect_count)
        if validation_errors:
            if self.view_callback:
                self.schedule_ui_update(self.view_callback.on_validation_error, "Invalid Input", "\n".join(validation_errors))
            return False

        settings_updates = {"grid.default_rect_count": int(default_rect_count)}
        return self._save_settings_generic(
            settings_updates=settings_updates,
            success_msg="Grid settings saved successfully!\n\nChanges take effect immediately.",
            category="Grid",
        )

    def save_markov_settings(self, confidence_threshold: str) -> bool:
        """Save Markov settings through the settings service"""
        validation_errors = self.validate_markov_settings(confidence_threshold=confidence_threshold)
        if validation_errors:
            if self.view_callback:
                self.schedule_ui_update(self.view_callback.on_validation_error, "Invalid Input", "\n".join(validation_errors))
            return False

        settings_updates = {"markov_predictor.confidence_threshold": float(confidence_threshold)}
        return self._save_settings_generic(
            settings_updates=settings_updates,
            success_msg="Markov settings saved successfully!\n\nChanges take effect immediately.",
            category="Markov",
        )

    def save_sound_settings(self, confidence_threshold: str, vote_threshold: str) -> bool:
        """Save Sound Recognizer settings through the settings service"""
        validation_errors = self.validate_sound_settings(confidence_threshold=confidence_threshold, vote_threshold=vote_threshold)
        if validation_errors:
            if self.view_callback:
                self.schedule_ui_update(self.view_callback.on_validation_error, "Invalid Input", "\n".join(validation_errors))
            return False

        settings_updates = {
            "sound_recognizer.confidence_threshold": float(confidence_threshold),
            "sound_recognizer.vote_threshold": float(vote_threshold),
        }
        return self._save_settings_generic(
            settings_updates=settings_updates,
            success_msg="Sound Recognizer settings saved successfully!\n\nChanges take effect immediately.",
            category="Sound Recognizer",
        )

    def save_dictation_settings(self, silent_chunks: str) -> bool:
        """Save Dictation settings through the settings service"""
        validation_errors = self.validate_dictation_settings(silent_chunks=silent_chunks)
        if validation_errors:
            if self.view_callback:
                self.schedule_ui_update(self.view_callback.on_validation_error, "Invalid Input", "\n".join(validation_errors))
            return False

        settings_updates = {"vad.dictation_silent_chunks_for_end": int(silent_chunks)}
        return self._save_settings_generic(
            settings_updates=settings_updates,
            success_msg="Dictation settings saved successfully!\n\nChanges take effect immediately.",
            category="Dictation",
        )

    # Public reset methods
    def reset_llm_to_defaults(self) -> bool:
        """Reset LLM settings to default values"""
        return self._reset_settings_generic(["llm.context_length", "llm.max_tokens"], "LLM")

    def reset_grid_to_defaults(self) -> bool:
        """Reset Grid settings to default values"""
        return self._reset_settings_generic(["grid.default_rect_count"], "Grid")

    def reset_markov_to_defaults(self) -> bool:
        """Reset Markov settings to default values"""
        return self._reset_settings_generic(["markov_predictor.confidence_threshold"], "Markov")

    def reset_sound_to_defaults(self) -> bool:
        """Reset Sound Recognizer settings to default values"""
        return self._reset_settings_generic(
            ["sound_recognizer.confidence_threshold", "sound_recognizer.vote_threshold"], "Sound Recognizer"
        )

    def reset_dictation_to_defaults(self) -> bool:
        """Reset Dictation settings to default values"""
        return self._reset_settings_generic(["vad.dictation_silent_chunks_for_end"], "Dictation")
