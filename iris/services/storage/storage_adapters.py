"""
Streamlined Storage Adapters

Simplified adapters that provide compatibility between existing services
and the unified storage backend.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple, Set

from iris.services.storage.unified_storage_service import (
    UnifiedStorageService, StorageType, StorageKey,
    read_settings, write_settings, read_commands, write_commands,
    read_agentic_prompts, write_agentic_prompts, read_sound_mappings, write_sound_mappings,
    read_marks, write_marks, read_grid_clicks, write_grid_clicks
)

logger = logging.getLogger(__name__)

from iris.config.command_types import AutomationCommand
from iris.config.automation_command_registry import AutomationCommandRegistry
import os
import soundfile as sf
import joblib

class SettingsStorageAdapter:
    """Adapter for settings service"""
    
    def __init__(self, unified_storage: UnifiedStorageService):
        self._storage = unified_storage
    
    async def load_user_settings(self) -> Dict[str, Any]:
        """Load user settings"""
        try:
            settings = await read_settings(self._storage, "user_settings", {})
            logger.debug(f"Loaded {len(settings)} user settings")
            return settings
        except Exception as e:
            logger.error(f"Failed to load user settings: {e}")
            return {}
    
    async def save_user_settings(self, settings: Dict[str, Any]) -> bool:
        """Save user settings"""
        try:
            success = await write_settings(self._storage, "user_settings", settings)
            if success:
                logger.debug(f"Saved {len(settings)} user settings")
            return success
        except Exception as e:
            logger.error(f"Failed to save user settings: {e}")
            return False
    
    async def update_setting(self, key: str, value: Any) -> bool:
        """Update single setting with dot notation"""
        try:
            settings = await self.load_user_settings()
            
            # Handle nested keys
            keys = key.split('.')
            current = settings
            
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            current[keys[-1]] = value
            return await self.save_user_settings(settings)
        except Exception as e:
            logger.error(f"Failed to update setting {key}: {e}")
            return False
    
    async def get_setting(self, key: str, default: Any = None) -> Any:
        """Get single setting with dot notation"""
        try:
            settings = await self.load_user_settings()
            
            keys = key.split('.')
            current = settings
            
            for k in keys:
                if isinstance(current, dict) and k in current:
                    current = current[k]
                else:
                    return default
            
            return current
        except Exception as e:
            logger.error(f"Failed to get setting {key}: {e}")
            return default


class CommandStorageAdapter:
    """Adapter for command storage service"""
    
    def __init__(self, unified_storage: UnifiedStorageService):
        self._storage = unified_storage
    
    async def load_custom_commands(self) -> Dict[str, Any]:
        """Load custom commands and convert back to AutomationCommand objects"""
        try:
            data = await read_commands(self._storage, {})
            custom_commands_data = data.get('custom_commands', {})
            
            # Convert dictionary data back to AutomationCommand objects
            custom_commands = {}
            for phrase, command_dict in custom_commands_data.items():
                if isinstance(command_dict, dict):
                    try:
                        # Create AutomationCommand from dictionary
                        command_obj = AutomationCommand(**command_dict)
                        custom_commands[phrase] = command_obj
                    except Exception as e:
                        logger.warning(f"Failed to deserialize command '{phrase}': {e}")
                        # Skip invalid commands
                        continue
                else:
                    # Handle legacy format or direct objects
                    custom_commands[phrase] = command_dict
            
            logger.debug(f"Loaded {len(custom_commands)} custom commands")
            return custom_commands
        except Exception as e:
            logger.error(f"Failed to load custom commands: {e}")
            return {}
    
    async def save_custom_commands(self, commands_data: Dict[str, Any]) -> bool:
        """Save custom commands, converting AutomationCommand objects to dictionaries"""
        try:
            # Convert AutomationCommand objects to dictionaries for JSON serialization
            serializable_commands = {}
            for phrase, command_data in commands_data.items():
                if hasattr(command_data, 'model_dump'):
                    # Pydantic v2 method
                    serializable_commands[phrase] = command_data.model_dump()
                elif hasattr(command_data, 'dict'):
                    # Pydantic v1 method
                    serializable_commands[phrase] = command_data.dict()
                elif isinstance(command_data, dict):
                    # Already a dictionary
                    serializable_commands[phrase] = command_data
                else:
                    # Fallback: convert to dict if possible
                    try:
                        serializable_commands[phrase] = dict(command_data)
                    except Exception as e:
                        logger.error(f"Failed to serialize command '{phrase}': {e}")
                        continue
            
            current_data = await read_commands(self._storage, {})
            current_data['custom_commands'] = serializable_commands
            success = await write_commands(self._storage, current_data)
            if success:
                logger.debug(f"Saved {len(serializable_commands)} custom commands")
            return success
        except Exception as e:
            logger.error(f"Failed to save custom commands: {e}")
            return False
    
    async def get_custom_commands(self) -> Dict[str, Any]:
        """Get custom commands"""
        return await self.load_custom_commands()
    
    async def add_custom_command(self, phrase: str, command_data: Any) -> bool:
        """Add single custom command"""
        try:
            commands = await self.load_custom_commands()
            commands[phrase] = command_data
            return await self.save_custom_commands(commands)
        except Exception as e:
            logger.error(f"Failed to add custom command: {e}")
            return False
    
    async def delete_custom_command(self, phrase: str) -> bool:
        """Delete custom command"""
        try:
            commands = await self.load_custom_commands()
            if phrase in commands:
                del commands[phrase]
                return await self.save_custom_commands(commands)
            return True
        except Exception as e:
            logger.error(f"Failed to delete custom command: {e}")
            return False
    
    async def update_command_phrase(self, old_phrase: str, new_phrase: str, display_phrase: str) -> bool:
        """Update command phrase"""
        try:
            commands = await self.load_custom_commands()
            if old_phrase in commands:
                command_data = commands[old_phrase]
                # Update the command_key in the command object if it's an AutomationCommand
                if hasattr(command_data, 'command_key'):
                    command_data.command_key = new_phrase
                del commands[old_phrase]
                commands[new_phrase] = command_data
                return await self.save_custom_commands(commands)
            return False
        except Exception as e:
            logger.error(f"Failed to update command phrase: {e}")
            return False
    
    async def get_phrase_overrides(self) -> Dict[str, str]:
        """Get phrase overrides"""
        try:
            data = await read_commands(self._storage, {})
            return data.get('phrase_overrides', {})
        except Exception as e:
            logger.error(f"Failed to get phrase overrides: {e}")
            return {}

    async def save_phrase_overrides(self, phrase_overrides: Dict[str, str]) -> bool:
        """Save phrase overrides for default commands"""
        try:
            current_data = await read_commands(self._storage, {})
            current_data['phrase_overrides'] = phrase_overrides
            success = await write_commands(self._storage, current_data)
            if success:
                logger.debug(f"Saved {len(phrase_overrides)} phrase overrides")
            return success
        except Exception as e:
            logger.error(f"Failed to save phrase overrides: {e}")
            return False

    async def set_phrase_override(self, original_phrase: str, new_phrase: str) -> bool:
        """Set a single phrase override for a default command"""
        try:
            phrase_overrides = await self.get_phrase_overrides()
            phrase_overrides[original_phrase] = new_phrase
            return await self.save_phrase_overrides(phrase_overrides)
        except Exception as e:
            logger.error(f"Failed to set phrase override for '{original_phrase}': {e}")
            return False

    async def remove_phrase_override(self, original_phrase: str) -> bool:
        """Remove a phrase override for a default command"""
        try:
            phrase_overrides = await self.get_phrase_overrides()
            if original_phrase in phrase_overrides:
                del phrase_overrides[original_phrase]
                return await self.save_phrase_overrides(phrase_overrides)
            return True
        except Exception as e:
            logger.error(f"Failed to remove phrase override for '{original_phrase}': {e}")
            return False
    
    async def store_custom_command(self, command_phrase: str, command_data: Any) -> bool:
        """Store a single custom command (alias for add_custom_command)"""
        return await self.add_custom_command(command_phrase, command_data)
    
    async def reset_to_defaults(self) -> bool:
        """Reset to defaults"""
        try:
            return await write_commands(self._storage, {})
        except Exception as e:
            logger.error(f"Failed to reset commands: {e}")
            return False
    
    async def get_action_map(self) -> Dict[str, Any]:
        """
        Get action map for fast command lookup - compatibility method for CentralizedCommandParser
        
        Returns:
            Dict mapping normalized phrases to AutomationCommand objects
        """
        try:
            action_map = {}
            
            # Load custom commands
            custom_commands = await self.get_custom_commands()
            for normalized_phrase, command_data in custom_commands.items():
                action_map[normalized_phrase] = command_data
            
            # Load default commands from configuration registry
            default_commands = AutomationCommandRegistry.get_default_commands()
            
            for command_data in default_commands:
                normalized_phrase = command_data.command_key.lower().strip()
                
                # Only add if not overridden by custom command
                if normalized_phrase not in action_map:
                    # Apply any phrase overrides
                    phrase_overrides = await self.get_phrase_overrides()
                    effective_phrase = phrase_overrides.get(command_data.command_key, command_data.command_key)
                    
                    if effective_phrase != command_data.command_key:
                        # Create a copy with the effective phrase
                        from iris.config.command_types import AutomationCommand
                        command_data = AutomationCommand(
                            command_key=effective_phrase,
                            action_type=command_data.action_type,
                            action_value=command_data.action_value,
                            short_description=command_data.short_description,
                            long_description=command_data.long_description,
                            is_custom=command_data.is_custom
                        )
                        normalized_phrase = effective_phrase.lower().strip()
                    
                    action_map[normalized_phrase] = command_data
            
            logger.debug(f"Built action map with {len(action_map)} commands")
            return action_map
            
        except Exception as e:
            logger.error(f"Failed to build action map: {e}")
            return {}


class AgenticPromptStorageAdapter:
    """Adapter for agentic prompt storage"""
    
    def __init__(self, unified_storage: UnifiedStorageService):
        self._storage = unified_storage
    
    async def load_prompts(self) -> Dict[str, Any]:
        """Load agentic prompts"""
        try:
            data = await read_agentic_prompts(self._storage, {})
            logger.debug(f"Loaded agentic prompts data")
            return data
        except Exception as e:
            logger.error(f"Failed to load agentic prompts: {e}")
            return {}
    
    async def save_prompts(self, prompts_data: Dict[str, Any]) -> bool:
        """Save agentic prompts"""
        try:
            success = await write_agentic_prompts(self._storage, prompts_data)
            if success:
                logger.debug("Saved agentic prompts")
            return success
        except Exception as e:
            logger.error(f"Failed to save agentic prompts: {e}")
            return False


class SoundStorageAdapter:
    """Adapter for sound storage"""
    
    def __init__(self, unified_storage: UnifiedStorageService):
        self._storage = unified_storage
        self._config = unified_storage._config
    
    async def load_sound_mappings(self) -> Dict[str, str]:
        """Load sound command mappings"""
        try:
            mappings = await read_sound_mappings(self._storage, {})
            logger.debug(f"Loaded {len(mappings)} sound mappings")
            return mappings
        except Exception as e:
            logger.error(f"Failed to load sound mappings: {e}")
            return {}
    
    async def save_sound_mappings(self, mappings: Dict[str, str]) -> bool:
        """Save sound command mappings"""
        try:
            success = await write_sound_mappings(self._storage, mappings)
            if success:
                logger.debug(f"Saved {len(mappings)} sound mappings")
            return success
        except Exception as e:
            logger.error(f"Failed to save sound mappings: {e}")
            return False
    
    async def set_sound_mapping(self, sound_label: str, command: str) -> bool:
        """Set single sound mapping"""
        try:
            mappings = await self.load_sound_mappings()
            mappings[sound_label] = command
            return await self.save_sound_mappings(mappings)
        except Exception as e:
            logger.error(f"Failed to set sound mapping: {e}")
            return False
    
    async def get_sound_mapping(self, sound_label: str) -> Optional[str]:
        """Get single sound mapping"""
        try:
            mappings = await self.load_sound_mappings()
            return mappings.get(sound_label)
        except Exception as e:
            logger.error(f"Failed to get sound mapping: {e}")
            return None
    
    def get_sound_model_path(self) -> str:
        """Get the path for sound model storage"""
        return self._config.storage.sound_model_dir
    
    def get_sound_samples_path(self) -> str:
        """Get the path for sound samples storage"""
        return self._config.storage.sound_samples_dir
    
    def get_external_sounds_path(self) -> str:
        """Get the path for external non-target sounds (ESC-50)"""
        return self._config.storage.external_non_target_sounds_dir
    
    async def save_training_samples(self, sound_label: str, samples: List[Tuple[Any, int]]) -> bool:
        """Save training samples to the sound samples directory"""
        import os
        import soundfile as sf
        import asyncio
        
        try:
            # Create sound-specific directory
            sound_dir = os.path.join(self.get_sound_samples_path(), sound_label)
            os.makedirs(sound_dir, exist_ok=True)
            
            # Save each sample as a WAV file
            def save_sample(sample_data, sample_rate, sample_index):
                filename = f"{sound_label}_sample_{sample_index:03d}.wav"
                filepath = os.path.join(sound_dir, filename)
                sf.write(filepath, sample_data, sample_rate)
                return filepath
            
            # Run file operations in executor to avoid blocking
            loop = asyncio.get_event_loop()
            saved_files = []
            
            for i, (audio_data, sample_rate) in enumerate(samples):
                filepath = await loop.run_in_executor(
                    None, save_sample, audio_data, sample_rate, i
                )
                saved_files.append(filepath)
            
            logger.info(f"Saved {len(saved_files)} training samples for '{sound_label}' to {sound_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save training samples for '{sound_label}': {e}")
            return False
    
    async def load_training_samples(self, sound_label: str) -> List[Tuple[Any, int]]:
        """Load training samples for a sound from the sound samples directory"""
        import os
        import soundfile as sf
        import asyncio
        
        try:
            sound_dir = os.path.join(self.get_sound_samples_path(), sound_label)
            if not os.path.exists(sound_dir):
                logger.debug(f"No training samples directory found for '{sound_label}'")
                return []
            
            # Find all WAV files for this sound
            wav_files = [f for f in os.listdir(sound_dir) if f.endswith('.wav')]
            wav_files.sort()  # Ensure consistent ordering
            
            def load_sample(filepath):
                audio_data, sample_rate = sf.read(filepath)
                return audio_data, sample_rate
            
            # Load files in executor
            loop = asyncio.get_event_loop()
            samples = []
            
            for wav_file in wav_files:
                filepath = os.path.join(sound_dir, wav_file)
                try:
                    audio_data, sample_rate = await loop.run_in_executor(
                        None, load_sample, filepath
                    )
                    samples.append((audio_data, sample_rate))
                except Exception as e:
                    logger.warning(f"Failed to load sample {wav_file}: {e}")
            
            logger.debug(f"Loaded {len(samples)} training samples for '{sound_label}'")
            return samples
            
        except Exception as e:
            logger.error(f"Failed to load training samples for '{sound_label}': {e}")
            return []
    
    async def save_sound_model(self, model_data: Dict[str, Any]) -> bool:
        """Save sound model data to file"""
        import os
        import joblib
        import asyncio
        
        try:
            model_file_path = os.path.join(self.get_sound_model_path(), "sound_model.pkl")
            os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
            
            # Save in executor to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, joblib.dump, model_data, model_file_path)
            
            logger.debug(f"Saved sound model to {model_file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save sound model: {e}")
            return False
    
    async def load_sound_model(self) -> Optional[Dict[str, Any]]:
        """Load sound model data from file"""
        import os
        import joblib
        import asyncio
        
        try:
            model_file_path = os.path.join(self.get_sound_model_path(), "sound_model.pkl")
            
            if not os.path.exists(model_file_path):
                logger.debug("No sound model file found")
                return None
            
            # Load in executor to avoid blocking
            loop = asyncio.get_event_loop()
            model_data = await loop.run_in_executor(None, joblib.load, model_file_path)
            
            logger.debug(f"Loaded sound model from {model_file_path}")
            return model_data
            
        except Exception as e:
            logger.error(f"Failed to load sound model: {e}")
            return None


class MarkStorageAdapter:
    """Adapter for mark storage"""
    
    def __init__(self, unified_storage: UnifiedStorageService):
        self._storage = unified_storage
    
    async def load_marks(self) -> Dict[str, Tuple[int, int]]:
        """Load marks as dict mapping name -> (x, y) coordinates"""
        try:
            marks_data = await read_marks(self._storage, {})
            # Convert from storage format {"name": {"x": x, "y": y}} to service format {"name": (x, y)}
            marks = {}
            for name, coords in marks_data.items():
                if isinstance(coords, dict) and "x" in coords and "y" in coords:
                    marks[name] = (coords["x"], coords["y"])
                elif isinstance(coords, (list, tuple)) and len(coords) >= 2:
                    marks[name] = (coords[0], coords[1])
            logger.debug(f"Loaded {len(marks)} marks")
            return marks
        except Exception as e:
            logger.error(f"Failed to load marks: {e}")
            return {}
    
    async def save_marks(self, marks_data: Dict[str, Tuple[int, int]]) -> bool:
        """Save marks from service format to storage format"""
        try:
            # Convert from service format {"name": (x, y)} to storage format {"name": {"x": x, "y": y}}
            storage_data = {}
            for name, coords in marks_data.items():
                if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                    storage_data[name] = {"x": coords[0], "y": coords[1]}
            
            success = await write_marks(self._storage, storage_data)
            if success:
                logger.debug(f"Saved {len(marks_data)} marks")
            return success
        except Exception as e:
            logger.error(f"Failed to save marks: {e}")
            return False
    
    async def set_mark(self, name: str, x: int, y: int) -> bool:
        """Set/create a mark with coordinates"""
        try:
            marks = await self.load_marks()
            marks[name] = (x, y)
            return await self.save_marks(marks)
        except Exception as e:
            logger.error(f"Failed to set mark {name}: {e}")
            return False
    
    async def get_mark_coordinates(self, name: str) -> Optional[Tuple[int, int]]:
        """Get coordinates for a specific mark"""
        try:
            marks = await self.load_marks()
            return marks.get(name)
        except Exception as e:
            logger.error(f"Failed to get mark coordinates for {name}: {e}")
            return None
    
    async def remove_mark(self, name: str) -> bool:
        """Remove a mark"""
        try:
            marks = await self.load_marks()
            if name in marks:
                del marks[name]
                return await self.save_marks(marks)
            return True
        except Exception as e:
            logger.error(f"Failed to remove mark {name}: {e}")
            return False
    
    async def get_all_mark_names(self) -> Set[str]:
        """Get all mark names as a set"""
        try:
            marks = await self.load_marks()
            return set(marks.keys())
        except Exception as e:
            logger.error(f"Failed to get mark names: {e}")
            return set()
    
    async def create_mark(self, name: str, x: int, y: int) -> bool:
        """Create a new mark (alias for set_mark for compatibility)"""
        return await self.set_mark(name, x, y)
    
    async def delete_mark(self, name: str) -> bool:
        """Delete a mark (alias for remove_mark for compatibility)"""
        return await self.remove_mark(name)
    
    async def clear_all_marks(self) -> bool:
        """Clear all marks"""
        try:
            return await self.save_marks({})
        except Exception as e:
            logger.error(f"Failed to clear all marks: {e}")
            return False
    
    async def get_mark_names(self) -> List[str]:
        """Get all mark names as a list"""
        try:
            mark_names = await self.get_all_mark_names()
            return list(mark_names)
        except Exception as e:
            logger.error(f"Failed to get mark names: {e}")
            return []


class GridClickStorageAdapter:
    """Adapter for grid click storage"""
    
    def __init__(self, unified_storage: UnifiedStorageService):
        self._storage = unified_storage
    
    async def load_clicks(self) -> List[Dict[str, Any]]:
        """Load click history"""
        try:
            clicks = await read_grid_clicks(self._storage, [])
            logger.debug(f"Loaded {len(clicks)} clicks")
            return clicks
        except Exception as e:
            logger.error(f"Failed to load clicks: {e}")
            return []
    
    async def save_clicks(self, clicks_data: List[Dict[str, Any]]) -> bool:
        """Save click history"""
        try:
            success = await write_grid_clicks(self._storage, clicks_data)
            if success:
                logger.debug(f"Saved {len(clicks_data)} clicks")
            return success
        except Exception as e:
            logger.error(f"Failed to save clicks: {e}")
            return False
    
    async def append_click(self, click_data: Dict[str, Any]) -> bool:
        """Append a new click to the history"""
        try:
            clicks = await self.load_clicks()
            clicks.append(click_data)
            return await self.save_clicks(clicks)
        except Exception as e:
            logger.error(f"Failed to append click: {e}")
            return False
    
    async def clear_clicks(self) -> bool:
        """Clear all click history"""
        try:
            return await self.save_clicks([])
        except Exception as e:
            logger.error(f"Failed to clear clicks: {e}")
            return False
    
    async def get_recent_clicks(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent clicks with limit"""
        try:
            clicks = await self.load_clicks()
            return clicks[-limit:] if len(clicks) > limit else clicks
        except Exception as e:
            logger.error(f"Failed to get recent clicks: {e}")
            return []


class SoundRecognizerStorageAdapter:
    """Adapter wrapper for sound recognizer compatibility"""
    
    def __init__(self, unified_storage: UnifiedStorageService):
        self._sound_adapter = SoundStorageAdapter(unified_storage)
    
    def get_model_path(self) -> str:
        """Get the path for sound model storage"""
        return self._sound_adapter.get_sound_model_path()
    
    def get_samples_path(self) -> str:
        """Get the path for sound samples storage"""
        return self._sound_adapter.get_sound_samples_path()
    
    def get_external_sounds_path(self) -> str:
        """Get the path for external non-target sounds"""
        return self._sound_adapter.get_external_sounds_path()
    
    async def load_sound_mappings(self) -> Dict[str, str]:
        """Load sound command mappings"""
        return await self._sound_adapter.load_sound_mappings()
    
    async def save_sound_mappings(self, mappings: Dict[str, str]) -> bool:
        """Save sound command mappings"""
        return await self._sound_adapter.save_sound_mappings(mappings)
    
    async def set_sound_mapping(self, sound_label: str, command: str) -> bool:
        """Set single sound mapping"""
        return await self._sound_adapter.set_sound_mapping(sound_label, command)
    
    async def get_sound_mapping(self, sound_label: str) -> Optional[str]:
        """Get single sound mapping"""
        return await self._sound_adapter.get_sound_mapping(sound_label)
    
    async def save_training_samples(self, sound_label: str, samples: List[Tuple[Any, int]]) -> bool:
        """Save training samples"""
        return await self._sound_adapter.save_training_samples(sound_label, samples)
    
    async def save_sound_model(self, model_data: Dict[str, Any]) -> bool:
        """Save sound model data"""
        return await self._sound_adapter.save_sound_model(model_data)
    
    async def load_sound_model(self) -> Optional[Dict[str, Any]]:
        """Load sound model data"""
        return await self._sound_adapter.load_sound_model()
    
    def get_sound_model_file_path(self) -> str:
        """Get the full path to the sound model file"""
        import os
        return os.path.join(self._sound_adapter.get_sound_model_path(), "sound_model.pkl")


class StorageAdapterFactory:
    """Factory for creating storage adapters"""
    
    def __init__(self, unified_storage: UnifiedStorageService):
        self._unified_storage = unified_storage
        self._adapters = {}
    
    def get_settings_adapter(self) -> SettingsStorageAdapter:
        """Get settings adapter"""
        if 'settings' not in self._adapters:
            self._adapters['settings'] = SettingsStorageAdapter(self._unified_storage)
        return self._adapters['settings']
    
    def get_command_adapter(self) -> CommandStorageAdapter:
        """Get command adapter"""
        if 'commands' not in self._adapters:
            self._adapters['commands'] = CommandStorageAdapter(self._unified_storage)
        return self._adapters['commands']
    
    def get_agentic_prompt_adapter(self) -> AgenticPromptStorageAdapter:
        """Get agentic prompt adapter"""
        if 'agentic_prompts' not in self._adapters:
            self._adapters['agentic_prompts'] = AgenticPromptStorageAdapter(self._unified_storage)
        return self._adapters['agentic_prompts']
    
    def get_sound_adapter(self) -> SoundStorageAdapter:
        """Get sound adapter"""
        if 'sounds' not in self._adapters:
            self._adapters['sounds'] = SoundStorageAdapter(self._unified_storage)
        return self._adapters['sounds']
    
    def get_mark_adapter(self) -> MarkStorageAdapter:
        """Get mark adapter"""
        if 'marks' not in self._adapters:
            self._adapters['marks'] = MarkStorageAdapter(self._unified_storage)
        return self._adapters['marks']
    
    def get_grid_click_adapter(self) -> GridClickStorageAdapter:
        """Get grid click adapter"""
        if 'grid_clicks' not in self._adapters:
            self._adapters['grid_clicks'] = GridClickStorageAdapter(self._unified_storage)
        return self._adapters['grid_clicks']
    
    def create_sound_recognizer_adapter(self) -> 'SoundRecognizerStorageAdapter':
        """Create sound recognizer adapter (legacy method name)"""
        if 'sound_recognizer' not in self._adapters:
            self._adapters['sound_recognizer'] = SoundRecognizerStorageAdapter(self._unified_storage)
        return self._adapters['sound_recognizer']