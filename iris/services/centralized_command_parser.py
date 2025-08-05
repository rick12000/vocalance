"""
Centralized Command Parser Service

This service provides a single point of command parsing for all text input.
It subscribes to text recognition events, parses the text into specific command types,
and publishes appropriate command events that individual services can subscribe to.

Key features:
- Single centralized parsing logic
- No code duplication across services
- Standardized command typing with Pydantic models
- Event-driven architecture for better separation of concerns
- Comprehensive logging and error handling
- Support for all command categories: dictation, automation, mark, grid, sound
- Sound command mapping (maps CustomSoundRecognizedEvent to commands)
"""

import logging
import time
import pyautogui
from typing import Optional, Dict
import inspect

from iris.event_bus import EventBus
from iris.config.app_config import GlobalAppConfig
from iris.config.automation_command_registry import AutomationCommandRegistry
from iris.config.command_types import (
    # Base classes
    BaseCommand, ParseResultType, NoMatchResult, ErrorResult,
    # Command types
    DictationStartCommand, DictationStopCommand, DictationTypeCommand, DictationSmartStartCommand,
    ExactMatchCommand, ParameterizedCommand,
    MarkCreateCommand, MarkExecuteCommand, MarkDeleteCommand, 
    MarkVisualizeCommand, MarkResetCommand, MarkVisualizeCancelCommand,
    GridShowCommand, GridSelectCommand, GridCancelCommand,
    SoundTrainCommand, SoundDeleteCommand, SoundResetAllCommand, 
    SoundListAllCommand, SoundMapCommand,
    # Unions
    DictationCommandType, AutomationCommandType, MarkCommandType, 
    GridCommandType, SoundCommandType, AnyCommand
)
from iris.events.dictation_events import DictationStatusChangedEvent

# Import events
from iris.events.core_events import (
    StartRecordingCommand, StopRecordingCommand,
    CustomSoundRecognizedEvent, ProcessCommandPhraseEvent
)
from iris.events.stt_events import CommandTextRecognizedEvent
from iris.events.command_management_events import CommandMappingsUpdatedEvent
from iris.events.sound_events import (
    SoundToCommandMappingUpdatedEvent, 
    RequestSoundMappingsEvent, 
    SoundMappingsResponseEvent
)
from iris.events.command_events import (
    # Specific command events
    DictationCommandParsedEvent, AutomationCommandParsedEvent,
    MarkCommandParsedEvent, GridCommandParsedEvent, SoundCommandParsedEvent,
    # Meta events
    CommandNoMatchEvent, CommandParseErrorEvent
)

# Import utilities
from iris.utils.number_parser import parse_number

logger = logging.getLogger(__name__)


class CentralizedCommandParser:
    """
    Centralized command parser that handles all text-to-command parsing and sound command mapping.
    
    This service:
    1. Subscribes to text recognition events and sound recognition events
    2. Maps custom sounds to command phrases
    3. Parses text through a hierarchy of specialized parsers
    4. Creates appropriate command objects
    5. Publishes specific command events for services to consume
    """
    
    def __init__(self, event_bus: EventBus, app_config: GlobalAppConfig, command_storage_adapter):
        self._event_bus = event_bus
        self._app_config = app_config
        self._command_storage_adapter = command_storage_adapter
        
        # Track dictation state via events
        self._dictation_active = False
        
        # Sound command mapping
        self._sound_to_command_mapping: Dict[str, str] = {}
        
        # Cache configuration data for performance
        self._cache_config_data()
        
        # Performance tracking
        self._last_text = None
        self._last_text_time = 0.0
        self._duplicate_interval = 1.0
        
        logger.info("CentralizedCommandParser initialized")

    def _cache_config_data(self) -> None:
        """Cache frequently accessed configuration data"""
        # Grid configuration
        self._grid_show_phrase = self._app_config.grid.show_grid_phrase.lower()
        self._grid_cancel_phrase = self._app_config.grid.cancel_grid_phrase.lower()
        
        # Mark configuration
        self._mark_create_prefix = self._app_config.mark.triggers.create_mark.lower()
        self._mark_delete_prefix = self._app_config.mark.triggers.delete_mark.lower()
        self._mark_visualize_phrases = [p.lower() for p in self._app_config.mark.triggers.visualize_marks]
        self._mark_reset_phrases = [p.lower() for p in self._app_config.mark.triggers.reset_marks]
        self._mark_cancel_visualize_phrases = [p.lower() for p in self._app_config.mark.triggers.visualization_cancel]
        
        # Dictation configuration
        self._dictation_start_trigger = self._app_config.dictation.start_trigger.lower()
        self._dictation_stop_trigger = self._app_config.dictation.stop_trigger.lower()
        self._dictation_type_trigger = self._app_config.dictation.type_trigger.lower()
        self._dictation_smart_trigger = self._app_config.dictation.smart_start_trigger.lower()

    def setup_subscriptions(self) -> None:
        """Set up event subscriptions"""
        subscriptions = [
            (CommandTextRecognizedEvent, self._handle_command_text_recognized),
            (ProcessCommandPhraseEvent, self._handle_process_command_phrase),
            (CommandMappingsUpdatedEvent, self._handle_command_mappings_updated),
            (DictationStatusChangedEvent, self._handle_dictation_status_changed),
            (CustomSoundRecognizedEvent, self._handle_custom_sound_recognized),
            (SoundToCommandMappingUpdatedEvent, self._handle_sound_mapping_updated),
            (SoundMappingsResponseEvent, self._handle_sound_mappings_response),
        ]
        
        for event_type, handler in subscriptions:
            self._event_bus.subscribe(event_type, handler)
        
        logger.info("CentralizedCommandParser subscriptions set up")

    async def initialize(self) -> bool:
        """Initialize the service by requesting current sound mappings"""
        try:
            await self._event_bus.publish(RequestSoundMappingsEvent())
            return True
        except Exception as e:
            logger.error(f"Error initializing command parser: {e}", exc_info=True)
            return False

    # ============================================================================
    # SOUND COMMAND MAPPING METHODS
    # ============================================================================

    async def _handle_custom_sound_recognized(self, event_data: CustomSoundRecognizedEvent) -> None:
        """Handle custom sound recognition events and map to commands"""
        sound_label = event_data.label
        
        if sound_label in self._sound_to_command_mapping:
            command_phrase = self._sound_to_command_mapping[sound_label]
            logger.info(f"Sound '{sound_label}' mapped to command: '{command_phrase}'")
            await self._process_text_input(command_phrase, source=f"sound:{sound_label}")
        else:
            logger.warning(f"No command mapping found for sound: {sound_label}")

    async def _handle_sound_mapping_updated(self, event_data: SoundToCommandMappingUpdatedEvent) -> None:
        """Handle sound mapping updates"""
        self._sound_to_command_mapping[event_data.sound_label] = event_data.command_phrase
        logger.info(f"Updated sound mapping: '{event_data.sound_label}' -> '{event_data.command_phrase}'")

    async def _handle_sound_mappings_response(self, event_data: SoundMappingsResponseEvent) -> None:
        """Handle sound mappings response from sound service"""
        try:
            self._sound_to_command_mapping = event_data.mappings
            logger.info(f"Updated sound mappings with {len(self._sound_to_command_mapping)} entries")
        except Exception as e:
            logger.error(f"Error handling sound mappings response: {e}", exc_info=True)

    # ============================================================================
    # TEXT PROCESSING METHODS
    # ============================================================================

    async def _handle_command_text_recognized(self, event_data: CommandTextRecognizedEvent) -> None:
        """Handle command text from STT service"""
        await self._process_text_input(event_data.text, source="speech_command")

    async def _handle_process_command_phrase(self, event_data: ProcessCommandPhraseEvent) -> None:
        """Handle command phrase processing requests"""
        await self._process_text_input(event_data.phrase, source=event_data.source)

    async def _handle_command_mappings_updated(self, event_data) -> None:
        """Handle custom command mappings updates"""
        logger.info("Received command mappings update")

    async def _handle_dictation_status_changed(self, event_data) -> None:
        """Handle dictation status changes via events"""
        self._dictation_active = event_data.is_active
        logger.debug(f"Command parser updated dictation state: active={self._dictation_active}")

    async def _process_text_input(self, text: str, source: Optional[str] = None) -> None:
        """Process text input through the parsing pipeline"""
        # Duplicate detection
        current_time = time.time()
        if text == self._last_text and current_time - self._last_text_time < self._duplicate_interval:
            logger.debug(f"Suppressing duplicate text: '{text}'")
            return
        
        self._last_text = text
        self._last_text_time = current_time
        
        logger.info(f"Processing text input: '{text}' from source: {source}")
        
        # Check if dictation is active and suppress non-stop commands
        if self._dictation_active and text.lower().strip() != self._dictation_stop_trigger:
            logger.info(f"Dictation active - suppressing command processing for: '{text}'")
            return
        
        try:
            # Parse through the hierarchy of parsers
            parse_result = await self._parse_text(text)
            
            # Handle parse result
            if isinstance(parse_result, BaseCommand):
                await self._publish_command_event(parse_result, source)
            elif isinstance(parse_result, NoMatchResult):
                await self._event_bus.publish(CommandNoMatchEvent(
                    source=source,
                    attempted_parsers=["dictation", "mark", "grid", "automation"]
                ))
            elif isinstance(parse_result, ErrorResult):
                await self._event_bus.publish(CommandParseErrorEvent(
                    source=source,
                    error_message=parse_result.error_message,
                    attempted_parser="CentralizedCommandParser"
                ))
                
        except Exception as e:
            await self._event_bus.publish(CommandParseErrorEvent(
                source=source,
                error_message=f"Parser exception: {str(e)}",
                attempted_parser="CentralizedCommandParser"
            ))
            logger.error(f"Error parsing text '{text}': {e}", exc_info=True)

    async def _parse_text(self, text: str) -> ParseResultType:
        """
        Parse text through hierarchical parsers in priority order.
        """
        normalized_text = text.lower().strip()
        
        if not normalized_text:
            return NoMatchResult()
        
        # Parse in priority order
        parsers = [
            self._parse_dictation_commands,
            self._parse_mark_commands,
            self._parse_grid_commands,
            self._parse_automation_commands,
            self._parse_mark_execute_fallback
        ]
        
        for parser in parsers:
            if inspect.iscoroutinefunction(parser):
                result = await parser(normalized_text)
            else:
                result = parser(normalized_text)
            if not isinstance(result, NoMatchResult):
                return result
        
        return NoMatchResult()

    def _parse_dictation_commands(self, normalized_text: str) -> ParseResultType:
        """Parse dictation commands"""
        # Check for specific dictation triggers
        if normalized_text == self._dictation_start_trigger:
            return DictationStartCommand(trigger_type="start")
        
        if normalized_text == self._dictation_stop_trigger:
            return DictationStopCommand()
        
        if normalized_text == self._dictation_type_trigger:
            return DictationTypeCommand()
        
        if normalized_text == self._dictation_smart_trigger:
            return DictationSmartStartCommand()
        
        return NoMatchResult()

    def _parse_mark_commands(self, normalized_text: str) -> ParseResultType:
        """Parse mark commands"""
        words = normalized_text.split()
        
        if not words:
            return NoMatchResult()
        
        # Mark create command ("mark <label>")
        if words[0] == self._mark_create_prefix and len(words) == 2:
            label = words[1]
            if label:
                x, y = pyautogui.position()
                return MarkCreateCommand(label=label, x=float(x), y=float(y))
            else:
                return ErrorResult(error_message="Mark label cannot be empty")
        
        # Mark delete command ("mark delete <label>")
        if normalized_text.startswith(f"{self._mark_delete_prefix} "):
            label_part = normalized_text[len(self._mark_delete_prefix):].strip()
            if label_part and len(label_part.split()) == 1:
                return MarkDeleteCommand(label=label_part)
            else:
                return ErrorResult(error_message="Mark delete requires a single word label")
        
        # Mark visualize commands
        if normalized_text in self._mark_visualize_phrases:
            return MarkVisualizeCommand()
        
        # Mark reset commands
        if normalized_text in self._mark_reset_phrases:
            return MarkResetCommand()
        
        # Mark visualization cancel commands
        if normalized_text in self._mark_cancel_visualize_phrases:
            return MarkVisualizeCancelCommand()
        
        return NoMatchResult()

    async def _parse_grid_commands(self, normalized_text: str) -> ParseResultType:
        """Parse grid commands"""
        words = normalized_text.split()
        
        if not words:
            return NoMatchResult()
        
        # Check for grid trigger phrase
        if normalized_text.startswith(self._grid_show_phrase):
            # Grid show command with optional number
            if normalized_text == self._grid_show_phrase:
                return GridShowCommand(num_rects=None)
            
            # Extract number after trigger phrase
            after_trigger = normalized_text[len(self._grid_show_phrase):].strip()
            if after_trigger:
                parsed_num = parse_number(after_trigger)
                if parsed_num is not None and parsed_num > 0:
                    return GridShowCommand(num_rects=parsed_num)
                else:
                    return ErrorResult(error_message=f"Invalid number of rectangles: '{after_trigger}'")
        
        # Grid cancel command
        if normalized_text == self._grid_cancel_phrase:
            return GridCancelCommand()
        
        # Grid select command (numbers)
        action_map = await self._command_storage_adapter.get_action_map()
        
        # Check if first word or any prefix is a known automation command
        is_automation_prefix = False
        for i in range(1, len(words) + 1):
            potential_prefix = " ".join(words[:i])
            if potential_prefix in action_map:
                is_automation_prefix = True
                break
        
        if not is_automation_prefix:
            # Try to parse as a number
            parsed_num = parse_number(normalized_text)
            if parsed_num is not None and parsed_num > 0:
                return GridSelectCommand(selected_number=parsed_num)
        
        return NoMatchResult()

    async def _parse_automation_commands(self, normalized_text: str) -> ParseResultType:
        """Parse automation commands (exact match and parameterized)"""
        words = normalized_text.split()
        
        if not words:
            return NoMatchResult()
        
        # Get action map from storage adapter
        action_map = await self._command_storage_adapter.get_action_map()
        
        # 1. Try exact match first
        if normalized_text in action_map:
            command_data = action_map[normalized_text]
            return ExactMatchCommand(
                command_key=normalized_text,
                action_type=command_data.action_type,
                action_value=command_data.action_value,
                is_custom=command_data.is_custom,
                short_description=command_data.short_description,
                long_description=command_data.long_description
            )
        
        # 2. Try parameterized commands (command + number)
        for i in range(len(words) - 1, 0, -1):
            potential_command = " ".join(words[:i])
            
            if potential_command in action_map:
                remaining_words = words[i:]
                if len(remaining_words) == 1:
                    count = parse_number(remaining_words[0])
                    if count is not None and count > 0:
                        command_data = action_map[potential_command]
                        return ParameterizedCommand(
                            command_key=potential_command,
                            action_type=command_data.action_type,
                            action_value=command_data.action_value,
                            count=count,
                            is_custom=command_data.is_custom,
                            short_description=command_data.short_description,
                            long_description=command_data.long_description
                        )
                # If longest match found but can't parse parameter, stop looking
                break
        
        return NoMatchResult()

    def _parse_mark_execute_fallback(self, normalized_text: str) -> ParseResultType:
        """Parse mark execute commands as fallback for single words"""
        words = normalized_text.split()
        
        # Only consider single words as potential mark execute commands
        if len(words) == 1:
            return MarkExecuteCommand(label=normalized_text)
        
        return NoMatchResult()

    async def _publish_command_event(self, command: BaseCommand, source: Optional[str]) -> None:
        """Publish specific command events based on command type"""
        base_kwargs = {"source": source}
        
        # Map command types to event classes
        command_type_map = {
            DictationStartCommand: DictationCommandParsedEvent,
            DictationStopCommand: DictationCommandParsedEvent,
            DictationTypeCommand: DictationCommandParsedEvent,
            DictationSmartStartCommand: DictationCommandParsedEvent,
            
            ExactMatchCommand: AutomationCommandParsedEvent,
            ParameterizedCommand: AutomationCommandParsedEvent,
            
            MarkCreateCommand: MarkCommandParsedEvent,
            MarkExecuteCommand: MarkCommandParsedEvent,
            MarkDeleteCommand: MarkCommandParsedEvent,
            MarkVisualizeCommand: MarkCommandParsedEvent,
            MarkResetCommand: MarkCommandParsedEvent,
            MarkVisualizeCancelCommand: MarkCommandParsedEvent,
            
            GridShowCommand: GridCommandParsedEvent,
            GridSelectCommand: GridCommandParsedEvent,
            GridCancelCommand: GridCommandParsedEvent,
            
            SoundTrainCommand: SoundCommandParsedEvent,
            SoundDeleteCommand: SoundCommandParsedEvent,
            SoundResetAllCommand: SoundCommandParsedEvent,
            SoundListAllCommand: SoundCommandParsedEvent,
            SoundMapCommand: SoundCommandParsedEvent,
        }
        
        command_type = type(command)
        if command_type in command_type_map:
            event_class = command_type_map[command_type]
            event = event_class(**base_kwargs, command=command)
            await self._event_bus.publish(event)
        else:
            logger.warning(f"Unknown command type: {command_type}")

    async def shutdown(self) -> None:
        """Clean up resources during service shutdown"""
        logger.info("CentralizedCommandParser shutting down")
        # No specific cleanup needed for this service
        logger.info("CentralizedCommandParser shutdown complete") 
