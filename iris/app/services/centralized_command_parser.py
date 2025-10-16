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

import inspect
import logging
import time
from typing import Dict, List, Optional, Tuple

import pyautogui

from iris.app.config.app_config import GlobalAppConfig
from iris.app.config.automation_command_registry import AutomationCommandRegistry
from iris.app.config.command_types import (  # Base classes; Command types; Unions
    BaseCommand,
    DictationSmartStartCommand,
    DictationStartCommand,
    DictationStopCommand,
    DictationTypeCommand,
    ErrorResult,
    ExactMatchCommand,
    GridCancelCommand,
    GridSelectCommand,
    GridShowCommand,
    MarkCreateCommand,
    MarkDeleteCommand,
    MarkExecuteCommand,
    MarkResetCommand,
    MarkVisualizeCancelCommand,
    MarkVisualizeCommand,
    NoMatchResult,
    ParameterizedCommand,
    ParseResultType,
    SoundDeleteCommand,
    SoundListAllCommand,
    SoundMapCommand,
    SoundResetAllCommand,
    SoundTrainCommand,
)
from iris.app.event_bus import EventBus
from iris.app.events.command_events import (  # Specific command events; Meta events
    AutomationCommandParsedEvent,
    CommandNoMatchEvent,
    CommandParseErrorEvent,
    DictationCommandParsedEvent,
    GridCommandParsedEvent,
    MarkCommandParsedEvent,
    SoundCommandParsedEvent,
)
from iris.app.events.command_management_events import CommandMappingsUpdatedEvent

# Import events
from iris.app.events.core_events import (
    CommandTextRecognizedEvent,
    CustomSoundRecognizedEvent,
    MarkovPredictionEvent,
    MarkovPredictionFeedbackEvent,
    ProcessCommandPhraseEvent,
)
from iris.app.events.sound_events import RequestSoundMappingsEvent, SoundMappingsResponseEvent, SoundToCommandMappingUpdatedEvent
from iris.app.services.storage.storage_models import CommandHistoryData, CommandHistoryEntry, CommandsData
from iris.app.services.storage.storage_service import StorageService

# Import utilities
from iris.app.utils.number_parser import parse_number

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

    def __init__(self, event_bus: EventBus, app_config: GlobalAppConfig, storage: StorageService):
        self._event_bus = event_bus
        self._app_config = app_config
        self._storage = storage

        # Sound command mapping
        self._sound_to_command_mapping: Dict[str, str] = {}

        # Cache configuration data for performance
        self._cache_config_data()

        # Duplicate detection
        self._last_text = None
        self._last_text_time = 0.0
        self._duplicate_interval = 1.0

        # In-memory command history (fast, no I/O during commands)
        # Accumulated throughout session, written once at shutdown
        self._session_command_history: List[CommandHistoryEntry] = []

        # Markov prediction deduplication
        self._recent_predictions: Dict[float, Tuple[str, float]] = {}
        self._prediction_window = 0.5  # 500ms deduplication window

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

    async def _get_action_map(self) -> Dict[str, any]:
        """Get action map for command lookup."""
        from iris.app.config.command_types import AutomationCommand

        action_map = {}

        # Load custom commands (already AutomationCommand objects)
        commands_data = await self._storage.read(model_type=CommandsData)
        for normalized_phrase, command_obj in commands_data.custom_commands.items():
            action_map[normalized_phrase] = command_obj

        # Load default commands
        default_commands = AutomationCommandRegistry.get_default_commands()

        for command_data in default_commands:
            normalized_phrase = command_data.command_key.lower().strip()

            if normalized_phrase not in action_map:
                # Apply any phrase overrides
                effective_phrase = commands_data.phrase_overrides.get(command_data.command_key, command_data.command_key)

                if effective_phrase != command_data.command_key:
                    command_data = AutomationCommand(
                        command_key=effective_phrase,
                        action_type=command_data.action_type,
                        action_value=command_data.action_value,
                        short_description=command_data.short_description,
                        long_description=command_data.long_description,
                        is_custom=command_data.is_custom,
                    )
                    normalized_phrase = effective_phrase.lower().strip()

                action_map[normalized_phrase] = command_data

        return action_map

    def setup_subscriptions(self) -> None:
        """Set up event subscriptions"""
        subscriptions = [
            (CommandTextRecognizedEvent, self._handle_command_text_recognized),
            (ProcessCommandPhraseEvent, self._handle_process_command_phrase),
            (CommandMappingsUpdatedEvent, self._handle_command_mappings_updated),
            (CustomSoundRecognizedEvent, self._handle_custom_sound_recognized),
            (SoundToCommandMappingUpdatedEvent, self._handle_sound_mapping_updated),
            (SoundMappingsResponseEvent, self._handle_sound_mappings_response),
            (MarkovPredictionEvent, self._handle_markov_prediction),
        ]

        for event_type, handler in subscriptions:
            self._event_bus.subscribe(event_type=event_type, handler=handler)

        logger.info("CentralizedCommandParser subscriptions set up")

    async def initialize(self) -> bool:
        """Initialize the service by requesting current sound mappings and loading existing history"""
        try:
            await self._event_bus.publish(RequestSoundMappingsEvent())

            # Load existing command history from storage into memory
            try:
                history_data = await self._storage.read(model_type=CommandHistoryData)
                self._session_command_history = list(history_data.history)
                logger.info(f"Loaded {len(self._session_command_history)} existing commands from history")
            except Exception as e:
                logger.warning(f"Could not load existing history (starting fresh): {e}")
                self._session_command_history = []

            return True
        except Exception as e:
            logger.error(f"Error initializing command parser: {e}", exc_info=True)
            return False

    # ============================================================================
    # SOUND COMMAND MAPPING METHODS
    # ============================================================================

    async def _handle_custom_sound_recognized(self, event_data: CustomSoundRecognizedEvent) -> None:
        """Handle custom sound recognition events - route to unified processing"""
        sound_label = event_data.label

        # Get mapped command (use event's mapped_command if available, otherwise check our mapping)
        command_text = None
        if event_data.mapped_command:
            command_text = event_data.mapped_command
        elif sound_label in self._sound_to_command_mapping:
            command_text = self._sound_to_command_mapping[sound_label]

        if command_text:
            logger.info(f"Sound '{sound_label}' → command: '{command_text}'")
            await self._process_recognized_command(command_text=command_text, source="sound", timestamp=time.time())
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

    async def _handle_process_command_phrase(self, event_data: ProcessCommandPhraseEvent) -> None:
        """Handle command phrase processing requests"""
        await self._process_text_input(event_data.phrase, source=event_data.source)

    async def _handle_command_mappings_updated(self, event_data) -> None:
        """Handle custom command mappings updates"""
        logger.info("Received command mappings update")

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

        try:
            # Parse through the hierarchy of parsers
            parse_result = await self._parse_text(text)

            # Handle parse result
            if isinstance(parse_result, BaseCommand):
                await self._publish_command_event(parse_result, source)
            elif isinstance(parse_result, NoMatchResult):
                await self._event_bus.publish(
                    CommandNoMatchEvent(source=source, attempted_parsers=["dictation", "mark", "grid", "automation"])
                )
            elif isinstance(parse_result, ErrorResult):
                await self._event_bus.publish(
                    CommandParseErrorEvent(
                        source=source, error_message=parse_result.error_message, attempted_parser="CentralizedCommandParser"
                    )
                )

        except Exception as e:
            await self._event_bus.publish(
                CommandParseErrorEvent(
                    source=source, error_message=f"Parser exception: {str(e)}", attempted_parser="CentralizedCommandParser"
                )
            )
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
            self._parse_mark_execute_fallback,
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
            return DictationStartCommand()

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
            label_part = normalized_text[len(self._mark_delete_prefix) :].strip()
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
            after_trigger = normalized_text[len(self._grid_show_phrase) :].strip()
            if after_trigger:
                parsed_num = parse_number(text=after_trigger)
                if parsed_num is not None and parsed_num > 0:
                    return GridShowCommand(num_rects=parsed_num)
                else:
                    return ErrorResult(error_message=f"Invalid number of rectangles: '{after_trigger}'")

        # Grid cancel command
        if normalized_text == self._grid_cancel_phrase:
            return GridCancelCommand()

        # Grid select command (numbers)
        action_map = await self._get_action_map()

        # Check if first word or any prefix is a known automation command
        is_automation_prefix = False
        for i in range(1, len(words) + 1):
            potential_prefix = " ".join(words[:i])
            if potential_prefix in action_map:
                is_automation_prefix = True
                break

        if not is_automation_prefix:
            # Try to parse as a number
            parsed_num = parse_number(text=normalized_text)
            if parsed_num is not None and parsed_num > 0:
                return GridSelectCommand(selected_number=parsed_num)

        return NoMatchResult()

    async def _parse_automation_commands(self, normalized_text: str) -> ParseResultType:
        """Parse automation commands (exact match and parameterized)"""
        words = normalized_text.split()

        if not words:
            return NoMatchResult()

        # Get action map from storage
        action_map = await self._get_action_map()

        # 1. Try exact match first
        if normalized_text in action_map:
            command_data = action_map[normalized_text]
            return ExactMatchCommand(
                command_key=normalized_text,
                action_type=command_data.action_type,
                action_value=command_data.action_value,
                is_custom=command_data.is_custom,
                short_description=command_data.short_description,
                long_description=command_data.long_description,
            )

        # 2. Try parameterized commands (command + number)
        for i in range(len(words) - 1, 0, -1):
            potential_command = " ".join(words[:i])

            if potential_command in action_map:
                remaining_words = words[i:]
                if len(remaining_words) == 1:
                    count = parse_number(text=remaining_words[0])
                    if count is not None and count > 0:
                        command_data = action_map[potential_command]
                        return ParameterizedCommand(
                            command_key=potential_command,
                            action_type=command_data.action_type,
                            action_value=command_data.action_value,
                            count=count,
                            is_custom=command_data.is_custom,
                            short_description=command_data.short_description,
                            long_description=command_data.long_description,
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

    # ============================================================================
    # EVENT HANDLERS - ROUTE TO UNIFIED PROCESSING
    # ============================================================================

    async def _handle_command_text_recognized(self, event: CommandTextRecognizedEvent) -> None:
        """Handle STT recognized text - route to unified processing"""
        command_text = event.text
        await self._process_recognized_command(command_text=command_text, source="stt", timestamp=time.time())

    # ============================================================================
    # MARKOV PREDICTION HANDLING
    # ============================================================================

    async def _handle_markov_prediction(self, event: MarkovPredictionEvent) -> None:
        """
        Handle Markov prediction events - execute immediately and store for deduplication.
        This provides ultra-fast command execution before STT completes.
        """
        try:
            predicted_command = event.predicted_command
            current_time = time.time()

            logger.info(f"Markov prediction received: '{predicted_command}' (confidence={event.confidence:.2%})")

            # Store prediction for deduplication (keyed by timestamp)
            self._recent_predictions[current_time] = (predicted_command, current_time)

            # Clean old predictions (outside deduplication window)
            self._clean_old_predictions(current_time)

            # Parse and execute the predicted command immediately
            await self._process_text_input(text=predicted_command, source="markov")

        except Exception as e:
            logger.error(f"Error handling Markov prediction: {e}", exc_info=True)

    def _clean_old_predictions(self, current_time: float) -> None:
        """Remove predictions outside the deduplication window"""
        cutoff_time = current_time - self._prediction_window
        keys_to_remove = [ts for ts in self._recent_predictions.keys() if ts < cutoff_time]
        for key in keys_to_remove:
            del self._recent_predictions[key]

    def _find_recent_prediction(self, command_time: float) -> Optional[Tuple[str, float]]:
        """
        Find a recent Markov prediction within the deduplication window.

        Returns:
            (predicted_command, prediction_time) if found, None otherwise
        """
        # Check predictions within the deduplication window
        for pred_time, (predicted_cmd, _) in self._recent_predictions.items():
            if command_time - pred_time < self._prediction_window:
                return (predicted_cmd, pred_time)

        return None

    # ============================================================================
    # UNIFIED COMMAND PROCESSING (STT + SOUND)
    # ============================================================================

    async def _process_recognized_command(self, command_text: str, source: str, timestamp: float) -> None:
        """
        Unified processing for STT and Sound recognized commands.
        Handles deduplication, feedback, history tracking, and execution.

        Args:
            command_text: The final command text to process
            source: "stt" or "sound"
            timestamp: When the command was recognized
        """
        try:
            # Check for recent Markov prediction
            recent_prediction = self._find_recent_prediction(timestamp)

            if recent_prediction:
                predicted_cmd, pred_time = recent_prediction

                # Send feedback to Markov predictor
                was_correct = command_text.lower().strip() == predicted_cmd.lower().strip()
                await self._send_markov_feedback(
                    predicted=predicted_cmd, actual=command_text, was_correct=was_correct, source=source
                )

                # Always record actual command to history (not the prediction)
                self._record_command_to_history(command=command_text, source=source)

                # Skip execution (Markov already executed)
                logger.info(
                    f"Skipping duplicate execution: Markov predicted '{predicted_cmd}', "
                    f"{source} recognized '{command_text}' "
                    f"({'CORRECT' if was_correct else 'INCORRECT'})"
                )

                # Remove the prediction from tracking (processed)
                if pred_time in self._recent_predictions:
                    del self._recent_predictions[pred_time]

                return  # DONE - don't execute again

            # No recent prediction - normal execution path
            logger.debug(f"No recent Markov prediction found for {source} command: '{command_text}'")

            # Parse and execute command normally
            await self._process_text_input(text=command_text, source=source)

            # Record to history
            self._record_command_to_history(command=command_text, source=source)

        except Exception as e:
            logger.error(f"Error processing recognized command from {source}: {e}", exc_info=True)

    async def _send_markov_feedback(self, predicted: str, actual: str, was_correct: bool, source: str) -> None:
        """Send feedback to Markov predictor about prediction accuracy"""
        try:
            feedback_event = MarkovPredictionFeedbackEvent(
                predicted_command=predicted, actual_command=actual, was_correct=was_correct, source=source
            )
            await self._event_bus.publish(feedback_event)

            logger.info(
                f"Markov feedback sent: {source} command '{actual}' vs prediction '{predicted}' - {'CORRECT' if was_correct else 'INCORRECT'}"
            )

        except Exception as e:
            logger.error(f"Error sending Markov feedback: {e}", exc_info=True)

    # ============================================================================
    # COMMAND HISTORY TRACKING
    # ============================================================================

    def _record_command_to_history(self, command: str, source: str) -> None:
        """
        Record a command to the in-memory session history (fast, no I/O).
        Only records actual commands (STT, Sound), NOT Markov predictions.
        History is written to storage once at shutdown.
        """
        try:
            entry = CommandHistoryEntry(command=command, timestamp=time.time(), success=None, metadata={"source": source})

            self._session_command_history.append(entry)
            logger.debug(
                f"Recorded to in-memory history: '{command}' (source={source}, total={len(self._session_command_history)})"
            )

        except Exception as e:
            logger.error(f"Error recording command to history: {e}", exc_info=True)

    async def shutdown(self) -> None:
        """Shutdown the parser and write all session history to storage"""
        try:
            if self._session_command_history:
                logger.info(f"Writing {len(self._session_command_history)} commands to storage at shutdown")

                # Create history data with all accumulated session commands
                history_data = CommandHistoryData(history=self._session_command_history)

                # Write to storage (overwrites with full history)
                success = await self._storage.write(data=history_data)

                if success:
                    logger.info(f"Successfully wrote command history ({len(self._session_command_history)} commands)")
                else:
                    logger.error("Failed to write command history to storage")
            else:
                logger.info("No commands to write at shutdown")

            logger.info("CentralizedCommandParser shutdown complete")
        except Exception as e:
            logger.error(f"Error during parser shutdown: {e}", exc_info=True)
