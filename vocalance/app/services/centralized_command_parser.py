import inspect
import logging
import time
from typing import Dict, Optional, Tuple

import pyautogui

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.config.command_types import (
    BaseCommand,
    DictationSmartStartCommand,
    DictationStartCommand,
    DictationStopCommand,
    DictationTypeCommand,
    ErrorResult,
    ExactMatchCommand,
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
from vocalance.app.event_bus import EventBus
from vocalance.app.events.command_events import (
    AutomationCommandParsedEvent,
    CommandNoMatchEvent,
    CommandParseErrorEvent,
    DictationCommandParsedEvent,
    GridCommandParsedEvent,
    MarkCommandParsedEvent,
    SoundCommandParsedEvent,
)
from vocalance.app.events.command_management_events import CommandMappingsUpdatedEvent
from vocalance.app.events.core_events import (
    CommandTextRecognizedEvent,
    CustomSoundRecognizedEvent,
    MarkovPredictionEvent,
    MarkovPredictionFeedbackEvent,
    ProcessCommandPhraseEvent,
)
from vocalance.app.events.sound_events import (
    RequestSoundMappingsEvent,
    SoundMappingsResponseEvent,
    SoundToCommandMappingUpdatedEvent,
)
from vocalance.app.services.command_action_map_provider import CommandActionMapProvider
from vocalance.app.services.command_history_manager import CommandHistoryManager
from vocalance.app.services.storage.storage_service import StorageService
from vocalance.app.utils.number_parser import parse_number

logger = logging.getLogger(__name__)


class CentralizedCommandParser:
    """Centralized text-to-command parser with sound mapping and prediction deduplication.

    Parses voice/text input through hierarchical command parsers (dictation, mark, grid,
    automation), maps custom sounds to commands, handles Markov prediction deduplication,
    and maintains command history for prediction training.
    """

    def __init__(
        self,
        event_bus: EventBus,
        app_config: GlobalAppConfig,
        storage: StorageService,
        action_map_provider: CommandActionMapProvider,
        history_manager: CommandHistoryManager,
    ) -> None:
        self._event_bus: EventBus = event_bus
        self._app_config: GlobalAppConfig = app_config
        self._storage: StorageService = storage
        self._action_map_provider: CommandActionMapProvider = action_map_provider
        self._history_manager: CommandHistoryManager = history_manager
        self._sound_to_command_mapping: Dict[str, str] = {}
        self._cache_config_data()
        self._last_text: Optional[str] = None
        self._last_text_time: float = 0.0
        self._duplicate_interval: float = app_config.command_parser.duplicate_detection_interval
        self._recent_predictions: Dict[float, Tuple[str, float]] = {}
        self._prediction_window: float = app_config.command_parser.prediction_deduplication_window

        logger.debug("CentralizedCommandParser initialized")

    def _cache_config_data(self) -> None:
        """Cache frequently accessed configuration values."""
        self._grid_show_phrase = self._app_config.grid.show_grid_phrase.lower()
        self._mark_create_prefix = self._app_config.mark.triggers.create_mark.lower()
        self._mark_delete_prefix = self._app_config.mark.triggers.delete_mark.lower()
        self._mark_visualize_phrases = [p.lower() for p in self._app_config.mark.triggers.visualize_marks]
        self._mark_reset_phrases = [p.lower() for p in self._app_config.mark.triggers.reset_marks]
        self._mark_cancel_visualize_phrases = [p.lower() for p in self._app_config.mark.triggers.visualization_cancel]
        self._dictation_start_trigger = self._app_config.dictation.start_trigger.lower()
        self._dictation_stop_trigger = self._app_config.dictation.stop_trigger.lower()
        self._dictation_type_trigger = self._app_config.dictation.type_trigger.lower()
        self._dictation_smart_trigger = self._app_config.dictation.smart_start_trigger.lower()

    def setup_subscriptions(self) -> None:
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
        """Initialize command parser by loading action map and setting up subscriptions.

        Returns:
            True if initialization succeeded, False otherwise.
        """
        try:
            await self._event_bus.publish(RequestSoundMappingsEvent())
            await self._history_manager.initialize()
            return True
        except Exception as e:
            logger.error(f"Error initializing command parser: {e}", exc_info=True)
            return False

    async def _handle_custom_sound_recognized(self, event_data: CustomSoundRecognizedEvent) -> None:
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
        self._sound_to_command_mapping = event_data.mappings
        logger.info(f"Updated sound mappings with {len(self._sound_to_command_mapping)} entries")

    # ============================================================================
    # TEXT PROCESSING METHODS
    # ============================================================================

    async def _handle_process_command_phrase(self, event_data: ProcessCommandPhraseEvent) -> None:
        """Handle command phrase processing requests"""
        await self._process_text_input(event_data.phrase, source=event_data.source)

    async def _handle_command_mappings_updated(self, event_data) -> None:
        """Handle custom command mappings updates"""
        logger.info("Received command mappings update")

    async def _is_valid_command(self, text: str) -> bool:
        """Check if text is a valid command without executing it.

        Args:
            text: Text to validate

        Returns:
            True if text parses to a valid command, False otherwise
        """
        parse_result = await self._parse_text(text)
        return isinstance(parse_result, BaseCommand)

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

        parse_result = await self._parse_text(text)

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

    async def _process_text_input_with_history(self, text: str, source: Optional[str] = None) -> None:
        """Process text input and record to history only if it's a valid command.

        Args:
            text: Text to process
            source: Source of the text (stt, sound, markov, etc.)
        """
        # Duplicate detection
        current_time = time.time()
        if text == self._last_text and current_time - self._last_text_time < self._duplicate_interval:
            logger.debug(f"Suppressing duplicate text: '{text}'")
            return

        self._last_text = text
        self._last_text_time = current_time

        logger.info(f"Processing text input: '{text}' from source: {source}")

        parse_result = await self._parse_text(text)

        if isinstance(parse_result, BaseCommand):
            # Valid command - record to history and execute
            await self._history_manager.record_command(command=text, source=source)
            await self._publish_command_event(parse_result, source)
            logger.debug(f"Recorded valid command to history: '{text}' (source={source})")
        elif isinstance(parse_result, NoMatchResult):
            logger.debug(f"Not recording to history - no match: '{text}' (source={source})")
            await self._event_bus.publish(
                CommandNoMatchEvent(source=source, attempted_parsers=["dictation", "mark", "grid", "automation"])
            )
        elif isinstance(parse_result, ErrorResult):
            logger.debug(f"Not recording to history - parse error: '{text}' (source={source})")
            await self._event_bus.publish(
                CommandParseErrorEvent(
                    source=source, error_message=parse_result.error_message, attempted_parser="CentralizedCommandParser"
                )
            )

    async def _parse_text(self, text: str) -> ParseResultType:
        """Parse text through hierarchical command parsers in priority order.

        Attempts to match text against dictation, mark, grid, automation, and mark
        execute commands in order, returning the first successful match.

        Args:
            text: Normalized lowercase text to parse

        Returns:
            Parsed command object, NoMatchResult, or ErrorResult
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

        if words[0] == self._mark_create_prefix and len(words) == 2:
            label = words[1]
            if label:
                x, y = pyautogui.position()
                return MarkCreateCommand(label=label, x=float(x), y=float(y))
            else:
                return ErrorResult(error_message="Mark label cannot be empty")

        if normalized_text.startswith(f"{self._mark_delete_prefix} "):
            label_part = normalized_text[len(self._mark_delete_prefix) :].strip()
            if label_part and len(label_part.split()) == 1:
                return MarkDeleteCommand(label=label_part)
            else:
                return ErrorResult(error_message="Mark delete requires a single word label")

        if normalized_text in self._mark_visualize_phrases:
            return MarkVisualizeCommand()

        if normalized_text in self._mark_reset_phrases:
            return MarkResetCommand()

        if normalized_text in self._mark_cancel_visualize_phrases:
            return MarkVisualizeCancelCommand()

        return NoMatchResult()

    async def _parse_grid_commands(self, normalized_text: str) -> ParseResultType:
        """Parse grid commands"""
        words = normalized_text.split()

        if not words:
            return NoMatchResult()

        if normalized_text.startswith(self._grid_show_phrase):
            if normalized_text == self._grid_show_phrase:
                return GridShowCommand(num_rects=None)

            after_trigger = normalized_text[len(self._grid_show_phrase) :].strip()
            if after_trigger:
                parsed_num = parse_number(text=after_trigger)
                if parsed_num is not None and parsed_num > 0:
                    return GridShowCommand(num_rects=parsed_num)
                else:
                    return ErrorResult(error_message=f"Invalid number of rectangles: '{after_trigger}'")

        action_map = await self._action_map_provider.get_action_map()

        is_automation_prefix = False
        for i in range(1, len(words) + 1):
            potential_prefix = " ".join(words[:i])
            if potential_prefix in action_map:
                is_automation_prefix = True
                break

        if not is_automation_prefix:
            parsed_num = parse_number(text=normalized_text)
            if parsed_num is not None and parsed_num > 0:
                return GridSelectCommand(selected_number=parsed_num)

        return NoMatchResult()

    async def _parse_automation_commands(self, normalized_text: str) -> ParseResultType:
        """Parse automation commands (exact match and parameterized)"""
        words = normalized_text.split()

        if not words:
            return NoMatchResult()

        action_map = await self._action_map_provider.get_action_map()

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
        """Handle Markov predictions by executing immediately and storing for deduplication.

        Executes predicted commands before STT completes for ultra-low latency,
        stores predictions in deduplication window for comparison with actual STT results.

        Args:
            event: Markov prediction event with predicted command and confidence
        """
        predicted_command = event.predicted_command
        current_time = time.time()

        logger.info(f"Markov prediction received: '{predicted_command}' (confidence={event.confidence:.2%})")

        self._recent_predictions[current_time] = (predicted_command, current_time)
        self._clean_old_predictions(current_time)

        await self._process_text_input(text=predicted_command, source="markov")

    def _clean_old_predictions(self, current_time: float) -> None:
        """Remove predictions outside the deduplication window."""
        cutoff = current_time - self._prediction_window
        to_remove = [ts for ts in self._recent_predictions if ts < cutoff]
        for ts in to_remove:
            del self._recent_predictions[ts]

    def _find_recent_prediction(self, cmd_time: float) -> Optional[Tuple[str, float]]:
        """Find recent prediction within deduplication window.

        Args:
            cmd_time: Timestamp of command recognition

        Returns:
            Tuple of (predicted_command, prediction_time) if found, None otherwise
        """
        for pred_time, (pred_cmd, _) in self._recent_predictions.items():
            if cmd_time - pred_time < self._prediction_window:
                return (pred_cmd, pred_time)
        return None

    # ============================================================================
    # UNIFIED COMMAND PROCESSING (STT + SOUND)
    # ============================================================================

    async def _process_recognized_command(self, command_text: str, source: str, timestamp: float) -> None:
        """Process recognized commands with Markov deduplication and history tracking.

        Checks for recent Markov predictions to avoid duplicate execution, sends
        prediction feedback, records ONLY VALID commands to history, and executes if not duplicate.

        Args:
            command_text: The final command text to process
            source: Source of recognition ("stt" or "sound")
            timestamp: When the command was recognized
        """
        recent_pred = self._find_recent_prediction(timestamp)

        if recent_pred:
            predicted_cmd, pred_time = recent_pred
            was_correct = command_text.lower().strip() == predicted_cmd.lower().strip()

            await self._send_markov_feedback(predicted=predicted_cmd, actual=command_text, was_correct=was_correct, source=source)

            # Only record if command is valid (will be checked in _process_text_input_with_history)
            is_valid = await self._is_valid_command(command_text)
            if is_valid:
                await self._history_manager.record_command(command=command_text, source=source)
                logger.info(
                    f"Skipping duplicate: Markov predicted '{predicted_cmd}', "
                    f"{source} recognized '{command_text}' "
                    f"({'CORRECT' if was_correct else 'INCORRECT'}) - recorded to history"
                )
            else:
                logger.info(
                    f"Skipping duplicate: Markov predicted '{predicted_cmd}', "
                    f"{source} recognized '{command_text}' "
                    f"({'CORRECT' if was_correct else 'INCORRECT'}) - NOT recorded (invalid command)"
                )

            if pred_time in self._recent_predictions:
                del self._recent_predictions[pred_time]

            return

        logger.debug(f"No recent Markov prediction found for {source} command: '{command_text}'")

        # Process and record only if valid
        await self._process_text_input_with_history(text=command_text, source=source)

    async def _send_markov_feedback(self, predicted: str, actual: str, was_correct: bool, source: str) -> None:
        """Send feedback to Markov predictor about prediction accuracy.

        Args:
            predicted: The predicted command
            actual: The actual recognized command
            was_correct: Whether prediction matched actual
            source: Source of actual command ("stt" or "sound")
        """
        feedback = MarkovPredictionFeedbackEvent(
            predicted_command=predicted, actual_command=actual, was_correct=was_correct, source=source
        )
        await self._event_bus.publish(feedback)

        logger.info(
            f"Markov feedback: {source} '{actual}' vs prediction '{predicted}' - " f"{'CORRECT' if was_correct else 'INCORRECT'}"
        )

    async def shutdown(self) -> None:
        """Shutdown parser and persist accumulated command history to storage.

        Writes all in-memory session command history to storage for future
        Markov training, ensuring no command data is lost.
        """
        await self._history_manager.shutdown()
        logger.info("CentralizedCommandParser shutdown complete")
