import asyncio
import gc
import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.config.command_types import (
    DictationHiddenStartCommand,
    DictationSmartStartCommand,
    DictationStartCommand,
    DictationStopCommand,
    DictationTypeCommand,
    DictationVisualStartCommand,
)
from vocalance.app.event_bus import EventBus
from vocalance.app.events.base_event import BaseEvent
from vocalance.app.events.command_events import DictationCommandParsedEvent
from vocalance.app.events.core_events import AudioChunkEvent, DictationTextRecognizedEvent
from vocalance.app.events.dictation_events import (
    AudioModeChangeRequestEvent,
    DictationModeDisableOthersEvent,
    DictationStatusChangedEvent,
    FinalDictationTextEvent,
    HiddenDictationStartedEvent,
    HiddenDictationStoppedEvent,
    LLMProcessingCompletedEvent,
    LLMProcessingFailedEvent,
    LLMProcessingReadyEvent,
    LLMProcessingStartedEvent,
    LLMTokenGeneratedEvent,
    PartialDictationTextEvent,
    SmartDictationRemoveCharactersEvent,
    SmartDictationStartedEvent,
    SmartDictationStoppedEvent,
    SmartDictationTextDisplayEvent,
    VisualDictationStartedEvent,
    VisualDictationStoppedEvent,
)
from vocalance.app.services.audio.dictation_handling.dictation_alias_service import DictationAliasService
from vocalance.app.services.audio.dictation_handling.llm_support.agentic_prompt_service import AgenticPromptService
from vocalance.app.services.audio.dictation_handling.llm_support.llm_service import LLMService
from vocalance.app.services.audio.dictation_handling.text_input_service import (
    TextInputService,
    clean_dictation_text,
    get_trailing_whitespace_count,
    lowercase_first_letter,
    remove_formatting,
    should_lowercase_current_start,
    should_remove_previous_period,
)
from vocalance.app.services.audio.streaming_audio_buffer import StreamingAudioBuffer
from vocalance.app.services.storage.storage_service import StorageService
from vocalance.app.utils.event_utils import EventSubscriptionManager, ThreadSafeEventPublisher

logger = logging.getLogger(__name__)


class DictationMode(Enum):
    """Dictation modes for different recognition and processing behavior.

    INACTIVE: No dictation active.
    STANDARD: Direct transcription without LLM processing.
    SMART: LLM-enhanced dictation with formatting and editing.
    TYPE: Direct typing of recognized text without formatting.
    VISUAL: Accumulated dictation with UI display but no LLM processing.
    HIDDEN: Silent accumulation without UI display, pastes on stop.
    """

    INACTIVE = "inactive"
    STANDARD = "standard"
    SMART = "smart"
    TYPE = "type"
    VISUAL = "visual"
    HIDDEN = "hidden"


class DictationState(Enum):
    """Explicit state machine for dictation coordinator.

    IDLE: No active session.
    RECORDING: Recording and accumulating dictation text.
    PROCESSING_LLM: Processing accumulated text through LLM.
    SHUTTING_DOWN: Service shutdown in progress.
    """

    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING_LLM = "processing_llm"
    SHUTTING_DOWN = "shutting_down"


# Valid state transitions for state machine validation
_VALID_TRANSITIONS = {
    DictationState.IDLE: {DictationState.RECORDING, DictationState.SHUTTING_DOWN},
    DictationState.RECORDING: {DictationState.PROCESSING_LLM, DictationState.IDLE, DictationState.SHUTTING_DOWN},
    DictationState.PROCESSING_LLM: {DictationState.IDLE, DictationState.SHUTTING_DOWN},
    DictationState.SHUTTING_DOWN: set(),
}


@dataclass
class DictationSession:
    """Immutable session snapshot capturing dictation state.

    Attributes:
        session_id: Unique session identifier.
        mode: Active dictation mode.
        start_time: Session start timestamp.
        accumulated_text: Accumulated dictation text from STT.
        last_text_time: Timestamp of last text segment.
        is_first_segment: Flag indicating if next segment is first.
    """

    session_id: str
    mode: DictationMode
    start_time: float
    accumulated_text: str = ""
    last_text_time: Optional[float] = None
    is_first_segment: bool = True


@dataclass
class LLMSession:
    """Immutable LLM processing session for state isolation.

    Attributes:
        session_id: Unique LLM session identifier.
        raw_text: Raw dictation text to process.
        agentic_prompt: Generated agentic prompt for LLM.
    """

    session_id: str
    raw_text: str
    agentic_prompt: str


class DictationCoordinator:
    """Production-ready dictation coordinator with thread-safe state management.

    Orchestrates all dictation workflows including standard/smart/type modes,
    integrates STT text recognition events, manages LLM processing for smart mode,
    coordinates with text input service for typing, and maintains strict state
    machine transitions. Thread-safe with RLock protecting all mutable state.

    Attributes:
        _current_state: Current dictation state (IDLE/RECORDING/PROCESSING_LLM/SHUTTING_DOWN).
        _current_session: Active dictation session or None.
        _pending_llm_session: LLM session awaiting processing.
        text_input: TextInputService for typing operations.
        llm_service: LLMService for smart dictation processing.
        agentic_prompt_service: AgenticPromptService for prompt generation.
    """

    def __init__(
        self,
        event_bus: EventBus,
        config: GlobalAppConfig,
        storage: StorageService,
        gui_event_loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        """Initialize dictation coordinator with services and state management.

        Args:
            event_bus: EventBus for pub/sub messaging.
            config: Global application configuration.
            storage: Storage service for persistent data.
            gui_event_loop: Optional GUI event loop for cross-thread operations.
        """
        self.event_bus = event_bus
        self.config = config
        self.gui_event_loop = gui_event_loop

        # Unified state lock for ALL mutable state - CRITICAL for race condition prevention
        self._state_lock = threading.RLock()

        # State machine state
        self._current_state = DictationState.IDLE
        self._current_session: Optional[DictationSession] = None
        self._pending_llm_session: Optional[LLMSession] = None
        self._type_silence_task: Optional[asyncio.Task] = None
        self._llm_processing_task: Optional[asyncio.Task] = None

        # Thread-safe token streaming with proper synchronization
        self._token_queue: queue.Queue = queue.Queue(maxsize=1000)
        self._streaming_active = False
        self._streaming_stop_event = threading.Event()
        self._streaming_thread: Optional[threading.Thread] = None
        self._direct_token_callback: Optional[callable] = None

        # Initialize services
        self.text_service = TextInputService(config=config.dictation)
        self.llm_service = LLMService(event_bus=event_bus, config=config)
        self.agentic_service = AgenticPromptService(event_bus=event_bus, config=config, storage=storage)
        self.alias_service = DictationAliasService(event_bus=event_bus, storage=storage, event_loop=gui_event_loop)

        # Track last text for smart dictation window concatenation logic
        self._last_smart_dictation_text: Optional[str] = None

        # Streaming dictation state for smart/visual modes
        self._streaming_buffer: Optional[StreamingAudioBuffer] = None
        self._streaming_task: Optional[asyncio.Task] = None
        self._streaming_segment_id: str = ""
        self._streaming_finalized_text: str = ""  # Concatenated final text for output
        self._streaming_finalized_segments: list[str] = []  # List of finalized segments

        # Same-output detection
        self._streaming_current_out: str = ""  # Current incomplete segment text
        self._streaming_prev_out: str = ""  # Previous incomplete segment text
        self._streaming_same_output_count: int = 0  # Repetition counter
        self._streaming_end_time_for_same_output: Optional[float] = None  # Timestamp when repetition FIRST occurred
        self._streaming_timestamp_offset: float = 0.0  # Cumulative offset for processed audio

        self._stt_service = None  # Will be injected via set_stt_service

        self.event_publisher = ThreadSafeEventPublisher(event_bus=event_bus, event_loop=gui_event_loop)
        self.subscription_manager = EventSubscriptionManager(event_bus=event_bus, component_name="DictationCoordinator")

        logger.debug("DictationCoordinator initialized with production-ready threading")

    @property
    def active_mode(self) -> DictationMode:
        with self._state_lock:
            return self._current_session.mode if self._current_session else DictationMode.INACTIVE

    def is_active(self) -> bool:
        return self.active_mode != DictationMode.INACTIVE

    def set_stt_service(self, stt_service) -> None:
        """Set STT service reference for streaming dictation.

        Args:
            stt_service: SpeechToTextService instance for streaming recognition.
        """
        self._stt_service = stt_service
        logger.debug("STT service reference set for streaming dictation")

    def _should_apply_formatting(self, mode: DictationMode) -> bool:
        """
        Determine if formatting should be applied based on mode and config.
        TYPE mode always disables formatting regardless of config.
        """
        if mode == DictationMode.TYPE:
            return False
        return self.config.dictation.enable_dictation_formatting

    def _get_state(self) -> DictationState:
        """Thread-safe state getter"""
        with self._state_lock:
            return self._current_state

    def _set_state(self, new_state: DictationState) -> None:
        """Thread-safe state setter with validation - MUST be called with lock held"""
        old_state = self._current_state

        # Validate state transition
        if new_state not in _VALID_TRANSITIONS[old_state]:
            error_msg = f"Invalid state transition: {old_state.value} -> {new_state.value}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        self._current_state = new_state
        logger.debug(f"State transition: {old_state.value} -> {new_state.value}")

    def set_direct_token_callback(self, callback: Optional[callable]) -> None:
        """Set a direct callback for token streaming that bypasses the event bus for minimal latency"""
        self._direct_token_callback = callback
        logger.info(f"Direct token callback {'registered' if callback else 'cleared'}")

    async def initialize(self) -> bool:
        """Initialize all dictation services concurrently.

        Returns:
            True if all services initialized successfully, False otherwise.
        """
        try:
            results = await asyncio.gather(
                self.text_service.initialize(),
                self.llm_service.initialize(),
                self.agentic_service.initialize(),
                self.alias_service.initialize(),
                return_exceptions=True,
            )

            if any(isinstance(r, Exception) or not r for r in results):
                logger.error("Service initialization failed")
                return False

            self.agentic_service.setup_subscriptions()
            logger.info("Dictation coordinator initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Initialization error: {e}", exc_info=True)
            return False

    def setup_subscriptions(self) -> None:
        """Set up event subscriptions for dictation coordinator.

        Subscribes to dictation text, command parsed events, LLM events,
        audio chunks for streaming, and agentic prompt ready events for
        comprehensive dictation workflow management.
        """
        subscriptions = [
            (DictationTextRecognizedEvent, self._handle_dictation_text),
            (AudioChunkEvent, self._handle_audio_chunk_for_streaming),
            (LLMProcessingCompletedEvent, self._handle_llm_completed),
            (LLMProcessingFailedEvent, self._handle_llm_failed),
            (DictationCommandParsedEvent, self._handle_dictation_command),
            (LLMProcessingReadyEvent, self._handle_llm_processing_ready),
        ]

        for event_type, handler in subscriptions:
            self.subscription_manager.subscribe(event_type, handler)

        logger.info("Event subscriptions configured")

    async def _handle_dictation_text(self, event: DictationTextRecognizedEvent) -> None:
        """Handle dictated text - centralized processing for all dictation modes"""
        try:
            text = event.text.strip()
            if not text:
                return

            with self._state_lock:
                session = self._current_session
                if not session:
                    return

                if self._current_state != DictationState.RECORDING:
                    return

                # Skip smart/visual/hidden modes - they use streaming flow
                if session.mode in (DictationMode.SMART, DictationMode.VISUAL, DictationMode.HIDDEN):
                    logger.debug(f"Skipping VAD-based text for streaming mode: {session.mode.value}")
                    return

            cleaned_text = self._clean_text(text)
            if not cleaned_text:
                return

            # Apply alias substitutions to all dictation modes
            cleaned_text = self.alias_service.apply_substitutions(cleaned_text)

            if not self._should_apply_formatting(mode=session.mode):
                cleaned_text = remove_formatting(text=cleaned_text, is_first_word_of_session=session.is_first_segment)

            updated_session = DictationSession(
                session_id=session.session_id,
                mode=session.mode,
                start_time=session.start_time,
                accumulated_text=self._append_text(session.accumulated_text, cleaned_text),
                last_text_time=time.time() if session.mode == DictationMode.TYPE else None,
                is_first_segment=False,
            )

            with self._state_lock:
                if self._current_session and self._current_session.session_id == session.session_id:
                    self._current_session = updated_session
                else:
                    return

            if updated_session.mode == DictationMode.SMART:
                display_text = clean_dictation_text(text=cleaned_text, add_trailing_space=True)

                if self._last_smart_dictation_text and should_remove_previous_period(
                    self._last_smart_dictation_text, display_text
                ):
                    trailing_whitespace_count = get_trailing_whitespace_count(self._last_smart_dictation_text)
                    chars_to_remove = 1 + trailing_whitespace_count

                    await self._publish_event(SmartDictationRemoveCharactersEvent(count=chars_to_remove))

                    display_text = " " + display_text

                if self._last_smart_dictation_text and should_lowercase_current_start(
                    self._last_smart_dictation_text, display_text
                ):
                    display_text = lowercase_first_letter(display_text)

                self._last_smart_dictation_text = display_text

                await self._publish_event(SmartDictationTextDisplayEvent(text=display_text))
            elif updated_session.mode == DictationMode.VISUAL:
                # Visual mode: display text in UI without LLM processing
                display_text = clean_dictation_text(text=cleaned_text, add_trailing_space=True)
                await self._publish_event(SmartDictationTextDisplayEvent(text=display_text))
            else:
                add_trailing_space = updated_session.mode != DictationMode.TYPE
                await self.text_service.input_text(text=cleaned_text, add_trailing_space=add_trailing_space)

        except Exception as e:
            logger.error(f"Dictation text error: {e}", exc_info=True)

    async def _cleanup_llm_session(self) -> None:
        """Common cleanup for LLM session completion or failure"""
        with self._state_lock:
            self._current_session = None
            self._pending_llm_session = None
            self._llm_processing_task = None
            self._last_smart_dictation_text = None
            self._set_state(DictationState.IDLE)
        await self._end_smart_session()

    async def _handle_llm_completed(self, event: LLMProcessingCompletedEvent) -> None:
        """Handle LLM completion - clear state and move to IDLE"""
        try:
            logger.info(f"LLM COMPLETION EVENT RECEIVED: '{event.processed_text[:100]}...'")
            logger.info("Inputting text via text service...")

            processed_text = event.processed_text

            # Apply formatting filter if enabled is False
            if not self.config.dictation.enable_dictation_formatting:
                processed_text = remove_formatting(text=processed_text, is_first_word_of_session=True)

            success = await self.text_service.input_text(processed_text)
            logger.info(f"Text input result: {success}")

            await self._cleanup_llm_session()
            logger.info("Smart session ended after LLM completion")
        except Exception as e:
            logger.error(f"LLM completion error: {e}", exc_info=True)
            await self._cleanup_llm_session()

    async def _handle_llm_failed(self, event: LLMProcessingFailedEvent) -> None:
        """Handle LLM failure - reset state and cleanup"""
        logger.warning(f"LLM processing failed: {event.error_message}")
        await self._cleanup_llm_session()
        await self._publish_error("Smart dictation processing failed")

    async def _handle_dictation_command(self, event: DictationCommandParsedEvent) -> None:
        """Handle dictation commands"""
        try:
            command = event.command
            if isinstance(command, DictationStartCommand):
                await self._start_session(DictationMode.STANDARD)
            elif isinstance(command, DictationStopCommand):
                await self._stop_session()
            elif isinstance(command, DictationTypeCommand):
                await self._start_session(DictationMode.TYPE)
            elif isinstance(command, DictationSmartStartCommand):
                await self._start_session(DictationMode.SMART)
            elif isinstance(command, DictationVisualStartCommand):
                await self._start_session(DictationMode.VISUAL)
            elif isinstance(command, DictationHiddenStartCommand):
                await self._start_session(DictationMode.HIDDEN)

        except Exception as e:
            logger.error(f"Command handling error: {e}", exc_info=True)

    async def _handle_llm_processing_ready(self, event: LLMProcessingReadyEvent) -> None:
        """Handle LLM processing ready signal from UI"""
        try:
            with self._state_lock:
                pending = self._pending_llm_session
                if not pending or pending.session_id != event.session_id:
                    logger.warning(f"Received ready signal for unknown session {event.session_id}")
                    return

                self._pending_llm_session = None

            logger.info(f"UI ready signal received for session {event.session_id}")
            self._llm_processing_task = asyncio.create_task(self._start_llm_processing(pending))

        except Exception as e:
            logger.error(f"LLM processing ready handling error: {e}", exc_info=True)

    async def _handle_audio_chunk_for_streaming(self, event: AudioChunkEvent) -> None:
        """Route audio chunks to streaming buffer for smart/visual/hidden modes.

        Args:
            event: AudioChunkEvent containing 50ms audio chunk.
        """
        try:
            with self._state_lock:
                session = self._current_session
                buffer = self._streaming_buffer

            # Only process if we're in streaming mode (smart/visual/hidden) and have a buffer
            if not session or not buffer:
                return

            if session.mode not in (DictationMode.SMART, DictationMode.VISUAL, DictationMode.HIDDEN):
                return

            # Add chunk to streaming buffer
            import numpy as np

            chunk_np = np.frombuffer(event.audio_chunk, dtype=np.int16)
            await buffer.add_chunk(chunk_np)

        except Exception as e:
            logger.error(f"Error handling audio chunk for streaming: {e}", exc_info=True)

    def _texts_are_similar(self, text1: str, text2: str, threshold: float = 0.8) -> bool:
        """Check if two texts are similar enough to be considered the same output.

        Whisper often returns slightly different text for the same audio:
        - "showing the transformed output." vs "showing the transformed"
        - "and then showing" vs "and showing"

        This uses a simple overlap-based similarity to handle these variations.

        Args:
            text1: First text to compare.
            text2: Second text to compare.
            threshold: Minimum similarity ratio (0.0 to 1.0).

        Returns:
            True if texts are similar enough.
        """
        if not text1 or not text2:
            return False

        t1 = text1.strip().lower()
        t2 = text2.strip().lower()

        # Exact match
        if t1 == t2:
            return True

        # One is a prefix/suffix of the other (common with trailing words)
        if t1.startswith(t2) or t2.startswith(t1):
            shorter = min(len(t1), len(t2))
            longer = max(len(t1), len(t2))
            if shorter / longer >= threshold:
                return True

        # Word-based overlap for longer texts
        words1 = set(t1.split())
        words2 = set(t2.split())
        if len(words1) >= 3 and len(words2) >= 3:
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            if union > 0 and intersection / union >= threshold:
                return True

        return False

    def _is_gibberish_text(self, text: str, strict: bool = False) -> bool:
        """Secondary gibberish detection at coordinator level.

        This provides a safety net in case hallucinations slip through the STT layer.
        Checks for common Whisper hallucination patterns that indicate no real speech.

        Args:
            text: Text to check for gibberish patterns.
            strict: If True, use stricter filtering (for partials shown to user).

        Returns:
            True if the text appears to be gibberish/hallucination.
        """
        if not text or not text.strip():
            return True

        text = text.strip()

        # Only punctuation/dots/symbols (very common hallucination)
        alpha_count = sum(1 for c in text if c.isalpha())
        if alpha_count == 0:
            return True

        # For strict mode (partials), require at least 2 alpha chars
        if strict and alpha_count < 2:
            return True

        # Repetitive patterns (e.g., "4.4.4.4", ". . . . .")
        stripped = text.replace(" ", "")
        if len(stripped) > 3:
            unique_chars = len(set(stripped))
            # Stricter: if mostly same chars repeated
            if unique_chars <= 2:
                return True
            # For strict mode, also catch 3 unique chars in long strings
            if strict and len(stripped) > 10 and unique_chars <= 3:
                return True

        # Check for repeating short patterns (e.g., "4.4.4", ". . .")
        if len(stripped) > 5:
            for pattern_len in range(1, 4):
                if pattern_len >= len(stripped):
                    continue
                pattern = stripped[:pattern_len]
                repetitions = stripped.count(pattern)
                threshold = len(stripped) / (pattern_len + 0.5) if strict else len(stripped) / (pattern_len + 1)
                if repetitions > threshold:
                    return True

        # Very low alpha ratio (mostly symbols/numbers in repetitive pattern)
        alpha_ratio = alpha_count / len(text)
        if len(text) > 5 and alpha_ratio < (0.4 if strict else 0.3):
            return True

        # Strict mode: check for suspicious short patterns like ". . . ."
        if strict:
            # Count dots and spaces vs other chars
            dots_spaces = sum(1 for c in text if c in ". ")
            if len(text) > 3 and dots_spaces / len(text) > 0.6:
                return True

        return False

    async def _streaming_transcription_loop(self) -> None:
        """Streaming transcription loop for continuous processing.

        Processes audio in a continuous loop, sleeping only when insufficient audio is available.
        Handles multi-segment processing, finalizing all but the last segment immediately.
        Detects same-output repetition for the last segment.
        """
        try:
            MIN_AUDIO_SECONDS = 1.0
            SAME_OUTPUT_THRESHOLD = 7
            NO_SPEECH_THRESH = 0.45

            # Conservative offset advancement settings
            MAX_CONSECUTIVE_EMPTY = 8  # After 8 empty results (~2s), assume true silence
            CONSERVATIVE_ADVANCE_SECONDS = 0.5  # Only advance by 0.5s at a time
            MAX_UNPROCESSED_BEFORE_CLIP = 15.0  # Only clip if unprocessed > 15s

            logger.info(
                f"Starting streaming loop (continuous, min_audio={MIN_AUDIO_SECONDS}s, same_output_threshold={SAME_OUTPUT_THRESHOLD})"
            )

            # State tracking for streaming transcription
            current_out = ""  # Current incomplete segment text
            prev_out = ""  # Previous incomplete segment text
            same_output_count = 0  # Repetition counter
            end_time_for_same_output: Optional[float] = None  # Timestamp when repetition FIRST occurred
            segment_id = ""

            # List of all processed text segments for duplicate detection
            all_text_segments: list[str] = []

            # Track finalized segments separately for UI emission
            finalized_segments: list[str] = []

            # Silence detection - track consecutive empty results
            consecutive_empty_results = 0
            time.time()

            while True:
                # Check session is still valid FIRST (before any processing)
                with self._state_lock:
                    session = self._current_session
                    buffer = self._streaming_buffer

                    if not session or not buffer:
                        logger.debug("Streaming loop: No session or buffer, exiting")
                        break

                    if session.mode not in (DictationMode.SMART, DictationMode.VISUAL, DictationMode.HIDDEN):
                        logger.debug("Streaming loop: Not in streaming mode, exiting")
                        break

                # Get unprocessed audio for transcription
                audio_result = await buffer.get_audio_for_transcription()

                if not audio_result:
                    # No audio yet - brief sleep and continue
                    await asyncio.sleep(0.05)
                    continue

                audio_bytes, duration = audio_result

                # Skip if not enough audio
                if duration < MIN_AUDIO_SECONDS:
                    await asyncio.sleep(0.1)  # Wait for more audio to arrive
                    continue

                # Transcribe with segment timestamps
                if not self._stt_service:
                    await asyncio.sleep(0.1)
                    continue

                # Get current timestamp_offset for absolute timestamp calculation
                current_timestamp_offset = await buffer.get_timestamp_offset()

                # Check if unprocessed audio is approaching context limit - force finalization
                # This prevents losing partial text when audio exceeds Whisper's window
                unprocessed_duration = await buffer.get_unprocessed_duration()
                MAX_UNPROCESSED_BEFORE_FORCE_FINALIZE = 20.0  # Force finalize at 20s

                if unprocessed_duration > MAX_UNPROCESSED_BEFORE_FORCE_FINALIZE and current_out:
                    # Force finalize the current partial to prevent loss
                    if not self._is_gibberish_text(current_out):
                        is_duplicate = (
                            any(self._texts_are_similar(ts, current_out, 0.8) for ts in all_text_segments[-5:])
                            if all_text_segments
                            else False
                        )
                        if not is_duplicate:
                            all_text_segments.append(current_out.strip())
                            finalized_segments.append(current_out.strip())
                            await self._emit_final_text_append(current_out.strip(), segment_id, session)
                            logger.info(
                                f"Force finalized due to long unprocessed audio ({unprocessed_duration:.1f}s): '{current_out[:50]}...'"
                            )

                            # Advance offset to the end of what we finalized
                            # Use the end_time_for_same_output if available, otherwise estimate
                            if end_time_for_same_output is not None:
                                await buffer.advance_timestamp_offset(end_time_for_same_output)

                            # Reset state for next segment
                            current_out = ""
                            prev_out = ""
                            same_output_count = 0
                            end_time_for_same_output = None
                            segment_id = str(uuid.uuid4())

                text, confidence, segments = await self._stt_service.recognize_streaming(
                    audio_bytes=audio_bytes,
                    sample_rate=self.config.audio.sample_rate,
                    return_segments=True,
                )

                if not segments:
                    # No valid segments returned - BUT DON'T advance full offset!
                    # VAD may have filtered audio that contains speech START
                    consecutive_empty_results += 1

                    # Only advance offset conservatively after prolonged silence
                    # This prevents losing speech that VAD filtered at the beginning
                    unprocessed_duration = await buffer.get_unprocessed_duration()

                    if consecutive_empty_results >= MAX_CONSECUTIVE_EMPTY:
                        # True prolonged silence - safe to advance conservatively
                        if unprocessed_duration > MAX_UNPROCESSED_BEFORE_CLIP:
                            # Only clip if we have a lot of unprocessed audio
                            advance_amount = min(CONSERVATIVE_ADVANCE_SECONDS, unprocessed_duration - 2.0)
                            if advance_amount > 0:
                                await buffer.advance_timestamp_offset(advance_amount)
                                logger.debug(f"Prolonged silence: advanced offset by {advance_amount:.2f}s")

                        # Finalize any pending valid text before clearing
                        if current_out and not self._is_gibberish_text(current_out):
                            is_duplicate = (
                                any(self._texts_are_similar(ts, current_out, 0.8) for ts in all_text_segments[-5:])
                                if all_text_segments
                                else False
                            )
                            if not is_duplicate:
                                all_text_segments.append(current_out.strip())
                                finalized_segments.append(current_out.strip())
                                await self._emit_final_text_append(current_out.strip(), segment_id, session)
                                logger.debug(f"Finalized due to silence: '{current_out[:50]}...'")

                        # Reset state for silence period
                        current_out = ""
                        prev_out = ""
                        same_output_count = 0
                        end_time_for_same_output = None
                        segment_id = str(uuid.uuid4())
                        consecutive_empty_results = 0  # Reset counter after handling

                    await asyncio.sleep(0.2)  # Wait for voice activity
                    continue

                # We got segments - reset silence counter and update last valid time
                consecutive_empty_results = 0
                time.time()

                offset_to_advance: Optional[float] = None
                all_segments_gibberish = (
                    True  # Track if all segments are gibberish                # Process complete segments (all but last)
                )
                # If multiple segments and last segment is valid, finalize all but last immediately
                if len(segments) > 1:
                    last_seg_no_speech = segments[-1].get("no_speech_prob", 0)
                    if last_seg_no_speech <= NO_SPEECH_THRESH:
                        for seg in segments[:-1]:
                            seg_text = seg["text"].strip()
                            seg_no_speech = seg.get("no_speech_prob", 0)
                            seg_start = seg["start"]
                            seg_end = seg["end"]

                            # Always append to all_text_segments for tracking
                            if seg_text:
                                all_text_segments.append(seg_text)

                            # Skip if start >= end
                            if seg_start >= seg_end:
                                continue

                            # Skip if no_speech_prob too high
                            if seg_no_speech > NO_SPEECH_THRESH:
                                continue
                            if not seg_text:
                                continue

                            # Secondary gibberish check - safety net
                            if self._is_gibberish_text(seg_text):
                                logger.debug(f"Filtered gibberish segment: '{seg_text[:30]}...'")
                                continue

                            # Calculate absolute timestamps
                            absolute_start = current_timestamp_offset + seg_start
                            absolute_end = current_timestamp_offset + min(duration, seg_end)

                            # Duplicate check against all_text_segments using fuzzy matching
                            # This catches near-duplicates that differ by a few words
                            is_duplicate = (
                                any(self._texts_are_similar(ts, seg_text, 0.8) for ts in all_text_segments[-10:])
                                if len(all_text_segments) > 1
                                else False
                            )
                            if is_duplicate:
                                logger.debug(f"Skipping duplicate segment: '{seg_text[:30]}...'")
                                continue

                            # Finalize this complete segment immediately
                            finalized_segments.append(seg_text)
                            await self._emit_final_text_append(seg_text, str(uuid.uuid4()), session)
                            logger.debug(
                                f"Finalized complete segment [{absolute_start:.2f}s-{absolute_end:.2f}s]: '{seg_text[:50]}...'"
                            )

                            # Track offset for advancement
                            offset_to_advance = min(duration, seg_end)
                            all_segments_gibberish = False  # We found a valid segment

                # Process last segment
                last_segment = segments[-1]
                last_seg_no_speech = last_segment.get("no_speech_prob", 0)
                last_seg_end = last_segment["end"]

                if last_seg_no_speech <= NO_SPEECH_THRESH:
                    # Set current_out first
                    candidate_text = last_segment["text"].strip()
                    # Filter gibberish before setting current_out (use strict mode for partials)
                    if self._is_gibberish_text(candidate_text, strict=True):
                        logger.debug(f"Filtered gibberish last segment: '{candidate_text[:30]}...'")
                        current_out = ""
                    else:
                        current_out = candidate_text
                        all_segments_gibberish = False  # At least one valid segment
                else:
                    current_out = ""

                # If ALL segments were gibberish, advance offset to prevent getting stuck
                # This is critical to break out of hallucination loops
                if all_segments_gibberish and offset_to_advance is None:
                    # Advance by a small amount to move past the problematic audio
                    advance_amount = min(duration, 2.0)  # Advance up to 2 seconds
                    await buffer.advance_timestamp_offset(advance_amount)
                    logger.warning(f"All segments gibberish, advancing offset by {advance_amount:.2f}s to break loop")
                    # Reset same-output tracking to prevent corruption
                    same_output_count = 0
                    prev_out = ""
                    end_time_for_same_output = None
                    continue

                # Same-output detection
                # Use fuzzy matching because Whisper often returns slightly different text:
                # "showing the transformed output." vs "showing the transformed"
                if current_out != "" and self._texts_are_similar(current_out, prev_out, threshold=0.75):
                    same_output_count += 1
                    # Capture end time on first repetition
                    if end_time_for_same_output is None:
                        end_time_for_same_output = last_seg_end
                else:
                    same_output_count = 0
                    end_time_for_same_output = None

                # Finalize if same output repeated enough
                if same_output_count > SAME_OUTPUT_THRESHOLD:
                    # Check against all_text_segments using fuzzy matching
                    is_duplicate = (
                        any(self._texts_are_similar(ts, current_out, 0.8) for ts in all_text_segments[-5:])
                        if all_text_segments
                        else False
                    )

                    if not is_duplicate and current_out.strip():
                        all_text_segments.append(current_out.strip())
                        finalized_segments.append(current_out.strip())
                        await self._emit_final_text_append(current_out.strip(), segment_id, session)
                        logger.debug(f"Finalized after {same_output_count} same outputs: '{current_out[:50]}...'")

                    # Reset state
                    current_out = ""
                    offset_to_advance = min(duration, end_time_for_same_output) if end_time_for_same_output else None
                    same_output_count = 0
                    end_time_for_same_output = None
                    segment_id = str(uuid.uuid4())
                else:
                    # Update prev_out only when not finalizing
                    prev_out = current_out

                # Advance offset if we finalized anything
                if offset_to_advance is not None:
                    await buffer.advance_timestamp_offset(offset_to_advance)

                # STEP 6: Emit partial for incomplete segment (skip for hidden mode)
                # Only emit if we have valid non-gibberish text
                if current_out and session.mode != DictationMode.HIDDEN:
                    if not segment_id:
                        segment_id = str(uuid.uuid4())
                    text_with_subs = self.alias_service.apply_substitutions(current_out)
                    await self._publish_event(PartialDictationTextEvent(text=text_with_subs, segment_id=segment_id))

                # Brief yield to allow other tasks to run (prevents UI freeze)
                # But keep it minimal for responsiveness
                await asyncio.sleep(0.01)

        except asyncio.CancelledError:
            # Save current unfinalized text before exiting
            if current_out.strip() and not self._is_gibberish_text(current_out):
                # Check for duplicate using fuzzy matching
                is_duplicate = (
                    any(self._texts_are_similar(ts, current_out, 0.8) for ts in all_text_segments[-5:])
                    if all_text_segments
                    else False
                )
                if not is_duplicate:
                    if self._streaming_finalized_text:
                        self._streaming_finalized_text += " " + current_out.strip()
                    else:
                        self._streaming_finalized_text = current_out.strip()
                    logger.info(f"Loop cancelled, saved remaining: '{current_out[:50]}...'")
            logger.info("Streaming transcription loop cancelled")
        except Exception as e:
            logger.error(f"Error in streaming transcription loop: {e}", exc_info=True)

    async def _emit_final_text(self, text: str, segment_id: str, session) -> None:
        """Emit final text event and update state (DEPRECATED - use _emit_final_text_append).

        WARNING: This method OVERWRITES _streaming_finalized_text. Use _emit_final_text_append
        for proper accumulation in streaming mode.
        """
        if not text:
            return

        # Emit final event (skip for hidden mode)
        if session.mode != DictationMode.HIDDEN:
            text_with_subs = self.alias_service.apply_substitutions(text)
            await self._publish_event(FinalDictationTextEvent(text=text_with_subs, segment_id=segment_id or str(uuid.uuid4())))

        # Update accumulator (OVERWRITES - legacy behavior)
        self._streaming_finalized_text = text

        # Add to STT context
        if self._stt_service:
            await self._stt_service.add_finalized_segment(text)

    async def _emit_final_text_append(self, text: str, segment_id: str, session) -> None:
        """Emit final text event and append to accumulated state.

        This method appends new text to the accumulator instead of overwriting it, preventing data loss.
        Also maintains segment list for deduplication.

        Args:
            text: New finalized text to append.
            segment_id: Segment ID for UI matching.
            session: Current dictation session.
        """
        if not text or not text.strip():
            return

        text = text.strip()

        # Duplicate check: don't add if identical to last segment
        if self._streaming_finalized_segments:
            if self._streaming_finalized_segments[-1].strip().lower() == text.lower():
                logger.debug(f"Skipping duplicate segment: '{text[:30]}...'")
                return

        # Emit final event (skip for hidden mode)
        if session.mode != DictationMode.HIDDEN:
            text_with_subs = self.alias_service.apply_substitutions(text)
            await self._publish_event(FinalDictationTextEvent(text=text_with_subs, segment_id=segment_id or str(uuid.uuid4())))

        # Add to segment list
        self._streaming_finalized_segments.append(text)

        # Append to accumulator to prevent data loss
        if self._streaming_finalized_text:
            self._streaming_finalized_text += " " + text
        else:
            self._streaming_finalized_text = text

        logger.debug(f"Appended finalized segment #{len(self._streaming_finalized_segments)}: '{text[:30]}...'")

        # Add to STT context for better subsequent transcriptions
        if self._stt_service:
            await self._stt_service.add_finalized_segment(text)

    async def _finalize_text(self, text: str, segment_id: str, session) -> None:
        """Finalize text (convert partial to final).

        Args:
            text: Text to finalize.
            segment_id: Segment ID for UI matching.
            session: Current dictation session.
        """
        if not text:
            return

        segment_id = segment_id or str(uuid.uuid4())

        # Emit final event (skip for hidden mode)
        if session.mode != DictationMode.HIDDEN:
            text_with_subs = self.alias_service.apply_substitutions(text)
            await self._publish_event(FinalDictationTextEvent(text=text_with_subs, segment_id=segment_id))

        logger.info(f"Finalized: '{text}'")

        # Add to context
        if self._stt_service:
            await self._stt_service.add_finalized_segment(text)

        # Update accumulator
        if self._streaming_finalized_text:
            self._streaming_finalized_text += " " + text
        else:
            self._streaming_finalized_text = text

    async def _flush_remaining_audio_buffer(self, mode: DictationMode) -> None:
        """Flush any remaining audio in the buffer and transcribe it.

        Called when stopping dictation to ensure we don't lose speech.
        Uses deduplication against last finalized segment.
        Also removes stop word from flushed text to prevent duplicate stop words.

        Args:
            mode: Current dictation mode.
        """
        if not self._streaming_buffer or not self._stt_service:
            return

        try:
            # Get unprocessed audio duration
            unprocessed_duration = await self._streaming_buffer.get_unprocessed_duration()

            if unprocessed_duration < 0.3:
                logger.debug(f"No significant unprocessed audio to flush: {unprocessed_duration:.2f}s")
                return

            logger.info(f"Flushing {unprocessed_duration:.2f}s of unprocessed audio")

            # Get unprocessed audio
            audio_result = await self._streaming_buffer.get_audio_for_transcription()

            if not audio_result:
                logger.debug("No audio available for flush")
                return

            audio_bytes, duration = audio_result

            # Transcribe
            text, confidence, segments = await self._stt_service.recognize_streaming(
                audio_bytes=audio_bytes,
                sample_rate=self.config.audio.sample_rate,
                return_segments=True,
            )

            if not text or not text.strip():
                logger.debug("No speech detected in flushed audio")
                return

            text = text.strip()

            # Remove stop word from flushed text BEFORE adding to finalized
            # This prevents "amber amber" duplication when stop word is in flush
            text = self._remove_stop_word(text)
            if not text or not text.strip():
                logger.debug("Flushed text was only stop word, skipping")
                return
            text = text.strip()

            # Duplicate check against last finalized segment
            if self._streaming_finalized_segments:
                last_segment = self._streaming_finalized_segments[-1].strip().lower()
                if text.lower() == last_segment:
                    logger.debug("Flushed text is duplicate of last segment, skipping")
                    return
                # Also check if flush text is contained in or contains last segment
                if text.lower() in last_segment or last_segment in text.lower():
                    # Use overlap stripping for partial matches
                    text = self._strip_overlap(text)
                    if not text:
                        logger.debug("Flushed text fully overlaps with finalized, skipping")
                        return
                # Check fuzzy similarity for near-duplicates
                if self._texts_are_similar(text, last_segment, threshold=0.8):
                    logger.debug("Flushed text is similar to last segment, skipping")
                    return

            logger.info(f"Flushed transcription: '{text}'")

            # Add to segment list
            self._streaming_finalized_segments.append(text)

            # Add to STT context
            await self._stt_service.add_finalized_segment(text)

            # APPEND to finalized text accumulator
            if self._streaming_finalized_text:
                self._streaming_finalized_text += " " + text
            else:
                self._streaming_finalized_text = text

        except Exception as e:
            logger.error(f"Error flushing remaining audio buffer: {e}", exc_info=True)

    def _remove_stop_word(self, text: str) -> str:
        """Remove stop word from text for hidden mode finalization.

        Removes the stop trigger word (e.g., "amber") from the final text.
        This ensures hidden mode captures everything except the stop word itself.

        Args:
            text: Text potentially containing the stop word.

        Returns:
            Text with stop word removed and trailing whitespace cleaned.
        """
        stop_word = self.config.dictation.stop_trigger
        if not stop_word or not text:
            return text

        # Remove stop word case-insensitively (with word boundaries)
        import re

        # Create pattern that matches stop word with word boundaries, case-insensitive
        pattern = r"\b" + re.escape(stop_word) + r"\b"
        result = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # Clean up extra whitespace
        result = " ".join(result.split())
        return result

    def _normalize_for_comparison(self, text: str) -> list[str]:
        """Normalize text for overlap comparison - removes punctuation.

        This ensures that variations like "the goal here is," vs "the goal here is"
        are correctly identified as overlapping, preventing duplication from
        Whisper's inconsistent punctuation.

        Args:
            text: Text to normalize.

        Returns:
            List of lowercase words with punctuation removed.
        """
        import re

        cleaned = re.sub(r"[^\w\s]", "", text.lower())
        return cleaned.split()

    def _strip_overlap(self, text: str) -> str:
        """Strip overlapping prefix that matches finalized text.

        Uses punctuation-insensitive comparison to handle Whisper's inconsistent
        punctuation (e.g., "the goal here is," vs "the goal here is").

        Handles two overlap scenarios:
        1. Full containment: New text starts with all of finalized text
        2. Suffix-prefix overlap: Last N words of finalized match first N words of new text

        Args:
            text: New predicted text that may overlap with finalized text.

        Returns:
            Text with overlapping prefix removed, or empty string if fully overlapping.
        """
        if not self._streaming_finalized_text or not text:
            return text

        # Use normalized (punctuation-free) comparison for matching
        finalized_normalized = self._normalize_for_comparison(self._streaming_finalized_text)
        text_words = text.split()  # Keep original words for output (preserves punctuation)
        text_normalized = self._normalize_for_comparison(text)

        # Case 1: Check if new text starts with all of finalized text (full re-transcription)
        # This happens when Whisper re-transcribes from the beginning with context
        if len(text_normalized) >= len(finalized_normalized):
            if text_normalized[: len(finalized_normalized)] == finalized_normalized:
                # New text contains all of finalized text at the start
                remaining_words = text_words[len(finalized_normalized) :]
                result = " ".join(remaining_words)
                if result:
                    logger.debug(f"Stripped full finalized text ({len(finalized_normalized)} words) from start of prediction")
                    return result
                else:
                    logger.debug("New text is identical to finalized text, returning empty")
                    return ""

        # Case 2: Suffix-prefix overlap - find longest matching suffix of finalized
        # that matches prefix of new text
        max_overlap = min(len(finalized_normalized), len(text_normalized))
        overlap_length = 0

        for i in range(1, max_overlap + 1):
            # Check if last i words of finalized match first i words of new text
            if finalized_normalized[-i:] == text_normalized[:i]:
                overlap_length = i

        if overlap_length > 0:
            # Remove overlapping prefix (use original word count for slicing)
            remaining_words = text_words[overlap_length:]
            result = " ".join(remaining_words)
            if result:
                logger.debug(f"Stripped {overlap_length} overlapping words from start of prediction")
                return result
            else:
                # Entire prediction was overlap - return empty to avoid duplication
                logger.debug("Entire prediction was overlap with finalized text")
                return ""

        return text

    async def _stop_streaming_mode(self, session: DictationSession) -> None:
        """Stop streaming dictation mode and handle finalization.

        Ensures all audio is transcribed before pasting:
        1. Cancel the streaming transcription loop (loop saves remaining text on cancel)
        2. Flush remaining audio buffer to STT (force transcribe any trailing audio)
        3. Collect all accumulated finalized text
        4. Paste accumulated text

        Args:
            session: Current dictation session.
        """
        try:
            # Cancel streaming task - the loop will save any unfinalized text on CancelledError
            if self._streaming_task and not self._streaming_task.done():
                self._streaming_task.cancel()
                try:
                    await self._streaming_task
                except asyncio.CancelledError:
                    pass
                self._streaming_task = None

            # Clear streaming state variables (the loop already saved text to _streaming_finalized_text)
            self._streaming_current_out = ""
            self._streaming_prev_out = ""
            self._streaming_same_output_count = 0
            self._streaming_segment_id = ""

            # Flush remaining audio - appends to _streaming_finalized_text
            if self._streaming_buffer and self._stt_service:
                await self._flush_remaining_audio_buffer(session.mode)

            # Get all finalized text (accumulated via append, not overwrite)
            final_text = self._streaming_finalized_text

            # For hidden mode, remove the stop word from the final text
            if session.mode == DictationMode.HIDDEN and final_text:
                final_text = self._remove_stop_word(final_text)
            if final_text:
                final_text = self.alias_service.apply_substitutions(final_text)
                # Clean up any double spaces from concatenation
                final_text = " ".join(final_text.split())
                logger.debug(f"Final text after cleanup: {len(final_text)} chars")

            # Clean up streaming state
            if self._streaming_buffer:
                await self._streaming_buffer.clear()
                self._streaming_buffer = None

            self._streaming_finalized_text = ""
            self._streaming_finalized_segments = []
            self._streaming_end_time_for_same_output = None
            self._streaming_timestamp_offset = 0.0

            # Update session state
            with self._state_lock:
                if session.mode == DictationMode.SMART and final_text:
                    # Smart mode: process through LLM
                    self._set_state(DictationState.PROCESSING_LLM)

                    agentic_prompt = self.agentic_service.get_current_prompt() or "Fix grammar and improve clarity."
                    llm_session_id = str(uuid.uuid4())
                    self._pending_llm_session = LLMSession(
                        session_id=llm_session_id,
                        raw_text=final_text,
                        agentic_prompt=agentic_prompt,
                    )
                else:
                    # Visual/Hidden mode or no text: finalize directly
                    self._current_session = None
                    self._set_state(DictationState.IDLE)

            # Emit stop events
            if session.mode == DictationMode.SMART and final_text:
                await self._publish_event(AudioModeChangeRequestEvent(mode="command", reason="Smart dictation processing"))
                await self._publish_event(SmartDictationStoppedEvent(raw_text=final_text))
                await self._publish_event(
                    LLMProcessingStartedEvent(
                        raw_text=final_text,
                        agentic_prompt=self._pending_llm_session.agentic_prompt,
                        session_id=self._pending_llm_session.session_id,
                    )
                )
            elif session.mode == DictationMode.SMART:
                # No text accumulated, just end session
                await self._end_smart_session()
            elif session.mode == DictationMode.VISUAL:
                # Visual mode: paste accumulated text
                if final_text:
                    await self._publish_event(VisualDictationStoppedEvent(accumulated_text=final_text))
                    await self.text_service.input_text(final_text)
                else:
                    await self._publish_event(VisualDictationStoppedEvent(accumulated_text=""))
                await self._finalize_session(session)
            elif session.mode == DictationMode.HIDDEN:
                # Hidden mode: paste accumulated text silently
                if final_text:
                    await self._publish_event(HiddenDictationStoppedEvent(accumulated_text=final_text))
                    await self.text_service.input_text(final_text)
                else:
                    await self._publish_event(HiddenDictationStoppedEvent(accumulated_text=""))
                await self._finalize_session(session)

            logger.info(
                f"Streaming {session.mode.value} mode stopped, finalized text: {len(final_text) if final_text else 0} chars"
            )

        except Exception as e:
            logger.error(f"Error stopping streaming mode: {e}", exc_info=True)
            # Ensure cleanup even on error
            with self._state_lock:
                self._current_session = None
                self._set_state(DictationState.IDLE)

    async def _monitor_type_silence(self) -> None:
        """Monitor silence timeout for TYPE dictation mode with safety limits"""
        try:
            timeout = self.config.dictation.type_dictation_silence_timeout
            max_runtime = 300
            start_time = time.time()

            while True:
                if time.time() - start_time > max_runtime:
                    logger.warning(f"Type silence monitoring exceeded max runtime ({max_runtime}s), auto-stopping")
                    break

                await asyncio.sleep(0.1)

                with self._state_lock:
                    session = self._current_session
                    if not session or session.mode != DictationMode.TYPE:
                        return

                    if session.last_text_time is None:
                        continue

                    time_since_last_text = time.time() - session.last_text_time

                    if time_since_last_text >= timeout:
                        logger.info(f"Type dictation silence timeout exceeded ({timeout}s), auto-stopping")
                        break

            await self._stop_session()

        except asyncio.CancelledError:
            logger.debug("Type silence monitoring task cancelled")
        except Exception as e:
            logger.error(f"Type silence monitoring error: {e}", exc_info=True)

    def _cancel_type_silence_task(self) -> None:
        """Cancel the type silence monitoring task properly"""
        if self._type_silence_task and not self._type_silence_task.done():
            self._type_silence_task.cancel()
            self._type_silence_task = None
            logger.debug("Type silence task cancelled")

    async def _start_session(self, mode: DictationMode) -> None:
        """Start dictation session with guards against concurrent starts"""
        try:
            session_id = str(uuid.uuid4())

            with self._state_lock:
                if self._current_session is not None:
                    logger.warning(
                        f"Cannot start {mode.value} dictation - session {self._current_session.mode.value} already active"
                    )
                    return

                if self._current_state != DictationState.IDLE:
                    logger.warning(f"Cannot start session - coordinator not in IDLE state (current: {self._current_state.value})")
                    return

                if mode == DictationMode.SMART:
                    self._last_smart_dictation_text = None

                # Reset text service session to prevent continuation logic from incorrectly lowercasing
                # the first text of a new session. This preserves Whisper's native capitalization.
                self.text_service.reset_session()

                self._current_session = DictationSession(
                    session_id=session_id,
                    mode=mode,
                    start_time=time.time(),
                    accumulated_text="",
                    last_text_time=None,
                    is_first_segment=True,
                )
                self._set_state(DictationState.RECORDING)

            await self._publish_event(AudioModeChangeRequestEvent(mode="dictation", reason=f"{mode.value} mode activated"))
            await self._publish_event(DictationModeDisableOthersEvent(dictation_mode_active=True, dictation_mode=mode.value))

            # Initialize streaming for smart/visual/hidden modes
            if mode in (DictationMode.SMART, DictationMode.VISUAL, DictationMode.HIDDEN):
                self._streaming_buffer = StreamingAudioBuffer(sample_rate=self.config.audio.sample_rate)
                self._streaming_finalized_text = ""
                self._streaming_finalized_segments = []
                self._streaming_segment_id = ""
                self._streaming_current_out = ""
                self._streaming_prev_out = ""
                self._streaming_same_output_count = 0
                self._streaming_end_time_for_same_output = None
                self._streaming_timestamp_offset = 0.0

                # Clear STT context for fresh session
                if self._stt_service:
                    await self._stt_service.clear_streaming_context()

                # Start streaming loop
                self._streaming_task = asyncio.create_task(self._streaming_transcription_loop())
                logger.info(f"Initialized streaming dictation for {mode.value} mode")

            # Emit mode-specific start events
            if mode == DictationMode.SMART:
                await self._publish_event(SmartDictationStartedEvent())
            elif mode == DictationMode.VISUAL:
                await self._publish_event(VisualDictationStartedEvent())
            elif mode == DictationMode.HIDDEN:
                await self._publish_event(HiddenDictationStartedEvent())

            if mode == DictationMode.TYPE:
                self._type_silence_task = asyncio.create_task(self._monitor_type_silence())
                logger.info("Started type dictation silence monitoring task")

            await self._publish_status(True, mode)
            logger.info(f"Started {mode.value} dictation")

        except Exception as e:
            logger.error(f"Session start error: {e}", exc_info=True)
            with self._state_lock:
                self._current_session = None
                self._set_state(DictationState.IDLE)

    async def _stop_session(self) -> None:
        """Stop dictation session with proper cleanup.

        Routes to appropriate handler based on mode:
        - SMART/VISUAL/HIDDEN: Use streaming mode handler
        - STANDARD/TYPE: Use simple VAD-based handler
        """
        try:
            with self._state_lock:
                session = self._current_session
                if not session:
                    return

                if self._current_state == DictationState.PROCESSING_LLM:
                    logger.warning("Stop session called while already processing LLM - ignoring duplicate call")
                    return

                if session.mode == DictationMode.TYPE:
                    self._cancel_type_silence_task()

                # Streaming modes (smart/visual/hidden) handle their own finalization
                if session.mode in (DictationMode.SMART, DictationMode.VISUAL, DictationMode.HIDDEN):
                    await self._stop_streaming_mode(session)
                    return

                # Non-streaming modes (standard/type): simple cleanup
                self._current_session = None
                self._set_state(DictationState.IDLE)

            # Finalize non-streaming session
            if session:
                await self._finalize_session(session)

        except Exception as e:
            logger.error(f"Session stop error: {e}", exc_info=True)
            with self._state_lock:
                self._current_session = None
                self._pending_llm_session = None
                self._set_state(DictationState.IDLE)

    async def _start_llm_processing(self, llm_session: LLMSession) -> None:
        """Actually start the LLM processing after UI is ready"""
        try:
            if hasattr(self.llm_service, "process_dictation_streaming"):
                logger.info("Starting LLM streaming processing...")

                self._start_streaming()

                try:
                    await self.llm_service.process_dictation_streaming(
                        llm_session.raw_text, llm_session.agentic_prompt, token_callback=self._stream_token
                    )
                finally:
                    self._stop_streaming()

                logger.info("LLM streaming processing completed")
            else:
                logger.info("Starting LLM non-streaming processing...")
                await self.llm_service.process_dictation(llm_session.raw_text, llm_session.agentic_prompt)
                logger.info("LLM non-streaming processing completed")

        except Exception as e:
            logger.error(f"LLM processing error: {e}", exc_info=True)
            self._stop_streaming()

    def _stream_token(self, token: str) -> None:
        """Thread-safe callback - queue token for publishing"""
        if self._streaming_active:
            try:
                self._token_queue.put_nowait(token)
                if self._token_queue.qsize() <= 5 or self._token_queue.qsize() % 10 == 0:
                    logger.debug(f"_stream_token: queued token (queue size: {self._token_queue.qsize()}): '{token}'")
            except queue.Full:
                logger.warning("Token queue full - dropping token to prevent blocking")
        else:
            logger.warning(f"_stream_token called but streaming not active! Token: '{token}'")

    def _streaming_worker(self) -> None:
        """Background thread that publishes tokens with proper synchronization"""
        logger.info("Streaming worker thread started")
        published_count = 0
        try:
            while not self._streaming_stop_event.is_set():
                try:
                    token = self._token_queue.get(timeout=0.1)
                    published_count += 1

                    if published_count <= 5 or published_count % 10 == 0:
                        logger.debug(f"_streaming_worker: publishing token #{published_count}: '{token}'")

                    if self._direct_token_callback:
                        try:
                            self._direct_token_callback(token)
                        except Exception as e:
                            logger.error(f"Direct callback error: {e}", exc_info=True)

                    self.event_publisher.publish(LLMTokenGeneratedEvent(token=token))

                except queue.Empty:
                    continue

                except Exception as e:
                    logger.error(f"Streaming worker error: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"Critical error in streaming worker: {e}", exc_info=True)
        finally:
            logger.info(f"Streaming worker thread stopped (published {published_count} tokens)")

    def _start_streaming(self) -> None:
        """Start background streaming thread with proper synchronization"""
        if not self._streaming_active:
            self._streaming_active = True
            self._streaming_stop_event.clear()
            self._streaming_thread = threading.Thread(target=self._streaming_worker, daemon=False, name="LLMTokenStreamer")
            self._streaming_thread.start()
            logger.info("Streaming thread started")

    def _stop_streaming(self) -> None:
        """Stop streaming thread and flush remaining tokens - properly synchronized"""
        logger.info("Stopping streaming thread")
        self._streaming_active = False
        self._streaming_stop_event.set()

        # Wait for thread to finish
        if self._streaming_thread:
            try:
                self._streaming_thread.join(timeout=2.0)
                if self._streaming_thread.is_alive():
                    logger.warning("Streaming thread did not terminate within timeout")
                else:
                    logger.info("Streaming thread terminated successfully")
            except Exception as e:
                logger.error(f"Error joining streaming thread: {e}")
            finally:
                self._streaming_thread = None

        # Flush any remaining tokens
        remaining = []
        while True:
            try:
                token = self._token_queue.get_nowait()
                remaining.append(token)
            except queue.Empty:
                break

        if remaining:
            batched_token = "".join(remaining)
            self.event_publisher.publish(LLMTokenGeneratedEvent(token=batched_token))
            logger.info(f"Flushed {len(remaining)} remaining tokens")

    async def _end_smart_session(self) -> None:
        """End smart dictation session"""
        try:
            await self._publish_event(AudioModeChangeRequestEvent(mode="command", reason="Smart dictation completed"))

            # Notify STT service about dictation mode deactivation
            await self._publish_event(DictationModeDisableOthersEvent(dictation_mode_active=False, dictation_mode="inactive"))

            await self._publish_status(False, DictationMode.INACTIVE)
            logger.info("Smart dictation session ended")
        except Exception as e:
            logger.error(f"Smart session end error: {e}", exc_info=True)

    async def _finalize_session(self, session: DictationSession) -> None:
        """Finalize non-smart session"""
        try:
            await self._publish_event(AudioModeChangeRequestEvent(mode="command", reason="Dictation stopped"))

            # Notify STT service about dictation mode deactivation
            await self._publish_event(DictationModeDisableOthersEvent(dictation_mode_active=False, dictation_mode="inactive"))

            await self._publish_status(False, DictationMode.INACTIVE)
            logger.info(f"{session.mode.value} dictation session completed")
        except Exception as e:
            logger.error(f"Session finalization error: {e}", exc_info=True)

    def _clean_text(self, text: str) -> str:
        """Clean dictated text by removing triggers"""
        if not text:
            return ""

        cfg = self.config.dictation
        triggers = {
            cfg.start_trigger.lower(),
            cfg.stop_trigger.lower(),
            cfg.type_trigger.lower(),
            cfg.smart_start_trigger.lower(),
            cfg.visual_start_trigger.lower(),
            cfg.hidden_start_trigger.lower(),
        }

        words = [w for w in text.split() if w.lower().strip('.,!?;:"()[]{}') not in triggers]
        return " ".join(words).strip()

    def _append_text(self, existing: str, new_text: str) -> str:
        """Append text with proper spacing"""
        if not existing:
            return new_text
        if not new_text:
            return existing
        return f"{existing} {new_text}"

    async def _publish_event(self, event: BaseEvent) -> None:
        """Publish event with error handling"""
        try:
            await self.event_bus.publish(event)
        except Exception as e:
            logger.error(f"Event publishing error: {e}", exc_info=True)

    async def _publish_status(self, is_active: bool, mode: DictationMode) -> None:
        """Publish status change event"""
        try:
            event = DictationStatusChangedEvent(
                is_active=is_active,
                mode=mode.value,
                show_ui=is_active,
                stop_command=self.config.dictation.stop_trigger if is_active else None,
            )
            await self._publish_event(event)
        except Exception as e:
            logger.error(f"Status publishing error: {e}", exc_info=True)

    async def _publish_error(self, message: str) -> None:
        """Log error message"""
        try:
            logger.error(f"Dictation error: {message}")
        except Exception as e:
            logger.error(f"Error logging error: {e}", exc_info=True)

    async def shutdown(self) -> None:
        """Shutdown coordinator with proper resource cleanup"""
        logger.info("Starting dictation coordinator shutdown")
        try:
            with self._state_lock:
                self._set_state(DictationState.SHUTTING_DOWN)
                has_active_session = self._current_session is not None

            # Cancel type silence task
            self._cancel_type_silence_task()

            # Cancel streaming task if active
            if self._streaming_task and not self._streaming_task.done():
                logger.info("Cancelling active streaming task")
                self._streaming_task.cancel()
                try:
                    await self._streaming_task
                except asyncio.CancelledError:
                    logger.info("Streaming task cancelled")
                except Exception as e:
                    logger.warning(f"Error cancelling streaming task: {e}")

            # Cancel LLM processing task if active
            if self._llm_processing_task and not self._llm_processing_task.done():
                logger.info("Cancelling active LLM processing task")
                self._llm_processing_task.cancel()
                try:
                    await asyncio.wait_for(asyncio.shield(self._llm_processing_task), timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    logger.info("LLM processing task cancelled")
                except Exception as e:
                    logger.warning(f"Error cancelling LLM task: {e}")

            # Stop streaming thread (for LLM tokens)
            self._stop_streaming()

            # Clean up streaming buffer
            if self._streaming_buffer:
                await self._streaming_buffer.clear()
                self._streaming_buffer = None

            # Stop current session if active (checked atomically above)
            if has_active_session:
                await self._stop_session()

            # Shutdown services
            await self.text_service.shutdown()
            await self.llm_service.shutdown()
            await self.agentic_service.shutdown()
            await self.alias_service.shutdown()

            # Clear pending sessions under lock
            with self._state_lock:
                self._current_session = None
                self._pending_llm_session = None

            # Force garbage collection
            gc.collect()

            logger.info("Dictation coordinator shutdown complete")
        except Exception as e:
            logger.error(f"Shutdown error: {e}", exc_info=True)
