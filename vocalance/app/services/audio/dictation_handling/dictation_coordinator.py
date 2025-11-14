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
    """

    INACTIVE = "inactive"
    STANDARD = "standard"
    SMART = "smart"
    TYPE = "type"
    VISUAL = "visual"


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

        # Track last text for smart dictation window concatenation logic
        self._last_smart_dictation_text: Optional[str] = None

        # Streaming dictation state for smart/visual modes (WhisperLive approach)
        self._streaming_buffer: Optional[StreamingAudioBuffer] = None
        self._streaming_task: Optional[asyncio.Task] = None
        self._streaming_segment_id: str = ""
        self._streaming_finalized_text: str = ""

        # Same-output detection (WhisperLive approach)
        self._streaming_current_out: str = ""  # Current incomplete segment text
        self._streaming_prev_out: str = ""  # Previous incomplete segment text
        self._streaming_same_output_count: int = 0  # Repetition counter
        self._streaming_end_time_for_same_output: Optional[float] = None  # Timestamp when repetition started

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

                # Skip smart/visual modes - they use streaming flow
                if session.mode in (DictationMode.SMART, DictationMode.VISUAL):
                    logger.debug(f"Skipping VAD-based text for streaming mode: {session.mode.value}")
                    return

            cleaned_text = self._clean_text(text)
            if not cleaned_text:
                return

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
        """Route audio chunks to streaming buffer for smart/visual modes.

        Args:
            event: AudioChunkEvent containing 50ms audio chunk.
        """
        try:
            with self._state_lock:
                session = self._current_session
                buffer = self._streaming_buffer

            # Only process if we're in streaming mode (smart/visual) and have a buffer
            if not session or not buffer:
                return

            if session.mode not in (DictationMode.SMART, DictationMode.VISUAL):
                return

            # Add chunk to streaming buffer
            import numpy as np

            chunk_np = np.frombuffer(event.audio_chunk, dtype=np.int16)
            await buffer.add_chunk(chunk_np)

        except Exception as e:
            logger.error(f"Error handling audio chunk for streaming: {e}", exc_info=True)

    async def _streaming_transcription_loop(self) -> None:
        """500ms streaming transcription loop with WhisperLive segment-based logic.

        Uses segment timestamps from Whisper to:
        - Finalize complete segments immediately (white text)
        - Track incomplete segment for same-output detection (gray text)
        - Advance timestamp after EVERY transcription using segment.end times
        """
        try:
            logger.info("Starting streaming transcription loop (500ms interval with segment timestamps)")

            while True:
                await asyncio.sleep(0.5)  # 500ms interval

                with self._state_lock:
                    session = self._current_session
                    buffer = self._streaming_buffer

                    if not session or not buffer:
                        logger.debug("Streaming loop: No session or buffer, exiting")
                        break

                    if session.mode not in (DictationMode.SMART, DictationMode.VISUAL):
                        logger.debug("Streaming loop: Not in streaming mode, exiting")
                        break

                # Get UNTRANSCRIBED audio from buffer (offset-based)
                audio_result = await buffer.get_audio_for_transcription()

                if not audio_result:
                    # No untranscribed audio available
                    continue

                audio_bytes, duration = audio_result

                if duration < 1.0:
                    # Not enough untranscribed audio yet (minimum 1 second)
                    continue

                # Check for silence timeout (2+ seconds without new audio chunks)
                last_chunk_time = buffer.get_last_chunk_time()
                silence_duration = time.time() - last_chunk_time

                if silence_duration > 2.0:
                    # Finalize any pending incomplete segment before silence
                    if self._streaming_current_out and self._streaming_same_output_count < 3:
                        logger.debug(f"Silence detected ({silence_duration:.1f}s), finalizing pending text")
                        await self._finalize_incomplete_segment()
                    # Continue loop but don't transcribe silence
                    continue

                # Perform streaming recognition - returns segments with timestamps
                if not self._stt_service:
                    logger.warning("STT service not set, cannot perform streaming recognition")
                    continue

                segments, confidence = await self._stt_service.recognize_streaming(
                    audio_bytes=audio_bytes, sample_rate=self.config.audio.sample_rate
                )

                if not segments:
                    # No speech detected - don't advance, wait for actual transcription
                    # (WhisperLive approach: prevents skipping over speech Whisper temporarily missed)
                    continue

                logger.debug(f"Received {len(segments)} segments (confidence: {confidence:.3f})")

                # Track offset for timestamp advancement (offset into current audio chunk)
                offset = None

                # Process complete segments (all except last if multiple segments)
                if len(segments) > 1:
                    for seg in segments[:-1]:
                        if seg["completed"]:
                            # Strip overlap and finalize immediately
                            text = self._strip_overlap(seg["text"])
                            if text:
                                await self._finalize_completed_segment(text, seg["end"])
                            # Use min(duration, seg["end"]) for offset (WhisperLive pattern)
                            offset = min(duration, seg["end"])
                            # Reset segment ID after finalizing completed segment
                            self._streaming_segment_id = ""

                # Process last segment (may be incomplete)
                last_seg = segments[-1]
                self._streaming_current_out = last_seg["text"]

                # Same-output detection (compare RAW text, not stripped)
                if self._streaming_current_out.strip() == self._streaming_prev_out.strip() and self._streaming_current_out:
                    self._streaming_same_output_count += 1

                    # Capture end time when repetition first starts
                    if self._streaming_end_time_for_same_output is None:
                        self._streaming_end_time_for_same_output = last_seg["end"]

                    await asyncio.sleep(0.1)  # Brief wait for voice activity
                else:
                    self._streaming_same_output_count = 0
                    self._streaming_end_time_for_same_output = None

                # If same output repeated 3+ times, finalize it
                if self._streaming_same_output_count >= 3:
                    text = self._strip_overlap(self._streaming_current_out)
                    if text:
                        await self._finalize_completed_segment(text, self._streaming_end_time_for_same_output)

                    self._streaming_current_out = ""
                    # Use min(duration, end_time) for offset (WhisperLive pattern)
                    offset = min(duration, self._streaming_end_time_for_same_output)
                    self._streaming_same_output_count = 0
                    self._streaming_end_time_for_same_output = None
                    # Reset segment ID after finalization
                    self._streaming_segment_id = ""
                else:
                    # Emit as partial (gray text) ONLY IF CHANGED
                    if self._streaming_current_out != self._streaming_prev_out:
                        self._streaming_prev_out = self._streaming_current_out
                        text = self._strip_overlap(self._streaming_current_out)

                        if text:
                            segment_id = self._streaming_segment_id or str(uuid.uuid4())
                            self._streaming_segment_id = segment_id

                            await self._publish_event(PartialDictationTextEvent(text=text, segment_id=segment_id))
                            logger.debug(f"Partial text ({self._streaming_same_output_count}/3): '{text[:50]}...'")
                    else:
                        self._streaming_prev_out = self._streaming_current_out

                # CRITICAL: Advance timestamp after EVERY transcription
                if offset is not None:
                    await buffer.advance_timestamp(offset)
                    logger.debug(f"Advanced timestamp_offset by {offset:.2f}s (of {duration:.2f}s chunk)")

        except asyncio.CancelledError:
            logger.info("Streaming transcription loop cancelled")
        except Exception as e:
            logger.error(f"Error in streaming transcription loop: {e}", exc_info=True)

    async def _finalize_completed_segment(self, text: str, end_time: float) -> None:
        """Finalize a completed segment (white text).

        Args:
            text: Segment text (already stripped of overlap).
            end_time: Segment end timestamp for offset advancement.
        """
        if not text:
            return

        # Use existing segment ID if available (to match partial text), or generate new one
        segment_id = self._streaming_segment_id or str(uuid.uuid4())

        # Emit final event
        await self._publish_event(FinalDictationTextEvent(text=text, segment_id=segment_id))

        logger.info(f"Finalized completed segment: '{text}' (end: {end_time:.2f}s)")

        # Add to STT context for future predictions
        if self._stt_service:
            await self._stt_service.add_finalized_segment(text)

        # Update finalized text accumulator
        if self._streaming_finalized_text:
            self._streaming_finalized_text += " " + text
        else:
            self._streaming_finalized_text = text

    async def _finalize_incomplete_segment(self) -> None:
        """Finalize pending incomplete segment (due to silence or stop).

        Finalizes the current_out text that hasn't reached threshold yet.
        """
        if not self._streaming_current_out:
            return

        text = self._strip_overlap(self._streaming_current_out)
        if not text:
            return

        # Generate segment ID (reuse if exists)
        segment_id = self._streaming_segment_id or str(uuid.uuid4())

        # Emit final event
        await self._publish_event(FinalDictationTextEvent(text=text, segment_id=segment_id))

        logger.info(f"Finalized incomplete segment: '{text}'")

        # Add to STT context
        if self._stt_service:
            await self._stt_service.add_finalized_segment(text)

        # Update finalized text accumulator
        if self._streaming_finalized_text:
            self._streaming_finalized_text += " " + text
        else:
            self._streaming_finalized_text = text

        # Reset state
        self._streaming_current_out = ""
        self._streaming_prev_out = ""
        self._streaming_same_output_count = 0
        self._streaming_segment_id = ""

    def _strip_overlap(self, text: str) -> str:
        """Strip overlapping prefix that matches finalized text.

        Args:
            text: New predicted text that may overlap with finalized text.

        Returns:
            Text with overlapping prefix removed.
        """
        if not self._streaming_finalized_text or not text:
            return text

        # Simple word-level overlap detection
        finalized_words = self._streaming_finalized_text.lower().split()
        text_words = text.split()
        text_words_lower = [w.lower() for w in text_words]

        # Find longest matching suffix of finalized that matches prefix of new text
        max_overlap = min(len(finalized_words), len(text_words))
        overlap_length = 0

        for i in range(1, max_overlap + 1):
            # Check if last i words of finalized match first i words of new text
            if finalized_words[-i:] == text_words_lower[:i]:
                overlap_length = i

        if overlap_length > 0:
            # Remove overlapping prefix
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

        Args:
            session: Current dictation session.
        """
        try:
            # Cancel streaming task
            if self._streaming_task and not self._streaming_task.done():
                self._streaming_task.cancel()
                try:
                    await self._streaming_task
                except asyncio.CancelledError:
                    pass
                self._streaming_task = None

            # Finalize any remaining incomplete segment
            if self._streaming_current_out:
                await self._finalize_incomplete_segment()

            # Get all finalized text
            final_text = self._streaming_finalized_text

            # Clean up streaming state
            if self._streaming_buffer:
                await self._streaming_buffer.clear()
                self._streaming_buffer = None

            self._streaming_finalized_text = ""
            self._streaming_segment_id = ""
            self._streaming_current_out = ""
            self._streaming_prev_out = ""
            self._streaming_same_output_count = 0
            self._streaming_end_time_for_same_output = None

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
                    # Visual mode or no text: finalize directly
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

            logger.info(f"Streaming {session.mode.value} mode stopped, finalized text: {len(final_text)} chars")

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

            # Initialize streaming for smart/visual modes
            if mode in (DictationMode.SMART, DictationMode.VISUAL):
                self._streaming_buffer = StreamingAudioBuffer(sample_rate=self.config.audio.sample_rate)
                self._streaming_finalized_text = ""
                self._streaming_segment_id = ""
                self._streaming_current_out = ""
                self._streaming_prev_out = ""
                self._streaming_same_output_count = 0
                self._streaming_end_time_for_same_output = None

                # Clear STT context for fresh session
                if self._stt_service:
                    await self._stt_service.clear_streaming_context()

                # Start streaming loop
                self._streaming_task = asyncio.create_task(self._streaming_transcription_loop())
                logger.info(f"Initialized streaming dictation for {mode.value} mode")

            if mode == DictationMode.SMART:
                await self._publish_event(SmartDictationStartedEvent())

            if mode == DictationMode.VISUAL:
                await self._publish_event(VisualDictationStartedEvent())

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
        """Stop dictation session with proper cleanup"""
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

                # Stop streaming for smart/visual modes
                if session.mode in (DictationMode.SMART, DictationMode.VISUAL):
                    await self._stop_streaming_mode(session)
                    return  # Early return - streaming modes handle their own finalization

                if session.mode == DictationMode.SMART:
                    if session.accumulated_text:
                        self._set_state(DictationState.PROCESSING_LLM)

                        agentic_prompt = self.agentic_service.get_current_prompt() or "Fix grammar and improve clarity."
                        llm_session_id = str(uuid.uuid4())
                        self._pending_llm_session = LLMSession(
                            session_id=llm_session_id,
                            raw_text=session.accumulated_text,
                            agentic_prompt=agentic_prompt,
                        )
                    else:
                        self._current_session = None
                        self._set_state(DictationState.IDLE)
                elif session.mode == DictationMode.VISUAL:
                    # Visual mode: accumulate text and paste directly without LLM
                    self._current_session = None
                    self._set_state(DictationState.IDLE)
                else:
                    self._current_session = None
                    self._set_state(DictationState.IDLE)

            if session and session.mode == DictationMode.VISUAL:
                # Visual mode: paste accumulated text if any, then finalize
                if session.accumulated_text:
                    logger.info(f"Publishing VisualDictationStoppedEvent with text: '{session.accumulated_text[:50]}...'")
                    await self._publish_event(VisualDictationStoppedEvent(accumulated_text=session.accumulated_text))
                    logger.info("VisualDictationStoppedEvent published - pasting accumulated text")
                    await self.text_service.input_text(session.accumulated_text)
                else:
                    logger.info("Visual dictation stopped with no accumulated text")
                    await self._publish_event(VisualDictationStoppedEvent(accumulated_text=""))
                # Always finalize the session to return to command mode
                await self._finalize_session(session)
            elif session and session.mode == DictationMode.SMART and session.accumulated_text:
                await self._publish_event(AudioModeChangeRequestEvent(mode="command", reason="Smart dictation processing"))

                logger.info(f"Publishing SmartDictationStoppedEvent with text: '{session.accumulated_text[:50]}...'")
                await self._publish_event(SmartDictationStoppedEvent(raw_text=session.accumulated_text))
                logger.info("SmartDictationStoppedEvent published successfully")

                logger.info(f"Publishing LLMProcessingStartedEvent with session ID: {self._pending_llm_session.session_id}")
                await self._publish_event(
                    LLMProcessingStartedEvent(
                        raw_text=session.accumulated_text,
                        agentic_prompt=self._pending_llm_session.agentic_prompt,
                        session_id=self._pending_llm_session.session_id,
                    )
                )
                logger.info("LLMProcessingStartedEvent published - waiting for UI ready signal...")
            elif session:
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
            except queue.Full:
                logger.warning("Token queue full - dropping token to prevent blocking")

    def _streaming_worker(self) -> None:
        """Background thread that publishes tokens with proper synchronization"""
        logger.info("Streaming worker thread started")
        try:
            while not self._streaming_stop_event.is_set():
                try:
                    token = self._token_queue.get(timeout=0.1)

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
            logger.info("Streaming worker thread stopped")

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

            # Clear pending sessions under lock
            with self._state_lock:
                self._current_session = None
                self._pending_llm_session = None

            # Force garbage collection
            gc.collect()

            logger.info("Dictation coordinator shutdown complete")
        except Exception as e:
            logger.error(f"Shutdown error: {e}", exc_info=True)
