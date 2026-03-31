"""Coordinates dictation sessions, Moonshine streaming, modifiers, LLM handoff, and text output."""

import asyncio
import gc
import logging
import queue
import re
import threading
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.config.command_types import (
    DictationHiddenStartCommand,
    DictationAmendStartCommand,
    DictationSmartStartCommand,
    DictationStartCommand,
    DictationStopCommand,
    DictationTypeCommand,
    DictationVisualStartCommand,
)
from vocalance.app.event_bus import EventBus
from vocalance.app.events.base_event import BaseEvent
from vocalance.app.events.command_events import DictationCommandParsedEvent
from vocalance.app.events.core_events import DictationTextRecognizedEvent
from vocalance.app.events.dictation_events import (
    AudioModeChangeRequestEvent,
    DictationModeDisableOthersEvent,
    DictationModifierId,
    DictationModifierPhraseEvent,
    DictationModifierStateChangedEvent,
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
    SmartDictationStartedEvent,
    SmartDictationStoppedEvent,
    VisualDictationStartedEvent,
    VisualDictationStoppedEvent,
)
from vocalance.app.services.audio.dictation_handling.dictation_alias_service import DictationAliasService
from vocalance.app.services.audio.dictation_handling.llm_support.agentic_prompt_service import AgenticPromptService
from vocalance.app.services.audio.dictation_handling.llm_support.llm_service import LLMService
from vocalance.app.services.audio.dictation_handling.dictation_postprocess import (
    apply_dictation_postprocess,
    apply_dictation_postprocess_partial,
    modifier_display_label,
)
from vocalance.app.services.audio.dictation_handling.text_input_service import TextInputService, remove_formatting
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
    AMEND: Copy selection to clipboard, dictate instructions, LLM applies them to the selection.
    """

    INACTIVE = "inactive"
    STANDARD = "standard"
    SMART = "smart"
    TYPE = "type"
    VISUAL = "visual"
    HIDDEN = "hidden"
    AMEND = "amend"


# Modes that stream PCM into Moonshine via the dedicated ingress thread (not the event bus).
MOONSHINE_CHUNK_DICTATION_MODES = frozenset(
    {
        DictationMode.STANDARD,
        DictationMode.TYPE,
        DictationMode.SMART,
        DictationMode.VISUAL,
        DictationMode.HIDDEN,
        DictationMode.AMEND,
    }
)
# Modes that use partial/final dictation events (skip duplicate DictationTextRecognized from segment path).
_STREAMING_STT_MODES = frozenset(
    {DictationMode.SMART, DictationMode.VISUAL, DictationMode.HIDDEN, DictationMode.AMEND}
)
# Subset that hands off to the LLM after stop.
_STREAMING_LLM_MODES = frozenset({DictationMode.SMART, DictationMode.AMEND})

MOONSHINE_MODIFIER_SUPPRESS_SEC: float = 0.55


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
        active_modifier: Optional voice post-process modifier (at most one).
    """

    session_id: str
    mode: DictationMode
    start_time: float
    accumulated_text: str = ""
    last_text_time: Optional[float] = None
    is_first_segment: bool = True
    active_modifier: Optional[DictationModifierId] = None


@dataclass
class LLMSession:
    """Immutable LLM processing session for state isolation.

    Attributes:
        session_id: Unique LLM session identifier.
        raw_text: Raw dictation text, or spoken instructions when ``clipboard_text`` is set.
        agentic_prompt: Generated agentic prompt for LLM.
        clipboard_text: If set, amend path: text captured from selection before dictation.
    """

    session_id: str
    raw_text: str
    agentic_prompt: str
    clipboard_text: Optional[str] = None


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

        self._state_lock = threading.RLock()

        self._current_state = DictationState.IDLE
        self._current_session: Optional[DictationSession] = None
        self._pending_llm_session: Optional[LLMSession] = None
        self._type_silence_task: Optional[asyncio.Task] = None
        self._llm_processing_task: Optional[asyncio.Task] = None

        self._token_queue: queue.Queue = queue.Queue(maxsize=1000)
        self._streaming_active = False
        self._streaming_stop_event = threading.Event()
        self._streaming_thread: Optional[threading.Thread] = None
        self._direct_token_callback: Optional[callable] = None

        self.text_service = TextInputService(config=config.dictation)
        self.llm_service = LLMService(event_bus=event_bus, config=config)
        self.agentic_service = AgenticPromptService(event_bus=event_bus, config=config, storage=storage)
        self.alias_service = DictationAliasService(event_bus=event_bus, storage=storage, event_loop=gui_event_loop)

        self._last_smart_dictation_text: Optional[str] = None
        self._amend_clipboard_snapshot: Optional[str] = None

        self._moonshine_session = None
        self._streaming_finalized_text: str = ""
        self._streaming_finalized_segments: list[str] = []

        self._stt_service = None

        # Moonshine audio is fed from the recorder thread via an unbounded queue and a dedicated
        # worker thread. Feeding through the asyncio event bus caused 100–500ms stalls per chunk.
        # We do not drop PCM when the ingress thread is briefly behind (accuracy over bounded memory).
        self._moonshine_ingress_queue: queue.Queue = queue.Queue()
        self._moonshine_ingress_stop = threading.Event()
        self._moonshine_ingress_thread: Optional[threading.Thread] = None
        # Serializes ingress drain/stop vs add_audio; never acquire _state_lock before this lock.
        self._moonshine_feed_lock = threading.Lock()
        # Bumped whenever the Moonshine stream is opened or torn down so stale chunks still
        # in the ingress queue cannot be fed into a new session (stop/start race).
        self._moonshine_ingress_epoch: int = 0
        self._moonshine_suppress_until: float = 0.0
        self._start_moonshine_ingress_thread()

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

    def _start_moonshine_ingress_thread(self) -> None:
        if self._moonshine_ingress_thread is not None and self._moonshine_ingress_thread.is_alive():
            return
        self._moonshine_ingress_stop.clear()
        self._moonshine_ingress_thread = threading.Thread(
            target=self._moonshine_ingress_loop, name="MoonshineAudioIngress", daemon=True
        )
        self._moonshine_ingress_thread.start()
        logger.debug("Moonshine audio ingress thread started")

    def _moonshine_ingress_loop(self) -> None:
        """Drain PCM chunks from the recorder thread and call Moonshine on a dedicated thread."""
        logger.debug("Moonshine audio ingress thread running")
        while not self._moonshine_ingress_stop.is_set():
            try:
                stamped = self._moonshine_ingress_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                epoch, audio_bytes, sample_rate = stamped
                rotate = False
                with self._moonshine_feed_lock:
                    with self._state_lock:
                        if epoch != self._moonshine_ingress_epoch:
                            continue
                        ms = self._moonshine_session
                    if ms is not None:
                        rotate = ms.add_audio_pcm16(audio_bytes, sample_rate)
                if rotate:
                    self._moonshine_rotate_line(reason="max_line_duration")
            except Exception as e:
                logger.error("Moonshine ingress feed error: %s", e, exc_info=True)
        logger.debug("Moonshine audio ingress thread exiting")

    def _moonshine_rotate_line(self, reason: str = "max_line_duration") -> None:
        """Stop the current Moonshine stream and open a new one (bounded line length for latency).

        Must run on the Moonshine audio ingress thread (same thread that calls ``add_audio_pcm16``).
        """
        loop = self.gui_event_loop
        if loop is None:
            logger.error("Moonshine stream rotation skipped: gui_event_loop is not set")
            return

        logger.info("Moonshine stream rotation (%s) — starting new native stream", reason)

        with self._moonshine_feed_lock:
            self._moonshine_ingress_epoch += 1
            self._drain_moonshine_ingress_queue()
            old = self._moonshine_session
            self._moonshine_session = None

        if old is not None:
            try:
                old.stop()
            except Exception as e:
                logger.warning("Moonshine stream rotation stop failed: %s", e, exc_info=True)

        with self._state_lock:
            session = self._current_session
            state = self._current_state

        if (
            session is None
            or session.mode not in MOONSHINE_CHUNK_DICTATION_MODES
            or state != DictationState.RECORDING
        ):
            return

        if not self._stt_service or not self._stt_service.moonshine_engine:
            return

        try:
            new_sess = self._stt_service.moonshine_engine.open_dictation_stream(
                loop,
                self._moonshine_on_partial,
                self._moonshine_on_final,
            )
        except Exception as e:
            logger.error("Moonshine stream rotation reopen failed: %s", e, exc_info=True)
            return

        with self._moonshine_feed_lock:
            self._moonshine_session = new_sess
            self._moonshine_ingress_epoch += 1

    def _drain_moonshine_ingress_queue(self) -> None:
        while True:
            try:
                self._moonshine_ingress_queue.get_nowait()
            except queue.Empty:
                break

    def feed_moonshine_audio_chunk(self, audio_bytes: bytes, sample_rate: int) -> None:
        """Hot path from the audio recorder thread: queue PCM for Moonshine.

        Uses an unbounded queue so we never drop audio for accuracy; memory grows only if the
        ingress thread falls behind for an extended time.
        """
        if not audio_bytes:
            return
        with self._state_lock:
            if self._current_state == DictationState.SHUTTING_DOWN:
                return
            session = self._current_session
            if session is None or session.mode not in MOONSHINE_CHUNK_DICTATION_MODES:
                return
            if self._moonshine_session is None:
                return
            epoch = self._moonshine_ingress_epoch
        self._moonshine_ingress_queue.put((epoch, audio_bytes, sample_rate))

    def _get_state(self) -> DictationState:
        """Thread-safe state getter"""
        with self._state_lock:
            return self._current_state

    def _set_state(self, new_state: DictationState) -> None:
        """Thread-safe state setter with validation.

        Must be called with _state_lock held to ensure atomic transitions.
        Validates transition against _VALID_TRANSITIONS state machine.

        Args:
            new_state: Target DictationState to transition to.

        Raises:
            ValueError: If transition is not valid for current state.
        """
        old_state = self._current_state

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
            (LLMProcessingCompletedEvent, self._handle_llm_completed),
            (LLMProcessingFailedEvent, self._handle_llm_failed),
            (DictationCommandParsedEvent, self._handle_dictation_command),
            (LLMProcessingReadyEvent, self._handle_llm_processing_ready),
            (DictationModifierPhraseEvent, self._handle_dictation_modifier_phrase),
        ]

        for event_type, handler in subscriptions:
            self.subscription_manager.subscribe(event_type, handler)

        logger.info("Event subscriptions configured")

    async def _handle_dictation_modifier_phrase(self, event: DictationModifierPhraseEvent) -> None:
        """Apply a Vosk-side modifier phrase: toggle off if repeated, else switch, publish UI state, suppress Moonshine."""
        try:
            with self._state_lock:
                session = self._current_session
                if not session or self._current_state != DictationState.RECORDING:
                    return
                mid = event.modifier_id
                current = session.active_modifier
                if current == mid:
                    new_mod: Optional[DictationModifierId] = None
                    label = ""
                    active = False
                else:
                    new_mod = mid
                    label = f"{modifier_display_label(mid)} active"
                    active = True
                self._current_session = DictationSession(
                    session_id=session.session_id,
                    mode=session.mode,
                    start_time=session.start_time,
                    accumulated_text=session.accumulated_text,
                    last_text_time=session.last_text_time,
                    is_first_segment=session.is_first_segment,
                    active_modifier=new_mod,
                )

            await self._publish_event(
                DictationModifierStateChangedEvent(active=active, modifier_id=new_mod, display_label=label)
            )
            logger.info("Dictation modifier: %s -> %s", current, new_mod)
            if session.mode in MOONSHINE_CHUNK_DICTATION_MODES:
                self._moonshine_suppress_until = time.monotonic() + MOONSHINE_MODIFIER_SUPPRESS_SEC
        except Exception as e:
            logger.error("Modifier phrase handling error: %s", e, exc_info=True)

    @staticmethod
    def _is_isolated_stt_noise_fragment(text: str) -> bool:
        """True for tiny punctuation-only tails Moonshine may emit after command-only modifier audio."""
        t = text.strip()
        if not t:
            return True
        if t in ("?", "？", "¿", "\ufffd", ""):
            return True
        if len(t) <= 2 and all(not (c.isalnum() or c == "_") for c in t):
            return True
        return False

    def _prepare_dictation_segment_final(self, raw_text: str, session: DictationSession) -> str:
        """Strip triggers, aliases, base number + modifier post-process (final/stable text)."""
        cleaned = self._clean_text(raw_text)
        if not cleaned or self._is_isolated_stt_noise_fragment(cleaned):
            return ""
        with_subs = self.alias_service.apply_substitutions(cleaned)
        if self._is_isolated_stt_noise_fragment(with_subs):
            return ""
        return apply_dictation_postprocess(with_subs, session.active_modifier)

    def _prepare_dictation_segment_partial(self, raw_text: str, session: DictationSession) -> str:
        """Same cleaning and filtering as finals, but spelling modifier is deferred (streaming UI)."""
        cleaned = self._clean_text(raw_text)
        if not cleaned or self._is_isolated_stt_noise_fragment(cleaned):
            return ""
        with_subs = self.alias_service.apply_substitutions(cleaned)
        if self._is_isolated_stt_noise_fragment(with_subs):
            return ""
        return apply_dictation_postprocess_partial(with_subs, session.active_modifier)

    async def _publish_modifier_cleared(self) -> None:
        """Emit inactive modifier state (session end and similar)."""
        await self._publish_event(
            DictationModifierStateChangedEvent(active=False, modifier_id=None, display_label="")
        )

    @staticmethod
    def _dictation_segment_input_options(
        mode: DictationMode, modifier: Optional[DictationModifierId]
    ) -> tuple[bool, bool]:
        """Return ``(add_trailing_space, skip_prose_segment_join_rules)`` for :meth:`TextInputService.input_text`.

        Camel, snake, and spelling modifiers disable trailing spaces and prose join rules (period removal
        and forced lowercase after a non-sentence boundary) so identifiers and spoken punctuation stay intact.
        """
        skip_join = modifier in ("camel", "snake", "spelling")
        add_trailing = mode != DictationMode.TYPE and not skip_join
        return add_trailing, skip_join

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

                if session.mode in _STREAMING_STT_MODES:
                    logger.debug("Skipping VAD-based text for Moonshine chunk-stream mode: %s", session.mode.value)
                    return

            cleaned_text = self._prepare_dictation_segment_final(text, session)
            if not cleaned_text:
                return

            updated_session = DictationSession(
                session_id=session.session_id,
                mode=session.mode,
                start_time=session.start_time,
                accumulated_text=self._append_text(session.accumulated_text, cleaned_text),
                last_text_time=time.time() if session.mode == DictationMode.TYPE else None,
                is_first_segment=False,
                active_modifier=session.active_modifier,
            )

            with self._state_lock:
                if self._current_session and self._current_session.session_id == session.session_id:
                    self._current_session = updated_session
                else:
                    return

            add_trailing, skip_join = self._dictation_segment_input_options(
                updated_session.mode, updated_session.active_modifier
            )
            await self.text_service.input_text(
                text=cleaned_text,
                add_trailing_space=add_trailing,
                skip_prose_segment_join_rules=skip_join,
            )

        except Exception as e:
            logger.error(f"Dictation text error: {e}", exc_info=True)

    async def _cleanup_llm_session(self) -> None:
        """Common cleanup for LLM session completion or failure"""
        with self._state_lock:
            self._current_session = None
            self._pending_llm_session = None
            self._llm_processing_task = None
            self._last_smart_dictation_text = None
            self._amend_clipboard_snapshot = None
            self._set_state(DictationState.IDLE)
        await self._publish_modifier_cleared()
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
        await self._publish_error("LLM processing failed")

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
            elif isinstance(command, DictationAmendStartCommand):
                await self._start_amend_session()

        except Exception as e:
            logger.error(f"Command handling error: {e}", exc_info=True)

    async def _start_amend_session(self) -> None:
        """Copy the foreground selection via Ctrl+C, then start streaming amend dictation."""
        loop = asyncio.get_event_loop()
        captured = await loop.run_in_executor(None, self.text_service.capture_selection_via_copy)
        if not captured or not captured.strip():
            logger.warning("Amend mode: no text captured from foreground selection")
            await self._publish_error("Amend: no text captured — keep focus on the app with the selection")
            return
        with self._state_lock:
            self._amend_clipboard_snapshot = captured.strip()
        await self._start_session(DictationMode.AMEND)

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

    def _moonshine_output_suppressed(self) -> bool:
        """Whether Moonshine partial/final handlers should drop output (post-modifier window)."""
        return time.monotonic() < self._moonshine_suppress_until

    async def _moonshine_on_partial(self, text: str, segment_id: str) -> None:
        """Moonshine line text update → partial dictation UI (smart/visual/amend)."""
        if self._moonshine_output_suppressed():
            return
        with self._state_lock:
            session = self._current_session

        if not session or session.mode not in MOONSHINE_CHUNK_DICTATION_MODES:
            return

        if session.mode in (DictationMode.HIDDEN, DictationMode.STANDARD, DictationMode.TYPE):
            return

        if self._current_state != DictationState.RECORDING:
            return

        if self._is_hallucination(text, ""):
            return

        with self._state_lock:
            live = self._current_session
            if not live or live.session_id != session.session_id:
                return
            session = live
        partial_text = self._prepare_dictation_segment_partial(text, session)
        if not partial_text:
            return
        await self._publish_event(PartialDictationTextEvent(text=partial_text, segment_id=segment_id))

    async def _moonshine_on_final(self, text: str, segment_id: str) -> None:
        """Moonshine completed line → segment typing or finalized chunk for LLM modes."""
        if self._moonshine_output_suppressed():
            return
        with self._state_lock:
            session = self._current_session

        if not session or session.mode not in MOONSHINE_CHUNK_DICTATION_MODES:
            return

        if self._current_state != DictationState.RECORDING:
            return

        if self._is_hallucination(text, ""):
            return

        line = text.strip()
        if not line:
            return

        if session.mode in (DictationMode.STANDARD, DictationMode.TYPE):
            await self._publish_event(
                DictationTextRecognizedEvent(
                    text=line,
                    processing_time_ms=0.0,
                    engine="moonshine",
                    mode="dictation",
                )
            )
            return

        with self._state_lock:
            live = self._current_session
            if not live or live.session_id != session.session_id:
                return
            session = live
        await self._emit_final_text_append(line, segment_id, session)

    async def _emit_final_text_append(self, text: str, segment_id: str, session) -> None:
        """Emit final text and append to accumulator (prevents data loss)."""
        if not text or not text.strip():
            return

        raw_line = text.strip()

        with self._state_lock:
            live = self._current_session
            if not live or live.session_id != session.session_id:
                return
            session = live

        processed = self._prepare_dictation_segment_final(raw_line, session)
        if not processed:
            return

        if self._streaming_finalized_segments:
            if self._streaming_finalized_segments[-1].strip().lower() == processed.lower():
                return

        if session.mode != DictationMode.HIDDEN:
            await self._publish_event(
                FinalDictationTextEvent(text=processed, segment_id=segment_id or str(uuid.uuid4()))
            )

        self._streaming_finalized_segments.append(processed)

        if self._streaming_finalized_text:
            self._streaming_finalized_text += " " + processed
        else:
            self._streaming_finalized_text = processed

    def _is_hallucination(self, text: str, prev_text: str = "") -> bool:
        """Detect likely ASR hallucination patterns (repeated short words or character spam)."""
        if not text or len(text) < 3:
            return False

        words = text.split()
        if len(words) > 10:
            last_words = words[-10:]
            unique_words = set(last_words)
            if len(unique_words) <= 2 and all(len(w) <= 2 for w in unique_words):
                return True

        if prev_text and not any(ord(c) > 127 for c in prev_text):
            ascii_count = sum(1 for c in text if ord(c) < 128)
            if len(text) > 10 and ascii_count < len(text) * 0.3:
                return True

        return False

    def _remove_stop_word(self, text: str) -> str:
        """Remove stop trigger word from text (hidden mode only)."""
        stop_word = self.config.dictation.stop_trigger
        if not stop_word or not text:
            return text

        import re

        pattern = r"\b" + re.escape(stop_word) + r"\b"
        result = re.sub(pattern, "", text, flags=re.IGNORECASE)
        return " ".join(result.split())

    async def _stop_streaming_mode(self, session: DictationSession) -> None:
        """Stop Moonshine chunk stream and finalize transcription for supported modes."""
        try:
            with self._moonshine_feed_lock:
                self._moonshine_ingress_epoch += 1
                self._drain_moonshine_ingress_queue()
                if self._moonshine_session:
                    self._moonshine_session.stop()
                    self._moonshine_session = None

            final_text = self._streaming_finalized_text

            if session.mode in (DictationMode.STANDARD, DictationMode.TYPE):
                with self._state_lock:
                    self._current_session = None
                    self._set_state(DictationState.IDLE)
                self._streaming_finalized_text = ""
                self._streaming_finalized_segments = []
                await self._finalize_session(session)
                logger.info("Moonshine chunk session stopped (%s)", session.mode.value)
                return

            if session.mode in (DictationMode.HIDDEN, DictationMode.AMEND) and final_text:
                final_text = self._remove_stop_word(final_text)
            if final_text:
                final_text = self.alias_service.apply_substitutions(final_text)
                final_text = " ".join(final_text.split())

            self._streaming_finalized_text = ""
            self._streaming_finalized_segments = []

            amend_clipboard_error = False
            with self._state_lock:
                if session.mode in _STREAMING_LLM_MODES and final_text:
                    if session.mode == DictationMode.AMEND and not self._amend_clipboard_snapshot:
                        logger.error("Amend mode: clipboard snapshot missing")
                        self._current_session = None
                        self._set_state(DictationState.IDLE)
                        amend_clipboard_error = True
                    else:
                        self._set_state(DictationState.PROCESSING_LLM)
                        default_prompt = (
                            "Fix grammar and improve clarity."
                            if session.mode is DictationMode.SMART
                            else "Follow the spoken instructions when transforming the text."
                        )
                        agentic_prompt = self.agentic_service.get_current_prompt() or default_prompt
                        llm_session_id = str(uuid.uuid4())
                        self._pending_llm_session = LLMSession(
                            session_id=llm_session_id,
                            raw_text=final_text,
                            agentic_prompt=agentic_prompt,
                            clipboard_text=self._amend_clipboard_snapshot if session.mode == DictationMode.AMEND else None,
                        )
                else:
                    self._current_session = None
                    self._set_state(DictationState.IDLE)

            if amend_clipboard_error:
                await self._publish_modifier_cleared()

            if session.mode in _STREAMING_LLM_MODES and final_text and self._pending_llm_session:
                await self._publish_modifier_cleared()
                dual = "amend" if session.mode is DictationMode.AMEND else "smart"
                reason = "Amend mode LLM processing" if dual == "amend" else "Smart dictation processing"
                await self._publish_event(AudioModeChangeRequestEvent(mode="command", reason=reason))
                await self._publish_event(SmartDictationStoppedEvent(raw_text=final_text, mode=dual))
                await self._publish_event(
                    LLMProcessingStartedEvent(
                        raw_text=final_text,
                        agentic_prompt=self._pending_llm_session.agentic_prompt,
                        session_id=self._pending_llm_session.session_id,
                    )
                )
            elif session.mode in _STREAMING_LLM_MODES:
                await self._end_smart_session()
            elif session.mode == DictationMode.VISUAL:
                if final_text:
                    await self._publish_event(VisualDictationStoppedEvent(accumulated_text=final_text))
                    await self.text_service.input_text(final_text)
                else:
                    await self._publish_event(VisualDictationStoppedEvent(accumulated_text=""))
                await self._finalize_session(session)
            elif session.mode == DictationMode.HIDDEN:
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
            with self._state_lock:
                self._current_session = None
                self._set_state(DictationState.IDLE)
            await self._publish_modifier_cleared()

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

                if mode != DictationMode.AMEND:
                    self._amend_clipboard_snapshot = None

                self.text_service.reset_session()

                self._current_session = DictationSession(
                    session_id=session_id,
                    mode=mode,
                    start_time=time.time(),
                    accumulated_text="",
                    last_text_time=None,
                    is_first_segment=True,
                    active_modifier=None,
                )
                self._set_state(DictationState.RECORDING)

            await self._publish_event(AudioModeChangeRequestEvent(mode="dictation", reason=f"{mode.value} mode activated"))
            await self._publish_event(DictationModeDisableOthersEvent(dictation_mode_active=True, dictation_mode=mode.value))

            if mode in MOONSHINE_CHUNK_DICTATION_MODES:
                self._streaming_finalized_text = ""
                self._streaming_finalized_segments = []
                if self._stt_service and self._stt_service.moonshine_engine:
                    self._moonshine_session = self._stt_service.moonshine_engine.open_dictation_stream(
                        asyncio.get_running_loop(),
                        self._moonshine_on_partial,
                        self._moonshine_on_final,
                    )
                    self._moonshine_ingress_epoch += 1
                    logger.info("Initialized Moonshine dictation stream for %s mode", mode.value)
                else:
                    logger.error("Moonshine engine unavailable — cannot start chunk dictation for %s", mode.value)

            if mode == DictationMode.SMART:
                await self._publish_event(SmartDictationStartedEvent())
            elif mode == DictationMode.AMEND:
                await self._publish_event(SmartDictationStartedEvent(mode="amend"))
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
                self._amend_clipboard_snapshot = None
                self._set_state(DictationState.IDLE)

    async def _stop_session(self) -> None:
        """Stop dictation session with proper cleanup.

        Routes to appropriate handler based on mode:
        - Chunk-stream modes (see MOONSHINE_CHUNK_DICTATION_MODES): Moonshine stop + mode-specific finalize
        - Other modes: simple VAD-based handler
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

                if session.mode in MOONSHINE_CHUNK_DICTATION_MODES:
                    await self._stop_streaming_mode(session)
                    return

                self._current_session = None
                self._set_state(DictationState.IDLE)

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
                    if llm_session.clipboard_text is not None:
                        await self.llm_service.process_amend_streaming(
                            llm_session.clipboard_text,
                            llm_session.raw_text,
                            llm_session.agentic_prompt,
                            token_callback=self._stream_token,
                        )
                    else:
                        await self.llm_service.process_dictation_streaming(
                            llm_session.raw_text, llm_session.agentic_prompt, token_callback=self._stream_token
                        )
                finally:
                    self._stop_streaming()

                logger.info("LLM streaming processing completed")
            else:
                logger.info("Starting LLM non-streaming processing...")
                if llm_session.clipboard_text is not None:
                    await self.llm_service.process_amend_streaming(
                        llm_session.clipboard_text,
                        llm_session.raw_text,
                        llm_session.agentic_prompt,
                        None,
                    )
                else:
                    await self.llm_service.process_dictation(llm_session.raw_text, llm_session.agentic_prompt)
                logger.info("LLM non-streaming processing completed")

        except Exception as e:
            logger.error(f"LLM processing error: {e}", exc_info=True)
            self._stop_streaming()

    def _stream_token(self, token: str) -> None:
        """Thread-safe callback to queue token for publishing.

        Args:
            token: Token string to publish asynchronously.

        Logs warning if streaming inactive or queue full.
        """
        if self._streaming_active:
            try:
                self._token_queue.put_nowait(token)
            except queue.Full:
                logger.warning("Token queue full - dropping token to prevent blocking")
        else:
            logger.warning(f"_stream_token called but streaming not active! Token: '{token}'")

    def _streaming_worker(self) -> None:
        """Background thread that publishes tokens from queue.

        Runs until stopped, dequeuing tokens and publishing via event bus or direct callback.
        Tolerates queue underflows gracefully and logs exceptional errors.
        """
        logger.info("Streaming worker thread started")
        published_count = 0
        try:
            while not self._streaming_stop_event.is_set():
                try:
                    token = self._token_queue.get(timeout=0.1)
                    published_count += 1

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
        """Stop streaming thread and flush remaining tokens"""
        logger.info("Stopping streaming thread")
        self._streaming_active = False
        self._streaming_stop_event.set()

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
            await self._publish_modifier_cleared()
            await self._publish_event(AudioModeChangeRequestEvent(mode="command", reason="Dictation stopped"))

            # Notify STT service about dictation mode deactivation
            await self._publish_event(DictationModeDisableOthersEvent(dictation_mode_active=False, dictation_mode="inactive"))

            await self._publish_status(False, DictationMode.INACTIVE)
            logger.info(f"{session.mode.value} dictation session completed")
        except Exception as e:
            logger.error(f"Session finalization error: {e}", exc_info=True)

    @staticmethod
    def _strip_config_phrases_case_insensitive(text: str, phrases: tuple[str, ...]) -> str:
        """Remove each non-empty phrase as a whole token run (word boundaries); case-insensitive.

        Multi-word phrases are stripped before single-word ones so e.g. ``smart green`` is removed
        entirely and the embedded ``green`` start trigger does not fire first.
        """
        s = " ".join(text.split()).strip()
        if not s:
            return ""
        nonempty = [p.strip() for p in phrases if p and p.strip()]
        nonempty.sort(key=lambda p: (len(p.split()), len(p)), reverse=True)
        for p in nonempty:
            pat = r"(?i)\b" + re.escape(p) + r"\b"
            s = re.sub(pat, " ", s)
            s = " ".join(s.split()).strip()
        return s

    def _clean_text(self, text: str) -> str:
        """Remove configured dictation triggers and modifier phrases (exact phrase matches only)."""
        if not text:
            return ""

        cfg = self.config.dictation
        trigger_phrases = (
            cfg.start_trigger,
            cfg.stop_trigger,
            cfg.type_trigger,
            cfg.smart_start_trigger,
            cfg.visual_start_trigger,
            cfg.hidden_start_trigger,
            cfg.amend_start_trigger,
        )
        s = self._strip_config_phrases_case_insensitive(text, trigger_phrases)

        modifier_phrases = (
            cfg.modifier_upper_phrase,
            cfg.modifier_capitals_phrase,
            cfg.modifier_camel_phrase,
            cfg.modifier_snake_phrase,
            cfg.modifier_spelling_phrase,
        )
        return self._strip_config_phrases_case_insensitive(s, modifier_phrases)

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

            self._cancel_type_silence_task()

            self._moonshine_ingress_stop.set()
            if self._moonshine_ingress_thread is not None:
                self._moonshine_ingress_thread.join(timeout=5.0)
                if self._moonshine_ingress_thread.is_alive():
                    logger.warning("Moonshine ingress thread did not stop within timeout")
                self._moonshine_ingress_thread = None

            with self._moonshine_feed_lock:
                self._moonshine_ingress_epoch += 1
                self._drain_moonshine_ingress_queue()
                if self._moonshine_session:
                    logger.info("Stopping active Moonshine dictation stream")
                    self._moonshine_session.stop()
                    self._moonshine_session = None

            if self._llm_processing_task and not self._llm_processing_task.done():
                logger.info("Cancelling active LLM processing task")
                self._llm_processing_task.cancel()
                try:
                    await asyncio.wait_for(asyncio.shield(self._llm_processing_task), timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    logger.info("LLM processing task cancelled")
                except Exception as e:
                    logger.warning(f"Error cancelling LLM task: {e}")

            self._stop_streaming()

            if has_active_session:
                await self._stop_session()

            await self.text_service.shutdown()
            await self.llm_service.shutdown()
            await self.agentic_service.shutdown()
            await self.alias_service.shutdown()

            with self._state_lock:
                self._current_session = None
                self._pending_llm_session = None

            gc.collect()

            logger.info("Dictation coordinator shutdown complete")
        except Exception as e:
            logger.error(f"Shutdown error: {e}", exc_info=True)
