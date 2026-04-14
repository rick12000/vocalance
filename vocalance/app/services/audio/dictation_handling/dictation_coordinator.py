"""Dictation sessions: Moonshine ingress thread, modifiers, LLM handoff, text output.

``MOONSHINE_CHUNK_DICTATION_MODES`` stream PCM off the event bus. ``_STREAMING_STT_MODES`` use
partial/final events. ``_STREAMING_LLM_MODES`` continue to the LLM after stop.
"""

import asyncio
import gc
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.config.command_types import (
    DictationAmendStartCommand,
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
from vocalance.app.events.core_events import DictationTextRecognizedEvent
from vocalance.app.events.dictation_events import (
    DictationModeDisableOthersEvent,
    DictationModifierId,
    DictationModifierPhraseEvent,
    DictationModifierStateChangedEvent,
    DictationSessionEvent,
    DictationStatusChangedEvent,
    FinalDictationTextEvent,
    LLMProcessingCompletedEvent,
    LLMProcessingFailedEvent,
    LLMProcessingReadyEvent,
    LLMProcessingStartedEvent,
    LLMTokenGeneratedEvent,
    PartialDictationTextEvent,
)
from vocalance.app.services.audio.dictation_handling.dictation_alias_service import DictationAliasService
from vocalance.app.services.audio.dictation_handling.dictation_postprocess import (
    apply_dictation_postprocess,
    apply_dictation_postprocess_partial,
    modifier_display_label,
)
from vocalance.app.services.audio.dictation_handling.llm_support.agentic_prompt_service import AgenticPromptService
from vocalance.app.services.audio.dictation_handling.llm_support.llm_service import LLMService
from vocalance.app.services.audio.dictation_handling.text_input_service import TextInputService, remove_formatting
from vocalance.app.services.base_service import Service
from vocalance.app.services.storage.storage_service import StorageService
from vocalance.app.utils.concurrency import SubscriptionTracker

logger = logging.getLogger(__name__)


class DictationMode(Enum):
    """Standard / smart / type / visual / hidden / amend dictation kinds."""

    INACTIVE = "inactive"
    STANDARD = "standard"
    SMART = "smart"
    TYPE = "type"
    VISUAL = "visual"
    HIDDEN = "hidden"
    AMEND = "amend"


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
_STREAMING_STT_MODES = frozenset({DictationMode.SMART, DictationMode.VISUAL, DictationMode.HIDDEN, DictationMode.AMEND})
_STREAMING_LLM_MODES = frozenset({DictationMode.SMART, DictationMode.AMEND})

MOONSHINE_MODIFIER_SUPPRESS_SEC: float = 0.55


class DictationState(Enum):
    """Coordinator FSM: idle, recording, LLM work, or shutdown."""

    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING_LLM = "processing_llm"
    SHUTTING_DOWN = "shutting_down"


_VALID_TRANSITIONS = {
    DictationState.IDLE: {DictationState.RECORDING, DictationState.SHUTTING_DOWN},
    DictationState.RECORDING: {DictationState.PROCESSING_LLM, DictationState.IDLE, DictationState.SHUTTING_DOWN},
    DictationState.PROCESSING_LLM: {DictationState.IDLE, DictationState.SHUTTING_DOWN},
    DictationState.SHUTTING_DOWN: set(),
}


@dataclass
class DictationSession:
    """One dictation run: mode, timing, accumulated text, optional modifier."""

    session_id: str
    mode: DictationMode
    start_time: float
    accumulated_text: str = ""
    last_text_time: Optional[float] = None
    is_first_segment: bool = True
    active_modifiers: set[DictationModifierId] = field(default_factory=set)


@dataclass
class LLMSession:
    """LLM request payload; ``clipboard_text`` set for amend flow."""

    session_id: str
    raw_text: str
    agentic_prompt: str
    clipboard_text: Optional[str] = None


class DictationCoordinator(Service):
    """Owns dictation modes, Moonshine streaming, modifiers, LLM and paste/typing paths (``RLock`` on state)."""

    def __init__(
        self,
        event_bus: EventBus,
        config: GlobalAppConfig,
        storage: StorageService,
        gui_event_loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        self.event_bus = event_bus
        self.config = config
        self.gui_event_loop = gui_event_loop

        self._state_lock = threading.RLock()

        self._current_state = DictationState.IDLE
        self._current_session: Optional[DictationSession] = None
        self._pending_llm_session: Optional[LLMSession] = None
        self._type_silence_task: Optional[asyncio.Task] = None
        self._llm_processing_task: Optional[asyncio.Task] = None

        self._direct_token_callback: Optional[Callable[[str], None]] = None

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

        self._moonshine_feed_lock = threading.Lock()
        self._moonshine_ingress_epoch: int = 0
        self._moonshine_suppress_until: float = 0.0

        self._subs = SubscriptionTracker(event_bus=event_bus)

        self._subs.subscribe(DictationTextRecognizedEvent, self._handle_dictation_text)
        self._subs.subscribe(LLMProcessingCompletedEvent, self._handle_llm_completed)
        self._subs.subscribe(LLMProcessingFailedEvent, self._handle_llm_failed)
        self._subs.subscribe(DictationCommandParsedEvent, self._handle_dictation_command)
        self._subs.subscribe(LLMProcessingReadyEvent, self._handle_llm_processing_ready)
        self._subs.subscribe(DictationModifierPhraseEvent, self._handle_dictation_modifier_phrase)

        logger.debug("DictationCoordinator initialized")

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

    def _moonshine_rotate_line(self, reason: str = "max_line_duration") -> None:
        """Stop the current Moonshine stream and open a new one (bounded line length for latency)."""
        loop = self.gui_event_loop
        if loop is None:
            logger.error("Moonshine stream rotation skipped: gui_event_loop is not set")
            return

        logger.info("Moonshine stream rotation (%s) — starting new native stream", reason)

        with self._moonshine_feed_lock:
            self._moonshine_ingress_epoch += 1
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

        if session is None or session.mode not in MOONSHINE_CHUNK_DICTATION_MODES or state != DictationState.RECORDING:
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

    def feed_moonshine_audio_chunk(self, audio_bytes: bytes, sample_rate: int) -> None:
        """Hot path from the audio recorder thread: feed PCM to Moonshine synchronously."""
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

        try:
            rotate = False
            with self._moonshine_feed_lock:
                with self._state_lock:
                    if epoch != self._moonshine_ingress_epoch:
                        return
                    ms = self._moonshine_session

                if ms is not None:
                    rotate = ms.add_audio_pcm16(audio_bytes, sample_rate)

            if rotate:
                self._moonshine_rotate_line(reason="max_line_duration")
        except Exception as e:
            logger.error("Moonshine ingress feed error: %s", e, exc_info=True)

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

    def set_direct_token_callback(self, callback: Optional[Callable[[str], None]]) -> None:
        """Set a direct callback for token streaming that bypasses the event bus for minimal latency"""
        self._direct_token_callback = callback
        logger.info(f"Direct token callback {'registered' if callback else 'cleared'}")

    async def initialize(self) -> bool:
        """Initialize all dictation services concurrently.

        Returns:
            True if all services initialized successfully, False otherwise.
        """
        try:
            text_init_result = self.text_service.initialize()
            llm_init_result = self.llm_service.initialize()
            results = await asyncio.gather(
                self.agentic_service.initialize(),
                self.alias_service.initialize(),
                return_exceptions=True,
            )
            results.append(text_init_result)
            results.append(llm_init_result)

            if any(isinstance(r, Exception) or not r for r in results):
                logger.error("Service initialization failed")
                return False

            logger.info("Dictation coordinator initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Initialization error: {e}", exc_info=True)
            return False

    # Typed accessors for UI layer — hides internal collaborator decomposition
    @property
    def prompts(self) -> AgenticPromptService:
        return self.agentic_service

    @property
    def aliases(self) -> DictationAliasService:
        return self.alias_service

    async def _handle_dictation_modifier_phrase(self, modifier_phrase: DictationModifierPhraseEvent) -> None:
        """Apply a Vosk-side modifier phrase: toggle off if repeated, else switch, publish UI state, suppress Moonshine."""
        try:
            with self._state_lock:
                session = self._current_session
                if not session or self._current_state != DictationState.RECORDING:
                    return
                mid = modifier_phrase.modifier_id
                current_mods = set(session.active_modifiers)

                casing_mods = {"upper", "capitals", "camel", "snake", "kebab", "diminish"}
                punct_mods = {"spelling", "strip"}

                if mid in current_mods:
                    current_mods.remove(mid)
                else:
                    if mid in casing_mods:
                        current_mods -= casing_mods
                    elif mid in punct_mods:
                        current_mods -= punct_mods
                    current_mods.add(mid)

                label = ", ".join(modifier_display_label(m) for m in current_mods) if current_mods else ""
                active = bool(current_mods)

                self._current_session = DictationSession(
                    session_id=session.session_id,
                    mode=session.mode,
                    start_time=session.start_time,
                    accumulated_text=session.accumulated_text,
                    last_text_time=session.last_text_time,
                    is_first_segment=session.is_first_segment,
                    active_modifiers=current_mods,
                )

            await self._publish_event(
                DictationModifierStateChangedEvent(active=active, active_modifiers=current_mods, display_label=label)
            )
            logger.info("Dictation modifiers: %s -> %s", session.active_modifiers, current_mods)
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

        text_with_placeholders, alias_map = self.alias_service.extract_aliases(cleaned)

        with_subs = text_with_placeholders
        for placeholder, alias_text in alias_map.items():
            pattern = re.compile(re.escape(placeholder), re.IGNORECASE)
            with_subs = pattern.sub(lambda m: alias_text, with_subs)

        if self._is_isolated_stt_noise_fragment(with_subs):
            return ""

        processed = apply_dictation_postprocess(text_with_placeholders, session.active_modifiers)

        for placeholder, alias_text in alias_map.items():
            pattern = re.compile(re.escape(placeholder), re.IGNORECASE)
            processed = pattern.sub(lambda m: alias_text, processed)

        return processed

    def _prepare_dictation_segment_partial(self, raw_text: str, session: DictationSession) -> str:
        """Same cleaning and filtering as finals, but spelling modifier is deferred (streaming UI)."""
        cleaned = self._clean_text(raw_text)
        if not cleaned or self._is_isolated_stt_noise_fragment(cleaned):
            return ""

        text_with_placeholders, alias_map = self.alias_service.extract_aliases(cleaned)

        with_subs = text_with_placeholders
        for placeholder, alias_text in alias_map.items():
            pattern = re.compile(re.escape(placeholder), re.IGNORECASE)
            with_subs = pattern.sub(lambda m: alias_text, with_subs)

        if self._is_isolated_stt_noise_fragment(with_subs):
            return ""

        processed = apply_dictation_postprocess_partial(text_with_placeholders, session.active_modifiers)

        for placeholder, alias_text in alias_map.items():
            pattern = re.compile(re.escape(placeholder), re.IGNORECASE)
            processed = pattern.sub(lambda m: alias_text, processed)

        return processed

    async def _publish_modifier_cleared(self) -> None:
        """Emit inactive modifier state (session end and similar)."""
        await self._publish_event(DictationModifierStateChangedEvent(active=False, active_modifiers=set(), display_label=""))

    @staticmethod
    def _dictation_segment_input_options(mode: DictationMode, modifiers: Optional[set[DictationModifierId]]) -> tuple[bool, bool]:
        """Return ``(add_trailing_space, skip_prose_segment_join_rules)`` for :meth:`TextInputService.input_text`.

        Camel, snake, kebab, and spelling modifiers disable trailing spaces and prose join rules (period removal
        and forced lowercase after a non-sentence boundary) so identifiers and spoken punctuation stay intact.
        """
        skip_join = False
        if modifiers:
            skip_join = bool(modifiers.intersection({"camel", "snake", "kebab", "spelling"}))
        add_trailing = mode != DictationMode.TYPE and not skip_join
        return add_trailing, skip_join

    async def _handle_dictation_text(self, text_recognized: DictationTextRecognizedEvent) -> None:
        """Handle dictated text - centralized processing for all dictation modes"""
        try:
            text = text_recognized.text.strip()
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
                accumulated_text=(f"{session.accumulated_text} {cleaned_text}" if session.accumulated_text else cleaned_text),
                last_text_time=time.time() if session.mode == DictationMode.TYPE else None,
                is_first_segment=False,
                active_modifiers=session.active_modifiers,
            )

            with self._state_lock:
                if self._current_session and self._current_session.session_id == session.session_id:
                    self._current_session = updated_session
                else:
                    return

            add_trailing, skip_join = self._dictation_segment_input_options(updated_session.mode, updated_session.active_modifiers)
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

    async def _handle_llm_completed(self, llm_completion: LLMProcessingCompletedEvent) -> None:
        """Handle LLM completion - clear state and move to IDLE"""
        try:
            logger.info(f"LLM COMPLETION EVENT RECEIVED: '{llm_completion.processed_text[:100]}...'")
            logger.info("Inputting text via text service...")

            processed_text = llm_completion.processed_text

            if not self.config.dictation.enable_dictation_formatting:
                processed_text = remove_formatting(text=processed_text, is_first_word_of_session=True)

            success = await self.text_service.input_text(processed_text)
            logger.info(f"Text input result: {success}")

            await self._cleanup_llm_session()
            logger.info("Smart session ended after LLM completion")
        except Exception as e:
            logger.error(f"LLM completion error: {e}", exc_info=True)
            await self._cleanup_llm_session()

    async def _handle_llm_failed(self, llm_failure: LLMProcessingFailedEvent) -> None:
        """Handle LLM failure - reset state and cleanup"""
        logger.warning(f"LLM processing failed: {llm_failure.error_message}")
        await self._cleanup_llm_session()

    async def _handle_dictation_command(self, parsed_dictation: DictationCommandParsedEvent) -> None:
        """Handle dictation commands"""
        try:
            command = parsed_dictation.command
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
            logger.warning("Amend mode: no text captured — keep focus on the app with the selection")
            return
        with self._state_lock:
            self._amend_clipboard_snapshot = captured.strip()
        await self._start_session(DictationMode.AMEND)

    async def _handle_llm_processing_ready(self, llm_ready: LLMProcessingReadyEvent) -> None:
        """Handle LLM processing ready signal from UI"""
        try:
            with self._state_lock:
                pending = self._pending_llm_session
                if not pending or pending.session_id != llm_ready.session_id:
                    logger.warning(f"Received ready signal for unknown session {llm_ready.session_id}")
                    return

                self._pending_llm_session = None

            logger.info(f"UI ready signal received for session {llm_ready.session_id}")
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
            await self._publish_event(FinalDictationTextEvent(text=processed, segment_id=segment_id or str(uuid.uuid4())))

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

        pattern = r"\b" + re.escape(stop_word) + r"\b"
        result = re.sub(pattern, "", text, flags=re.IGNORECASE)
        return " ".join(result.split())

    async def _stop_streaming_mode(self, session: DictationSession) -> None:
        """Stop Moonshine chunk stream and finalize transcription for supported modes."""
        try:
            with self._moonshine_feed_lock:
                self._moonshine_ingress_epoch += 1
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
                await self._publish_event(DictationSessionEvent(mode=dual, state="stopped", raw_text=final_text))
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
                    await self._publish_event(DictationSessionEvent(mode="visual", state="stopped", accumulated_text=final_text))
                    await self.text_service.input_text(final_text)
                else:
                    await self._publish_event(DictationSessionEvent(mode="visual", state="stopped", accumulated_text=""))
                await self._finalize_session(session)
            elif session.mode == DictationMode.HIDDEN:
                if final_text:
                    await self._publish_event(DictationSessionEvent(mode="hidden", state="stopped", accumulated_text=final_text))
                    await self.text_service.input_text(final_text)
                else:
                    await self._publish_event(DictationSessionEvent(mode="hidden", state="stopped", accumulated_text=""))
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

                initial_modifiers = {"strip", "diminish"} if mode == DictationMode.TYPE else set()

                self._current_session = DictationSession(
                    session_id=session_id,
                    mode=mode,
                    start_time=time.time(),
                    accumulated_text="",
                    last_text_time=None,
                    is_first_segment=True,
                    active_modifiers=initial_modifiers,
                )
                self._set_state(DictationState.RECORDING)

            await self._publish_event(DictationModeDisableOthersEvent(dictation_mode_active=True, dictation_mode=mode.value))

            if initial_modifiers:
                label = ", ".join(modifier_display_label(m) for m in initial_modifiers)
                await self._publish_event(
                    DictationModifierStateChangedEvent(active=True, active_modifiers=initial_modifiers, display_label=label)
                )

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
                await self._publish_event(DictationSessionEvent(mode="smart", state="started"))
            elif mode == DictationMode.AMEND:
                await self._publish_event(DictationSessionEvent(mode="amend", state="started"))
            elif mode == DictationMode.VISUAL:
                await self._publish_event(DictationSessionEvent(mode="visual", state="started"))
            elif mode == DictationMode.HIDDEN:
                await self._publish_event(DictationSessionEvent(mode="hidden", state="started"))

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
        """Start LLM inference after the UI signals ready."""
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
        except Exception as e:
            logger.error(f"LLM processing error: {e}", exc_info=True)

    async def _stream_token(self, token: str) -> None:
        """Async callback to publish token.

        Args:
            token: Token string to publish asynchronously.
        """
        if self._direct_token_callback:
            try:
                self._direct_token_callback(token)
            except Exception as e:
                logger.error(f"Direct callback error: {e}", exc_info=True)

        await self._publish_event(LLMTokenGeneratedEvent(token=token))

    async def _end_smart_session(self) -> None:
        """End smart dictation session"""
        try:
            await self._publish_event(DictationModeDisableOthersEvent(dictation_mode_active=False, dictation_mode="inactive"))

            await self._publish_status(False, DictationMode.INACTIVE)
            logger.info("Smart dictation session ended")
        except Exception as e:
            logger.error(f"Smart session end error: {e}", exc_info=True)

    async def _finalize_session(self, session: DictationSession) -> None:
        """Finalize non-smart session"""
        try:
            await self._publish_modifier_cleared()
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
            cfg.modifier_kebab_phrase,
            cfg.modifier_diminish_phrase,
            cfg.modifier_strip_phrase,
        )
        return self._strip_config_phrases_case_insensitive(s, modifier_phrases)

    async def _publish_event(self, published_event: BaseEvent) -> None:
        """Publish event with error handling"""
        try:
            await self.event_bus.publish(published_event)
        except Exception as e:
            logger.error(f"Event publishing error: {e}", exc_info=True)

    async def _publish_status(self, is_active: bool, mode: DictationMode) -> None:
        await self._publish_event(
            DictationStatusChangedEvent(
                is_active=is_active,
                mode=mode.value,
                show_ui=is_active,
                stop_command=self.config.dictation.stop_trigger if is_active else None,
            )
        )

    async def shutdown(self) -> None:
        """Shutdown coordinator with proper resource cleanup"""
        logger.info("Starting dictation coordinator shutdown")
        try:
            with self._state_lock:
                self._set_state(DictationState.SHUTTING_DOWN)
                has_active_session = self._current_session is not None

            self._cancel_type_silence_task()

            with self._moonshine_feed_lock:
                self._moonshine_ingress_epoch += 1
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

            if has_active_session:
                await self._stop_session()

            self.text_service.shutdown()
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
