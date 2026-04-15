from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
import uuid
from typing import Optional

from vocalance.app.config.app_config import DictationConfig, GlobalAppConfig
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
from vocalance.app.events.command_events import DictationCommandParsedEvent
from vocalance.app.events.core_events import DictationTextRecognizedEvent
from vocalance.app.events.dictation_events import (
    DictationModeDisableOthersEvent,
    DictationModifierPhraseEvent,
    DictationModifierStateChangedEvent,
    DictationSessionEvent,
    DictationStatusChangedEvent,
    FinalDictationTextEvent,
    LLMProcessingCompletedEvent,
    LLMProcessingFailedEvent,
    LLMProcessingReadyEvent,
    LLMProcessingStartedEvent,
    PartialDictationTextEvent,
)
from vocalance.app.services.audio.dictation_handling.dictation_alias_service import DictationAliasService
from vocalance.app.services.audio.dictation_handling.llm_support.agentic_prompt_service import AgenticPromptService
from vocalance.app.services.audio.dictation_handling.llm_support.llm_service import LLMService
from vocalance.app.services.audio.dictation_handling.text_input_service import DictationTextInput
from vocalance.app.services.audio.dictation_handling.types import DictationMode, DictationSession, DictationState, LLMSession
from vocalance.app.services.audio.dictation_handling.utils.coordinator_segment_filters import (
    dictation_segment_input_options,
    is_isolated_stt_noise_fragment,
    is_likely_hallucination_fragment,
    remove_stop_trigger_word,
)
from vocalance.app.services.audio.dictation_handling.utils.modifier_postprocess import modifier_display_label
from vocalance.app.services.audio.dictation_handling.utils.postprocess_pipeline import (
    apply_dictation_postprocess,
    apply_dictation_postprocess_partial,
)
from vocalance.app.services.audio.dictation_handling.utils.segment_text import remove_formatting
from vocalance.app.services.audio.dictation_handling.utils.trigger_strip import strip_dictation_triggers
from vocalance.app.services.audio.stt.stt_service import SpeechToTextService
from vocalance.app.services.base_service import Service
from vocalance.app.services.storage.storage_service import StorageService
from vocalance.app.utils.concurrency import SubscriptionTracker

MOONSHINE_CHUNK_DICTATION_MODES: tuple[DictationMode, ...] = (
    DictationMode.STANDARD,
    DictationMode.TYPE,
    DictationMode.SMART,
    DictationMode.VISUAL,
    DictationMode.HIDDEN,
    DictationMode.AMEND,
)

STREAMING_STT_MODES: tuple[DictationMode, ...] = (
    DictationMode.SMART,
    DictationMode.VISUAL,
    DictationMode.HIDDEN,
    DictationMode.AMEND,
)

STREAMING_LLM_MODES: tuple[DictationMode, ...] = (DictationMode.SMART, DictationMode.AMEND)

VALID_DICTATION_STATE_TRANSITIONS: dict[DictationState, frozenset[DictationState]] = {
    DictationState.IDLE: frozenset({DictationState.RECORDING, DictationState.SHUTTING_DOWN}),
    DictationState.RECORDING: frozenset({DictationState.PROCESSING_LLM, DictationState.IDLE, DictationState.SHUTTING_DOWN}),
    DictationState.PROCESSING_LLM: frozenset({DictationState.IDLE, DictationState.SHUTTING_DOWN}),
    DictationState.SHUTTING_DOWN: frozenset(),
}

logger = logging.getLogger(__name__)


def substitute_alias_placeholders(text: str, alias_map: dict[str, str]) -> str:
    out = text
    for placeholder, alias_text in alias_map.items():
        pattern = re.compile(re.escape(placeholder), re.IGNORECASE)
        out = pattern.sub(lambda m, rep=alias_text: rep, out)
    return out


class DictationSegmentPipeline:
    def __init__(self, dictation_config: DictationConfig, alias_service: DictationAliasService) -> None:
        self.dictation_config = dictation_config
        self.alias_service = alias_service

    def clean_text(self, text: str) -> str:
        return strip_dictation_triggers(text, self.dictation_config)

    def prepare_final(self, raw_text: str, session: DictationSession) -> str:
        cleaned = self.clean_text(raw_text)
        if not cleaned or is_isolated_stt_noise_fragment(cleaned):
            return ""

        text_with_placeholders, alias_map = self.alias_service.extract_aliases(cleaned)

        with_subs = substitute_alias_placeholders(text_with_placeholders, alias_map)

        if is_isolated_stt_noise_fragment(with_subs):
            return ""

        processed = apply_dictation_postprocess(text_with_placeholders, session.active_modifiers)

        return substitute_alias_placeholders(processed, alias_map)

    def prepare_partial(self, raw_text: str, session: DictationSession) -> str:
        cleaned = self.clean_text(raw_text)
        if not cleaned or is_isolated_stt_noise_fragment(cleaned):
            return ""

        text_with_placeholders, alias_map = self.alias_service.extract_aliases(cleaned)

        with_subs = substitute_alias_placeholders(text_with_placeholders, alias_map)

        if is_isolated_stt_noise_fragment(with_subs):
            return ""

        processed = apply_dictation_postprocess_partial(text_with_placeholders, session.active_modifiers)

        return substitute_alias_placeholders(processed, alias_map)


class DictationMoonshineController:
    def __init__(self, coordinator: DictationCoordinator) -> None:
        self.coordinator = coordinator
        self.moonshine_session = None
        self.moonshine_feed_lock = threading.Lock()
        self.moonshine_ingress_epoch: int = 0
        self.moonshine_suppress_until: float = 0.0
        self.streaming_finalized_text: str = ""
        self.streaming_finalized_segments: list[str] = []

    def note_modifier_suppress(self, duration_sec: float) -> None:
        self.moonshine_suppress_until = time.monotonic() + duration_sec

    def output_suppressed(self) -> bool:
        return time.monotonic() < self.moonshine_suppress_until

    def reset_streaming_buffers(self) -> None:
        self.streaming_finalized_text = ""
        self.streaming_finalized_segments = []

    def halt_streaming_capturer(self) -> str:
        with self.moonshine_feed_lock:
            self.moonshine_ingress_epoch += 1
            if self.moonshine_session:
                self.moonshine_session.stop()
                self.moonshine_session = None
        return self.streaming_finalized_text

    def clear_streaming_accumulators(self) -> None:
        self.streaming_finalized_text = ""
        self.streaming_finalized_segments = []

    def shutdown_ingress(self) -> None:
        with self.moonshine_feed_lock:
            self.moonshine_ingress_epoch += 1
            if self.moonshine_session:
                self.moonshine_session.stop()
                self.moonshine_session = None

    def try_open_dictation_stream(self) -> None:
        engine = self.coordinator.stt_service.moonshine_engine
        if not engine:
            logger.error("Moonshine engine unavailable — cannot start chunk dictation stream")
            return
        self.moonshine_session = engine.open_dictation_stream(
            self.coordinator.gui_event_loop,
            self.on_partial,
            self.on_final,
        )
        self.moonshine_ingress_epoch += 1

    def rotate_line(self) -> None:
        loop = self.coordinator.gui_event_loop

        with self.moonshine_feed_lock:
            self.moonshine_ingress_epoch += 1
            old = self.moonshine_session
            self.moonshine_session = None

        if old is not None:
            old.stop()

        with self.coordinator.state_lock:
            session = self.coordinator.current_session
            state = self.coordinator.current_state

        if session is None or session.mode not in MOONSHINE_CHUNK_DICTATION_MODES or state != DictationState.RECORDING:
            return

        engine = self.coordinator.stt_service.moonshine_engine
        if not engine:
            return

        new_sess = engine.open_dictation_stream(
            loop,
            self.on_partial,
            self.on_final,
        )

        with self.moonshine_feed_lock:
            self.moonshine_session = new_sess
            self.moonshine_ingress_epoch += 1

    def feed_moonshine_audio_chunk(self, audio_bytes: bytes, sample_rate: int) -> None:
        if not audio_bytes:
            return
        with self.coordinator.state_lock:
            if self.coordinator.current_state == DictationState.SHUTTING_DOWN:
                return
            session = self.coordinator.current_session
            if session is None or session.mode not in MOONSHINE_CHUNK_DICTATION_MODES:
                return
            if self.moonshine_session is None:
                return
            epoch = self.moonshine_ingress_epoch

        rotate = False
        with self.moonshine_feed_lock:
            with self.coordinator.state_lock:
                if epoch != self.moonshine_ingress_epoch:
                    return
                ms = self.moonshine_session

            if ms is not None:
                rotate = ms.add_audio_pcm16(audio_bytes, sample_rate)

        if rotate:
            self.rotate_line()

    async def on_partial(self, text: str, segment_id: str) -> None:
        if self.output_suppressed():
            return
        with self.coordinator.state_lock:
            session = self.coordinator.current_session

        if not session or session.mode not in MOONSHINE_CHUNK_DICTATION_MODES:
            return

        if session.mode in (DictationMode.HIDDEN, DictationMode.STANDARD, DictationMode.TYPE):
            return

        if self.coordinator.current_state != DictationState.RECORDING:
            return

        if is_likely_hallucination_fragment(text, ""):
            return

        with self.coordinator.state_lock:
            live = self.coordinator.current_session
            if not live or live.session_id != session.session_id:
                return
            session = live
        partial_text = self.coordinator.segment_pipeline.prepare_partial(text, session)
        if not partial_text:
            return
        await self.coordinator.event_bus.publish(PartialDictationTextEvent(text=partial_text, segment_id=segment_id))

    async def on_final(self, text: str, segment_id: str) -> None:
        if self.output_suppressed():
            return
        with self.coordinator.state_lock:
            session = self.coordinator.current_session

        if not session or session.mode not in MOONSHINE_CHUNK_DICTATION_MODES:
            return

        if self.coordinator.current_state != DictationState.RECORDING:
            return

        if is_likely_hallucination_fragment(text, ""):
            return

        line = text.strip()
        if not line:
            return

        if session.mode in (DictationMode.STANDARD, DictationMode.TYPE):
            await self.coordinator.event_bus.publish(
                DictationTextRecognizedEvent(
                    text=line,
                    processing_time_ms=0.0,
                    engine="moonshine",
                    mode="dictation",
                )
            )
            return

        with self.coordinator.state_lock:
            live = self.coordinator.current_session
            if not live or live.session_id != session.session_id:
                return
            session = live
        await self.emit_final_text_append(line, segment_id, session)

    async def emit_final_text_append(self, text: str, segment_id: str, session: DictationSession) -> None:
        if not text or not text.strip():
            return

        raw_line = text.strip()

        with self.coordinator.state_lock:
            live = self.coordinator.current_session
            if not live or live.session_id != session.session_id:
                return
            session = live

        processed = self.coordinator.segment_pipeline.prepare_final(raw_line, session)
        if not processed:
            return

        if self.streaming_finalized_segments:
            if self.streaming_finalized_segments[-1].strip().lower() == processed.lower():
                return

        if session.mode != DictationMode.HIDDEN:
            await self.coordinator.event_bus.publish(
                FinalDictationTextEvent(text=processed, segment_id=segment_id or str(uuid.uuid4()))
            )

        self.streaming_finalized_segments.append(processed)

        if self.streaming_finalized_text:
            self.streaming_finalized_text += " " + processed
        else:
            self.streaming_finalized_text = processed


class DictationLlmRuntime:
    def __init__(self, coordinator: DictationCoordinator) -> None:
        self.coordinator = coordinator

    async def handle_completed(self, llm_completion: LLMProcessingCompletedEvent) -> None:
        processed_text = llm_completion.processed_text

        if not self.coordinator.config.dictation.enable_dictation_formatting:
            processed_text = remove_formatting(text=processed_text, is_first_word_of_session=True)

        await self.coordinator.text_service.input_text(processed_text)

        await self.cleanup_session()

    async def handle_failed(self, llm_failure: LLMProcessingFailedEvent) -> None:
        logger.warning(f"LLM processing failed: {llm_failure.error_message}")
        await self.cleanup_session()

    async def handle_ready(self, llm_ready: LLMProcessingReadyEvent) -> None:
        with self.coordinator.state_lock:
            pending = self.coordinator.pending_llm_session
            if not pending or pending.session_id != llm_ready.session_id:
                logger.warning("Received ready signal for unknown session %s", llm_ready.session_id)
                return

            self.coordinator.pending_llm_session = None

        self.coordinator.llm_processing_task = asyncio.create_task(self.start_processing(pending))

    async def start_processing(self, llm_session: LLMSession) -> None:
        sid = llm_session.session_id
        if llm_session.clipboard_text is not None:
            await self.coordinator.llm_service.process_amend_streaming(
                llm_session.clipboard_text,
                llm_session.raw_text,
                llm_session.agentic_prompt,
                stream_session_id=sid,
            )
        else:
            await self.coordinator.llm_service.process_dictation_streaming(
                llm_session.raw_text, llm_session.agentic_prompt, stream_session_id=sid
            )

    async def cleanup_session(self) -> None:
        with self.coordinator.state_lock:
            self.coordinator.current_session = None
            self.coordinator.pending_llm_session = None
            self.coordinator.llm_processing_task = None
            self.coordinator.amend_clipboard_snapshot = None
            self.coordinator.set_state(DictationState.IDLE)
        await self.coordinator.exit_dictation_ui(reset_modifiers=True)


class DictationCoordinator(Service):
    def __init__(
        self,
        event_bus: EventBus,
        config: GlobalAppConfig,
        storage: StorageService,
        gui_event_loop: asyncio.AbstractEventLoop,
        stt_service: SpeechToTextService,
    ) -> None:
        self.event_bus = event_bus
        self.config = config
        self.gui_event_loop = gui_event_loop
        self.stt_service = stt_service

        self.state_lock = threading.RLock()

        self.current_state = DictationState.IDLE
        self.current_session: Optional[DictationSession] = None
        self.pending_llm_session: Optional[LLMSession] = None
        self.type_silence_task: Optional[asyncio.Task] = None
        self.llm_processing_task: Optional[asyncio.Task] = None

        self.text_service = DictationTextInput(config=config.dictation, loop=gui_event_loop)
        self.llm_service = LLMService(event_bus=event_bus, config=config, gui_event_loop=gui_event_loop)
        self.agentic_service = AgenticPromptService(event_bus=event_bus, config=config, storage=storage)
        self.alias_service = DictationAliasService(event_bus=event_bus, storage=storage, event_loop=gui_event_loop)

        self.segment_pipeline = DictationSegmentPipeline(config.dictation, self.alias_service)
        self.moonshine = DictationMoonshineController(self)
        self.llm_runtime = DictationLlmRuntime(self)

        self.amend_clipboard_snapshot: Optional[str] = None

        self.subs = SubscriptionTracker(event_bus=event_bus)

        self.subs.subscribe(DictationTextRecognizedEvent, self.handle_dictation_text)
        self.subs.subscribe(LLMProcessingCompletedEvent, self.llm_runtime.handle_completed)
        self.subs.subscribe(LLMProcessingFailedEvent, self.llm_runtime.handle_failed)
        self.subs.subscribe(DictationCommandParsedEvent, self.handle_dictation_command)
        self.subs.subscribe(LLMProcessingReadyEvent, self.llm_runtime.handle_ready)
        self.subs.subscribe(DictationModifierPhraseEvent, self.handle_dictation_modifier_phrase)

    @property
    def active_mode(self) -> DictationMode:
        with self.state_lock:
            return self.current_session.mode if self.current_session else DictationMode.INACTIVE

    def is_active(self) -> bool:
        return self.active_mode != DictationMode.INACTIVE

    def feed_moonshine_audio_chunk(self, audio_bytes: bytes, sample_rate: int) -> None:
        self.moonshine.feed_moonshine_audio_chunk(audio_bytes, sample_rate)

    def set_state(self, new_state: DictationState) -> None:
        """Set ``current_state`` after validating transition (caller must hold ``state_lock``)."""
        old_state = self.current_state

        if new_state not in VALID_DICTATION_STATE_TRANSITIONS[old_state]:
            error_msg = f"Invalid state transition: {old_state} -> {new_state}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        self.current_state = new_state

    async def initialize(self) -> bool:
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

        return True

    @property
    def prompts(self) -> AgenticPromptService:
        return self.agentic_service

    @property
    def aliases(self) -> DictationAliasService:
        return self.alias_service

    async def handle_dictation_modifier_phrase(self, modifier_phrase: DictationModifierPhraseEvent) -> None:
        with self.state_lock:
            session = self.current_session
            if not session or self.current_state != DictationState.RECORDING:
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

            self.current_session = DictationSession(
                session_id=session.session_id,
                mode=session.mode,
                start_time=session.start_time,
                accumulated_text=session.accumulated_text,
                last_text_time=session.last_text_time,
                is_first_segment=session.is_first_segment,
                active_modifiers=current_mods,
            )
        await self.event_bus.publish(
            DictationModifierStateChangedEvent(active=active, active_modifiers=current_mods, display_label=label)
        )
        if session.mode in MOONSHINE_CHUNK_DICTATION_MODES:
            self.moonshine.note_modifier_suppress(self.config.dictation.moonshine_modifier_suppress_sec)

    def clean_text(self, text: str) -> str:
        return self.segment_pipeline.clean_text(text)

    async def publish_modifier_cleared(self) -> None:
        await self.event_bus.publish(DictationModifierStateChangedEvent(active=False, active_modifiers=set(), display_label=""))

    async def handle_dictation_text(self, text_recognized: DictationTextRecognizedEvent) -> None:
        text = text_recognized.text.strip()
        if not text:
            return

        with self.state_lock:
            session = self.current_session
            if not session:
                return

            if self.current_state != DictationState.RECORDING:
                return

            if session.mode in STREAMING_STT_MODES:
                return

        cleaned_text = self.segment_pipeline.prepare_final(text, session)
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

        with self.state_lock:
            if self.current_session and self.current_session.session_id == session.session_id:
                self.current_session = updated_session
            else:
                return

        add_trailing, skip_join = dictation_segment_input_options(updated_session.mode, updated_session.active_modifiers)
        await self.text_service.input_text(
            text=cleaned_text,
            add_trailing_space=add_trailing,
            skip_prose_segment_join_rules=skip_join,
        )

    async def handle_dictation_command(self, parsed_dictation: DictationCommandParsedEvent) -> None:
        command = parsed_dictation.command
        if isinstance(command, DictationStartCommand):
            await self.start_session(DictationMode.STANDARD)
        elif isinstance(command, DictationStopCommand):
            await self.stop_session()
        elif isinstance(command, DictationTypeCommand):
            await self.start_session(DictationMode.TYPE)
        elif isinstance(command, DictationSmartStartCommand):
            await self.start_session(DictationMode.SMART)
        elif isinstance(command, DictationVisualStartCommand):
            await self.start_session(DictationMode.VISUAL)
        elif isinstance(command, DictationHiddenStartCommand):
            await self.start_session(DictationMode.HIDDEN)
        elif isinstance(command, DictationAmendStartCommand):
            await self.start_amend_session()

    async def start_amend_session(self) -> None:
        captured = await self.gui_event_loop.run_in_executor(None, self.text_service.capture_selection_via_copy)
        if not captured or not captured.strip():
            logger.warning("Amend mode: no text captured — keep focus on the app with the selection")
            return
        with self.state_lock:
            self.amend_clipboard_snapshot = captured.strip()
        await self.start_session(DictationMode.AMEND)

    async def stop_streaming_mode(self, session: DictationSession) -> None:
        final_text = self.moonshine.halt_streaming_capturer()

        if session.mode in (DictationMode.STANDARD, DictationMode.TYPE):
            with self.state_lock:
                self.current_session = None
                self.set_state(DictationState.IDLE)
            self.moonshine.clear_streaming_accumulators()
            await self.exit_dictation_ui(reset_modifiers=True)
            return

        if session.mode in (DictationMode.HIDDEN, DictationMode.AMEND) and final_text:
            final_text = remove_stop_trigger_word(final_text, self.config.dictation.stop_trigger)
        if final_text:
            final_text = self.alias_service.apply_substitutions(final_text)
            final_text = " ".join(final_text.split())

        self.moonshine.clear_streaming_accumulators()

        amend_clipboard_error = False
        with self.state_lock:
            if session.mode in STREAMING_LLM_MODES and final_text:
                if session.mode == DictationMode.AMEND and not self.amend_clipboard_snapshot:
                    logger.error("Amend mode: clipboard snapshot missing")
                    self.current_session = None
                    self.set_state(DictationState.IDLE)
                    amend_clipboard_error = True
                else:
                    self.set_state(DictationState.PROCESSING_LLM)
                    default_prompt = (
                        "Fix grammar and improve clarity."
                        if session.mode == DictationMode.SMART
                        else "Follow the spoken instructions when transforming the text."
                    )
                    agentic_prompt = self.agentic_service.get_current_prompt() or default_prompt
                    llm_session_id = str(uuid.uuid4())
                    self.pending_llm_session = LLMSession(
                        session_id=llm_session_id,
                        raw_text=final_text,
                        agentic_prompt=agentic_prompt,
                        clipboard_text=self.amend_clipboard_snapshot if session.mode == DictationMode.AMEND else None,
                    )
            else:
                self.current_session = None
                self.set_state(DictationState.IDLE)

        llm_started = session.mode in STREAMING_LLM_MODES and final_text and self.pending_llm_session
        if amend_clipboard_error or llm_started:
            await self.publish_modifier_cleared()

        if llm_started:
            dual = "amend" if session.mode == DictationMode.AMEND else "smart"
            await self.event_bus.publish(DictationSessionEvent(mode=dual, state="stopped", raw_text=final_text))
            await self.event_bus.publish(
                LLMProcessingStartedEvent(
                    raw_text=final_text,
                    agentic_prompt=self.pending_llm_session.agentic_prompt,
                    session_id=self.pending_llm_session.session_id,
                )
            )
        elif session.mode in STREAMING_LLM_MODES:
            await self.exit_dictation_ui(reset_modifiers=False)
        elif session.mode == DictationMode.VISUAL:
            if final_text:
                await self.event_bus.publish(DictationSessionEvent(mode="visual", state="stopped", accumulated_text=final_text))
                await self.text_service.input_text(final_text)
            else:
                await self.event_bus.publish(DictationSessionEvent(mode="visual", state="stopped", accumulated_text=""))
            await self.exit_dictation_ui(reset_modifiers=True)
        elif session.mode == DictationMode.HIDDEN:
            if final_text:
                await self.event_bus.publish(DictationSessionEvent(mode="hidden", state="stopped", accumulated_text=final_text))
                await self.text_service.input_text(final_text)
            else:
                await self.event_bus.publish(DictationSessionEvent(mode="hidden", state="stopped", accumulated_text=""))
            await self.exit_dictation_ui(reset_modifiers=True)

    async def monitor_type_silence(self) -> None:
        try:
            timeout = self.config.dictation.type_dictation_silence_timeout
            max_runtime = self.config.dictation.type_silence_monitor_max_seconds
            start_time = time.time()

            while True:
                if time.time() - start_time > max_runtime:
                    logger.warning("Type silence monitoring exceeded max runtime (%ss), auto-stopping", max_runtime)
                    break

                await asyncio.sleep(0.1)

                with self.state_lock:
                    session = self.current_session
                    if not session or session.mode != DictationMode.TYPE:
                        return

                    if session.last_text_time is None:
                        continue

                    time_since_last_text = time.time() - session.last_text_time

                    if time_since_last_text >= timeout:
                        break

            await self.stop_session()

        except asyncio.CancelledError:
            return

    def cancel_type_silence_task(self) -> None:
        if self.type_silence_task and not self.type_silence_task.done():
            self.type_silence_task.cancel()
            self.type_silence_task = None

    async def start_session(self, mode: DictationMode) -> None:
        session_id = str(uuid.uuid4())

        with self.state_lock:
            if self.current_session is not None:
                logger.warning("Cannot start %s dictation - session %s already active", mode, self.current_session.mode)
                return

            if self.current_state != DictationState.IDLE:
                logger.warning("Cannot start session - coordinator not in IDLE state (current: %s)", self.current_state)
                return

            if mode != DictationMode.AMEND:
                self.amend_clipboard_snapshot = None

            self.text_service.reset_session()

            initial_modifiers = {"strip", "diminish"} if mode == DictationMode.TYPE else set()

            self.current_session = DictationSession(
                session_id=session_id,
                mode=mode,
                start_time=time.time(),
                accumulated_text="",
                last_text_time=None,
                is_first_segment=True,
                active_modifiers=initial_modifiers,
            )
            self.set_state(DictationState.RECORDING)

        await self.event_bus.publish(DictationModeDisableOthersEvent(dictation_mode_active=True, dictation_mode=str(mode)))

        if initial_modifiers:
            label = ", ".join(modifier_display_label(m) for m in initial_modifiers)
            await self.event_bus.publish(
                DictationModifierStateChangedEvent(active=True, active_modifiers=initial_modifiers, display_label=label)
            )

        if mode in MOONSHINE_CHUNK_DICTATION_MODES:
            self.moonshine.reset_streaming_buffers()
            self.moonshine.try_open_dictation_stream()

        if mode == DictationMode.SMART:
            await self.event_bus.publish(DictationSessionEvent(mode="smart", state="started"))
        elif mode == DictationMode.AMEND:
            await self.event_bus.publish(DictationSessionEvent(mode="amend", state="started"))
        elif mode == DictationMode.VISUAL:
            await self.event_bus.publish(DictationSessionEvent(mode="visual", state="started"))
        elif mode == DictationMode.HIDDEN:
            await self.event_bus.publish(DictationSessionEvent(mode="hidden", state="started"))

        if mode == DictationMode.TYPE:
            self.type_silence_task = asyncio.create_task(self.monitor_type_silence())

        await self.publish_status(True, mode)

    async def stop_session(self) -> None:
        streaming_session: Optional[DictationSession] = None
        session_to_finalize: Optional[DictationSession] = None

        with self.state_lock:
            session = self.current_session
            if not session:
                return

            if self.current_state == DictationState.PROCESSING_LLM:
                logger.warning("Stop session called while already processing LLM - ignoring duplicate call")
                return

            if session.mode == DictationMode.TYPE:
                self.cancel_type_silence_task()

            if session.mode in MOONSHINE_CHUNK_DICTATION_MODES:
                streaming_session = session
            else:
                self.current_session = None
                self.set_state(DictationState.IDLE)
                session_to_finalize = session

        if streaming_session is not None:
            await self.stop_streaming_mode(streaming_session)
            return

        if session_to_finalize is not None:
            await self.exit_dictation_ui(reset_modifiers=True)

    async def exit_dictation_ui(self, reset_modifiers: bool) -> None:
        if reset_modifiers:
            await self.publish_modifier_cleared()
        await self.event_bus.publish(DictationModeDisableOthersEvent(dictation_mode_active=False, dictation_mode="inactive"))
        await self.publish_status(False, DictationMode.INACTIVE)

    async def publish_status(self, is_active: bool, mode: DictationMode) -> None:
        await self.event_bus.publish(
            DictationStatusChangedEvent(
                is_active=is_active,
                mode=str(mode),
                show_ui=is_active,
                stop_command=self.config.dictation.stop_trigger if is_active else None,
            )
        )

    async def shutdown(self) -> None:
        with self.state_lock:
            self.set_state(DictationState.SHUTTING_DOWN)
            has_active_session = self.current_session is not None

        self.cancel_type_silence_task()

        self.moonshine.shutdown_ingress()

        if self.llm_processing_task and not self.llm_processing_task.done():
            self.llm_processing_task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(self.llm_processing_task), timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        if has_active_session:
            await self.stop_session()

        self.text_service.shutdown()
        await self.llm_service.shutdown()
        await self.agentic_service.shutdown()
        await self.alias_service.shutdown()

        with self.state_lock:
            self.current_session = None
            self.pending_llm_session = None
