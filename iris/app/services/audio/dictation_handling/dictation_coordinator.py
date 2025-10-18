"""
Streamlined Dictation Coordinator

Simplified coordinator that handles dictation modes with clean event-driven architecture.
Follows DRY principles and maintains all essential functionality with minimal code.
"""

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

from iris.app.config.app_config import GlobalAppConfig
from iris.app.config.command_types import (
    DictationSmartStartCommand,
    DictationStartCommand,
    DictationStopCommand,
    DictationTypeCommand,
)
from iris.app.event_bus import EventBus
from iris.app.events.base_event import BaseEvent
from iris.app.events.command_events import DictationCommandParsedEvent
from iris.app.events.core_events import DictationTextRecognizedEvent
from iris.app.events.dictation_events import (
    AudioModeChangeRequestEvent,
    DictationModeDisableOthersEvent,
    DictationStatusChangedEvent,
    LLMProcessingCompletedEvent,
    LLMProcessingFailedEvent,
    LLMProcessingReadyEvent,
    LLMProcessingStartedEvent,
    LLMTokenGeneratedEvent,
    SmartDictationStartedEvent,
    SmartDictationStoppedEvent,
    SmartDictationTextDisplayEvent,
)
from iris.app.services.audio.dictation_handling.llm_support.agentic_prompt_service import AgenticPromptService
from iris.app.services.audio.dictation_handling.llm_support.llm_service import LLMService
from iris.app.services.audio.dictation_handling.text_input_service import TextInputService
from iris.app.services.storage.storage_service import StorageService
from iris.app.utils.event_utils import EventSubscriptionManager, ThreadSafeEventPublisher

logger = logging.getLogger(__name__)


class DictationMode(Enum):
    INACTIVE = "inactive"
    STANDARD = "standard"
    SMART = "smart"
    TYPE = "type"


class DictationState(Enum):
    """Explicit state machine for dictation coordinator"""

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
    """Immutable session snapshot - never modified after creation"""

    session_id: str
    mode: DictationMode
    start_time: float
    accumulated_text: str = ""
    last_text_time: Optional[float] = None


@dataclass
class LLMSession:
    """Separate immutable LLM session for proper state isolation"""

    session_id: str
    raw_text: str
    agentic_prompt: str


class DictationCoordinator:
    """Production-ready dictation coordinator with proper thread-safe state management"""

    def __init__(
        self,
        event_bus: EventBus,
        config: GlobalAppConfig,
        storage: StorageService,
        gui_event_loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
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

        self.event_publisher = ThreadSafeEventPublisher(event_bus=event_bus, event_loop=gui_event_loop)
        self.subscription_manager = EventSubscriptionManager(event_bus=event_bus, component_name="DictationCoordinator")

        logger.info("DictationCoordinator initialized with production-ready threading")

    @property
    def active_mode(self) -> DictationMode:
        with self._state_lock:
            return self._current_session.mode if self._current_session else DictationMode.INACTIVE

    def is_active(self) -> bool:
        return self.active_mode != DictationMode.INACTIVE

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
        """Initialize all services"""
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
        """Set up event subscriptions"""
        subscriptions = [
            (DictationTextRecognizedEvent, self._handle_dictation_text),
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

            # Snapshot session under lock - prevents race conditions
            with self._state_lock:
                session = self._current_session
                if not session:
                    return

                # Only process if we're actually recording (not already processing LLM)
                if self._current_state != DictationState.RECORDING:
                    return

            # Process text outside lock (non-blocking)
            cleaned_text = self._clean_text(text)
            if not cleaned_text:
                return

            # Update session with NEW data
            updated_session = DictationSession(
                session_id=session.session_id,
                mode=session.mode,
                start_time=session.start_time,
                accumulated_text=self._append_text(session.accumulated_text, cleaned_text),
                last_text_time=time.time() if session.mode == DictationMode.TYPE else None,
            )

            # Replace session under lock
            with self._state_lock:
                if self._current_session and self._current_session.session_id == session.session_id:
                    self._current_session = updated_session
                else:
                    # Session changed, ignore this text
                    return

            # Publish events outside lock
            if updated_session.mode == DictationMode.SMART:
                await self._publish_event(SmartDictationTextDisplayEvent(text=cleaned_text))
            else:
                await self.text_service.input_text(cleaned_text)

        except Exception as e:
            logger.error(f"Dictation text error: {e}", exc_info=True)

    async def _cleanup_llm_session(self) -> None:
        """Common cleanup for LLM session completion or failure"""
        with self._state_lock:
            self._current_session = None
            self._pending_llm_session = None
            self._llm_processing_task = None
            self._set_state(DictationState.IDLE)
        await self._end_smart_session()

    async def _handle_llm_completed(self, event: LLMProcessingCompletedEvent) -> None:
        """Handle LLM completion - clear state and move to IDLE"""
        try:
            logger.info(f"LLM COMPLETION EVENT RECEIVED: '{event.processed_text[:100]}...'")
            logger.info("Inputting text via text service...")
            success = await self.text_service.input_text(event.processed_text)
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

        except Exception as e:
            logger.error(f"Command handling error: {e}", exc_info=True)

    async def _handle_llm_processing_ready(self, event: LLMProcessingReadyEvent) -> None:
        """Handle LLM processing ready signal from UI"""
        try:
            # Get pending session and clear it INSIDE lock to prevent duplicate processing
            with self._state_lock:
                pending = self._pending_llm_session
                if not pending or pending.session_id != event.session_id:
                    logger.warning(f"Received ready signal for unknown session {event.session_id}")
                    return

                # CRITICAL: Clear pending session INSIDE lock immediately after validation
                # This prevents duplicate processing if ready signal arrives twice
                self._pending_llm_session = None

            logger.info(f"UI ready signal received for session {event.session_id}")
            # Start LLM processing outside lock with cleared state and track the task
            self._llm_processing_task = asyncio.create_task(self._start_llm_processing(pending))

        except Exception as e:
            logger.error(f"LLM processing ready handling error: {e}", exc_info=True)

    async def _monitor_type_silence(self) -> None:
        """Monitor silence timeout for TYPE dictation mode with safety limits"""
        try:
            timeout = self.config.dictation.type_dictation_silence_timeout
            max_runtime = 300  # 5 minutes safety limit
            start_time = time.time()

            while True:
                # Safety check: prevent infinite loops
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
                # Guard: prevent concurrent session starts
                if self._current_session is not None:
                    logger.warning(
                        f"Cannot start {mode.value} dictation - session {self._current_session.mode.value} already active"
                    )
                    return

                # State validation: only start from idle state
                if self._current_state != DictationState.IDLE:
                    logger.warning(f"Cannot start session - coordinator not in IDLE state (current: {self._current_state.value})")
                    return

                # Create new session - immutable snapshot
                self._current_session = DictationSession(
                    session_id=session_id,
                    mode=mode,
                    start_time=time.time(),
                    accumulated_text="",
                    last_text_time=None,
                )
                self._set_state(DictationState.RECORDING)

            # Publish events outside lock
            await self._publish_event(AudioModeChangeRequestEvent(mode="dictation", reason=f"{mode.value} mode activated"))
            await self._publish_event(DictationModeDisableOthersEvent(dictation_mode_active=True, dictation_mode=mode.value))

            if mode == DictationMode.SMART:
                await self._publish_event(SmartDictationStartedEvent())

            # Start silence monitoring for TYPE mode
            if mode == DictationMode.TYPE:
                self._type_silence_task = asyncio.create_task(self._monitor_type_silence())
                logger.info("Started type dictation silence monitoring task")

            await self._publish_status(True, mode)
            logger.info(f"Started {mode.value} dictation")

        except Exception as e:
            logger.error(f"Session start error: {e}", exc_info=True)
            # Reset state on error
            with self._state_lock:
                self._current_session = None
                self._set_state(DictationState.IDLE)

    async def _stop_session(self) -> None:
        """Stop dictation session with proper cleanup"""
        try:
            # Snapshot session and prepare LLM session under lock
            with self._state_lock:
                session = self._current_session
                if not session:
                    return

                # CRITICAL: Prevent duplicate stop calls during LLM processing
                if self._current_state == DictationState.PROCESSING_LLM:
                    logger.warning("Stop session called while already processing LLM - ignoring duplicate call")
                    return

                # Cancel type silence monitoring task if active
                if session.mode == DictationMode.TYPE:
                    self._cancel_type_silence_task()

                if session.mode == DictationMode.SMART:
                    if session.accumulated_text:
                        # Transition to LLM processing state
                        self._set_state(DictationState.PROCESSING_LLM)

                        # Create immutable LLM session for UI coordination
                        agentic_prompt = self.agentic_service.get_current_prompt() or "Fix grammar and improve clarity."
                        llm_session_id = str(uuid.uuid4())
                        self._pending_llm_session = LLMSession(
                            session_id=llm_session_id,
                            raw_text=session.accumulated_text,
                            agentic_prompt=agentic_prompt,
                        )
                    else:
                        # No text, clean up and end session
                        self._current_session = None
                        self._set_state(DictationState.IDLE)
                else:
                    # Standard/Type: clear session immediately
                    self._current_session = None
                    self._set_state(DictationState.IDLE)

            # Publish events outside lock
            if session and session.mode == DictationMode.SMART and session.accumulated_text:
                # Publish dictation stop events for smart mode WITH text
                await self._publish_event(AudioModeChangeRequestEvent(mode="command", reason="Smart dictation processing"))

                logger.info(f"Publishing SmartDictationStoppedEvent with text: '{session.accumulated_text[:50]}...'")
                await self._publish_event(SmartDictationStoppedEvent(raw_text=session.accumulated_text))
                logger.info("SmartDictationStoppedEvent published successfully")

                # Publish LLM processing started event
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
                # Non-smart session OR smart session with no text - finalize it
                await self._finalize_session(session)

        except Exception as e:
            logger.error(f"Session stop error: {e}", exc_info=True)
            # Ensure state is reset on error
            with self._state_lock:
                self._current_session = None
                self._pending_llm_session = None
                self._set_state(DictationState.IDLE)

    async def _start_llm_processing(self, llm_session: LLMSession) -> None:
        """Actually start the LLM processing after UI is ready"""
        try:
            # Use streaming if available
            if hasattr(self.llm_service, "process_dictation_streaming"):
                logger.info("Starting LLM streaming processing...")

                # Start background streaming thread
                self._start_streaming()

                try:
                    await self.llm_service.process_dictation_streaming(
                        llm_session.raw_text, llm_session.agentic_prompt, token_callback=self._stream_token
                    )
                finally:
                    # Stop streaming and flush remaining tokens
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
                    # Wait for token with timeout to allow checking stop event
                    token = self._token_queue.get(timeout=0.1)

                    # Fast path: direct callback if available (bypasses event bus)
                    if self._direct_token_callback:
                        try:
                            self._direct_token_callback(token)
                        except Exception as e:
                            logger.error(f"Direct callback error: {e}", exc_info=True)

                    # Standard path: event bus for other subscribers
                    self.event_publisher.publish(LLMTokenGeneratedEvent(token=token))

                except queue.Empty:
                    # No token available, continue to check stop event
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
        triggers = {cfg.start_trigger.lower(), cfg.stop_trigger.lower(), cfg.type_trigger.lower(), cfg.smart_start_trigger.lower()}

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

            # Stop streaming thread
            self._stop_streaming()

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
