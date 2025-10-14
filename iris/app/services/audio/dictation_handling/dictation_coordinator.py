"""
Streamlined Dictation Coordinator

Simplified coordinator that handles dictation modes with clean event-driven architecture.
Follows DRY principles and maintains all essential functionality with minimal code.
"""

import asyncio
import logging
import time
import threading
import uuid
from typing import Optional
from enum import Enum
from dataclasses import dataclass
from collections import deque

from iris.app.event_bus import EventBus
from iris.app.config.app_config import GlobalAppConfig
from iris.app.services.storage.unified_storage_service import UnifiedStorageService
from iris.app.services.audio.dictation_handling.text_input_service import TextInputService
from iris.app.services.audio.dictation_handling.llm_support.llm_service import LLMService
from iris.app.services.audio.dictation_handling.llm_support.agentic_prompt_service import AgenticPromptService
from iris.app.config.command_types import (
    DictationStartCommand, DictationStopCommand,
    DictationTypeCommand, DictationSmartStartCommand
)
from iris.app.events.dictation_events import (
    DictationStatusChangedEvent,
    SmartDictationStartedEvent, SmartDictationStoppedEvent,
    AudioModeChangeRequestEvent,
    LLMProcessingStartedEvent, LLMProcessingCompletedEvent, LLMProcessingFailedEvent, LLMTokenGeneratedEvent,
    DictationModeDisableOthersEvent, SmartDictationTextDisplayEvent, LLMProcessingReadyEvent
)
from iris.app.events.core_events import CommandTextRecognizedEvent, DictationTextRecognizedEvent
from iris.app.events.core_events import CustomSoundRecognizedEvent
from iris.app.events.command_events import DictationCommandParsedEvent
from iris.app.utils.event_utils import ThreadSafeEventPublisher, EventSubscriptionManager
from iris.app.events.base_event import BaseEvent

logger = logging.getLogger(__name__)

class DictationMode(Enum):
    INACTIVE = "inactive"
    STANDARD = "standard" 
    SMART = "smart"
    TYPE = "type"

@dataclass
class DictationSession:
    mode: DictationMode
    start_time: float
    accumulated_text: str = ""
    is_processing: bool = False
    last_text_time: Optional[float] = None

class DictationCoordinator:
    """Streamlined dictation coordinator with minimal complexity"""
    
    def __init__(self, event_bus: EventBus, config: GlobalAppConfig, storage: UnifiedStorageService, gui_event_loop: Optional[asyncio.AbstractEventLoop] = None):
        self.event_bus = event_bus
        self.config = config
        self.gui_event_loop = gui_event_loop
        
        self._lock = threading.RLock()
        self._current_session: Optional[DictationSession] = None
        self._pending_llm_session: Optional[DictationSession] = None
        self._session_id: Optional[str] = None
        self._type_silence_task: Optional[asyncio.Task] = None
        
        # Lock-free token streaming - deque is thread-safe for append/popleft
        self._token_queue = deque()  # Lock-free queue for tokens
        self._streaming_active = False
        self._streaming_thread: Optional[threading.Thread] = None
        self._direct_token_callback: Optional[callable] = None
        
        # Initialize services
        self.text_service = TextInputService(config=config.dictation)
        self.llm_service = LLMService(event_bus=event_bus, config=config)
        self.agentic_service = AgenticPromptService(event_bus=event_bus, config=config, storage=storage)
        
        self.event_publisher = ThreadSafeEventPublisher(event_bus=event_bus, event_loop=gui_event_loop)
        self.subscription_manager = EventSubscriptionManager(event_bus=event_bus, component_name="DictationCoordinator")
        
        logger.info("DictationCoordinator initialized")
    
    @property
    def active_mode(self) -> DictationMode:
        with self._lock:
            return self._current_session.mode if self._current_session else DictationMode.INACTIVE
    
    def is_active(self) -> bool:
        return self.active_mode != DictationMode.INACTIVE
    
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
                return_exceptions=True
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
            (CommandTextRecognizedEvent, self._handle_trigger_text),
            (DictationTextRecognizedEvent, self._handle_dictation_text),
            (CustomSoundRecognizedEvent, self._handle_sound_trigger),
            (LLMProcessingCompletedEvent, self._handle_llm_completed),
            (LLMProcessingFailedEvent, self._handle_llm_failed),
            (DictationCommandParsedEvent, self._handle_dictation_command),
            (LLMProcessingReadyEvent, self._handle_llm_processing_ready),
        ]
        
        for event_type, handler in subscriptions:
            self.subscription_manager.subscribe(event_type, handler)
        
        logger.info("Event subscriptions configured")
    
    async def _handle_trigger_text(self, event: CommandTextRecognizedEvent) -> None:
        """Handle trigger word detection"""
        try:
            await self._process_trigger(event.text.strip().lower())
        except Exception as e:
            logger.error(f"Trigger handling error: {e}", exc_info=True)
    
    async def _handle_dictation_text(self, event: DictationTextRecognizedEvent) -> None:
        """Handle dictated text - centralized processing for all dictation modes"""
        try:
            text = event.text.strip()
            if not text:
                return
                
            with self._lock:
                session = self._current_session
                
            if not session:
                return
            
            # Don't process new text if smart session is already being processed
            if session.mode == DictationMode.SMART and getattr(session, 'is_processing', False):
                logger.debug("Ignoring dictation text - smart session already processing")
                return
                
            # Clean and process text for all modes
            cleaned_text = self._clean_text(text)
            if not cleaned_text:
                return
                
            # Accumulate text for session tracking
            session.accumulated_text = self._append_text(session.accumulated_text, cleaned_text)
            
            # Update last text time for TYPE mode silence detection
            if session.mode == DictationMode.TYPE:
                session.last_text_time = time.time()
                logger.debug(f"Type dictation: updated last_text_time, reset silence timer")
            
            # Handle based on dictation mode
            if session.mode == DictationMode.SMART:
                # Smart dictation: send cleaned text to UI for display, accumulate for LLM processing
                logger.debug(f"Smart dictation: accumulated text now: '{session.accumulated_text}'")
                await self._publish_event(SmartDictationTextDisplayEvent(text=cleaned_text))
            else:
                # Standard/Type dictation: input text immediately
                await self.text_service.input_text(cleaned_text)
                
        except Exception as e:
            logger.error(f"Dictation text error: {e}", exc_info=True)
    
    async def _handle_sound_trigger(self, event: CustomSoundRecognizedEvent) -> None:
        """Handle sound-based triggers"""
        try:
            trigger_text = event.mapped_command.lower() if event.mapped_command else event.label.lower()
            await self._process_trigger(trigger_text)
        except Exception as e:
            logger.error(f"Sound trigger error: {e}", exc_info=True)
    
    async def _handle_llm_completed(self, event: LLMProcessingCompletedEvent) -> None:
        """Handle LLM completion"""
        try:
            logger.info(f"LLM COMPLETION EVENT RECEIVED: '{event.processed_text[:100]}...'")
            logger.info(f"Inputting text via text service...")
            success = await self.text_service.input_text(event.processed_text)
            logger.info(f"Text input result: {success}")
            
            # Clear the session now that processing is complete
            with self._lock:
                self._current_session = None
                
            await self._end_smart_session()
            logger.info("Smart session ended after LLM completion")
        except Exception as e:
            logger.error(f"LLM completion error: {e}", exc_info=True)
    
    async def _handle_llm_failed(self, event: LLMProcessingFailedEvent) -> None:
        """Handle LLM failure"""
        logger.warning(f"LLM processing failed: {event.error_message}")
        await self._end_smart_session()
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
        """Handle LLM processing ready signal from UI - fire and forget to avoid blocking event bus"""
        try:
            if self._session_id and event.session_id == self._session_id and self._pending_llm_session:
                logger.info(f"UI ready signal received for session {event.session_id} - starting LLM processing now")
                # Use create_task to avoid blocking the event bus worker
                # This allows token events to be processed in real-time
                asyncio.create_task(self._start_llm_processing(self._pending_llm_session))
                self._pending_llm_session = None
                self._session_id = None
            else:
                logger.warning(f"Received ready signal for unknown session {event.session_id}")
        except Exception as e:
            logger.error(f"LLM processing ready handling error: {e}", exc_info=True)
    
    async def _monitor_type_silence(self) -> None:
        """Monitor silence timeout for TYPE dictation mode"""
        try:
            timeout = self.config.dictation.type_dictation_silence_timeout
            
            while True:
                await asyncio.sleep(0.1)
                
                with self._lock:
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
        """Cancel the type silence monitoring task"""
        if self._type_silence_task and not self._type_silence_task.done():
            self._type_silence_task.cancel()
            self._type_silence_task = None
            logger.debug("Type silence task cancelled")
    
    async def _process_trigger(self, text: str) -> None:
        """Process trigger words"""
        cfg = self.config.dictation
        
        if self.is_active():
            if text == cfg.stop_trigger.lower():
                await self._stop_session()
        else:
            if text == cfg.smart_start_trigger.lower():
                await self._start_session(DictationMode.SMART)
            elif text == cfg.start_trigger.lower():
                await self._start_session(DictationMode.STANDARD)
            elif text == cfg.type_trigger.lower():
                await self._start_session(DictationMode.TYPE)
    
    async def _start_session(self, mode: DictationMode) -> None:
        """Start dictation session"""
        try:
            with self._lock:
                self._current_session = DictationSession(mode=mode, start_time=time.time())
            
            # Request audio mode change
            await self._publish_event(
                AudioModeChangeRequestEvent(mode="dictation", reason=f"{mode.value} mode activated"))
            
            # Notify STT service about dictation mode activation
            await self._publish_event(
                DictationModeDisableOthersEvent(
                    dictation_mode_active=True,
                    dictation_mode=mode.value
                ))
            
            # Publish mode-specific events (only for smart dictation which needs UI coordination)
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
    
    async def _stop_session(self) -> None:
        """Stop dictation session"""
        try:
            with self._lock:
                session = self._current_session
                if not session:
                    return
                
                # Cancel type silence monitoring task if active
                if session.mode == DictationMode.TYPE:
                    self._cancel_type_silence_task()
                
                # Don't clear session yet for smart dictation - keep it until LLM processing completes
                if session.mode != DictationMode.SMART:
                    self._current_session = None
            
            if session.mode == DictationMode.SMART:
                if session.accumulated_text:
                    await self._process_smart_dictation(session)
                    return  # Don't publish stop events yet - wait for LLM
                else:
                    # Clear session now since no LLM processing needed
                    with self._lock:
                        self._current_session = None
                    await self._end_smart_session()
            else:
                await self._finalize_session(session)
                
        except Exception as e:
            logger.error(f"Session stop error: {e}", exc_info=True)
    
    async def _process_smart_dictation(self, session: DictationSession) -> None:
        """Process smart dictation with LLM - wait for UI ready signal"""
        try:
            session.is_processing = True
            self._session_id = str(uuid.uuid4())
            self._pending_llm_session = session
            
            # Switch back to command audio mode
            await self._publish_event(AudioModeChangeRequestEvent(mode="command", reason="Smart dictation processing"))
            
            # Publish smart dictation stopped event (triggers UI)
            logger.info(f"Publishing SmartDictationStoppedEvent with text: '{session.accumulated_text[:50]}...'")
            await self._publish_event(SmartDictationStoppedEvent(raw_text=session.accumulated_text))
            logger.info("SmartDictationStoppedEvent published successfully")
            
            # Publish LLM processing started event with session ID
            agentic_prompt = self.agentic_service.get_current_prompt() or "Fix grammar and improve clarity."
            logger.info(f"Publishing LLMProcessingStartedEvent with session ID: {self._session_id}")
            await self._publish_event(LLMProcessingStartedEvent(
                raw_text=session.accumulated_text, 
                agentic_prompt=agentic_prompt,
                session_id=self._session_id
            ))
            logger.info("LLMProcessingStartedEvent published - waiting for UI ready signal...")
            
            # LLM processing will start when UI signals ready via LLMProcessingReadyEvent

        except Exception as e:
            logger.error(f"Smart dictation processing error: {e}", exc_info=True)
            
    async def _start_llm_processing(self, session: DictationSession) -> None:
        """Actually start the LLM processing after UI is ready"""
        try:
            agentic_prompt = self.agentic_service.get_current_prompt() or "Fix grammar and improve clarity."
            
            # Use streaming if available
            if hasattr(self.llm_service, 'process_dictation_streaming'):
                logger.info("Starting LLM streaming processing...")
                
                # Start background streaming thread
                self._start_streaming()
                
                try:
                    await self.llm_service.process_dictation_streaming(
                        session.accumulated_text, 
                        agentic_prompt,
                        token_callback=self._stream_token
                    )
                finally:
                    # Stop streaming and flush remaining tokens
                    self._stop_streaming()
                
                logger.info("LLM streaming processing completed")
            else:
                logger.info("Starting LLM non-streaming processing...")
                await self.llm_service.process_dictation(session.accumulated_text, agentic_prompt)
                logger.info("LLM non-streaming processing completed")

        except Exception as e:
            logger.error(f"LLM processing error: {e}", exc_info=True)
            self._stop_streaming()
    
    def _stream_token(self, token: str) -> None:
        """Fastest possible callback - just append to lock-free queue and return"""
        self._token_queue.append(token)
    
    def _streaming_worker(self) -> None:
        """Background thread that publishes tokens immediately - supports direct callback for zero-latency path"""
        while self._streaming_active:
            try:
                try:
                    token = self._token_queue.popleft()
                    
                    # Fast path: direct callback if available (bypasses event bus)
                    if self._direct_token_callback:
                        try:
                            self._direct_token_callback(token)
                        except Exception as e:
                            logger.error(f"Direct callback error: {e}", exc_info=True)
                    
                    # Standard path: event bus for other subscribers
                    self.event_publisher.publish(LLMTokenGeneratedEvent(token=token))
                    
                except IndexError:
                    time.sleep(0.0001)  # 0.1ms sleep when queue empty - ultra-responsive
                    continue
                    
            except Exception as e:
                logger.error(f"Streaming worker error: {e}", exc_info=True)
    
    def _start_streaming(self) -> None:
        """Start background streaming thread"""
        if not self._streaming_active:
            self._streaming_active = True
            self._streaming_thread = threading.Thread(
                target=self._streaming_worker,
                daemon=True,
                name="LLMTokenStreamer"
            )
            self._streaming_thread.start()
    
    def _stop_streaming(self) -> None:
        """Stop streaming and flush remaining tokens"""
        self._streaming_active = False
        if self._streaming_thread:
            self._streaming_thread.join(timeout=1.0)
            self._streaming_thread = None
        
        # Flush any remaining tokens
        remaining = []
        while True:
            try:
                remaining.append(self._token_queue.popleft())
            except IndexError:
                break
        
        if remaining:
            batched_token = ''.join(remaining)
            self.event_publisher.publish(LLMTokenGeneratedEvent(token=batched_token))
    
    async def _end_smart_session(self) -> None:
        """End smart dictation session"""
        try:
            await self._publish_event(AudioModeChangeRequestEvent(mode="command", reason="Smart dictation completed"))
            
            # Notify STT service about dictation mode deactivation
            await self._publish_event(
                DictationModeDisableOthersEvent(
                    dictation_mode_active=False,
                    dictation_mode="inactive"
                ))
            
            await self._publish_status(False, DictationMode.INACTIVE)
            logger.info("Smart dictation session ended")
        except Exception as e:
            logger.error(f"Smart session end error: {e}", exc_info=True)

    async def _finalize_session(self, session: DictationSession) -> None:
        """Finalize non-smart session"""
        try:
            await self._publish_event(AudioModeChangeRequestEvent(mode="command", reason="Dictation stopped"))
            
            # Notify STT service about dictation mode deactivation
            await self._publish_event(
                DictationModeDisableOthersEvent(
                    dictation_mode_active=False,
                    dictation_mode="inactive"
                ))
            
            await self._publish_status(False, DictationMode.INACTIVE)
            logger.info(f"{session.mode.value} dictation session completed")
        except Exception as e:
            logger.error(f"Session finalization error: {e}", exc_info=True)
    
    def _clean_text(self, text: str) -> str:
        """Clean dictated text by removing triggers"""
        if not text:
            return ""
            
        cfg = self.config.dictation
        triggers = {cfg.start_trigger.lower(), cfg.stop_trigger.lower(), 
                   cfg.type_trigger.lower(), cfg.smart_start_trigger.lower()}
        
        words = [w for w in text.split() if w.lower().strip('.,!?;:"()[]{}') not in triggers]
        return ' '.join(words).strip()
    
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
                stop_command=self.config.dictation.stop_trigger if is_active else None
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
        """Shutdown coordinator"""
        try:
            # Cancel type silence task if running
            self._cancel_type_silence_task()
            
            if self._current_session:
                await self._stop_session()
            await self.text_service.shutdown()
            await self.llm_service.shutdown()
            await self.agentic_service.shutdown()
            logger.info("Dictation coordinator shutdown complete")
        except Exception as e:
            logger.error(f"Shutdown error: {e}", exc_info=True)
