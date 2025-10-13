"""
Streamlined Sound Recognition Service.

Simplified service using the streamlined recognizer with focus on core functionality.
"""
import numpy as np
import logging
from typing import Optional

from iris.app.event_bus import EventBus
from iris.app.config.app_config import GlobalAppConfig
from iris.app.events.core_events import ProcessAudioChunkForSoundRecognitionEvent, CustomSoundRecognizedEvent
from iris.app.events.sound_events import (
    SoundTrainingRequestEvent, RequestSoundListEvent, RequestSoundMappingsEvent,
    DeleteSoundCommand, ResetAllSoundsCommand, MapSoundToCommandPhraseCommand,
    SoundListUpdatedEvent, SoundMappingsResponseEvent, SoundDeletedEvent,
    AllSoundsResetEvent, SoundToCommandMappingUpdatedEvent,
    SoundTrainingInitiatedEvent, SoundTrainingCompleteEvent, SoundTrainingFailedEvent,
    SoundTrainingProgressEvent
)
from iris.app.services.audio.sound_recognizer.streamlined_sound_recognizer import StreamlinedSoundRecognizer
from iris.app.services.storage.unified_storage_service import UnifiedStorageService

logger = logging.getLogger(__name__)


class StreamlinedSoundService:
    """Streamlined sound recognition service focused on core functionality."""
    
    def __init__(self, event_bus: EventBus, config: GlobalAppConfig, storage: UnifiedStorageService):
        self.event_bus = event_bus
        self.config = config
        self.recognizer = StreamlinedSoundRecognizer(config, storage)
        
        # State
        self.is_initialized = False
        self._training_active = False
        self._current_training_label: Optional[str] = None
        self._training_samples = []
        self._target_samples = 0
        
        # Subscribe to events
        logger.debug("StreamlinedSoundService subscribing to events...")
        self.event_bus.subscribe(ProcessAudioChunkForSoundRecognitionEvent, self._handle_audio_chunk)
        self.event_bus.subscribe(SoundTrainingRequestEvent, self._handle_training_request)
        self.event_bus.subscribe(RequestSoundListEvent, self._handle_sound_list_request)
        self.event_bus.subscribe(RequestSoundMappingsEvent, self._handle_mappings_request)
        self.event_bus.subscribe(DeleteSoundCommand, self._handle_delete_sound)
        self.event_bus.subscribe(ResetAllSoundsCommand, self._handle_reset_all_sounds)
        self.event_bus.subscribe(MapSoundToCommandPhraseCommand, self._handle_map_sound_command)
        logger.debug("StreamlinedSoundService event subscriptions complete")
    
    async def initialize(self) -> bool:
        """Initialize the sound recognition service."""
        try:
            logger.info("Initializing StreamlinedSoundService...")
            self.is_initialized = await self.recognizer.initialize()
            if self.is_initialized:
                logger.info("StreamlinedSoundService initialized successfully")
            else:
                logger.error("Failed to initialize StreamlinedSoundService - recognizer initialization failed")
            return self.is_initialized
        except Exception as e:
            logger.error(f"Error initializing StreamlinedSoundService: {e}", exc_info=True)
            return False
    
    async def _handle_audio_chunk(self, event_data: ProcessAudioChunkForSoundRecognitionEvent):
        """Handle incoming audio chunks for recognition or training."""
        logger.debug(f"StreamlinedSoundService received audio chunk: {len(event_data.audio_chunk)} bytes")
        
        if not self.is_initialized:
            logger.warning("Service not initialized, ignoring audio chunk")
            return
        
        try:
            # Convert audio chunk to float32
            audio_float32 = self._preprocess_audio_chunk(event_data.audio_chunk)
            sample_rate = event_data.sample_rate
            
            logger.debug(f"Processing audio chunk: training_active={self._training_active}, audio_size={len(audio_float32)}")
            
            # Handle training mode
            if self._training_active:
                logger.debug(f"Training mode active for '{self._current_training_label}', collecting sample")
                await self._collect_training_sample(audio_float32, sample_rate)
                return
            
            # Recognize sound
            result = self.recognizer.recognize_sound(audio_float32, sample_rate)
            
            if result:
                sound_label, confidence = result
                # Only publish custom sounds (not ESC-50 background sounds)
                if not sound_label.startswith('esc50_'):
                    command = self.recognizer.get_mapping(sound_label)
                    
                    recognition_event = CustomSoundRecognizedEvent(
                        label=sound_label,
                        confidence=confidence,
                        mapped_command=command or ""
                    )
                    
                    await self.event_bus.publish(recognition_event)
                    logger.info(f"Recognized: {sound_label} (confidence: {confidence:.3f})")
                
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}", exc_info=True)
    
    def _preprocess_audio_chunk(self, audio_bytes: bytes) -> np.ndarray:
        """Convert audio bytes to float32 numpy array."""
        # Convert bytes to int16 array
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # Convert to float32 in range [-1, 1]
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        
        return audio_float32
    
    async def start_training(self, sound_label: str) -> bool:
        """Start training mode for a specific sound."""
        if not self.is_initialized:
            logger.error("Service not initialized")
            return False
        
        if self._training_active:
            logger.warning(f"Training already active for '{self._current_training_label}'")
            return False
        
        self._training_active = True
        self._current_training_label = sound_label
        self._training_samples = []
        
        logger.info(f"Started training for sound: '{sound_label}'")
        return True
    
    async def _collect_training_sample(self, audio: np.ndarray, sample_rate: int):
        """Collect a training sample."""
        if not self._training_active:
            logger.debug("Training not active, ignoring sample")
            return
        
        logger.debug(f"Collecting training sample for '{self._current_training_label}' - audio length: {len(audio)}")
        self._training_samples.append((audio.copy(), sample_rate))
        sample_count = len(self._training_samples)
        
        logger.info(f"Collected training sample {sample_count}/{self._target_samples} for '{self._current_training_label}'")
        
        # Publish progress event
        is_last = sample_count >= self._target_samples
        await self.event_bus.publish(SoundTrainingProgressEvent(
            label=self._current_training_label,
            current_sample=sample_count,
            total_samples=self._target_samples,
            is_last_sample=is_last
        ))
        
        # Auto-finish training after collecting enough samples
        if sample_count >= self._target_samples:
            await self.finish_training()
    
    async def finish_training(self) -> bool:
        """Finish training and train the recognizer."""
        if not self._training_active:
            logger.warning("No training session active")
            return False
        
        if not self._training_samples:
            logger.warning("No training samples collected")
            self._reset_training_state()
            return False
        
        try:
            # Train the recognizer
            success = await self.recognizer.train_sound(
                self._current_training_label, 
                self._training_samples
            )
            
            if success:
                logger.info(f"Training completed for '{self._current_training_label}' "
                           f"with {len(self._training_samples)} samples")
                await self.event_bus.publish(SoundTrainingCompleteEvent(
                    sound_name=self._current_training_label,
                    success=True
                ))
            else:
                logger.error(f"Training failed for '{self._current_training_label}'")
                await self.event_bus.publish(SoundTrainingFailedEvent(
                    sound_name=self._current_training_label,
                    reason="Training failed"
                ))
            
            self._reset_training_state()
            return success
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            await self.event_bus.publish(SoundTrainingFailedEvent(
                sound_name=self._current_training_label,
                reason=str(e)
            ))
            self._reset_training_state()
            return False
    
    def cancel_training(self):
        """Cancel current training session."""
        if self._training_active:
            logger.info(f"Cancelled training for '{self._current_training_label}'")
            self._reset_training_state()
    
    def _reset_training_state(self):
        """Reset training state."""
        self._training_active = False
        self._current_training_label = None
        self._training_samples = []
    
    def set_sound_mapping(self, sound_label: str, command: str):
        """Set command mapping for a sound."""
        self.recognizer.set_mapping(sound_label, command)
        logger.info(f"Mapped sound '{sound_label}' to command '{command}'")
    
    def get_sound_mapping(self, sound_label: str) -> Optional[str]:
        """Get command mapping for a sound."""
        return self.recognizer.get_mapping(sound_label)
    
    def get_stats(self) -> dict:
        """Get service statistics."""
        stats = self.recognizer.get_stats()
        stats.update({
            'service_initialized': self.is_initialized,
            'training_active': self._training_active,
            'current_training_label': self._current_training_label,
            'training_samples_collected': len(self._training_samples)
        })
        return stats
    
    def is_training_active(self) -> bool:
        """Check if training is currently active."""
        return self._training_active
    
    def get_current_training_label(self) -> Optional[str]:
        """Get the label of the currently training sound."""
        return self._current_training_label
    
    # Event handlers
    async def _handle_training_request(self, event_data: SoundTrainingRequestEvent):
        """Handle sound training request."""
        try:
            logger.info(f"Starting training for sound: {event_data.sound_label}")
            
            # Publish training initiated event
            await self.event_bus.publish(SoundTrainingInitiatedEvent(
                sound_name=event_data.sound_label,
                total_samples=event_data.num_samples
            ))
            
            # Start training mode
            self._training_active = True
            self._current_training_label = event_data.sound_label
            self._training_samples = []
            self._target_samples = event_data.num_samples
            
            logger.info(f"Training initiated for '{event_data.sound_label}' - collecting {event_data.num_samples} samples")
            
        except Exception as e:
            logger.error(f"Error handling training request: {e}")
            await self.event_bus.publish(SoundTrainingFailedEvent(
                sound_name=event_data.sound_label,
                reason=str(e)
            ))
    
    async def _handle_sound_list_request(self, event_data: RequestSoundListEvent):
        """Handle request for sound list."""
        try:
            sounds = list(self.recognizer.get_stats().get('trained_sounds', {}).keys())
            await self.event_bus.publish(SoundListUpdatedEvent(sounds=sounds))
            logger.debug(f"Published sound list with {len(sounds)} sounds")
        except Exception as e:
            logger.error(f"Error handling sound list request: {e}")
    
    async def _handle_mappings_request(self, event_data: RequestSoundMappingsEvent):
        """Handle request for sound mappings."""
        try:
            stats = self.recognizer.get_stats()
            mappings = stats.get('sound_mappings', {})
            await self.event_bus.publish(SoundMappingsResponseEvent(mappings=mappings))
            logger.debug(f"Published sound mappings: {mappings}")
        except Exception as e:
            logger.error(f"Error handling mappings request: {e}")
    
    async def _handle_delete_sound(self, event_data: DeleteSoundCommand):
        """Handle delete sound command."""
        try:
            success = self.recognizer.delete_sound(event_data.label)
            await self.event_bus.publish(SoundDeletedEvent(
                label=event_data.label,
                success=success
            ))
            logger.info(f"Delete sound '{event_data.label}' - success: {success}")
        except Exception as e:
            logger.error(f"Error deleting sound: {e}")
            await self.event_bus.publish(SoundDeletedEvent(
                label=event_data.label,
                success=False
            ))
    
    async def _handle_reset_all_sounds(self, event_data: ResetAllSoundsCommand):
        """Handle reset all sounds command."""
        try:
            success = self.recognizer.reset_all_sounds()
            await self.event_bus.publish(AllSoundsResetEvent(success=success))
            logger.info(f"Reset all sounds - success: {success}")
        except Exception as e:
            logger.error(f"Error resetting sounds: {e}")
            await self.event_bus.publish(AllSoundsResetEvent(success=False))
    
    async def _handle_map_sound_command(self, event_data: MapSoundToCommandPhraseCommand):
        """Handle map sound to command phrase."""
        try:
            self.recognizer.set_mapping(event_data.sound_label, event_data.command_phrase)
            await self.event_bus.publish(SoundToCommandMappingUpdatedEvent(
                sound_label=event_data.sound_label,
                command_phrase=event_data.command_phrase,
                success=True
            ))
            logger.info(f"Mapped sound '{event_data.sound_label}' to command '{event_data.command_phrase}'")
        except Exception as e:
            logger.error(f"Error mapping sound to command: {e}")
            await self.event_bus.publish(SoundToCommandMappingUpdatedEvent(
                sound_label=event_data.sound_label,
                command_phrase=event_data.command_phrase,
                success=False
            ))
    
    async def shutdown(self) -> None:
        """Shutdown sound service and cleanup resources"""
        try:
            logger.info("Shutting down StreamlinedSoundService")
            
            # Cancel any active training
            if self._training_active:
                self.cancel_training()
            
            # Shutdown the recognizer (which will cleanup TensorFlow resources)
            if hasattr(self, 'recognizer') and self.recognizer:
                await self.recognizer.shutdown()
            
            logger.info("StreamlinedSoundService shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during StreamlinedSoundService shutdown: {e}", exc_info=True)