"""
Streamlined Sound Recognition Service.

Simplified service using the streamlined recognizer with focus on core functionality.
Implements proper async patterns and thread-safe operations.
"""
import asyncio
import logging
from threading import RLock
from typing import Optional

import numpy as np

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.event_bus import EventBus
from vocalance.app.events.core_events import CustomSoundRecognizedEvent, ProcessAudioChunkForSoundRecognitionEvent
from vocalance.app.events.sound_events import (
    AllSoundsResetEvent,
    DeleteSoundCommand,
    MapSoundToCommandPhraseCommand,
    RequestSoundListEvent,
    RequestSoundMappingsEvent,
    ResetAllSoundsCommand,
    SoundDeletedEvent,
    SoundListUpdatedEvent,
    SoundMappingsResponseEvent,
    SoundToCommandMappingUpdatedEvent,
    SoundTrainingCompleteEvent,
    SoundTrainingFailedEvent,
    SoundTrainingInitiatedEvent,
    SoundTrainingProgressEvent,
    SoundTrainingRequestEvent,
)
from vocalance.app.services.audio.sound_recognizer.streamlined_sound_recognizer import StreamlinedSoundRecognizer
from vocalance.app.services.storage.storage_service import StorageService

logger = logging.getLogger(__name__)


class StreamlinedSoundService:
    """Streamlined sound recognition service focused on core functionality."""

    def __init__(self, event_bus: EventBus, config: GlobalAppConfig, storage: StorageService):
        """Initialize service with thread-safe state management."""
        self.event_bus = event_bus
        self.config = config
        self.recognizer = StreamlinedSoundRecognizer(config=config, storage=storage)

        # State management
        self.is_initialized = False

        # Training state - protected by lock
        self._training_lock = RLock()
        self._training_active = False
        self._current_training_label: Optional[str] = None
        self._training_samples = []
        self._target_samples = 0

        # Shutdown coordination
        self._shutdown_event = asyncio.Event()

        logger.debug("StreamlinedSoundService created")

    def setup_subscriptions(self) -> None:
        """Set up all event subscriptions."""
        logger.debug("StreamlinedSoundService subscribing to events...")
        self.event_bus.subscribe(
            event_type=ProcessAudioChunkForSoundRecognitionEvent,
            handler=self._handle_audio_chunk,
        )
        self.event_bus.subscribe(
            event_type=SoundTrainingRequestEvent,
            handler=self._handle_training_request,
        )
        self.event_bus.subscribe(
            event_type=RequestSoundListEvent,
            handler=self._handle_sound_list_request,
        )
        self.event_bus.subscribe(
            event_type=RequestSoundMappingsEvent,
            handler=self._handle_mappings_request,
        )
        self.event_bus.subscribe(
            event_type=DeleteSoundCommand,
            handler=self._handle_delete_sound,
        )
        self.event_bus.subscribe(
            event_type=ResetAllSoundsCommand,
            handler=self._handle_reset_all_sounds,
        )
        self.event_bus.subscribe(
            event_type=MapSoundToCommandPhraseCommand,
            handler=self._handle_map_sound_command,
        )
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

    async def _handle_audio_chunk(self, event_data: ProcessAudioChunkForSoundRecognitionEvent) -> None:
        """Handle incoming audio chunks for recognition or training."""
        if not self.is_initialized:
            logger.debug("Service not initialized, ignoring audio chunk")
            return

        try:
            # Convert audio chunk to float32
            audio_float32 = self._preprocess_audio_chunk(audio_bytes=event_data.audio_chunk)
            sample_rate = event_data.sample_rate

            # Check if training is active
            with self._training_lock:
                training_active = self._training_active
                current_label = self._current_training_label

            if training_active:
                logger.debug(f"Training mode active for '{current_label}', collecting sample")
                await self._collect_training_sample(audio=audio_float32, sample_rate=sample_rate)
                return

            # Recognize sound
            result = self.recognizer.recognize_sound(audio=audio_float32, sr=sample_rate)

            if result:
                sound_label, confidence = result
                # Only publish custom sounds (not ESC-50 background sounds)
                if not sound_label.startswith("esc50_"):
                    command = self.recognizer.get_mapping(sound_label=sound_label)

                    recognition_event = CustomSoundRecognizedEvent(
                        label=sound_label,
                        confidence=confidence,
                        mapped_command=command or "",
                    )

                    await self.event_bus.publish(recognition_event)
                    logger.info(f"Recognized: {sound_label} (confidence: {confidence:.3f})")

        except ValueError as e:
            logger.error(f"Invalid audio format: {e}")
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}", exc_info=True)

    def _preprocess_audio_chunk(self, audio_bytes: bytes) -> np.ndarray:
        """Convert audio bytes to float32 numpy array."""
        if not isinstance(audio_bytes, bytes):
            raise ValueError("Audio must be bytes")

        if len(audio_bytes) == 0:
            raise ValueError("Audio bytes are empty")

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

        if not sound_label or not isinstance(sound_label, str):
            logger.error("Invalid sound label")
            return False

        with self._training_lock:
            if self._training_active:
                logger.warning(f"Training already active for '{self._current_training_label}'")
                return False

            self._training_active = True
            self._current_training_label = sound_label
            self._training_samples = []

        logger.info(f"Started training for sound: '{sound_label}'")
        return True

    async def _collect_training_sample(self, audio: np.ndarray, sample_rate: int) -> None:
        """Collect a training sample - thread-safe."""
        with self._training_lock:
            if not self._training_active:
                logger.debug("Training not active, ignoring sample")
                return

            self._training_samples.append((audio.copy(), sample_rate))
            sample_count = len(self._training_samples)
            target = self._target_samples
            label = self._current_training_label

        logger.info(f"Collected training sample {sample_count}/{target} for '{label}'")

        # Publish progress event
        is_last = sample_count >= target
        await self.event_bus.publish(
            SoundTrainingProgressEvent(
                label=label,
                current_sample=sample_count,
                total_samples=target,
                is_last_sample=is_last,
            )
        )

        # Auto-finish training after collecting enough samples
        if sample_count >= target:
            await self.finish_training()

    async def finish_training(self) -> bool:
        """Finish training and train the recognizer."""
        # Get training state
        with self._training_lock:
            if not self._training_active:
                logger.warning("No training session active")
                return False

            if not self._training_samples:
                logger.warning("No training samples collected")
                self._reset_training_state()
                return False

            # Copy data before releasing lock
            label = self._current_training_label
            samples = self._training_samples.copy()

        try:
            # Train the recognizer
            success = await self.recognizer.train_sound(label=label, samples=samples)

            if success:
                logger.info(f"Training completed for '{label}' with {len(samples)} samples")
                await self.event_bus.publish(SoundTrainingCompleteEvent(sound_name=label, success=True))
            else:
                logger.error(f"Training failed for '{label}'")
                await self.event_bus.publish(SoundTrainingFailedEvent(sound_name=label, reason="Training failed"))

            self._reset_training_state()
            return success

        except Exception as e:
            logger.error(f"Error during training: {e}", exc_info=True)
            with self._training_lock:
                label = self._current_training_label
            await self.event_bus.publish(SoundTrainingFailedEvent(sound_name=label, reason=str(e)))
            self._reset_training_state()
            return False

    def cancel_training(self) -> None:
        """Cancel current training session."""
        with self._training_lock:
            if self._training_active:
                label = self._current_training_label
                self._reset_training_state()
                logger.info(f"Cancelled training for '{label}'")

    def _reset_training_state(self) -> None:
        """Reset training state - must be called with lock held."""
        self._training_active = False
        self._current_training_label = None
        self._training_samples = []

    def set_sound_mapping(self, sound_label: str, command: str) -> None:
        """Set command mapping for a sound."""
        if not sound_label or not isinstance(sound_label, str):
            logger.error("Invalid sound label")
            return

        if not command or not isinstance(command, str):
            logger.error("Invalid command")
            return

        self.recognizer.set_mapping(sound_label=sound_label, command=command)
        logger.info(f"Mapped sound '{sound_label}' to command '{command}'")

    def get_sound_mapping(self, sound_label: str) -> Optional[str]:
        """Get command mapping for a sound."""
        if not sound_label or not isinstance(sound_label, str):
            return None

        return self.recognizer.get_mapping(sound_label=sound_label)

    def get_stats(self) -> dict:
        """Get service statistics."""
        stats = self.recognizer.get_stats()

        with self._training_lock:
            training_active = self._training_active
            current_label = self._current_training_label
            samples_collected = len(self._training_samples)

        stats.update(
            {
                "service_initialized": self.is_initialized,
                "training_active": training_active,
                "current_training_label": current_label,
                "training_samples_collected": samples_collected,
            }
        )
        return stats

    def is_training_active(self) -> bool:
        """Check if training is currently active."""
        with self._training_lock:
            return self._training_active

    def get_current_training_label(self) -> Optional[str]:
        """Get the label of the currently training sound."""
        with self._training_lock:
            return self._current_training_label

    # Event handlers
    async def _handle_training_request(self, event_data: SoundTrainingRequestEvent) -> None:
        """Handle sound training request."""
        try:
            logger.info(f"Starting training for sound: {event_data.sound_label}")

            # Publish training initiated event
            await self.event_bus.publish(
                SoundTrainingInitiatedEvent(
                    sound_name=event_data.sound_label,
                    total_samples=event_data.num_samples,
                )
            )

            # Start training mode - use lock
            with self._training_lock:
                self._training_active = True
                self._current_training_label = event_data.sound_label
                self._training_samples = []
                self._target_samples = event_data.num_samples

            logger.info(f"Training initiated for '{event_data.sound_label}' - collecting {event_data.num_samples} samples")

        except Exception as e:
            logger.error(f"Error handling training request: {e}", exc_info=True)
            await self.event_bus.publish(SoundTrainingFailedEvent(sound_name=event_data.sound_label, reason=str(e)))

    async def _handle_sound_list_request(self, event_data: RequestSoundListEvent) -> None:
        """Handle request for sound list."""
        try:
            sounds = list(self.recognizer.get_stats().get("trained_sounds", {}).keys())
            await self.event_bus.publish(SoundListUpdatedEvent(sounds=sounds))
            logger.debug(f"Published sound list with {len(sounds)} sounds")
        except Exception as e:
            logger.error(f"Error handling sound list request: {e}", exc_info=True)

    async def _handle_mappings_request(self, event_data: RequestSoundMappingsEvent) -> None:
        """Handle request for sound mappings."""
        try:
            stats = self.recognizer.get_stats()
            mappings = stats.get("sound_mappings", {})
            await self.event_bus.publish(SoundMappingsResponseEvent(mappings=mappings))
            logger.debug(f"Published sound mappings: {mappings}")
        except Exception as e:
            logger.error(f"Error handling mappings request: {e}", exc_info=True)

    async def _handle_delete_sound(self, event_data: DeleteSoundCommand) -> None:
        """Handle delete sound command."""
        try:
            success = await self.recognizer.delete_sound(sound_label=event_data.label)
            await self.event_bus.publish(SoundDeletedEvent(label=event_data.label, success=success))
            logger.info(f"Delete sound '{event_data.label}' - success: {success}")
        except Exception as e:
            logger.error(f"Error deleting sound: {e}", exc_info=True)
            await self.event_bus.publish(SoundDeletedEvent(label=event_data.label, success=False))

    async def _handle_reset_all_sounds(self, event_data: ResetAllSoundsCommand) -> None:
        """Handle reset all sounds command."""
        try:
            success = await self.recognizer.reset_all_sounds()
            await self.event_bus.publish(AllSoundsResetEvent(success=success))
            logger.info(f"Reset all sounds - success: {success}")
        except Exception as e:
            logger.error(f"Error resetting sounds: {e}", exc_info=True)
            await self.event_bus.publish(AllSoundsResetEvent(success=False))

    async def _handle_map_sound_command(self, event_data: MapSoundToCommandPhraseCommand) -> None:
        """Handle map sound to command phrase."""
        try:
            self.recognizer.set_mapping(
                sound_label=event_data.sound_label,
                command=event_data.command_phrase,
            )
            await self.event_bus.publish(
                SoundToCommandMappingUpdatedEvent(
                    sound_label=event_data.sound_label,
                    command_phrase=event_data.command_phrase,
                    success=True,
                )
            )
            logger.info(f"Mapped sound '{event_data.sound_label}' to command '{event_data.command_phrase}'")
        except Exception as e:
            logger.error(f"Error mapping sound to command: {e}", exc_info=True)
            await self.event_bus.publish(
                SoundToCommandMappingUpdatedEvent(
                    sound_label=event_data.sound_label,
                    command_phrase=event_data.command_phrase,
                    success=False,
                )
            )

    def on_confidence_threshold_updated(self, threshold: float) -> None:
        """
        Called by SettingsUpdateCoordinator when confidence threshold is updated.
        Forwards the update to the recognizer.
        """
        self.recognizer.on_confidence_threshold_updated(threshold=threshold)

    def on_vote_threshold_updated(self, threshold: float) -> None:
        """
        Called by SettingsUpdateCoordinator when vote threshold is updated.
        Forwards the update to the recognizer.
        """
        self.recognizer.on_vote_threshold_updated(threshold=threshold)

    async def shutdown(self) -> None:
        """Shutdown sound service and cleanup resources."""
        try:
            logger.info("Shutting down StreamlinedSoundService")

            # Signal shutdown
            self._shutdown_event.set()

            # Cancel any active training
            self.cancel_training()

            # Shutdown the recognizer (which will cleanup TensorFlow resources)
            if self.recognizer:
                await self.recognizer.shutdown()

            logger.info("StreamlinedSoundService shutdown complete")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)
