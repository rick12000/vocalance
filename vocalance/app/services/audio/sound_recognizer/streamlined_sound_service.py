import asyncio
import logging
from threading import RLock
from typing import Optional

import numpy as np

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.event_bus import EventBus
from vocalance.app.events.core_events import CustomSoundRecognizedEvent, ProcessAudioChunkForSoundRecognitionEvent
from vocalance.app.events.sound_events import (
    SoundListUpdatedEvent,
    SoundMappingsResponseEvent,
    SoundToCommandMappingUpdatedEvent,
    SoundTrainingCompleteEvent,
    SoundTrainingFailedEvent,
    SoundTrainingInitiatedEvent,
    SoundTrainingProgressEvent,
)
from vocalance.app.services.audio.sound_recognizer.streamlined_sound_recognizer import SoundRecognizer
from vocalance.app.services.base_service import Service
from vocalance.app.services.storage.storage_service import StorageService

logger = logging.getLogger(__name__)


class SoundService(Service):
    """Sound recognition, training, and sound→command mappings (thread-safe training state)."""

    def __init__(self, event_bus: EventBus, config: GlobalAppConfig, storage: StorageService) -> None:
        self.event_bus = event_bus
        self.config = config
        self.recognizer = SoundRecognizer(config=config, storage=storage)

        self.is_initialized = False

        self._training_lock = RLock()
        self._training_active = False
        self._current_training_label: Optional[str] = None
        self._training_samples = []
        self._target_samples = 0

        event_bus.subscribe(ProcessAudioChunkForSoundRecognitionEvent, self._handle_audio_chunk)

    async def initialize(self) -> bool:
        try:
            logger.info("Initializing SoundService...")
            self.is_initialized = await self.recognizer.initialize()
            if self.is_initialized:
                logger.info("SoundService initialized successfully")
                await self._publish_mappings()
            else:
                logger.error("Failed to initialize SoundService - recognizer initialization failed")
            return self.is_initialized
        except Exception as e:
            logger.error(f"Error initializing SoundService: {e}", exc_info=True)
            return False

    def get_sound_list(self) -> list:
        return list(self.recognizer.get_stats().get("trained_sounds", {}).keys())

    def get_sound_mappings(self) -> dict:
        return self.recognizer.get_stats().get("sound_mappings", {})

    async def _publish_mappings(self) -> None:
        await self.event_bus.publish(SoundMappingsResponseEvent(mappings=self.get_sound_mappings()))
        await self.event_bus.publish(SoundListUpdatedEvent(sounds=self.get_sound_list()))

    async def _handle_audio_chunk(self, audio_chunk: ProcessAudioChunkForSoundRecognitionEvent) -> None:
        try:
            audio_float32 = self._preprocess_audio_chunk(audio_bytes=audio_chunk.audio_chunk)
            sample_rate = audio_chunk.sample_rate

            with self._training_lock:
                training_active = self._training_active

            if training_active:
                await self._collect_training_sample(audio=audio_float32, sample_rate=sample_rate)
                return

            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, self.recognizer.recognize_sound, audio_float32, sample_rate)

            if result:
                sound_label, confidence = result
                if not sound_label.startswith("esc50_"):
                    command = self.recognizer.get_mapping(sound_label=sound_label)
                    await self.event_bus.publish(
                        CustomSoundRecognizedEvent(
                            label=sound_label,
                            confidence=confidence,
                            mapped_command=command or "",
                        )
                    )
                    logger.info(f"Recognized: {sound_label} (confidence: {confidence:.3f})")

        except ValueError as e:
            logger.error(f"Invalid audio format: {e}")
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}", exc_info=True)

    def _preprocess_audio_chunk(self, audio_bytes: bytes) -> np.ndarray:
        if not isinstance(audio_bytes, bytes):
            raise ValueError("Audio must be bytes")
        if len(audio_bytes) == 0:
            raise ValueError("Audio bytes are empty")
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        return audio_int16.astype(np.float32) / 32768.0

    # ── Public interface for direct callers ───────────────────────────────────

    async def start_training_session(self, sound_label: str, num_samples: int) -> bool:
        """Initiate a training session for a sound. Returns False if already training."""
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
            self._target_samples = num_samples

        await self.event_bus.publish(SoundTrainingInitiatedEvent(sound_name=sound_label, total_samples=num_samples))
        logger.info(f"Training initiated for '{sound_label}' - collecting {num_samples} samples")
        return True

    async def delete_sound(self, sound_label: str) -> bool:
        """Delete a trained sound. Returns True on success."""
        try:
            success = await self.recognizer.delete_sound(sound_label=sound_label)
            if success:
                await self._publish_mappings()
            logger.info(f"Delete sound '{sound_label}' - success: {success}")
            return success
        except Exception as e:
            logger.error(f"Error deleting sound: {e}", exc_info=True)
            return False

    async def reset_all_sounds(self) -> bool:
        """Reset all trained sounds. Returns True on success."""
        try:
            success = await self.recognizer.reset_all_sounds()
            if success:
                await self._publish_mappings()
            logger.info(f"Reset all sounds - success: {success}")
            return success
        except Exception as e:
            logger.error(f"Error resetting sounds: {e}", exc_info=True)
            return False

    async def map_sound_to_command(self, sound_label: str, command_phrase: str) -> bool:
        """Map a sound to a command phrase. Returns True on success."""
        try:
            success = await self.recognizer.set_mapping(
                sound_label=sound_label,
                command=command_phrase,
            )
            await self.event_bus.publish(
                SoundToCommandMappingUpdatedEvent(
                    sound_label=sound_label,
                    command_phrase=command_phrase,
                    success=success,
                )
            )
            if success:
                await self._publish_mappings()
                logger.info(f"Mapped sound '{sound_label}' to command '{command_phrase}'")
            else:
                logger.warning(f"Failed to save mapping for sound '{sound_label}'")
            return success
        except Exception as e:
            logger.error(f"Error mapping sound to command: {e}", exc_info=True)
            await self.event_bus.publish(
                SoundToCommandMappingUpdatedEvent(
                    sound_label=sound_label,
                    command_phrase=command_phrase,
                    success=False,
                )
            )
            return False

    async def _collect_training_sample(self, audio: np.ndarray, sample_rate: int) -> None:
        with self._training_lock:
            if not self._training_active:
                return

            sample_count = len(self._training_samples)
            target = self._target_samples

            if sample_count >= target:
                return

            try:
                preprocessed = self.recognizer.preprocessor.preprocess_audio(audio=audio.copy(), sr=sample_rate)
                self._training_samples.append((preprocessed, self.recognizer.target_sr))
            except Exception as e:
                logger.error(f"Failed to preprocess training sample: {e}")
                return

            sample_count = len(self._training_samples)
            label = self._current_training_label

        logger.info(f"Collected training sample {sample_count}/{target} for '{label}'")

        is_last = sample_count >= target
        await self.event_bus.publish(
            SoundTrainingProgressEvent(
                label=label,
                current_sample=sample_count,
                total_samples=target,
                is_last_sample=is_last,
            )
        )

        if is_last:
            logger.info(f"Target samples reached ({sample_count}/{target}), auto-finalizing training")
            await self.finish_training()

    async def finish_training(self) -> bool:
        with self._training_lock:
            if not self._training_active:
                logger.warning("No training session active")
                return False
            if not self._training_samples:
                logger.warning("No training samples collected")
                self._reset_training_state()
                return False
            label = self._current_training_label
            samples = self._training_samples.copy()

        try:
            success = await self.recognizer.train_sound(label=label, samples=samples)
            if success:
                logger.info(f"Training completed for '{label}' with {len(samples)} samples")
                await self.event_bus.publish(SoundTrainingCompleteEvent(sound_name=label, success=True))
                await self._publish_mappings()
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
        with self._training_lock:
            if self._training_active:
                label = self._current_training_label
                self._reset_training_state()
                logger.info(f"Cancelled training for '{label}'")

    def _reset_training_state(self) -> None:
        self._training_active = False
        self._current_training_label = None
        self._training_samples = []

    def start_training(self, sound_label: str) -> bool:
        """Synchronous training-state activation (used internally)."""
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

    async def set_sound_mapping(self, sound_label: str, command: str) -> bool:
        if not sound_label or not isinstance(sound_label, str):
            logger.error("Invalid sound label")
            return False
        if not command or not isinstance(command, str):
            logger.error("Invalid command")
            return False
        success = await self.recognizer.set_mapping(sound_label=sound_label, command=command)
        if success:
            logger.info(f"Mapped sound '{sound_label}' to command '{command}' and saved to storage")
        else:
            logger.warning(f"Failed to save mapping for sound '{sound_label}'")
        return success

    def get_sound_mapping(self, sound_label: str) -> Optional[str]:
        if not sound_label or not isinstance(sound_label, str):
            return None
        return self.recognizer.get_mapping(sound_label=sound_label)

    def get_stats(self) -> dict:
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
        with self._training_lock:
            return self._training_active

    def get_current_training_label(self) -> Optional[str]:
        with self._training_lock:
            return self._current_training_label

    def on_confidence_threshold_updated(self, threshold: float) -> None:
        self.recognizer.on_confidence_threshold_updated(threshold=threshold)

    def on_vote_threshold_updated(self, threshold: float) -> None:
        self.recognizer.on_vote_threshold_updated(threshold=threshold)

    async def shutdown(self) -> None:
        self.event_bus.unsubscribe(ProcessAudioChunkForSoundRecognitionEvent, self._handle_audio_chunk)
        try:
            self.cancel_training()
            if self.recognizer:
                await self.recognizer.shutdown()
        except Exception as e:
            logger.error("Error during SoundService shutdown: %s", e, exc_info=True)
