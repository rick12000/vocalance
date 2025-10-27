"""
Streamlined Sound Recognizer - Single cohesive implementation.

Optimized for lip-popping vs tongue-clicking discrimination with noise rejection.
Keeps only essential components: YAMNet embeddings, k-NN classification,
silence trimming, and ESC-50 negative examples.
"""
import asyncio
import gc
import logging
import os
import shutil
from collections import Counter
from threading import RLock
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import joblib
import librosa
import numpy as np
import soundfile as sf
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.services.storage.storage_models import SoundMappingsData
from vocalance.app.services.storage.storage_service import StorageService

if TYPE_CHECKING:
    from vocalance.app.config.app_config import SoundRecognizerConfig

# TensorFlow import - can be mocked for testing
try:
    import tensorflow as tf
except ImportError:
    tf = None

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """Essential audio preprocessing for consistent embeddings."""

    def __init__(self, config: "SoundRecognizerConfig"):
        """Initialize preprocessor from configuration."""
        self.target_sr = config.target_sample_rate
        self.silence_threshold = config.silence_threshold
        self.min_sound_duration = config.min_sound_duration
        self.max_sound_duration = config.max_sound_duration
        self.frame_length = config.frame_length
        self.hop_length = config.hop_length
        self.normalization_level = config.normalization_level

    def preprocess_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Essential preprocessing pipeline: resample, trim silence, normalize."""
        if not isinstance(audio, np.ndarray):
            raise TypeError("Audio must be a numpy array")

        if len(audio) == 0:
            raise ValueError("Audio array is empty")

        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        if not isinstance(sr, (int, np.integer)) or sr <= 0:
            raise ValueError(f"Invalid sample rate: {sr}")

        if sr != self.target_sr:
            try:
                audio = librosa.resample(y=audio, orig_sr=sr, target_sr=self.target_sr)
            except Exception as e:
                logger.error(f"Resample failed: sr={sr}, target={self.target_sr}, audio_shape={audio.shape}, error={e}")
                raise ValueError(f"Failed to resample audio: {e}")

        audio = self._trim_silence(audio=audio)

        duration = len(audio) / self.target_sr
        if duration < self.min_sound_duration:
            target_samples = int(self.min_sound_duration * self.target_sr)
            audio = np.pad(audio, (0, target_samples - len(audio)), mode="constant")
        elif duration > self.max_sound_duration:
            target_samples = int(self.max_sound_duration * self.target_sr)
            audio = audio[:target_samples]

        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio * (self.normalization_level / peak)

        return audio

    def _trim_silence(self, audio: np.ndarray) -> np.ndarray:
        """Trim silence using RMS energy analysis."""
        rms = librosa.feature.rms(y=audio, frame_length=self.frame_length, hop_length=self.hop_length)[0]

        sorted_rms = np.sort(rms)
        noise_floor = np.mean(sorted_rms[: len(sorted_rms) // 4])
        threshold = max(self.silence_threshold, noise_floor * 3)

        sound_frames = rms > threshold

        if not np.any(sound_frames):
            return audio

        sound_indices = np.where(sound_frames)[0]
        start_frame = max(0, sound_indices[0] - 2)
        end_frame = min(len(rms) - 1, sound_indices[-1] + 2)

        start_sample = start_frame * self.hop_length
        end_sample = min(len(audio), (end_frame + 1) * self.hop_length)

        return audio[start_sample:end_sample]


class StreamlinedSoundRecognizer:
    """Streamlined sound recognizer focused on core functionality."""

    def __init__(self, config: GlobalAppConfig, storage: StorageService):
        """Initialize recognizer with thread-safe state management."""
        self.asset_path_config = config.asset_paths
        self.config = config.sound_recognizer
        self._storage = storage

        storage_config = storage.storage_config
        self.model_path = storage_config.sound_model_dir
        self.external_sounds_path = storage_config.external_non_target_sounds_dir

        self.yamnet_model = None
        self.scaler = StandardScaler()
        self.embeddings: np.ndarray = np.empty((0, 1024))
        self.labels: List[str] = []
        self.mappings: Dict[str, str] = {}

        self._model_lock = RLock()
        self._shutdown_event = asyncio.Event()

        self.target_sr = self.config.target_sample_rate
        self.confidence_threshold = self.config.confidence_threshold
        self.k_neighbors = self.config.k_neighbors
        self.vote_threshold = self.config.vote_threshold

        self.esc50_categories = self.config.esc50_categories
        self.max_esc50_per_cat = self.config.max_esc50_samples_per_category
        self.max_total_esc50 = self.config.max_total_esc50_samples

        self.preprocessor = AudioPreprocessor(config=self.config)

        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.external_sounds_path, exist_ok=True)

        logger.info("StreamlinedSoundRecognizer initialized")

    async def initialize(self) -> bool:
        """Initialize YAMNet model and load existing data."""
        try:
            logger.info("Initializing StreamlinedSoundRecognizer...")

            if tf is None:
                logger.error("TensorFlow not available")
                return False

            if not await self._initialize_yamnet_model():
                return False

            await self._load_model_data_async()

            await self._copy_esc50_samples()

            logger.info(f"StreamlinedSoundRecognizer initialized: {len(self.embeddings)} embeddings")
            return True

        except ValueError as e:
            logger.error(f"Configuration error during initialization: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize recognizer: {e}", exc_info=True)
            return False

    async def _load_model_data_async(self) -> None:
        """Load saved model data asynchronously."""
        try:
            embeddings_path = os.path.join(self.model_path, "embeddings.npy")
            labels_path = os.path.join(self.model_path, "labels.joblib")
            scaler_path = os.path.join(self.model_path, "scaler.joblib")

            all_exist = all(os.path.exists(path) for path in [embeddings_path, labels_path, scaler_path])

            if not all_exist:
                logger.info("No existing model files found, starting with empty model")
                return

            with self._model_lock:
                self.embeddings = np.load(embeddings_path)
                self.labels = joblib.load(labels_path)
                self.scaler = joblib.load(scaler_path)

            unique_sounds = len(set(self.labels))
            logger.info(f"Loaded model data: {len(self.embeddings)} embeddings, {unique_sounds} unique sounds")

            await self._load_mappings_from_storage()

        except FileNotFoundError as e:
            logger.error(f"Model file not found: {e}")
        except Exception as e:
            logger.error(f"Failed to load model data: {e}", exc_info=True)
            with self._model_lock:
                self.embeddings = np.empty((0, 1024))
                self.labels = []
                self.mappings = {}
                self.scaler = StandardScaler()

    async def _load_mappings_from_storage(self) -> None:
        """Load sound mappings from storage service."""
        try:
            mappings_data = await self._storage.read(model_type=SoundMappingsData)
            if mappings_data:
                with self._model_lock:
                    self.mappings = mappings_data.mappings
                logger.info(f"Loaded {len(self.mappings)} sound mappings from storage")
            else:
                logger.info("No mappings found in storage")
        except Exception as e:
            logger.warning(f"Failed to load sound mappings from storage: {e}")
            with self._model_lock:
                self.mappings = {}

    async def _save_model_data_async(self) -> bool:
        """Save model data asynchronously."""
        try:
            with self._model_lock:
                embeddings = self.embeddings.copy()
                labels = self.labels.copy()
                scaler_obj = self.scaler
                mappings = self.mappings.copy()

            # Save numpy and joblib files
            np.save(os.path.join(self.model_path, "embeddings.npy"), embeddings)
            joblib.dump(labels, os.path.join(self.model_path, "labels.joblib"))
            joblib.dump(scaler_obj, os.path.join(self.model_path, "scaler.joblib"))

            logger.debug("Saved embeddings, labels, and scaler")

            # Save mappings through storage service
            try:
                mappings_data = SoundMappingsData(mappings=mappings)
                success = await self._storage.write(data=mappings_data)
                if success:
                    logger.debug("Successfully saved sound mappings to storage")
                else:
                    logger.warning("Failed to save sound mappings to storage")
                return success
            except Exception as e:
                logger.error(f"Error saving sound mappings: {e}")
                return False

        except Exception as e:
            logger.error(f"Failed to save model data: {e}", exc_info=True)
            return False

    async def _initialize_yamnet_model(self) -> bool:
        """Initialize YAMNet model by copying from assets."""
        try:
            # Get YAMNet path from config
            assets_yamnet_path = self.asset_path_config.yamnet_model_path
            app_yamnet_path = os.path.join(self.model_path, "yamnet")

            if await self._copy_yamnet_from_assets(assets_path=assets_yamnet_path, app_path=app_yamnet_path):
                # Load from app directory
                self.yamnet_model = tf.saved_model.load(app_yamnet_path)
                logger.info("YAMNet model copied from assets and loaded successfully")
                return True

            raise ValueError(f"YAMNet model not found in assets at {assets_yamnet_path}")

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to initialize YAMNet model: {e}")
            return False

    async def _copy_yamnet_from_assets(self, assets_path: str, app_path: str) -> bool:
        """Copy YAMNet model from assets to app directory."""
        try:
            if not os.path.exists(assets_path):
                logger.info(f"YAMNet model not found in assets at {assets_path}")
                return False

            # Check if already copied and valid
            if os.path.exists(app_path) and self._validate_yamnet_model(app_path):
                logger.info("YAMNet model already exists in app directory")
                return True

            # Copy the entire model directory
            if os.path.exists(app_path):
                shutil.rmtree(app_path)

            shutil.copytree(src=assets_path, dst=app_path)
            logger.info(f"YAMNet model copied from {assets_path} to {app_path}")

            # Validate the copied model
            if self._validate_yamnet_model(app_path):
                return True

            logger.error("Copied YAMNet model failed validation")
            return False

        except OSError as e:
            logger.error(f"File system error copying YAMNet model: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to copy YAMNet model from assets: {e}")
            return False

    def _validate_yamnet_model(self, model_path: str) -> bool:
        """Validate that the YAMNet model directory contains required files."""
        try:
            variables_dir = os.path.join(model_path, "variables")

            # Check for saved_model.pb
            if not os.path.exists(os.path.join(model_path, "saved_model.pb")):
                return False

            # Check for variables directory and files
            if not os.path.exists(variables_dir):
                return False

            # Check for variables files
            variables_files = os.listdir(variables_dir)
            if not any(f.startswith("variables.data") for f in variables_files):
                return False
            if not any(f == "variables.index" for f in variables_files):
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating YAMNet model: {e}")
            return False

    async def _copy_esc50_samples(self) -> None:
        """Copy ESC-50 samples from assets to app directory if needed."""
        try:
            # Get ESC-50 path from config
            assets_esc50_path = self.asset_path_config.esc50_samples_path

            # Check what categories we need
            needed_categories = []
            for category in self.esc50_categories.keys():
                # Check if any files exist for this category in app directory
                category_files = [
                    f for f in os.listdir(self.external_sounds_path) if f.startswith(f"esc50_{category}_") and f.endswith(".wav")
                ]
                if len(category_files) < self.max_esc50_per_cat:
                    needed_categories.append(category)

            if not needed_categories:
                logger.info("ESC-50 samples already present in app directory")
                return

            logger.info(f"Copying ESC-50 samples for categories: {needed_categories}")
            copied_count = await self._copy_categories_from_assets(assets_path=assets_esc50_path, categories=needed_categories)
            logger.info(f"Successfully copied {copied_count} ESC-50 samples from assets")

        except Exception as e:
            logger.error(f"Failed to copy ESC-50 samples: {e}", exc_info=True)

    async def _copy_categories_from_assets(self, assets_path: str, categories: list) -> int:
        """Copy specific ESC-50 categories from assets to app directory."""
        if not os.path.exists(assets_path):
            raise ValueError(f"ESC-50 assets not found at {assets_path}")

        copied_count = 0

        for category in categories:
            category_path = os.path.join(assets_path, category)
            if not os.path.exists(category_path):
                logger.warning(f"Category {category} not found in assets, skipping")
                continue

            # Get wav files from assets
            wav_files = [f for f in os.listdir(category_path) if f.endswith(".wav")]
            files_to_copy = wav_files[: self.max_esc50_per_cat]

            # Copy files to app directory
            for wav_file in files_to_copy:
                src = os.path.join(category_path, wav_file)
                dst = os.path.join(self.external_sounds_path, f"esc50_{category}_{wav_file}")

                if not os.path.exists(dst):
                    shutil.copy2(src=src, dst=dst)
                    copied_count += 1

        return copied_count

    def recognize_sound(self, audio: np.ndarray, sr: int) -> Optional[Tuple[str, float]]:
        """Core recognition method - thread-safe."""
        if not isinstance(audio, np.ndarray) or sr <= 0:
            logger.warning("Invalid audio input")
            return None

        with self._model_lock:
            if len(self.embeddings) == 0:
                logger.debug("No trained sounds available for recognition")
                return None

            embeddings_copy = self.embeddings.copy()
            labels_copy = self.labels.copy()
            scaler_obj = self.scaler

        # Extract embedding with preprocessing
        embedding = self._extract_embedding(audio=audio, sr=sr)
        if embedding is None:
            return None

        # Scale embedding
        try:
            scaled_embedding = scaler_obj.transform(embedding.reshape(1, -1))[0]
        except Exception as e:
            logger.error(f"Failed to scale embedding: {e}")
            return None

        # Calculate similarities
        similarities = cosine_similarity(scaled_embedding.reshape(1, -1), embeddings_copy)[0]

        # Get top-k neighbors
        top_indices = np.argsort(similarities)[-self.k_neighbors :][::-1]
        top_similarities = similarities[top_indices]
        top_labels = [labels_copy[i] for i in top_indices]

        # Confidence check
        best_similarity = top_similarities[0]
        if best_similarity < self.confidence_threshold:
            logger.debug(f"Recognition failed: similarity {best_similarity:.3f} < threshold {self.confidence_threshold}")
            return None

        # Voting - prioritize custom sounds over ESC-50
        custom_labels = [label for label in top_labels if not label.startswith("esc50_")]

        if not custom_labels:
            logger.debug("Only background sounds detected")
            return None

        # Simple majority voting among custom sounds
        votes = Counter(custom_labels)
        majority_label, vote_count = votes.most_common(1)[0]
        vote_ratio = vote_count / len(custom_labels)

        # Debug logging
        logger.debug(f"Recognition debug: top_labels={top_labels}, custom_labels={custom_labels}")
        logger.debug(f"Votes: {votes}, majority: {majority_label}, ratio: {vote_ratio:.3f}")

        if vote_ratio >= self.vote_threshold:
            # Calculate confidence as average similarity of majority votes
            majority_indices = [i for i, label in enumerate(top_labels) if label == majority_label]
            confidence = np.mean([top_similarities[i] for i in majority_indices])

            logger.info(f"Sound recognized: '{majority_label}' (confidence: {confidence:.3f})")
            return majority_label, confidence

        logger.debug(f"Insufficient vote alignment: {vote_ratio:.2f}")
        return None

    def _extract_embedding(self, audio: np.ndarray, sr: int) -> Optional[np.ndarray]:
        """Extract YAMNet embedding with preprocessing."""
        try:
            # Preprocess audio
            processed_audio = self.preprocessor.preprocess_audio(audio=audio, sr=sr)

            # Convert to tensor
            if tf is None:
                logger.error("TensorFlow not available for embedding extraction")
                return None

            audio_tensor = tf.convert_to_tensor(processed_audio, dtype=tf.float32)

            # Get YAMNet embeddings
            _, embeddings, _ = self.yamnet_model(audio_tensor)

            # Average embeddings across time
            embedding = tf.reduce_mean(embeddings, axis=0).numpy()

            return embedding

        except ValueError as e:
            logger.error(f"Invalid audio for embedding: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to extract embedding: {e}")
            return None

    async def train_sound(self, label: str, samples: List[Tuple[np.ndarray, int]]) -> bool:
        """Train the recognizer with sound samples."""
        try:
            if not label or not isinstance(label, str):
                raise ValueError("Sound label must be a non-empty string")

            if not samples or not isinstance(samples, list):
                raise ValueError("Samples must be a non-empty list")

            new_embeddings = []
            new_labels = []

            logger.info(f"Training '{label}' with {len(samples)} samples...")

            for i, sample_data in enumerate(samples):
                if not isinstance(sample_data, tuple) or len(sample_data) != 2:
                    logger.warning(f"  Sample {i+1}: invalid format, skipping")
                    continue

                audio, sr = sample_data
                embedding = self._extract_embedding(audio=audio, sr=sr)
                if embedding is not None:
                    new_embeddings.append(embedding)
                    new_labels.append(label)
                    logger.debug(f"  Sample {i+1}: embedding extracted")
                else:
                    logger.warning(f"  Sample {i+1}: failed to extract embedding")

            if not new_embeddings:
                logger.error(f"No valid embeddings extracted for '{label}'")
                return False

            # Add to existing data - thread-safe
            with self._model_lock:
                if len(self.embeddings) == 0:
                    self.embeddings = np.array(new_embeddings)
                else:
                    self.embeddings = np.vstack([self.embeddings, new_embeddings])

                self.labels.extend(new_labels)

                # Retrain scaler with all data
                self.scaler.fit(self.embeddings)

            logger.info(f"Training completed: {len(self.embeddings)} total embeddings")

            # Load and add ESC-50 samples as negative examples
            await self._add_esc50_samples()

            # Save updated model
            return await self._save_model_data_async()

        except ValueError as e:
            logger.error(f"Training input validation failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            return False

    async def _add_esc50_samples(self) -> None:
        """Add ESC-50 samples as negative examples."""
        if not os.path.exists(self.external_sounds_path):
            logger.debug("No external sounds path found, skipping ESC-50 samples")
            return

        esc50_files = [f for f in os.listdir(self.external_sounds_path) if f.startswith("esc50_") and f.endswith(".wav")]

        if not esc50_files:
            logger.debug("No ESC-50 files found, skipping negative examples")
            return

        # Limit total ESC-50 samples
        esc50_files = esc50_files[: self.max_total_esc50]

        esc50_embeddings = []
        esc50_labels = []

        for wav_file in esc50_files:
            try:
                audio_data = sf.read(os.path.join(self.external_sounds_path, wav_file))

                # soundfile.read() returns (audio, sr) tuple
                if isinstance(audio_data, tuple) and len(audio_data) == 2:
                    audio, sr = audio_data
                else:
                    logger.warning(f"Unexpected format from soundfile.read() for {wav_file}")
                    continue

                # Validate audio format
                if not isinstance(audio, np.ndarray):
                    logger.warning(f"Audio from {wav_file} is not numpy array, skipping")
                    continue

                if len(audio) == 0:
                    logger.warning(f"Audio from {wav_file} is empty, skipping")
                    continue

                if not isinstance(sr, (int, np.integer)) or sr <= 0:
                    logger.warning(f"Invalid sample rate {sr} from {wav_file}, skipping")
                    continue

                embedding = self._extract_embedding(audio=audio, sr=sr)

                if embedding is not None:
                    esc50_embeddings.append(embedding)
                    # Extract category from filename
                    category = wav_file.split("_")[1]  # esc50_category_file.wav
                    esc50_labels.append(f"esc50_{category}")

            except (FileNotFoundError, ValueError, TypeError) as e:
                logger.warning(f"Failed to read/validate ESC-50 file {wav_file}: {e}")
            except Exception as e:
                logger.warning(f"Failed to process ESC-50 file {wav_file}: {e}")

        if esc50_embeddings:
            with self._model_lock:
                # Add ESC-50 embeddings
                self.embeddings = np.vstack([self.embeddings, esc50_embeddings])
                self.labels.extend(esc50_labels)

                # Retrain scaler
                self.scaler.fit(self.embeddings)

            logger.info(f"Added {len(esc50_embeddings)} ESC-50 negative examples")
        else:
            logger.debug("No valid ESC-50 embeddings extracted")

    async def set_mapping(self, sound_label: str, command: str) -> bool:
        """Set command mapping for a sound and persist to storage - thread-safe.

        Returns:
            True if mapping was set and saved successfully, False otherwise.
        """
        if not sound_label or not isinstance(sound_label, str):
            raise ValueError("Sound label must be a non-empty string")
        if not command or not isinstance(command, str):
            raise ValueError("Command must be a non-empty string")

        with self._model_lock:
            self.mappings[sound_label] = command

        # Persist mappings to storage
        success = await self._save_model_data_async()
        if success:
            logger.info(f"Successfully saved mapping '{sound_label}' -> '{command}' to storage")
        else:
            logger.warning(f"Failed to save mapping '{sound_label}' -> '{command}' to storage")

        return success

    def get_mapping(self, sound_label: str) -> Optional[str]:
        """Get command mapping for a sound - thread-safe."""
        if not sound_label or not isinstance(sound_label, str):
            return None

        with self._model_lock:
            return self.mappings.get(sound_label)

    async def reset_all_sounds(self) -> bool:
        """Reset all trained sounds and mappings."""
        try:
            # Clear in-memory data - thread-safe
            with self._model_lock:
                self.embeddings = np.empty((0, 1024))
                self.labels = []
                self.mappings = {}
                self.scaler = StandardScaler()

            # Remove saved model files
            model_files = ["embeddings.npy", "labels.joblib", "scaler.joblib"]
            for filename in model_files:
                filepath = os.path.join(self.model_path, filename)
                if os.path.exists(filepath):
                    os.remove(filepath)
                    logger.debug(f"Removed model file: {filepath}")

            # Clear mappings through storage
            try:
                empty_mappings = SoundMappingsData(mappings={})
                success = await self._storage.write(data=empty_mappings)

                if success:
                    logger.debug("Successfully cleared sound mappings in storage")
                else:
                    logger.warning("Failed to clear sound mappings in storage")
                    return False
            except Exception as e:
                logger.error(f"Error clearing sound mappings: {e}")
                return False

            logger.info("Successfully reset all sounds and mappings")
            return True

        except OSError as e:
            logger.error(f"File system error during reset: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to reset sounds: {e}", exc_info=True)
            return False

    async def delete_sound(self, sound_label: str) -> bool:
        """Delete a specific trained sound - thread-safe."""
        try:
            if not sound_label or not isinstance(sound_label, str):
                raise ValueError("Sound label must be a non-empty string")

            with self._model_lock:
                if sound_label not in self.labels:
                    logger.warning(f"Sound '{sound_label}' not found in trained sounds")
                    return False

                # Find indices of embeddings for this sound
                indices_to_remove = [i for i, label in enumerate(self.labels) if label == sound_label]

                if not indices_to_remove:
                    logger.warning(f"No embeddings found for sound '{sound_label}'")
                    return False

                # Remove embeddings and labels
                mask = np.ones(len(self.embeddings), dtype=bool)
                mask[indices_to_remove] = False

                self.embeddings = self.embeddings[mask]
                self.labels = [label for i, label in enumerate(self.labels) if i not in indices_to_remove]

                # Remove mapping if it exists
                if sound_label in self.mappings:
                    del self.mappings[sound_label]

                # Retrain scaler if we still have data
                if len(self.embeddings) > 0:
                    self.scaler.fit(self.embeddings)
                else:
                    self.scaler = StandardScaler()

            # Save updated model
            success = await self._save_model_data_async()

            if success:
                logger.info(f"Successfully deleted sound '{sound_label}' ({len(indices_to_remove)} embeddings removed)")
            else:
                logger.error(f"Failed to save model after deleting '{sound_label}'")

            return success

        except ValueError as e:
            logger.error(f"Delete validation failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to delete sound '{sound_label}': {e}", exc_info=True)
            return False

    def get_stats(self) -> Dict:
        """Get recognizer statistics - thread-safe."""
        with self._model_lock:
            custom_sounds = [label for label in self.labels if not label.startswith("esc50_")]
            esc50_sounds = [label for label in self.labels if label.startswith("esc50_")]
            trained_sounds = list(set(custom_sounds))  # Unique custom sound names

            return {
                "total_embeddings": len(self.embeddings),
                "custom_sounds": len(set(custom_sounds)),
                "trained_sounds": {sound: self.labels.count(sound) for sound in trained_sounds},
                "esc50_samples": len(esc50_sounds),
                "mappings": len(self.mappings),
                "sound_mappings": self.mappings.copy(),
                "model_ready": len(self.embeddings) > 0,
            }

    def on_confidence_threshold_updated(self, threshold: float) -> None:
        """
        Called by SettingsUpdateCoordinator when confidence threshold is updated.
        Config is already updated - this updates the recognizer's instance variable.
        """
        if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1:
            logger.warning(f"Invalid confidence threshold: {threshold}")
            return

        old_threshold = self.confidence_threshold
        self.confidence_threshold = threshold
        logger.info(f"Sound recognizer confidence threshold updated: {old_threshold:.3f} -> {threshold:.3f}")

    def on_vote_threshold_updated(self, threshold: float) -> None:
        """
        Called by SettingsUpdateCoordinator when vote threshold is updated.
        Config is already updated - this updates the recognizer's instance variable.
        """
        if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1:
            logger.warning(f"Invalid vote threshold: {threshold}")
            return

        old_threshold = self.vote_threshold
        self.vote_threshold = threshold
        logger.info(f"Sound recognizer vote threshold updated: {old_threshold:.3f} -> {threshold:.3f}")

    async def shutdown(self) -> None:
        """Shutdown sound recognizer and cleanup TensorFlow resources."""
        try:
            logger.info("Shutting down StreamlinedSoundRecognizer")

            # Signal shutdown
            self._shutdown_event.set()

            # Clear TensorFlow model and free GPU/CPU memory
            if self.yamnet_model is not None:
                del self.yamnet_model
                self.yamnet_model = None
                logger.info("YAMNet model deleted")

            # Clear TensorFlow session using modern TensorFlow 2.x API
            if tf is not None:
                try:
                    tf.keras.backend.clear_session()
                    logger.info("TensorFlow Keras session cleared")
                except Exception as e:
                    logger.warning(f"Error clearing TensorFlow session: {e}")

            # Clear numpy arrays and other resources
            with self._model_lock:
                if self.embeddings is not None:
                    del self.embeddings
                    self.embeddings = None

                if self.labels:
                    self.labels.clear()
                    self.labels = None

                if self.scaler is not None:
                    del self.scaler
                    self.scaler = None

                if self.mappings:
                    self.mappings.clear()
                    self.mappings = None

            # Force garbage collection
            gc.collect()

            logger.info("StreamlinedSoundRecognizer shutdown complete")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)
