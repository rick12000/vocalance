import asyncio
import gc
import logging
import os
import pickle
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from threading import RLock
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
from scipy.spatial.distance import cosine

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.services.storage.storage_models import SoundMappingsData
from vocalance.app.services.storage.storage_service import StorageService

if TYPE_CHECKING:
    from vocalance.app.config.app_config import SoundRecognizerConfig

import tensorflow as tf

logger = logging.getLogger(__name__)

_EMBEDDING_DIM = 5120


class SimpleStandardScaler:
    def __init__(self) -> None:
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self._is_fitted = False

    def fit(self, X: np.ndarray) -> "SimpleStandardScaler":
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std = np.maximum(self.std, 0.01)
        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted or self.mean is None or self.std is None:
            raise ValueError("Scaler must be fit before transform")
        return (X - self.mean) / self.std

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class AudioPreprocessor:
    """Resample, optional silence trim, pad/crop duration, peak-normalize for YAMNet."""

    def __init__(self, config: "SoundRecognizerConfig") -> None:
        self.target_sr = config.target_sample_rate
        self.silence_threshold = config.silence_threshold
        self.min_sound_duration = config.min_sound_duration
        self.max_sound_duration = config.max_sound_duration
        self.frame_length = config.frame_length
        self.hop_length = config.hop_length
        self.normalization_level = config.normalization_level
        self.skip_silence_trimming = True

    def preprocess_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
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

        if not self.skip_silence_trimming:
            audio = self._trim_silence(audio=audio)

        duration = len(audio) / self.target_sr

        if duration < self.min_sound_duration:
            target_samples = int(self.min_sound_duration * self.target_sr)
            pad_total = target_samples - len(audio)
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            audio = np.pad(audio, (pad_left, pad_right), mode="constant")
            logger.debug(f"Padded audio symmetrically: {pad_left} left, {pad_right} right")
        elif duration > self.max_sound_duration:
            target_samples = int(self.max_sound_duration * self.target_sr)
            start_idx = (len(audio) - target_samples) // 2
            audio = audio[start_idx : start_idx + target_samples]
            logger.debug(f"Center-cropped audio from sample {start_idx} to {start_idx + target_samples}")

        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio * (self.normalization_level / peak)

        return audio

    def _trim_silence(self, audio: np.ndarray) -> np.ndarray:
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


class SoundRecognizer:
    """YAMNet temporal embeddings, k-NN + voting, persisted model and mappings (thread-safe)."""

    def __init__(self, config: GlobalAppConfig, storage: StorageService) -> None:
        self.asset_path_config = config.asset_paths
        self.config = config.sound_recognizer
        self._storage = storage

        storage_config = storage.storage_config
        self.model_path = storage_config.sound_model_dir
        self.external_sounds_path = storage_config.external_non_target_sounds_dir

        self.yamnet_model = None
        self.scaler = SimpleStandardScaler()
        self.embeddings: np.ndarray = np.empty((0, _EMBEDDING_DIM))
        self.labels: List[str] = []
        self.mappings: Dict[str, str] = {}

        self._model_lock = RLock()
        self._shutdown_event = asyncio.Event()

        self._file_io_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sound-file-io")

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

        logger.info("SoundRecognizer initialized")

    async def initialize(self) -> bool:
        try:
            logger.info("Initializing SoundRecognizer...")

            if tf is None:
                logger.error("TensorFlow not available")
                return False

            if not await self._initialize_yamnet_model():
                return False

            await self._load_model_data_async()

            logger.info(f"SoundRecognizer initialized: {len(self.embeddings)} embeddings")
            return True

        except ValueError as e:
            logger.error(f"Configuration error during initialization: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize recognizer: {e}", exc_info=True)
            return False

    async def warm_start_esc50_samples(self) -> None:
        try:
            logger.info("Warm-starting ESC-50 sample cache...")
            await self._copy_esc50_samples()
            logger.info("ESC-50 warm-start completed")
        except Exception as e:
            logger.error(f"Failed to warm-start ESC-50 samples (non-critical): {e}")

    async def _load_model_data_async(self) -> None:
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
                with open(labels_path, "rb") as f:
                    self.labels = pickle.load(f)
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)

            unique_sounds = len(set(self.labels))
            if unique_sounds > 0:
                logger.info(f"Loaded model data: {len(self.embeddings)} embeddings, {unique_sounds} unique sounds")

            await self._load_mappings_from_storage()

        except FileNotFoundError as e:
            logger.error(f"Model file not found: {e}")
        except Exception as e:
            logger.error(f"Failed to load model data: {e}", exc_info=True)
            with self._model_lock:
                self.embeddings = np.empty((0, _EMBEDDING_DIM))
                self.labels = []
                self.mappings = {}
                self.scaler = SimpleStandardScaler()

    async def _load_mappings_from_storage(self) -> None:
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

    def _save_model_files_sync(self, embeddings: np.ndarray, labels: List[str], scaler_obj: "SimpleStandardScaler") -> bool:
        try:
            np.save(os.path.join(self.model_path, "embeddings.npy"), embeddings)
            with open(os.path.join(self.model_path, "labels.joblib"), "wb") as f:
                pickle.dump(labels, f)
            with open(os.path.join(self.model_path, "scaler.joblib"), "wb") as f:
                pickle.dump(scaler_obj, f)
            logger.debug(f"Saved model files: {len(embeddings)} embeddings, {len(labels)} labels")
            return True
        except Exception as e:
            logger.error(f"Failed to save model files: {e}", exc_info=True)
            return False

    async def _save_model_data_async(self) -> bool:
        try:
            with self._model_lock:
                embeddings = self.embeddings.copy()
                labels = self.labels.copy()
                scaler_obj = self.scaler
                mappings = self.mappings.copy()

            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                self._file_io_executor, self._save_model_files_sync, embeddings, labels, scaler_obj
            )

            if not success:
                return False

            mappings_data = SoundMappingsData(mappings=mappings)
            return await self._storage.write(data=mappings_data)

        except Exception as e:
            logger.error(f"Failed to save model data: {e}", exc_info=True)
            return False

    async def _initialize_yamnet_model(self) -> bool:
        try:
            assets_yamnet_path = self.asset_path_config.yamnet_model_path
            app_yamnet_path = os.path.join(self.model_path, "yamnet")

            if await self._copy_yamnet_from_assets(assets_path=assets_yamnet_path, app_path=app_yamnet_path):
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
        try:
            if not os.path.exists(assets_path):
                logger.info(f"YAMNet model not found in assets at {assets_path}")
                return False

            if os.path.exists(app_path) and self._validate_yamnet_model(app_path):
                logger.info("YAMNet model already exists in app directory")
                return True

            if os.path.exists(app_path):
                shutil.rmtree(app_path)

            shutil.copytree(src=assets_path, dst=app_path)
            logger.info(f"YAMNet model copied from {assets_path} to {app_path}")

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
        try:
            variables_dir = os.path.join(model_path, "variables")

            if not os.path.exists(os.path.join(model_path, "saved_model.pb")):
                return False

            if not os.path.exists(variables_dir):
                return False

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
        try:
            assets_esc50_path = self.asset_path_config.esc50_samples_path

            needed_categories = []
            for category in self.esc50_categories.keys():
                try:
                    category_files = [
                        f
                        for f in os.listdir(self.external_sounds_path)
                        if f.startswith(f"esc50_{category}_") and f.endswith(".wav")
                    ]
                    if len(category_files) < self.max_esc50_per_cat:
                        needed_categories.append(category)
                except FileNotFoundError:
                    needed_categories.append(category)

            if not needed_categories:
                logger.debug("ESC-50 samples already present in app directory")
                return

            logger.info(f"Copying ESC-50 samples for categories: {needed_categories}")
            copied_count = await self._copy_categories_from_assets(assets_path=assets_esc50_path, categories=needed_categories)
            logger.info(f"Successfully copied {copied_count} ESC-50 samples from assets")

        except Exception as e:
            logger.debug(f"Failed to copy ESC-50 samples (non-critical): {e}")

    async def _copy_categories_from_assets(self, assets_path: str, categories: list) -> int:
        if not os.path.exists(assets_path):
            raise ValueError(f"ESC-50 assets not found at {assets_path}")

        copied_count = 0

        for category in categories:
            category_path = os.path.join(assets_path, category)
            if not os.path.exists(category_path):
                logger.warning(f"Category {category} not found in assets, skipping")
                continue

            wav_files = [f for f in os.listdir(category_path) if f.endswith(".wav")]
            files_to_copy = wav_files[: self.max_esc50_per_cat]

            for wav_file in files_to_copy:
                src = os.path.join(category_path, wav_file)
                dst = os.path.join(self.external_sounds_path, f"esc50_{category}_{wav_file}")

                if not os.path.exists(dst):
                    shutil.copy2(src=src, dst=dst)
                    copied_count += 1

        return copied_count

    def recognize_sound(self, audio: np.ndarray, sr: int) -> Optional[Tuple[str, float]]:
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

        embedding = self._extract_embedding(audio=audio, sr=sr)
        if embedding is None:
            return None

        try:
            scaled_embedding = scaler_obj.transform(embedding.reshape(1, -1))[0]
        except Exception as e:
            logger.error(f"Failed to scale embedding: {e}")
            return None

        similarities = np.array([1 - cosine(scaled_embedding, emb) for emb in embeddings_copy])

        top_indices = np.argsort(similarities)[-self.k_neighbors :][::-1]
        top_similarities = similarities[top_indices]
        top_labels = [labels_copy[i] for i in top_indices]

        best_similarity = top_similarities[0]
        if best_similarity < self.confidence_threshold:
            logger.debug(f"Recognition failed: similarity {best_similarity:.3f} < threshold {self.confidence_threshold}")
            return None

        all_votes = Counter(top_labels)
        total_votes = len(top_labels)

        custom_votes = {k: v for k, v in all_votes.items() if not k.startswith("esc50_")}

        if not custom_votes:
            logger.debug("Only background sounds detected")
            return None

        best_custom_label, custom_vote_count = max(custom_votes.items(), key=lambda x: x[1])

        vote_ratio = custom_vote_count / total_votes

        logger.debug(f"Recognition debug: top_labels={top_labels}")
        logger.debug(
            f"All votes: {all_votes}, best custom: {best_custom_label}, votes: {custom_vote_count}/{total_votes}, ratio: {vote_ratio:.3f}"
        )

        if vote_ratio >= self.vote_threshold:
            majority_indices = [i for i, label in enumerate(top_labels) if label == best_custom_label]
            confidence = np.mean([top_similarities[i] for i in majority_indices])

            logger.info(
                f"Sound recognized: '{best_custom_label}' (confidence: {confidence:.3f}, votes: {custom_vote_count}/{total_votes})"
            )
            return best_custom_label, confidence

        logger.debug(f"Insufficient vote alignment: {vote_ratio:.2f} (need {self.vote_threshold})")
        return None

    def _extract_embedding(self, audio: np.ndarray, sr: int) -> Optional[np.ndarray]:
        """YAMNet frame embeddings aggregated to shape (_EMBEDDING_DIM,)."""
        try:
            processed_audio = self.preprocessor.preprocess_audio(audio=audio, sr=sr)

            if tf is None:
                logger.error("TensorFlow not available for embedding extraction")
                return None

            audio_tensor = tf.convert_to_tensor(processed_audio, dtype=tf.float32)

            _, embeddings, _ = self.yamnet_model(audio_tensor)

            if hasattr(embeddings, "numpy"):
                embeddings_np = embeddings.numpy()
            else:
                embeddings_np = np.array(embeddings)

            temporal_embedding = self._aggregate_temporal_embeddings(embeddings_np)

            return temporal_embedding

        except ValueError as e:
            logger.error(f"Invalid audio for embedding: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to extract embedding: {e}")
            return None

    def _aggregate_temporal_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Mean/std over frames plus per-third max-pool (5×1024 = _EMBEDDING_DIM)."""
        num_frames = embeddings.shape[0]

        global_mean = np.mean(embeddings, axis=0)
        global_std = np.std(embeddings, axis=0)

        third = max(1, num_frames // 3)

        early = embeddings[:third]
        middle = embeddings[third : 2 * third]
        late = embeddings[2 * third :]

        early_max = np.max(early, axis=0) if len(early) > 0 else np.zeros(1024, dtype=np.float32)
        middle_max = np.max(middle, axis=0) if len(middle) > 0 else np.zeros(1024, dtype=np.float32)
        late_max = np.max(late, axis=0) if len(late) > 0 else np.zeros(1024, dtype=np.float32)

        temporal_embedding = np.concatenate([global_mean, global_std, early_max, middle_max, late_max])

        return temporal_embedding

    async def train_sound(self, label: str, samples: List[Tuple[np.ndarray, int]]) -> bool:
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
                    logger.warning(f"  Sample {i + 1}: invalid format, skipping")
                    continue

                audio, sr = sample_data
                embedding = self._extract_embedding(audio=audio, sr=sr)
                if embedding is not None:
                    new_embeddings.append(embedding)
                    new_labels.append(label)
                    logger.debug(f"  Sample {i + 1}: embedding extracted")
                else:
                    logger.warning(f"  Sample {i + 1}: failed to extract embedding")

            if not new_embeddings:
                logger.error(f"No valid embeddings extracted for '{label}'")
                return False

            with self._model_lock:
                if len(self.embeddings) == 0:
                    self.embeddings = np.array(new_embeddings)
                else:
                    self.embeddings = np.vstack([self.embeddings, new_embeddings])

                self.labels.extend(new_labels)

                self.scaler.fit(self.embeddings)

            logger.info(f"Training completed: {len(self.embeddings)} total embeddings")

            await self._add_esc50_samples()

            return await self._save_model_data_async()

        except ValueError as e:
            logger.error(f"Training input validation failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            return False

    def _extract_esc50_embeddings_sync(self) -> Tuple[List[np.ndarray], List[str]]:
        try:
            esc50_files = [f for f in os.listdir(self.external_sounds_path) if f.startswith("esc50_") and f.endswith(".wav")]
            if not esc50_files:
                return [], []

            esc50_embeddings = []
            esc50_labels = []

            for wav_file in esc50_files[: self.max_total_esc50]:
                try:
                    audio_data = sf.read(os.path.join(self.external_sounds_path, wav_file))
                    if not isinstance(audio_data, tuple) or len(audio_data) != 2:
                        continue

                    audio, sr = audio_data
                    if not isinstance(audio, np.ndarray) or len(audio) == 0:
                        continue
                    if not isinstance(sr, (int, np.integer)) or sr <= 0:
                        continue

                    embedding = self._extract_embedding(audio=audio, sr=sr)
                    if embedding is not None:
                        esc50_embeddings.append(embedding)
                        category = wav_file.split("_")[1]
                        esc50_labels.append(f"esc50_{category}")
                except Exception:
                    continue

            return esc50_embeddings, esc50_labels
        except Exception as e:
            logger.warning(f"Failed to extract ESC-50 embeddings: {e}")
            return [], []

    async def _add_esc50_samples(self) -> None:
        if not os.path.exists(self.external_sounds_path):
            return

        try:
            loop = asyncio.get_event_loop()
            esc50_embeddings, esc50_labels = await loop.run_in_executor(
                self._file_io_executor, self._extract_esc50_embeddings_sync
            )

            if not esc50_embeddings:
                return

            with self._model_lock:
                self.embeddings = np.vstack([self.embeddings, esc50_embeddings])
                self.labels.extend(esc50_labels)
                self.scaler.fit(self.embeddings)

            logger.info(f"Added {len(esc50_embeddings)} ESC-50 negative examples")
        except Exception as e:
            logger.warning(f"Failed to add ESC-50 samples: {e}")

    async def set_mapping(self, sound_label: str, command: str) -> bool:
        if not sound_label or not isinstance(sound_label, str):
            raise ValueError("Sound label must be a non-empty string")
        if not command or not isinstance(command, str):
            raise ValueError("Command must be a non-empty string")

        with self._model_lock:
            self.mappings[sound_label] = command

        success = await self._save_model_data_async()
        if success:
            logger.info(f"Successfully saved mapping '{sound_label}' -> '{command}' to storage")
        else:
            logger.warning(f"Failed to save mapping '{sound_label}' -> '{command}' to storage")

        return success

    def get_mapping(self, sound_label: str) -> Optional[str]:
        if not sound_label or not isinstance(sound_label, str):
            return None

        with self._model_lock:
            return self.mappings.get(sound_label)

    async def reset_all_sounds(self) -> bool:
        try:
            with self._model_lock:
                self.embeddings = np.empty((0, _EMBEDDING_DIM))
                self.labels = []
                self.mappings = {}
                self.scaler = SimpleStandardScaler()

            model_files = ["embeddings.npy", "labels.joblib", "scaler.joblib"]
            for filename in model_files:
                filepath = os.path.join(self.model_path, filename)
                if os.path.exists(filepath):
                    os.remove(filepath)
                    logger.debug(f"Removed model file: {filepath}")

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
        try:
            if not sound_label or not isinstance(sound_label, str):
                raise ValueError("Sound label must be a non-empty string")

            with self._model_lock:
                if sound_label not in self.labels:
                    logger.warning(f"Sound '{sound_label}' not found in trained sounds")
                    return False

                indices_to_remove = [i for i, label in enumerate(self.labels) if label == sound_label]

                if not indices_to_remove:
                    logger.warning(f"No embeddings found for sound '{sound_label}'")
                    return False

                mask = np.ones(len(self.embeddings), dtype=bool)
                mask[indices_to_remove] = False

                self.embeddings = self.embeddings[mask]
                self.labels = [label for i, label in enumerate(self.labels) if i not in indices_to_remove]

                if sound_label in self.mappings:
                    del self.mappings[sound_label]

                if len(self.embeddings) > 0:
                    self.scaler.fit(self.embeddings)
                else:
                    self.scaler = SimpleStandardScaler()

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
        """Sync instance threshold after settings coordinator updates config."""
        if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1:
            logger.warning(f"Invalid confidence threshold: {threshold}")
            return

        old_threshold = self.confidence_threshold
        self.confidence_threshold = threshold
        logger.info(f"Sound recognizer confidence threshold updated: {old_threshold:.3f} -> {threshold:.3f}")

    def on_vote_threshold_updated(self, threshold: float) -> None:
        """Sync instance vote threshold after settings coordinator updates config."""
        if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1:
            logger.warning(f"Invalid vote threshold: {threshold}")
            return

        old_threshold = self.vote_threshold
        self.vote_threshold = threshold
        logger.info(f"Sound recognizer vote threshold updated: {old_threshold:.3f} -> {threshold:.3f}")

    async def shutdown(self) -> None:
        try:
            logger.info("Shutting down SoundRecognizer")

            self._shutdown_event.set()

            if self._file_io_executor is not None:
                try:
                    loop = asyncio.get_event_loop()
                    await asyncio.wait_for(loop.run_in_executor(None, self._file_io_executor.shutdown, True), timeout=5.0)
                    logger.debug("File I/O executor shutdown complete")
                except asyncio.TimeoutError:
                    logger.warning("File I/O executor shutdown timed out, forcing shutdown")
                except Exception as e:
                    logger.warning(f"Error during executor shutdown: {e}")

            if self.yamnet_model is not None:
                del self.yamnet_model
                self.yamnet_model = None
                logger.info("YAMNet model deleted")

            if tf is not None:
                try:
                    tf.keras.backend.clear_session()
                    logger.info("TensorFlow Keras session cleared")
                except Exception as e:
                    logger.warning(f"Error clearing TensorFlow session: {e}")

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

            gc.collect()

            logger.info("SoundRecognizer shutdown complete")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)
