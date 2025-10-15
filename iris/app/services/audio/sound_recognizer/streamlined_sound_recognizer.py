"""
Streamlined Sound Recognizer - Single cohesive implementation.

Optimized for lip-popping vs tongue-clicking discrimination with noise rejection.
Keeps only essential components: YAMNet embeddings, k-NN classification, 
silence trimming, and ESC-50 negative examples.
"""
import asyncio
import concurrent.futures
import gc
import os
import sys
import requests
import zipfile
import tempfile
import shutil
import csv
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import logging

from iris.app.services.storage.storage_service import StorageService
from iris.app.services.storage.storage_models import SoundMappingsData
from iris.app.config.app_config import GlobalAppConfig

# TensorFlow import - can be mocked for testing
try:
    import tensorflow as tf
except ImportError:
    tf = None

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """Essential audio preprocessing for consistent embeddings."""
    
    def __init__(self, 
                 target_sr: int = 16000,
                 silence_threshold: float = 0.005,
                 min_sound_duration: float = 0.1,
                 max_sound_duration: float = 2.0):
        self.target_sr = target_sr
        self.silence_threshold = silence_threshold
        self.min_sound_duration = min_sound_duration
        self.max_sound_duration = max_sound_duration
    
    def preprocess_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Essential preprocessing pipeline: resample, trim silence, normalize."""
        # Convert to mono if needed
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        
        # Resample if needed
        if sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
        
        # Trim silence - CRITICAL for consistent embeddings
        audio = self._trim_silence(audio)
        
        # Validate and adjust duration
        duration = len(audio) / self.target_sr
        if duration < self.min_sound_duration:
            # Pad to minimum duration
            target_samples = int(self.min_sound_duration * self.target_sr)
            audio = np.pad(audio, (0, target_samples - len(audio)), mode='constant')
        elif duration > self.max_sound_duration:
            # Truncate to maximum duration
            target_samples = int(self.max_sound_duration * self.target_sr)
            audio = audio[:target_samples]
        
        # Simple normalization
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio * (0.7 / peak)  # Normalize to 70% of max amplitude
        
        return audio
    
    def _trim_silence(self, audio: np.ndarray) -> np.ndarray:
        """Trim silence using RMS energy analysis."""
        frame_length = 1024
        hop_length = 512
        
        # Calculate RMS energy per frame
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Adaptive threshold
        sorted_rms = np.sort(rms)
        noise_floor = np.mean(sorted_rms[:len(sorted_rms)//4])  # Bottom 25%
        threshold = max(self.silence_threshold, noise_floor * 3)
        
        # Find sound boundaries
        sound_frames = rms > threshold
        
        if not np.any(sound_frames):
            return audio  # No trimming if no sound detected
        
        # Get first and last sound frames with padding
        sound_indices = np.where(sound_frames)[0]
        start_frame = max(0, sound_indices[0] - 2)  # Small padding
        end_frame = min(len(rms) - 1, sound_indices[-1] + 2)
        
        # Convert to sample indices
        start_sample = start_frame * hop_length
        end_sample = min(len(audio), (end_frame + 1) * hop_length)
        
        return audio[start_sample:end_sample]


class StreamlinedSoundRecognizer:
    """Streamlined sound recognizer focused on core functionality."""
    
    def __init__(self, config: GlobalAppConfig, storage: StorageService):
        self.asset_path_config = config.asset_paths  # Store full config to access asset_paths
        self.config = config.sound_recognizer
        self._storage = storage
        
        # Get storage paths from storage service
        storage_config = storage.storage_config
        self.model_path = storage_config.sound_model_dir
        self.external_sounds_path = storage_config.external_non_target_sounds_dir
        
        # Core components
        self.yamnet_model = None
        self.scaler = StandardScaler()
        self.embeddings: np.ndarray = np.empty((0, 1024))  # YAMNet embedding size
        self.labels: List[str] = []
        self.mappings: Dict[str, str] = {}
        
        # Configuration
        self.target_sr = self.config.target_sample_rate
        self.confidence_threshold = self.config.confidence_threshold
        self.k_neighbors = self.config.k_neighbors
        self.vote_threshold = self.config.vote_threshold
        
        # ESC-50 configuration
        self.esc50_categories = self.config.esc50_categories
        self.max_esc50_per_cat = self.config.max_esc50_samples_per_category
        self.max_total_esc50 = self.config.max_total_esc50_samples
        
        # Audio preprocessing
        self.preprocessor = AudioPreprocessor(
            target_sr=self.target_sr,
            silence_threshold=0.005,
            min_sound_duration=0.1,
            max_sound_duration=2.0
        )
        
        # Create directories once
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.external_sounds_path, exist_ok=True)
    
    async def initialize(self) -> bool:
        """Initialize YAMNet model and load existing data."""
        try:
            # Load YAMNet model
            if tf is None:
                logger.error("TensorFlow not available")
                return False
            
            # Initialize YAMNet model with copy-first strategy
            if not await self._initialize_yamnet_model():
                return False
            
            # Load existing data
            self._load_model_data()
            
            # Copy ESC-50 samples if needed
            await self._copy_esc50_samples()
            
            logger.info(f"StreamlinedSoundRecognizer initialized: {len(self.embeddings)} embeddings")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize recognizer: {e}")
            return False
    
    def _load_model_data(self):
        """Load saved model data."""
        try:
            embeddings_path = os.path.join(self.model_path, "embeddings.npy")
            labels_path = os.path.join(self.model_path, "labels.joblib")
            scaler_path = os.path.join(self.model_path, "scaler.joblib")
            
            # Load embeddings, labels, and scaler from files
            if all(os.path.exists(path) for path in [embeddings_path, labels_path, scaler_path]):
                self.embeddings = np.load(embeddings_path)
                self.labels = joblib.load(labels_path)
                self.scaler = joblib.load(scaler_path)
                logger.info(f"Loaded model data: {len(self.embeddings)} embeddings, {len(set(self.labels))} unique sounds")
            else:
                logger.info("No existing model files found, starting with empty model")
            
            # Load mappings from storage
            try:
                # Use a thread pool to run the async operation
                async def load_mappings():
                    mappings_data = await self._storage.read(model_type=SoundMappingsData)
                    return mappings_data.mappings
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, load_mappings())
                    mappings = future.result(timeout=10)  # 10-second timeout
                
                self.mappings = mappings
                logger.info(f"Loaded {len(self.mappings)} sound mappings from storage")
            except Exception as mapping_error:
                logger.warning(f"Failed to load sound mappings from storage: {mapping_error}")
                self.mappings = {}
            
        except Exception as e:
            logger.error(f"Failed to load model data: {e}")
            # Reset to empty state on load failure
            self.embeddings = np.empty((0, 1024))
            self.labels = []
            self.mappings = {}
            self.scaler = StandardScaler()
    
    def _save_model_data(self):
        """Save model data."""
        try:
            # Save embeddings, labels, and scaler to files
            np.save(os.path.join(self.model_path, "embeddings.npy"), self.embeddings)
            joblib.dump(self.labels, os.path.join(self.model_path, "labels.joblib"))
            joblib.dump(self.scaler, os.path.join(self.model_path, "scaler.joblib"))
            
            # Save mappings through storage
            try:
                # Use a thread pool to run the async operation
                async def save_mappings():
                    mappings_data = SoundMappingsData(mappings=self.mappings)
                    return await self._storage.write(data=mappings_data)
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, save_mappings())
                    success = future.result(timeout=10)  # 10-second timeout
                
                if success:
                    logger.debug("Successfully saved sound mappings to storage")
                else:
                    logger.warning("Failed to save sound mappings to storage")
            except Exception as mapping_error:
                logger.error(f"Error saving sound mappings: {mapping_error}")
            
        except Exception as e:
            logger.error(f"Failed to save model data: {e}")
    
    async def _initialize_yamnet_model(self) -> bool:
        """Initialize YAMNet model by copying from assets."""
        try:
            # Get YAMNet path from config
            assets_yamnet_path = self.asset_path_config.yamnet_model_path
            app_yamnet_path = os.path.join(self.model_path, "yamnet")
            if await self._copy_yamnet_from_assets(assets_yamnet_path, app_yamnet_path):
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
            
            shutil.copytree(assets_path, app_path)
            logger.info(f"YAMNet model copied from {assets_path} to {app_path}")
            
            # Validate the copied model
            if self._validate_yamnet_model(app_path):
                return True
            else:
                logger.error("Copied YAMNet model failed validation")
                return False
                
        except Exception as e:
            logger.error(f"Failed to copy YAMNet model from assets: {e}")
            return False
    
    def _validate_yamnet_model(self, model_path: str) -> bool:
        """Validate that the YAMNet model directory contains required files."""
        try:
            required_files = ["saved_model.pb"]
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
    
    async def _copy_esc50_samples(self):
        """Copy ESC-50 samples from assets to app directory if needed."""
        try:
            # Get ESC-50 path from config
            assets_esc50_path = self.asset_path_config.esc50_samples_path

            # Check what categories we need
            needed_categories = []
            for category in self.esc50_categories.keys():
                # Check if any files exist for this category in app directory
                category_files = [f for f in os.listdir(self.external_sounds_path)
                                if f.startswith(f'esc50_{category}_') and f.endswith('.wav')]
                if len(category_files) < self.max_esc50_per_cat:
                    needed_categories.append(category)

            if not needed_categories:
                logger.info("ESC-50 samples already present in app directory")
                return

            logger.info(f"Copying ESC-50 samples for categories: {needed_categories}")
            copied_count = await self._copy_categories_from_assets(assets_esc50_path, needed_categories)
            logger.info(f"Successfully copied {copied_count} ESC-50 samples from assets")

        except Exception as e:
            logger.error(f"Failed to copy ESC-50 samples: {e}")
            raise
    
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
            wav_files = [f for f in os.listdir(category_path) if f.endswith('.wav')]
            files_to_copy = wav_files[:self.max_esc50_per_cat]
            
            # Copy files to app directory
            for wav_file in files_to_copy:
                src = os.path.join(category_path, wav_file)
                dst = os.path.join(self.external_sounds_path, f"esc50_{category}_{wav_file}")
                
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    copied_count += 1
        
        return copied_count
    
    def recognize_sound(self, audio: np.ndarray, sr: int) -> Optional[Tuple[str, float]]:
        """Core recognition method."""
        if len(self.embeddings) == 0:
            logger.warning("No trained sounds available for recognition")
            return None
        
        # Extract embedding with preprocessing
        embedding = self._extract_embedding(audio, sr)
        if embedding is None:
            return None
        
        # Scale embedding
        scaled_embedding = self.scaler.transform(embedding.reshape(1, -1))[0]
        
        # Calculate similarities
        similarities = cosine_similarity(scaled_embedding.reshape(1, -1), self.embeddings)[0]
        
        # Get top-k neighbors
        top_indices = np.argsort(similarities)[-self.k_neighbors:][::-1]
        top_similarities = similarities[top_indices]
        top_labels = [self.labels[i] for i in top_indices]
        
        # Confidence check
        best_similarity = top_similarities[0]
        if best_similarity < self.confidence_threshold:
            logger.debug(f"Recognition failed: similarity {best_similarity:.3f} < threshold {self.confidence_threshold}")
            return None
        
        # Voting - prioritize custom sounds over ESC-50
        custom_labels = [label for label in top_labels if not label.startswith('esc50_')]
        
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
            processed_audio = self.preprocessor.preprocess_audio(audio, sr)
            
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
            
        except Exception as e:
            logger.error(f"Failed to extract embedding: {e}")
            return None
    
    async def train_sound(self, label: str, samples: List[Tuple[np.ndarray, int]]) -> bool:
        """Train the recognizer with sound samples."""
        try:
            new_embeddings = []
            new_labels = []
            
            logger.info(f"Training '{label}' with {len(samples)} samples...")
            
            for i, (audio, sr) in enumerate(samples):
                embedding = self._extract_embedding(audio, sr)
                if embedding is not None:
                    new_embeddings.append(embedding)
                    new_labels.append(label)
                    logger.debug(f"  Sample {i+1}: embedding extracted")
                else:
                    logger.warning(f"  Sample {i+1}: failed to extract embedding")
            
            if not new_embeddings:
                logger.error(f"No valid embeddings extracted for '{label}'")
                return False
            
            # Add to existing data
            if len(self.embeddings) == 0:
                self.embeddings = np.array(new_embeddings)
            else:
                self.embeddings = np.vstack([self.embeddings, new_embeddings])
            
            self.labels.extend(new_labels)
            
            # Retrain scaler with all data
            self.scaler.fit(self.embeddings)
            
            # Load and add ESC-50 samples as negative examples
            await self._add_esc50_samples()
            
            # Save updated model
            self._save_model_data()
            
            logger.info(f"Training completed: {len(self.embeddings)} total embeddings")
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False
    
    async def _add_esc50_samples(self):
        """Add ESC-50 samples as negative examples."""
        if not os.path.exists(self.external_sounds_path):
            logger.info("No external sounds path found, skipping ESC-50 samples")
            return
        
        esc50_files = [f for f in os.listdir(self.external_sounds_path) if f.startswith('esc50_') and f.endswith('.wav')]
        
        if not esc50_files:
            logger.info("No ESC-50 files found, skipping negative examples")
            return
        
        # Limit total ESC-50 samples
        esc50_files = esc50_files[:self.max_total_esc50]
        
        esc50_embeddings = []
        esc50_labels = []
        
        for wav_file in esc50_files:
            try:
                audio, sr = sf.read(os.path.join(self.external_sounds_path, wav_file))
                embedding = self._extract_embedding(audio, sr)
                
                if embedding is not None:
                    esc50_embeddings.append(embedding)
                    # Extract category from filename
                    category = wav_file.split('_')[1]  # esc50_category_file.wav
                    esc50_labels.append(f"esc50_{category}")
                    
            except Exception as e:
                logger.warning(f"Failed to process ESC-50 file {wav_file}: {e}")
        
        if esc50_embeddings:
            # Add ESC-50 embeddings
            self.embeddings = np.vstack([self.embeddings, esc50_embeddings])
            self.labels.extend(esc50_labels)
            
            # Retrain scaler
            self.scaler.fit(self.embeddings)
            
            logger.info(f"Added {len(esc50_embeddings)} ESC-50 negative examples")
        else:
            logger.info("No valid ESC-50 embeddings extracted")
    
    def set_mapping(self, sound_label: str, command: str):
        """Set command mapping for a sound."""
        self.mappings[sound_label] = command
        self._save_model_data()
    
    def get_mapping(self, sound_label: str) -> Optional[str]:
        """Get command mapping for a sound."""
        return self.mappings.get(sound_label)
    
    def reset_all_sounds(self) -> bool:
        """Reset all trained sounds and mappings."""
        try:
            # Clear in-memory data
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
                # Use a thread pool to run the async operation
                async def clear_mappings():
                    from iris.app.services.storage.storage_models import SoundMappingsData
                    empty_mappings = SoundMappingsData(mappings={})
                    return await self._storage.write(data=empty_mappings)
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, clear_mappings())
                    success = future.result(timeout=10)  # 10-second timeout
                
                if success:
                    logger.debug("Successfully cleared sound mappings in storage")
                else:
                    logger.warning("Failed to clear sound mappings in storage")
            except Exception as mapping_error:
                logger.error(f"Error clearing sound mappings: {mapping_error}")
            
            logger.info("Successfully reset all sounds and mappings")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset sounds: {e}")
            return False
    
    def delete_sound(self, sound_label: str) -> bool:
        """Delete a specific trained sound."""
        try:
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
            self._save_model_data()
            
            logger.info(f"Successfully deleted sound '{sound_label}' ({len(indices_to_remove)} embeddings removed)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete sound '{sound_label}': {e}")
            return False

    def get_stats(self) -> Dict:
        """Get recognizer statistics."""
        custom_sounds = [label for label in self.labels if not label.startswith('esc50_')]
        esc50_sounds = [label for label in self.labels if label.startswith('esc50_')]
        trained_sounds = list(set(custom_sounds))  # Unique custom sound names
        
        return {
            'total_embeddings': len(self.embeddings),
            'custom_sounds': len(set(custom_sounds)),
            'trained_sounds': {sound: self.labels.count(sound) for sound in trained_sounds},
            'esc50_samples': len(esc50_sounds),
            'mappings': len(self.mappings),
            'sound_mappings': self.mappings.copy(),
            'model_ready': len(self.embeddings) > 0
        }
    
    def on_confidence_threshold_updated(self, threshold: float) -> None:
        """
        Called by SettingsUpdateCoordinator when confidence threshold is updated.
        Config is already updated - this updates the recognizer's instance variable.
        """
        old_threshold = self.confidence_threshold
        self.confidence_threshold = threshold
        logger.info(f"Sound recognizer confidence threshold updated: {old_threshold:.3f} -> {threshold:.3f}")
    
    def on_vote_threshold_updated(self, threshold: float) -> None:
        """
        Called by SettingsUpdateCoordinator when vote threshold is updated.
        Config is already updated - this updates the recognizer's instance variable.
        """
        old_threshold = self.vote_threshold
        self.vote_threshold = threshold
        logger.info(f"Sound recognizer vote threshold updated: {old_threshold:.3f} -> {threshold:.3f}")
    
    async def shutdown(self) -> None:
        """Shutdown sound recognizer and cleanup TensorFlow resources"""
        try:
            logger.info("Shutting down StreamlinedSoundRecognizer")
            
            # Clear TensorFlow model and free GPU/CPU memory
            if hasattr(self, 'yamnet_model') and self.yamnet_model is not None:
                # Force deletion of the model
                del self.yamnet_model
                self.yamnet_model = None
                logger.info("YAMNet model deleted")
            
            # Clear TensorFlow session and backend - CRITICAL for memory release
            if tf is not None:
                try:
                    # Clear all TensorFlow graphs and sessions
                    tf.keras.backend.clear_session()
                    
                    # Reset default graph (releases graph memory)
                    if hasattr(tf, 'compat') and hasattr(tf.compat, 'v1'):
                        tf.compat.v1.reset_default_graph()
                    
                    logger.info("TensorFlow session and graphs cleared")
                except Exception as e:
                    logger.warning(f"Error clearing TensorFlow session: {e}")
            
            # Clear numpy arrays to free memory
            if hasattr(self, 'embeddings') and self.embeddings is not None:
                del self.embeddings
                self.embeddings = None
            
            if hasattr(self, 'labels') and self.labels is not None:
                self.labels.clear()
                self.labels = None
            
            # Clear scaler
            if hasattr(self, 'scaler') and self.scaler is not None:
                del self.scaler
                self.scaler = None
            
            # Clear mappings
            if hasattr(self, 'mappings') and self.mappings is not None:
                self.mappings.clear()
                self.mappings = None
            
            # Force garbage collection
            gc.collect()
            
            logger.info("StreamlinedSoundRecognizer shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during StreamlinedSoundRecognizer shutdown: {e}", exc_info=True)