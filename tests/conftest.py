"""
Test fixtures for sound recognition tests.

Provides isolated fixtures for testing sound recognition components
without dependencies on external storage or persistent state.
"""
import pytest
import numpy as np
import soundfile as sf
from pathlib import Path
from unittest.mock import Mock, MagicMock
from typing import Dict, List, Tuple
import tempfile
import os

# Add iris to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from iris.config.app_config import GlobalAppConfig
from iris.event_bus import EventBus
from iris.services.storage.unified_storage_service import UnifiedStorageService
from iris.services.storage.storage_adapters import StorageAdapterFactory


@pytest.fixture
def sample_rate():
    """Standard sample rate for audio processing."""
    return 16000


@pytest.fixture
def audio_samples():
    """Load and provide audio samples from test assets."""
    assets_path = Path(__file__).parent / "assets" / "sound_recognizer"
    
    samples = {
        'lip_popping': [],
        'tongue_clicking': [],
        'noise': []
    }
    
    # Load target sound samples
    for wav_file in sorted(assets_path.glob("*.wav")):
        try:
            audio, sr = sf.read(wav_file)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=-1)
            
            filename = wav_file.name.lower()
            if 'lip_popping' in filename or 'lip_pop' in filename:
                samples['lip_popping'].append((audio, sr, wav_file.name))
            elif 'tongue_clicking' in filename or 'tongue_click' in filename:
                samples['tongue_clicking'].append((audio, sr, wav_file.name))
        except Exception as e:
            print(f"Failed to load {wav_file}: {e}")
    
    # Load noise samples
    noise_path = assets_path / "noise"
    if noise_path.exists():
        for wav_file in sorted(noise_path.glob("*.wav")):
            try:
                audio, sr = sf.read(wav_file)
                if audio.ndim > 1:
                    audio = np.mean(audio, axis=-1)
                samples['noise'].append((audio, sr, wav_file.name))
            except Exception as e:
                print(f"Failed to load noise sample {wav_file}: {e}")
    
    return samples


@pytest.fixture
def user_prompt_sample(audio_samples):
    """Extract the user prompt sample specifically."""
    for audio, sr, name in audio_samples['lip_popping']:
        if 'user_prompt' in name.lower():
            return audio, sr, name
    pytest.fail("User prompt sample not found in audio samples")


@pytest.fixture
def training_samples(audio_samples):
    """Provide training samples (excluding user prompt)."""
    training = {
        'lip_popping': [],
        'tongue_clicking': []
    }
    
    # Get lip_popping samples (excluding user prompt)
    for audio, sr, name in audio_samples['lip_popping']:
        if 'user_prompt' not in name.lower() and len(training['lip_popping']) < 3:
            training['lip_popping'].append((audio, sr))
    
    # Get tongue_clicking samples
    for audio, sr, name in audio_samples['tongue_clicking'][:3]:
        training['tongue_clicking'].append((audio, sr))
    
    return training


@pytest.fixture
def mock_yamnet_model():
    """Mock YAMNet model that returns consistent embeddings."""
    mock_model = Mock()
    
    def mock_yamnet_call(audio_tensor):
        # Return consistent embeddings based on audio characteristics
        # This simulates YAMNet behavior for testing
        audio_np = audio_tensor.numpy() if hasattr(audio_tensor, 'numpy') else audio_tensor
        
        # Create deterministic embeddings based on audio content
        # Use audio statistics to create different embeddings for different sounds
        rms = np.sqrt(np.mean(audio_np**2))
        spectral_centroid = np.mean(np.abs(np.fft.fft(audio_np)[:len(audio_np)//2]))
        
        # Create a 1024-dimensional embedding (YAMNet size)
        embedding = np.random.RandomState(int(rms * 1000 + spectral_centroid)).normal(0, 1, (1, 1024))
        
        return None, embedding, None
    
    mock_model.side_effect = mock_yamnet_call
    return mock_model


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    config = Mock()
    config.sound_recognizer = Mock()
    config.sound_recognizer.target_sample_rate = 16000
    config.sound_recognizer.confidence_threshold = 0.15
    config.sound_recognizer.k_neighbors = 7
    config.sound_recognizer.vote_threshold = 0.35
    config.sound_recognizer.esc50_categories = {
        "breathing": "breathing",
        "coughing": "coughing",
        "brushing_teeth": "brushing_teeth"
    }
    config.sound_recognizer.max_esc50_samples_per_category = 15
    config.sound_recognizer.max_total_esc50_samples = 40
    return config


@pytest.fixture
def mock_storage_factory():
    """Mock storage factory that doesn't persist data."""
    factory = Mock()
    adapter = Mock()
    
    # Create temporary directories for testing
    temp_dir = tempfile.mkdtemp()
    model_path = os.path.join(temp_dir, "model")
    external_sounds_path = os.path.join(temp_dir, "external_sounds")
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(external_sounds_path, exist_ok=True)
    
    adapter.get_model_path.return_value = model_path
    adapter.get_external_sounds_path.return_value = external_sounds_path
    factory.create_sound_recognizer_adapter.return_value = adapter
    
    return factory


@pytest.fixture
def sample_embeddings():
    """Provide sample embeddings for testing."""
    np.random.seed(42)  # For reproducible results
    
    # Create distinct embeddings for different sound types
    lip_popping_embeddings = np.random.normal(0, 1, (3, 1024))
    tongue_clicking_embeddings = np.random.normal(2, 1, (3, 1024))  # Different distribution
    noise_embeddings = np.random.normal(-1, 0.5, (5, 1024))  # Another different distribution
    
    return {
        'lip_popping': lip_popping_embeddings,
        'tongue_clicking': tongue_clicking_embeddings,
        'noise': noise_embeddings
    }


@pytest.fixture
def mock_tensorflow():
    """Mock TensorFlow for testing without actual model loading."""
    tf_mock = Mock()
    tf_mock.convert_to_tensor = lambda x, dtype=None: Mock(numpy=lambda: x)
    tf_mock.reduce_mean = lambda x, axis=None: Mock(numpy=lambda: np.mean(x.numpy() if hasattr(x, 'numpy') else x, axis=axis))
    tf_mock.saved_model = Mock()
    tf_mock.saved_model.load = Mock()
    return tf_mock


@pytest.fixture
def isolated_recognizer(mock_config, mock_storage_factory, mock_yamnet_model, monkeypatch):
    """Create an isolated recognizer instance for testing."""
    # Mock TensorFlow import at the module level where it's imported
    tf_mock = Mock()
    tf_mock.convert_to_tensor = lambda x, dtype=None: Mock(numpy=lambda: x)
    tf_mock.reduce_mean = lambda x, axis=None: Mock(numpy=lambda: np.mean(x.numpy() if hasattr(x, 'numpy') else x, axis=axis))
    tf_mock.saved_model = Mock()
    tf_mock.saved_model.load.return_value = mock_yamnet_model
    
    # Mock tensorflow module in sys.modules
    import sys
    monkeypatch.setitem(sys.modules, "tensorflow", tf_mock)
    monkeypatch.setattr("iris.services.audio.sound_recognizer.streamlined_sound_recognizer.tf", tf_mock)
    
    # Import after mocking
    from iris.services.audio.sound_recognizer.streamlined_sound_recognizer import StreamlinedSoundRecognizer
    
    recognizer = StreamlinedSoundRecognizer(mock_config, mock_storage_factory)
    recognizer.yamnet_model = mock_yamnet_model
    
    return recognizer