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

from iris.config.app_config import GlobalAppConfig, AudioConfig, VADConfig
from iris.config.stt_config import STTConfig
from iris.config.dictation_config import DictationConfig
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


@pytest.fixture
def mock_audio_config():
    """Mock audio configuration for testing."""
    return AudioConfig(
        sample_rate=16000,
        chunk_size=320,
        command_chunk_size=960,
        channels=1,
        dtype="int16",
        device=None,
        enable_dual_mode_processing=True
    )


@pytest.fixture
def mock_vad_config():
    """Mock VAD configuration for testing."""
    return VADConfig(
        command_energy_threshold=0.003,
        dictation_energy_threshold=0.002,
        command_silence_timeout=0.4,
        dictation_silence_timeout=1.5,
        command_max_recording_duration=5.0,
        dictation_max_recording_duration=30.0,
        command_pre_roll_buffers=3,
        dictation_pre_roll_buffers=5
    )


@pytest.fixture
def mock_stt_config():
    """Mock STT configuration for testing."""
    return STTConfig(
        default_engine="vosk",
        model_path="test/model/path",
        whisper_model="base",
        whisper_device="cpu",
        sample_rate=16000,
        command_debounce_interval=0.02,
        dictation_debounce_interval=0.1
    )


@pytest.fixture
def mock_dictation_config():
    """Mock dictation configuration for testing."""
    return DictationConfig(
        start_trigger="green",
        stop_trigger="amber",
        type_trigger="type",
        smart_start_trigger="smart green",
        dictation_stt_engine="whisper",
        command_stt_engine="vosk"
    )


@pytest.fixture
def mock_global_config(mock_audio_config, mock_vad_config, mock_stt_config, mock_dictation_config):
    """Mock global app configuration for testing."""
    config = Mock()
    config.audio = mock_audio_config
    config.vad = mock_vad_config
    config.stt = mock_stt_config
    config.dictation = mock_dictation_config
    config.model_paths = Mock()
    config.model_paths.vosk_model = "test/vosk/model"
    return config


@pytest.fixture
def mock_event_bus():
    """Mock event bus for testing."""
    return Mock()


@pytest.fixture
def test_audio_bytes():
    """Generate test audio bytes for testing."""
    sample_rate = 16000
    duration = 1.0  # 1 second
    samples = int(sample_rate * duration)
    
    # Generate sine wave test audio
    t = np.linspace(0, duration, samples)
    frequency = 440  # A4 note
    audio_data = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
    
    return audio_data.tobytes()


@pytest.fixture
def quiet_audio_bytes():
    """Generate quiet audio bytes for testing silence detection."""
    sample_rate = 16000
    duration = 0.5
    samples = int(sample_rate * duration)
    
    # Generate low-amplitude noise
    audio_data = (np.random.normal(0, 0.001, samples) * 32767).astype(np.int16)
    
    return audio_data.tobytes()


@pytest.fixture
def command_test_phrases():
    """Common command phrases for testing."""
    return [
        "click",
        "enter",
        "right click", 
        "down",
        "ctrl c",
        "mark a",
        "go to b",
        "1",
        "25",
        "golf"
    ]


@pytest.fixture
def mock_vosk_model():
    """Mock Vosk model for testing."""
    model = Mock()
    recognizer = Mock()
    
    # Mock recognition results
    def mock_accept_waveform(audio_bytes):
        return True
    
    def mock_final_result():
        return '{"text": "test command"}'
    
    def mock_result():
        return '{"text": "test command"}'
    
    def mock_partial_result():
        return '{"partial": "test"}'
    
    recognizer.AcceptWaveform = mock_accept_waveform
    recognizer.FinalResult = mock_final_result
    recognizer.Result = mock_result
    recognizer.PartialResult = mock_partial_result
    recognizer.Reset = Mock()
    
    return model, recognizer


@pytest.fixture
def mock_whisper_model():
    """Mock Whisper model for testing."""
    model = Mock()
    
    # Mock segment for transcription results
    mock_segment = Mock()
    mock_segment.text = "test dictation text"
    mock_segment.avg_logprob = -0.5
    
    # Mock transcribe method
    def mock_transcribe(audio, **kwargs):
        segments = [mock_segment]
        info = Mock()
        return segments, info
    
    model.transcribe = mock_transcribe
    
    return model