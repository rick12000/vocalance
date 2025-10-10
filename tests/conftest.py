"""
Test fixtures for sound recognition and audio flow tests.

Provides isolated fixtures for testing sound recognition components
and audio flow integration without dependencies on external storage or persistent state.
"""
import pytest
import pytest_asyncio
import numpy as np
import soundfile as sf
from pathlib import Path
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from typing import Dict, List, Tuple
import tempfile
import os

# Add iris to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from iris.app.config.app_config import GlobalAppConfig
from iris.app.config.stt_config import STTConfig
from iris.app.config.dictation_config import DictationConfig
from iris.app.event_bus import EventBus
from iris.app.services.storage.unified_storage_service import UnifiedStorageService
from iris.app.services.storage.storage_adapters import StorageAdapterFactory
from iris.app.services.audio.vosk_stt import EnhancedVoskSTT
from iris.app.services.audio.whisper_stt import WhisperSpeechToText
from iris.app.services.audio.stt_service import SpeechToTextService


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

    # Predefined embeddings for different sound types
    np.random.seed(42)  # Fixed seed for reproducibility
    base_embeddings = {
        'lip_popping': np.random.normal(0, 1, 1024),
        'tongue_clicking': np.random.normal(100, 1, 1024),  # Different seed for different sound
        'noise': np.random.normal(200, 1, 1024),
        'default': np.random.normal(300, 1, 1024)
    }

    # Normalize base embeddings
    for key in base_embeddings:
        base_embeddings[key] = base_embeddings[key] / np.linalg.norm(base_embeddings[key])

    def mock_yamnet_call(audio_tensor):
        # Return consistent embeddings based on audio characteristics
        audio_np = audio_tensor.numpy() if hasattr(audio_tensor, 'numpy') else audio_tensor

        # Simple heuristic to classify audio type based on RMS and duration
        rms = np.sqrt(np.mean(audio_np**2))
        duration = len(audio_np) / 16000  # Assume 16kHz

        # Classify based on simple heuristics
        if 0.05 < duration < 0.3 and 0.01 < rms < 0.5:
            # Use spectral_centroid as a simple differentiator
            spectral_centroid = np.mean(np.abs(np.fft.fft(audio_np)[:len(audio_np)//4]))
            
            # Different sounds have different spectral characteristics
            if spectral_centroid > 1000:  # Arbitrary threshold for differentiation
                sound_type = 'tongue_clicking'
            else:
                sound_type = 'lip_popping'
        elif rms < 0.01:
            sound_type = 'noise'  # Very quiet = noise
        else:
            sound_type = 'default'

        # Return base embedding with tiny variation for same sound type
        embedding = base_embeddings[sound_type] + np.random.normal(0, 0.01, 1024)
        embedding = embedding / np.linalg.norm(embedding)

        return None, embedding.reshape(1, -1), None

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
    from unittest.mock import AsyncMock
    
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
    
    # Mock async methods properly
    adapter.load_sound_mappings = AsyncMock(return_value={})
    adapter.save_sound_mappings = AsyncMock(return_value=True)
    
    factory.create_sound_recognizer_adapter.return_value = adapter
    
    return factory




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
    monkeypatch.setattr("iris.app.services.audio.sound_recognizer.streamlined_sound_recognizer.tf", tf_mock)
    
    # Import after mocking
    from iris.app.services.audio.sound_recognizer.streamlined_sound_recognizer import StreamlinedSoundRecognizer
    
    recognizer = StreamlinedSoundRecognizer(mock_config, mock_storage_factory)
    recognizer.yamnet_model = mock_yamnet_model
    
    return recognizer


@pytest.fixture
def vosk_model_path():
    """Get the path to the Vosk model."""
    return "iris/app/assets/vosk-model-small-en-us-0.15"


@pytest.fixture
def stt_config():
    """Create GlobalAppConfig for testing."""
    return GlobalAppConfig()


@pytest.fixture
def vosk_stt(vosk_model_path, sample_rate, stt_config):
    """Initialize Vosk STT instance."""
    return EnhancedVoskSTT(
        model_path=vosk_model_path,
        sample_rate=sample_rate,
        config=stt_config
    )


@pytest.fixture
def audio_samples_path():
    """Get path to audio samples directory."""
    return Path(__file__).parent / "assets" / "audio_processing" / "stt_models"


@pytest.fixture
def vosk_test_files(audio_samples_path):
    """Get list of test files for Vosk (excludes dictation file)."""
    all_files = list(audio_samples_path.glob("*.bytes"))
    return [
        f for f in all_files
        if f.name != "this_is_a_test_of_the_dictation_capabilities.bytes"
    ]


@pytest.fixture
def whisper_stt(sample_rate, stt_config):
    """Initialize Whisper STT instance."""
    return WhisperSpeechToText(
        model_name="base",
        device="cpu",
        sample_rate=sample_rate,
        config=stt_config
    )


@pytest.fixture
def dictation_file(audio_samples_path):
    """Get the dictation test file."""
    return audio_samples_path / "this_is_a_test_of_the_dictation_capabilities.bytes"


@pytest.fixture
def audio_flow_samples_path():
    """Get path to audio flow test samples."""
    return Path(__file__).parent / "assets" / "audio_processing" / "audio_flow"


@pytest.fixture
def audio_flow_samples(audio_flow_samples_path):
    """
    Load audio flow test samples (wav files).

    Returns dict mapping expected text to (audio_data, sample_rate, filename) tuples.
    """
    samples = {}

    for wav_file in sorted(audio_flow_samples_path.glob("*.wav")):
        try:
            audio_data, sample_rate = sf.read(wav_file)

            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=-1)

            expected_text = wav_file.stem.replace("_", " ").lower().strip()
            samples[expected_text] = (audio_data, sample_rate, wav_file.name)

        except Exception as e:
            pytest.fail(f"Failed to load audio flow sample {wav_file}: {e}")

    return samples


@pytest.fixture
def app_config():
    """Create application configuration for testing."""
    return GlobalAppConfig()


@pytest.fixture
def event_bus():
    """Create event bus for testing."""
    bus = EventBus()
    return bus


@pytest_asyncio.fixture
async def stt_service(event_bus, app_config):
    """Create and initialize STT service."""
    service = SpeechToTextService(event_bus, app_config)
    service.initialize_engines()
    service.setup_subscriptions()

    await event_bus.start_worker()

    yield service

    await event_bus.stop_worker()


@pytest.fixture
def command_audio_bytes():
    """Generate sample command audio bytes."""
    return np.random.randint(0, 256, size=16000, dtype=np.uint8).tobytes()


@pytest.fixture
def dictation_audio_bytes():
    """Generate sample dictation audio bytes."""
    return np.random.randint(0, 256, size=32000, dtype=np.uint8).tobytes()


@pytest.fixture
def mock_command_storage_adapter():
    """Mock command storage adapter for testing."""
    
    adapter = Mock()
    adapter.get_action_map = AsyncMock(return_value={
        "copy": Mock(
            action_type="hotkey",
            action_value="ctrl+c",
            is_custom=False,
            short_description="Copy text",
            long_description="Copy selected text to clipboard"
        ),
        "paste": Mock(
            action_type="hotkey",
            action_value="ctrl+v",
            is_custom=False,
            short_description="Paste text",
            long_description="Paste clipboard contents"
        ),
        "scroll up": Mock(
            action_type="scroll",
            action_value="scroll_up",
            is_custom=False,
            short_description="Scroll up",
            long_description="Scroll page upward"
        ),
    })
    adapter.get_custom_commands = AsyncMock(return_value={})
    adapter.get_phrase_overrides = AsyncMock(return_value={})
    return adapter


@pytest.fixture
def mock_storage_adapters(mock_command_storage_adapter):
    """Mock storage adapter factory for testing."""
    
    factory = Mock()
    
    mark_adapter = Mock()
    mark_adapter.get_marks = AsyncMock(return_value={})
    mark_adapter.get_all_mark_names = AsyncMock(return_value=[])
    mark_adapter.set_mark = AsyncMock(return_value=True)
    mark_adapter.save_mark = AsyncMock(return_value=True)
    mark_adapter.delete_mark = AsyncMock(return_value=True)
    mark_adapter.remove_mark = AsyncMock(return_value=True)
    mark_adapter.delete_all_marks = AsyncMock(return_value=True)
    mark_adapter.load_marks = AsyncMock(return_value={})
    
    factory.get_mark_adapter.return_value = mark_adapter
    factory.get_command_adapter.return_value = mock_command_storage_adapter

    return factory


@pytest.fixture
def mock_vosk_model():
    """Mock Vosk model."""
    return Mock()


@pytest.fixture
def mock_vosk_recognizer():
    """Mock Vosk recognizer."""
    import json
    recognizer = Mock()
    recognizer.Reset = Mock()
    recognizer.AcceptWaveform = Mock(return_value=True)
    recognizer.FinalResult = Mock(return_value=json.dumps({"text": "test"}))
    recognizer.Result = Mock(return_value=json.dumps({"text": "test"}))
    recognizer.PartialResult = Mock(return_value=json.dumps({"partial": "test"}))
    return recognizer


@pytest.fixture
def mock_duplicate_filter():
    """Mock duplicate filter."""
    filter_mock = Mock()
    filter_mock.is_duplicate = Mock(return_value=False)
    return filter_mock


@pytest.fixture
def vosk_stt_instance(mock_vosk_model, mock_vosk_recognizer, mock_duplicate_filter, stt_config):
    """Create Vosk STT instance with mocked dependencies."""
    with patch('iris.app.services.audio.vosk_stt.vosk.Model', return_value=mock_vosk_model), \
         patch('iris.app.services.audio.vosk_stt.vosk.KaldiRecognizer', return_value=mock_vosk_recognizer), \
         patch('iris.app.services.audio.vosk_stt.DuplicateTextFilter', return_value=mock_duplicate_filter):

        from iris.app.services.audio.vosk_stt import EnhancedVoskSTT
        instance = EnhancedVoskSTT(
            model_path="fake_model_path",
            sample_rate=16000,
            config=stt_config
        )
        instance._recognizer = mock_vosk_recognizer
        return instance


@pytest.fixture
def mock_whisper_model():
    """Mock faster-whisper model."""
    model = Mock()

    mock_segment = Mock()
    mock_segment.text = "test"
    mock_segment.avg_logprob = -0.5

    mock_info = Mock()

    model.transcribe = Mock(return_value=([mock_segment], mock_info))

    return model


@pytest.fixture
def whisper_stt_instance(mock_whisper_model, mock_duplicate_filter, stt_config):
    """Create Whisper STT instance with mocked dependencies."""
    with patch('iris.app.services.audio.whisper_stt.WhisperModel', return_value=mock_whisper_model), \
         patch('iris.app.services.audio.whisper_stt.DuplicateTextFilter', return_value=mock_duplicate_filter):

        from iris.app.services.audio.whisper_stt import WhisperSpeechToText
        instance = WhisperSpeechToText(
            model_name="base",
            device="cpu",
            sample_rate=16000,
            config=stt_config
        )
        instance._model = mock_whisper_model
        instance._duplicate_filter = mock_duplicate_filter
        return instance


@pytest.fixture
def mock_event_bus():
    """Create a mock event bus."""
    event_bus = Mock()
    event_bus.subscribe = Mock()
    event_bus.publish = AsyncMock()
    return event_bus


@pytest.fixture
def mock_recognizer():
    """Create a mock recognizer."""
    recognizer = Mock()
    recognizer.initialize = AsyncMock(return_value=True)
    recognizer.recognize_sound = Mock(return_value=None)
    recognizer.train_sound = AsyncMock(return_value=True)
    recognizer.set_mapping = Mock()
    recognizer.get_mapping = Mock(return_value=None)
    recognizer.get_stats = Mock(return_value={
        'service_initialized': False,
        'training_active': False,
        'current_training_label': None,
        'training_samples_collected': 0
    })
    return recognizer


@pytest.fixture
def preprocessor():
    """Create a standard AudioPreprocessor instance."""
    from iris.app.services.audio.sound_recognizer.streamlined_sound_recognizer import AudioPreprocessor
    return AudioPreprocessor(
        target_sr=16000,
        silence_threshold=0.005,
        min_sound_duration=0.1,
        max_sound_duration=2.0
    )