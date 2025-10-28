import os

# Add vocalance to path for imports
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest
import pytest_asyncio
import soundfile as sf

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.event_bus import EventBus
from vocalance.app.services.audio.stt.stt_service import SpeechToTextService
from vocalance.app.services.audio.stt.vosk_stt import VoskSTT
from vocalance.app.services.audio.stt.whisper_stt import WhisperSTT

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_rate():
    """Standard sample rate for audio processing."""
    return 16000


@pytest.fixture
def audio_samples():
    """Load and provide audio samples from test assets."""
    assets_path = Path(__file__).parent / "assets" / "sound_recognizer"

    samples = {"lip_popping": [], "tongue_clicking": [], "noise": []}

    # Load target sound samples
    for wav_file in sorted(assets_path.glob("*.wav")):
        audio, sr = sf.read(wav_file)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)

        filename = wav_file.name.lower()
        if "lip_popping" in filename or "lip_pop" in filename:
            samples["lip_popping"].append((audio, sr, wav_file.name))
        elif "tongue_clicking" in filename or "tongue_click" in filename:
            samples["tongue_clicking"].append((audio, sr, wav_file.name))

    # Load noise samples
    noise_path = assets_path / "noise"
    if noise_path.exists():
        for wav_file in sorted(noise_path.glob("*.wav")):
            audio, sr = sf.read(wav_file)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=-1)
            samples["noise"].append((audio, sr, wav_file.name))

    return samples


@pytest.fixture
def user_prompt_sample(audio_samples):
    """Extract the user prompt sample specifically."""
    for audio, sr, name in audio_samples["lip_popping"]:
        if "user_prompt" in name.lower():
            return audio, sr, name
    pytest.fail("User prompt sample not found in audio samples")


@pytest.fixture
def training_samples(audio_samples):
    """Provide training samples (excluding user prompt)."""
    training = {"lip_popping": [], "tongue_clicking": []}

    # Get lip_popping samples (excluding user prompt)
    for audio, sr, name in audio_samples["lip_popping"]:
        if "user_prompt" not in name.lower() and len(training["lip_popping"]) < 3:
            training["lip_popping"].append((audio, sr))

    # Get tongue_clicking samples
    for audio, sr, name in audio_samples["tongue_clicking"][:3]:
        training["tongue_clicking"].append((audio, sr))

    return training


@pytest.fixture
def mock_yamnet_model():
    """Mock YAMNet model that returns consistent embeddings."""
    mock_model = Mock()

    # Predefined embeddings for different sound types
    # Use distinct random seeds for each sound type to create differentiation
    base_embeddings = {}
    for sound_type, seed in [("lip_popping", 42), ("tongue_clicking", 123), ("noise", 456), ("default", 789)]:
        np.random.seed(seed)
        embedding = np.random.normal(0, 1, 1024)
        base_embeddings[sound_type] = embedding / np.linalg.norm(embedding)

    def mock_yamnet_call(audio_tensor):
        # Return consistent embeddings based on audio characteristics
        audio_np = audio_tensor.numpy() if hasattr(audio_tensor, "numpy") else audio_tensor

        # Ensure we have a numpy array
        if not isinstance(audio_np, np.ndarray):
            audio_np = np.array(audio_np)

        # Use audio spectral features to differentiate
        # Lip-popping: lower frequency, burst-like
        # Tongue-clicking: higher frequency, sharper transients

        rms = np.sqrt(np.mean(audio_np**2))

        # Spectral analysis
        if len(audio_np) > 0:
            fft_result = np.fft.fft(audio_np)
            power_spectrum = np.abs(fft_result[: len(fft_result) // 2])

            # Spectral centroid (center of mass of spectrum)
            freqs = np.fft.fftfreq(len(audio_np), 1 / 16000)[: len(audio_np) // 2]
            spectral_centroid = np.sum(freqs * power_spectrum) / (np.sum(power_spectrum) + 1e-10)

            # Zero-crossing rate (rapid signal changes)
            zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_np)))) / len(audio_np)
        else:
            spectral_centroid = 0
            zero_crossings = 0

        # Classification logic:
        # Tongue clicks tend to have higher spectral centroid and more zero crossings
        # Lip pops tend to have lower spectral centroid and fewer zero crossings
        if rms < 0.005:
            sound_type = "noise"
        elif spectral_centroid > 1500 or zero_crossings > 0.15:
            sound_type = "tongue_clicking"
        elif spectral_centroid < 1500 and rms > 0.01:
            sound_type = "lip_popping"
        else:
            sound_type = "default"

        # Return base embedding with small consistent variation for same sound type
        # Use hash of audio for consistent variation per sample
        audio_hash = hash(tuple(audio_np[: min(100, len(audio_np))].tobytes())) % 1000
        np.random.seed(audio_hash)
        variation = np.random.normal(0, 0.02, 1024)

        embedding = base_embeddings[sound_type] + variation
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
    config.sound_recognizer.silence_threshold = 0.005
    config.sound_recognizer.min_sound_duration = 0.1
    config.sound_recognizer.max_sound_duration = 2.0
    config.sound_recognizer.frame_length = 1024
    config.sound_recognizer.hop_length = 512
    config.sound_recognizer.normalization_level = 0.7
    config.sound_recognizer.esc50_categories = {
        "breathing": "breathing",
        "coughing": "coughing",
        "brushing_teeth": "brushing_teeth",
    }
    config.sound_recognizer.max_esc50_samples_per_category = 15
    config.sound_recognizer.max_total_esc50_samples = 40

    # Add asset paths pointing to real assets for integration tests
    config.asset_paths = Mock()
    # Determine the project root (3 levels up from this file: tests/conftest.py -> vocalance/)
    project_root = Path(__file__).parent.parent
    assets_root = project_root / "vocalance" / "app" / "assets"
    config.asset_paths.yamnet_model_path = str(assets_root / "sound_processing" / "yamnet")
    config.asset_paths.esc50_samples_path = str(assets_root / "sound_processing" / "esc50")

    return config


@pytest.fixture
def mock_storage_factory():
    """Mock storage service for sound recognizer tests."""
    from vocalance.app.services.storage.storage_models import SoundMappingsData

    storage = Mock()

    # Create temporary directories for testing
    temp_dir = tempfile.mkdtemp()

    # Mock storage_config attribute with proper string paths
    storage.storage_config = Mock()
    storage.storage_config.sound_model_dir = os.path.join(temp_dir, "model")
    storage.storage_config.external_non_target_sounds_dir = os.path.join(temp_dir, "external_sounds")

    # Ensure directories exist
    os.makedirs(storage.storage_config.sound_model_dir, exist_ok=True)
    os.makedirs(storage.storage_config.external_non_target_sounds_dir, exist_ok=True)

    # Mock async read/write methods
    async def mock_read(model_type):
        if model_type == SoundMappingsData:
            return SoundMappingsData(sound_to_command={})
        return model_type()

    storage.read = AsyncMock(side_effect=mock_read)
    storage.write = AsyncMock(return_value=True)

    return storage


@pytest.fixture
def isolated_recognizer(mock_config, mock_storage_factory, mock_yamnet_model, monkeypatch):
    """Create an isolated recognizer instance for testing."""
    # Create temporary directory for YAMNet model
    temp_yamnet_dir = tempfile.mkdtemp()

    # Mock asset paths
    mock_config.asset_paths = Mock()
    mock_config.asset_paths.yamnet_model_path = os.path.join(temp_yamnet_dir, "yamnet")
    mock_config.asset_paths.esc50_samples_path = os.path.join(temp_yamnet_dir, "esc50")
    os.makedirs(mock_config.asset_paths.yamnet_model_path, exist_ok=True)
    os.makedirs(mock_config.asset_paths.esc50_samples_path, exist_ok=True)

    # Create proper YAMNet model structure to pass validation
    with open(os.path.join(mock_config.asset_paths.yamnet_model_path, "saved_model.pb"), "w") as f:
        f.write("fake model")

    # Create variables directory with required files
    variables_dir = os.path.join(mock_config.asset_paths.yamnet_model_path, "variables")
    os.makedirs(variables_dir, exist_ok=True)
    with open(os.path.join(variables_dir, "variables.data-00000-of-00001"), "w") as f:
        f.write("fake variables data")
    with open(os.path.join(variables_dir, "variables.index"), "w") as f:
        f.write("fake variables index")

    # Mock TensorFlow import at the module level where it's imported
    tf_mock = Mock()

    # Create a proper tensor mock that carries the audio data
    class TensorMock:
        def __init__(self, data):
            self._data = np.array(data) if not isinstance(data, np.ndarray) else data

        def numpy(self):
            return self._data

    tf_mock.convert_to_tensor = lambda x, dtype=None: TensorMock(x)

    def reduce_mean_mock(x, axis=None):
        data = x.numpy() if hasattr(x, "numpy") else x
        result = np.mean(data, axis=axis)
        return TensorMock(result)

    tf_mock.reduce_mean = reduce_mean_mock
    tf_mock.saved_model = Mock()
    tf_mock.saved_model.load.return_value = mock_yamnet_model

    # Mock tensorflow module in sys.modules
    import sys

    monkeypatch.setitem(sys.modules, "tensorflow", tf_mock)
    monkeypatch.setattr("vocalance.app.services.audio.sound_recognizer.streamlined_sound_recognizer.tf", tf_mock)

    # Import after mocking
    from vocalance.app.services.audio.sound_recognizer.streamlined_sound_recognizer import SoundRecognizer

    recognizer = SoundRecognizer(config=mock_config, storage=mock_storage_factory)
    recognizer.yamnet_model = mock_yamnet_model

    return recognizer


@pytest.fixture
def vosk_model_path():
    """Get the path to the Vosk model."""
    return "vocalance/app/assets/vosk-model-small-en-us-0.15"


@pytest.fixture
def stt_config():
    """Create GlobalAppConfig for testing."""
    return GlobalAppConfig()


@pytest.fixture
def vosk_stt(vosk_model_path, sample_rate, stt_config):
    """Initialize Vosk STT instance."""
    return VoskSTT(model_path=vosk_model_path, sample_rate=sample_rate, config=stt_config)


@pytest.fixture
def audio_samples_path():
    """Get path to audio samples directory."""
    return Path(__file__).parent / "assets" / "audio_processing" / "stt_models"


@pytest.fixture
def vosk_test_files(audio_samples_path):
    """Get list of test files for Vosk (excludes dictation file)."""
    all_files = list(audio_samples_path.glob("*.bytes"))
    return [f for f in all_files if f.name != "this_is_a_test_of_the_dictation_capabilities.bytes"]


@pytest.fixture
def whisper_stt(sample_rate, stt_config):
    """Initialize Whisper STT instance."""
    return WhisperSTT(model_name="base", device="cpu", sample_rate=sample_rate, config=stt_config)


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
    await service.initialize_engines()
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
def mock_storage_service():
    """Mock unified storage service for testing."""
    from vocalance.app.services.storage.storage_models import CommandHistoryData, CommandsData, MarksData

    storage = Mock()

    # Mock the read method to return appropriate data models
    async def mock_read(model_type):
        if model_type == MarksData:
            return MarksData(marks={})
        elif model_type == CommandsData:
            return CommandsData(custom_commands={}, phrase_overrides={})
        elif model_type == CommandHistoryData:
            return CommandHistoryData(history=[])
        return None

    storage.read = AsyncMock(side_effect=mock_read)
    storage.write = AsyncMock(return_value=True)

    return storage


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
def vosk_stt_instance(mock_vosk_model, mock_vosk_recognizer, stt_config):
    """Create Vosk STT instance with mocked dependencies."""
    with patch("vocalance.app.services.audio.stt.vosk_stt.vosk.Model", return_value=mock_vosk_model), patch(
        "vocalance.app.services.audio.stt.vosk_stt.vosk.KaldiRecognizer", return_value=mock_vosk_recognizer
    ):
        from vocalance.app.services.audio.stt.vosk_stt import VoskSTT

        instance = VoskSTT(model_path="fake_model_path", sample_rate=16000, config=stt_config)
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
def whisper_stt_instance(mock_whisper_model, stt_config):
    """Create Whisper STT instance with mocked dependencies."""
    with patch("vocalance.app.services.audio.stt.whisper_stt.WhisperModel", return_value=mock_whisper_model):
        from vocalance.app.services.audio.stt.whisper_stt import WhisperSTT

        instance = WhisperSTT(model_name="base", device="cpu", sample_rate=16000, config=stt_config)
        instance._model = mock_whisper_model
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
    recognizer.set_mapping = AsyncMock(return_value=True)
    recognizer.get_mapping = Mock(return_value=None)
    recognizer.get_stats = Mock(
        return_value={
            "service_initialized": False,
            "training_active": False,
            "current_training_label": None,
            "training_samples_collected": 0,
        }
    )
    return recognizer


@pytest.fixture
def preprocessor(mock_config):
    """Create a standard AudioPreprocessor instance."""
    from vocalance.app.services.audio.sound_recognizer.streamlined_sound_recognizer import AudioPreprocessor

    return AudioPreprocessor(config=mock_config.sound_recognizer)


@pytest.fixture
def mock_protected_terms_validator():
    """Mock protected terms validator for testing."""
    validator = Mock()
    validator.validate_term = AsyncMock(return_value=(True, None))
    validator.is_term_protected = AsyncMock(return_value=False)
    validator.get_all_protected_terms = AsyncMock(return_value={"start dictation", "stop dictation", "show grid"})
    return validator


@pytest.fixture
def mock_action_map_provider():
    """Mock CommandActionMapProvider for testing."""
    from vocalance.app.config.command_types import AutomationCommand

    provider = Mock()

    async def mock_get_action_map():
        return {
            "copy": AutomationCommand(
                command_key="copy",
                action_type="hotkey",
                action_value="ctrl+c",
                is_custom=False,
                short_description="Copy text",
                long_description="Copy selected text to clipboard",
            ),
            "paste": AutomationCommand(
                command_key="paste",
                action_type="hotkey",
                action_value="ctrl+v",
                is_custom=False,
                short_description="Paste text",
                long_description="Paste clipboard contents",
            ),
            "scroll up": AutomationCommand(
                command_key="scroll up",
                action_type="scroll",
                action_value="up",
                is_custom=False,
                short_description="Scroll up",
                long_description="Scroll up",
            ),
        }

    provider.get_action_map = AsyncMock(side_effect=mock_get_action_map)
    return provider


@pytest.fixture
def mock_command_history_manager():
    """Mock CommandHistoryManager for testing."""
    manager = Mock()
    manager.initialize = AsyncMock(return_value=True)
    manager.record_command = AsyncMock()
    manager.get_recent_history = AsyncMock(return_value=[])
    manager.get_full_history = AsyncMock(return_value=[])
    manager.shutdown = AsyncMock(return_value=True)
    return manager
