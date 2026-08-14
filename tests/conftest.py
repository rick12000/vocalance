import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest
import pytest_asyncio
import soundfile as sf

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.event_bus import EventBus
from vocalance.app.services.capture.vad import AdaptiveVADThreshold, AudioProcessor, SegmentConfig, UtteranceSegmenter
from vocalance.app.services.command_flow.speech_recognition.command_speech_service import CommandSpeechService
from vocalance.app.services.command_flow.speech_recognition.vosk_engine import VoskEngine

# import sys

# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def skip_if_headless() -> None:
    """Skip the calling test module when display or audio hardware is unavailable.

    Place immediately after `from conftest import skip_if_headless` at the top of
    every test module, before any production-code imports.  On a headless CI runner
    both pyautogui (DISPLAY) and sounddevice (PortAudio) fail at import time; calling
    this function once per module causes pytest to mark the entire module as skipped
    before any failing import is attempted.
    """
    try:
        import pyautogui  # noqa: F401
    except Exception:
        pytest.skip("requires display (pyautogui)", allow_module_level=True)
    try:
        import sounddevice  # noqa: F401
    except OSError:
        pytest.skip("requires audio hardware (sounddevice)", allow_module_level=True)
    try:
        from PySide6.QtGui import QFont  # noqa: F401
    except ImportError:
        pytest.skip("requires OpenGL libraries (PySide6.QtGui)", allow_module_level=True)


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
    config.sound_recognizer.max_training_samples = 1000

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
    monkeypatch.setattr("vocalance.app.services.command_flow.sound_recognition.sound_recognizer.tf", tf_mock)

    # Import after mocking
    from vocalance.app.services.command_flow.sound_recognition.sound_recognizer import SoundRecognizer

    recognizer = SoundRecognizer(config=mock_config, storage=mock_storage_factory)
    recognizer.yamnet_model = mock_yamnet_model

    return recognizer


@pytest.fixture
def recognizer_identity_query(isolated_recognizer, sample_rate):
    """Recognizer with an identity scaler and a deterministic query embedding.

    The identity scaler makes the scaled query equal the raw embedding, so cosine
    similarity against stored copies of the same embedding is exactly 1, allowing
    deterministic assertions on the k-NN voting logic.
    """
    t = np.linspace(0, 0.5, int(0.5 * sample_rate))
    audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    embedding = isolated_recognizer._extract_embedding(audio=audio, sr=sample_rate)
    dim = embedding.shape[0]
    isolated_recognizer.scaler.mean = np.zeros(dim, dtype=np.float32)
    isolated_recognizer.scaler.std = np.ones(dim, dtype=np.float32)
    isolated_recognizer.scaler._is_fitted = True
    return isolated_recognizer, audio, embedding


@pytest.fixture
def vosk_model_path():
    """Get the path to the Vosk model."""
    return "vocalance/app/assets/vosk-model-small-en-us-0.15"


@pytest.fixture
def stt_config():
    """Create GlobalAppConfig for testing."""
    return GlobalAppConfig()


@pytest.fixture
def vosk_engine(vosk_model_path, sample_rate, stt_config):
    """Initialize VoskEngine instance."""
    return VoskEngine(model_path=vosk_model_path, sample_rate=sample_rate, config=stt_config)


@pytest.fixture
def audio_samples_path():
    """Get path to audio samples directory."""
    return Path(__file__).parent / "assets" / "audio_processing" / "stt_models"


@pytest.fixture
def vosk_test_files(audio_samples_path):
    """Get list of test files for Vosk (excludes dictation file)."""
    all_files = list(audio_samples_path.glob("*.bytes"))
    return [f for f in all_files if f.name != "this_is_a_test_of_the_dictation_capabilities.bytes"]


@pytest.fixture(scope="module")
def xasr_engine(stt_config):
    """Real XASREngine backed by the bundled model assets (integration tests only)."""
    from vocalance.app.services.dictation_flow.speech_recognition.xasr_engine import XASREngine

    return XASREngine(config=stt_config)


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


@pytest_asyncio.fixture
async def event_bus():
    """Create a started event bus bound to the test asyncio loop."""
    bus = EventBus()
    loop = asyncio.get_running_loop()
    bus.start(loop)
    yield bus
    await bus.shutdown()


@pytest_asyncio.fixture
async def stt_service(event_bus, app_config):
    """Create and initialize STT service."""
    service = CommandSpeechService(event_bus, app_config)
    await service.initialize()
    yield service


@pytest_asyncio.fixture
async def command_speech_service(event_bus, app_config):
    """CommandSpeechService wired to a real bus with a mocked Vosk engine.

    Tests override ``service.vosk_engine.recognize`` to drive the recognized
    text for each routing scenario without loading the real Vosk model.
    """
    service = CommandSpeechService(event_bus, app_config)
    service.vosk_engine = Mock()
    service.vosk_engine.recognize = AsyncMock(return_value="copy")
    return service


@pytest.fixture
def command_audio_bytes():
    """Generate sample command audio bytes."""
    return np.random.randint(0, 256, size=16000, dtype=np.uint8).tobytes()


@pytest.fixture
def mock_storage_service():
    """Mock unified storage service for testing."""
    from vocalance.app.services.storage.storage_models import CommandsData, MarksData

    storage = Mock()
    _store: dict = {}

    async def mock_read(model_type):
        if model_type in _store:
            return _store[model_type]
        if model_type == MarksData:
            return MarksData(marks={})
        elif model_type == CommandsData:
            return CommandsData(custom_commands={}, phrase_overrides={})
        return None

    async def mock_write(data):
        _store[type(data)] = data
        return True

    storage.read = AsyncMock(side_effect=mock_read)
    storage.write = AsyncMock(side_effect=mock_write)

    return storage


@pytest.fixture
def isolated_storage_config():
    """Create isolated storage config that NEVER touches production data.

    SAFETY: This fixture ensures tests use temporary directories only.
    All storage paths are explicitly set to temp locations.
    """
    import tempfile
    from pathlib import Path

    from vocalance.app.config.app_config import GlobalAppConfig

    with tempfile.TemporaryDirectory() as temp_dir:
        config = GlobalAppConfig()
        temp_path = Path(temp_dir)

        # Override ALL storage paths - CRITICAL for data safety
        config.storage.user_data_root = temp_dir
        config.storage.marks_dir = str(temp_path / "marks")
        config.storage.settings_dir = str(temp_path / "settings")
        config.storage.click_tracker_dir = str(temp_path / "click_tracker")
        config.storage.sound_model_dir = str(temp_path / "sound_model")

        # Verify every storage path stays inside the temp directory
        for path_attr in [
            "user_data_root",
            "marks_dir",
            "settings_dir",
            "click_tracker_dir",
            "sound_model_dir",
        ]:
            path_value = Path(getattr(config.storage, path_attr)).resolve()
            if not path_value.is_relative_to(temp_path.resolve()):
                raise RuntimeError(f"SAFETY VIOLATION: Test config path {path_attr} escapes temp dir: {path_value}")

        yield config


@pytest.fixture
def storage_service(isolated_storage_config):
    """Real StorageService bound to an isolated temp-directory config."""
    from vocalance.app.services.storage.storage_service import StorageService

    return StorageService(config=isolated_storage_config)


@pytest_asyncio.fixture
async def runtime_config_store(event_bus, isolated_storage_config, storage_service):
    """RuntimeConfigurationStore backed by a real isolated StorageService."""
    from vocalance.app.services.storage.runtime_configuration import RuntimeConfigurationStore

    return RuntimeConfigurationStore(event_bus=event_bus, config=isolated_storage_config, storage=storage_service)


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
def vosk_engine_instance(mock_vosk_model, mock_vosk_recognizer, stt_config):
    """Create Vosk STT instance with mocked dependencies."""
    with patch(
        "vocalance.app.services.command_flow.speech_recognition.vosk_engine.vosk.Model", return_value=mock_vosk_model
    ), patch(
        "vocalance.app.services.command_flow.speech_recognition.vosk_engine.vosk.KaldiRecognizer",
        return_value=mock_vosk_recognizer,
    ):
        from vocalance.app.services.command_flow.speech_recognition.vosk_engine import VoskEngine

        instance = VoskEngine(model_path="fake_model_path", sample_rate=16000, config=stt_config)
        instance._recognizer = mock_vosk_recognizer
        return instance


@pytest.fixture
def xasr_engine_instance(stt_config):
    """XASREngine with sherpa_onnx.OnlineRecognizer.from_transducer mocked."""
    with patch("sherpa_onnx.OnlineRecognizer.from_transducer") as mock_factory:
        mock_factory.return_value = Mock()
        from vocalance.app.services.dictation_flow.speech_recognition.xasr_engine import XASREngine

        return XASREngine(config=stt_config)


@pytest.fixture
def tsm():
    from vocalance.app.services.dictation_flow.speech_recognition.transcript_state_manager import TranscriptStateManager

    return TranscriptStateManager(stability_window=2, provisional_words=2)


@pytest.fixture
def mock_xasr_recognizer():
    """Mock sherpa_onnx.OnlineRecognizer + stream pair for XASRStreamSession tests."""
    stream = Mock()
    stream.accept_waveform = Mock()
    stream.input_finished = Mock()

    recognizer = Mock()
    recognizer.create_stream.return_value = stream
    recognizer.is_ready.return_value = False
    recognizer.decode_stream = Mock()
    recognizer.get_result.return_value = ""
    return recognizer, stream


@pytest.fixture
def stream_loop():
    """Standalone event loop for stream-session tests (never run, closed on teardown)."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def xasr_stream_session(mock_xasr_recognizer, stream_loop):
    """XASRStreamSession wired to mocked recognizer and async callbacks."""
    from vocalance.app.config.app_config import XASRConfig
    from vocalance.app.services.dictation_flow.speech_recognition.xasr_engine import XASRStreamSession

    recognizer, _ = mock_xasr_recognizer
    return XASRStreamSession(
        recognizer=recognizer,
        loop=stream_loop,
        on_committed=AsyncMock(),
        on_provisional=AsyncMock(),
        xasr_config=XASRConfig(),
    )


@pytest.fixture
def mock_event_bus():
    """Create a mock event bus."""
    event_bus = Mock()
    event_bus.subscribe = Mock()
    event_bus.publish = AsyncMock()
    return event_bus


@pytest.fixture
def pause_state_manager(mock_event_bus):
    """PauseStateManager wired to a mock event bus for direct handler testing."""
    from vocalance.app.services.command_flow.pause_state_manager import PauseStateManager

    return PauseStateManager(event_bus=mock_event_bus)


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
def sound_service(mock_event_bus, mock_config, mock_storage_factory, mock_recognizer):
    """SoundService with its SoundRecognizer replaced by a mock recognizer."""
    from vocalance.app.services.command_flow.sound_recognition.sound_service import SoundService

    mock_config.asset_paths = Mock()
    mock_config.asset_paths.yamnet_model_path = "/fake/yamnet/path"

    with patch(
        "vocalance.app.services.command_flow.sound_recognition.sound_service.SoundRecognizer",
        return_value=mock_recognizer,
    ):
        return SoundService(mock_event_bus, mock_config, mock_storage_factory)


@pytest.fixture
def preprocessor(mock_config):
    """Create a standard AudioPreprocessor instance."""
    from vocalance.app.services.command_flow.sound_recognition.sound_recognizer import AudioPreprocessor

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
def audio_processor(sample_rate):
    """AudioProcessor with normalization disabled for predictable energies."""
    return AudioProcessor(sample_rate=sample_rate, enable_normalization=False)


@pytest.fixture
def vad_threshold():
    """AdaptiveVADThreshold with representative multipliers and bounds."""
    return AdaptiveVADThreshold(speech_multiplier=4.0, silence_multiplier=2.0, min_threshold=0.0003, max_threshold=0.1)


@pytest.fixture
def silence_chunk_bytes():
    """Quiet PCM chunk whose RMS sits below the silence threshold."""
    return np.tile([5, -5], 400).astype(np.int16).tobytes()


@pytest.fixture
def speech_chunk_bytes():
    """Loud PCM chunk whose RMS sits well above the speech threshold."""
    return np.tile([5000, -5000], 400).astype(np.int16).tobytes()


@pytest.fixture
def segment_config():
    """SegmentConfig tuned so a short utterance can be captured deterministically."""
    return SegmentConfig(
        speech_multiplier=4.0,
        silence_multiplier=2.0,
        min_threshold=0.0003,
        max_threshold=0.1,
        silent_chunks_for_end=3,
        pre_roll_chunks=2,
        min_duration_chunks=2,
        max_duration_chunks=100,
    )


@pytest.fixture
def utterance_segmenter(segment_config, audio_processor, sample_rate):
    """UtteranceSegmenter wired to the predictable analyzer and config."""
    return UtteranceSegmenter(segment_config=segment_config, analyzer=audio_processor, sample_rate=sample_rate)


@pytest_asyncio.fixture
async def audio_capture_service(event_bus, app_config):
    """AudioCaptureService with the PortAudio input stream patched out."""
    from vocalance.app.services.capture.audio_capture_service import AudioCaptureService

    loop = asyncio.get_running_loop()
    with patch("vocalance.app.services.capture.audio_capture_service.sd.InputStream"):
        yield AudioCaptureService(event_bus=event_bus, config=app_config, main_event_loop=loop)


@pytest.fixture
def event_collector(event_bus):
    """Factory subscribing a collector to an event type and returning its list."""

    def _subscribe(event_type):
        received = []

        async def _handler(event):
            received.append(event)

        event_bus.subscribe(event_type, _handler)
        return received

    return _subscribe


@pytest_asyncio.fixture
async def command_segmenter_service(event_bus, app_config):
    """CommandSegmenterService wired to a started event bus and default config."""
    from vocalance.app.services.command_flow.segmenting.command_segmenter_service import CommandSegmenterService

    return CommandSegmenterService(event_bus=event_bus, config=app_config)


@pytest_asyncio.fixture
async def sound_segmenter_service(event_bus, app_config):
    """SoundSegmenterService wired to a started event bus and default config."""
    from vocalance.app.services.command_flow.segmenting.sound_segmenter_service import SoundSegmenterService

    return SoundSegmenterService(event_bus=event_bus, config=app_config)


@pytest_asyncio.fixture
async def automation_service(event_bus, app_config):
    """AutomationService backed by a real serialised KeyboardInputService."""
    from vocalance.app.services.command_flow.execution.automation_service import AutomationService
    from vocalance.app.services.keyboard_input_service import KeyboardInputService

    input_service = KeyboardInputService(event_bus=event_bus)
    service = AutomationService(event_bus, app_config, input_service=input_service, activity_tracker=Mock())
    yield service
    await input_service.shutdown()


@pytest.fixture
def grid_service(mock_event_bus, app_config):
    """GridService wired to a mock event bus so published grid state events can be asserted."""
    from vocalance.app.services.command_flow.execution.grid.grid_service import GridService

    return GridService(event_bus=mock_event_bus, config=app_config)


@pytest_asyncio.fixture
async def click_tracker_service(mock_event_bus, mock_storage_service):
    """ClickTrackerService on the running loop with debounce scheduling stubbed out."""
    from vocalance.app.services.command_flow.execution.grid.click_tracker_service import ClickTrackerService

    loop = asyncio.get_running_loop()

    def _spawn(coro, name=None):
        coro.close()
        return Mock()

    lifecycle = Mock()
    lifecycle.spawn = Mock(side_effect=_spawn)
    return ClickTrackerService(
        event_bus=mock_event_bus,
        storage=mock_storage_service,
        gui_event_loop=loop,
        lifecycle=lifecycle,
        ui_refresh_debounce_s=0.0,
        persist_debounce_s=9999.0,
    )


@pytest.fixture
def parser_triggers():
    """Explicit lowercased trigger phrases for command-parsing grammar tests."""
    from vocalance.app.services.command_flow.parsing.text_command_parse import CommandParserTriggers

    return CommandParserTriggers(
        grid_show_phrase="go",
        grid_hover_phrase="hover",
        grid_drag_phrase="move",
        mark_create_prefix="mark",
        mark_delete_prefix="delete mark",
        mark_visualize_phrases=("show marks", "visualize marks"),
        mark_reset_phrases=("reset marks", "clear all marks"),
        mark_cancel_visualize_phrases=("cancel marks", "hide marks"),
        dictation_start_trigger="green",
        dictation_stop_trigger="amber",
        dictation_type_trigger="type",
        dictation_smart_trigger="smart green",
        dictation_visual_trigger="visual green",
        dictation_hidden_trigger="hidden green",
        dictation_amend_trigger="amend",
    )


@pytest.fixture
def parser_action_map():
    """Small phrase to AutomationCommand map for parser grammar tests."""
    from vocalance.app.config.command_types import AutomationCommand

    return {
        "copy": AutomationCommand(command_key="copy", action_type="hotkey", action_value="ctrl+c"),
        "scroll down": AutomationCommand(command_key="scroll down", action_type="scroll", action_value="down"),
    }


@pytest.fixture
def command_parser(mock_event_bus, app_config, mock_storage_service):
    """CentralizedCommandParser on a mock bus and in-memory storage for direct method calls."""
    from vocalance.app.services.command_flow.parsing.parser import CentralizedCommandParser

    return CentralizedCommandParser(event_bus=mock_event_bus, app_config=app_config, storage=mock_storage_service)


@pytest.fixture
def command_storage():
    """Plain async storage mock whose read result each test sets explicitly."""
    storage = Mock()
    storage.read = AsyncMock()
    storage.write = AsyncMock(return_value=True)
    return storage


@pytest.fixture
def command_management_service(mock_event_bus, command_storage, mock_protected_terms_validator):
    """CommandManagementService on a mock bus, settable storage, and permissive validator."""
    from vocalance.app.services.command_flow.management.command_management_service import CommandManagementService

    return CommandManagementService(
        event_bus=mock_event_bus,
        storage=command_storage,
        protected_terms_validator=mock_protected_terms_validator,
    )


@pytest.fixture
def protected_terms_storage():
    """Async storage mock returning empty marks and sound mappings by default."""
    from vocalance.app.services.storage.storage_models import MarksData, SoundMappingsData

    storage = Mock()

    def _read(model_type):
        return MarksData(marks={}) if model_type == MarksData else SoundMappingsData(mappings={})

    storage.read = AsyncMock(side_effect=_read)
    return storage


@pytest.fixture
def protected_terms_validator(app_config, protected_terms_storage):
    """Real ProtectedTermsValidator over config defaults and empty storage."""
    from vocalance.app.services.command_flow.management.protected_terms_validator import ProtectedTermsValidator

    return ProtectedTermsValidator(config=app_config, storage=protected_terms_storage)


@pytest_asyncio.fixture
async def mark_service(event_bus, app_config, mock_storage_service, mock_protected_terms_validator):
    """MarkService wired to in-memory storage and a permissive terms validator."""
    from vocalance.app.services.command_flow.execution.mark_service import MarkService
    from vocalance.app.services.keyboard_input_service import KeyboardInputService

    input_service = KeyboardInputService(event_bus=event_bus)
    service = MarkService(
        event_bus=event_bus,
        config=app_config,
        storage=mock_storage_service,
        protected_terms_validator=mock_protected_terms_validator,
        input_service=input_service,
    )
    yield service
    await input_service.shutdown()


@pytest_asyncio.fixture
async def dictation_alias_service(mock_event_bus):
    """DictationAliasService backed by an empty in-memory alias store.

    ``service.storage.write`` can be flipped to ``False`` by individual tests to
    exercise the persistence-failure rollback paths.
    """
    from vocalance.app.services.dictation_flow.dictation_alias_service import DictationAliasService
    from vocalance.app.services.storage.storage_models import DictationAliasData

    storage = Mock()
    storage.read = AsyncMock(return_value=DictationAliasData(aliases={}))
    storage.write = AsyncMock(return_value=True)
    service = DictationAliasService(event_bus=mock_event_bus, storage=storage, event_loop=asyncio.get_running_loop())
    await service.initialize()
    return service


@pytest.fixture
def dictation_text_input():
    """DictationTextInput in keyboard (non-clipboard) mode with zeroed delays.

    The injected input service runs callables inline so the real prose-join logic
    in ``input_text`` executes synchronously; tests patch the pyautogui boundary.
    """
    from vocalance.app.config.app_config import DictationConfig
    from vocalance.app.services.dictation_flow.text_input_service import DictationTextInput

    config = DictationConfig(
        use_clipboard=False,
        typing_delay=0.0,
        type_text_post_delay=0.0,
        clipboard_paste_delay_pre=0.0,
        clipboard_paste_delay_post=0.0,
        pyautogui_pause=0.0,
    )

    async def _run(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    input_service = Mock()
    input_service.run = _run
    return DictationTextInput(config=config, input_service=input_service)


@pytest.fixture
def patched_keyboard():
    """Patch the pyautogui keystroke boundary used by DictationTextInput."""
    with patch("pyautogui.write") as write, patch("pyautogui.press") as press:
        yield write, press


@pytest.fixture
def noop_alias_service():
    """Alias service stub whose ``extract_aliases`` returns the text unchanged with no map."""
    alias = Mock()
    alias.extract_aliases = lambda text: (text, {})
    return alias


@pytest.fixture
def agentic_prompts_storage():
    """Async storage mock returning empty agentic prompt data by default."""
    from vocalance.app.services.storage.storage_models import AgenticPromptsData

    storage = Mock()
    storage.read = AsyncMock(return_value=AgenticPromptsData(prompts=[]))
    storage.write = AsyncMock(return_value=True)
    return storage


@pytest_asyncio.fixture
async def agentic_prompt_service(mock_event_bus, app_config, agentic_prompts_storage):
    """Initialized AgenticPromptService starting from empty storage (seeds one default prompt)."""
    from vocalance.app.services.dictation_flow.llm.agentic_prompt_service import AgenticPromptService

    service = AgenticPromptService(event_bus=mock_event_bus, config=app_config, storage=agentic_prompts_storage)
    await service.initialize()
    return service


@pytest.fixture
def llm_service(mock_event_bus, isolated_storage_config):
    """LLMService on an isolated temp config; model loading and downloads are never invoked."""
    from vocalance.app.services.dictation_flow.llm.llm_service import LLMService

    return LLMService(event_bus=mock_event_bus, config=isolated_storage_config)


@pytest.fixture
def dictation_coordinator(mock_event_bus, app_config, mock_storage_service):
    """DictationCoordinator with heavy collaborators patched out for state-machine tests."""
    from vocalance.app.services.dictation_flow.dictation_coordinator import DictationCoordinator

    loop = asyncio.new_event_loop()
    with patch("vocalance.app.services.dictation_flow.dictation_coordinator.DictationTextInput"), patch(
        "vocalance.app.services.dictation_flow.dictation_coordinator.DictationAliasService"
    ), patch(
        "vocalance.app.services.dictation_flow.dictation_coordinator.llm_deps_available",
        return_value=False,
    ):
        coordinator = DictationCoordinator(
            event_bus=mock_event_bus,
            config=app_config,
            storage=mock_storage_service,
            gui_event_loop=loop,
            input_service=Mock(),
            lifecycle=Mock(),
            activity_tracker=Mock(),
        )
    yield coordinator
    loop.close()


@pytest.fixture
def theme_manager():
    """Fresh ThemeManager with default token config (no Qt app required)."""
    from vocalance.app.ui.qt_theme import ThemeManager

    return ThemeManager()


@pytest_asyncio.fixture
async def app_lifecycle():
    """AppLifecycle constructed on the running test loop."""
    from vocalance.app.lifecycle.lifecycle import AppLifecycle

    return AppLifecycle()


@pytest_asyncio.fixture
async def cancellation_token():
    """CancellationToken bound to the running test loop."""
    from vocalance.app.lifecycle.cancellation import CancellationToken

    return CancellationToken(asyncio.get_running_loop())


@pytest.fixture
def teardown_sink():
    """Shared list recording resource shutdown order for lifecycle tests."""
    return []


@pytest.fixture
def recording_resource_factory(teardown_sink):
    """Factory for AsyncCloseable test doubles that log their shutdown order.

    The returned callable accepts a ``tag`` and a ``mode`` of ``"async"``
    (default), ``"sync"``, or ``"slow"``. Async and sync resources append their
    tag to the shared ``teardown_sink`` when shut down; slow resources sleep past
    the lifecycle grace period without recording.
    """

    class _RecordingResource:
        def __init__(self, tag, mode="async"):
            self.tag = tag
            self.mode = mode
            self.shutdown_calls = 0

        def shutdown(self):
            self.shutdown_calls += 1
            if self.mode == "sync":
                teardown_sink.append(self.tag)
                return None
            return self._async_shutdown()

        async def _async_shutdown(self):
            if self.mode == "slow":
                await asyncio.sleep(60)
            else:
                teardown_sink.append(self.tag)

    def _make(tag, mode="async"):
        return _RecordingResource(tag, mode=mode)

    return _make
