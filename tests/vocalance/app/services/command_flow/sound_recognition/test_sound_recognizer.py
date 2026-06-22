import numpy as np
import pytest

from vocalance.app.services.command_flow.sound_recognition.sound_recognizer import SimpleStandardScaler


def test_scaler_fit_transform_standardizes():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    out = SimpleStandardScaler().fit_transform(X)

    assert out.shape == X.shape
    assert np.allclose(out.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(out.std(axis=0), 1.0, atol=1e-6)


def test_scaler_floors_zero_variance_std():
    X = np.array([[1.0, 5.0], [1.0, 5.0]])

    scaler = SimpleStandardScaler().fit(X)

    assert np.allclose(scaler.std, 0.01)


def test_scaler_transform_before_fit_raises():
    with pytest.raises(ValueError):
        SimpleStandardScaler().transform(np.zeros((1, 2)))


def test_scaler_roundtrip_preserves_transform():
    X = np.array([[1.0, 2.0], [3.0, 8.0], [5.0, 6.0]])
    scaler = SimpleStandardScaler().fit(X)

    restored = SimpleStandardScaler.from_dict(scaler.to_dict())

    assert np.allclose(restored.transform(X), scaler.transform(X))


@pytest.mark.parametrize("num_frames", [1, 9, 1000])
def test_aggregate_temporal_embeddings_shape_and_composition(isolated_recognizer, num_frames):
    frames = np.random.randn(num_frames, 1024).astype(np.float32)

    out = isolated_recognizer._aggregate_temporal_embeddings(frames)

    assert out.shape == (5120,)
    assert np.allclose(out[:1024], frames.mean(axis=0))
    assert np.allclose(out[1024:2048], frames.std(axis=0))


def test_extract_embedding_shape(isolated_recognizer, sample_rate):
    t = np.linspace(0, 0.5, int(0.5 * sample_rate))
    audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)

    embedding = isolated_recognizer._extract_embedding(audio=audio, sr=sample_rate)

    assert embedding.shape == (5120,)


@pytest.mark.parametrize("duration", [0.02, 0.5, 3.0])
def test_preprocess_clamps_duration(preprocessor, sample_rate, duration):
    t = np.linspace(0, duration, max(1, int(duration * sample_rate)))
    audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)

    out = preprocessor.preprocess_audio(audio=audio, sr=sample_rate)

    min_samples = int(preprocessor.min_sound_duration * preprocessor.target_sr)
    max_samples = int(preprocessor.max_sound_duration * preprocessor.target_sr)
    assert out.ndim == 1
    assert min_samples <= len(out) <= max_samples


def test_preprocess_peak_normalizes(preprocessor, sample_rate):
    t = np.linspace(0, 0.5, int(0.5 * sample_rate))
    audio = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)

    out = preprocessor.preprocess_audio(audio=audio, sr=sample_rate)

    assert np.isclose(np.max(np.abs(out)), preprocessor.normalization_level, atol=1e-4)


def test_preprocess_converts_stereo_to_mono(preprocessor, sample_rate):
    stereo = (np.random.randn(sample_rate // 2, 2) * 0.3).astype(np.float32)

    out = preprocessor.preprocess_audio(audio=stereo, sr=sample_rate)

    assert out.ndim == 1


def test_preprocess_resamples_to_target(preprocessor):
    original_sr = 44100
    duration = 0.5
    t = np.linspace(0, duration, int(duration * original_sr))
    audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)

    out = preprocessor.preprocess_audio(audio=audio, sr=original_sr)

    expected = int(duration * preprocessor.target_sr)
    assert abs(len(out) - expected) < 200


def test_preprocess_rejects_empty(preprocessor, sample_rate):
    with pytest.raises(ValueError):
        preprocessor.preprocess_audio(audio=np.array([], dtype=np.float32), sr=sample_rate)


def test_preprocess_rejects_non_array(preprocessor, sample_rate):
    with pytest.raises(TypeError):
        preprocessor.preprocess_audio(audio=[1, 2, 3], sr=sample_rate)


@pytest.mark.parametrize("sr", [0, -1])
def test_preprocess_rejects_invalid_sample_rate(preprocessor, sr):
    with pytest.raises(ValueError):
        preprocessor.preprocess_audio(audio=np.ones(2000, dtype=np.float32), sr=sr)


def test_recognize_returns_none_without_training(isolated_recognizer, sample_rate):
    t = np.linspace(0, 0.5, int(0.5 * sample_rate))
    audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)

    assert isolated_recognizer.recognize_sound(audio=audio, sr=sample_rate) is None


def test_recognize_majority_vote_returns_label(recognizer_identity_query):
    recognizer, audio, embedding = recognizer_identity_query
    recognizer.embeddings = np.array([embedding] * 5 + [-embedding] * 2)
    recognizer.labels = ["click"] * 5 + ["other"] * 2

    result = recognizer.recognize_sound(audio=audio, sr=recognizer.target_sr)

    assert result is not None
    label, confidence = result
    assert label == "click"
    assert confidence > 0.9


def test_recognize_below_confidence_threshold(recognizer_identity_query):
    recognizer, audio, embedding = recognizer_identity_query
    recognizer.embeddings = np.array([-embedding])
    recognizer.labels = ["click"]
    recognizer.confidence_threshold = 0.5

    assert recognizer.recognize_sound(audio=audio, sr=recognizer.target_sr) is None


def test_recognize_ignores_esc50_only_neighbors(recognizer_identity_query):
    recognizer, audio, embedding = recognizer_identity_query
    recognizer.embeddings = np.array([embedding] * 7)
    recognizer.labels = ["esc50_x"] * 7

    assert recognizer.recognize_sound(audio=audio, sr=recognizer.target_sr) is None


@pytest.mark.asyncio
async def test_train_sound_adds_embeddings(isolated_recognizer, sample_rate):
    t = np.linspace(0, 0.5, int(0.5 * sample_rate))
    first = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    second = (np.sin(2 * np.pi * 880 * t) * 0.5).astype(np.float32)

    result = await isolated_recognizer.train_sound("click", [(first, sample_rate), (second, sample_rate)])

    assert result is True
    assert len(isolated_recognizer.embeddings) == 2
    assert isolated_recognizer.labels == ["click", "click"]


@pytest.mark.asyncio
@pytest.mark.parametrize("label,samples", [("", [(np.zeros(10, dtype=np.float32), 16000)]), ("click", [])])
async def test_train_sound_rejects_invalid(isolated_recognizer, label, samples):
    result = await isolated_recognizer.train_sound(label, samples)

    assert result is False
    assert len(isolated_recognizer.embeddings) == 0


@pytest.mark.asyncio
async def test_delete_sound_removes_label(isolated_recognizer):
    isolated_recognizer.embeddings = np.random.randn(5, 8)
    isolated_recognizer.labels = ["a", "a", "a", "b", "b"]
    isolated_recognizer.mappings = {"a": "copy"}

    result = await isolated_recognizer.delete_sound("a")

    assert result is True
    assert len(isolated_recognizer.embeddings) == 2
    assert isolated_recognizer.labels == ["b", "b"]
    assert "a" not in isolated_recognizer.mappings


@pytest.mark.asyncio
async def test_delete_sound_missing_returns_false(isolated_recognizer):
    isolated_recognizer.embeddings = np.random.randn(2, 8)
    isolated_recognizer.labels = ["a", "a"]

    result = await isolated_recognizer.delete_sound("z")

    assert result is False
    assert len(isolated_recognizer.embeddings) == 2


def test_get_stats_counts(isolated_recognizer):
    isolated_recognizer.embeddings = np.random.randn(5, 8)
    isolated_recognizer.labels = ["c1", "c1", "esc50_a", "c2", "esc50_b"]
    isolated_recognizer.mappings = {"c1": "copy"}

    stats = isolated_recognizer.get_stats()

    assert stats["total_embeddings"] == 5
    assert stats["custom_sounds"] == 2
    assert stats["esc50_samples"] == 2
    assert stats["mappings"] == 1
    assert stats["trained_sounds"] == {"c1": 2, "c2": 1}
    assert stats["model_ready"] is True


@pytest.mark.asyncio
async def test_set_and_get_mapping(isolated_recognizer):
    result = await isolated_recognizer.set_mapping("click", "copy")

    assert result is True
    assert isolated_recognizer.get_mapping("click") == "copy"
    assert isolated_recognizer.get_mapping("missing") is None


@pytest.mark.asyncio
async def test_reset_all_sounds_clears_state(isolated_recognizer):
    isolated_recognizer.embeddings = np.random.randn(3, 8)
    isolated_recognizer.labels = ["a", "a", "b"]
    isolated_recognizer.mappings = {"a": "copy"}

    result = await isolated_recognizer.reset_all_sounds()

    assert result is True
    assert len(isolated_recognizer.embeddings) == 0
    assert isolated_recognizer.labels == []
    assert isolated_recognizer.mappings == {}


@pytest.mark.parametrize("value", [-0.1, 1.5, "x"])
def test_confidence_threshold_rejects_invalid(isolated_recognizer, value):
    original = isolated_recognizer.confidence_threshold

    isolated_recognizer.on_confidence_threshold_updated(value)

    assert isolated_recognizer.confidence_threshold == original


def test_confidence_threshold_updates_valid(isolated_recognizer):
    isolated_recognizer.on_confidence_threshold_updated(0.5)

    assert isolated_recognizer.confidence_threshold == 0.5
