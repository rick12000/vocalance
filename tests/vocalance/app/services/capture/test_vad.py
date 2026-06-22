import numpy as np
import pytest

from vocalance.app.services.capture.vad import AudioProcessor, Clip, Onset, SegmentConfig, UtteranceSegmenter


@pytest.mark.parametrize("enable_normalization", [True, False])
def test_process_chunk_removes_dc_offset_and_preserves_length(sample_rate, enable_normalization):
    processor = AudioProcessor(sample_rate=sample_rate, enable_normalization=enable_normalization)
    pcm = np.full(800, 1000, dtype=np.int16).tobytes()

    audio, energy = processor.process_chunk(pcm)

    assert len(audio) == 800
    assert abs(float(np.mean(audio))) < 1e-6
    assert energy >= 0.0


@pytest.mark.parametrize(
    "signal, expected",
    [
        (np.array([], dtype=np.float32), 0.0),
        (np.full(1000, 0.5, dtype=np.float32), 0.5),
        (np.tile([0.3, -0.3], 500).astype(np.float32), 0.3),
    ],
)
def test_calculate_rms_energy_matches_root_mean_square(audio_processor, signal, expected):
    assert audio_processor.calculate_rms_energy(signal) == pytest.approx(expected)


def test_noise_floor_becomes_stable_after_min_samples(audio_processor, silence_chunk_bytes):
    energy = audio_processor.process_chunk(silence_chunk_bytes)[1]

    estimate = None
    for _ in range(AudioProcessor.MIN_SAMPLES_FOR_STABLE + 5):
        estimate = audio_processor.update_noise_floor(energy, is_likely_speech=False)

    assert estimate.is_stable is True
    assert estimate.sample_count >= AudioProcessor.MIN_SAMPLES_FOR_STABLE
    assert estimate.value == pytest.approx(energy, abs=1e-6)


def test_noise_floor_ignores_speech_after_bootstrap(audio_processor, silence_chunk_bytes, speech_chunk_bytes):
    silence_energy = audio_processor.process_chunk(silence_chunk_bytes)[1]
    speech_energy = audio_processor.process_chunk(speech_chunk_bytes)[1]

    for _ in range(AudioProcessor.BOOTSTRAP_CHUNKS + 1):
        audio_processor.update_noise_floor(silence_energy, is_likely_speech=False)
    count_before = audio_processor.get_noise_floor().sample_count

    for _ in range(10):
        audio_processor.update_noise_floor(speech_energy, is_likely_speech=True)

    assert audio_processor.get_noise_floor().sample_count == count_before


@pytest.mark.parametrize("noise_floor", [0.00001, 0.01, 1.0])
def test_threshold_update_stays_within_bounds(vad_threshold, noise_floor):
    speech, silence = vad_threshold.update(noise_floor)

    assert vad_threshold.min_threshold <= speech <= vad_threshold.max_threshold
    assert vad_threshold.min_threshold * 0.5 <= silence <= vad_threshold.max_threshold * 0.5
    assert silence <= speech


def test_speech_and_silence_classification_follow_thresholds(vad_threshold):
    vad_threshold.update(0.01)

    assert vad_threshold.is_speech(vad_threshold.speech_threshold + 0.001)
    assert not vad_threshold.is_speech(vad_threshold.speech_threshold - 0.0001)
    assert vad_threshold.is_silence(vad_threshold.silence_threshold - 0.0001)
    assert not vad_threshold.is_silence(vad_threshold.silence_threshold + 0.001)


def test_segmenter_emits_clip_after_speech_then_silence(utterance_segmenter, silence_chunk_bytes, speech_chunk_bytes, sample_rate):
    hits = []
    sequence = [silence_chunk_bytes] * 2 + [speech_chunk_bytes] * 5 + [silence_chunk_bytes] * 3
    for ts, chunk in enumerate(sequence):
        hits.extend(utterance_segmenter.feed_pcm_chunk(chunk, ts=float(ts)))

    clips = [hit for hit in hits if isinstance(hit, Clip)]
    assert len(clips) == 1
    assert clips[0].sample_rate == sample_rate
    assert len(clips[0].pcm_bytes) > 0
    assert len(clips[0].pcm_bytes) % 2 == 0


def test_segmenter_discards_utterance_below_min_duration(audio_processor, silence_chunk_bytes, speech_chunk_bytes, sample_rate):
    config = SegmentConfig(
        speech_multiplier=4.0,
        silence_multiplier=2.0,
        min_threshold=0.0003,
        max_threshold=0.1,
        silent_chunks_for_end=3,
        pre_roll_chunks=2,
        min_duration_chunks=100,
        max_duration_chunks=1000,
    )
    segmenter = UtteranceSegmenter(segment_config=config, analyzer=audio_processor, sample_rate=sample_rate)

    hits = []
    sequence = [silence_chunk_bytes] * 2 + [speech_chunk_bytes] * 2 + [silence_chunk_bytes] * 3
    for ts, chunk in enumerate(sequence):
        hits.extend(segmenter.feed_pcm_chunk(chunk, ts=float(ts)))

    assert [hit for hit in hits if isinstance(hit, Clip)] == []


def test_segmenter_emits_onset_when_enabled(audio_processor, silence_chunk_bytes, speech_chunk_bytes, sample_rate):
    config = SegmentConfig(
        speech_multiplier=4.0,
        silence_multiplier=2.0,
        min_threshold=0.0003,
        max_threshold=0.1,
        silent_chunks_for_end=3,
        pre_roll_chunks=2,
        min_duration_chunks=2,
        max_duration_chunks=100,
        emit_onset=True,
    )
    segmenter = UtteranceSegmenter(segment_config=config, analyzer=audio_processor, sample_rate=sample_rate)

    segmenter.feed_pcm_chunk(silence_chunk_bytes, ts=0.0)
    segmenter.feed_pcm_chunk(silence_chunk_bytes, ts=1.0)
    hits = segmenter.feed_pcm_chunk(speech_chunk_bytes, ts=2.0)

    onsets = [hit for hit in hits if isinstance(hit, Onset)]
    assert len(onsets) == 1
    assert onsets[0].ts == 2.0
