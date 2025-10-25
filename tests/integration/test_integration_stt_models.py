"""
Integration tests for STT models (Vosk and Whisper).

Tests the recognition accuracy and performance of both Vosk and Whisper STT models
using real audio samples recorded with the system's audio recorder.
"""
import time
from pathlib import Path


def _get_expected_text(filename: str) -> str:
    """Extract expected text from filename."""
    name = filename.replace(".bytes", "").replace("_", " ")
    return name.lower().strip()


def _load_audio_bytes(file_path: Path) -> bytes:
    """Load audio bytes from file."""
    with open(file_path, "rb") as f:
        return f.read()


def _calculate_word_count(text: str) -> int:
    """Calculate number of words in text."""
    return len(text.strip().split())


def _calculate_word_accuracy(expected: str, recognized: str):
    """
    Calculate per-word accuracy.

    Returns:
        (accuracy, correct_words, total_words)
    """
    expected_words = expected.lower().strip().split()
    recognized_words = [w.lower().rstrip(".,!?;:") for w in recognized.strip().split()]

    total_words = len(expected_words)

    if total_words == 0:
        return 0.0, 0, 0

    correct_words = sum(1 for exp_word in expected_words if exp_word in recognized_words)

    accuracy = correct_words / total_words
    return accuracy


def test_vosk_recognition_accuracy_and_performance(vosk_stt, vosk_test_files, sample_rate):
    single_word_results = []
    multi_word_results = []

    for file_path in vosk_test_files:
        expected_text = _get_expected_text(file_path.name)
        word_count = _calculate_word_count(expected_text)

        audio_bytes = _load_audio_bytes(file_path)

        start_time = time.time()
        recognized_text = vosk_stt.recognize_sync(audio_bytes, sample_rate)
        runtime_ms = (time.time() - start_time) * 1000

        is_correct = recognized_text.strip().lower() == expected_text
        accuracy = 1.0 if is_correct else 0.0

        result = {
            "filename": file_path.name,
            "expected": expected_text,
            "recognized": recognized_text.strip().lower(),
            "correct": is_correct,
            "accuracy": accuracy,
            "runtime_ms": runtime_ms,
            "word_count": word_count,
        }

        if word_count == 1:
            single_word_results.append(result)
        else:
            multi_word_results.append(result)

    # Calculate overall metrics
    all_results = single_word_results + multi_word_results
    overall_accuracy = sum(r["accuracy"] for r in all_results) / len(all_results)
    overall_avg_runtime = sum(r["runtime_ms"] for r in all_results) / len(all_results)

    # Calculate single-word metrics
    if single_word_results:
        single_word_accuracy = sum(r["accuracy"] for r in single_word_results) / len(single_word_results)
        single_word_avg_runtime = sum(r["runtime_ms"] for r in single_word_results) / len(single_word_results)
    else:
        single_word_accuracy = None
        single_word_avg_runtime = None

    # Calculate multi-word metrics
    if multi_word_results:
        multi_word_accuracy = sum(r["accuracy"] for r in multi_word_results) / len(multi_word_results)
        multi_word_avg_runtime = sum(r["runtime_ms"] for r in multi_word_results) / len(multi_word_results)
    else:
        multi_word_accuracy = None
        multi_word_avg_runtime = None

    # Assertions
    assert overall_accuracy == 1.0
    assert overall_avg_runtime < 500

    if single_word_results:
        assert single_word_accuracy == 1.0
        assert single_word_avg_runtime < 500

    if multi_word_results:
        assert multi_word_accuracy == 1.0
        assert multi_word_avg_runtime < 500


def test_whisper_dictation_accuracy_and_performance(whisper_stt, dictation_file, sample_rate):
    audio_bytes = _load_audio_bytes(dictation_file)
    expected_text = "this is a test of the dictation capabilities"

    start_time = time.time()
    recognized_text = whisper_stt.recognize(audio_bytes, sample_rate)
    runtime_ms = (time.time() - start_time) * 1000
    runtime_s = runtime_ms / 1000

    # NOTE: We use per word accuracy on a single prompt for Whisper:
    accuracy = _calculate_word_accuracy(expected_text, recognized_text)

    assert runtime_s < 2.0
    assert accuracy == 1.0
