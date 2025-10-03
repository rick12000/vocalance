"""
Unit tests for WhisperSpeechToText
"""
import pytest
import numpy as np
import threading
from unittest.mock import Mock, patch, MagicMock

from iris.services.audio.whisper_stt import WhisperSpeechToText


class TestWhisperSpeechToText:
    """Test WhisperSpeechToText functionality"""
    
    @pytest.fixture
    def mock_whisper_model(self):
        """Mock faster-whisper model"""
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
    
    @pytest.fixture
    def mock_stt_config(self):
        """Mock STT configuration"""
        config = Mock()
        config.whisper_beam_size = 5
        config.whisper_temperature = 0.0
        config.whisper_no_speech_threshold = 0.6
        return config
    
    @pytest.fixture
    def whisper_stt(self, mock_whisper_model, mock_stt_config):
        """Create WhisperSpeechToText instance for testing"""
        with patch('iris.services.audio.whisper_stt.WhisperModel') as mock_whisper_class:
            mock_whisper_class.return_value = mock_whisper_model
            
            stt = WhisperSpeechToText(
                model_name="base",
                device="cpu",
                sample_rate=16000,
                config=mock_stt_config
            )
            stt._model = mock_whisper_model
            return stt
    
    def test_initialization(self, mock_stt_config):
        """Test WhisperSpeechToText initialization"""
        with patch('iris.services.audio.whisper_stt.WhisperModel') as mock_whisper_class:
            mock_model = Mock()
            mock_whisper_class.return_value = mock_model
            
            stt = WhisperSpeechToText(
                model_name="small",
                device="cpu",
                sample_rate=22050,
                config=mock_stt_config
            )
            
            assert stt._model_name == "small"
            assert stt._device == "cpu"
            assert stt._sample_rate == 22050
            assert stt._config == mock_stt_config
            assert isinstance(stt._model_lock, threading.RLock)
            assert stt._duplicate_filter is not None
    
    def test_successful_recognition(self, whisper_stt, test_audio_bytes):
        """Test successful speech recognition"""
        result = whisper_stt.recognize(test_audio_bytes, 16000)
        
        assert result == "test dictation text"
        whisper_stt._model.transcribe.assert_called_once()
    
    def test_empty_audio_handling(self, whisper_stt):
        """Test handling of empty audio input"""
        result = whisper_stt.recognize(b"", 16000)
        assert result == ""
        
        result = whisper_stt.recognize(None, 16000)
        assert result == ""
    
    def test_short_audio_filtering(self, whisper_stt):
        """Test that very short audio segments are filtered out"""
        # Create very short audio (< 0.3 seconds)
        short_audio = np.random.randint(-1000, 1000, size=2000, dtype=np.int16).tobytes()
        
        result = whisper_stt.recognize(short_audio, 16000)
        assert result == ""
    
    def test_sample_rate_mismatch_warning(self, whisper_stt, test_audio_bytes):
        """Test warning on sample rate mismatch"""
        with patch('iris.services.audio.whisper_stt.logger') as mock_logger:
            result = whisper_stt.recognize(test_audio_bytes, 44100)
            
            mock_logger.warning.assert_called_once()
            assert "Sample rate mismatch" in str(mock_logger.warning.call_args)
    
    def test_audio_preparation(self, whisper_stt):
        """Test audio bytes to numpy array conversion"""
        # Create test audio bytes
        audio_data = np.random.randint(-32768, 32767, size=1000, dtype=np.int16)
        audio_bytes = audio_data.tobytes()
        
        prepared_audio = whisper_stt._prepare_audio(audio_bytes)
        
        assert isinstance(prepared_audio, np.ndarray)
        assert prepared_audio.dtype == np.float32
        assert len(prepared_audio) == len(audio_data)
        assert np.all(prepared_audio >= -1.0) and np.all(prepared_audio <= 1.0)
    
    def test_transcription_options_short_audio(self, whisper_stt):
        """Test transcription options for short audio"""
        options = whisper_stt._get_transcription_options(3.0)
        
        assert options["language"] == "en"
        assert options["beam_size"] == 4  # Reduced for short audio
        assert options["word_timestamps"] is True
        assert options["vad_filter"] is True
    
    def test_transcription_options_long_audio(self, whisper_stt):
        """Test transcription options for longer audio"""
        options = whisper_stt._get_transcription_options(10.0)
        
        assert options["language"] == "en"
        assert options["beam_size"] == 5  # Full beam size
        assert options["word_timestamps"] is True
        assert options["vad_filter"] is True
    
    def test_text_extraction_from_segments(self, whisper_stt):
        """Test text extraction from multiple segments"""
        # Create mock segments
        segment1 = Mock()
        segment1.text = "Hello"
        segment1.avg_logprob = -0.3
        
        segment2 = Mock()
        segment2.text = " world"
        segment2.avg_logprob = -0.7
        
        segments = [segment1, segment2]
        
        text, confidence = whisper_stt._extract_text_from_segments(segments)
        
        assert text == "Hello world"
        assert 0.0 <= confidence <= 1.0
    
    def test_text_extraction_empty_segments(self, whisper_stt):
        """Test text extraction with empty segments"""
        segment1 = Mock()
        segment1.text = ""
        segment1.avg_logprob = -0.5
        
        segment2 = Mock()
        segment2.text = "   "  # Whitespace only
        segment2.avg_logprob = -0.3
        
        segments = [segment1, segment2]
        
        text, confidence = whisper_stt._extract_text_from_segments(segments)
        
        assert text == ""
        assert confidence == 0.0  # No valid segments
    
    def test_text_normalization(self, whisper_stt):
        """Test text normalization functionality"""
        # Test whitespace normalization
        assert whisper_stt._normalize_text("  hello   world  ") == "hello world"
        
        # Test artifact removal
        assert whisper_stt._normalize_text("um hello world") == "hello world"
        assert whisper_stt._normalize_text("hello world uh") == "hello world"
        
        # Test consecutive duplicate removal
        assert whisper_stt._normalize_text("hello hello world") == "hello world"
        assert whisper_stt._normalize_text("the the the end") == "the end"
    
    def test_text_normalization_edge_cases(self, whisper_stt):
        """Test text normalization edge cases"""
        # Empty text
        assert whisper_stt._normalize_text("") == ""
        assert whisper_stt._normalize_text("   ") == ""
        
        # Single word
        assert whisper_stt._normalize_text("hello") == "hello"
        
        # All artifacts
        assert whisper_stt._normalize_text("um uh like so") == ""
    
    def test_duplicate_filtering(self, whisper_stt, test_audio_bytes):
        """Test duplicate text filtering"""
        # Mock duplicate filter to return True
        whisper_stt._duplicate_filter.is_duplicate = Mock(return_value=True)
        
        result = whisper_stt.recognize(test_audio_bytes, 16000)
        assert result == ""
        
        whisper_stt._duplicate_filter.is_duplicate.assert_called_once()
    
    def test_recognition_error_handling(self, whisper_stt, test_audio_bytes):
        """Test error handling during recognition"""
        whisper_stt._model.transcribe.side_effect = Exception("Recognition error")
        
        result = whisper_stt.recognize(test_audio_bytes, 16000)
        assert result == ""
    
    def test_empty_recognition_result(self, whisper_stt, test_audio_bytes):
        """Test handling of empty recognition results"""
        # Mock empty segments
        whisper_stt._model.transcribe.return_value = ([], Mock())
        
        result = whisper_stt.recognize(test_audio_bytes, 16000)
        assert result == ""
    
    def test_thread_safety(self, whisper_stt, test_audio_bytes):
        """Test thread safety of recognition operations"""
        results = []
        
        def recognition_task():
            result = whisper_stt.recognize(test_audio_bytes, 16000)
            results.append(result)
        
        # Run multiple recognition tasks concurrently
        threads = []
        for _ in range(3):  # Fewer threads for Whisper due to processing overhead
            thread = threading.Thread(target=recognition_task)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All should succeed
        assert len(results) == 3
        assert all(result == "test dictation text" for result in results)
    
    def test_context_reset(self, whisper_stt):
        """Test context and cache reset functionality"""
        # Add some cache
        whisper_stt._text_cache = {"key": "value"}
        
        whisper_stt.reset_context()
        
        # Should clear cache if it exists
        if hasattr(whisper_stt, '_text_cache'):
            assert len(whisper_stt._text_cache) == 0
    
    @pytest.mark.asyncio
    async def test_shutdown(self, whisper_stt):
        """Test proper shutdown and resource cleanup"""
        await whisper_stt.shutdown()
        
        # Should clear model and related resources
        assert whisper_stt._model is None
    
    def test_confidence_calculation(self, whisper_stt):
        """Test confidence calculation from avg_logprob"""
        # Test segment with good confidence
        good_segment = Mock()
        good_segment.text = "clear speech"
        good_segment.avg_logprob = -0.2
        
        # Test segment with poor confidence
        poor_segment = Mock()
        poor_segment.text = "unclear speech"
        poor_segment.avg_logprob = -1.5
        
        # Test segment without logprob
        no_logprob_segment = Mock()
        no_logprob_segment.text = "no confidence"
        no_logprob_segment.avg_logprob = None
        
        text, confidence = whisper_stt._extract_text_from_segments([good_segment])
        assert confidence > 0.7
        
        text, confidence = whisper_stt._extract_text_from_segments([poor_segment])
        assert confidence < 0.3
        
        text, confidence = whisper_stt._extract_text_from_segments([no_logprob_segment])
        assert confidence == 0.8  # Default confidence
    
    def test_device_configuration(self, mock_stt_config):
        """Test device configuration is properly handled"""
        with patch('iris.services.audio.whisper_stt.WhisperModel') as mock_whisper_class:
            mock_model = Mock()
            mock_whisper_class.return_value = mock_model
            
            # Should force CPU regardless of input
            stt = WhisperSpeechToText(
                model_name="base",
                device="cuda",  # This should be overridden
                config=mock_stt_config
            )
            
            assert stt._device == "cpu"
            
            # Verify model was initialized with CPU
            mock_whisper_class.assert_called_once()
            call_args = mock_whisper_class.call_args
            assert call_args[1]["device"] == "cpu"
    
    def test_model_warmup(self, mock_stt_config):
        """Test model warmup during initialization"""
        with patch('iris.services.audio.whisper_stt.WhisperModel') as mock_whisper_class:
            mock_model = Mock()
            mock_model.transcribe.return_value = ([], Mock())
            mock_whisper_class.return_value = mock_model
            
            stt = WhisperSpeechToText(
                model_name="base",
                device="cpu",
                config=mock_stt_config
            )
            
            # Should call transcribe during warmup
            mock_model.transcribe.assert_called()
            
            # Warmup call should use dummy audio
            warmup_args = mock_model.transcribe.call_args[0]
            assert isinstance(warmup_args[0], np.ndarray)
            assert len(warmup_args[0]) == 16000  # 1 second of dummy audio

