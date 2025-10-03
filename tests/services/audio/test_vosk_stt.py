"""
Unit tests for EnhancedVoskSTT
"""
import pytest
import json
import threading
from unittest.mock import Mock, patch, MagicMock

from iris.services.audio.vosk_stt import EnhancedVoskSTT


class TestEnhancedVoskSTT:
    """Test EnhancedVoskSTT functionality"""
    
    @pytest.fixture
    def mock_vosk_components(self):
        """Mock Vosk model and recognizer components"""
        mock_model = Mock()
        mock_recognizer = Mock()
        
        # Mock recognition responses
        mock_recognizer.AcceptWaveform.return_value = True
        mock_recognizer.FinalResult.return_value = '{"text": "test command"}'
        mock_recognizer.Result.return_value = '{"text": "test command"}'
        mock_recognizer.PartialResult.return_value = '{"partial": "test"}'
        mock_recognizer.Reset = Mock()
        
        return mock_model, mock_recognizer
    
    @pytest.fixture
    def vosk_stt(self, mock_global_config, mock_vosk_components):
        """Create EnhancedVoskSTT instance for testing"""
        mock_model, mock_recognizer = mock_vosk_components
        
        with patch('iris.services.audio.vosk_stt.vosk') as mock_vosk:
            mock_vosk.Model.return_value = mock_model
            mock_vosk.KaldiRecognizer.return_value = mock_recognizer
            
            stt = EnhancedVoskSTT(
                model_path="test/model/path",
                sample_rate=16000,
                config=mock_global_config
            )
            stt._recognizer = mock_recognizer
            return stt
    
    def test_initialization(self, mock_global_config):
        """Test VoskSTT initialization"""
        with patch('iris.services.audio.vosk_stt.vosk') as mock_vosk:
            mock_model = Mock()
            mock_recognizer = Mock()
            mock_vosk.Model.return_value = mock_model
            mock_vosk.KaldiRecognizer.return_value = mock_recognizer
            
            stt = EnhancedVoskSTT(
                model_path="test/model/path",
                sample_rate=16000,
                config=mock_global_config
            )
            
            assert stt._sample_rate == 16000
            assert stt._model_path == "test/model/path"
            assert stt._config == mock_global_config
            assert isinstance(stt._recognizer_lock, threading.RLock)
            assert stt._duplicate_filter is not None
    
    def test_successful_recognition(self, vosk_stt, test_audio_bytes):
        """Test successful speech recognition"""
        result = vosk_stt.recognize(test_audio_bytes, 16000)
        
        assert result == "test command"
        vosk_stt._recognizer.Reset.assert_called_once()
        vosk_stt._recognizer.AcceptWaveform.assert_called_once_with(test_audio_bytes)
        vosk_stt._recognizer.FinalResult.assert_called_once()
    
    def test_empty_audio_handling(self, vosk_stt):
        """Test handling of empty audio input"""
        result = vosk_stt.recognize(b"", 16000)
        assert result == ""
        
        result = vosk_stt.recognize(None, 16000)
        assert result == ""
    
    def test_recognition_error_handling(self, vosk_stt, test_audio_bytes):
        """Test error handling during recognition"""
        vosk_stt._recognizer.AcceptWaveform.side_effect = Exception("Recognition error")
        
        result = vosk_stt.recognize(test_audio_bytes, 16000)
        assert result == ""
    
    def test_empty_recognition_result(self, vosk_stt, test_audio_bytes):
        """Test handling of empty recognition results"""
        vosk_stt._recognizer.FinalResult.return_value = '{"text": ""}'
        
        result = vosk_stt.recognize(test_audio_bytes, 16000)
        assert result == ""
    
    def test_duplicate_filtering(self, vosk_stt, test_audio_bytes):
        """Test duplicate text filtering"""
        # Mock duplicate filter to return True for duplicate
        vosk_stt._duplicate_filter.is_duplicate = Mock(return_value=True)
        
        result = vosk_stt.recognize(test_audio_bytes, 16000)
        assert result == ""
        
        vosk_stt._duplicate_filter.is_duplicate.assert_called_once_with("test command")
    
    def test_streaming_recognition_final_result(self, vosk_stt, test_audio_bytes):
        """Test streaming recognition with final result available"""
        vosk_stt._recognizer.AcceptWaveform.return_value = True
        vosk_stt._recognizer.Result.return_value = '{"text": "final command"}'
        
        result = vosk_stt.recognize_streaming(test_audio_bytes, True)
        
        assert result == "final command"
        vosk_stt._recognizer.AcceptWaveform.assert_called_once_with(test_audio_bytes)
        vosk_stt._recognizer.Result.assert_called_once()
    
    def test_streaming_recognition_partial_result(self, vosk_stt, test_audio_bytes):
        """Test streaming recognition with partial result"""
        vosk_stt._recognizer.AcceptWaveform.return_value = False
        vosk_stt._recognizer.PartialResult.return_value = '{"partial": "partial text"}'
        
        result = vosk_stt.recognize_streaming(test_audio_bytes, False)
        
        assert result == "partial text"
        vosk_stt._recognizer.AcceptWaveform.assert_called_once_with(test_audio_bytes)
        vosk_stt._recognizer.PartialResult.assert_called_once()
    
    def test_streaming_empty_audio(self, vosk_stt):
        """Test streaming recognition with empty audio"""
        result = vosk_stt.recognize_streaming(b"", False)
        assert result is None
        
        result = vosk_stt.recognize_streaming(None, False)
        assert result is None
    
    def test_streaming_recognition_error(self, vosk_stt, test_audio_bytes):
        """Test error handling in streaming recognition"""
        vosk_stt._recognizer.AcceptWaveform.side_effect = Exception("Streaming error")
        
        result = vosk_stt.recognize_streaming(test_audio_bytes, False)
        assert result is None
    
    @pytest.mark.parametrize("text,expected", [
        ("", True),
        ("a", True),
        ("  ", True),
        ("hello world", False),
        ("test test test", True),  # Repetitive pattern
        ("good morning", False)
    ])
    def test_nonsense_detection(self, vosk_stt, text, expected):
        """Test nonsense text detection"""
        result = vosk_stt._is_nonsense(text)
        assert result == expected
    
    def test_streaming_nonsense_filtering(self, vosk_stt, test_audio_bytes):
        """Test that nonsense text is filtered in streaming recognition"""
        vosk_stt._recognizer.AcceptWaveform.return_value = False
        vosk_stt._recognizer.PartialResult.return_value = '{"partial": "a"}'  # Nonsense
        
        result = vosk_stt.recognize_streaming(test_audio_bytes, False)
        assert result is None
    
    def test_case_normalization_in_streaming(self, vosk_stt, test_audio_bytes):
        """Test case normalization in streaming recognition"""
        vosk_stt._recognizer.AcceptWaveform.return_value = False
        vosk_stt._recognizer.PartialResult.return_value = '{"partial": "HELLO WORLD"}'
        
        result = vosk_stt.recognize_streaming(test_audio_bytes, False)
        assert result == "hello world"
    
    def test_thread_safety(self, vosk_stt, test_audio_bytes):
        """Test thread safety of recognition operations"""
        results = []
        
        def recognition_task():
            result = vosk_stt.recognize(test_audio_bytes, 16000)
            results.append(result)
        
        # Run multiple recognition tasks concurrently
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=recognition_task)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All should succeed
        assert len(results) == 5
        assert all(result == "test command" for result in results)
    
    @pytest.mark.asyncio
    async def test_shutdown(self, vosk_stt):
        """Test proper shutdown and resource cleanup"""
        await vosk_stt.shutdown()
        
        # Should clear recognizer and model
        assert vosk_stt._recognizer is None
        assert vosk_stt._model is None
    
    def test_malformed_json_handling(self, vosk_stt, test_audio_bytes):
        """Test handling of malformed JSON responses"""
        vosk_stt._recognizer.FinalResult.return_value = '{"text": malformed json'
        
        result = vosk_stt.recognize(test_audio_bytes, 16000)
        assert result == ""
    
    def test_sample_rate_parameter_ignored(self, vosk_stt, test_audio_bytes):
        """Test that sample rate parameter is accepted but internal rate is used"""
        # Should work regardless of provided sample rate
        result = vosk_stt.recognize(test_audio_bytes, 44100)
        assert result == "test command"
        
        result = vosk_stt.recognize(test_audio_bytes, 8000)
        assert result == "test command"
    
    def test_configuration_parameters(self, mock_global_config):
        """Test that configuration parameters are properly used"""
        with patch('iris.services.audio.vosk_stt.vosk') as mock_vosk:
            mock_model = Mock()
            mock_recognizer = Mock()
            mock_vosk.Model.return_value = mock_model
            mock_vosk.KaldiRecognizer.return_value = mock_recognizer
            
            stt = EnhancedVoskSTT(
                model_path="custom/path",
                sample_rate=22050,
                config=mock_global_config
            )
            
            # Verify initialization parameters
            mock_vosk.Model.assert_called_once_with("custom/path")
            mock_vosk.KaldiRecognizer.assert_called_once_with(mock_model, 22050)
            
            # Verify duplicate filter configuration
            assert stt._duplicate_filter._cache_size == 5
            assert stt._duplicate_filter._duplicate_threshold_ms == 300

