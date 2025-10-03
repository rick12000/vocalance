"""
Unit tests for SimpleAudioService
"""
import pytest
import asyncio
import threading
from unittest.mock import Mock, patch, MagicMock

from iris.services.audio.simple_audio_service import SimpleAudioService
from iris.events.core_events import (
    CommandAudioSegmentReadyEvent,
    DictationAudioSegmentReadyEvent,
    StartRecordingCommand,
    StopRecordingCommand,
    AudioModeChangeRequestEvent
)


class TestSimpleAudioService:
    """Test SimpleAudioService functionality"""
    
    @pytest.fixture
    def mock_recorder(self):
        """Mock AudioRecorder for testing"""
        recorder = Mock()
        recorder.start = Mock()
        recorder.stop = Mock()
        recorder.set_active = Mock()
        recorder.is_active = Mock(return_value=True)
        recorder.is_recording = Mock(return_value=True)
        return recorder
    
    @pytest.fixture
    def audio_service(self, mock_event_bus, mock_global_config):
        """Create SimpleAudioService instance for testing"""
        with patch('iris.services.audio.simple_audio_service.AudioRecorder') as mock_recorder_class:
            mock_recorder_class.return_value = Mock()
            service = SimpleAudioService(mock_event_bus, mock_global_config)
            return service
    
    def test_initialization(self, mock_event_bus, mock_global_config):
        """Test service initialization"""
        with patch('iris.services.audio.simple_audio_service.AudioRecorder') as mock_recorder_class:
            mock_command_recorder = Mock()
            mock_dictation_recorder = Mock()
            mock_recorder_class.side_effect = [mock_command_recorder, mock_dictation_recorder]
            
            service = SimpleAudioService(mock_event_bus, mock_global_config)
            
            assert service._event_bus == mock_event_bus
            assert service._config == mock_global_config
            assert not service._is_dictation_mode
            assert service._command_recorder == mock_command_recorder
            assert service._dictation_recorder == mock_dictation_recorder
            
            # Verify recorder initialization calls
            assert mock_recorder_class.call_count == 2
    
    def test_command_audio_segment_handling(self, audio_service, mock_event_bus, test_audio_bytes):
        """Test command audio segment processing"""
        audio_service._on_command_audio_segment(test_audio_bytes)
        
        # Should publish event via async mechanism
        assert hasattr(audio_service, '_publish_audio_event')
    
    def test_dictation_audio_segment_handling(self, audio_service, mock_event_bus, test_audio_bytes):
        """Test dictation audio segment processing"""
        audio_service._on_dictation_audio_segment(test_audio_bytes)
        
        # Should publish event via async mechanism
        assert hasattr(audio_service, '_publish_audio_event')
    
    def test_streaming_chunk_processing(self, audio_service, test_audio_bytes):
        """Test streaming chunk processing for early termination"""
        # Without streaming STT engine
        result = audio_service._on_command_streaming_chunk(test_audio_bytes, False)
        assert result == ""
        
        # With streaming STT engine
        mock_stt_engine = Mock()
        mock_stt_engine.recognize_streaming = Mock(return_value="click")
        audio_service._streaming_stt_engine = mock_stt_engine
        
        result = audio_service._on_command_streaming_chunk(test_audio_bytes, True)
        assert result == "click"
        mock_stt_engine.recognize_streaming.assert_called_once_with(test_audio_bytes, True)
    
    @pytest.mark.asyncio
    async def test_audio_mode_change_to_dictation(self, audio_service):
        """Test switching to dictation mode"""
        event = AudioModeChangeRequestEvent(mode="dictation", reason="user_request")
        
        await audio_service._handle_audio_mode_change_request(event)
        
        assert audio_service._is_dictation_mode
        audio_service._command_recorder.set_active.assert_called_with(True)
        audio_service._dictation_recorder.set_active.assert_called_with(True)
    
    @pytest.mark.asyncio
    async def test_audio_mode_change_to_command(self, audio_service):
        """Test switching to command mode"""
        # Start in dictation mode
        audio_service._is_dictation_mode = True
        
        event = AudioModeChangeRequestEvent(mode="command", reason="dictation_ended")
        
        await audio_service._handle_audio_mode_change_request(event)
        
        assert not audio_service._is_dictation_mode
        audio_service._command_recorder.set_active.assert_called_with(True)
        audio_service._dictation_recorder.set_active.assert_called_with(False)
    
    def test_start_processing_command_mode(self, audio_service):
        """Test starting audio processing in command mode"""
        audio_service._is_dictation_mode = False
        
        audio_service.start_processing()
        
        audio_service._command_recorder.start.assert_called_once()
        audio_service._dictation_recorder.start.assert_called_once()
        audio_service._command_recorder.set_active.assert_called_with(True)
        audio_service._dictation_recorder.set_active.assert_called_with(False)
    
    def test_start_processing_dictation_mode(self, audio_service):
        """Test starting audio processing in dictation mode"""
        audio_service._is_dictation_mode = True
        
        audio_service.start_processing()
        
        audio_service._command_recorder.start.assert_called_once()
        audio_service._dictation_recorder.start.assert_called_once()
        audio_service._command_recorder.set_active.assert_called_with(True)
        audio_service._dictation_recorder.set_active.assert_called_with(True)
    
    def test_stop_processing(self, audio_service):
        """Test stopping audio processing"""
        audio_service.stop_processing()
        
        audio_service._command_recorder.stop.assert_called_once()
        audio_service._dictation_recorder.stop.assert_called_once()
    
    def test_get_status(self, audio_service):
        """Test getting service status"""
        status = audio_service.get_status()
        
        expected_keys = {
            "command_recorder_active",
            "dictation_recorder_active", 
            "command_recording",
            "dictation_recording",
            "dictation_mode"
        }
        
        assert set(status.keys()) == expected_keys
        assert isinstance(status["dictation_mode"], bool)
    
    @pytest.mark.asyncio
    async def test_shutdown(self, audio_service):
        """Test service shutdown"""
        await audio_service.shutdown()
        
        audio_service._command_recorder.stop.assert_called_once()
        audio_service._dictation_recorder.stop.assert_called_once()
    
    def test_set_streaming_stt_engine(self, audio_service):
        """Test setting streaming STT engine"""
        mock_engine = Mock()
        
        audio_service.set_streaming_stt_engine(mock_engine)
        
        assert audio_service._streaming_stt_engine == mock_engine
    
    def test_event_subscriptions(self, audio_service, mock_event_bus):
        """Test event subscription setup"""
        audio_service.init_listeners()
        
        # Verify subscriptions were made
        expected_events = [
            StartRecordingCommand,
            StopRecordingCommand, 
            AudioModeChangeRequestEvent
        ]
        
        for event_type in expected_events:
            mock_event_bus.subscribe.assert_any_call(event_type, audio_service._handle_start_recording)

