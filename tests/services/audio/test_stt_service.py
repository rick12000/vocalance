"""
Unit tests for StreamlinedSpeechToTextService
"""
import pytest
import asyncio
import threading
from unittest.mock import Mock, patch, AsyncMock

from iris.services.audio.stt_service import StreamlinedSpeechToTextService, STTMode
from iris.events.core_events import CommandAudioSegmentReadyEvent, DictationAudioSegmentReadyEvent
from iris.events.stt_events import CommandTextRecognizedEvent, DictationTextRecognizedEvent
from iris.events.dictation_events import DictationModeDisableOthersEvent


class TestStreamlinedSpeechToTextService:
    """Test StreamlinedSpeechToTextService functionality"""
    
    @pytest.fixture
    def stt_service(self, mock_event_bus, mock_global_config):
        """Create STT service instance for testing"""
        service = StreamlinedSpeechToTextService(mock_event_bus, mock_global_config)
        return service
    
    @pytest.fixture
    def initialized_stt_service(self, stt_service, mock_vosk_model, mock_whisper_model):
        """Create initialized STT service with mocked engines"""
        with patch('iris.services.audio.stt_service.EnhancedVoskSTT') as mock_vosk_class:
            with patch('iris.services.audio.stt_service.WhisperSpeechToText') as mock_whisper_class:
                mock_vosk_class.return_value = Mock()
                mock_whisper_class.return_value = Mock()
                
                stt_service.initialize_engines()
                
                return stt_service
    
    def test_initialization(self, mock_event_bus, mock_global_config):
        """Test STT service initialization"""
        service = StreamlinedSpeechToTextService(mock_event_bus, mock_global_config)
        
        assert service.event_bus == mock_event_bus
        assert service.config == mock_global_config
        assert service.stt_config == mock_global_config.stt
        assert not service._dictation_active
        assert isinstance(service._processing_lock, threading.RLock)
        assert service.vosk_engine is None
        assert service.whisper_engine is None
        assert not service._engines_initialized
    
    def test_engine_initialization(self, stt_service):
        """Test STT engine initialization"""
        with patch('iris.services.audio.stt_service.EnhancedVoskSTT') as mock_vosk_class:
            with patch('iris.services.audio.stt_service.WhisperSpeechToText') as mock_whisper_class:
                mock_vosk_engine = Mock()
                mock_whisper_engine = Mock()
                mock_vosk_class.return_value = mock_vosk_engine
                mock_whisper_class.return_value = mock_whisper_engine
                
                stt_service.initialize_engines()
                
                assert stt_service.vosk_engine == mock_vosk_engine
                assert stt_service.whisper_engine == mock_whisper_engine
                assert stt_service._engines_initialized
                
                # Verify initialization parameters
                mock_vosk_class.assert_called_once_with(
                    model_path=stt_service.config.model_paths.vosk_model,
                    sample_rate=stt_service.stt_config.sample_rate,
                    config=stt_service.config
                )
                
                mock_whisper_class.assert_called_once_with(
                    model_name=stt_service.stt_config.whisper_model,
                    device=stt_service.stt_config.whisper_device,
                    sample_rate=stt_service.stt_config.sample_rate,
                    config=stt_service.stt_config
                )
    
    def test_double_initialization_ignored(self, stt_service):
        """Test that double initialization is ignored"""
        with patch('iris.services.audio.stt_service.EnhancedVoskSTT') as mock_vosk_class:
            with patch('iris.services.audio.stt_service.WhisperSpeechToText') as mock_whisper_class:
                mock_vosk_class.return_value = Mock()
                mock_whisper_class.return_value = Mock()
                
                stt_service.initialize_engines()
                first_vosk = stt_service.vosk_engine
                first_whisper = stt_service.whisper_engine
                
                stt_service.initialize_engines()  # Should be ignored
                
                assert stt_service.vosk_engine is first_vosk
                assert stt_service.whisper_engine is first_whisper
                assert mock_vosk_class.call_count == 1
                assert mock_whisper_class.call_count == 1
    
    def test_event_subscriptions(self, stt_service, mock_event_bus):
        """Test event subscription setup"""
        stt_service.setup_subscriptions()
        
        expected_subscriptions = [
            (CommandAudioSegmentReadyEvent, stt_service._handle_command_audio_segment),
            (DictationAudioSegmentReadyEvent, stt_service._handle_dictation_audio_segment),
            (DictationModeDisableOthersEvent, stt_service._handle_dictation_mode_change)
        ]
        
        for event_type, handler in expected_subscriptions:
            mock_event_bus.subscribe.assert_any_call(event_type, handler)
    
    @pytest.mark.asyncio
    async def test_command_audio_processing_normal_mode(self, initialized_stt_service, test_audio_bytes):
        """Test command audio processing in normal mode"""
        service = initialized_stt_service
        service._dictation_active = False
        service.vosk_engine.recognize = Mock(return_value="test command")
        service._duplicate_filter.is_duplicate = Mock(return_value=False)
        
        event = CommandAudioSegmentReadyEvent(
            audio_bytes=test_audio_bytes,
            sample_rate=16000
        )
        
        with patch.object(service, '_publish_recognition_result') as mock_publish:
            await service._handle_command_audio_segment(event)
            
            service.vosk_engine.recognize.assert_called_once_with(test_audio_bytes, 16000)
            mock_publish.assert_called_once_with("test command", pytest.approx(0, abs=1000), "vosk", STTMode.COMMAND)
    
    @pytest.mark.asyncio
    async def test_command_audio_processing_dictation_mode(self, initialized_stt_service, test_audio_bytes):
        """Test command audio processing in dictation mode (amber detection only)"""
        service = initialized_stt_service
        service._dictation_active = True
        service.vosk_engine.recognize = Mock(return_value="amber")
        
        event = CommandAudioSegmentReadyEvent(
            audio_bytes=test_audio_bytes,
            sample_rate=16000
        )
        
        with patch.object(service, '_publish_recognition_result') as mock_publish:
            await service._handle_command_audio_segment(event)
            
            service.vosk_engine.recognize.assert_called_once_with(test_audio_bytes, 16000)
            mock_publish.assert_called_once_with("amber", 0, "vosk", STTMode.COMMAND)
    
    @pytest.mark.asyncio
    async def test_command_audio_processing_dictation_mode_no_amber(self, initialized_stt_service, test_audio_bytes):
        """Test command audio processing in dictation mode without amber trigger"""
        service = initialized_stt_service
        service._dictation_active = True
        service.vosk_engine.recognize = Mock(return_value="regular command")
        
        event = CommandAudioSegmentReadyEvent(
            audio_bytes=test_audio_bytes,
            sample_rate=16000
        )
        
        with patch.object(service, '_publish_recognition_result') as mock_publish:
            await service._handle_command_audio_segment(event)
            
            service.vosk_engine.recognize.assert_called_once_with(test_audio_bytes, 16000)
            mock_publish.assert_not_called()  # Should not publish non-amber commands
    
    @pytest.mark.asyncio
    async def test_dictation_audio_processing(self, initialized_stt_service, test_audio_bytes):
        """Test dictation audio processing"""
        service = initialized_stt_service
        service.whisper_engine.recognize = Mock(return_value="dictated text")
        service._duplicate_filter.is_duplicate = Mock(return_value=False)
        
        event = DictationAudioSegmentReadyEvent(
            audio_bytes=test_audio_bytes,
            sample_rate=16000
        )
        
        with patch.object(service, '_publish_recognition_result') as mock_publish:
            await service._handle_dictation_audio_segment(event)
            
            service.whisper_engine.recognize.assert_called_once_with(test_audio_bytes, 16000)
            mock_publish.assert_called_once_with("dictated text", pytest.approx(0, abs=1000), "whisper", STTMode.DICTATION)
    
    @pytest.mark.asyncio
    async def test_sound_recognition_fallback(self, initialized_stt_service, test_audio_bytes):
        """Test sound recognition fallback when no speech detected"""
        service = initialized_stt_service
        service._dictation_active = False
        service.vosk_engine.recognize = Mock(return_value="")  # No speech detected
        
        event = CommandAudioSegmentReadyEvent(
            audio_bytes=test_audio_bytes,
            sample_rate=16000
        )
        
        with patch.object(service, '_publish_sound_recognition_event') as mock_sound_publish:
            await service._handle_command_audio_segment(event)
            
            mock_sound_publish.assert_called_once_with(test_audio_bytes, 16000)
    
    @pytest.mark.asyncio
    async def test_duplicate_filtering(self, initialized_stt_service, test_audio_bytes):
        """Test duplicate text filtering"""
        service = initialized_stt_service
        service._dictation_active = False
        service.vosk_engine.recognize = Mock(return_value="duplicate text")
        service._duplicate_filter.is_duplicate = Mock(return_value=True)
        
        event = CommandAudioSegmentReadyEvent(
            audio_bytes=test_audio_bytes,
            sample_rate=16000
        )
        
        with patch.object(service, '_publish_recognition_result') as mock_publish:
            await service._handle_command_audio_segment(event)
            
            service._duplicate_filter.is_duplicate.assert_called_once_with("duplicate text")
            mock_publish.assert_not_called()
    
    @pytest.mark.parametrize("text,expected", [
        ("amber", True),
        ("stop", True),
        ("end", True),
        ("AMBER", True),  # Case insensitive
        ("hello amber", True),  # Contains trigger word
        ("regular command", False),
        ("", False),
        (None, False)
    ])
    def test_amber_trigger_detection(self, initialized_stt_service, text, expected):
        """Test amber trigger word detection"""
        result = initialized_stt_service._is_amber_trigger(text)
        assert result == expected
    
    @pytest.mark.asyncio
    async def test_dictation_mode_change_to_active(self, initialized_stt_service):
        """Test changing to dictation mode"""
        service = initialized_stt_service
        assert not service._dictation_active
        
        event = DictationModeDisableOthersEvent(dictation_mode_active=True)
        
        await service._handle_dictation_mode_change(event)
        
        assert service._dictation_active
    
    @pytest.mark.asyncio
    async def test_dictation_mode_change_to_inactive(self, initialized_stt_service):
        """Test changing from dictation mode to command mode"""
        service = initialized_stt_service
        service._dictation_active = True
        
        event = DictationModeDisableOthersEvent(dictation_mode_active=False)
        
        await service._handle_dictation_mode_change(event)
        
        assert not service._dictation_active
    
    @pytest.mark.asyncio
    async def test_engines_not_initialized_error_handling(self, stt_service, test_audio_bytes):
        """Test error handling when engines are not initialized"""
        event = CommandAudioSegmentReadyEvent(
            audio_bytes=test_audio_bytes,
            sample_rate=16000
        )
        
        # Should not raise exception
        await stt_service._handle_command_audio_segment(event)
        
        # Should handle gracefully without crashing
        assert not stt_service._engines_initialized
    
    def test_get_status(self, initialized_stt_service):
        """Test status reporting"""
        service = initialized_stt_service
        service._dictation_active = True
        service._last_recognized_text = "test text for status"
        
        status = service.get_status()
        
        expected_keys = {
            "dictation_active",
            "engines_initialized",
            "vosk_initialized",
            "whisper_initialized",
            "last_recognized_text"
        }
        
        assert set(status.keys()) == expected_keys
        assert status["dictation_active"] is True
        assert status["engines_initialized"] is True
        assert status["vosk_initialized"] is True
        assert status["whisper_initialized"] is True
        assert "test text" in status["last_recognized_text"]
    
    def test_connect_to_audio_service(self, initialized_stt_service):
        """Test connecting to audio service for streaming"""
        service = initialized_stt_service
        mock_audio_service = Mock()
        mock_audio_service.set_streaming_stt_engine = Mock()
        
        service.connect_to_audio_service(mock_audio_service)
        
        mock_audio_service.set_streaming_stt_engine.assert_called_once_with(service.vosk_engine)
    
    def test_connect_to_audio_service_no_vosk(self, stt_service):
        """Test connecting to audio service without Vosk engine"""
        mock_audio_service = Mock()
        mock_audio_service.set_streaming_stt_engine = Mock()
        
        # Should not raise exception
        stt_service.connect_to_audio_service(mock_audio_service)
        
        mock_audio_service.set_streaming_stt_engine.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_shutdown(self, initialized_stt_service):
        """Test service shutdown"""
        service = initialized_stt_service
        service.vosk_engine.shutdown = AsyncMock()
        service.whisper_engine.shutdown = AsyncMock()
        
        await service.shutdown()
        
        service.vosk_engine.shutdown.assert_called_once()
        service.whisper_engine.shutdown.assert_called_once()
        assert service.vosk_engine is None
        assert service.whisper_engine is None
    
    @pytest.mark.asyncio
    async def test_shutdown_error_handling(self, initialized_stt_service):
        """Test shutdown error handling"""
        service = initialized_stt_service
        service.vosk_engine.shutdown = AsyncMock(side_effect=Exception("Shutdown error"))
        service.whisper_engine.shutdown = AsyncMock()
        
        # Should not raise exception
        await service.shutdown()
        
        # Should still attempt to shutdown whisper
        service.whisper_engine.shutdown.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_recognition_result_publishing(self, initialized_stt_service):
        """Test recognition result publishing with correct event types"""
        service = initialized_stt_service
        service.event_bus.publish = AsyncMock()
        
        # Test command mode publishing
        await service._publish_recognition_result("test command", 100.0, "vosk", STTMode.COMMAND)
        
        published_event = service.event_bus.publish.call_args[0][0]
        assert isinstance(published_event, CommandTextRecognizedEvent)
        assert published_event.text == "test command"
        assert published_event.processing_time_ms == 100.0
        assert published_event.engine == "vosk"
        assert published_event.mode == "command"
        
        # Test dictation mode publishing
        await service._publish_recognition_result("dictated text", 200.0, "whisper", STTMode.DICTATION)
        
        published_event = service.event_bus.publish.call_args[0][0]
        assert isinstance(published_event, DictationTextRecognizedEvent)
        assert published_event.text == "dictated text"
        assert published_event.processing_time_ms == 200.0
        assert published_event.engine == "whisper"
        assert published_event.mode == "dictation"
    
    @pytest.mark.asyncio
    async def test_processing_events_published(self, initialized_stt_service, test_audio_bytes):
        """Test that processing start/complete events are published"""
        service = initialized_stt_service
        service._dictation_active = False
        service.vosk_engine.recognize = Mock(return_value="test")
        service._duplicate_filter.is_duplicate = Mock(return_value=False)
        service.event_bus.publish = AsyncMock()
        
        event = CommandAudioSegmentReadyEvent(
            audio_bytes=test_audio_bytes,
            sample_rate=16000
        )
        
        await service._handle_command_audio_segment(event)
        
        # Should publish processing started and completed events
        published_calls = service.event_bus.publish.call_args_list
        assert len(published_calls) >= 3  # Started, recognition result, completed
    
    def test_thread_safety(self, initialized_stt_service):
        """Test thread safety of mode changes"""
        service = initialized_stt_service
        
        def change_mode():
            event = DictationModeDisableOthersEvent(dictation_mode_active=True)
            asyncio.run(service._handle_dictation_mode_change(event))
        
        # Run multiple mode changes concurrently
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=change_mode)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should end up in dictation mode
        assert service._dictation_active

