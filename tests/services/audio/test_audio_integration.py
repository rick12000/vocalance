"""
Integration tests for audio system end-to-end workflows

Tests the complete audio processing pipeline from recording to text recognition.
"""
import pytest
import asyncio
import threading
import time
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List

from iris.services.audio.simple_audio_service import SimpleAudioService
from iris.services.audio.stt_service import StreamlinedSpeechToTextService
from iris.services.audio.recorder import AudioRecorder
from iris.events.core_events import (
    CommandAudioSegmentReadyEvent,
    DictationAudioSegmentReadyEvent,
    AudioModeChangeRequestEvent
)
from iris.events.stt_events import CommandTextRecognizedEvent, DictationTextRecognizedEvent
from iris.events.dictation_events import DictationModeDisableOthersEvent


class TestAudioSystemIntegration:
    """Integration tests for complete audio processing workflows"""
    
    @pytest.fixture
    def mock_audio_pipeline(self, mock_event_bus, mock_global_config):
        """Create a complete mocked audio processing pipeline"""
        # Mock sounddevice
        with patch('iris.services.audio.recorder.sd') as mock_sd:
            mock_stream = Mock()
            mock_stream.start = Mock()
            mock_stream.stop = Mock()
            mock_stream.close = Mock()
            mock_stream.active = True
            
            # Generate realistic audio data
            def generate_audio_chunk():
                return np.random.randint(-1000, 1000, 320, dtype=np.int16), None
            
            mock_stream.read.side_effect = lambda size: generate_audio_chunk()
            mock_sd.InputStream.return_value = mock_stream
            
            # Create services
            audio_service = SimpleAudioService(mock_event_bus, mock_global_config)
            stt_service = StreamlinedSpeechToTextService(mock_event_bus, mock_global_config)
            
            # Mock STT engines
            with patch('iris.services.audio.stt_service.EnhancedVoskSTT') as mock_vosk:
                with patch('iris.services.audio.stt_service.WhisperSpeechToText') as mock_whisper:
                    mock_vosk_engine = Mock()
                    mock_whisper_engine = Mock()
                    mock_vosk.return_value = mock_vosk_engine
                    mock_whisper.return_value = mock_whisper_engine
                    
                    stt_service.initialize_engines()
                    
                    return {
                        'audio_service': audio_service,
                        'stt_service': stt_service,
                        'vosk_engine': mock_vosk_engine,
                        'whisper_engine': mock_whisper_engine,
                        'event_bus': mock_event_bus,
                        'stream': mock_stream
                    }
    
    @pytest.fixture
    def event_collector(self):
        """Collect events for verification"""
        events = []
        
        def collect_event(event):
            events.append(event)
        
        return events, collect_event
    
    def test_command_mode_end_to_end_workflow(self, mock_audio_pipeline, event_collector):
        """Test complete command processing workflow"""
        pipeline = mock_audio_pipeline
        events, collect_event = event_collector
        
        # Set up event collection
        pipeline['event_bus'].publish = AsyncMock(side_effect=collect_event)
        
        # Configure recognition result
        pipeline['vosk_engine'].recognize.return_value = "click"
        pipeline['stt_service']._duplicate_filter.is_duplicate = Mock(return_value=False)
        
        # Set up event subscriptions
        pipeline['stt_service'].setup_subscriptions()
        pipeline['audio_service'].init_listeners()
        
        # Start services
        pipeline['audio_service'].start_processing()
        
        # Simulate audio segment ready event
        audio_bytes = np.random.randint(-1000, 1000, 3200, dtype=np.int16).tobytes()
        event = CommandAudioSegmentReadyEvent(
            audio_bytes=audio_bytes,
            sample_rate=16000
        )
        
        # Process the event
        asyncio.run(pipeline['stt_service']._handle_command_audio_segment(event))
        
        # Verify the workflow
        pipeline['vosk_engine'].recognize.assert_called_once_with(audio_bytes, 16000)
        
        # Should have published recognition result
        published_events = [call[0][0] for call in pipeline['event_bus'].publish.call_args_list]
        text_events = [e for e in published_events if isinstance(e, CommandTextRecognizedEvent)]
        assert len(text_events) > 0
        assert text_events[0].text == "click"
        assert text_events[0].engine == "vosk"
        
        # Clean up
        pipeline['audio_service'].stop_processing()
    
    def test_dictation_mode_end_to_end_workflow(self, mock_audio_pipeline, event_collector):
        """Test complete dictation processing workflow"""
        pipeline = mock_audio_pipeline
        events, collect_event = event_collector
        
        # Set up event collection
        pipeline['event_bus'].publish = AsyncMock(side_effect=collect_event)
        
        # Configure recognition result
        pipeline['whisper_engine'].recognize.return_value = "hello world this is dictation"
        pipeline['stt_service']._duplicate_filter.is_duplicate = Mock(return_value=False)
        
        # Set up event subscriptions
        pipeline['stt_service'].setup_subscriptions()
        
        # Simulate dictation audio segment ready event
        audio_bytes = np.random.randint(-1000, 1000, 16000, dtype=np.int16).tobytes()
        event = DictationAudioSegmentReadyEvent(
            audio_bytes=audio_bytes,
            sample_rate=16000
        )
        
        # Process the event
        asyncio.run(pipeline['stt_service']._handle_dictation_audio_segment(event))
        
        # Verify the workflow
        pipeline['whisper_engine'].recognize.assert_called_once_with(audio_bytes, 16000)
        
        # Should have published dictation result
        published_events = [call[0][0] for call in pipeline['event_bus'].publish.call_args_list]
        text_events = [e for e in published_events if isinstance(e, DictationTextRecognizedEvent)]
        assert len(text_events) > 0
        assert text_events[0].text == "hello world this is dictation"
        assert text_events[0].engine == "whisper"
    
    @pytest.mark.asyncio
    async def test_mode_switching_workflow(self, mock_audio_pipeline):
        """Test switching between command and dictation modes"""
        pipeline = mock_audio_pipeline
        audio_service = pipeline['audio_service']
        stt_service = pipeline['stt_service']
        
        # Set up subscriptions
        stt_service.setup_subscriptions()
        audio_service.init_listeners()
        
        # Start in command mode
        assert not stt_service._dictation_active
        assert not audio_service._is_dictation_mode
        
        # Switch to dictation mode
        dictation_event = DictationModeDisableOthersEvent(dictation_mode_active=True)
        await stt_service._handle_dictation_mode_change(dictation_event)
        
        audio_mode_event = AudioModeChangeRequestEvent(mode="dictation", reason="user_request")
        await audio_service._handle_audio_mode_change_request(audio_mode_event)
        
        # Verify dictation mode is active
        assert stt_service._dictation_active
        assert audio_service._is_dictation_mode
        
        # In dictation mode, both recorders should be active
        audio_service._command_recorder.set_active.assert_called_with(True)
        audio_service._dictation_recorder.set_active.assert_called_with(True)
        
        # Switch back to command mode
        dictation_event = DictationModeDisableOthersEvent(dictation_mode_active=False)
        await stt_service._handle_dictation_mode_change(dictation_event)
        
        audio_mode_event = AudioModeChangeRequestEvent(mode="command", reason="dictation_ended")
        await audio_service._handle_audio_mode_change_request(audio_mode_event)
        
        # Verify command mode is active
        assert not stt_service._dictation_active
        assert not audio_service._is_dictation_mode
    
    def test_amber_detection_during_dictation(self, mock_audio_pipeline, event_collector):
        """Test that amber commands are detected during dictation mode"""
        pipeline = mock_audio_pipeline
        events, collect_event = event_collector
        
        # Set up event collection
        pipeline['event_bus'].publish = AsyncMock(side_effect=collect_event)
        
        # Set dictation mode active
        pipeline['stt_service']._dictation_active = True
        
        # Configure amber detection
        pipeline['vosk_engine'].recognize.return_value = "amber"
        
        # Set up subscriptions
        pipeline['stt_service'].setup_subscriptions()
        
        # Simulate command audio during dictation
        audio_bytes = np.random.randint(-1000, 1000, 3200, dtype=np.int16).tobytes()
        event = CommandAudioSegmentReadyEvent(
            audio_bytes=audio_bytes,
            sample_rate=16000
        )
        
        # Process the event
        asyncio.run(pipeline['stt_service']._handle_command_audio_segment(event))
        
        # Should have processed amber command even in dictation mode
        pipeline['vosk_engine'].recognize.assert_called_once_with(audio_bytes, 16000)
        
        # Should have published the amber command
        published_events = [call[0][0] for call in pipeline['event_bus'].publish.call_args_list]
        text_events = [e for e in published_events if isinstance(e, CommandTextRecognizedEvent)]
        assert len(text_events) > 0
        assert text_events[0].text == "amber"
    
    def test_non_amber_command_ignored_during_dictation(self, mock_audio_pipeline, event_collector):
        """Test that non-amber commands are ignored during dictation mode"""
        pipeline = mock_audio_pipeline
        events, collect_event = event_collector
        
        # Set up event collection
        pipeline['event_bus'].publish = AsyncMock(side_effect=collect_event)
        
        # Set dictation mode active
        pipeline['stt_service']._dictation_active = True
        
        # Configure non-amber command
        pipeline['vosk_engine'].recognize.return_value = "click"
        
        # Set up subscriptions
        pipeline['stt_service'].setup_subscriptions()
        
        # Simulate command audio during dictation
        audio_bytes = np.random.randint(-1000, 1000, 3200, dtype=np.int16).tobytes()
        event = CommandAudioSegmentReadyEvent(
            audio_bytes=audio_bytes,
            sample_rate=16000
        )
        
        # Process the event
        asyncio.run(pipeline['stt_service']._handle_command_audio_segment(event))
        
        # Should have processed the audio but not published the result
        pipeline['vosk_engine'].recognize.assert_called_once_with(audio_bytes, 16000)
        
        # Should NOT have published any text recognition events
        published_events = [call[0][0] for call in pipeline['event_bus'].publish.call_args_list if call[0]]
        text_events = [e for e in published_events if isinstance(e, CommandTextRecognizedEvent)]
        assert len(text_events) == 0
    
    def test_duplicate_filtering_integration(self, mock_audio_pipeline, event_collector):
        """Test duplicate filtering in the complete workflow"""
        pipeline = mock_audio_pipeline
        events, collect_event = event_collector
        
        # Set up event collection
        pipeline['event_bus'].publish = AsyncMock(side_effect=collect_event)
        
        # Configure recognition result
        pipeline['vosk_engine'].recognize.return_value = "duplicate command"
        
        # First call: not duplicate
        pipeline['stt_service']._duplicate_filter.is_duplicate = Mock(return_value=False)
        
        # Set up subscriptions
        pipeline['stt_service'].setup_subscriptions()
        
        # Process first event
        audio_bytes = np.random.randint(-1000, 1000, 3200, dtype=np.int16).tobytes()
        event = CommandAudioSegmentReadyEvent(
            audio_bytes=audio_bytes,
            sample_rate=16000
        )
        
        asyncio.run(pipeline['stt_service']._handle_command_audio_segment(event))
        
        # Should publish first result
        published_events = [call[0][0] for call in pipeline['event_bus'].publish.call_args_list]
        text_events = [e for e in published_events if isinstance(e, CommandTextRecognizedEvent)]
        initial_count = len(text_events)
        assert initial_count > 0
        
        # Reset mock for second call
        pipeline['event_bus'].publish.reset_mock()
        
        # Second call: is duplicate
        pipeline['stt_service']._duplicate_filter.is_duplicate = Mock(return_value=True)
        
        # Process second event (duplicate)
        asyncio.run(pipeline['stt_service']._handle_command_audio_segment(event))
        
        # Should NOT publish duplicate result
        published_events = [call[0][0] for call in pipeline['event_bus'].publish.call_args_list if call[0]]
        text_events = [e for e in published_events if isinstance(e, CommandTextRecognizedEvent)]
        assert len(text_events) == 0  # No new text events should be published
    
    def test_error_handling_integration(self, mock_audio_pipeline):
        """Test error handling throughout the pipeline"""
        pipeline = mock_audio_pipeline
        
        # Configure engine to raise exception
        pipeline['vosk_engine'].recognize.side_effect = Exception("Recognition error")
        
        # Set up subscriptions
        pipeline['stt_service'].setup_subscriptions()
        
        # Process event with error
        audio_bytes = np.random.randint(-1000, 1000, 3200, dtype=np.int16).tobytes()
        event = CommandAudioSegmentReadyEvent(
            audio_bytes=audio_bytes,
            sample_rate=16000
        )
        
        # Should not raise exception
        asyncio.run(pipeline['stt_service']._handle_command_audio_segment(event))
        
        # Engine should have been called despite error
        pipeline['vosk_engine'].recognize.assert_called_once()
    
    def test_concurrent_processing_integration(self, mock_audio_pipeline):
        """Test concurrent processing of multiple audio segments"""
        pipeline = mock_audio_pipeline
        
        # Configure different results for different calls
        results = ["first", "second", "third"]
        pipeline['vosk_engine'].recognize.side_effect = results
        pipeline['stt_service']._duplicate_filter.is_duplicate = Mock(return_value=False)
        
        # Set up subscriptions
        pipeline['stt_service'].setup_subscriptions()
        
        # Create multiple events
        events = []
        for i in range(3):
            audio_bytes = np.random.randint(-1000, 1000, 3200, dtype=np.int16).tobytes()
            event = CommandAudioSegmentReadyEvent(
                audio_bytes=audio_bytes,
                sample_rate=16000
            )
            events.append(event)
        
        # Process events concurrently
        async def process_all():
            tasks = []
            for event in events:
                task = asyncio.create_task(
                    pipeline['stt_service']._handle_command_audio_segment(event)
                )
                tasks.append(task)
            
            await asyncio.gather(*tasks)
        
        asyncio.run(process_all())
        
        # All events should have been processed
        assert pipeline['vosk_engine'].recognize.call_count == 3
    
    def test_service_lifecycle_integration(self, mock_audio_pipeline):
        """Test complete service lifecycle"""
        pipeline = mock_audio_pipeline
        audio_service = pipeline['audio_service']
        stt_service = pipeline['stt_service']
        
        # Start services
        audio_service.start_processing()
        
        # Verify recorders are started
        audio_service._command_recorder.start.assert_called_once()
        audio_service._dictation_recorder.start.assert_called_once()
        
        # Get status
        audio_status = audio_service.get_status()
        stt_status = stt_service.get_status()
        
        assert isinstance(audio_status, dict)
        assert isinstance(stt_status, dict)
        assert "dictation_mode" in audio_status
        assert "engines_initialized" in stt_status
        
        # Shutdown services
        asyncio.run(audio_service.shutdown())
        
        # Verify proper shutdown
        audio_service._command_recorder.stop.assert_called()
        audio_service._dictation_recorder.stop.assert_called()
    
    def test_streaming_integration(self, mock_audio_pipeline):
        """Test streaming recognition integration"""
        pipeline = mock_audio_pipeline
        audio_service = pipeline['audio_service']
        stt_service = pipeline['stt_service']
        
        # Connect services for streaming
        stt_service.connect_to_audio_service(audio_service)
        
        # Should have set up streaming connection
        assert hasattr(audio_service, '_streaming_stt_engine')
        assert audio_service._streaming_stt_engine == stt_service.vosk_engine
        
        # Test streaming callback
        audio_bytes = np.random.randint(-1000, 1000, 3200, dtype=np.int16).tobytes()
        
        # Mock streaming recognition
        stt_service.vosk_engine.recognize_streaming = Mock(return_value="stream result")
        
        result = audio_service._on_command_streaming_chunk(audio_bytes, False)
        
        assert result == "stream result"
        stt_service.vosk_engine.recognize_streaming.assert_called_once_with(audio_bytes, False)
    
    @pytest.mark.asyncio
    async def test_full_pipeline_realistic_scenario(self, mock_audio_pipeline):
        """Test a realistic end-to-end scenario"""
        pipeline = mock_audio_pipeline
        
        # Configure realistic responses
        pipeline['vosk_engine'].recognize.side_effect = ["", "click", "right click"]
        pipeline['whisper_engine'].recognize.return_value = "hello world"
        pipeline['stt_service']._duplicate_filter.is_duplicate = Mock(return_value=False)
        
        # Set up services
        pipeline['stt_service'].setup_subscriptions()
        pipeline['audio_service'].init_listeners()
        pipeline['stt_service'].connect_to_audio_service(pipeline['audio_service'])
        
        # Start processing
        pipeline['audio_service'].start_processing()
        
        # Scenario 1: No speech detected -> sound recognition
        audio_bytes = np.random.randint(-100, 100, 1600, dtype=np.int16).tobytes()
        event = CommandAudioSegmentReadyEvent(audio_bytes=audio_bytes, sample_rate=16000)
        
        with patch.object(pipeline['stt_service'], '_publish_sound_recognition_event') as mock_sound:
            await pipeline['stt_service']._handle_command_audio_segment(event)
            mock_sound.assert_called_once()
        
        # Scenario 2: Command recognized
        audio_bytes = np.random.randint(-1000, 1000, 3200, dtype=np.int16).tobytes()
        event = CommandAudioSegmentReadyEvent(audio_bytes=audio_bytes, sample_rate=16000)
        
        with patch.object(pipeline['stt_service'], '_publish_recognition_result') as mock_publish:
            await pipeline['stt_service']._handle_command_audio_segment(event)
            mock_publish.assert_called_once_with("click", pytest.approx(0, abs=1000), "vosk", pipeline['stt_service'].STTMode.COMMAND)
        
        # Scenario 3: Switch to dictation and process
        await pipeline['stt_service']._handle_dictation_mode_change(
            DictationModeDisableOthersEvent(dictation_mode_active=True)
        )
        
        dictation_bytes = np.random.randint(-1000, 1000, 16000, dtype=np.int16).tobytes()
        dictation_event = DictationAudioSegmentReadyEvent(audio_bytes=dictation_bytes, sample_rate=16000)
        
        with patch.object(pipeline['stt_service'], '_publish_recognition_result') as mock_publish:
            await pipeline['stt_service']._handle_dictation_audio_segment(dictation_event)
            mock_publish.assert_called_once_with("hello world", pytest.approx(0, abs=1000), "whisper", pipeline['stt_service'].STTMode.DICTATION)
        
        # Clean up
        pipeline['audio_service'].stop_processing()

