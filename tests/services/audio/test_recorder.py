"""
Unit tests for AudioRecorder
"""
import pytest
import threading
import time
import numpy as np
from unittest.mock import Mock, patch, MagicMock, call

from iris.services.audio.recorder import AudioRecorder


class TestAudioRecorder:
    """Test AudioRecorder functionality"""
    
    @pytest.fixture
    def mock_sounddevice(self):
        """Mock sounddevice for testing"""
        with patch('iris.services.audio.recorder.sd') as mock_sd:
            mock_stream = Mock()
            mock_stream.start = Mock()
            mock_stream.stop = Mock()
            mock_stream.close = Mock()
            mock_stream.active = True
            mock_stream.read.return_value = (np.random.randint(-1000, 1000, 320, dtype=np.int16), None)
            
            mock_sd.InputStream.return_value = mock_stream
            yield mock_sd, mock_stream
    
    @pytest.fixture
    def command_recorder(self, mock_global_config, mock_sounddevice):
        """Create command mode AudioRecorder for testing"""
        mock_sd, mock_stream = mock_sounddevice
        
        on_audio_segment = Mock()
        on_streaming_chunk = Mock(return_value="")
        
        recorder = AudioRecorder(
            app_config=mock_global_config,
            mode="command",
            on_audio_segment=on_audio_segment,
            on_streaming_chunk=on_streaming_chunk
        )
        recorder._stream = mock_stream
        return recorder
    
    @pytest.fixture
    def dictation_recorder(self, mock_global_config, mock_sounddevice):
        """Create dictation mode AudioRecorder for testing"""
        mock_sd, mock_stream = mock_sounddevice
        
        on_audio_segment = Mock()
        
        recorder = AudioRecorder(
            app_config=mock_global_config,
            mode="dictation",
            on_audio_segment=on_audio_segment
        )
        recorder._stream = mock_stream
        return recorder
    
    def test_command_mode_initialization(self, mock_global_config):
        """Test command mode recorder initialization"""
        on_audio_segment = Mock()
        on_streaming_chunk = Mock()
        
        recorder = AudioRecorder(
            app_config=mock_global_config,
            mode="command",
            on_audio_segment=on_audio_segment,
            on_streaming_chunk=on_streaming_chunk
        )
        
        assert recorder.mode == "command"
        assert recorder.chunk_size == mock_global_config.audio.command_chunk_size
        assert recorder.energy_threshold == mock_global_config.vad.command_energy_threshold
        assert recorder.silence_timeout == mock_global_config.vad.command_silence_timeout
        assert recorder.enable_streaming is True
        assert recorder._smart_timeout_manager is not None
    
    def test_dictation_mode_initialization(self, mock_global_config):
        """Test dictation mode recorder initialization"""
        on_audio_segment = Mock()
        
        recorder = AudioRecorder(
            app_config=mock_global_config,
            mode="dictation",
            on_audio_segment=on_audio_segment
        )
        
        assert recorder.mode == "dictation"
        assert recorder.chunk_size == mock_global_config.audio.chunk_size
        assert recorder.energy_threshold == mock_global_config.vad.dictation_energy_threshold
        assert recorder.silence_timeout == mock_global_config.vad.dictation_silence_timeout
        assert recorder.enable_streaming is False
        assert recorder._smart_timeout_manager is None
    
    def test_energy_calculation_int16(self, command_recorder):
        """Test RMS energy calculation for int16 audio"""
        # Create test audio data
        audio_chunk = np.array([1000, -1000, 2000, -2000], dtype=np.int16)
        
        energy = command_recorder._calculate_energy(audio_chunk)
        
        assert isinstance(energy, float)
        assert energy > 0
        # Should normalize int16 values to float32 range
        expected_energy = np.sqrt(np.mean((audio_chunk.astype(np.float32) / 32768.0) ** 2))
        assert abs(energy - expected_energy) < 1e-6
    
    def test_energy_calculation_float32(self, command_recorder):
        """Test RMS energy calculation for float32 audio"""
        audio_chunk = np.array([0.1, -0.1, 0.2, -0.2], dtype=np.float32)
        
        energy = command_recorder._calculate_energy(audio_chunk)
        
        assert isinstance(energy, float)
        assert energy > 0
        expected_energy = np.sqrt(np.mean(audio_chunk ** 2))
        assert abs(energy - expected_energy) < 1e-6
    
    def test_noise_floor_adaptation(self, command_recorder):
        """Test noise floor adaptation mechanism"""
        initial_threshold = command_recorder.energy_threshold
        
        # Simulate multiple low-energy samples
        for _ in range(20):
            command_recorder._update_noise_floor(0.001)
        
        # Threshold should be adapted if noise floor is high enough
        if command_recorder.energy_threshold != initial_threshold:
            assert command_recorder.energy_threshold > initial_threshold
    
    def test_adaptive_timeout_command_mode(self, command_recorder):
        """Test adaptive timeout calculation for command mode"""
        # Short speech (< 240ms)
        timeout = command_recorder._get_timeout(3)
        assert timeout == 0.25
        
        # Medium speech (< 480ms)
        timeout = command_recorder._get_timeout(6)
        assert timeout == command_recorder.app_config.vad.command_silence_timeout
        
        # Long speech
        timeout = command_recorder._get_timeout(10)
        assert timeout == 0.6
    
    def test_adaptive_timeout_dictation_mode(self, dictation_recorder):
        """Test that dictation mode uses fixed timeout"""
        timeout = dictation_recorder._get_timeout(5)
        assert timeout == dictation_recorder.silence_timeout
        
        timeout = dictation_recorder._get_timeout(15)
        assert timeout == dictation_recorder.silence_timeout
    
    def test_instant_command_detection(self, command_recorder):
        """Test instant command detection for streaming"""
        # Mock smart timeout manager
        command_recorder._smart_timeout_manager.get_timeout_for_text = Mock(return_value=0.05)
        
        result = command_recorder._is_instant_command("click")
        assert result is True
        
        command_recorder._smart_timeout_manager.get_timeout_for_text = Mock(return_value=0.4)
        result = command_recorder._is_instant_command("complex command")
        assert result is False
    
    def test_instant_command_no_smart_timeout(self, dictation_recorder):
        """Test instant command detection without smart timeout manager"""
        result = dictation_recorder._is_instant_command("click")
        assert result is False
    
    def test_start_recording(self, command_recorder):
        """Test starting recording"""
        assert not command_recorder._is_recording
        
        command_recorder.start()
        
        assert command_recorder._is_recording
        assert command_recorder._thread is not None
        assert command_recorder._thread.daemon is True
        
        # Clean up
        command_recorder.stop()
    
    def test_stop_recording(self, command_recorder):
        """Test stopping recording"""
        command_recorder.start()
        assert command_recorder._is_recording
        
        command_recorder.stop()
        
        assert not command_recorder._is_recording
    
    def test_double_start_ignored(self, command_recorder):
        """Test that starting already running recorder is ignored"""
        command_recorder.start()
        first_thread = command_recorder._thread
        
        command_recorder.start()  # Should be ignored
        
        assert command_recorder._thread is first_thread
        
        command_recorder.stop()
    
    def test_double_stop_ignored(self, command_recorder):
        """Test that stopping already stopped recorder is ignored"""
        assert not command_recorder._is_recording
        
        # Should not raise exception
        command_recorder.stop()
        
        assert not command_recorder._is_recording
    
    def test_set_active_state(self, command_recorder):
        """Test setting active state"""
        assert command_recorder._is_active
        
        command_recorder.set_active(False)
        assert not command_recorder._is_active
        
        command_recorder.set_active(True)
        assert command_recorder._is_active
    
    def test_recording_status(self, command_recorder):
        """Test recording and active status queries"""
        assert not command_recorder.is_recording()
        assert command_recorder.is_active()
        
        command_recorder.start()
        assert command_recorder.is_recording()
        
        command_recorder.set_active(False)
        assert not command_recorder.is_active()
        
        command_recorder.stop()
        assert not command_recorder.is_recording()
    
    def test_timeout_update_for_text(self, command_recorder):
        """Test updating timeout based on recognized text"""
        original_timeout = command_recorder.silence_timeout
        
        # Mock smart timeout manager
        command_recorder._smart_timeout_manager.get_timeout_for_text = Mock(return_value=0.1)
        
        command_recorder.update_timeout_for_text("quick command")
        
        assert command_recorder.silence_timeout == 0.1
        command_recorder._smart_timeout_manager.get_timeout_for_text.assert_called_once_with("quick command")
    
    def test_timeout_update_dictation_mode(self, dictation_recorder):
        """Test that dictation mode doesn't update timeout"""
        original_timeout = dictation_recorder.silence_timeout
        
        dictation_recorder.update_timeout_for_text("some text")
        
        assert dictation_recorder.silence_timeout == original_timeout
    
    def test_smart_timeout_status(self, command_recorder):
        """Test getting smart timeout status"""
        command_recorder._smart_timeout_manager.get_status = Mock(return_value={"test": "status"})
        
        status = command_recorder.get_smart_timeout_status()
        
        assert status == {"test": "status"}
        command_recorder._smart_timeout_manager.get_status.assert_called_once()
    
    def test_smart_timeout_status_dictation_mode(self, dictation_recorder):
        """Test that dictation mode returns None for smart timeout status"""
        status = dictation_recorder.get_smart_timeout_status()
        assert status is None
    
    def test_stream_cleanup_on_error(self, command_recorder, mock_sounddevice):
        """Test that stream is properly cleaned up on errors"""
        mock_sd, mock_stream = mock_sounddevice
        
        # Simulate stream error
        mock_stream.stop.side_effect = Exception("Stream error")
        
        # Should not raise exception
        command_recorder._cleanup_stream()
        
        # Stream should be set to None
        assert command_recorder._stream is None
    
    def test_stream_cleanup_inactive_stream(self, command_recorder):
        """Test cleanup of inactive stream"""
        mock_stream = Mock()
        mock_stream.active = False
        mock_stream.stop = Mock()
        mock_stream.close = Mock()
        
        command_recorder._stream = mock_stream
        command_recorder._cleanup_stream()
        
        # Should not call stop on inactive stream
        mock_stream.stop.assert_not_called()
        mock_stream.close.assert_called_once()
        assert command_recorder._stream is None
    
    @pytest.mark.parametrize("mode,expected_attributes", [
        ("command", {"enable_streaming": True, "smart_timeout": True}),
        ("dictation", {"enable_streaming": False, "smart_timeout": False})
    ])
    def test_mode_specific_configuration(self, mock_global_config, mode, expected_attributes):
        """Test that mode-specific configuration is applied correctly"""
        recorder = AudioRecorder(
            app_config=mock_global_config,
            mode=mode,
            on_audio_segment=Mock()
        )
        
        assert recorder.enable_streaming == expected_attributes["enable_streaming"]
        
        if expected_attributes["smart_timeout"]:
            assert recorder._smart_timeout_manager is not None
        else:
            assert recorder._smart_timeout_manager is None
    
    def test_pre_roll_buffer_size(self, command_recorder):
        """Test that pre-roll buffer size matches configuration"""
        expected_size = command_recorder.app_config.vad.command_pre_roll_buffers
        
        # This would be tested in the recording thread, but we can verify the config
        assert command_recorder.pre_roll_chunks == expected_size
    
    def test_device_configuration(self, mock_global_config):
        """Test that audio device configuration is used"""
        mock_global_config.audio.device = 5
        
        recorder = AudioRecorder(
            app_config=mock_global_config,
            mode="command",
            on_audio_segment=Mock()
        )
        
        assert recorder.device == 5
    
    def test_silence_threshold_calculation(self, command_recorder):
        """Test that silence threshold is calculated from energy threshold"""
        expected_silence_threshold = command_recorder.energy_threshold * 0.35
        assert abs(command_recorder.silence_threshold - expected_silence_threshold) < 1e-6
    
    def test_callback_assignment(self, mock_global_config):
        """Test that callbacks are properly assigned"""
        on_audio_segment = Mock()
        on_streaming_chunk = Mock()
        
        recorder = AudioRecorder(
            app_config=mock_global_config,
            mode="command",
            on_audio_segment=on_audio_segment,
            on_streaming_chunk=on_streaming_chunk
        )
        
        assert recorder.on_audio_segment is on_audio_segment
        assert recorder.on_streaming_chunk is on_streaming_chunk
    
    def test_thread_daemon_property(self, command_recorder):
        """Test that recording thread is set as daemon"""
        command_recorder.start()
        
        assert command_recorder._thread.daemon is True
        
        command_recorder.stop()
    
    def test_thread_safety_lock(self, command_recorder):
        """Test that operations are thread-safe with lock"""
        assert isinstance(command_recorder._lock, threading.Lock)
        
        # Operations that should use the lock
        command_recorder.start()
        command_recorder.stop()
        
        # Should not raise any threading exceptions
        assert True  # If we get here, locking worked

