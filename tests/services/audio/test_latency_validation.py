"""
Latency validation tests for command system

Tests STT processing times for various command types to validate performance requirements.
"""
import pytest
import time
import numpy as np
import asyncio
from unittest.mock import Mock, patch
from typing import Dict, List, Tuple

from iris.services.audio.stt_service import StreamlinedSpeechToTextService
from iris.services.audio.vosk_stt import EnhancedVoskSTT
from iris.services.audio.whisper_stt import WhisperSpeechToText
from iris.events.core_events import CommandAudioSegmentReadyEvent, DictationAudioSegmentReadyEvent


class TestLatencyValidation:
    """Test latency requirements for command processing"""
    
    # Latency requirements (in milliseconds)
    INSTANT_COMMAND_LATENCY_MS = 50    # Single words like "click"
    QUICK_COMMAND_LATENCY_MS = 150     # Multi-word commands like "right click"  
    AMBIGUOUS_COMMAND_LATENCY_MS = 400 # Ambiguous commands like "down"
    DICTATION_LATENCY_MS = 1000        # Dictation processing
    
    @pytest.fixture
    def audio_generator(self):
        """Generate test audio for different command types"""
        def generate_audio(duration_ms: int, frequency: int = 440) -> bytes:
            """Generate sine wave audio for testing"""
            sample_rate = 16000
            samples = int(sample_rate * duration_ms / 1000)
            t = np.linspace(0, duration_ms / 1000, samples)
            
            # Generate sine wave with some noise for realism
            audio_data = np.sin(2 * np.pi * frequency * t) * 0.8
            audio_data += np.random.normal(0, 0.05, samples)  # Add noise
            
            # Convert to int16
            audio_data = (audio_data * 32767).astype(np.int16)
            return audio_data.tobytes()
        
        return generate_audio
    
    @pytest.fixture
    def command_test_cases(self, audio_generator):
        """Test cases for different command categories"""
        return {
            "instant_commands": {
                "click": audio_generator(200, 800),
                "enter": audio_generator(300, 600),
                "1": audio_generator(150, 900),
                "5": audio_generator(180, 750)
            },
            "quick_commands": {
                "right click": audio_generator(400, 650),
                "ctrl c": audio_generator(350, 700),
                "mark a": audio_generator(450, 600),
                "go to": audio_generator(380, 680)
            },
            "ambiguous_commands": {
                "down": audio_generator(250, 550),
                "mark": audio_generator(300, 600),  # Prefix of "mark a", "mark b", etc.
                "select": audio_generator(350, 650),
                "go": audio_generator(200, 700)
            },
            "dictation_samples": {
                "hello world": audio_generator(800, 500),
                "this is a test": audio_generator(1200, 450),
                "longer dictation text": audio_generator(1500, 480)
            }
        }
    
    @pytest.fixture
    def mock_vosk_with_timing(self):
        """Mock Vosk engine that simulates realistic processing times"""
        def create_mock_vosk():
            vosk_engine = Mock()
            
            def mock_recognize(audio_bytes, sample_rate=None):
                # Simulate processing time based on audio length
                audio_length = len(audio_bytes)
                processing_time = min(0.02 + (audio_length / 100000), 0.1)  # 20-100ms
                time.sleep(processing_time)
                
                # Return different results based on audio characteristics
                if audio_length < 5000:  # Short audio
                    return "click"
                elif audio_length < 8000:  # Medium audio
                    return "right click"
                else:  # Longer audio
                    return "complex command"
            
            vosk_engine.recognize = mock_recognize
            return vosk_engine
        
        return create_mock_vosk
    
    @pytest.fixture
    def mock_whisper_with_timing(self):
        """Mock Whisper engine that simulates realistic processing times"""
        def create_mock_whisper():
            whisper_engine = Mock()
            
            def mock_recognize(audio_bytes, sample_rate=None):
                # Whisper is slower but more accurate
                audio_length = len(audio_bytes)
                processing_time = 0.1 + (audio_length / 50000)  # 100ms+ base time
                time.sleep(processing_time)
                
                # Return dictation-like results
                if audio_length < 8000:
                    return "hello world"
                elif audio_length < 15000:
                    return "this is a test"
                else:
                    return "longer dictation text with multiple words"
            
            whisper_engine.recognize = mock_recognize
            return whisper_engine
        
        return create_mock_whisper
    
    @pytest.fixture
    def latency_stt_service(self, mock_event_bus, mock_global_config, mock_vosk_with_timing, mock_whisper_with_timing):
        """Create STT service with timing-aware mocks"""
        service = StreamlinedSpeechToTextService(mock_event_bus, mock_global_config)
        
        # Inject timing-aware engines
        service.vosk_engine = mock_vosk_with_timing()
        service.whisper_engine = mock_whisper_with_timing()
        service._engines_initialized = True
        service._duplicate_filter.is_duplicate = Mock(return_value=False)
        
        return service
    
    @pytest.mark.asyncio
    async def test_instant_command_latency(self, latency_stt_service, command_test_cases):
        """Test that instant commands meet latency requirements"""
        service = latency_stt_service
        service._dictation_active = False
        
        latencies = []
        
        for command, audio_bytes in command_test_cases["instant_commands"].items():
            event = CommandAudioSegmentReadyEvent(
                audio_bytes=audio_bytes,
                sample_rate=16000
            )
            
            start_time = time.time()
            await service._handle_command_audio_segment(event)
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append((command, latency_ms))
            
            assert latency_ms < self.INSTANT_COMMAND_LATENCY_MS, \
                f"Instant command '{command}' took {latency_ms:.1f}ms (limit: {self.INSTANT_COMMAND_LATENCY_MS}ms)"
        
        avg_latency = sum(lat[1] for lat in latencies) / len(latencies)
        print(f"Instant commands average latency: {avg_latency:.1f}ms")
    
    @pytest.mark.asyncio
    async def test_quick_command_latency(self, latency_stt_service, command_test_cases):
        """Test that quick commands meet latency requirements"""
        service = latency_stt_service
        service._dictation_active = False
        
        latencies = []
        
        for command, audio_bytes in command_test_cases["quick_commands"].items():
            event = CommandAudioSegmentReadyEvent(
                audio_bytes=audio_bytes,
                sample_rate=16000
            )
            
            start_time = time.time()
            await service._handle_command_audio_segment(event)
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append((command, latency_ms))
            
            assert latency_ms < self.QUICK_COMMAND_LATENCY_MS, \
                f"Quick command '{command}' took {latency_ms:.1f}ms (limit: {self.QUICK_COMMAND_LATENCY_MS}ms)"
        
        avg_latency = sum(lat[1] for lat in latencies) / len(latencies)
        print(f"Quick commands average latency: {avg_latency:.1f}ms")
    
    @pytest.mark.asyncio
    async def test_ambiguous_command_latency(self, latency_stt_service, command_test_cases):
        """Test that ambiguous commands meet latency requirements"""
        service = latency_stt_service
        service._dictation_active = False
        
        latencies = []
        
        for command, audio_bytes in command_test_cases["ambiguous_commands"].items():
            event = CommandAudioSegmentReadyEvent(
                audio_bytes=audio_bytes,
                sample_rate=16000
            )
            
            start_time = time.time()
            await service._handle_command_audio_segment(event)
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append((command, latency_ms))
            
            assert latency_ms < self.AMBIGUOUS_COMMAND_LATENCY_MS, \
                f"Ambiguous command '{command}' took {latency_ms:.1f}ms (limit: {self.AMBIGUOUS_COMMAND_LATENCY_MS}ms)"
        
        avg_latency = sum(lat[1] for lat in latencies) / len(latencies)
        print(f"Ambiguous commands average latency: {avg_latency:.1f}ms")
    
    @pytest.mark.asyncio
    async def test_dictation_latency(self, latency_stt_service, command_test_cases):
        """Test that dictation processing meets latency requirements"""
        service = latency_stt_service
        
        latencies = []
        
        for text, audio_bytes in command_test_cases["dictation_samples"].items():
            event = DictationAudioSegmentReadyEvent(
                audio_bytes=audio_bytes,
                sample_rate=16000
            )
            
            start_time = time.time()
            await service._handle_dictation_audio_segment(event)
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append((text, latency_ms))
            
            assert latency_ms < self.DICTATION_LATENCY_MS, \
                f"Dictation '{text}' took {latency_ms:.1f}ms (limit: {self.DICTATION_LATENCY_MS}ms)"
        
        avg_latency = sum(lat[1] for lat in latencies) / len(latencies)
        print(f"Dictation average latency: {avg_latency:.1f}ms")
    
    @pytest.mark.asyncio
    async def test_concurrent_processing_latency(self, latency_stt_service, command_test_cases):
        """Test latency under concurrent processing load"""
        service = latency_stt_service
        service._dictation_active = False
        
        # Process multiple commands concurrently
        tasks = []
        start_times = []
        
        for command, audio_bytes in list(command_test_cases["instant_commands"].items())[:3]:
            event = CommandAudioSegmentReadyEvent(
                audio_bytes=audio_bytes,
                sample_rate=16000
            )
            
            start_times.append(time.time())
            task = asyncio.create_task(service._handle_command_audio_segment(event))
            tasks.append((command, task))
        
        # Wait for all tasks to complete
        results = []
        for i, (command, task) in enumerate(tasks):
            await task
            end_time = time.time()
            latency_ms = (end_time - start_times[i]) * 1000
            results.append((command, latency_ms))
        
        # Even under concurrent load, latency should be reasonable
        max_latency = max(lat[1] for lat in results)
        assert max_latency < self.QUICK_COMMAND_LATENCY_MS * 2, \
            f"Concurrent processing caused excessive latency: {max_latency:.1f}ms"
        
        print(f"Concurrent processing max latency: {max_latency:.1f}ms")
    
    @pytest.mark.parametrize("audio_length_ms,expected_max_latency", [
        (100, 80),   # Very short audio
        (300, 120),  # Short audio
        (500, 180),  # Medium audio
        (1000, 300), # Long audio
    ])
    @pytest.mark.asyncio
    async def test_latency_scales_with_audio_length(self, latency_stt_service, audio_generator, audio_length_ms, expected_max_latency):
        """Test that latency scales appropriately with audio length"""
        service = latency_stt_service
        service._dictation_active = False
        
        audio_bytes = audio_generator(audio_length_ms)
        event = CommandAudioSegmentReadyEvent(
            audio_bytes=audio_bytes,
            sample_rate=16000
        )
        
        start_time = time.time()
        await service._handle_command_audio_segment(event)
        end_time = time.time()
        
        latency_ms = (end_time - start_time) * 1000
        
        assert latency_ms < expected_max_latency, \
            f"Audio length {audio_length_ms}ms took {latency_ms:.1f}ms (expected < {expected_max_latency}ms)"
    
    def test_vosk_engine_direct_latency(self, mock_global_config):
        """Test direct Vosk engine latency without service overhead"""
        with patch('iris.services.audio.vosk_stt.vosk') as mock_vosk:
            # Mock Vosk components
            mock_model = Mock()
            mock_recognizer = Mock()
            mock_recognizer.AcceptWaveform.return_value = True
            mock_recognizer.FinalResult.return_value = '{"text": "click"}'
            mock_recognizer.Reset = Mock()
            
            mock_vosk.Model.return_value = mock_model
            mock_vosk.KaldiRecognizer.return_value = mock_recognizer
            
            vosk_engine = EnhancedVoskSTT(
                model_path="test/path",
                sample_rate=16000,
                config=mock_global_config
            )
            
            # Test recognition latency
            test_audio = np.random.randint(-1000, 1000, 3200, dtype=np.int16).tobytes()
            
            start_time = time.time()
            result = vosk_engine.recognize(test_audio)
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000
            
            # Direct engine should be very fast (mocked)
            assert latency_ms < 10, f"Direct Vosk recognition took {latency_ms:.1f}ms"
            assert result == "click"
    
    def test_whisper_engine_direct_latency(self, mock_stt_config):
        """Test direct Whisper engine latency without service overhead"""
        with patch('iris.services.audio.whisper_stt.WhisperModel') as mock_whisper_class:
            # Mock Whisper components
            mock_model = Mock()
            mock_segment = Mock()
            mock_segment.text = "hello world"
            mock_segment.avg_logprob = -0.5
            
            mock_model.transcribe.return_value = ([mock_segment], Mock())
            mock_whisper_class.return_value = mock_model
            
            whisper_engine = WhisperSpeechToText(
                model_name="base",
                device="cpu",
                config=mock_stt_config
            )
            
            # Test recognition latency
            test_audio = np.random.randint(-1000, 1000, 8000, dtype=np.int16).tobytes()
            
            start_time = time.time()
            result = whisper_engine.recognize(test_audio)
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000
            
            # Direct engine should be fast (mocked)
            assert latency_ms < 50, f"Direct Whisper recognition took {latency_ms:.1f}ms"
            assert result == "hello world"
    
    @pytest.mark.asyncio
    async def test_latency_percentiles(self, latency_stt_service, audio_generator):
        """Test latency percentiles for performance analysis"""
        service = latency_stt_service
        service._dictation_active = False
        
        # Run multiple iterations to get latency distribution
        latencies = []
        num_iterations = 20
        
        for _ in range(num_iterations):
            audio_bytes = audio_generator(250)  # Standard command length
            event = CommandAudioSegmentReadyEvent(
                audio_bytes=audio_bytes,
                sample_rate=16000
            )
            
            start_time = time.time()
            await service._handle_command_audio_segment(event)
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        latencies.sort()
        
        p50 = latencies[int(0.5 * len(latencies))]
        p95 = latencies[int(0.95 * len(latencies))]
        p99 = latencies[int(0.99 * len(latencies))]
        
        print(f"Latency percentiles - P50: {p50:.1f}ms, P95: {p95:.1f}ms, P99: {p99:.1f}ms")
        
        # Performance requirements
        assert p50 < 100, f"P50 latency {p50:.1f}ms exceeds 100ms"
        assert p95 < 200, f"P95 latency {p95:.1f}ms exceeds 200ms"
        assert p99 < 400, f"P99 latency {p99:.1f}ms exceeds 400ms"
    
    def test_generate_latency_test_script(self, tmp_path):
        """Generate a script for manual latency testing with real audio"""
        script_content = '''#!/usr/bin/env python3
"""
Manual latency testing script

Run this script to test latency with real audio input.
Speak commands and measure the response time.
"""
import time
import sounddevice as sd
import numpy as np
from iris.services.audio.vosk_stt import EnhancedVoskSTT
from iris.config.app_config import GlobalAppConfig

def record_and_test_latency(duration_seconds=2):
    """Record audio and measure STT latency"""
    print(f"Recording for {duration_seconds} seconds...")
    
    sample_rate = 16000
    audio_data = sd.rec(
        int(duration_seconds * sample_rate), 
        samplerate=sample_rate, 
        channels=1, 
        dtype=np.int16
    )
    sd.wait()
    
    print("Processing...")
    
    # Initialize STT engine
    config = GlobalAppConfig()
    stt_engine = EnhancedVoskSTT(
        model_path=config.model_paths.vosk_model,
        sample_rate=sample_rate,
        config=config
    )
    
    # Measure latency
    start_time = time.time()
    result = stt_engine.recognize(audio_data.tobytes())
    end_time = time.time()
    
    latency_ms = (end_time - start_time) * 1000
    
    print(f"Recognized: '{result}'")
    print(f"Latency: {latency_ms:.1f}ms")
    
    return result, latency_ms

if __name__ == "__main__":
    print("Latency Testing Script")
    print("======================")
    print("Commands to try:")
    print("- Single words: 'click', 'enter', 'space'")
    print("- Multi-word: 'right click', 'ctrl c'")
    print("- Numbers: '1', '5', '25'")
    print("- Ambiguous: 'down', 'mark'")
    print()
    
    while True:
        input("Press Enter to record (Ctrl+C to exit)...")
        try:
            record_and_test_latency()
            print()
        except KeyboardInterrupt:
            print("\\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
'''
        
        script_path = tmp_path / "latency_test.py"
        script_path.write_text(script_content)
        
        assert script_path.exists()
        print(f"Latency test script generated: {script_path}")
        
        return str(script_path)

