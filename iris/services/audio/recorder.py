import sounddevice as sd
import threading
from typing import Optional, Callable
import numpy as np
import time
import logging
from iris.config.app_config import GlobalAppConfig
from iris.services.audio.smart_timeout_manager import SmartTimeoutManager

class AudioRecorder:
    """Simplified audio recorder optimized for either command or dictation mode"""
    
    def __init__(self, 
                 app_config: GlobalAppConfig,
                 mode: str = "command",
                 on_audio_segment: Optional[Callable[[bytes], None]] = None,
                 on_streaming_chunk: Optional[Callable[[bytes, bool], str]] = None):
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{mode}")
        
        self.app_config = app_config
        self.mode = mode
        self.on_audio_segment = on_audio_segment
        self.on_streaming_chunk = on_streaming_chunk
        
        # Mode-specific configuration
        if mode == "command":
            self.chunk_size = 960  # 60ms at 16kHz
            self.energy_threshold = getattr(app_config.vad, 'command_energy_threshold', 0.002)
            self.silence_timeout = 0.35
            self.max_duration = 3.0
            self.pre_roll_chunks = 4
            self.enable_streaming = True
        else:  # dictation
            self.chunk_size = 2048
            self.energy_threshold = getattr(app_config.vad, 'dictation_energy_threshold', app_config.vad.energy_threshold * 0.6)
            self.silence_timeout = 3.0
            self.max_duration = 12.0
            self.pre_roll_chunks = 3
            self.enable_streaming = False
        
        self.sample_rate = app_config.audio.sample_rate
        self.device = getattr(app_config.audio, 'device', None)
        self.silence_threshold = self.energy_threshold * 0.35
        
        # Recording state
        self._is_recording = False
        self._is_active = True
        self._thread: Optional[threading.Thread] = None
        self._stream: Optional[sd.InputStream] = None
        self._lock = threading.Lock()
        
        # Noise floor adaptation
        self._noise_floor = 0.002
        self._noise_samples = []
        self._max_noise_samples = 20
        
        # Smart timeout for command mode
        self._smart_timeout_manager = SmartTimeoutManager(app_config) if mode == "command" else None
        
        self.logger.info(f"AudioRecorder initialized for {mode} mode")

    def _calculate_energy(self, audio_chunk: np.ndarray) -> float:
        """Calculate RMS energy"""
        if audio_chunk.dtype == np.int16:
            return np.sqrt(np.mean((audio_chunk.astype(np.float32) / 32768.0) ** 2))
        return np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2))
    
    def _update_noise_floor(self, energy: float):
        """Update noise floor estimation"""
        if len(self._noise_samples) < self._max_noise_samples:
            self._noise_samples.append(energy)
            
            if len(self._noise_samples) == self._max_noise_samples:
                self._noise_floor = np.percentile(self._noise_samples, 75)
                margin_multiplier = 3.0 if self.mode == "command" else 2.5
                adaptive_threshold = self._noise_floor * margin_multiplier
                
                if adaptive_threshold > self.energy_threshold * 2.0:
                    old_threshold = self.energy_threshold
                    self.energy_threshold = adaptive_threshold
                    self.silence_threshold = self.energy_threshold * 0.4
                    self.logger.info(f"Adapted thresholds: {old_threshold:.6f} -> {self.energy_threshold:.6f}")

    def _recording_thread(self):
        """Main recording thread"""
        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                channels=1,
                dtype='int16',
                device=self.device
            )
            self._stream.start()
            self.logger.info(f"{self.mode} recording started")
            
            while self._is_recording:
                if not self._is_active:
                    time.sleep(0.1)
                    continue
                
                # Wait for speech detection
                audio_buffer = []
                pre_roll_buffer = []
                speech_detected = False
                
                while self._is_recording and self._is_active and not speech_detected:
                    try:
                        data, _ = self._stream.read(self.chunk_size)
                        energy = self._calculate_energy(data)
                        
                        # Maintain pre-roll buffer
                        pre_roll_buffer.append(data)
                        if len(pre_roll_buffer) > self.pre_roll_chunks:
                            pre_roll_buffer.pop(0)
                        
                        # Update noise floor occasionally
                        if energy <= self.energy_threshold and len(self._noise_samples) < self._max_noise_samples:
                            self._update_noise_floor(energy)
                        
                        # Speech detection
                        if energy > self.energy_threshold:
                            speech_detected = True
                            audio_buffer.extend(pre_roll_buffer)
                            
                    except Exception as e:
                        self.logger.error(f"Error reading audio: {e}")
                        break
                
                if not speech_detected:
                    continue
                
                # Collect audio until silence
                silence_start = None
                recording_start = time.time()
                chunks_collected = 0
                
                while self._is_recording and self._is_active:
                    try:
                        data, _ = self._stream.read(self.chunk_size)
                        energy = self._calculate_energy(data)
                        audio_buffer.append(data)
                        chunks_collected += 1
                        
                        # Streaming recognition for command mode
                        if (self.enable_streaming and self.on_streaming_chunk and 
                            chunks_collected >= 3 and chunks_collected % 2 == 0):
                            current_audio = np.concatenate(audio_buffer)
                            recognized_command = self.on_streaming_chunk(current_audio.tobytes(), False)
                            
                            if recognized_command and self._is_instant_command(recognized_command):
                                self.logger.info(f"Instant command '{recognized_command}' detected")
                                break
                        
                        # Silence detection
                        current_timeout = self._get_timeout(chunks_collected)
                        if energy < self.silence_threshold:
                            if silence_start is None:
                                silence_start = time.time()
                            elif time.time() - silence_start > current_timeout:
                                break
                        else:
                            silence_start = None
                        
                        # Max duration check
                        if time.time() - recording_start > self.max_duration:
                            break
                            
                    except Exception as e:
                        self.logger.error(f"Error during recording: {e}")
                        break
                
                # Process collected audio
                if audio_buffer and self.on_audio_segment:
                    audio_data = np.concatenate(audio_buffer)
                    duration = len(audio_data) / self.sample_rate
                    
                    if duration >= 0.1:  # Minimum duration filter
                        audio_bytes = audio_data.tobytes()
                        self.logger.info(f"Segment captured: {duration:.3f}s")
                        self.on_audio_segment(audio_bytes)
                
        except Exception as e:
            self.logger.error(f"Recording thread error: {e}", exc_info=True)
        finally:
            self._cleanup_stream()

    def _cleanup_stream(self):
        """Properly cleanup audio stream with detailed error handling"""
        if self._stream:
            try:
                if hasattr(self._stream, 'active') and self._stream.active:
                    self._stream.stop()
                    self.logger.debug(f"{self.mode} stream stopped")
                
                self._stream.close()
                self.logger.debug(f"{self.mode} stream closed")
                
            except Exception as e:
                self.logger.error(f"Error cleaning up {self.mode} audio stream: {e}", exc_info=True)
            finally:
                self._stream = None
                self.logger.info(f"{self.mode} audio stream cleanup completed")

    def _get_timeout(self, chunks_collected: int) -> float:
        """Get adaptive timeout based on speech length"""
        if self.mode != "command":
            return self.silence_timeout
            
        # Adaptive timeout for command mode
        if chunks_collected <= 4:  # < 240ms
            return 0.25
        elif chunks_collected <= 8:  # < 480ms
            return self.silence_timeout
        else:
            return 0.6

    def _is_instant_command(self, recognized_text: str) -> bool:
        """Check if command should execute instantly"""
        if not self._smart_timeout_manager or not recognized_text:
            return False
        timeout = self._smart_timeout_manager.get_timeout_for_text(recognized_text)
        return timeout <= 0.05

    def start(self):
        """Start recording"""
        with self._lock:
            if self._is_recording:
                return
            self._is_recording = True
            self._thread = threading.Thread(target=self._recording_thread, daemon=True)
            self._thread.start()

    def stop(self):
        """Stop recording"""
        with self._lock:
            if not self._is_recording:
                return
            self._is_recording = False
            if self._thread:
                self._thread.join(timeout=2.0)
                if self._thread.is_alive():
                    self.logger.warning(f"{self.mode} recording thread did not terminate cleanly")
            
            # Ensure stream is cleaned up even if thread didn't terminate properly
            self._cleanup_stream()

    def set_active(self, active: bool):
        """Set recorder active state"""
        self._is_active = active

    def is_recording(self) -> bool:
        """Check if recording"""
        return self._is_recording

    def is_active(self) -> bool:
        """Check if active"""
        return self._is_active
    
    def update_timeout_for_text(self, recognized_text: str) -> None:
        """Update timeout based on recognized text"""
        if self._smart_timeout_manager and self.mode == "command":
            new_timeout = self._smart_timeout_manager.get_timeout_for_text(recognized_text)
            if new_timeout != self.silence_timeout:
                self.silence_timeout = new_timeout

    def get_smart_timeout_status(self) -> Optional[dict]:
        """Get smart timeout status"""
        return self._smart_timeout_manager.get_status() if self._smart_timeout_manager else None



