from pydantic import BaseModel, Field
from typing import Literal

class STTConfig(BaseModel):
    # Engine selection
    default_engine: Literal["vosk", "whisper"] = Field("vosk", description="Default STT engine to use")
    model_path: str = Field("assets/vosk-model-small-en-us-0.15", description="Path to the Vosk STT model.")
    whisper_model: Literal["tiny", "base", "small", "medium"] = Field("base", description="Whisper model size (tiny for speed, base/small/medium for accuracy)")
    whisper_device: str = Field("cpu", description="Device for Whisper inference (cpu, cuda)")
    
    # Common configuration
    sample_rate: int = Field(16000, description="Sample rate for STT.")
    max_segment_duration_sec: int = Field(3, description="Maximum duration of audio segment for STT.")
    debounce_interval: float = Field(0.05, description="Default minimum interval between processing audio segments (optimized for commands).")
    duplicate_text_interval: float = Field(0.3, description="Default interval to suppress duplicate recognized text (optimized for commands).")
    
    # Engine switching configuration
    enable_engine_switching: bool = Field(True, description="Enable automatic engine switching based on dictation mode")
    dictation_engine: Literal["vosk", "whisper"] = Field("whisper", description="STT engine to use during dictation mode")
    command_engine: Literal["vosk", "whisper"] = Field("vosk", description="STT engine to use for command recognition")
    
    # Dual-mode processing settings
    enable_dual_mode_processing: bool = Field(default=True, description="Enable separate processing paths for commands vs dictation.")
    
    # Command mode settings (optimized for ULTRA-LOW LATENCY)
    command_debounce_interval: float = Field(default=0.02, description="Ultra-aggressive debounce for command mode (20ms for real-time response).")
    command_duplicate_text_interval: float = Field(default=0.2, description="Very short duplicate suppression for commands (200ms).")
    command_max_segment_duration_sec: float = Field(default=1.5, description="Short max duration for fast command execution.")
    
    # Enhanced dictation mode settings (optimized for accuracy and continuity)
    dictation_debounce_interval: float = Field(default=0.1, description="Reduced debounce for better dictation responsiveness.")
    dictation_duplicate_text_interval: float = Field(default=4.0, description="Extended duplicate suppression to catch phrase repetitions - increased from 3.0s.")
    dictation_max_segment_duration_sec: float = Field(default=20.0, description="Extended max duration for longer dictation segments - increased from 15.0s.")
    dictation_context_enabled: bool = Field(default=True, description="Enable context preservation between dictation segments.")
    dictation_min_segment_duration_sec: float = Field(default=0.8, description="Minimum segment duration for quality dictation - ensures sufficient context.")
