from pydantic import BaseModel, Field

class DictationConfig(BaseModel):
    """Configuration for dictation functionality"""
    
    # Trigger words
    start_trigger: str = Field(default="green", description="Trigger word to start standard dictation")
    stop_trigger: str = Field(default="amber", description="Trigger word to stop any dictation mode")
    type_trigger: str = Field(default="type", description="Trigger word to start type mode")
    smart_start_trigger: str = Field(default="smart green", description="Trigger phrase to start LLM-assisted dictation")
    
    # STT Engine switching for dictation
    enable_stt_switching: bool = Field(default=True, description="Enable automatic STT engine switching for dictation")
    dictation_stt_engine: str = Field(default="whisper", description="STT engine to use during dictation (vosk, whisper)")
    command_stt_engine: str = Field(default="vosk", description="STT engine to use for command recognition (vosk, whisper)")
    
    # Text filtering and processing
    min_text_length: int = Field(default=1, description="Minimum length of text to process")
    remove_trigger_words: bool = Field(default=True, description="Whether to remove trigger words from dictated text")
    
    # Text input settings (used by text_input_service)
    use_clipboard: bool = Field(default=True, description="Use clipboard for text input instead of typing")
    typing_delay: float = Field(default=0.01, description="Delay between keystrokes when typing (if not using clipboard)")