from pydantic import BaseModel, Field
import yaml
from typing import Optional, Dict, Any, List, ClassVar, Literal
import logging
import os

from iris.app.config.logging_config import LoggingConfigModel

logger = logging.getLogger(__name__)

class AudioConfig(BaseModel):
    sample_rate: int = Field(16000, description="Audio sample rate in Hz.")
    chunk_size: int = Field(320, description="Audio chunk size for command mode.")
    channels: int = Field(1, description="Number of audio channels.")
    dtype: Literal["int16", "float32", "int32"] = Field("int16", description="Data type of audio samples (e.g., 'int16', 'float32').")
    device: Optional[int] = Field(None, description="Input device index for sounddevice. None means default device.")
    command_chunk_size: int = Field(default=960, description="Ultra-optimized chunk size for maximum short-word performance - 60ms at 16kHz.")

class STTConfig(BaseModel):
    # Engine configuration
    whisper_model: Literal["tiny", "base", "small", "medium"] = Field("base", description="Whisper model size (tiny for speed, base/small/medium for accuracy)")
    whisper_device: Literal["cpu", "cuda"] = Field("cpu", description="Device for Whisper inference (cpu, cuda)")
    
    # Common configuration
    sample_rate: int = Field(16000, description="Sample rate for STT.")
    
    # Command mode settings (optimized for ULTRA-LOW LATENCY)
    command_debounce_interval: float = Field(default=0.02, description="Ultra-aggressive debounce for command mode (20ms for real-time response).")
    command_duplicate_text_interval: float = Field(default=0.2, description="Very short duplicate suppression for commands (200ms).")
    command_max_segment_duration_sec: float = Field(default=1.5, description="Short max duration for fast command execution.")
    
    # Enhanced dictation mode settings (optimized for accuracy and continuity)
    dictation_debounce_interval: float = Field(default=0.1, description="Reduced debounce for better dictation responsiveness.")
    dictation_duplicate_text_interval: float = Field(default=4.0, description="Extended duplicate suppression to catch phrase repetitions - increased from 3.0s.")
    dictation_max_segment_duration_sec: float = Field(default=20.0, description="Extended max duration for longer dictation segments - increased from 15.0s.")



class SoundRecognizerConfig(BaseModel):
    """
    Clean sound recognizer configuration using only ESC-50 for non-target sounds.
    """
    
    target_sample_rate: int = Field(
        16000, 
        description="Target sample rate for YAMNet (do not change)"
    )
    energy_threshold: float = Field(
        0.001, 
        description="Minimum audio energy for processing"
    )
    
    confidence_threshold: float = Field(
        0.15, 
        description="Minimum similarity for recognition (optimized for enhanced features)"
    )
    k_neighbors: int = Field(
        7, 
        description="Number of neighbors for k-NN voting (increased for better discrimination)"
    )
    vote_threshold: float = Field(
        0.35, 
        description="Minimum vote alignment percentage (optimized for enhanced voting)"
    )
    
    default_samples_per_sound: int = Field(
        12, 
        description="Default training samples per sound (increased for better discrimination)"
    )
    sample_duration_sec: float = Field(
        2.0, 
        description="Duration of training samples in seconds"
    )
    
    max_esc50_samples_per_category: int = Field(
        15, 
        description="Max samples per ESC-50 category"
    )
    max_total_esc50_samples: int = Field(
        40, 
        description="Maximum total ESC-50 samples (2:1 negative:positive ratio)"
    )
    
    esc50_categories: Dict[str, str] = Field(
        default_factory=lambda: {
            "keyboard_typing": "keyboard_typing",
            "mouse_click": "mouse_click",
            "wind": "wind",
            "breathing": "breathing",
            "coughing": "coughing",
            "brushing_teeth": "brushing_teeth",
            "drinking_sipping": "drinking_sipping"
        },
        description="ESC-50 categories used as negative examples"
    )


class MarkTriggersConfig(BaseModel):
    create_mark: str = Field(default="mark", description="Trigger word to create a mark.")
    delete_mark: str = Field(default="delete mark", description="Trigger phrase to delete a mark.")
    visualize_marks: List[str] = Field(default_factory=lambda: ["show marks", "visualize marks"], description="Phrases to visualize marks.")
    reset_marks: List[str] = Field(default_factory=lambda: ["reset marks", "clear all marks"], description="Phrases to reset all marks.")
    visualization_cancel: List[str] = Field(default_factory=lambda: ["cancel marks", "hide marks"], description="Phrases to cancel mark visualization.")

class MarkConfig(BaseModel):
    triggers: MarkTriggersConfig = Field(default_factory=MarkTriggersConfig, description="Trigger phrases for mark commands.")
    visualization_duration_seconds: int = Field(default=15, description="Duration in seconds for mark visualization overlay before auto-hide.")


class GridConfig(BaseModel):
    rows: int = Field(3, description="Number of grid rows.")
    cols: int = Field(3, description="Number of grid columns.")
    line_color: str = Field("#00FF00", description="Color of grid lines.")
    label_color: str = Field("#FFFFFF", description="Color of grid labels.")
    font_size: int = Field(16, description="Font size for grid labels.")
    show_labels: bool = Field(True, description="Whether to show grid cell labels.")
    default_rect_count: int = Field(default=500, description="Default number of rectangles (cells) to show in the grid if not specified by command.")

    # Triggers for grid commands
    show_grid_phrase: str = Field(default="golf", description="Phrase to show the grid.")
    select_cell_phrase: str = Field(default="select", description="Phrase to select grid cells.")
    cancel_grid_phrase: str = Field(default="cancel", description="Phrase to hide/cancel the grid.")


class ErrorHandlingConfig(BaseModel):
    """Configuration for error handling service."""
    
    # Notification settings
    notify_ui_on_error: bool = Field(default=True, description="Whether to send UI notifications for errors")
    auto_dismiss_notifications: bool = Field(default=True, description="Whether error notifications should auto-dismiss")
    notification_dismiss_timeout_ms: int = Field(default=5000, description="How long notifications should display before auto-dismissing (ms)")
    
    # Logging settings
    log_error_details: bool = Field(default=True, description="Whether to include full error details in logs")



class DictationConfig(BaseModel):
    """Configuration for dictation functionality"""
    
    # Trigger words
    start_trigger: str = Field(default="green", description="Trigger word to start standard dictation")
    stop_trigger: str = Field(default="amber", description="Trigger word to stop any dictation mode")
    type_trigger: str = Field(default="type", description="Trigger word to start type mode")
    smart_start_trigger: str = Field(default="smart green", description="Trigger phrase to start LLM-assisted dictation")
    
    # Text filtering and processing
    min_text_length: int = Field(default=1, description="Minimum length of text to process")
    
    # Text input settings (used by text_input_service)
    use_clipboard: bool = Field(default=True, description="Use clipboard for text input instead of typing")
    typing_delay: float = Field(default=0.01, description="Delay between keystrokes when typing (if not using clipboard)")

class LLMConfig(BaseModel):
    """Configuration for LLM service"""
    model_size: Literal["XS", "S", "M", "L"] = Field(default="S", description="LLM model size: XS, S, M, or L")
    context_length: int = Field(default=4096, description="Context length for LLM processing")
    max_tokens: int = Field(default=1024, description="Maximum tokens to generate")
    n_threads: int = Field(default=8, description="Number of threads for LLM processing")
    
    # Startup configuration for faster app startup
    startup_mode: Literal["startup", "background", "lazy"] = Field(default="startup", description="When to initialize LLM: 'startup', 'background', 'lazy'")
    
    # Hugging Face model repository mapping
    HF_MODEL_MAPPING: ClassVar[Dict[str, Dict[str, str]]] = {
        "XS": {
            "repo_id": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
            "filename": "qwen2.5-1.5b-instruct-q5_k_m.gguf"
        },
        "S": {
            "repo_id": "Qwen/Qwen2.5-1.5B-Instruct-GGUF", 
            "filename": "qwen2.5-1.5b-instruct-q8_0.gguf"
        }
    }
    
    def get_model_info(self) -> Dict[str, str]:
        """Get the Hugging Face model info for the configured size"""
        return self.HF_MODEL_MAPPING.get(self.model_size, self.HF_MODEL_MAPPING["XS"])
    
    def get_model_filename(self) -> str:
        """Get the model filename for the configured size"""
        model_info = self.get_model_info()
        return model_info["filename"]

class VADConfig(BaseModel):
    # Base VAD settings
    energy_threshold: float = Field(default=0.006, description="Base energy threshold for speech detection.")
    max_recording_duration: float = Field(default=4.0, description="Maximum recording duration in seconds.")
    pre_roll_buffers: int = Field(default=2, description="Number of audio chunks to buffer before speech detection.")
    
    # Energy-based detection
    continuation_energy_threshold: float = Field(default=0.002, description="Threshold for detecting speech continuation.")
    noise_floor_estimation: bool = Field(default=True, description="Enable automatic noise floor estimation.")
    
    # Command mode specific settings
    command_energy_threshold: float = Field(default=0.002, description="Energy threshold for command mode speech detection.")
    command_silent_chunks_for_end: int = Field(default=3, description="Number of consecutive silent chunks to end recording in command mode (3 chunks = 180ms at 60ms/chunk).")
    command_max_recording_duration: float = Field(default=3, description="Maximum recording duration for command mode.")
    command_pre_roll_buffers: int = Field(default=4, description="Pre-roll buffers for command mode (240ms at 60ms chunks).")
    
    # Dictation mode specific settings
    dictation_energy_threshold: float = Field(default=0.0035, description="Energy threshold for dictation mode.")
    dictation_silent_chunks_for_end: int = Field(default=25, description="Number of consecutive silent chunks to end recording in dictation mode (25 chunks = 500ms at 20ms/chunk).")
    dictation_max_recording_duration: float = Field(default=8.0, description="Maximum recording duration for dictation mode.")
    dictation_pre_roll_buffers: int = Field(default=2, description="Pre-roll buffers for dictation mode.")
    
    # Training mode specific settings
    training_energy_threshold: float = Field(default=0.003, description="Energy threshold for training sample collection.")
    training_silent_chunks_for_end: int = Field(default=40, description="Number of consecutive silent chunks to end training recording (40 chunks = 800ms).")
    training_max_wait_for_sound_duration: float = Field(default=10.0, description="Maximum time to wait for sound during training.")
    training_pre_roll_buffers: int = Field(default=4, description="Pre-roll buffers for training sample collection.")

class MarkovPredictorConfig(BaseModel):
    """Configuration for Markov chain command predictor with backoff"""
    
    enabled: bool = Field(
        default=False,
        description="Enable Markov chain command prediction"
    )
    
    confidence_threshold: float = Field(
        default=0.95,
        description="Minimum probability threshold for prediction (0.0-1.0)"
    )
    
    training_window_commands: Dict[int, int] = Field(
        default_factory=lambda: {2: 500, 3: 1000, 4: 1500},
        description="Number of recent commands to train on per order {order: count}"
    )
    
    training_window_days: Dict[int, int] = Field(
        default_factory=lambda: {2: 3, 3: 5, 4: 7},
        description="Number of days of command history to train on per order {order: days}"
    )
    
    max_order: int = Field(
        default=4,
        description="Maximum order of Markov chain (backoff from this)"
    )
    
    min_order: int = Field(
        default=2,
        description="Minimum order of Markov chain (backoff to this)"
    )
    
    min_command_frequency: Dict[int, int] = Field(
        default_factory=lambda: {2: 30, 3: 15, 4: 10},
        description="Minimum transition frequency per order {order: min_count}"
    )
    
    incorrect_prediction_cooldown: int = Field(
        default=2,
        description="Number of commands to skip Markov after incorrect prediction"
    )

class AppInfoConfig(BaseModel):
    default_app_name_for_data_dir: str = "iris_voice_assistant"
    user_data_dir_suffix: str = "_data"
    dev_cache_dir_name: str = "dev_cache"
    user_data_dir: str = Field(default="data", description="User data directory for all persistent files.")

class ModelPathsConfig(BaseModel):
    vosk_model: str = Field(default="", description="Path to Vosk model, auto-detected if empty")
    
    def __init__(self, **data):
        super().__init__(**data)
        if not self.vosk_model:
            self.vosk_model = self._get_default_vosk_path()
    
    def _get_default_vosk_path(self) -> str:
        """Get default vosk model path, works in both dev and installed mode"""
        from pathlib import Path
        import os
        
        # Try package-relative path first (works in both modes)
        try:
            # Get path relative to this config file
            config_dir = Path(__file__).resolve().parent
            iris_app_dir = config_dir.parent
            vosk_path = iris_app_dir / "assets" / "vosk-model-small-en-us-0.15"
            
            if vosk_path.exists():
                return str(vosk_path)
        except Exception:
            pass
        
        # Fallback to checking common locations
        possible_paths = [
            "iris/app/assets/vosk-model-small-en-us-0.15",
            "assets/vosk-model-small-en-us-0.15",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # Return a reasonable default even if not found (error will occur later with helpful message)
        return "iris/app/assets/vosk-model-small-en-us-0.15" 

class StorageConfig(BaseModel):
    sound_model_subdir: str = "sound_models"
    sound_samples_subdir: str = "sound_samples"
    marks_subdir: str = "marks"
    click_tracker_subdir: str = "click_tracker"
    settings_subdir: str = "settings"
    llm_models_subdir: str = "llm_models"
    external_non_target_sounds_subdir: str = "external_non_target_sounds"
    command_history_subdir: str = "command_history"
    marks_filename: str = "marks.json"
    click_history_filename: str = "click_history.json"
    command_history_filename: str = "command_history.json"
    sound_model_dir: Optional[str] = None  # Directory to store sound models, set at runtime
    sound_samples_dir: Optional[str] = None  # Directory to store sound samples, set at runtime
    external_non_target_sounds_dir: Optional[str] = None  # Directory for external non-target sound samples, set at runtime
    user_data_root: Optional[str] = None  # Will be set dynamically at runtime
    settings_dir: Optional[str] = None  # Will be set dynamically at runtime
    marks_dir: Optional[str] = None  # New: directory for marks
    llm_models_dir: Optional[str] = None  # Directory for LLM models, set at runtime
    click_tracker_dir: Optional[str] = None  # New: directory for click tracker
    command_history_dir: Optional[str] = None  # Directory for command history



class GlobalAppConfig(BaseModel):
    logging: LoggingConfigModel = Field(default_factory=LoggingConfigModel)
    app_info: AppInfoConfig = Field(default_factory=AppInfoConfig)
    model_paths: ModelPathsConfig = Field(default_factory=ModelPathsConfig)
    vad: VADConfig = Field(default_factory=VADConfig) # Added VADConfig
    grid: GridConfig = Field(default_factory=GridConfig) # Now uses the imported GridConfig
    storage: StorageConfig = Field(default_factory=StorageConfig)
    sound_recognizer: ImportedSoundRecognizerConfig = Field(default_factory=ImportedSoundRecognizerConfig)

    error_handling: ErrorHandlingConfig = Field(default_factory=ErrorHandlingConfig)
    mark: MarkConfig = Field(default_factory=MarkConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    dictation: DictationConfig = Field(default_factory=DictationConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    markov_predictor: MarkovPredictorConfig = Field(default_factory=MarkovPredictorConfig)
    scroll_amount_vertical: int = Field(default=120, description="The amount to scroll vertically for 'sky' and 'earth' commands.")
    automation_cooldown_seconds: float = Field(default=0.5, description="Cooldown period between automation command executions.")

    def __init__(self, **data):
        super().__init__(**data)
        self._setup_storage_paths()

    def _setup_storage_paths(self):
        app_info = self.app_info
        storage = self.storage
        user_data_root = get_default_user_data_root(app_info)
        sound_model_dir = os.path.join(user_data_root, storage.sound_model_subdir)
        sound_samples_dir = os.path.join(user_data_root, storage.sound_samples_subdir)
        external_non_target_sounds_dir = os.path.join(sound_samples_dir, storage.external_non_target_sounds_subdir)
        settings_dir = os.path.join(user_data_root, storage.settings_subdir)
        marks_dir = os.path.join(user_data_root, storage.marks_subdir)
        click_tracker_dir = os.path.join(user_data_root, storage.click_tracker_subdir)
        llm_models_dir = os.path.join(user_data_root, storage.llm_models_subdir)
        command_history_dir = os.path.join(user_data_root, storage.command_history_subdir)
        # Ensure directories exist
        for d in [sound_model_dir, sound_samples_dir, external_non_target_sounds_dir, settings_dir, marks_dir, click_tracker_dir, llm_models_dir, command_history_dir]:
            os.makedirs(d, exist_ok=True)
        storage.sound_model_dir = sound_model_dir
        storage.sound_samples_dir = sound_samples_dir
        storage.external_non_target_sounds_dir = external_non_target_sounds_dir
        storage.user_data_root = user_data_root
        storage.settings_dir = settings_dir
        storage.marks_dir = marks_dir
        storage.click_tracker_dir = click_tracker_dir
        storage.llm_models_dir = llm_models_dir
        storage.command_history_dir = command_history_dir

# --- Config Loader ---

CONFIG_FILE_NAME = "settings.yaml"
DEFAULT_CONFIG_DIR_NAME = "config"

def get_config_path(config_dir: Optional[str] = None, config_file: str = CONFIG_FILE_NAME, app_info: Optional[AppInfoConfig] = None) -> str:
    if config_dir:
        return os.path.join(config_dir, config_file)
    # Use app_info if provided, otherwise fallback to repo config
    if app_info is not None:
        user_data_root = get_default_user_data_root(app_info)
        settings_dir = os.path.join(user_data_root, "settings")
        os.makedirs(settings_dir, exist_ok=True)
        return os.path.join(settings_dir, config_file)
    # Fallback: look for 'config/settings.yaml' relative to the current working directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # iris/app/config -> iris (package root)
    return os.path.join(project_root, DEFAULT_CONFIG_DIR_NAME, config_file)

def load_app_config(config_path: Optional[str] = None, app_info: Optional[AppInfoConfig] = None) -> GlobalAppConfig:
    actual_config_path = config_path or get_config_path(app_info=app_info)
    logger.info(f"Loading application configuration from: {actual_config_path}")
    try:
        with open(actual_config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        if not config_data or 'app' not in config_data: # Assuming global config is under an 'app' key
            logger.warning(f"Configuration file {actual_config_path} is empty or missing 'app' root. Using default GlobalAppConfig.")
            return GlobalAppConfig()
        return GlobalAppConfig(**config_data.get('app', {}))
    except FileNotFoundError:
        logger.warning(f"Configuration file not found at {actual_config_path}. Using default GlobalAppConfig.")
        return GlobalAppConfig()
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration file {actual_config_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to load configuration from {actual_config_path}: {e}")
        raise

def get_default_user_data_root(app_info: AppInfoConfig) -> str:
    # Use AppData on Windows, otherwise home dir
    if os.name == "nt":
        base = os.environ.get("APPDATA", os.path.expanduser("~"))
    else:
        base = os.path.expanduser("~")
    return os.path.join(base, app_info.default_app_name_for_data_dir + app_info.user_data_dir_suffix)
