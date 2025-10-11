from pydantic import BaseModel, Field
import yaml
from typing import Optional, Dict, Any, List, ClassVar, Literal
import logging
import os

from iris.app.config.logging_config import LoggingConfigModel

logger = logging.getLogger(__name__)

class AudioConfig(BaseModel):
    sample_rate: int = 16000
    chunk_size: int = Field(320, description="Audio chunk size for command mode.")
    channels: int = 1
    dtype: Literal["int16", "float32", "int32"] = Field("int16", description="Data type of audio samples (e.g., 'int16', 'float32').")
    device: Optional[int] = None
    command_chunk_size: int = Field(default=960, description="Ultra-optimized chunk size for maximum short-word performance - 60ms at 16kHz.")

class STTConfig(BaseModel):
    # Engine configuration
    whisper_model: Literal["tiny", "base", "small", "medium"] = "base"
    whisper_device: Literal["cpu", "cuda"] = "cpu"

    # Common configuration
    sample_rate: int = 16000

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
    create_mark: str = "mark"
    delete_mark: str = "delete mark"
    visualize_marks: List[str] = ["show marks", "visualize marks"]
    reset_marks: List[str] = ["reset marks", "clear all marks"]
    visualization_cancel: List[str] = ["cancel marks", "hide marks"]

class MarkConfig(BaseModel):
    triggers: MarkTriggersConfig = MarkTriggersConfig()
    visualization_duration_seconds: int = Field(default=15, description="Duration in seconds for mark visualization overlay before auto-hide.")


class GridConfig(BaseModel):
    rows: int = 3
    cols: int = 3
    line_color: str = "#00FF00"
    label_color: str = "#FFFFFF"
    font_size: int = 16
    show_labels: bool = True
    default_rect_count: int = Field(default=500, description="Default number of rectangles (cells) to show in the grid if not specified by command.")

    # Triggers for grid commands
    show_grid_phrase: str = "golf"
    select_cell_phrase: str = "select"
    cancel_grid_phrase: str = "cancel"


class ErrorHandlingConfig(BaseModel):
    """Configuration for error handling service."""

    # Notification settings
    notify_ui_on_error: bool = True
    auto_dismiss_notifications: bool = True
    notification_dismiss_timeout_ms: int = 5000

    # Logging settings
    log_error_details: bool = True



class DictationConfig(BaseModel):
    """Configuration for dictation functionality"""

    # Trigger words
    start_trigger: str = "green"
    stop_trigger: str = "amber"
    type_trigger: str = "type"
    smart_start_trigger: str = "smart green"

    # Text filtering and processing
    min_text_length: int = 1

    # Text input settings (used by text_input_service)
    use_clipboard: bool = True
    typing_delay: float = 0.01
    
    # Type dictation specific settings
    type_dictation_silence_timeout: float = 1.0

class LLMConfig(BaseModel):
    """Configuration for LLM service"""
    
    model_info: Dict[str, str] = Field(
        default={
            "repo_id": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
            "filename": "qwen2.5-1.5b-instruct-q5_k_m.gguf"
        },
        description="Internal model configuration"
    )
    
    context_length: int = Field(
        default=2048,
        description="Model context window - 2048 is optimal for dictation (faster than 4096)"
    )
    
    max_tokens: int = Field(
        default=1024,
        description="Max output tokens - sufficient for most dictation, faster than 2600"
    )
    
    n_threads: Optional[int] = Field(
        default=None,
        description="Threads for token generation (None = auto: cpu_count - 1, max 6)"
    )
    
    n_threads_batch: Optional[int] = Field(
        default=None,
        description="Threads for prompt processing (None = auto: same as n_threads). CRITICAL for performance!"
    )
    
    n_batch: int = Field(
        default=2048,
        description="Prompt processing batch size - 2048 matches Ollama optimal, 4x faster than 512"
    )
    
    use_mlock: bool = Field(
        default=False,
        description="Lock model in RAM - disable on 8GB systems to prevent OOM"
    )
    
    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="Low temperature for faster, more deterministic generation"
    )
    top_p: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Lower top_p = fewer tokens to consider = faster"
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Very aggressive top_k for maximum speed (10 tokens only)"
    )
    min_p: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Disabled - adds overhead on some systems"
    )
    repeat_penalty: float = Field(
        default=1.05,
        ge=1.0,
        le=2.0,
        description="Minimal penalty for speed (lower = faster)"
    )
    frequency_penalty: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Disabled - adds overhead"
    )
    
    mirostat_mode: int = Field(
        default=0,
        ge=0,
        le=2,
        description="Disabled - standard sampling is faster on most CPUs"
    )
    mirostat_tau: float = Field(
        default=5.0,
        ge=0.0,
        le=10.0,
        description="Not used when mirostat_mode=0"
    )
    mirostat_eta: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Not used when mirostat_mode=0"
    )
    
    n_gpu_layers: int = Field(
        default=0,
        ge=0,
        description="Number of layers to offload to GPU (0 = CPU only, -1 = all layers)"
    )
    
    verbose: bool = Field(
        default=False,
        description="Enable verbose llama.cpp logging for debugging"
    )
    
    flash_attn: bool = Field(
        default=True,
        description="Enable flash attention for faster computation (recommended)"
    )
    
    type_k: int = Field(
        default=1,
        ge=0,
        le=2,
        description="KV cache key type: 0=f32, 1=f16 (recommended), 2=q8_0. Lower = faster with less memory"
    )
    type_v: int = Field(
        default=1,
        ge=0,
        le=2,
        description="KV cache value type: 0=f32, 1=f16 (recommended), 2=q8_0. Lower = faster with less memory"
    )
    
    generation_timeout_sec: float = Field(
        default=45.0,
        description="Max time for generation before timeout"
    )

    startup_mode: Literal["startup", "background", "lazy"] = Field(
        default="startup", 
        description="When to initialize LLM: 'startup', 'background', 'lazy'"
    )
    
    def get_model_filename(self) -> str:
        """Get the model filename"""
        return self.model_info["filename"]

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
    default_app_name_for_data_dir: str = Field(default="iris_voice_assistant", description="Default app name for data directory")
    user_data_dir_suffix: str = Field(default="_data", description="Suffix for user data directory")
    dev_cache_dir_name: str = "dev_cache"
    user_data_dir: str = "data"

class ModelPathsConfig(BaseModel):
    vosk_model: str = ""
    
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
    logging: LoggingConfigModel = LoggingConfigModel()
    app_info: AppInfoConfig = AppInfoConfig()
    model_paths: ModelPathsConfig = ModelPathsConfig()
    vad: VADConfig = VADConfig()
    grid: GridConfig = GridConfig()
    storage: StorageConfig = StorageConfig()
    sound_recognizer: SoundRecognizerConfig = SoundRecognizerConfig()

    error_handling: ErrorHandlingConfig = ErrorHandlingConfig()
    mark: MarkConfig = MarkConfig()
    stt: STTConfig = STTConfig()
    audio: AudioConfig = AudioConfig()
    dictation: DictationConfig = DictationConfig()
    llm: LLMConfig = LLMConfig()
    markov_predictor: MarkovPredictorConfig = MarkovPredictorConfig()
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
