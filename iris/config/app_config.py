from pydantic import BaseModel, Field
import yaml
from typing import Optional, Dict, Any, List, ClassVar
import logging
import os

from iris.config.logging_config import LoggingConfigModel
from iris.config.stt_config import STTConfig
from iris.config.sound_recognizer_config import SoundRecognizerConfig as ImportedSoundRecognizerConfig
from iris.config.dictation_config import DictationConfig

logger = logging.getLogger(__name__)

class AudioConfig(BaseModel):
    sample_rate: int = Field(16000, description="Audio sample rate in Hz.")
    chunk_size: int = Field(320, description="Audio chunk size for command mode.")
    channels: int = Field(1, description="Number of audio channels.")
    dtype: str = Field("int16", description="Data type of audio samples (e.g., 'int16', 'float32').")
    device: int | None = Field(None, description="Input device index for sounddevice. None means default device.")
    enable_dual_mode_processing: bool = Field(default=True, description="Enable separate processing paths for commands vs dictation.")
    command_chunk_size: int = Field(default=960, description="Ultra-optimized chunk size for maximum short-word performance - 60ms at 16kHz.")
    enable_partial: bool = Field(default=True, description="Enable partial recognition fast-tracking for unambiguous command prefixes.")



class MarkTriggersConfig(BaseModel):
    create_mark: str = Field(default="mark", description="Trigger word to create a mark.")
    delete_mark: str = Field(default="delete mark", description="Trigger phrase to delete a mark.")
    visualize_marks: List[str] = Field(default_factory=lambda: ["show marks", "visualize marks"], description="Phrases to visualize marks.")
    reset_marks: List[str] = Field(default_factory=lambda: ["reset marks", "clear all marks"], description="Phrases to reset all marks.")
    visualization_cancel: List[str] = Field(default_factory=lambda: ["cancel marks", "hide marks"], description="Phrases to cancel mark visualization.")

class MarkConfig(BaseModel):
    triggers: MarkTriggersConfig = Field(default_factory=MarkTriggersConfig, description="Trigger phrases for mark commands.")
    storage_filename: str = Field(default="marks.json", description="Filename for storing marks.")
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
    cancel_phrase: str = Field(default="cancel", description="Alternative cancel phrase for grid operations.")


class ErrorHandlingConfig(BaseModel):
    """Configuration for error handling service."""
    
    # Notification settings
    notify_ui_on_error: bool = Field(default=True, description="Whether to send UI notifications for errors")
    auto_dismiss_notifications: bool = Field(default=True, description="Whether error notifications should auto-dismiss")
    notification_dismiss_timeout_ms: int = Field(default=5000, description="How long notifications should display before auto-dismissing (ms)")
    
    # Logging settings
    log_error_details: bool = Field(default=True, description="Whether to include full error details in logs")





class LLMConfig(BaseModel):
    """Configuration for LLM service"""
    model_size: str = Field(default="S", description="LLM model size: XS, S, M, or L")
    context_length: int = Field(default=4096, description="Context length for LLM processing")
    max_tokens: int = Field(default=1024, description="Maximum tokens to generate")
    n_threads: int = Field(default=8, description="Number of threads for LLM processing")
    
    # Startup configuration for faster app startup
    startup_mode: str = Field(default="startup", description="When to initialize LLM: 'startup', 'background', 'lazy'")
    
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
    vosk_model: str = "assets/vosk-model-small-en-us-0.15" # Default, can be overridden 

class MarkVisualizationConfig(BaseModel):
    auto_hide_duration_ms: int = 5000

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
    mark_visualization: MarkVisualizationConfig = Field(default_factory=MarkVisualizationConfig)
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
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # src -> project_root
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
