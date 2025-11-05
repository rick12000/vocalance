import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field

from vocalance.app.config.logging_config import LoggingConfigModel

logger = logging.getLogger(__name__)


class AudioConfig(BaseModel):
    """Configuration for audio capture settings and chunk sizing.

    Controls sample rate, audio format, and device selection for the audio service.
    Audio chunk size is fixed at 50ms (800 samples at 16kHz) for all modes.
    """

    sample_rate: int = 16000
    channels: int = 1
    dtype: Literal["int16", "float32", "int32"] = Field(
        "int16", description="Data type of audio samples (e.g., 'int16', 'float32')."
    )
    device: Optional[int] = None


class STTConfig(BaseModel):
    """Configuration for speech-to-text engines and processing parameters.

    Manages Whisper model selection and retry behavior, configures debouncing and
    duplicate suppression intervals separately for command and dictation modes to
    optimize responsiveness vs accuracy tradeoffs.
    """

    whisper_model: Literal["tiny", "base", "small", "medium"] = "base"
    whisper_device: Literal["cpu", "cuda"] = "cpu"
    whisper_max_retries: int = Field(default=3, description="Maximum retry attempts for Whisper model loading")
    whisper_retry_delay_seconds: int = Field(default=5, description="Delay in seconds between Whisper retry attempts")

    sample_rate: int = 16000


class SoundRecognizerConfig(BaseModel):
    """Sound recognizer configuration using ESC-50 for non-target sounds.

    Configures the YAMNet-based sound recognition system with k-NN classification,
    ESC-50 negative examples, and audio preprocessing parameters. Controls confidence
    thresholds, training sample counts, and sound detection parameters.
    """

    target_sample_rate: int = Field(16000, description="Target sample rate for YAMNet (do not change)")
    energy_threshold: float = Field(0.001, description="Minimum audio energy for processing")

    confidence_threshold: float = Field(0.15, description="Minimum similarity for recognition (optimized for enhanced features)")
    k_neighbors: int = Field(7, description="Number of neighbors for k-NN voting (increased for better discrimination)")
    vote_threshold: float = Field(0.35, description="Minimum vote alignment percentage (optimized for enhanced voting)")

    default_samples_per_sound: int = Field(
        12, description="Default training samples per sound (increased for better discrimination)"
    )
    sample_duration_sec: float = Field(2.0, description="Duration of training samples in seconds")

    max_esc50_samples_per_category: int = Field(15, description="Max samples per ESC-50 category")
    max_total_esc50_samples: int = Field(40, description="Maximum total ESC-50 samples (2:1 negative:positive ratio)")

    esc50_categories: Dict[str, str] = Field(
        default_factory=lambda: {
            "keyboard_typing": "keyboard_typing",
            "mouse_click": "mouse_click",
            "wind": "wind",
            "breathing": "breathing",
            "coughing": "coughing",
            "brushing_teeth": "brushing_teeth",
            "drinking_sipping": "drinking_sipping",
        },
        description="ESC-50 categories used as negative examples",
    )

    silence_threshold: float = Field(0.005, description="RMS energy threshold for silence detection")
    min_sound_duration: float = Field(0.1, description="Minimum sound duration in seconds")
    max_sound_duration: float = Field(2.0, description="Maximum sound duration in seconds")
    frame_length: int = Field(1024, description="Frame length for RMS energy analysis")
    hop_length: int = Field(512, description="Hop length for RMS energy analysis")
    normalization_level: float = Field(0.7, description="Peak normalization level (0.0-1.0)")


class MarkTriggersConfig(BaseModel):
    """Voice command triggers for mark system operations.

    Defines the voice phrases that trigger mark creation, deletion, visualization,
    and reset operations in the mark service.
    """

    create_mark: str = "mark"
    delete_mark: str = "delete mark"
    visualize_marks: List[str] = ["show marks", "visualize marks"]
    reset_marks: List[str] = ["reset marks", "clear all marks"]
    visualization_cancel: List[str] = ["cancel marks", "hide marks"]


class MarkConfig(BaseModel):
    """Configuration for the mark system including triggers and timing parameters.

    Controls voice command phrases for mark operations, visualization overlay duration,
    and shutdown grace period for persisting mark data to storage.
    """

    triggers: MarkTriggersConfig = MarkTriggersConfig()
    shutdown_grace_period_seconds: float = Field(
        default=0.1, description="Time to wait for pending writes during service shutdown"
    )


class GridConfig(BaseModel):
    """Configuration for the click grid overlay system.

    Controls grid appearance (colors, labels, dimensions), default cell count,
    and voice command phrases for showing the grid and selecting cells.
    """

    rows: int = 3
    cols: int = 3
    line_color: str = "#00FF00"
    label_color: str = "#FFFFFF"
    font_size: int = 16
    show_labels: bool = True
    default_rect_count: int = Field(
        default=500, description="Default number of rectangles (cells) to show in the grid if not specified by command."
    )

    show_grid_phrase: str = "go"
    select_cell_phrase: str = "select"


class ErrorHandlingConfig(BaseModel):
    """Configuration for error handling service.

    Controls UI notifications for errors, auto-dismiss behavior, timeout durations,
    and whether to log detailed error information.
    """

    notify_ui_on_error: bool = True
    auto_dismiss_notifications: bool = True
    notification_dismiss_timeout_ms: int = 5000

    log_error_details: bool = True


class DictationConfig(BaseModel):
    """Configuration for dictation functionality.

    Defines voice triggers for starting/stopping dictation, typing mode, clipboard behavior,
    timing delays for text input operations, and whether to enable automatic formatting
    through the LLM service.
    """

    start_trigger: str = "green"
    stop_trigger: str = "amber"
    type_trigger: str = "type"
    smart_start_trigger: str = "smart green"
    visual_start_trigger: str = "visual green"

    use_clipboard: bool = True
    typing_delay: float = 0.01

    type_dictation_silence_timeout: float = 0.1

    pyautogui_pause: float = Field(default=0.01, description="Global pause interval between pyautogui operations (seconds)")
    clipboard_paste_delay_pre: float = Field(default=0.05, description="Delay before clipboard paste operation (seconds)")
    clipboard_paste_delay_post: float = Field(default=0.1, description="Delay after clipboard paste operation (seconds)")
    type_text_post_delay: float = Field(default=0.1, description="Delay after typing text (seconds)")

    enable_dictation_formatting: bool = Field(
        default=True, description="Enable automatic formatting (punctuation, capitalization) in dictation output"
    )


class LLMConfig(BaseModel):
    """Configuration for LLM service.

    Comprehensive configuration for llama.cpp-based LLM inference including model selection,
    context/generation limits, threading/batching parameters, quantization settings,
    sampling parameters, GPU offloading, and startup initialization mode. Tuned for
    dictation formatting on CPU with optimal speed/quality balance.
    """

    model_info: Dict[str, str] = Field(
        default={"repo_id": "Qwen/Qwen2.5-1.5B-Instruct-GGUF", "filename": "qwen2.5-1.5b-instruct-q5_k_m.gguf"},
        description="Internal model configuration",
    )

    context_length: int = Field(
        default=2048, description="Model context window - 2048 is optimal for dictation (faster than 4096)"
    )

    max_tokens: int = Field(default=1024, description="Max output tokens - sufficient for most dictation, faster than 2600")

    n_threads: Optional[int] = Field(default=None, description="Threads for token generation (None = auto: cpu_count - 1, max 6)")

    n_threads_batch: Optional[int] = Field(
        default=None, description="Threads for prompt processing (None = auto: same as n_threads). CRITICAL for performance!"
    )

    n_batch: int = Field(
        default=2048, description="Prompt processing batch size - 2048 matches Ollama optimal, 4x faster than 512"
    )

    use_mlock: bool = Field(default=False, description="Lock model in RAM - disable on 8GB systems to prevent OOM")

    temperature: float = Field(
        default=0.3, ge=0.0, le=2.0, description="Low temperature for faster, more deterministic generation"
    )
    top_p: float = Field(default=0.8, ge=0.0, le=1.0, description="Lower top_p = fewer tokens to consider = faster")
    top_k: int = Field(default=10, ge=1, le=100, description="Very aggressive top_k for maximum speed (10 tokens only)")
    min_p: float = Field(default=0.0, ge=0.0, le=1.0, description="Disabled - adds overhead on some systems")
    repeat_penalty: float = Field(default=1.05, ge=1.0, le=2.0, description="Minimal penalty for speed (lower = faster)")
    frequency_penalty: float = Field(default=0.0, ge=0.0, le=2.0, description="Disabled - adds overhead")

    mirostat_mode: int = Field(default=0, ge=0, le=2, description="Disabled - standard sampling is faster on most CPUs")
    mirostat_tau: float = Field(default=5.0, ge=0.0, le=10.0, description="Not used when mirostat_mode=0")
    mirostat_eta: float = Field(default=0.1, ge=0.0, le=1.0, description="Not used when mirostat_mode=0")

    n_gpu_layers: int = Field(default=0, ge=0, description="Number of layers to offload to GPU (0 = CPU only, -1 = all layers)")

    verbose: bool = Field(default=False, description="Enable verbose llama.cpp logging for debugging")

    flash_attn: bool = Field(default=True, description="Enable flash attention for faster computation (recommended)")

    type_k: int = Field(
        default=1, ge=0, le=2, description="KV cache key type: 0=f32, 1=f16 (recommended), 2=q8_0. Lower = faster with less memory"
    )
    type_v: int = Field(
        default=1,
        ge=0,
        le=2,
        description="KV cache value type: 0=f32, 1=f16 (recommended), 2=q8_0. Lower = faster with less memory",
    )

    generation_timeout_sec: float = Field(default=45.0, description="Max time for generation before timeout")

    startup_mode: Literal["startup", "background", "lazy"] = Field(
        default="startup", description="When to initialize LLM: 'startup', 'background', 'lazy'"
    )

    def get_model_filename(self) -> str:
        """Get the GGUF model filename from model_info dictionary.

        Returns:
            Model filename string extracted from model_info.
        """
        return self.model_info["filename"]


class VADConfig(BaseModel):
    """Configuration for Voice Activity Detection (VAD) across multiple modes.

    Defines energy thresholds, recording durations, silence detection parameters, and
    pre-roll buffering separately optimized for command mode (low latency), dictation mode
    (longer speech), and training mode (sample collection). Includes adaptive noise floor
    estimation for robust speech detection in varying acoustic environments.
    """

    noise_floor_estimation: bool = Field(default=True, description="Enable automatic noise floor estimation.")

    command_energy_threshold: float = Field(
        default=0.0008,
        description="Energy threshold for command mode speech detection - further lowered for more sensitive detection.",
    )
    command_silent_chunks_for_end: int = Field(
        default=5,
        description="Number of consecutive silent chunks to end recording in command mode (4 chunks = 200ms at 50ms/chunk).",
    )
    command_max_recording_duration: float = Field(default=4, description="Maximum recording duration for command mode.")
    command_pre_roll_buffers: int = Field(
        default=5,
        description="Pre-roll buffers for command mode (200ms at 50ms chunks) - captures full word attack including initial consonants.",
    )

    dictation_energy_threshold: float = Field(default=0.0035, description="Energy threshold for dictation mode.")
    dictation_silent_chunks_for_end: int = Field(
        default=16,
        description="Number of consecutive silent chunks to end recording in dictation mode (16 chunks = 800ms at 50ms/chunk).",
    )
    dictation_max_recording_duration: float = Field(default=30.0, description="Maximum recording duration for dictation mode.")
    dictation_pre_roll_buffers: int = Field(default=5, description="Pre-roll buffers for dictation mode (250ms at 50ms/chunk).")

    silence_threshold_multiplier: float = Field(
        default=0.55, description="Multiplier for silence threshold relative to energy threshold"
    )
    command_adaptive_margin_multiplier: float = Field(
        default=3.5, description="Multiplier for adaptive noise floor in command mode - increased for lower threshold robustness."
    )
    dictation_adaptive_margin_multiplier: float = Field(
        default=2.5, description="Multiplier for adaptive noise floor in dictation mode"
    )
    adaptive_threshold_max_multiplier: float = Field(
        default=2.0, description="Maximum multiplier before applying adaptive threshold"
    )
    adaptive_silence_threshold_multiplier: float = Field(
        default=0.6, description="Adjustment factor for silence threshold after adaptation"
    )

    command_min_recording_duration: float = Field(
        default=0.05, description="Minimum recording duration for command mode in seconds"
    )
    dictation_min_recording_duration: float = Field(
        default=0.1, description="Minimum recording duration for dictation mode in seconds"
    )

    max_noise_samples: int = Field(default=20, description="Maximum number of samples to collect for noise floor estimation")
    noise_floor_initial_value: float = Field(default=0.002, description="Initial noise floor value before estimation")
    noise_floor_percentile: int = Field(default=75, description="Percentile to use for noise floor calculation")


class MarkovPredictorConfig(BaseModel):
    """Configuration for Markov chain command predictor with backoff."""

    enabled: bool = Field(default=False, description="Enable Markov chain command prediction")

    confidence_threshold: float = Field(default=1.0, description="Minimum probability threshold for prediction (0.0-1.0)")

    training_window_commands: Dict[int, int] = Field(
        default_factory=lambda: {2: 500, 3: 1000, 4: 1500},
        description="Number of recent commands to train on per order {order: count}",
    )

    training_window_days: Dict[int, int] = Field(
        default_factory=lambda: {2: 7, 3: 21, 4: 60},
        description="Number of days of command history to train on per order {order: days}",
    )

    max_order: int = Field(default=4, description="Maximum order of Markov chain (backoff from this)")

    min_order: int = Field(default=2, description="Minimum order of Markov chain (backoff to this)")

    min_command_frequency: Dict[int, int] = Field(
        default_factory=lambda: {2: 15, 3: 10, 4: 10}, description="Minimum transition frequency per order {order: min_count}"
    )

    incorrect_prediction_cooldown: int = Field(
        default=2, description="Number of commands to skip Markov after incorrect prediction"
    )

    prediction_cooldown_seconds: float = Field(
        default=0.05, description="Minimum time in seconds between consecutive Markov predictions to prevent spam"
    )


class CommandParserConfig(BaseModel):
    """Configuration for centralized command parser behavior."""

    duplicate_detection_window_ms: float = Field(
        default=600, description="Time window in milliseconds for command deduplication across Vosk, sound, and Markov sources"
    )


class AutomationServiceConfig(BaseModel):
    """Configuration for automation command execution."""

    thread_pool_max_workers: int = Field(default=2, description="Maximum number of worker threads for automation action execution")

    key_sequence_delay_seconds: float = Field(
        default=0.25, description="Delay in seconds between individual key presses in a key sequence"
    )

    scroll_total_clicks: int = Field(default=600, description="Total number of scroll clicks for animated scrolling")
    scroll_animation_steps: int = Field(default=20, description="Number of animation steps for scrolling")
    scroll_animation_delay_seconds: float = Field(default=0.01, description="Delay between scroll animation steps in seconds")


class ProtectedTermsValidatorConfig(BaseModel):
    """Configuration for protected terms validator."""

    cache_ttl_seconds: float = Field(default=60.0, description="Cache time-to-live in seconds for protected terms")


class AppInfoConfig(BaseModel):
    default_app_name_for_data_dir: str = Field(
        default="vocalance_voice_assistant", description="Default app name for data directory"
    )
    user_data_dir_suffix: str = Field(default="_data", description="Suffix for user data directory")
    dev_cache_dir_name: str = "dev_cache"
    user_data_dir: str = "data"


class AssetPathsConfig(BaseModel):
    """Centralized asset path resolution for both dev and PyInstaller bundle modes.

    Automatically detects whether running from source (development) or from a PyInstaller
    bundle and provides consistent path resolution for all application assets including
    logos, icons, fonts, ML models, and audio samples.
    """

    def __init__(self, **data: any) -> None:
        """Initialize asset paths configuration and resolve assets root directory.

        Args:
            **data: Arbitrary keyword arguments passed to Pydantic BaseModel.
        """
        super().__init__(**data)
        self._assets_root: Optional[Path] = self._get_assets_root()

    def _get_assets_root(self) -> Optional[Path]:
        """Get the assets root directory adaptively for dev or bundled execution.

        Checks for PyInstaller bundle (_MEIPASS) first, then falls back to development
        mode by navigating relative to this config file's location.

        Returns:
            Path to assets root directory, or None if not found.
        """
        if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
            bundle_dir: Path = Path(sys._MEIPASS)
            assets_path: Path = bundle_dir / "vocalance" / "app" / "assets"

            if assets_path.exists():
                logging.debug(f"Found assets in PyInstaller bundle: {assets_path}")
                return assets_path
            else:
                logging.warning(f"Assets not found in bundle at: {assets_path}")
        else:
            config_dir: Path = Path(__file__).resolve().parent
            vocalance_app_dir: Path = config_dir.parent
            assets_path: Path = vocalance_app_dir / "assets"

            if assets_path.exists():
                logging.debug(f"Found assets in dev mode: {assets_path}")
                return assets_path

        return None

    @property
    def logo_dir(self) -> Optional[str]:
        """Logo directory path.

        Returns:
            Path to logo directory or None.
        """
        if self._assets_root:
            return str(self._assets_root / "logo")
        return None

    @property
    def icons_dir(self) -> Optional[str]:
        """Icons directory path.

        Returns:
            Path to icons directory or None.
        """
        if self._assets_root:
            return str(self._assets_root / "icons")
        return None

    @property
    def fonts_dir(self) -> Optional[str]:
        """Fonts directory path.

        Returns:
            Path to fonts directory or None.
        """
        if self._assets_root:
            return str(self._assets_root / "fonts" / "Manrope")
        return None

    @property
    def vosk_model_path(self) -> Optional[str]:
        """Vosk model directory path.

        Returns:
            Path to Vosk model directory or None.
        """
        if self._assets_root:
            return str(self._assets_root / "vosk-model-small-en-us-0.15")
        return None

    @property
    def yamnet_model_path(self) -> Optional[str]:
        """YAMNet model directory path.

        Returns:
            Path to YAMNet model directory or None.
        """
        if self._assets_root:
            return str(self._assets_root / "sound_processing" / "yamnet")
        return None

    @property
    def esc50_samples_path(self) -> Optional[str]:
        """ESC-50 samples directory path.

        Returns:
            Path to ESC-50 samples directory or None.
        """
        if self._assets_root:
            return str(self._assets_root / "sound_processing" / "esc50")
        return None

    @property
    def logo_image_path(self) -> Optional[str]:
        """Main logo image path.

        Returns:
            Path to logo image or None.
        """
        if self.logo_dir:
            logo_path: Path = Path(self.logo_dir) / "logo_full_text_full_size.png"
            return str(logo_path)
        return None

    @property
    def icon_logo_image_path(self) -> Optional[str]:
        """Icon logo image path.

        Returns:
            Path to icon logo image or None.
        """
        if self.logo_dir:
            icon_path: Path = Path(self.logo_dir) / "grey_red_icon_full_size.png"
            return str(icon_path)
        return None

    @property
    def icon_path(self) -> Optional[str]:
        """Application icon path.

        Returns:
            Path to application icon or None.
        """
        if self.logo_dir:
            icon_path: Path = Path(self.logo_dir) / "icon.ico"
            return str(icon_path)
        return None

    def get_vosk_model_path(self) -> str:
        """Get the Vosk model path with fallback for missing assets root.

        Returns:
            Absolute path to Vosk model directory, or fallback relative path if assets
            root is not properly initialized.
        """
        path = self.vosk_model_path
        if path:
            return path
        # Fallback for cases where assets root is not properly set
        return "vocalance/app/assets/vosk-model-small-en-us-0.15"


class StorageConfig(BaseModel):
    """Configuration for persistent storage paths and caching behavior.

    Defines subdirectory names and full paths for all user data storage including
    sound models/samples, marks, click tracking history, settings, LLM models, and
    command history. Paths are initialized automatically in GlobalAppConfig.__init__.
    """

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
    sound_model_dir: Optional[str] = None
    sound_samples_dir: Optional[str] = None
    external_non_target_sounds_dir: Optional[str] = None
    user_data_root: Optional[str] = None
    settings_dir: Optional[str] = None
    marks_dir: Optional[str] = None
    llm_models_dir: Optional[str] = None
    click_tracker_dir: Optional[str] = None
    command_history_dir: Optional[str] = None
    cache_ttl_seconds: float = Field(
        default=300.0, description="Cache time-to-live in seconds for storage service read operations"
    )


class GlobalAppConfig(BaseModel):
    """Main application configuration container aggregating all subsystem configs.

    Central configuration object containing nested configuration models for every
    subsystem: audio, STT, VAD, LLM, grid, marks, storage, error handling, etc.
    Automatically initializes storage directory structure on instantiation.
    """

    logging: LoggingConfigModel = LoggingConfigModel()
    app_info: AppInfoConfig = AppInfoConfig()
    asset_paths: AssetPathsConfig = AssetPathsConfig()
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
    command_parser: CommandParserConfig = CommandParserConfig()
    automation_service: AutomationServiceConfig = AutomationServiceConfig()
    protected_terms_validator: ProtectedTermsValidatorConfig = ProtectedTermsValidatorConfig()
    automation_cooldown_seconds: float = Field(default=0.5, description="Cooldown period between automation command executions.")

    def __init__(self, **data: any) -> None:
        """Initialize global configuration and create storage directory structure.

        Args:
            **data: Arbitrary keyword arguments passed to Pydantic for config overrides.
        """
        super().__init__(**data)
        self._setup_storage_paths()

    def _setup_storage_paths(self) -> None:
        """Setup storage directory paths and create directories if they don't exist.

        Constructs absolute paths for all storage subdirectories, creates them using
        os.makedirs with exist_ok=True, and updates the storage config object with
        the computed paths.
        """
        app_info = self.app_info
        storage = self.storage
        user_data_root = get_default_user_data_root(app_info=app_info)
        sound_model_dir = os.path.join(user_data_root, storage.sound_model_subdir)
        sound_samples_dir = os.path.join(user_data_root, storage.sound_samples_subdir)
        external_non_target_sounds_dir = os.path.join(sound_samples_dir, storage.external_non_target_sounds_subdir)
        settings_dir = os.path.join(user_data_root, storage.settings_subdir)
        marks_dir = os.path.join(user_data_root, storage.marks_subdir)
        click_tracker_dir = os.path.join(user_data_root, storage.click_tracker_subdir)
        llm_models_dir = os.path.join(user_data_root, storage.llm_models_subdir)
        command_history_dir = os.path.join(user_data_root, storage.command_history_subdir)

        for d in [
            sound_model_dir,
            sound_samples_dir,
            external_non_target_sounds_dir,
            settings_dir,
            marks_dir,
            click_tracker_dir,
            llm_models_dir,
            command_history_dir,
        ]:
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


CONFIG_FILE_NAME = "settings.yaml"
DEFAULT_CONFIG_DIR_NAME = "config"


def get_config_path(
    config_dir: Optional[str] = None, config_file: str = CONFIG_FILE_NAME, app_info: Optional[AppInfoConfig] = None
) -> str:
    """Get configuration file path with fallback hierarchy.

    Determines configuration file location with the following priority:
    1. Custom config_dir if provided
    2. User data settings directory if app_info provided
    3. Project repository config directory as fallback

    Args:
        config_dir: Optional custom config directory path.
        config_file: Configuration filename (defaults to settings.yaml).
        app_info: Application info for resolving user data root directory.

    Returns:
        Absolute path to configuration file.
    """
    if config_dir:
        return os.path.join(config_dir, config_file)

    if app_info is not None:
        user_data_root = get_default_user_data_root(app_info=app_info)
        settings_dir = os.path.join(user_data_root, "settings")
        os.makedirs(settings_dir, exist_ok=True)
        return os.path.join(settings_dir, config_file)

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(project_root, DEFAULT_CONFIG_DIR_NAME, config_file)


def load_app_config(config_path: Optional[str] = None, app_info: Optional[AppInfoConfig] = None) -> GlobalAppConfig:
    """Load application configuration from YAML file with fallback to defaults.

    Attempts to load configuration from the specified or computed path. Returns default
    GlobalAppConfig if the file is missing, empty, or lacks the required 'app' root key.
    Raises exceptions for YAML parsing errors or other unexpected failures.

    Args:
        config_path: Optional explicit path to configuration file.
        app_info: Application info for computing default configuration path.

    Returns:
        Loaded GlobalAppConfig instance with overrides applied, or default instance on failure.
    """
    actual_config_path = config_path or get_config_path(app_info=app_info)
    logger.debug(f"Loading application configuration from: {actual_config_path}")

    try:
        with open(actual_config_path, "r") as f:
            config_data = yaml.safe_load(f)
        if not config_data or "app" not in config_data:
            logger.warning(
                f"Configuration file {actual_config_path} is empty or missing 'app' root. Using default GlobalAppConfig."
            )
            return GlobalAppConfig()
        return GlobalAppConfig(**config_data.get("app", {}))
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
    """Get default user data root directory based on operating system conventions.

    Uses %APPDATA% on Windows for application data storage, and home directory on
    Unix-like systems. Appends the configured application name and suffix.

    Args:
        app_info: Application info configuration containing name and suffix.

    Returns:
        Absolute path to user data root directory.
    """
    if os.name == "nt":
        base = os.environ.get("APPDATA", os.path.expanduser("~"))
    else:
        base = os.path.expanduser("~")
    return os.path.join(base, app_info.default_app_name_for_data_dir + app_info.user_data_dir_suffix)
