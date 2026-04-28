import logging
import os
import sys
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from vocalance.app.config.logging_config import LoggingConfigModel

logger = logging.getLogger(__name__)


class AudioDeviceCaptureMessages(BaseModel):
    """User-visible copy when the default input device cannot be opened."""

    model_config = ConfigDict(extra="ignore")

    mic_unavailable_unknown_device: str = Field(
        default=(
            "The default microphone that was in use when Vocalance started is no longer available "
            "or could not be opened.\n\n"
            "Please reconnect your microphone or fix your system audio settings, then "
            "completely quit and restart Vocalance."
        )
    )
    mic_unavailable_named_device: str = Field(
        default=(
            "The microphone that was in use when Vocalance started ({device_name}) is no longer "
            "available or could not be opened.\n\n"
            "Please reconnect your microphone or fix your system audio settings, then "
            "completely quit and restart Vocalance."
        )
    )

    def message_for_launch_device(self, launch_device_name: Optional[str]) -> str:
        """Return the user-visible mic-unavailable message for a launch device name.

        Args:
            launch_device_name: Device name captured at launch, if known.

        Returns:
            Formatted named-device message, or the generic unknown-device message.
        """
        if launch_device_name:
            return self.mic_unavailable_named_device.format(device_name=launch_device_name)
        return self.mic_unavailable_unknown_device


class AudioConfig(BaseModel):
    """Configuration for audio capture settings and chunk sizing.

    Controls sample rate and format. Input uses the host default device; there is no
    in-app microphone selection.
    """

    model_config = ConfigDict(extra="ignore")

    sample_rate: int = 16000
    channels: int = 1
    dtype: Literal["int16", "float32", "int32"] = Field(
        "int16", description="Data type of audio samples (e.g., 'int16', 'float32')."
    )
    capture_chunk_duration_seconds: float = Field(
        default=0.03,
        ge=0.01,
        le=0.2,
        description="PortAudio input block duration in seconds (chunk length for capture callback).",
    )
    device_capture_messages: AudioDeviceCaptureMessages = Field(default_factory=AudioDeviceCaptureMessages)


class MoonshineStreamingConfig(BaseModel):
    """Tuning for Moonshine streaming dictation (partial cadence, native VAD, decode gating).

    Values map to string options passed to ``moonshine_load_transcriber_from_files`` (see Moonshine
    ``parse_transcriber_options``). ``stream_update_interval`` is the Python transcriber/stream
    update cadence, not a native option key.
    """

    model_config = ConfigDict(extra="ignore")

    stream_update_interval: float = Field(
        default=0.5,
        ge=0.05,
        le=2.0,
        description="Seconds of stream time between partial refresh calls (higher = fewer partials, less CPU).",
    )
    transcription_interval: float = Field(
        default=0.6,
        ge=0.05,
        le=2.0,
        description="Native transcription_interval: minimum seconds of buffered audio before each stream decode pass.",
    )
    vad_threshold: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Native vad_threshold; lower than 0.5 tends to reduce premature line finals on short pauses.",
    )
    vad_window_duration: Optional[float] = Field(
        default=None,
        ge=0.05,
        le=2.0,
        description="Optional native vad_window_duration (seconds); omit for library default.",
    )
    vad_max_segment_duration: float = Field(
        default=32.0,
        ge=5.0,
        le=120.0,
        description="Native vad_max_segment_duration (seconds) before a forced VAD segment split; default raised from library ~15s.",
    )
    max_tokens_per_second: Optional[float] = Field(
        default=None,
        ge=1.0,
        le=30.0,
        description="Optional native max_tokens_per_second; omit for library default.",
    )

    def transcriber_load_options(self) -> dict[str, str]:
        """Key/value strings for ``Transcriber(..., options=...)`` at model load."""
        opts: dict[str, str] = {
            "transcription_interval": str(self.transcription_interval),
            "vad_threshold": str(self.vad_threshold),
        }
        if self.vad_window_duration is not None:
            opts["vad_window_duration"] = str(self.vad_window_duration)
        opts["vad_max_segment_duration"] = str(self.vad_max_segment_duration)
        if self.max_tokens_per_second is not None:
            opts["max_tokens_per_second"] = str(self.max_tokens_per_second)
        return opts


class STTConfig(BaseModel):
    """Configuration for speech-to-text engines and processing parameters.

    Dictation uses Moonshine Voice (streaming + batch). Command mode uses Vosk.
    """

    model_config = ConfigDict(extra="ignore")

    moonshine_language: str = Field(default="en", description="Two-letter language code for Moonshine models")
    moonshine_model_arch: str = Field(
        default="medium-streaming",
        description=(
            "Moonshine architecture id: tiny, base, tiny-streaming, base-streaming, small-streaming, medium-streaming. "
            "medium-streaming is larger than small-streaming and usually more accurate (higher latency, bigger download)."
        ),
    )
    moonshine_streaming: MoonshineStreamingConfig = Field(
        default_factory=MoonshineStreamingConfig,
        description="Streaming partial cadence and native Moonshine transcriber options (VAD, decode interval).",
    )
    moonshine_max_stream_line_duration_seconds: float = Field(
        default=45.0,
        ge=0.0,
        le=600.0,
        description=(
            "After this many seconds of audio on one Moonshine stream line, start a new native stream. "
            "Decoder cost grows with unbounded line length; rotation keeps partial latency stable. "
            "0 disables rotation."
        ),
    )
    moonshine_max_retries: int = Field(default=3, description="Maximum retry attempts for Moonshine model loading")
    moonshine_retry_delay_seconds: int = Field(default=5, description="Delay in seconds between Moonshine load retries")

    sample_rate: int = 16000

    @field_validator("moonshine_model_arch", mode="before")
    @classmethod
    def default_moonshine_arch_if_empty(cls, v: object) -> object:
        if v is None or (isinstance(v, str) and not v.strip()):
            return "medium-streaming"
        return v


class SoundRecognizerConfig(BaseModel):
    """Sound recognizer configuration using ESC-50 for non-target sounds.

    Configures the YAMNet-based sound recognition system with k-NN classification,
    ESC-50 negative examples, and audio preprocessing parameters. Controls confidence
    thresholds, training sample counts, and sound detection parameters.
    """

    target_sample_rate: int = Field(16000, description="Target sample rate for YAMNet (do not change)")
    energy_threshold: float = Field(0.005, description="Minimum audio energy for processing")

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
    hover_grid_phrase: str = "hover"
    drag_grid_phrase: str = Field(
        default="move",
        description="Voice phrase to show the grid in drag mode (click-hold from pointer at show time to chosen cell).",
    )
    select_cell_phrase: str = "select"
    click_history_ui_refresh_debounce_s: float = Field(
        default=0.05,
        ge=0.0,
        le=2.0,
        description="Debounce before notifying UI to refresh grid labels after new clicks.",
    )
    click_history_persist_debounce_s: float = Field(
        default=1.5,
        ge=0.05,
        le=120.0,
        description="Debounce before writing click history JSON via StorageService (async, non-blocking).",
    )


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
    hidden_start_trigger: str = "hidden green"
    amend_start_trigger: str = "amend"

    use_clipboard: bool = True
    typing_delay: float = 0.01

    type_dictation_silence_timeout: float = 0.1

    moonshine_modifier_suppress_sec: float = Field(
        default=0.55,
        ge=0.0,
        le=10.0,
        description="After a modifier phrase, drop Moonshine partial/final output for this many seconds",
    )
    type_silence_monitor_max_seconds: int = Field(
        default=300,
        ge=30,
        le=3600,
        description="Safety cap for TYPE dictation silence watcher (seconds)",
    )

    pyautogui_pause: float = Field(default=0.01, description="Global pause interval between pyautogui operations (seconds)")
    clipboard_paste_delay_pre: float = Field(default=0.05, description="Delay before clipboard paste operation (seconds)")
    clipboard_paste_delay_post: float = Field(default=0.1, description="Delay after clipboard paste operation (seconds)")
    type_text_post_delay: float = Field(default=0.1, description="Delay after typing text (seconds)")

    enable_dictation_formatting: bool = Field(
        default=True, description="Enable automatic formatting (punctuation, capitalization) in dictation output"
    )

    modifier_upper_phrase: str = Field(default="upper", description="Voice phrase to toggle title-case modifier")
    modifier_capitals_phrase: str = Field(default="capitals", description="Voice phrase to toggle ALL CAPS modifier")
    modifier_camel_phrase: str = Field(
        default="camel", description="Voice phrase to toggle UpperCamelCase (PascalCase) identifier modifier"
    )
    modifier_snake_phrase: str = Field(default="snake", description="Voice phrase to toggle snake_case modifier")
    modifier_spelling_phrase: str = Field(default="spelling", description="Voice phrase to toggle spoken-punctuation modifier")
    modifier_kebab_phrase: str = Field(default="kebab", description="Voice phrase to toggle kebab-case modifier")
    modifier_diminish_phrase: str = Field(default="diminish", description="Voice phrase to toggle lowercase modifier")
    modifier_strip_phrase: str = Field(default="strip", description="Voice phrase to toggle strip punctuation modifier")


class LocalLLMArtifact(BaseModel):
    """One built-in GGUF bundle (Hugging Face repo + filenames + UI label)."""

    model_config = ConfigDict(frozen=True)

    id: str
    label: str
    repo_id: str
    gguf_filenames: tuple[str, ...]
    model_card_url: str

    @property
    def load_path_filename(self) -> str:
        return self.gguf_filenames[0]


class LocalLLMAllowList(BaseModel):
    """Built-in local LLM bundles the app may download and run."""

    model_config = ConfigDict(frozen=True)

    artifacts: tuple[LocalLLMArtifact, ...]

    def artifact_for(self, model_id: str) -> Optional[LocalLLMArtifact]:
        for artifact in self.artifacts:
            if artifact.id == model_id:
                return artifact
        return None

    def has_id(self, model_id: str) -> bool:
        return any(a.id == model_id for a in self.artifacts)

    @property
    def default_id(self) -> str:
        return self.artifacts[0].id


def _builtin_local_llm_allowlist() -> LocalLLMAllowList:
    return LocalLLMAllowList(
        artifacts=(
            LocalLLMArtifact(
                id="qwen2.5-1.5b-q5km",
                label="Qwen 2.5 1.5B Instruct (Q5_K_M, CPU)",
                repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
                gguf_filenames=("qwen2.5-1.5b-instruct-q5_k_m.gguf",),
                model_card_url="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF",
            ),
            LocalLLMArtifact(
                id="qwen2.5-3b-q5km",
                label="Qwen 2.5 3B Instruct (Q5_K_M, CPU)",
                repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
                gguf_filenames=("qwen2.5-3b-instruct-q5_k_m.gguf",),
                model_card_url="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF",
            ),
            LocalLLMArtifact(
                id="qwen2.5-7b-q5km",
                label="Qwen 2.5 7B Instruct (Q5_K_M, CPU)",
                repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
                gguf_filenames=(
                    "qwen2.5-7b-instruct-q5_k_m-00001-of-00002.gguf",
                    "qwen2.5-7b-instruct-q5_k_m-00002-of-00002.gguf",
                ),
                model_card_url="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF",
            ),
        )
    )


_LOCAL_LLM_ALLOWLIST: LocalLLMAllowList = _builtin_local_llm_allowlist()
DEFAULT_LLM_MODEL_ID: str = _LOCAL_LLM_ALLOWLIST.default_id


def get_whitelisted_llm_model(model_id: str) -> Optional[LocalLLMArtifact]:
    return _LOCAL_LLM_ALLOWLIST.artifact_for(model_id)


def is_whitelisted_llm_model_id(model_id: str) -> bool:
    return _LOCAL_LLM_ALLOWLIST.has_id(model_id)


def local_llm_allowlist() -> LocalLLMAllowList:
    return _LOCAL_LLM_ALLOWLIST


class LLMConfig(BaseModel):
    """Local LLM (llama.cpp, CPU): built-in Qwen GGUF bundles only."""

    model_config = ConfigDict(extra="ignore")

    selected_model_id: str = Field(
        default=DEFAULT_LLM_MODEL_ID,
        description="Id of a built-in GGUF bundle (see LocalLLMAllowList).",
    )

    context_length: int = Field(
        default=2048, description="Model context window - 2048 is optimal for dictation (faster than 4096)"
    )

    max_tokens: int = Field(default=1500, description="Max output tokens - sufficient for most dictation, faster than 2600")

    n_threads: Optional[int] = Field(default=None, description="Threads for token generation (None = auto: cpu_count - 1, max 6)")

    n_threads_batch: Optional[int] = Field(
        default=None, description="Threads for prompt processing (None = auto: same as n_threads). CRITICAL for performance!"
    )

    n_batch: int = Field(
        default=2048, description="Prompt processing batch size - 2048 matches Ollama optimal, 4x faster than 512"
    )

    use_mlock: bool = Field(default=False, description="Lock model in RAM - disable on 8GB systems to prevent OOM")

    temperature: float = Field(
        default=0.5, ge=0.0, le=2.0, description="High temperature for creative rewriting and instruction-following"
    )
    top_p: float = Field(default=0.95, ge=0.0, le=1.0, description="High top_p for diverse vocabulary and creative rewrites")
    top_k: int = Field(default=30, ge=1, le=100, description="Higher top_k for creative vocabulary choices")
    min_p: float = Field(default=0.05, ge=0.0, le=1.0, description="Filter low-probability tokens for quality")
    repeat_penalty: float = Field(default=1.15, ge=1.0, le=2.0, description="Strong penalty to prevent copying input text")
    frequency_penalty: float = Field(default=0.2, ge=0.0, le=2.0, description="Strong encouragement for vocabulary diversity")

    mirostat_mode: int = Field(default=0, ge=0, le=2, description="Disabled - standard sampling is faster on most CPUs")
    mirostat_tau: float = Field(default=5.0, ge=0.0, le=10.0, description="Not used when mirostat_mode=0")
    mirostat_eta: float = Field(default=0.1, ge=0.0, le=1.0, description="Not used when mirostat_mode=0")

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

    @field_validator("selected_model_id")
    @classmethod
    def validate_selected_model_id(cls, v: str) -> str:
        if not _LOCAL_LLM_ALLOWLIST.has_id(v):
            raise ValueError(f"selected_model_id must be a built-in LLM id, got {v!r}")
        return v


class VADConfig(BaseModel):
    """Configuration for Voice Activity Detection (VAD) for command and sound modes.

    Dictation uses continuous audio chunks fed to Moonshine streaming (no separate VAD segment pipeline).

    The VAD system uses continuous adaptive noise floor estimation with audio normalization
    to work robustly across different microphones. Key features:
    - DC offset removal and peak normalization before energy calculation
    - Rolling window noise floor estimation with bootstrap period
    - Thresholds adapt continuously to changing acoustic environments
    - Works with dynamic mics, condensers, headsets, and USB devices

    NOTE: All chunk-based parameters assume 30ms chunks (industry standard for VAD).
    """

    noise_floor_estimation: bool = Field(default=True, description="Enable automatic noise floor estimation.")

    enable_audio_normalization: bool = Field(
        default=True,
        description="Enable audio preprocessing (DC offset removal, peak normalization) for microphone-robust VAD.",
    )

    command_energy_threshold: float = Field(
        default=0.0005,
        description="Minimum energy threshold for command mode (used as floor when noise is very low).",
    )
    sound_energy_threshold: float = Field(
        default=0.003,
        description="Minimum energy threshold for sound recognition - higher than command to reduce false triggers.",
    )
    command_silent_chunks_for_end: int = Field(
        default=5,
        description="Number of consecutive silent chunks to end recording in command mode (5 chunks = 150ms at 30ms/chunk).",
    )
    command_max_recording_duration: float = Field(default=4, description="Maximum recording duration for command mode.")
    command_pre_roll_buffers: int = Field(
        default=7,
        description="Pre-roll buffers for command mode (210ms at 30ms chunks) - captures word attack.",
    )
    command_min_recording_duration: float = Field(
        default=0.05, description="Minimum recording duration for command mode in seconds"
    )
    command_max_threshold: float = Field(default=0.1, description="Upper clamp on the adaptive speech threshold for command mode.")

    sound_silent_chunks_for_end: int = Field(
        default=5,
        description="Number of consecutive silent chunks to end recording in sound mode (5 chunks = 150ms at 30ms/chunk).",
    )
    sound_max_recording_duration: float = Field(
        default=1.02, description="Maximum recording duration for sound mode in seconds (~34 chunks at 30ms/chunk)."
    )
    sound_pre_roll_buffers: int = Field(
        default=5,
        description="Pre-roll buffers for sound mode (150ms at 30ms chunks) - captures the leading edge of a transient.",
    )
    sound_min_recording_duration: float = Field(
        default=0.15, description="Minimum recording duration for sound mode in seconds (~5 chunks at 30ms/chunk)."
    )
    sound_min_peak_ratio: float = Field(
        default=1.5,
        description="Minimum ratio of clip peak energy to speech threshold for sound clips - filters background noise.",
    )
    sound_max_threshold: float = Field(default=0.15, description="Upper clamp on the adaptive speech threshold for sound mode.")

    silence_threshold_multiplier: float = Field(
        default=0.45, description="Multiplier for silence threshold relative to speech threshold"
    )
    command_adaptive_margin_multiplier: float = Field(
        default=3.5, description="Multiplier applied to noise floor for command speech threshold."
    )
    sound_adaptive_margin_multiplier: float = Field(
        default=5.0,
        description="Multiplier applied to noise floor for sound detection - higher than speech to reduce false triggers.",
    )
    adaptive_threshold_max_multiplier: float = Field(
        default=2.0, description="Maximum multiplier before applying adaptive threshold (legacy, kept for compatibility)"
    )
    adaptive_silence_threshold_multiplier: float = Field(
        default=0.65, description="Adjustment factor for silence threshold after adaptation (legacy, kept for compatibility)"
    )

    max_noise_samples: int = Field(
        default=100,
        description="Rolling window size for noise floor estimation (~5 seconds at 50ms chunks)",
    )
    noise_floor_initial_value: float = Field(default=0.002, description="Initial noise floor value before estimation")
    noise_floor_percentile: int = Field(default=50, description="Percentile for noise floor calculation (50=median, more robust)")


class CommandParserConfig(BaseModel):
    """Configuration for centralized command parser behavior."""

    model_config = ConfigDict(extra="ignore")

    min_command_interval_ms: float = Field(
        default=100.0,
        description="Minimum milliseconds between executed parsed commands; later commands in the window are ignored.",
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
    """Configuration for application identity and data directory naming.

    Controls the base names and suffixes used for constructing user-specific data directories
    where application state and user data are persisted.
    """

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

    @cached_property
    def assets_root(self) -> Optional[Path]:
        """Root directory containing ``assets`` (dev tree or PyInstaller bundle)."""
        return self.resolve_assets_root()

    def resolve_assets_root(self) -> Optional[Path]:
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
                return assets_path
            else:
                logging.warning(f"Assets not found in bundle at: {assets_path}")
        else:
            config_dir: Path = Path(__file__).resolve().parent
            vocalance_app_dir: Path = config_dir.parent
            assets_path: Path = vocalance_app_dir / "assets"

            if assets_path.exists():
                return assets_path

        return None

    @property
    def logo_dir(self) -> Optional[str]:
        """Logo directory path.

        Returns:
            Path to logo directory or None.
        """
        if self.assets_root:
            return str(self.assets_root / "logo")
        return None

    @property
    def icons_dir(self) -> Optional[str]:
        """Icons directory path.

        Returns:
            Path to icons directory or None.
        """
        if self.assets_root:
            return str(self.assets_root / "icons")
        return None

    @property
    def fonts_dir(self) -> Optional[str]:
        """Fonts directory path.

        Returns:
            Path to base fonts directory or None.
        """
        if self.assets_root:
            return str(self.assets_root / "fonts")
        return None

    @property
    def vosk_model_path(self) -> Optional[str]:
        """Vosk model directory path.

        Returns:
            Path to Vosk model directory or None.
        """
        if self.assets_root:
            return str(self.assets_root / "vosk-model-small-en-us-0.15")
        return None

    @property
    def yamnet_model_path(self) -> Optional[str]:
        """YAMNet model directory path.

        Returns:
            Path to YAMNet model directory or None.
        """
        if self.assets_root:
            return str(self.assets_root / "sound_processing" / "yamnet")
        return None

    @property
    def esc50_samples_path(self) -> Optional[str]:
        """ESC-50 samples directory path.

        Returns:
            Path to ESC-50 samples directory or None.
        """
        if self.assets_root:
            return str(self.assets_root / "sound_processing" / "esc50")
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
            icon_path: Path = Path(self.logo_dir) / "grey_icon_full_size.png"
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
        return "vocalance/app/assets/vosk-model-small-en-us-0.15"


class StorageConfig(BaseModel):
    """Configuration for persistent storage paths and caching behavior.

    Defines subdirectory names and full paths for all user data storage including
    sound models/samples, marks, click tracking history, settings, and LLM models.
    Paths are initialized automatically in GlobalAppConfig.__init__.
    """

    sound_model_subdir: str = "sound_models"
    sound_samples_subdir: str = "sound_samples"
    marks_subdir: str = "marks"
    click_tracker_subdir: str = "click_tracker"
    settings_subdir: str = "settings"
    llm_models_subdir: str = "llm_models"
    external_non_target_sounds_subdir: str = "external_non_target_sounds"
    marks_filename: str = "marks.json"
    click_history_filename: str = "click_history.json"
    sound_model_dir: Optional[str] = None
    sound_samples_dir: Optional[str] = None
    external_non_target_sounds_dir: Optional[str] = None
    user_data_root: Optional[str] = None
    settings_dir: Optional[str] = None
    marks_dir: Optional[str] = None
    llm_models_dir: Optional[str] = None
    click_tracker_dir: Optional[str] = None
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
    command_parser: CommandParserConfig = CommandParserConfig()
    automation_service: AutomationServiceConfig = AutomationServiceConfig()
    protected_terms_validator: ProtectedTermsValidatorConfig = ProtectedTermsValidatorConfig()
    automation_cooldown_seconds: float = Field(default=0.5, description="Cooldown period between automation command executions.")

    def __init__(self, **data: Any) -> None:
        """Initialize global configuration and create storage directory structure.

        Args:
            **data: Arbitrary keyword arguments passed to Pydantic for config overrides.
        """
        super().__init__(**data)
        self.setup_storage_paths()

    def setup_storage_paths(self) -> None:
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

        for d in [
            sound_model_dir,
            sound_samples_dir,
            external_non_target_sounds_dir,
            settings_dir,
            marks_dir,
            click_tracker_dir,
            llm_models_dir,
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

    @property
    def local_llm_allowlist(self) -> LocalLLMAllowList:
        return _LOCAL_LLM_ALLOWLIST


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
    actual_config_path: str = config_path or get_config_path(app_info=app_info)

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
