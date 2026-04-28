from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import asyncio
import contextlib
import importlib.util
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from PySide6.QtWidgets import QApplication

from vocalance.app.config.app_config import AppInfoConfig, GlobalAppConfig
from vocalance.app.config.logging_config import setup_logging
from vocalance.app.event_bus import EventBus
from vocalance.app.lifecycle.lifecycle import AppLifecycle
from vocalance.app.lifecycle.registry import ServiceSpec, build_services, register_services_for_teardown
from vocalance.app.ui.qt_startup_window import StartupProgressTracker, StartupWindow
from vocalance.app.ui.qt_theme import theme
from vocalance.app.ui.utils.window_icon_manager import WindowIconManager

logger = logging.getLogger(__name__)


def _build_validator(c: Dict[str, Any]) -> Any:
    from vocalance.app.services.protected_terms_validator import ProtectedTermsValidator

    validator = ProtectedTermsValidator(config=c["config"], storage=c["storage"])
    validator.setup_invalidation_subscriptions(c["event_bus"])
    return validator


def _service_specs() -> List[ServiceSpec]:
    """Declare every service in construction order.

    Teardown is the reverse of this order (LIFO), so adding a new service is a
    single-line edit: construction, registration, and teardown are all derived
    from this list. Heavy module imports stay scoped to this function so they
    only fire when ``build_services`` runs, not on ``qt_main`` import.
    """
    from vocalance.app.services.audio.audio_capture_service import AudioCaptureService
    from vocalance.app.services.audio.dictation_handling.dictation_coordinator import DictationCoordinator
    from vocalance.app.services.audio.segmenting.command_segmenter_service import CommandSegmenterService
    from vocalance.app.services.audio.segmenting.sound_segmenter_service import SoundSegmenterService
    from vocalance.app.services.audio.sound_recognizer.streamlined_sound_service import SoundService
    from vocalance.app.services.audio.stt.stt_service import SpeechToTextService
    from vocalance.app.services.automation_service import AutomationService
    from vocalance.app.services.commands.management import CommandManagementService
    from vocalance.app.services.commands.parser import CentralizedCommandParser
    from vocalance.app.services.commands.utilities.input_executor import KeyboardInputService
    from vocalance.app.services.grid.click_tracker_service import ClickTrackerService
    from vocalance.app.services.grid.grid_service import GridService
    from vocalance.app.services.mark_service import MarkService
    from vocalance.app.services.pause_state_manager import PauseStateManager
    from vocalance.app.services.storage.runtime_configuration import RuntimeConfigurationStore
    from vocalance.app.services.storage.storage_service import StorageService

    return [
        ServiceSpec(name="input_service", factory=lambda c: KeyboardInputService(event_bus=c["event_bus"])),
        ServiceSpec(name="storage", factory=lambda c: StorageService(config=c["config"])),
        ServiceSpec(
            name="runtime_config",
            factory=lambda c: RuntimeConfigurationStore(event_bus=c["event_bus"], config=c["config"], storage=c["storage"]),
        ),
        ServiceSpec(name="validator", factory=_build_validator),
        ServiceSpec(name="grid", factory=lambda c: GridService(event_bus=c["event_bus"], config=c["config"])),
        ServiceSpec(
            name="automation",
            factory=lambda c: AutomationService(event_bus=c["event_bus"], config=c["config"], input_service=c["input_service"]),
        ),
        ServiceSpec(
            name="click_tracker",
            factory=lambda c: ClickTrackerService(
                event_bus=c["event_bus"],
                storage=c["storage"],
                gui_event_loop=c["gui_loop"],
                lifecycle=c["lifecycle"],
                ui_refresh_debounce_s=c["config"].grid.click_history_ui_refresh_debounce_s,
                persist_debounce_s=c["config"].grid.click_history_persist_debounce_s,
            ),
        ),
        ServiceSpec(
            name="mark",
            factory=lambda c: MarkService(
                event_bus=c["event_bus"],
                config=c["config"],
                storage=c["storage"],
                protected_terms_validator=c["validator"],
                input_service=c["input_service"],
            ),
        ),
        ServiceSpec(
            name="command_management",
            factory=lambda c: CommandManagementService(
                event_bus=c["event_bus"], storage=c["storage"], protected_terms_validator=c["validator"]
            ),
        ),
        ServiceSpec(name="pause_state_manager", factory=lambda c: PauseStateManager(event_bus=c["event_bus"])),
        ServiceSpec(
            name="centralized_parser",
            factory=lambda c: CentralizedCommandParser(
                event_bus=c["event_bus"],
                app_config=c["config"],
                storage=c["storage"],
                pause_state_manager=c["pause_state_manager"],
            ),
        ),
        ServiceSpec(name="stt", factory=lambda c: SpeechToTextService(event_bus=c["event_bus"], config=c["config"])),
        ServiceSpec(
            name="audio_capture",
            factory=lambda c: AudioCaptureService(event_bus=c["event_bus"], config=c["config"], main_event_loop=c["gui_loop"]),
        ),
        ServiceSpec(
            name="command_segmenter",
            factory=lambda c: CommandSegmenterService(event_bus=c["event_bus"], config=c["config"]),
        ),
        ServiceSpec(
            name="sound_segmenter",
            factory=lambda c: SoundSegmenterService(event_bus=c["event_bus"], config=c["config"]),
        ),
        ServiceSpec(
            name="dictation",
            factory=lambda c: DictationCoordinator(
                event_bus=c["event_bus"],
                config=c["config"],
                storage=c["storage"],
                gui_event_loop=c["gui_loop"],
                stt_service=c["stt"],
                input_service=c["input_service"],
                lifecycle=c["lifecycle"],
                cancel_token=c["cancel_token"],
            ),
        ),
        ServiceSpec(
            name="sound_service",
            factory=lambda c: SoundService(event_bus=c["event_bus"], config=c["config"], storage=c["storage"]),
        ),
    ]


def _moonshine_model_cache_ready() -> bool:
    """Return True if the Moonshine cache directory exists and is non-empty."""
    if importlib.util.find_spec("moonshine_voice.download_file") is None:
        return False
    from moonshine_voice.download_file import get_cache_dir

    cache_dir = get_cache_dir()
    return cache_dir.is_dir() and any(cache_dir.iterdir())


def _validate_critical_assets(config: GlobalAppConfig) -> bool:
    """Return False if required on-disk assets (e.g. Vosk) are missing."""
    vosk_path = config.asset_paths.get_vosk_model_path()
    if not os.path.exists(vosk_path):
        logger.critical(
            "Vosk model not found: %s - download from https://alphacephei.com/vosk/models",
            vosk_path,
        )
        return False
    return True


async def _run_initialization(
    services: SimpleNamespace,
    config: GlobalAppConfig,
    lifecycle: AppLifecycle,
    progress: StartupProgressTracker,
) -> None:
    """Run the staged async initialization for already-constructed services."""
    progress.start_step(step_name="Initializing storage...")
    progress.update_status_animated(status="Loading user settings")
    await services.runtime_config.initialize()
    progress.update_status_animated(status="Initializing click tracking")
    await services.click_tracker.initialize()
    await services.centralized_parser.initialize()
    progress.complete_step()

    progress.start_step(step_name="Starting audio processing...")

    yamnet_path = os.path.join(config.storage.sound_model_dir, "yamnet")
    yamnet_ready = (
        os.path.exists(yamnet_path)
        and os.path.exists(os.path.join(yamnet_path, "saved_model.pb"))
        and os.path.exists(os.path.join(yamnet_path, "variables"))
    )
    progress.update_status_animated(
        status="Initializing sound recognition"
        if yamnet_ready
        else "Loading YAMNet model. This should take 1-2 minutes on first use."
    )
    await services.sound_service.initialize()
    lifecycle.spawn(
        lifecycle.run_blocking(services.sound_service.recognizer.warm_start_esc50_samples, name="esc50-warmup"),
        name="esc50-warmup",
    )

    progress.update_status_animated(
        status="Initializing speech-to-text"
        if _moonshine_model_cache_ready()
        else "Fetching Moonshine STT model. This may take several minutes on first use."
    )
    await services.stt.initialize()

    progress.update_sub_step(sub_step_name="Preparing dictation system...")
    allow = config.local_llm_allowlist
    spec = allow.artifact_for(config.llm.selected_model_id) or allow.artifact_for(allow.default_id)
    if spec is not None:
        from vocalance.app.services.storage.llm_model_downloader import LLMModelDownloader

        downloader = LLMModelDownloader(config)
        if not downloader.model_bundle_complete(spec.gguf_filenames):
            progress.update_sub_step(
                sub_step_name="Fetching AI Model. First launch may take several minutes.",
                progress=0.35,
            )
            ok = await downloader.download_model_bundle(
                repo_id=spec.repo_id,
                filenames=list(spec.gguf_filenames),
                cancel_event=lifecycle.cancel_token.threading_event(),
            )
            if not ok:
                raise RuntimeError("Critical asset download failed: LLM model")

    progress.update_sub_step(sub_step_name="Initializing dictation", progress=0.55)
    if not await services.dictation.initialize():
        raise RuntimeError("Critical dictation initialization failed")

    progress.complete_step()


async def _show_terminal_message(startup_window: Optional[StartupWindow], message: str, hold_s: float) -> None:
    """Display a terminal message on the startup window briefly before tearing down."""
    if startup_window is not None:
        with contextlib.suppress(RuntimeError):
            startup_window.update_progress(0.0, message, animate=False)
    await asyncio.sleep(hold_s)


async def main() -> None:
    """Configure the runtime, run startup, await shutdown, then tear everything down."""
    logging.getLogger("numba").setLevel(logging.WARNING)

    qt_app = QApplication.instance()
    if qt_app is None:
        raise RuntimeError("QApplication instance is missing; create it before calling main()")

    lifecycle = AppLifecycle()
    lifecycle.install_signal_handlers()

    startup_window: Optional[StartupWindow] = None

    try:
        config = GlobalAppConfig()
        AppInfoConfig()
        setup_logging(config=config.logging)
        os.makedirs(config.storage.user_data_root, exist_ok=True)
        logger.info("Vocalance starting")

        qt_app.setQuitOnLastWindowClosed(False)

        icon_path = Path(config.asset_paths.icon_path) if config.asset_paths.icon_path else None
        icon_manager = WindowIconManager(icon_path=icon_path)
        if icon_manager.load_icon():
            icon_manager.apply_to_application(qt_app)
        else:
            logger.warning("Failed to load application icon")

        theme.load_fonts(config.asset_paths.fonts_dir)
        theme.load_stylesheet()
        theme.apply_stylesheet(qt_app)
        qt_app.setFont(theme.get_font(size="medium", weight="regular"))

        startup_window = StartupWindow(
            logger=logging.getLogger("StartupWindow"),
            asset_paths_config=config.asset_paths,
            lifecycle=lifecycle,
            icon_manager=icon_manager,
        )
        startup_window.show()
        qt_app.processEvents()

        if not _validate_critical_assets(config):
            await _show_terminal_message(startup_window, "Critical assets missing. Please check logs.", 3.0)
            return

        event_bus = EventBus()
        gui_loop = asyncio.get_running_loop()

        progress = StartupProgressTracker(startup_window=startup_window, total_steps=3)
        progress.start_step(step_name="Loading core components...")
        progress.update_status_animated(status="Initializing services")

        specs = _service_specs()
        ctx: Dict[str, Any] = {
            "event_bus": event_bus,
            "config": config,
            "gui_loop": gui_loop,
            "cancel_token": lifecycle.cancel_token,
            "lifecycle": lifecycle,
        }
        build_services(specs, ctx)
        services = SimpleNamespace(**{spec.name: ctx[spec.name] for spec in specs})

        lifecycle.register_resource(event_bus)
        register_services_for_teardown(specs, ctx, lifecycle)
        progress.complete_step()

        event_bus.start(gui_loop)

        init_task = asyncio.create_task(
            _run_initialization(services, config, lifecycle, progress),
            name="initialize-services",
        )
        lifecycle.register_init_task(init_task)
        try:
            await init_task
        except asyncio.CancelledError:
            logger.info("Initialization cancelled")
            await _show_terminal_message(startup_window, "Startup cancelled by user", 1.0)
            return
        except RuntimeError as exc:
            logger.critical("Critical initialization error: %s", exc)
            await _show_terminal_message(
                startup_window,
                "Initialization failed. Please check your internet connection and try again.",
                3.0,
            )
            return
        finally:
            lifecycle.clear_init_task()

        services.audio_capture.start()
        startup_window.update_progress(1.0, "Ready!", animate=False)
        await asyncio.sleep(0.5)

        from vocalance.app.ui.qt_main_window import VocalanceMainWindow

        main_window = VocalanceMainWindow(
            event_bus=event_bus,
            logger=logging.getLogger("MainWindow"),
            config=config,
            input_service=services.input_service,
            icon_manager=icon_manager,
            lifecycle=lifecycle,
        )
        lifecycle.register_resource(main_window)
        await services.click_tracker.publish_click_history_snapshot()
        main_window.show()
        main_window.raise_()
        main_window.activateWindow()
        startup_window.close_after_initialization()

        logger.info("Application running")
        await lifecycle.wait()
        logger.info("Shutdown signal received")

    except Exception:
        logger.exception("Unhandled error during application lifecycle")
    finally:
        if startup_window is not None:
            with contextlib.suppress(RuntimeError):
                startup_window.close()
        await lifecycle.teardown()
        with contextlib.suppress(Exception):
            qt_app.quit()
        logger.info("Application shutdown complete")
