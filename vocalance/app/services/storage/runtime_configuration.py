import asyncio
import copy
import logging
from typing import Any, Awaitable, Callable, Dict, List, Tuple

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.event_bus import EventBus
from vocalance.app.events.core_events import RuntimeConfigRequestEvent, RuntimeConfigResponseEvent, SettingsChangedEvent
from vocalance.app.services.base_service import Service
from vocalance.app.services.storage.storage_models import AppUserConfigDocument
from vocalance.app.services.storage.storage_service import StorageService
from vocalance.app.services.storage.user_configurable_settings import (
    ALLOWED_USER_SETTING_PATHS,
    LLM_SESSION_SETTING_PATHS,
    apply_dot_path_to_config,
    effective_settings_projection,
    sanitize_user_overrides,
)

logger = logging.getLogger(__name__)

ConfigurationListener = Callable[[frozenset[str]], Awaitable[None]]


class RuntimeConfigurationStore(Service):
    """Loads user overrides from disk, merges into GlobalAppConfig, persists updates, notifies listeners."""

    def __init__(self, event_bus: EventBus, config: GlobalAppConfig, storage: StorageService) -> None:
        self.event_bus = event_bus
        self.config = config
        self.storage = storage
        self.overrides: Dict[str, Dict[str, Any]] = {}
        self.listeners: List[Tuple[int, str, ConfigurationListener]] = []
        self.lock = asyncio.Lock()
        event_bus.subscribe(RuntimeConfigRequestEvent, self._handle_runtime_config_request)

    async def _handle_runtime_config_request(self, event: RuntimeConfigRequestEvent) -> None:
        op = event.op
        if op == "get_effective":
            all_settings = self.get_effective_settings()
            await self.event_bus.publish(RuntimeConfigResponseEvent(op="effective_snapshot", all_settings=all_settings))
        elif op == "update":
            ok = await self.update_multiple_settings(dict(event.updates))
            msg = "OK" if ok else "Update failed"
            await self.event_bus.publish(
                RuntimeConfigResponseEvent(
                    op="update_result",
                    correlation_id=event.correlation_id,
                    success=ok,
                    message=msg,
                )
            )
        elif op == "reset_defaults":
            await self.reset_to_defaults()
        elif op == "reset_section":
            for setting_key in event.setting_keys:
                await self.reset_setting(setting_key)

    def register_listener(self, order: int, name: str, listener: ConfigurationListener) -> None:
        self.listeners.append((order, name, listener))
        self.listeners.sort(key=lambda t: (t[0], t[1]))

    def current(self) -> GlobalAppConfig:
        return self.config

    async def shutdown(self) -> None:
        self.event_bus.unsubscribe(RuntimeConfigRequestEvent, self._handle_runtime_config_request)
        self.listeners.clear()

    async def initialize(self) -> bool:
        async with self.lock:
            doc = await self.storage.read(model_type=AppUserConfigDocument)
            self.overrides = sanitize_user_overrides(doc.overrides)
            changed: List[str] = []
            for category, items in self.overrides.items():
                if not isinstance(items, dict):
                    continue
                for key, value in items.items():
                    path = f"{category}.{key}"
                    if path not in ALLOWED_USER_SETTING_PATHS:
                        continue
                    try:
                        apply_dot_path_to_config(self.config, path, value)
                        changed.append(path)
                    except Exception as e:
                        logger.warning("Skipping invalid persisted override %s=%r: %s", path, value, e)

            await self.dispatch_configuration_listeners(frozenset(changed))
            if changed:
                await self.publish_settings_changed_event(frozenset(changed))
            logger.info("RuntimeConfigurationStore initialized (%d override paths)", len(changed))
            return True

    def get_effective_settings(self) -> Dict[str, Any]:
        return effective_settings_projection(self.config)

    def get_setting(self, setting_path: str, default: Any = None) -> Any:
        if setting_path not in ALLOWED_USER_SETTING_PATHS:
            return default
        try:
            category, key = setting_path.split(".", 1)
            return getattr(getattr(self.config, category), key)
        except AttributeError:
            return default

    async def update_multiple_settings(self, settings_updates: Dict[str, Any]) -> bool:
        for path in settings_updates:
            if path not in ALLOWED_USER_SETTING_PATHS:
                logger.warning("Setting %s is not user-configurable", path)
                return False

        async with self.lock:
            try:
                new_overrides = copy.deepcopy(self.overrides)
                for path, value in settings_updates.items():
                    category, key = path.split(".", 1)
                    new_overrides.setdefault(category, {})[key] = value

                test_cfg = self.config.model_copy(deep=True)
                for path, value in settings_updates.items():
                    apply_dot_path_to_config(test_cfg, path, value)

                if not await self.storage.write(AppUserConfigDocument(overrides=new_overrides)):
                    logger.error("Failed to persist user configuration")
                    return False

                self.overrides = new_overrides
                for path, value in settings_updates.items():
                    apply_dot_path_to_config(self.config, path, value)

                self.storage.clear_cache(AppUserConfigDocument)
                await self.dispatch_configuration_listeners(frozenset(settings_updates.keys()))
                await self.publish_settings_changed_event(frozenset(settings_updates.keys()))
                return True
            except Exception as e:
                logger.error("Failed to update settings: %s", e, exc_info=True)
                return False

    async def reset_setting(self, setting_path: str) -> bool:
        if setting_path not in ALLOWED_USER_SETTING_PATHS:
            return False

        async with self.lock:
            try:
                category, key = setting_path.split(".", 1)
                new_overrides = copy.deepcopy(self.overrides)
                if category in new_overrides and key in new_overrides[category]:
                    del new_overrides[category][key]
                    if not new_overrides[category]:
                        del new_overrides[category]

                fresh = GlobalAppConfig()
                default_val = getattr(getattr(fresh, category), key)
                test_cfg = self.config.model_copy(deep=True)
                apply_dot_path_to_config(test_cfg, setting_path, default_val)

                if not await self.storage.write(AppUserConfigDocument(overrides=new_overrides)):
                    return False

                self.overrides = new_overrides
                apply_dot_path_to_config(self.config, setting_path, default_val)
                self.storage.clear_cache(AppUserConfigDocument)
                await self.dispatch_configuration_listeners(frozenset([setting_path]))
                await self.publish_settings_changed_event(frozenset([setting_path]))
                return True
            except Exception as e:
                logger.error("Failed to reset setting %s: %s", setting_path, e, exc_info=True)
                return False

    async def reset_to_defaults(self) -> Tuple[bool, str]:
        async with self.lock:
            try:
                if not await self.storage.write(AppUserConfigDocument(overrides={})):
                    return False, "Failed to save reset configuration"
                self.overrides = {}
                fresh = GlobalAppConfig()
                for name in ("llm", "grid", "vad", "sound_recognizer"):
                    setattr(self.config, name, getattr(fresh, name))

                self.storage.clear_cache(AppUserConfigDocument)
                paths = frozenset(ALLOWED_USER_SETTING_PATHS)
                await self.dispatch_configuration_listeners(paths)
                await self.publish_settings_changed_event(paths)
                logger.info("All user configuration reset to code defaults")
                return True, "Settings reset to defaults successfully"
            except Exception as e:
                msg = f"Failed to reset settings: {e}"
                logger.error(msg, exc_info=True)
                return False, msg

    async def dispatch_configuration_listeners(self, paths: frozenset[str]) -> None:
        if not paths:
            return
        for order, name, fn in self.listeners:
            await fn(paths)

    async def publish_settings_changed_event(self, paths: frozenset[str]) -> None:
        resolved = {p: self.get_setting(p) for p in paths}
        all_settings = self.get_effective_settings()
        await self.event_bus.publish(SettingsChangedEvent(updated_settings=resolved, all_settings=all_settings))


def register_configuration_listeners(
    store: RuntimeConfigurationStore,
    *,
    sound_service: Any,
    audio_service: Any,
    llm_service: Any,
) -> None:
    """Register ordered listeners. Call after services exist, before ``event_bus.start(loop)``."""

    async def sound_listener(paths: frozenset[str]) -> None:
        cfg = store.current()
        if "sound_recognizer.confidence_threshold" in paths:
            sound_service.on_confidence_threshold_updated(cfg.sound_recognizer.confidence_threshold)
        if "sound_recognizer.vote_threshold" in paths:
            sound_service.on_vote_threshold_updated(cfg.sound_recognizer.vote_threshold)

    async def audio_listener(paths: frozenset[str]) -> None:
        cfg = store.current()
        if "vad.command_silent_chunks_for_end" in paths:
            audio_service.on_command_silent_chunks_updated(cfg.vad.command_silent_chunks_for_end)

    async def llm_listener(paths: frozenset[str]) -> None:
        if paths & LLM_SESSION_SETTING_PATHS:
            await llm_service.dispose_loaded_model()

    store.register_listener(10, "sound", sound_listener)
    store.register_listener(20, "audio", audio_listener)
    store.register_listener(30, "llm", llm_listener)
