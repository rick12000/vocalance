import asyncio
import gc
import logging
import multiprocessing
import os
import threading
from typing import AsyncGenerator, Callable, Dict, List, Optional, Tuple

from llama_cpp import Llama

from vocalance.app.config.app_config import DEFAULT_LLM_MODEL_ID, GlobalAppConfig, LocalLLMArtifact, get_whitelisted_llm_model
from vocalance.app.event_bus import EventBus
from vocalance.app.events.core_events import LlmUiNotificationEvent, LlmUiRequestEvent
from vocalance.app.events.dictation_events import LLMProcessingCompletedEvent, LLMProcessingFailedEvent, LLMTokenGeneratedEvent
from vocalance.app.services.storage.llm_model_downloader import LLMModelDownloader
from vocalance.app.utils.concurrency import schedule_on_loop

logger = logging.getLogger(__name__)

_AMEND_SYSTEM_BASE = (
    "You transform text according to instructions. The user message contains two labeled parts: "
    "TEXT TO TRANSFORM (the source material copied from the user's selection) and "
    "USER PROMPT (instructions they spoke).\n\n"
    "Apply USER PROMPT to TEXT TO TRANSFORM. "
    "Respond with only the transformed text: no explanations, preamble, markdown fences, "
    "headings, apologies, or any text before or after the result."
)


class LLMService:
    """llama.cpp CPU inference: load GGUF per request, unload after."""

    def __init__(
        self,
        event_bus: EventBus,
        config: GlobalAppConfig,
        gui_event_loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        self.event_bus = event_bus
        self.config = config
        self._gui_event_loop = gui_event_loop
        self.llm: Optional[Llama] = None
        self.model_loaded = False
        self.model_downloader = LLMModelDownloader(config)
        cpu_count = multiprocessing.cpu_count()
        self.n_threads = config.llm.n_threads if config.llm.n_threads else max(4, min(int(cpu_count * 0.75), 12))
        self.n_threads_batch = config.llm.n_threads_batch if config.llm.n_threads_batch else self.n_threads
        self.request_lock = asyncio.Lock()
        self._download_cancel_events: Dict[str, threading.Event] = {}
        self._active_download_request_id: Optional[str] = None
        event_bus.subscribe(LlmUiRequestEvent, self._handle_llm_ui_request)

    def _download_progress_cb(self, request_id: str) -> Optional[Callable[[str], None]]:
        if self._gui_event_loop is None or self._gui_event_loop.is_closed():
            return None

        def cb(message: str) -> None:
            async def pub() -> None:
                await self.event_bus.publish(
                    LlmUiNotificationEvent(kind="download_progress", request_id=request_id, message=message)
                )

            schedule_on_loop(self._gui_event_loop, pub())

        return cb

    async def _handle_llm_ui_request(self, event: LlmUiRequestEvent) -> None:
        op = event.op
        if op == "refresh_bundle_status":
            status = {a.id: self.is_whitelisted_bundle_on_disk(a.id) for a in self.config.local_llm_allowlist.artifacts}
            await self.event_bus.publish(LlmUiNotificationEvent(kind="bundle_status", status=status))
        elif op == "start_download":
            cancel = threading.Event()
            self._download_cancel_events[event.request_id] = cancel
            self._active_download_request_id = event.request_id
            progress_cb = self._download_progress_cb(event.request_id)
            try:
                ok, msg = await self.download_whitelisted_model_cancellable(event.model_id, cancel, progress_cb)
            except Exception as e:
                ok, msg = False, str(e)
            await self.event_bus.publish(
                LlmUiNotificationEvent(kind="download_finished", request_id=event.request_id, ok=ok, message=msg)
            )
            self._download_cancel_events.pop(event.request_id, None)
            if self._active_download_request_id == event.request_id:
                self._active_download_request_id = None
        elif op == "cancel_download":
            rid = event.request_id or self._active_download_request_id
            if not rid:
                return
            ev = self._download_cancel_events.get(rid)
            if ev is not None:
                ev.set()

    def active_spec(self) -> Optional[LocalLLMArtifact]:
        spec = get_whitelisted_llm_model(self.config.llm.selected_model_id)
        if spec:
            return spec
        return get_whitelisted_llm_model(DEFAULT_LLM_MODEL_ID)

    def initialize(self) -> bool:
        return True

    def load_model(self, model_path: str) -> Llama:
        cfg = self.config.llm
        return Llama(
            model_path=model_path,
            n_ctx=cfg.context_length,
            n_threads=self.n_threads,
            n_threads_batch=self.n_threads_batch,
            n_batch=cfg.n_batch,
            n_gpu_layers=0,
            flash_attn=cfg.flash_attn,
            use_mmap=True,
            use_mlock=cfg.use_mlock,
            chat_format="chatml",
            seed=-1,
            type_k=cfg.type_k,
            type_v=cfg.type_v,
            verbose=cfg.verbose,
        )

    async def dispose_loaded_model(self) -> None:
        if not self.llm:
            self.model_loaded = False
            return
        llm_ref = self.llm
        self.llm = None
        self.model_loaded = False
        try:
            async with asyncio.timeout(5.0):
                loop = asyncio.get_running_loop()
                if hasattr(llm_ref, "close"):
                    await loop.run_in_executor(None, llm_ref.close)
        except asyncio.TimeoutError:
            logger.warning("LLM close timed out after 5s")
        gc.collect()

    def is_whitelisted_bundle_on_disk(self, model_id: str) -> bool:
        spec = get_whitelisted_llm_model(model_id)
        if not spec:
            return False
        return self.model_downloader.model_bundle_complete(spec.gguf_filenames)

    async def download_whitelisted_model_cancellable(
        self,
        model_id: str,
        cancel_event: threading.Event,
        progress_message_cb: Optional[Callable[[str], None]] = None,
    ) -> Tuple[bool, str]:
        spec = get_whitelisted_llm_model(model_id)
        if not spec:
            return False, f"Unknown model id: {model_id}"
        fns = list(spec.gguf_filenames)
        had_before = {fn: self.model_downloader.model_exists(fn) for fn in fns}
        try:
            primary = await self.model_downloader.download_model_bundle(
                repo_id=spec.repo_id,
                filenames=fns,
                force_download=False,
                max_retries=1,
                retry_delay_seconds=0,
                cancel_event=cancel_event,
                progress_message_cb=progress_message_cb,
            )
            if cancel_event.is_set():
                self.model_downloader.revert_partial_bundle(fns, had_before)
                return False, "Download cancelled"
            if primary:
                return True, f"Downloaded: {spec.label}"
            self.model_downloader.revert_partial_bundle(fns, had_before)
            return False, f"Download failed for {spec.label}"
        except Exception:
            self.model_downloader.revert_partial_bundle(fns, had_before)
            raise

    async def run_chat_completion(
        self,
        messages: List[Dict[str, str]],
        fallback_text: str,
        agentic_prompt: str,
        stream_session_id: Optional[str],
    ) -> Optional[str]:
        async with self.request_lock:
            spec = self.active_spec()
            if not spec:
                await self.emit_failed("Invalid LLM configuration", fallback_text)
                return fallback_text.strip()

            if not self.model_downloader.model_bundle_complete(spec.gguf_filenames):
                msg = "LLM model files missing. Download the selected model in Settings."
                await self.emit_failed(msg, fallback_text)
                return fallback_text.strip()

            model_path = self.model_downloader.get_model_path(spec.load_path_filename)
            if not os.path.exists(model_path) or os.path.getsize(model_path) <= 0:
                await self.emit_failed("LLM model file missing or empty", fallback_text)
                return fallback_text.strip()

            loop = asyncio.get_running_loop()
            try:
                self.llm = await loop.run_in_executor(None, self.load_model, model_path)
            except Exception as e:
                await self.emit_failed(f"LLM failed to load: {e}", fallback_text)
                return fallback_text.strip()

            self.model_loaded = True
            try:
                full_text: List[str] = []
                async for token in self.generate_streaming(messages):
                    full_text.append(token)
                    if stream_session_id:
                        await self.event_bus.publish(LLMTokenGeneratedEvent(token=token, session_id=stream_session_id))

                result = "".join(full_text).strip()
                final_result = result if result else fallback_text.strip()
                await self.emit_completed(final_result, agentic_prompt)
                return final_result
            except Exception as e:
                await self.emit_failed(str(e), fallback_text)
                return fallback_text.strip()
            finally:
                await self.dispose_loaded_model()

    async def process_dictation_streaming(self, raw_text: str, agentic_prompt: str, stream_session_id: str) -> Optional[str]:
        messages = self.build_messages(agentic_prompt, raw_text)
        return await self.run_chat_completion(messages, raw_text, agentic_prompt, stream_session_id)

    async def process_dictation(self, raw_text: str, agentic_prompt: str) -> Optional[str]:
        messages = self.build_messages(agentic_prompt, raw_text)
        return await self.run_chat_completion(messages, raw_text, agentic_prompt, None)

    async def process_amend_streaming(
        self,
        clipboard_text: str,
        spoken_prompt: str,
        agentic_prompt: str,
        stream_session_id: str,
    ) -> Optional[str]:
        messages = self.build_amend_messages(agentic_prompt, clipboard_text, spoken_prompt)
        return await self.run_chat_completion(messages, spoken_prompt, agentic_prompt, stream_session_id)

    def build_messages(self, agentic_prompt: str, raw_text: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": f"{agentic_prompt}"},
            {"role": "user", "content": raw_text},
        ]

    def build_amend_messages(self, agentic_prompt: str, clipboard_text: str, spoken_prompt: str) -> List[Dict[str, str]]:
        extra = (agentic_prompt or "").strip()
        system_content = _AMEND_SYSTEM_BASE if not extra else f"{_AMEND_SYSTEM_BASE}\n\n{extra}"
        user_content = (
            "--- TEXT TO TRANSFORM (clipboard) ---\n"
            f"{clipboard_text}\n"
            "--- END TEXT ---\n\n"
            "--- USER PROMPT (spoken instructions) ---\n"
            f"{spoken_prompt}\n"
            "--- END USER PROMPT ---"
        )
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    async def generate_streaming(self, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        cfg = self.config.llm
        loop = asyncio.get_running_loop()
        token_queue: asyncio.Queue[Optional[str]] = asyncio.Queue(maxsize=50)

        if not self.llm:
            return

        llm_ref = self.llm

        def sync_generate() -> None:
            stream = llm_ref.create_chat_completion(
                messages=messages,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                min_p=cfg.min_p,
                repeat_penalty=cfg.repeat_penalty,
                frequency_penalty=cfg.frequency_penalty,
                mirostat_mode=cfg.mirostat_mode,
                mirostat_tau=cfg.mirostat_tau,
                mirostat_eta=cfg.mirostat_eta,
                stop=[],
                stream=True,
            )
            for chunk in stream:
                if chunk and chunk.get("choices"):
                    delta = chunk["choices"][0].get("delta", {})
                    token = delta.get("content", "")
                    if token:
                        loop.call_soon_threadsafe(token_queue.put_nowait, token)
            loop.call_soon_threadsafe(token_queue.put_nowait, None)

        executor_task = loop.run_in_executor(None, sync_generate)

        try:
            while True:
                token = await asyncio.wait_for(token_queue.get(), timeout=cfg.generation_timeout_sec)
                if token is None:
                    break
                yield token
        except asyncio.TimeoutError:
            pass
        finally:
            await executor_task

    async def emit_completed(self, processed_text: str, agentic_prompt: str) -> None:
        await self.event_bus.publish(LLMProcessingCompletedEvent(processed_text=processed_text, agentic_prompt=agentic_prompt))

    async def emit_failed(self, error_message: str, original_text: str) -> None:
        await self.event_bus.publish(LLMProcessingFailedEvent(error_message=error_message, original_text=original_text))

    def is_ready(self) -> bool:
        return self.model_loaded and self.llm is not None

    async def shutdown(self) -> None:
        self.event_bus.unsubscribe(LlmUiRequestEvent, self._handle_llm_ui_request)
        await self.dispose_loaded_model()
