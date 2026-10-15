import asyncio
import gc
import logging
import multiprocessing
import os
import sys
import threading
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Set, Tuple

from llama_cpp import Llama, llama_supports_gpu_offload

from vocalance.app.config.app_config import DEFAULT_LLM_MODEL_ID, GlobalAppConfig, LocalLLMArtifact, get_whitelisted_llm_model
from vocalance.app.event_bus import EventBus
from vocalance.app.events.core_events import LlmUiNotificationEvent, LlmUiNotificationKind, LlmUiRequestEvent, SettingsChangedEvent
from vocalance.app.events.dictation_events import LLMProcessingCompletedEvent, LLMProcessingFailedEvent
from vocalance.app.lifecycle.cancellation import CancellationToken
from vocalance.app.lifecycle.worker import run_blocking, schedule_on_loop
from vocalance.app.services.base_service import Service
from vocalance.app.services.dictation_flow.llm.llm_model_downloader import IntegrityError, LLMModelDownloader
from vocalance.app.services.dictation_flow.llm.dictation_messages import (
    build_dictation_chat_messages,
    is_dictation_formatter_messages,
)
from vocalance.app.services.dictation_flow.llm.dictation_llm_output import (
    dictation_stop_sequences_for_messages,
    postprocess_llm_completion,
)
from vocalance.app.services.dictation_flow.llm.local_endpoint import validate_local_endpoint_base_url
from vocalance.app.services.dictation_flow.llm.openai_compatible_client import (
    OpenAICompatibleChatError,
    chat_completions_sync,
)
from vocalance.app.services.dictation_flow.llm.formatter_output_rules import OUTPUT_SLOP_AND_FORMATTING_RULE
from vocalance.app.services.dictation_flow.llm.reasoning_stream import ReasoningTokenFilter
from vocalance.app.services.storage.user_configurable_settings import LLM_SESSION_SETTING_PATHS

logger = logging.getLogger(__name__)

_AMEND_SYSTEM_BASE = (
    "You transform text according to instructions. The user message contains two labeled parts.\n"
    "- TEXT TO TRANSFORM (clipboard selection, source material)\n"
    "- USER PROMPT (spoken instructions)\n\n"
    "Apply USER PROMPT to TEXT TO TRANSFORM.\n\n"
    "Reply with only the transformed text. No preamble, postscript, summary of edits, markdown "
    "fences, headings, apologies, or other meta commentary before or after the result."
)


class LLMService(Service):
    """Loads a GGUF model with llama.cpp (GPU when available) and unloads when idle."""

    def __init__(
        self,
        event_bus: EventBus,
        config: GlobalAppConfig,
        gui_event_loop: Optional[asyncio.AbstractEventLoop] = None,
        cancel_token: Optional[CancellationToken] = None,
    ) -> None:
        super().__init__(event_bus)
        self.config = config
        self.gui_event_loop = gui_event_loop
        self.cancel_token = cancel_token
        self.llm: Optional[Llama] = None
        self.loaded_spec: Optional[LocalLLMArtifact] = None
        self.model_loaded = False
        self.model_downloader = LLMModelDownloader(config)
        cpu_count = multiprocessing.cpu_count()
        self.n_threads = config.llm.n_threads if config.llm.n_threads else max(4, min(int(cpu_count * 0.75), 12))
        self.n_threads_batch = config.llm.n_threads_batch if config.llm.n_threads_batch else self.n_threads
        self.request_lock = asyncio.Lock()
        self.download_cancel_events: Dict[str, threading.Event] = {}
        self.active_download_request_id: Optional[str] = None
        self.active_download_tasks: Set[asyncio.Task[Any]] = set()
        self.subscribe(LlmUiRequestEvent, self.handle_llm_ui_request)
        self.subscribe(SettingsChangedEvent, self._handle_settings_changed)

    async def _handle_settings_changed(self, event: SettingsChangedEvent) -> None:
        """Drop the in-memory model when any setting that invalidates the session changes."""
        if event.updated_settings.keys() & LLM_SESSION_SETTING_PATHS:
            await self.dispose_loaded_model()

    def download_progress_callback(self, request_id: str) -> Optional[Callable[[str], None]]:
        if self.gui_event_loop is None or self.gui_event_loop.is_closed():
            return None

        def cb(message: str) -> None:
            async def pub() -> None:
                await self.event_bus.publish(
                    LlmUiNotificationEvent(kind="download_progress", request_id=request_id, message=message)
                )

            schedule_on_loop(self.gui_event_loop, pub())

        return cb

    async def handle_llm_ui_request(self, event: LlmUiRequestEvent) -> None:
        op = event.op
        if op == "refresh_bundle_status":
            status = {a.id: self.is_whitelisted_bundle_on_disk(a.id) for a in self.config.local_llm_allowlist.artifacts}
            await self.event_bus.publish(LlmUiNotificationEvent(kind="bundle_status", status=status))
        elif op == "start_download":
            cancel = threading.Event()
            unlink = self.cancel_token.link_event(cancel) if self.cancel_token is not None else (lambda: None)
            self.download_cancel_events[event.request_id] = cancel
            self.active_download_request_id = event.request_id
            progress_cb = self.download_progress_callback(event.request_id)
            task = asyncio.create_task(
                self._run_download_workflow(event, cancel, progress_cb, unlink),
                name=f"llm-download-{event.request_id}",
            )
            self.active_download_tasks.add(task)
            task.add_done_callback(self.active_download_tasks.discard)
        elif op == "cancel_download":
            rid = event.request_id or self.active_download_request_id
            if not rid:
                return
            ev = self.download_cancel_events.get(rid)
            if ev is not None:
                ev.set()
        elif op == "test_local_endpoint":
            schedule_on_loop(asyncio.get_running_loop(), self._run_endpoint_test(event.request_id))

    async def _run_download_workflow(
        self,
        event: LlmUiRequestEvent,
        cancel: threading.Event,
        progress_cb: Optional[Callable[[str], None]],
        unlink: Callable[[], None],
    ) -> None:
        """Drive one whitelisted-model download to completion off the bus dispatcher.

        The handler that triggers this returns immediately so the event bus stays
        free to dispatch other events (audio chunks, dictation, settings) while
        the download runs for minutes.
        """
        try:
            try:
                ok, msg = await self.download_whitelisted_model_cancellable(event.model_id, cancel, progress_cb)
            except IntegrityError as e:
                logger.error("Integrity check failed: %s", e)
                await self.event_bus.publish(
                    LlmUiNotificationEvent(
                        kind="download_integrity_error",
                        request_id=event.request_id,
                        ok=False,
                        message=(
                            "The downloaded file's SHA-256 hash does not match the expected value. "
                            "The file has been deleted. This may indicate a compromised download "
                            "source or a corrupted transfer. Please try again or contact support."
                        ),
                    )
                )
                return
            except Exception:
                logger.exception("Whitelisted model download failed")
                ok, msg = False, "Download failed"

            kind: LlmUiNotificationKind = "download_cancelled" if not ok and cancel.is_set() else "download_finished"
            await self.event_bus.publish(LlmUiNotificationEvent(kind=kind, request_id=event.request_id, ok=ok, message=msg))
        finally:
            unlink()
            self.download_cancel_events.pop(event.request_id, None)
            if self.active_download_request_id == event.request_id:
                self.active_download_request_id = None

    def active_spec(self) -> Optional[LocalLLMArtifact]:
        spec = get_whitelisted_llm_model(self.config.llm.selected_model_id)
        if spec:
            return spec
        return get_whitelisted_llm_model(DEFAULT_LLM_MODEL_ID)

    def initialize(self) -> bool:
        return True

    def load_model(self, model_path: str, spec: LocalLLMArtifact) -> Llama:
        cfg = self.config.llm
        model: Llama
        n_gpu_layers = 0
        for i, attempt_layers in enumerate((-1, 0)):
            try:
                model = Llama(
                    model_path=model_path,
                    n_ctx=cfg.context_length,
                    n_threads=self.n_threads,
                    n_threads_batch=self.n_threads_batch,
                    n_batch=cfg.n_batch,
                    n_gpu_layers=attempt_layers,
                    flash_attn=cfg.flash_attn,
                    use_mmap=True,
                    use_mlock=cfg.use_mlock,
                    chat_format=None,
                    seed=-1,
                    type_k=cfg.type_k,
                    type_v=cfg.type_v,
                    verbose=cfg.verbose,
                )
                n_gpu_layers = attempt_layers
                break
            except Exception:
                if i == 1:
                    raise
                logger.warning("LLM GPU offload failed; falling back to CPU", exc_info=True)
        if n_gpu_layers == -1 and llama_supports_gpu_offload():
            logger.info("LLM loaded with GPU offload")
        elif n_gpu_layers == -1:
            logger.warning(
                "LLM requested GPU offload but this llama-cpp-python build has no GPU backend; running on CPU. "
                "Install a CUDA/Vulkan build into the Python that runs Vocalance (%s).",
                sys.executable,
            )
        else:
            logger.info("LLM loaded on CPU (n_gpu_layers=0)")
        self.bind_jinja_chat_template(model, spec)
        return model

    @staticmethod
    def bind_jinja_chat_template(model: Llama, spec: LocalLLMArtifact) -> None:
        """Use the GGUF Jinja chat template and pass the model's thinking flag."""
        handler = model._chat_handlers.get("chat_template.default")
        if handler is None:
            logger.error(
                "GGUF has no Jinja chat template (chat_format=%s). System instructions may not reach the model.",
                model.chat_format,
            )
            return
        template_kwargs = spec.chat_template_kwargs()

        def chat_handler(**kwargs: Any) -> Any:
            return handler(**{**template_kwargs, **kwargs})

        model.chat_format = "chat_template.default"
        model.chat_handler = chat_handler

    async def dispose_loaded_model(self) -> None:
        self.loaded_spec = None
        if not self.llm:
            self.model_loaded = False
            return
        llm_ref = self.llm
        self.llm = None
        self.model_loaded = False
        try:
            async with asyncio.timeout(5.0):
                if hasattr(llm_ref, "close"):
                    await run_blocking(llm_ref.close, name="llm-close")
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

    async def _run_endpoint_test(self, request_id: str) -> None:
        task = asyncio.current_task()
        if task is not None:
            self.active_download_tasks.add(task)
            task.add_done_callback(self.active_download_tasks.discard)
        ok, msg = await self.test_local_endpoint()
        kind: LlmUiNotificationKind = "endpoint_test_finished"
        await self.event_bus.publish(
            LlmUiNotificationEvent(kind=kind, request_id=request_id, ok=ok, message=msg)
        )

    async def test_local_endpoint(self) -> Tuple[bool, str]:
        cfg = self.config.llm
        if not cfg.uses_local_endpoint():
            return False, "Configure a loopback endpoint URL and model name first."
        try:
            validate_local_endpoint_base_url(cfg.local_endpoint_base_url)
        except ValueError as exc:
            return False, str(exc)
        messages = [{"role": "user", "content": "Reply with the single word: ok"}]
        try:
            await run_blocking(
                chat_completions_sync,
                cfg.local_endpoint_base_url,
                cfg.local_endpoint_model.strip(),
                messages,
                16,
                cfg.local_endpoint_timeout_sec,
                None,
                name="llm-endpoint-test",
            )
            return True, "Endpoint responded successfully."
        except OpenAICompatibleChatError as exc:
            return False, str(exc)
        except Exception as exc:
            return False, f"Endpoint request failed: {exc}"

    async def _complete_via_local_endpoint(
        self,
        messages: List[Dict[str, str]],
        fallback_text: str,
        agentic_prompt: str,
    ) -> Optional[str]:
        cfg = self.config.llm
        stop_sequences = dictation_stop_sequences_for_messages(messages)
        try:
            raw = await run_blocking(
                chat_completions_sync,
                cfg.local_endpoint_base_url,
                cfg.local_endpoint_model.strip(),
                messages,
                cfg.max_tokens,
                cfg.local_endpoint_timeout_sec,
                stop_sequences or None,
                name="llm-endpoint-chat",
            )
        except OpenAICompatibleChatError as exc:
            await self.emit_failed(str(exc), fallback_text)
            return fallback_text.strip()
        except Exception as exc:
            await self.emit_failed(f"Local endpoint request failed: {exc}", fallback_text)
            return fallback_text.strip()

        result = postprocess_llm_completion(raw, messages, start_inside_reasoning=False)
        final_result = result if result else fallback_text.strip()
        await self.emit_completed(final_result, agentic_prompt)
        return final_result

    async def _complete_via_shipped_model(
        self,
        messages: List[Dict[str, str]],
        fallback_text: str,
        agentic_prompt: str,
    ) -> Optional[str]:
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

        try:
            self.llm = await run_blocking(self.load_model, model_path, spec, name="llm-load")
            self.loaded_spec = spec
        except Exception as e:
            await self.emit_failed(f"LLM failed to load: {e}", fallback_text)
            return fallback_text.strip()

        self.model_loaded = True
        stop_sequences = dictation_stop_sequences_for_messages(messages)
        try:
            full_text: List[str] = []
            async for token in self.generate_streaming(messages, stop=stop_sequences):
                full_text.append(token)

            raw = "".join(full_text)
            result = postprocess_llm_completion(
                raw,
                messages,
                start_inside_reasoning=spec.stream_starts_in_reasoning,
            )
            final_result = result if result else fallback_text.strip()
            await self.emit_completed(final_result, agentic_prompt)
            return final_result
        except Exception as e:
            await self.emit_failed(str(e), fallback_text)
            return fallback_text.strip()
        finally:
            await self.dispose_loaded_model()

    async def run_chat_completion(
        self,
        messages: List[Dict[str, str]],
        fallback_text: str,
        agentic_prompt: str,
        _stream_session_id: Optional[str],
    ) -> Optional[str]:
        async with self.request_lock:
            if self.config.llm.provider == "local_endpoint":
                if not self.config.llm.uses_local_endpoint():
                    await self.emit_failed(
                        "Local endpoint mode is enabled but URL or model name is missing.",
                        fallback_text,
                    )
                    return fallback_text.strip()
                return await self._complete_via_local_endpoint(messages, fallback_text, agentic_prompt)
            return await self._complete_via_shipped_model(messages, fallback_text, agentic_prompt)

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
        return build_dictation_chat_messages(agentic_prompt, raw_text)

    def build_amend_messages(self, agentic_prompt: str, clipboard_text: str, spoken_prompt: str) -> List[Dict[str, str]]:
        extra = (agentic_prompt or "").strip()
        system_parts = [_AMEND_SYSTEM_BASE, OUTPUT_SLOP_AND_FORMATTING_RULE]
        if extra:
            system_parts.append(extra)
        system_content = "\n\n".join(system_parts)
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

    async def generate_streaming(
        self, messages: List[Dict[str, str]], stop: Optional[List[str]] = None
    ) -> AsyncGenerator[str, None]:
        cfg = self.config.llm
        spec = self.loaded_spec or self.active_spec()
        if spec is None or self.llm is None:
            return
        sampling = (
            spec.formatter_sampling_profile()
            if is_dictation_formatter_messages(messages)
            else spec.amend_sampling_profile()
        )
        reasoning_filter = ReasoningTokenFilter(start_inside=spec.stream_starts_in_reasoning)
        loop = asyncio.get_running_loop()
        token_queue: asyncio.Queue[Optional[str]] = asyncio.Queue(maxsize=50)
        llm_ref = self.llm

        def sync_generate() -> None:
            stream = llm_ref.create_chat_completion(
                messages=messages,
                max_tokens=cfg.max_tokens,
                temperature=sampling.temperature,
                top_p=sampling.top_p,
                top_k=sampling.top_k,
                min_p=sampling.min_p,
                repeat_penalty=sampling.repeat_penalty,
                presence_penalty=sampling.presence_penalty,
                frequency_penalty=sampling.frequency_penalty,
                mirostat_mode=cfg.mirostat_mode,
                mirostat_tau=cfg.mirostat_tau,
                mirostat_eta=cfg.mirostat_eta,
                stop=stop or [],
                stream=True,
            )
            for chunk in stream:
                if not chunk or not chunk.get("choices"):
                    continue
                delta = chunk["choices"][0].get("delta") or {}
                token = delta.get("content") or ""
                if token:
                    loop.call_soon_threadsafe(token_queue.put_nowait, token)
            loop.call_soon_threadsafe(token_queue.put_nowait, None)

        executor_task = asyncio.create_task(run_blocking(sync_generate, name="llm-generate"))

        try:
            while True:
                token = await asyncio.wait_for(token_queue.get(), timeout=cfg.generation_timeout_sec)
                if token is None:
                    break
                visible = reasoning_filter.push(token)
                if visible:
                    yield visible
            tail = reasoning_filter.finish()
            if tail:
                yield tail
        except asyncio.TimeoutError:
            logger.debug("LLM token stream timed out waiting for next token")
            tail = reasoning_filter.finish()
            if tail:
                yield tail
        finally:
            await executor_task

    async def emit_completed(self, processed_text: str, agentic_prompt: str) -> None:
        await self.event_bus.publish(LLMProcessingCompletedEvent(processed_text=processed_text, agentic_prompt=agentic_prompt))

    async def emit_failed(self, error_message: str, original_text: str) -> None:
        await self.event_bus.publish(LLMProcessingFailedEvent(error_message=error_message, original_text=original_text))

    def is_ready(self) -> bool:
        return self.model_loaded and self.llm is not None

    async def shutdown(self) -> None:
        for task in list(self.active_download_tasks):
            task.cancel()
        for ev in self.download_cancel_events.values():
            ev.set()
        if self.active_download_tasks:
            await asyncio.gather(*self.active_download_tasks, return_exceptions=True)
        await self.dispose_loaded_model()
        if hasattr(self, "model_downloader") and self.model_downloader:
            self.model_downloader.shutdown()
        await super().shutdown()
