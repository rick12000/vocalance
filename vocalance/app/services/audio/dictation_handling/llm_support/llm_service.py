import asyncio
import gc
import logging
import multiprocessing
import os
import threading
from typing import Callable, Dict, List, Optional, Tuple

from llama_cpp import Llama

from vocalance.app.config.app_config import DEFAULT_LLM_MODEL_ID, GlobalAppConfig, LocalLLMArtifact, get_whitelisted_llm_model
from vocalance.app.event_bus import EventBus
from vocalance.app.events.dictation_events import LLMProcessingCompletedEvent, LLMProcessingFailedEvent
from vocalance.app.services.storage.llm_model_downloader import LLMModelDownloader

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

    def __init__(self, event_bus: EventBus, config: GlobalAppConfig) -> None:
        self.event_bus = event_bus
        self.config = config
        self.llm: Optional[Llama] = None
        self._model_loaded = False

        self.model_downloader = LLMModelDownloader(config)

        cpu_count = multiprocessing.cpu_count()
        self.n_threads = config.llm.n_threads if config.llm.n_threads else max(4, min(int(cpu_count * 0.75), 12))
        self.n_threads_batch = config.llm.n_threads_batch if config.llm.n_threads_batch else self.n_threads

        self._request_lock = asyncio.Lock()

        logger.debug("LLMService initialized")

    def _active_spec(self) -> Optional[LocalLLMArtifact]:
        spec = get_whitelisted_llm_model(self.config.llm.selected_model_id)
        if spec:
            return spec
        return get_whitelisted_llm_model(DEFAULT_LLM_MODEL_ID)

    async def initialize(self) -> bool:
        return True

    def _load_model(self, model_path: str) -> Optional[Llama]:
        try:
            cfg = self.config.llm
            model = Llama(
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
            logger.info("Model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Model load error: {e}", exc_info=True)
            return None

    async def _dispose_loaded_model(self) -> None:
        if not self.llm:
            self._model_loaded = False
            return
        try:
            async with asyncio.timeout(5.0):
                loop = asyncio.get_event_loop()
                if hasattr(self.llm, "close"):
                    await loop.run_in_executor(None, self.llm.close)
        except asyncio.TimeoutError:
            logger.warning("LLM close timed out after 5s")
        except Exception as e:
            logger.warning(f"Error closing LLM model: {e}")
        finally:
            self.llm = None
            self._model_loaded = False
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
        except Exception as e:
            logger.error(f"LLM download error: {e}", exc_info=True)
            self.model_downloader.revert_partial_bundle(fns, had_before)
            return False, str(e)

    async def _run_chat_completion(
        self,
        messages: List[Dict[str, str]],
        fallback_text: str,
        agentic_prompt: str,
        token_callback: Optional[Callable[[str], None]],
    ) -> Optional[str]:
        """Run one completion, emit bus events, return final text."""
        async with self._request_lock:
            spec = self._active_spec()
            if not spec:
                logger.error("No valid whitelisted LLM configuration")
                await self._publish_failed("Invalid LLM configuration", fallback_text)
                return fallback_text.strip()

            if not self.model_downloader.model_bundle_complete(spec.gguf_filenames):
                msg = "LLM model files missing. Download the selected model in Settings."
                logger.error(msg)
                await self._publish_failed(msg, fallback_text)
                return fallback_text.strip()

            model_path = self.model_downloader.get_model_path(spec.load_path_filename)
            if not os.path.exists(model_path) or os.path.getsize(model_path) <= 0:
                await self._publish_failed("LLM model file missing or empty", fallback_text)
                return fallback_text.strip()

            loop = asyncio.get_event_loop()
            self.llm = await loop.run_in_executor(None, self._load_model, model_path)
            if not self.llm:
                self._model_loaded = False
                await self._publish_failed("LLM failed to load", fallback_text)
                return fallback_text.strip()

            self._model_loaded = True
            try:
                result = await self._generate_streaming(messages, token_callback)
                final_result = result if result else fallback_text.strip()
                await self._publish_completed(final_result, agentic_prompt)
                return final_result
            except Exception as e:
                logger.error(f"Processing error: {e}", exc_info=True)
                await self._publish_failed(str(e), fallback_text)
                return fallback_text.strip()
            finally:
                await self._dispose_loaded_model()

    async def process_dictation_streaming(
        self, raw_text: str, agentic_prompt: str, token_callback: Optional[Callable[[str], None]] = None
    ) -> Optional[str]:
        messages = self._build_messages(agentic_prompt, raw_text)
        return await self._run_chat_completion(messages, raw_text, agentic_prompt, token_callback)

    async def process_dictation(self, raw_text: str, agentic_prompt: str) -> Optional[str]:
        return await self.process_dictation_streaming(raw_text, agentic_prompt, None)

    async def process_amend_streaming(
        self,
        clipboard_text: str,
        spoken_prompt: str,
        agentic_prompt: str,
        token_callback: Optional[Callable[[str], None]] = None,
    ) -> Optional[str]:
        messages = self._build_amend_messages(agentic_prompt, clipboard_text, spoken_prompt)
        return await self._run_chat_completion(messages, spoken_prompt, agentic_prompt, token_callback)

    def _build_messages(self, agentic_prompt: str, raw_text: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": f"{agentic_prompt}"},
            {"role": "user", "content": raw_text},
        ]

    def _build_amend_messages(self, agentic_prompt: str, clipboard_text: str, spoken_prompt: str) -> List[Dict[str, str]]:
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

    async def _generate_streaming(
        self, messages: List[Dict[str, str]], token_callback: Optional[Callable[[str], None]] = None
    ) -> Optional[str]:
        try:
            cfg = self.config.llm
            loop = asyncio.get_event_loop()
            token_queue = asyncio.Queue(maxsize=50)
            full_text = []

            if not self.llm:
                return None

            llm_ref = self.llm

            def sync_generate():
                try:
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

                    token_count = 0
                    for chunk in stream:
                        if chunk and chunk.get("choices"):
                            delta = chunk["choices"][0].get("delta", {})
                            token = delta.get("content", "")
                            if token:
                                token_count += 1
                                if token_count <= 5 or token_count % 10 == 0:
                                    logger.debug(f"LLM generated token #{token_count}: '{token}'")
                                try:
                                    asyncio.run_coroutine_threadsafe(token_queue.put(token), loop)
                                except RuntimeError:
                                    logger.warning("Event loop closed during token streaming")
                                    break
                    logger.info(f"LLM generation completed: {token_count} tokens generated")

                    try:
                        asyncio.run_coroutine_threadsafe(token_queue.put(None), loop)
                    except RuntimeError:
                        logger.warning("Event loop closed during streaming completion")

                except Exception as e:
                    logger.error(f"Generation error: {e}", exc_info=True)
                    try:
                        asyncio.run_coroutine_threadsafe(token_queue.put(None), loop)
                    except RuntimeError:
                        pass

            executor_task = loop.run_in_executor(None, sync_generate)

            try:
                callback_count = 0
                while True:
                    token = await asyncio.wait_for(token_queue.get(), timeout=cfg.generation_timeout_sec)
                    if token is None:
                        logger.debug(f"Token stream ended (received {callback_count} tokens)")
                        break

                    full_text.append(token)
                    if token_callback:
                        try:
                            callback_count += 1
                            if callback_count <= 5 or callback_count % 10 == 0:
                                logger.debug(f"Calling token_callback #{callback_count} with: '{token}'")
                            token_callback(token)
                        except Exception as e:
                            logger.error(f"Token callback error: {e}", exc_info=True)

                await executor_task
                result = "".join(full_text).strip()
                return result if result else None

            except asyncio.TimeoutError:
                logger.warning(f"Timeout after {cfg.generation_timeout_sec}s")
                return None

        except Exception as e:
            logger.error(f"Generation error: {e}", exc_info=True)
            return None

    async def _publish_completed(self, processed_text: str, agentic_prompt: str) -> None:
        event = LLMProcessingCompletedEvent(processed_text=processed_text, agentic_prompt=agentic_prompt)
        await self.event_bus.publish(event)

    async def _publish_failed(self, error_message: str, original_text: str) -> None:
        event = LLMProcessingFailedEvent(error_message=error_message, original_text=original_text)
        await self.event_bus.publish(event)

    def is_ready(self) -> bool:
        return self._model_loaded and self.llm is not None

    async def shutdown(self) -> None:
        try:
            logger.info("LLM service shutting down")
            await self._dispose_loaded_model()
            for _ in range(2):
                gc.collect()
            logger.info("LLM service shutdown complete")
        except Exception as e:
            logger.error(f"Shutdown error: {e}", exc_info=True)
