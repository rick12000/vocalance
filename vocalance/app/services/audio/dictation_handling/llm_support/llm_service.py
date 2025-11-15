import asyncio
import gc
import logging
import multiprocessing
import os
from typing import Callable, Dict, List, Optional

from llama_cpp import Llama

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.event_bus import EventBus
from vocalance.app.events.dictation_events import LLMProcessingCompletedEvent, LLMProcessingFailedEvent
from vocalance.app.services.storage.llm_model_downloader import LLMModelDownloader

logger = logging.getLogger(__name__)


class LLMService:
    """High-performance LLM service optimized for speed and quality with proper resource management.

    Manages local LLM model (llama.cpp) for smart dictation formatting and editing.
    Handles model downloading, loading with performance optimizations (threading, GPU
    layers, flash attention), warm-up, streaming generation, and resource cleanup.

    Attributes:
        llm: Loaded Llama model instance.
        _model_loaded: Flag indicating successful model load.
        _warmed_up: Flag indicating model warm-up completion.
        model_downloader: LLMModelDownloader for model acquisition.
    """

    def __init__(self, event_bus: EventBus, config: GlobalAppConfig) -> None:
        """Initialize LLM service with configuration.

        Args:
            event_bus: EventBus for pub/sub messaging.
            config: Global application configuration.
        """
        self.event_bus = event_bus
        self.config = config
        self.llm: Optional[Llama] = None
        self._model_loaded = False
        self._warmed_up = False

        self.model_downloader = LLMModelDownloader(config)
        self.model_info = config.llm.model_info
        self.model_filename = config.llm.get_model_filename()
        self.model_path: Optional[str] = None

        cpu_count = multiprocessing.cpu_count()
        self.n_threads = config.llm.n_threads if config.llm.n_threads else max(4, min(int(cpu_count * 0.75), 12))
        self.n_threads_batch = config.llm.n_threads_batch if config.llm.n_threads_batch else self.n_threads

        logger.debug(f"LLMService initialized: {self.model_filename}")

    async def initialize(self) -> bool:
        """Initialize LLM model with atomic download and retry logic.

        Downloads model if not present, loads with performance optimizations, and
        performs warm-up inference to prepare for real requests.

        Returns:
            True if initialization and warm-up successful, False otherwise.
        """
        try:
            if not self.model_downloader.model_exists(self.model_filename):
                logger.debug(f"Downloading model: {self.model_filename}")
                model_path = await self.model_downloader.download_model(
                    repo_id=self.model_info["repo_id"], filename=self.model_info["filename"]
                )
                if not model_path:
                    logger.error("Model download failed after all retry attempts")
                    return False

            self.model_path = self.model_downloader.get_model_path(self.model_filename)

            if not os.path.exists(self.model_path):
                logger.error(f"Model not found: {self.model_path}")
                return False

            logger.debug(f"Loading model: {os.path.basename(self.model_path)}")

            loop = asyncio.get_event_loop()
            self.llm = await loop.run_in_executor(None, self._load_model, self.model_path)

            if self.llm:
                self._model_loaded = True
                logger.debug("Model loaded successfully, warming up...")
                await self._warmup_model()
                self._warmed_up = True
                logger.debug("Model initialization and warmup complete")
                return True

            logger.error("Model loading failed")
            return False

        except Exception as e:
            logger.error(f"Initialization error: {e}", exc_info=True)
            return False

    def _load_model(self, model_path: str) -> Optional[Llama]:
        """Load model with performance-optimized settings.

        Configures llama.cpp with threading, GPU layers, flash attention, quantization
        types, and memory mapping for optimal inference performance.

        Args:
            model_path: Path to GGUF model file.

        Returns:
            Loaded Llama model instance if successful, None otherwise.
        """
        try:
            cfg = self.config.llm
            model = Llama(
                model_path=model_path,
                n_ctx=cfg.context_length,
                n_threads=self.n_threads,
                n_threads_batch=self.n_threads_batch,
                n_batch=cfg.n_batch,
                n_gpu_layers=cfg.n_gpu_layers,
                flash_attn=cfg.flash_attn,
                use_mmap=True,
                use_mlock=cfg.use_mlock,
                chat_format="chatml",
                seed=42,
                type_k=cfg.type_k,
                type_v=cfg.type_v,
                verbose=cfg.verbose,
            )
            logger.info("Model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Model load error: {e}", exc_info=True)
            return None

    async def _warmup_model(self) -> None:
        """Quick warmup using chat completion API"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.llm.create_chat_completion(
                    messages=[{"role": "user", "content": "Hi"}], max_tokens=5, temperature=0.3
                ),
            )
            logger.info("Model warmed up")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

    async def process_dictation_streaming(
        self, raw_text: str, agentic_prompt: str, token_callback: Optional[Callable[[str], None]] = None
    ) -> Optional[str]:
        """Process text using chat completion API with streaming"""
        if not self._model_loaded or not self.llm:
            logger.error("Model not loaded")
            return raw_text.strip()

        try:
            messages = self._build_messages(agentic_prompt, raw_text)
            result = await self._generate_streaming(messages, token_callback)

            final_result = result if result else raw_text.strip()
            await self._publish_completed(final_result, agentic_prompt)
            return final_result

        except Exception as e:
            logger.error(f"Processing error: {e}", exc_info=True)
            await self._publish_failed(str(e), raw_text)
            return raw_text.strip()

    async def process_dictation(self, raw_text: str, agentic_prompt: str) -> Optional[str]:
        """Non-streaming processing"""
        return await self.process_dictation_streaming(raw_text, agentic_prompt, None)

    def _build_messages(self, agentic_prompt: str, raw_text: str) -> List[Dict[str, str]]:
        """Build messages for chat completion API - optimized for prompt caching"""
        return [
            {
                "role": "system",
                "content": f"{agentic_prompt}. Do not include any meta descriptions or explanations. Output ONLY the processed text.",
            },
            {"role": "user", "content": raw_text},
        ]

    async def _generate_streaming(
        self, messages: List[Dict[str, str]], token_callback: Optional[Callable[[str], None]] = None
    ) -> Optional[str]:
        """Generate using chat completion API with async streaming"""
        try:
            cfg = self.config.llm
            loop = asyncio.get_event_loop()
            token_queue = asyncio.Queue(maxsize=50)
            full_text = []

            def sync_generate():
                """Run llama.cpp generation in thread and feed tokens to async queue"""
                try:
                    stream = self.llm.create_chat_completion(
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
                        stream=True,
                    )

                    for chunk in stream:
                        if chunk and chunk.get("choices"):
                            delta = chunk["choices"][0].get("delta", {})
                            token = delta.get("content", "")
                            if token:
                                try:
                                    asyncio.run_coroutine_threadsafe(token_queue.put(token), loop)
                                except RuntimeError:
                                    logger.warning("Event loop closed during token streaming")
                                    break

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
                while True:
                    token = await asyncio.wait_for(token_queue.get(), timeout=cfg.generation_timeout_sec)
                    if token is None:
                        break

                    full_text.append(token)
                    if token_callback:
                        try:
                            token_callback(token)
                        except Exception as e:
                            logger.debug(f"Token callback error: {e}")

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
        """Publish completion event"""
        event = LLMProcessingCompletedEvent(processed_text=processed_text, agentic_prompt=agentic_prompt)
        await self.event_bus.publish(event)

    async def _publish_failed(self, error_message: str, original_text: str) -> None:
        """Publish failure event"""
        event = LLMProcessingFailedEvent(error_message=error_message, original_text=original_text)
        await self.event_bus.publish(event)

    def is_ready(self) -> bool:
        """Check if ready"""
        return self._model_loaded and self.llm is not None

    async def shutdown(self) -> None:
        """Shutdown and cleanup with proper resource management"""
        try:
            logger.info("LLM service shutting down - cleaning up model and GPU memory")
            self._model_loaded = False

            if self.llm:
                try:
                    # Properly close the llama model to release resources
                    if hasattr(self.llm, "close"):
                        self.llm.close()
                        logger.info("LLM model closed successfully")
                except Exception as e:
                    logger.warning(f"Error closing LLM model: {e}")
                finally:
                    # Delete reference to allow garbage collection
                    del self.llm
                    self.llm = None

            # Force garbage collection to free memory immediately
            # Multiple rounds to catch cyclic references
            for i in range(2):
                gc.collect()
                logger.debug(f"Garbage collection round {i+1} performed")

            logger.info("LLM service shutdown complete")
        except Exception as e:
            logger.error(f"Shutdown error: {e}", exc_info=True)
