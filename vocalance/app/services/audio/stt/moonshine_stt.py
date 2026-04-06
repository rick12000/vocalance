"""Moonshine Voice STT for dictation (streaming + batch)."""

from __future__ import annotations

import asyncio
import ctypes
import logging
import time
from typing import Any, Awaitable, Callable, List, Optional

import numpy as np
from moonshine_voice.errors import check_error

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.services.audio.stt.dictation_text_normalize import normalize_dictation_text

logger = logging.getLogger(__name__)


class MoonshineDictationStreamSession:
    """One native Moonshine stream with partial/final callbacks bridged to asyncio."""

    def __init__(
        self,
        transcriber: Any,
        update_interval: float,
        loop: asyncio.AbstractEventLoop,
        on_partial: Callable[[str, str], Awaitable[None]],
        on_final: Callable[[str, str], Awaitable[None]],
        max_line_duration_sec: Optional[float] = None,
    ) -> None:
        """Create a stream on ``transcriber`` and register a listener that forwards transcript events."""
        from moonshine_voice.transcriber import LineCompleted, LineTextChanged, TranscriptEventListener

        self._loop = loop
        self._on_partial = on_partial
        self._on_final = on_final
        self._stream = transcriber.create_stream(update_interval=update_interval)

        class _Listener(TranscriptEventListener):
            def __init__(self, outer: MoonshineDictationStreamSession) -> None:
                self._outer = outer

            def on_line_text_changed(self, event: LineTextChanged) -> None:
                self._outer._dispatch_partial(event)

            def on_line_completed(self, event: LineCompleted) -> None:
                self._outer._dispatch_final(event)

        self._stream.add_listener(_Listener(self))
        self._stream.start()
        self._max_line_duration_sec = max_line_duration_sec

    def _dispatch_partial(self, event: Any) -> None:
        """Normalize line text and schedule ``on_partial`` on the asyncio loop."""
        text = (event.line.text or "").strip()
        if not text:
            return
        text = normalize_dictation_text(text)
        if not text:
            return
        segment_id = str(event.line.line_id)
        coro = self._on_partial(text, segment_id)
        try:
            self._loop.call_soon_threadsafe(lambda: asyncio.create_task(coro))
        except RuntimeError:
            coro.close()

    def _dispatch_final(self, event: Any) -> None:
        """Normalize completed line text and schedule ``on_final`` on the asyncio loop."""
        text = (event.line.text or "").strip()
        if not text:
            return
        text = normalize_dictation_text(text)
        if not text:
            return
        segment_id = str(event.line.line_id)
        coro = self._on_final(text, segment_id)
        try:
            self._loop.call_soon_threadsafe(lambda: asyncio.create_task(coro))
        except RuntimeError:
            coro.close()

    def add_audio_pcm16(self, audio_bytes: bytes, sample_rate: int) -> bool:
        """Feed int16 mono PCM. Returns True when ``max_line_duration_sec`` was exceeded (rotate stream)."""
        if not audio_bytes:
            return False
        samples = np.frombuffer(audio_bytes, dtype=np.int16)
        n = int(samples.shape[0])
        if n == 0:
            return False
        arr = (samples.astype(np.float32) * (1.0 / 32768.0)).reshape(-1)
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)

        stream = self._stream
        audio_array = (ctypes.c_float * n).from_buffer_copy(arr)
        error = stream._lib.moonshine_transcribe_add_audio_to_stream(
            stream._transcriber._handle,
            stream._handle,
            audio_array,
            n,
            sample_rate,
            0,
        )
        check_error(error)
        stream._stream_time += n / float(sample_rate)
        if stream._stream_time - stream._last_update_time >= stream._update_interval:
            stream.update_transcription(0)
            stream._last_update_time = stream._stream_time

        max_d = self._max_line_duration_sec
        if max_d is not None and max_d > 0 and stream._stream_time >= max_d:
            return True
        return False

    def stop(self) -> None:
        """Stop the stream, flush, and release native resources."""
        try:
            self._stream.stop()
        except Exception as e:
            logger.warning("Moonshine stream stop: %s", e, exc_info=True)
        try:
            self._stream.close()
        except Exception as e:
            logger.warning("Moonshine stream close: %s", e, exc_info=True)


class MoonshineSTT:
    """Load Moonshine models; expose batch ``recognize`` and ``open_dictation_stream``."""

    def __init__(self, sample_rate: int, config: GlobalAppConfig) -> None:
        """Load the configured model (with retries)."""
        self._sample_rate = sample_rate
        self._config = config
        self._transcriber: Optional[Any] = None
        self._model_lock = asyncio.Lock()
        self._load_model_with_retry()

    def _resolve_model_arch(self) -> Any:
        """Return a ``ModelArch`` enum value from config, or None for default."""
        from moonshine_voice.moonshine_api import string_to_model_arch

        raw = self._config.stt.moonshine_model_arch
        if raw is None or raw == "":
            return None
        return string_to_model_arch(raw)

    def _load_model_with_retry(self) -> None:
        """Download/load Moonshine; set ``self._transcriber`` or raise after max retries."""
        from moonshine_voice.download import get_model_for_language

        max_retries = self._config.stt.moonshine_max_retries
        delay = self._config.stt.moonshine_retry_delay_seconds
        lang = self._config.stt.moonshine_language
        arch = self._resolve_model_arch()

        last_err: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                logger.debug("Loading Moonshine model (attempt %s/%s)", attempt, max_retries)
                model_path, model_arch = get_model_for_language(lang, arch)
                from moonshine_voice.transcriber import Transcriber

                ms = self._config.stt.moonshine_streaming
                self._transcriber = Transcriber(
                    model_path=model_path,
                    model_arch=model_arch,
                    update_interval=ms.stream_update_interval,
                    options=ms.transcriber_load_options(),
                )
                logger.info("Moonshine transcriber loaded: %s arch=%s", model_path, model_arch)
                return
            except Exception as e:
                last_err = e
                logger.error("Moonshine load failed (attempt %s/%s): %s", attempt, max_retries, e, exc_info=True)
                if attempt < max_retries:
                    time.sleep(delay)
        raise RuntimeError(f"Failed to load Moonshine after {max_retries} attempts") from last_err

    def _prepare_float_list(self, audio_bytes: bytes) -> List[float]:
        """Convert int16 PCM bytes to float samples in [-1, 1] for batch transcribe."""
        samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return samples.tolist()

    def recognize_sync(self, audio_bytes: bytes, sample_rate: Optional[int] = None) -> str:
        """Synchronous batch transcription; returns normalized joined text (empty if too short or error)."""
        if sample_rate and sample_rate != self._sample_rate:
            logger.warning("Sample rate mismatch. Expected %s, got %s", self._sample_rate, sample_rate)

        if not audio_bytes or not self._transcriber:
            return ""

        duration_sec = len(audio_bytes) / (self._sample_rate * 2)
        if duration_sec < 0.2:
            return ""

        audio_list = self._prepare_float_list(audio_bytes)
        try:
            transcript = self._transcriber.transcribe_without_streaming(audio_list, self._sample_rate)
        except Exception as e:
            logger.error("Moonshine batch transcribe failed: %s", e, exc_info=True)
            return ""

        parts = [normalize_dictation_text(line.text) for line in transcript.lines if line.text]
        parts = [p for p in parts if p]
        combined = " ".join(parts).strip()
        if combined:
            logger.info("Moonshine recognized: '%s'", combined[:120])
        return combined

    async def recognize(self, audio_bytes: bytes, sample_rate: Optional[int] = None) -> str:
        """Run ``recognize_sync`` under ``_model_lock`` on a worker thread."""
        async with self._model_lock:
            return await asyncio.to_thread(self.recognize_sync, audio_bytes, sample_rate)

    def open_dictation_stream(
        self,
        loop: asyncio.AbstractEventLoop,
        on_partial: Callable[[str, str], Awaitable[None]],
        on_final: Callable[[str, str], Awaitable[None]],
    ) -> MoonshineDictationStreamSession:
        """Open a streaming session using config update interval and optional line rotation duration."""
        if not self._transcriber:
            raise RuntimeError("Moonshine transcriber not loaded")
        max_line = self._config.stt.moonshine_max_stream_line_duration_seconds
        max_line_f = float(max_line) if max_line is not None else 0.0
        use_max = max_line_f if max_line_f > 0 else None
        return MoonshineDictationStreamSession(
            self._transcriber,
            self._config.stt.moonshine_streaming.stream_update_interval,
            loop,
            on_partial,
            on_final,
            max_line_duration_sec=use_max,
        )

    async def shutdown(self) -> None:
        """Close the transcriber with a short timeout."""
        logger.info("Shutting down MoonshineSTT")
        try:
            async with asyncio.timeout(5.0):
                async with self._model_lock:
                    if self._transcriber is not None:
                        self._transcriber.close()
                        self._transcriber = None
        except asyncio.TimeoutError:
            logger.warning("MoonshineSTT shutdown timed out")
            self._transcriber = None
        logger.info("MoonshineSTT shutdown complete")
