from __future__ import annotations

import asyncio
import ctypes
import logging
import queue
import threading
import time
from typing import TYPE_CHECKING, Awaitable, Callable, List, Optional

import numpy as np
from moonshine_voice.errors import check_error

from vocalance.app.config.app_config import GlobalAppConfig, MoonshineStreamingConfig
from vocalance.app.lifecycle.worker import run_blocking, schedule_on_loop
from vocalance.app.services.dictation_flow.speech_recognition.dictation_text_normalize import normalize_dictation_text

if TYPE_CHECKING:
    from moonshine_voice.moonshine_api import ModelArch
    from moonshine_voice.transcriber import Transcriber

logger = logging.getLogger(__name__)


_AUDIO_QUEUE_MAXSIZE = 1024
_INT16_SCALE = 1.0 / 32768.0
_MOONSHINE_FLAG_FORCE_UPDATE = 1


class MoonshineStreamSession:
    """One native Moonshine stream with audio ingestion off-loaded to a worker thread.

    Producers call ``add_audio_pcm16`` from any thread; the call only enqueues bytes
    and returns immediately, so the asyncio loop is never blocked by the streaming
    decoder. A single worker thread drains the queue in batches and runs the native
    ``add_audio_to_stream`` and ``update_transcription`` calls. Segment boundaries
    (mid-pause line finalization, hard duration cap) are owned entirely by the
    Moonshine native VAD; no Python-side rotation is involved.
    """

    def __init__(
        self,
        transcriber: Transcriber,
        loop: asyncio.AbstractEventLoop,
        on_partial: Callable[[str, str], Awaitable[None]],
        on_final: Callable[[str, str], Awaitable[None]],
        ms_config: MoonshineStreamingConfig,
    ) -> None:
        from moonshine_voice.transcriber import LineCompleted, LineTextChanged, TranscriptEventListener

        self._loop = loop
        self._on_partial = on_partial
        self._on_final = on_final
        self._stream = transcriber.create_stream(update_interval=ms_config.stream_update_interval)

        class _Listener(TranscriptEventListener):
            def __init__(self, outer: MoonshineStreamSession) -> None:
                self._outer = outer

            def on_line_text_changed(self, line_update: LineTextChanged) -> None:
                self._outer._dispatch(line_update.line, self._outer._on_partial)

            def on_line_completed(self, line_completed: LineCompleted) -> None:
                self._outer._dispatch(line_completed.line, self._outer._on_final)

        self._stream.add_listener(_Listener(self))
        self._stream.start()

        self._queue: queue.Queue[Optional[tuple[bytes, int]]] = queue.Queue(maxsize=_AUDIO_QUEUE_MAXSIZE)
        self._closed = False
        self._dropped_chunks = 0
        self._worker = threading.Thread(target=self._worker_loop, name="moonshine-feeder", daemon=True)
        self._worker.start()

    def _dispatch(self, line, callback: Callable[[str, str], Awaitable[None]]) -> None:
        text = (line.text or "").strip()
        if not text:
            return
        text = normalize_dictation_text(text)
        if not text:
            return
        coro = callback(text, str(line.line_id))
        try:
            schedule_on_loop(self._loop, coro)
        except RuntimeError:
            coro.close()

    def add_audio_pcm16(self, audio_bytes: bytes, sample_rate: int) -> None:
        """Non-blocking: enqueue raw PCM for the worker. Drops oldest on overflow."""
        if not audio_bytes or self._closed:
            return
        try:
            self._queue.put_nowait((audio_bytes, sample_rate))
            return
        except queue.Full:
            pass
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        self._dropped_chunks += 1
        if self._dropped_chunks % 50 == 1:
            logger.warning("Moonshine feeder backlog: dropped %s oldest audio chunks", self._dropped_chunks)
        try:
            self._queue.put_nowait((audio_bytes, sample_rate))
        except queue.Full:
            pass

    def _worker_loop(self) -> None:
        while True:
            try:
                first = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if first is None:
                return
            batch: list[tuple[bytes, int]] = [first]
            saw_sentinel = False
            while True:
                try:
                    more = self._queue.get_nowait()
                except queue.Empty:
                    break
                if more is None:
                    saw_sentinel = True
                    break
                batch.append(more)
            try:
                self._feed_batch(batch)
            except Exception as e:
                logger.error("Moonshine feeder error: %s", e, exc_info=True)
            if saw_sentinel:
                return

    def _feed_batch(self, batch: list[tuple[bytes, int]]) -> None:
        stream = self._stream
        handle_t = stream._transcriber._handle
        handle_s = stream._handle
        for audio_bytes, sample_rate in batch:
            samples = np.frombuffer(audio_bytes, dtype=np.int16)
            n = int(samples.shape[0])
            if n == 0:
                continue
            arr = (samples.astype(np.float32) * _INT16_SCALE).reshape(-1)
            if not arr.flags.c_contiguous:
                arr = np.ascontiguousarray(arr)
            audio_array = (ctypes.c_float * n).from_buffer_copy(arr)
            error = stream._lib.moonshine_transcribe_add_audio_to_stream(
                handle_t,
                handle_s,
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

    def stop(self) -> None:
        """Drain queued audio, force-flush the native buffer, then close the stream.

        The Moonshine Python ``Stream.stop()`` deactivates VAD and then calls
        ``update_transcription(0)`` again internally. This second call re-emits
        ``LineCompleted`` for every already-finalized segment, producing duplicate
        events. We therefore deactivate VAD directly via the C API, issue our own
        ``update_transcription(FORCE_UPDATE)`` to capture the tail of the utterance,
        and then call ``stream.close()`` to free resources — skipping the Python
        ``stop()`` wrapper entirely.
        """
        self._closed = True
        try:
            self._queue.put(None, timeout=1.0)
        except queue.Full:
            logger.warning("Moonshine feeder queue full on stop; some audio may be lost")
        if self._worker.is_alive():
            self._worker.join(timeout=3.0)
        try:
            self._stream._lib.moonshine_stop_stream(self._stream._transcriber._handle, self._stream._handle)
        except Exception as e:
            logger.warning("Moonshine native stop: %s", e, exc_info=True)
        try:
            self._stream.update_transcription(_MOONSHINE_FLAG_FORCE_UPDATE)
        except Exception as e:
            logger.warning("Moonshine final flush: %s", e, exc_info=True)
        try:
            self._stream.close()
        except Exception as e:
            logger.warning("Moonshine stream close: %s", e, exc_info=True)


class MoonshineEngine:
    """Load Moonshine models; expose batch ``recognize`` and ``open_stream``."""

    def __init__(self, sample_rate: int, config: GlobalAppConfig) -> None:
        self._sample_rate = sample_rate
        self._config = config
        self._transcriber: Optional[Transcriber] = None
        self._model_lock = asyncio.Lock()
        self._load_model_with_retry()

    def _resolve_model_arch(self) -> Optional[ModelArch]:
        from moonshine_voice.moonshine_api import string_to_model_arch

        raw = self._config.stt.moonshine_model_arch
        if raw is None or raw == "":
            return None
        return string_to_model_arch(raw)

    def _load_model_with_retry(self) -> None:
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
        async with self._model_lock:
            return await run_blocking(self.recognize_sync, audio_bytes, sample_rate, name="moonshine-recognize")

    def open_stream(
        self,
        loop: asyncio.AbstractEventLoop,
        on_partial: Callable[[str, str], Awaitable[None]],
        on_final: Callable[[str, str], Awaitable[None]],
    ) -> MoonshineStreamSession:
        """Open a streaming session that feeds audio via a dedicated worker thread."""
        if not self._transcriber:
            raise RuntimeError("Moonshine transcriber not loaded")
        return MoonshineStreamSession(
            self._transcriber,
            loop,
            on_partial,
            on_final,
            ms_config=self._config.stt.moonshine_streaming,
        )

    async def shutdown(self) -> None:
        logger.info("Shutting down MoonshineEngine")
        try:
            async with asyncio.timeout(5.0):
                async with self._model_lock:
                    if self._transcriber is not None:
                        self._transcriber.close()
                        self._transcriber = None
        except asyncio.TimeoutError:
            logger.warning("MoonshineEngine shutdown timed out")
            self._transcriber = None
        logger.info("MoonshineEngine shutdown complete")
