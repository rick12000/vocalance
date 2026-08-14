from __future__ import annotations

import asyncio
import logging
import pathlib
import queue
import threading
import time
from typing import TYPE_CHECKING, Awaitable, Callable, Optional

import numpy as np

from vocalance.app.config.app_config import GlobalAppConfig, XASRConfig
from vocalance.app.lifecycle.worker import schedule_on_loop
from vocalance.app.services.dictation_flow.speech_recognition.transcript_state_manager import TranscriptStateManager

if TYPE_CHECKING:
    import sherpa_onnx

logger = logging.getLogger(__name__)

ASR_ASSETS = pathlib.Path(__file__).parent.parent.parent.parent / "assets" / "asr"
INT16_SCALE = 1.0 / 32768.0
FEATURE_DIM = 80
DECODING_METHOD = "greedy_search"
PROVIDER = "cpu"
WORKER_STOP_TIMEOUT_SEC = 5.0
QUEUE_PUT_TIMEOUT_SEC = 1.0
WORKER_POLL_TIMEOUT_SEC = 0.1


class XASRStreamSession:

    def __init__(
        self,
        recognizer: sherpa_onnx.OnlineRecognizer,
        loop: asyncio.AbstractEventLoop,
        on_committed: Callable[[str], Awaitable[None]],
        on_provisional: Callable[[str], Awaitable[None]],
        xasr_config: XASRConfig,
        provisional_words_override: Optional[int] = None,
        silence_commit_threshold_override: Optional[float] = None,
    ) -> None:
        self.recognizer = recognizer
        self.loop = loop
        self.on_committed = on_committed
        self.on_provisional = on_provisional

        provisional_words = provisional_words_override if provisional_words_override is not None else xasr_config.provisional_words
        self.tsm = TranscriptStateManager(
            stability_window=xasr_config.stability_window,
            provisional_words=provisional_words,
        )

        self.stream = recognizer.create_stream()
        self.audio_queue: queue.Queue[Optional[tuple[bytes, int]]] = queue.Queue(maxsize=xasr_config.audio_queue_maxsize)
        self.closed = False
        self.dropped_chunks = 0
        self.last_provisional = ""

        self.silence_commit_threshold = (
            silence_commit_threshold_override
            if silence_commit_threshold_override is not None
            else xasr_config.silence_commit_threshold_sec
        )
        self.last_hypothesis = ""
        self.hypothesis_stable_since = time.monotonic()
        self.silence_committed = False
        self.finalization_delta: str = ""

        self.session_start = time.monotonic()

        self.worker = threading.Thread(target=self.worker_loop, name="xasr-worker", daemon=True)
        self.worker.start()

    def add_audio_pcm16(self, audio_bytes: bytes, sample_rate: int) -> None:
        if not audio_bytes or self.closed:
            return
        try:
            self.audio_queue.put_nowait((audio_bytes, sample_rate))
            return
        except queue.Full:
            pass
        try:
            self.audio_queue.get_nowait()
        except queue.Empty:
            pass
        self.dropped_chunks += 1
        if self.dropped_chunks % 50 == 1:
            logger.warning("XASR worker backlog: dropped %s oldest audio chunks", self.dropped_chunks)
        try:
            self.audio_queue.put_nowait((audio_bytes, sample_rate))
        except queue.Full:
            pass

    def stop(self) -> str:
        self.closed = True
        try:
            self.audio_queue.put(None, timeout=QUEUE_PUT_TIMEOUT_SEC)
        except queue.Full:
            logger.warning("XASR queue full on stop; some tail audio may be lost")
        if self.worker.is_alive():
            self.worker.join(timeout=WORKER_STOP_TIMEOUT_SEC)
        self.finalize()
        return self.finalization_delta

    def worker_loop(self) -> None:
        while True:
            try:
                first = self.audio_queue.get(timeout=WORKER_POLL_TIMEOUT_SEC)
            except queue.Empty:
                continue
            if first is None:
                return
            batch: list[tuple[bytes, int]] = [first]
            saw_sentinel = False
            while True:
                try:
                    item = self.audio_queue.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    saw_sentinel = True
                    break
                batch.append(item)
            try:
                self.process_batch(batch)
            except Exception:
                logger.exception("XASR worker error during batch processing")
            if saw_sentinel:
                return

    def process_batch(self, batch: list[tuple[bytes, int]]) -> None:
        decode_start = time.monotonic()
        for audio_bytes, sample_rate in batch:
            samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) * INT16_SCALE
            if not samples.flags.c_contiguous:
                samples = np.ascontiguousarray(samples)
            self.stream.accept_waveform(sample_rate, samples)

        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)

        decode_ms = (time.monotonic() - decode_start) * 1000.0
        batch_audio_ms = sum(
            len(np.frombuffer(ab, dtype=np.int16)) / sr * 1000.0 for ab, sr in batch
        )
        if batch_audio_ms > 0:
            rtf = decode_ms / batch_audio_ms
            if rtf > 1.0:
                logger.warning(
                    "XASR inference slower than real-time: RTF=%.2f (decode=%.1f ms, audio=%.1f ms)",
                    rtf, decode_ms, batch_audio_ms,
                )
            else:
                logger.debug(
                    "XASR decode RTF=%.3f (decode=%.1f ms, audio=%.1f ms, queue_depth=%d)",
                    rtf, decode_ms, batch_audio_ms, self.audio_queue.qsize(),
                )

        self.dispatch_hypothesis()

    def dispatch_hypothesis(self) -> None:
        hypothesis = self.recognizer.get_result(self.stream)
        if not isinstance(hypothesis, str):
            hypothesis = str(hypothesis)
        hypothesis = hypothesis.strip()

        committed_delta, provisional_text = self.tsm.update(hypothesis)

        if committed_delta:
            self.fire(self.on_committed, committed_delta)

        if provisional_text != self.last_provisional:
            self.last_provisional = provisional_text
            self.fire(self.on_provisional, provisional_text)

        self.maybe_silence_commit(hypothesis)

    def maybe_silence_commit(self, hypothesis: str) -> None:
        now = time.monotonic()
        if hypothesis != self.last_hypothesis:
            self.last_hypothesis = hypothesis
            self.hypothesis_stable_since = now
            self.silence_committed = False
            return

        if self.silence_committed:
            return

        elapsed = now - self.hypothesis_stable_since
        if elapsed < self.silence_commit_threshold:
            return

        self.silence_committed = True
        silence_delta = self.tsm.finalize(hypothesis)
        # Clear provisional before firing committed so the UI removes gray text
        # before appending white text, preventing momentary visual duplication.
        if self.last_provisional:
            self.last_provisional = ""
            self.fire(self.on_provisional, "")
        if silence_delta:
            self.fire(self.on_committed, silence_delta)
        logger.debug(
            "XASR silence commit after %.1f s: delta=%r",
            elapsed,
            silence_delta[:60] if silence_delta else "",
        )

    def finalize(self) -> None:
        try:
            self.stream.input_finished()
        except Exception:
            logger.warning("XASR stream.input_finished() raised", exc_info=True)

        try:
            while self.recognizer.is_ready(self.stream):
                self.recognizer.decode_stream(self.stream)
        except Exception:
            logger.warning("XASR final decode raised", exc_info=True)

        try:
            final_hypothesis = self.recognizer.get_result(self.stream)
            if not isinstance(final_hypothesis, str):
                final_hypothesis = str(final_hypothesis)
            final_hypothesis = final_hypothesis.strip()
            final_delta = self.tsm.finalize(final_hypothesis)
            logger.debug(
                "XASR finalization: delta=%r, session_elapsed=%.1f s",
                final_delta[:60] if final_delta else "",
                time.monotonic() - self.session_start,
            )
            self.finalization_delta = final_delta
        except Exception:
            logger.exception("XASR finalization error")

    def fire(self, callback: Callable[[str], Awaitable[None]], text: str) -> None:
        coro = callback(text)
        try:
            schedule_on_loop(self.loop, coro)
        except RuntimeError:
            coro.close()


class XASREngine:

    def __init__(self, config: GlobalAppConfig) -> None:
        self.config = config
        self.recognizer: Optional[sherpa_onnx.OnlineRecognizer] = None
        self.load_recognizer()

    def load_recognizer(self) -> None:
        import sherpa_onnx

        xasr = self.config.stt.xasr
        logger.info("Loading X-ASR model from %s (num_threads=%d)", ASR_ASSETS, xasr.num_threads)
        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(ASR_ASSETS / "tokens.txt"),
            encoder=str(ASR_ASSETS / "encoder-480ms.onnx"),
            decoder=str(ASR_ASSETS / "decoder-480ms.onnx"),
            joiner=str(ASR_ASSETS / "joiner-480ms.onnx"),
            num_threads=xasr.num_threads,
            sample_rate=self.config.stt.sample_rate,
            feature_dim=FEATURE_DIM,
            enable_endpoint_detection=False,
            decoding_method=DECODING_METHOD,
            provider=PROVIDER,
        )
        logger.info("X-ASR recognizer loaded")

    def create_session(
        self,
        loop: asyncio.AbstractEventLoop,
        on_committed: Callable[[str], Awaitable[None]],
        on_provisional: Callable[[str], Awaitable[None]],
        provisional_words_override: Optional[int] = None,
        silence_commit_threshold_override: Optional[float] = None,
    ) -> XASRStreamSession:
        if self.recognizer is None:
            raise RuntimeError("XASREngine: recognizer not loaded")
        return XASRStreamSession(
            recognizer=self.recognizer,
            loop=loop,
            on_committed=on_committed,
            on_provisional=on_provisional,
            xasr_config=self.config.stt.xasr,
            provisional_words_override=provisional_words_override,
            silence_commit_threshold_override=silence_commit_threshold_override,
        )

    async def shutdown(self) -> None:
        logger.info("XASREngine shutting down")
        self.recognizer = None
        logger.info("XASREngine shutdown complete")
