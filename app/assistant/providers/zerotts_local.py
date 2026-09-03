"""Local ZeroTTS streaming provider for Vietnamese speech output."""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from app.assistant.providers.streaming_tts import TTSChunk, float32_to_s16_mono

logger = logging.getLogger(__name__)


class ZeroTTSLocalProvider:
    """Bridge ZeroTTS's synchronous generator to an async bounded stream.

    ZeroTTS itself is CPU/ONNX oriented. The generator runs in one worker
    thread so model inference never blocks Pipecat's event loop. The queue is
    bounded to keep a slow LiveKit consumer from accumulating audio in RAM.
    """

    name = "zerotts"
    sample_rate = 48_000

    def __init__(
        self,
        *,
        model_name: str = "zeroweight-ai/ZeroTTS",
        voice: str = "maichi",
        cache_dir: str | None = None,
        queue_max_chunks: int = 8,
        startup_buffer_ms: float = 250.0,
        intra_op_num_threads: int = 4,
        codec_intra_op_num_threads: int | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("ZeroTTS model name must not be empty.")
        if not voice.strip():
            raise ValueError("ZeroTTS voice must not be empty.")
        if queue_max_chunks <= 0:
            raise ValueError("ZeroTTS queue_max_chunks must be positive.")
        if startup_buffer_ms < 0:
            raise ValueError("ZeroTTS startup_buffer_ms must not be negative.")
        if intra_op_num_threads <= 0:
            raise ValueError("ZeroTTS intra_op_num_threads must be positive.")
        if codec_intra_op_num_threads is not None and codec_intra_op_num_threads <= 0:
            raise ValueError("ZeroTTS codec_intra_op_num_threads must be positive.")
        self._model_name = model_name.strip()
        self._voice = voice.strip()
        self._cache_dir = Path(cache_dir).expanduser() if cache_dir else None
        self._queue_max_chunks = queue_max_chunks
        self._startup_buffer_ms = startup_buffer_ms
        self._intra_op_num_threads = intra_op_num_threads
        self._codec_intra_op_num_threads = codec_intra_op_num_threads
        self._model: Any | None = None
        self._model_lock = threading.Lock()
        self._active_stops: set[threading.Event] = set()
        self._active_stops_lock = threading.Lock()

    async def warmup(self) -> None:
        await asyncio.to_thread(self._warmup_sync)

    async def stream(self, text: str) -> AsyncIterator[TTSChunk]:
        if not text.strip():
            raise ValueError("Cannot synthesize empty text.")
        model = await asyncio.to_thread(self._model_or_create)
        output_queue: queue.Queue[object] = queue.Queue(maxsize=self._queue_max_chunks)
        stop_event = threading.Event()
        sentinel = object()
        error_holder: list[BaseException] = []
        with self._active_stops_lock:
            self._active_stops.add(stop_event)

        def produce() -> None:
            try:
                with self._model_lock:
                    generated = model.synthesize_stream(text, voice=self._voice)
                    for raw_chunk in generated:
                        if stop_event.is_set():
                            break
                        pcm = float32_to_s16_mono(raw_chunk)
                        if pcm:
                            item = TTSChunk(
                                audio=pcm,
                                sample_rate=self.sample_rate,
                                num_channels=1,
                            )
                            while not stop_event.is_set():
                                try:
                                    output_queue.put(item, timeout=0.1)
                                    break
                                except queue.Full:
                                    continue
            except BaseException as error:  # surfaced by the async consumer
                error_holder.append(error)
            finally:
                while not stop_event.is_set():
                    try:
                        output_queue.put(sentinel, timeout=0.1)
                        break
                    except queue.Full:
                        continue

        worker = asyncio.get_running_loop().run_in_executor(None, produce)
        startup_buffer_bytes = int(
            self.sample_rate * 2 * self._startup_buffer_ms / 1000
        )
        pending: list[TTSChunk] = []
        pending_bytes = 0
        started = startup_buffer_bytes <= 0
        try:
            while True:
                item = output_queue.get_nowait() if not output_queue.empty() else None
                if item is None:
                    if worker.done() and output_queue.empty():
                        # The producer can finish between its last audio put
                        # and its sentinel put. Do not lose a short utterance
                        # that is still waiting in the startup buffer.
                        for pending_item in pending:
                            yield pending_item
                        pending.clear()
                        break
                    await asyncio.sleep(0.005)
                    continue
                if item is sentinel:
                    # A short utterance may finish before the target buffer is
                    # full. Do not drop its already-generated audio.
                    for pending_item in pending:
                        yield pending_item
                    pending.clear()
                    break
                chunk = item  # type: ignore[assignment]
                if not started:
                    pending.append(chunk)
                    pending_bytes += len(chunk.audio)
                    if pending_bytes < startup_buffer_bytes:
                        continue
                    started = True
                    for pending_item in pending:
                        yield pending_item
                    pending.clear()
                    pending_bytes = 0
                    continue
                yield chunk
            await worker
            if error_holder:
                raise error_holder[0]
        finally:
            stop_event.set()
            with self._active_stops_lock:
                self._active_stops.discard(stop_event)
            if not worker.done():
                try:
                    await asyncio.wait_for(asyncio.shield(worker), timeout=1.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    logger.warning("zerotts_worker_stop_timeout")

    async def close(self) -> None:
        with self._active_stops_lock:
            stops = tuple(self._active_stops)
            self._active_stops.clear()
        for stop_event in stops:
            stop_event.set()
        self._model = None

    def _warmup_sync(self) -> None:
        model = self._model_or_create()
        voices = model.list_voices()
        if self._voice not in voices:
            available = ", ".join(str(value) for value in voices)
            raise ValueError(
                f"ZeroTTS voice {self._voice!r} is unavailable. Available voices: {available}"
            )
        # Run one short generation to initialize ONNX sessions before a room
        # receives its first conversational turn.
        generated = model.synthesize_stream("Xin chào.", voice=self._voice)
        next(iter(generated), None)

    def _model_or_create(self) -> Any:
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is not None:
                return self._model
            try:
                from zerotts import ZeroTTS
            except ImportError as error:
                raise RuntimeError(
                    "ZeroTTS is unavailable. Install the providers extra with zerotts."
                ) from error
            if self._cache_dir is not None:
                self._cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                "zerotts_loading model=%s voice=%s cache_dir=%s",
                self._model_name,
                self._voice,
                self._cache_dir,
            )
            self._model = ZeroTTS.from_pretrained(
                self._model_name,
                cache_dir=str(self._cache_dir) if self._cache_dir is not None else None,
                providers=["CPUExecutionProvider"],
                intra_op_num_threads=self._intra_op_num_threads,
                codec_intra_op_num_threads=self._codec_intra_op_num_threads,
            )
            return self._model
