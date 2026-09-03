"""Spawned RawNet3 worker so Streamlit reruns never reload native model state."""

from __future__ import annotations

import multiprocessing
from multiprocessing.connection import Connection
from threading import RLock
from typing import Any

import numpy as np

from voiceauth.errors import ModelWorkerError
from voiceauth.rawnet3 import RawNet3Runtime


def _speaker_worker(connection: Connection, runtime: RawNet3Runtime) -> None:
    try:
        from voiceauth.rawnet3 import RawNet3SpeakerEncoder

        engine = RawNet3SpeakerEncoder(runtime)
        while True:
            command, payload = connection.recv()
            if command == "close":
                connection.send((True, None))
                return
            try:
                if command == "prepare":
                    engine.prepare()
                    result: Any = True
                elif command == "embed":
                    waveform, sample_rate = payload
                    result = engine.embed(waveform, sample_rate)
                else:
                    raise ValueError(f"Unknown speaker-worker command: {command}")
            except Exception as error:
                connection.send((False, str(error)))
            else:
                connection.send((True, result))
    except EOFError:
        return
    finally:
        connection.close()


class ProcessSpeakerEngine:
    """Synchronous proxy backed by one persistent spawned RawNet3 process."""

    def __init__(self, runtime: RawNet3Runtime, *, timeout_seconds: float = 180.0) -> None:
        self.runtime = runtime
        self.timeout_seconds = timeout_seconds
        self._context = multiprocessing.get_context("spawn")
        self._connection: Connection | None = None
        self._process: multiprocessing.Process | None = None
        self._lock = RLock()

    def prepare(self) -> None:
        self._request("prepare", None)

    def embed(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        return self._request("embed", (np.ascontiguousarray(waveform, dtype=np.float32), sample_rate))

    def close(self) -> None:
        with self._lock:
            connection, process = self._connection, self._process
            self._connection, self._process = None, None
            if connection is not None:
                try:
                    if process is not None and process.is_alive():
                        connection.send(("close", None))
                        if connection.poll(2.0):
                            connection.recv()
                except (BrokenPipeError, EOFError, OSError):
                    pass
                finally:
                    connection.close()
            if process is not None:
                process.join(2.0)
                if process.is_alive():
                    process.terminate()
                    process.join(2.0)

    def _request(self, command: str, payload: Any) -> Any:
        with self._lock:
            self._ensure_started()
            if self._connection is None or self._process is None:
                raise ModelWorkerError("The speaker-model worker could not be started.")
            try:
                self._connection.send((command, payload))
                if not self._connection.poll(self.timeout_seconds):
                    self.close()
                    raise ModelWorkerError("The speaker-model worker timed out. Restart the app.")
                succeeded, result = self._connection.recv()
            except (BrokenPipeError, EOFError, OSError) as error:
                self.close()
                raise ModelWorkerError("The speaker-model worker stopped unexpectedly.") from error
            if not succeeded:
                raise ModelWorkerError(str(result))
            return result

    def _ensure_started(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        self.close()
        parent, child = self._context.Pipe()
        process = self._context.Process(target=_speaker_worker, args=(child, self.runtime), daemon=True)
        process.start()
        child.close()
        self._connection, self._process = parent, process
