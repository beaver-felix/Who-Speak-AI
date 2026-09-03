"""Local Whisper ASR adapter; the model is loaded only in the Agent process."""

from __future__ import annotations

import numpy as np

from voiceauth.audio import AudioRecording


class LocalWhisperProvider:
    def __init__(self, *, model_name: str = "base", device: str = "cpu", model=None) -> None:
        self._model_name = model_name
        self._device = device
        self._model = model

    def _model_or_create(self):
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except ImportError as error:
            raise RuntimeError("Install the [providers] extra to enable local Whisper ASR.") from error
        self._model = WhisperModel(self._model_name, device=self._device)
        return self._model

    def warmup(self) -> None:
        """Load the local model before the participant's first spoken turn."""
        self._model_or_create()

    def transcribe(self, recording: AudioRecording, *, language: str = "vi") -> str:
        segments, _info = self._model_or_create().transcribe(
            np.ascontiguousarray(recording.waveform, dtype=np.float32), language=language
        )
        return " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
