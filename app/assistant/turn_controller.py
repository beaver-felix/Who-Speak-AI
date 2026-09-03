"""Framework-neutral routing for one participant's conversational audio."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

import numpy as np

from app.assistant.audio_turn import LocalSpeechTurnBuffer
from app.assistant.contracts import AuthState
from voiceauth.audio import AudioRecording


@dataclass(frozen=True)
class TurnIngress:
    turn_id: str | None = None
    started: bool = False
    recording: AudioRecording | None = None


class ConversationTurnController:
    """Prevent the auth challenge from entering the conversational pipeline."""

    def __init__(
        self,
        *,
        min_speech_seconds: float = 0.3,
        silence_seconds: float = 0.45,
        maximum_seconds: float = 15.0,
    ) -> None:
        self._buffer = LocalSpeechTurnBuffer(
            min_speech_seconds=min_speech_seconds,
            silence_seconds=silence_seconds,
            maximum_seconds=maximum_seconds,
        )
        self._active_turn_id: str | None = None
        self._processing = False

    @property
    def active_turn_id(self) -> str | None:
        return self._active_turn_id

    def begin_authentication(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._buffer.reset()
        self._active_turn_id = None
        self._processing = False

    def begin_processing(self) -> None:
        self._processing = True
        self._active_turn_id = None

    def finish_processing(self) -> None:
        self._processing = False

    def append_pcm(self, samples: np.ndarray, *, sample_rate: int, auth_state: AuthState) -> TurnIngress:
        if auth_state is AuthState.AUTH_PENDING:
            self.reset()
            return TurnIngress()
        if self._processing:
            return TurnIngress()

        was_listening = self._buffer.has_audio
        recording = self._buffer.append_pcm(samples, sample_rate=sample_rate)
        if not was_listening and self._buffer.has_audio:
            self._active_turn_id = str(uuid4())
            return TurnIngress(turn_id=self._active_turn_id, started=True)
        if recording is not None:
            turn_id = self._active_turn_id or str(uuid4())
            self._active_turn_id = None
            return TurnIngress(turn_id=turn_id, recording=recording)
        return TurnIngress(turn_id=self._active_turn_id)
