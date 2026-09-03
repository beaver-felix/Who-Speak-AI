from __future__ import annotations

import numpy as np

from app.assistant.contracts import AuthState
from app.assistant.turn_controller import ConversationTurnController


def test_auth_pending_discards_any_conversation_audio() -> None:
    controller = ConversationTurnController()

    pending = controller.append_pcm(
        np.full(48_000, 0.1, dtype=np.float32), sample_rate=48_000, auth_state=AuthState.AUTH_PENDING
    )

    assert pending.started is False
    assert pending.recording is None
    assert controller.active_turn_id is None


def test_turn_starts_then_finalizes_after_short_silence() -> None:
    controller = ConversationTurnController(min_speech_seconds=0.2, silence_seconds=0.2)

    started = controller.append_pcm(
        np.full(10_000, 0.1, dtype=np.float32), sample_rate=48_000, auth_state=AuthState.GUEST
    )
    completed = controller.append_pcm(
        np.zeros(10_000, dtype=np.float32), sample_rate=48_000, auth_state=AuthState.GUEST
    )

    assert started.started is True
    assert started.turn_id is not None
    assert completed.turn_id == started.turn_id
    assert completed.recording is not None


def test_begin_authentication_resets_a_partially_captured_turn() -> None:
    controller = ConversationTurnController(min_speech_seconds=0.2)
    controller.append_pcm(np.full(10_000, 0.1, dtype=np.float32), sample_rate=48_000, auth_state=AuthState.GUEST)

    controller.begin_authentication()

    assert controller.active_turn_id is None
    assert controller.append_pcm(np.zeros(48_000, dtype=np.float32), sample_rate=48_000, auth_state=AuthState.GUEST).recording is None


def test_authentication_cancellation_allows_a_new_conversation_turn() -> None:
    controller = ConversationTurnController(min_speech_seconds=0.2, silence_seconds=0.2)

    started = controller.append_pcm(
        np.full(10_000, 0.1, dtype=np.float32), sample_rate=48_000, auth_state=AuthState.GUEST
    )
    controller.begin_processing()
    controller.begin_authentication()

    new_started = controller.append_pcm(
        np.full(10_000, 0.1, dtype=np.float32), sample_rate=48_000, auth_state=AuthState.AUTHENTICATED
    )

    assert started.started is True
    assert new_started.started is True
    assert new_started.turn_id is not None
