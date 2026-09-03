from __future__ import annotations

import asyncio

import numpy as np
import pytest

from app.assistant.audio_buffer import AuthChallengeBuffer
from app.assistant.config import OpenAISettings
from app.assistant.contracts import AuthDecision, AuthState
from app.assistant.providers.openai_llm import OpenAIResponsesProvider
from voiceauth.errors import AudioValidationError


def test_challenge_buffer_is_bounded_and_requires_four_seconds() -> None:
    buffer = AuthChallengeBuffer()
    assert buffer.append_pcm(np.ones(48_000 * 7, dtype=np.float32), sample_rate=48_000) is True
    assert buffer.duration_seconds == 6.0
    recording = buffer.finish()
    assert recording.sample_rate == 16_000
    assert recording.duration_seconds == 6.0


def test_challenge_buffer_rejects_short_audio() -> None:
    buffer = AuthChallengeBuffer()
    buffer.append_pcm(np.ones(3_000, dtype=np.float32), sample_rate=1_000)
    with pytest.raises(AudioValidationError, match="at least"):
        buffer.finish()


def test_openai_payload_contains_transcript_and_capability_not_sensitive_voice_data() -> None:
    class Responses:
        def __init__(self) -> None:
            self.kwargs = None

        async def create(self, **kwargs):
            self.kwargs = kwargs
            return type("Response", (), {"output_text": "xin chào"})()

    responses = Responses()
    fake_client = type("Client", (), {"responses": responses})()
    provider = OpenAIResponsesProvider(OpenAISettings("not-a-real-key", "test-model"), client=fake_client)

    answer = asyncio.run(
        provider.respond(
            "thời tiết hôm nay",
            allowed_tools={"weather"},
            auth_context=AuthDecision(AuthState.GUEST),
        )
    )

    assert answer == "xin chào"
    assert responses.kwargs["input"] == "thời tiết hôm nay"
    assert responses.kwargs["store"] is False
    serialized = repr(responses.kwargs)
    assert "embedding" not in serialized
    assert "audio" not in serialized
    assert "Trusted VoiceAuth state for this request: guest" in responses.kwargs["instructions"]
    assert "Private Calendar access: not granted" in responses.kwargs["instructions"]
