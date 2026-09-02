from __future__ import annotations

import json

from app.assistant.events import VoiceAgentState, event_payload


def test_voice_agent_event_contains_only_browser_safe_fields() -> None:
    payload = json.loads(
        event_payload(
            event_type="state",
            turn_id="turn-1",
            state=VoiceAgentState.THINKING,
        )
    )

    assert payload["type"] == "state"
    assert payload["turn_id"] == "turn-1"
    assert payload["state"] == "thinking"
    assert payload["message_id"]
    serialized = repr(payload).casefold()
    for forbidden in ("embedding", "ciphertext", "he_context", "matcher_token", "score", "audio"):
        assert forbidden not in serialized


def test_voice_agent_event_can_carry_ordering_metadata_without_private_fields() -> None:
    payload = json.loads(
        event_payload(
            event_type="assistant_response",
            turn_id="turn-1",
            sequence=4,
            text="Xin chào",
            is_final=False,
        )
    )

    assert payload["sequence"] == 4
    assert payload["is_final"] is False
