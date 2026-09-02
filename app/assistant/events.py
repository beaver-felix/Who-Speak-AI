"""Trusted, browser-safe event contract for the voice conversation UI."""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Any
from uuid import uuid4


class VoiceAgentState(StrEnum):
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    SPEAKING = "speaking"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    ERROR = "error"


class VoiceAgentSessionStatus(StrEnum):
    """Lifecycle of the local voice worker, separate from a conversation turn."""

    IDLE = "idle"
    CONNECTING = "connecting"
    STARTING = "starting"
    READY = "ready"
    RECONNECTING = "reconnecting"
    STOPPING = "stopping"
    FAILED = "failed"


def event_id() -> str:
    return str(uuid4())


def event_payload(*, event_type: str, turn_id: str | None = None, **values: Any) -> str:
    """Serialize only UI-safe conversation metadata.

    This boundary deliberately has no score, embedding, raw audio, HE context,
    matcher token, or identity template fields.
    """
    payload: dict[str, Any] = {"type": event_type, "message_id": event_id()}
    if turn_id is not None:
        payload["turn_id"] = turn_id
    payload.update(values)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
