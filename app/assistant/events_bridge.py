"""Safe event bridge shared by LiveKit Agent runtimes."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Awaitable, Callable

from app.assistant.contracts import AuthDecision
from app.assistant.events import event_payload


def auth_status_payload(decision: AuthDecision) -> str:
    """Serialize only the browser-safe part of an auth decision."""

    return json.dumps(
        {
            "state": decision.state.value,
            "display_name": decision.display_name if decision.may_use_private_tools else None,
            "expires_at": decision.expires_at.isoformat() if decision.expires_at else None,
        },
        separators=(",", ":"),
    )


class SessionEventSink:
    """Send ordered, UI-safe JSON events to one LiveKit participant."""

    def __init__(self, send: Callable[[str, str | None], Awaitable[None]], participant_id: str) -> None:
        self._send = send
        self._participant_id = participant_id
        self._sequences: defaultdict[str, int] = defaultdict(int)

    async def send_auth(self, decision: AuthDecision) -> None:
        await self._send(auth_status_payload(decision), self._participant_id)

    async def send_event(self, event_type: str, *, turn_id: str | None = None, **values) -> None:
        if turn_id is not None:
            self._sequences[turn_id] += 1
            values["sequence"] = self._sequences[turn_id]
        await self._send(event_payload(event_type=event_type, turn_id=turn_id, **values), self._participant_id)
