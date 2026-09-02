"""Safe event bridge shared by LiveKit Agent runtimes."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Awaitable, Callable

from app.assistant.contracts import AuthChallengePhase, AuthChallengeStatus, AuthDecision
from app.assistant.events import event_payload


def auth_status_payload(
    decision: AuthDecision,
    *,
    challenge: AuthChallengeStatus | None = None,
    session_id: str | None = None,
    sequence: int | None = None,
) -> str:
    """Serialize only the browser-safe part of an auth decision."""

    status = challenge or AuthChallengeStatus.idle(target_ms=5_000)
    if status.phase is AuthChallengePhase.CAPTURING:
        message = "Đang thu giọng nói để xác thực"
    elif status.phase is AuthChallengePhase.PROCESSING:
        message = "Đã thu đủ 5 giây · Đang kiểm tra giọng nói…"
    elif status.phase is AuthChallengePhase.WAITING_FOR_RESUME:
        message = (
            "Xác thực thành công · Bấm ‘Bắt đầu nói với Agent’"
            if decision.may_use_private_tools
            else "Chưa xác thực được · Thử lại hoặc tiếp tục ở Guest"
        )
    elif status.phase is AuthChallengePhase.CONVERSATION_READY:
        message = "Sẵn sàng lắng nghe"
    else:
        message = None

    payload = {
        "state": decision.state.value,
        "display_name": decision.display_name if decision.may_use_private_tools else None,
        "expires_at": decision.expires_at.isoformat() if decision.expires_at else None,
        "phase": status.phase.value,
        "elapsed_ms": status.elapsed_ms,
        "target_ms": status.target_ms,
        "can_resume": status.can_resume,
        "message": message,
    }
    if session_id is not None:
        payload["session_id"] = session_id
    if sequence is not None:
        payload["sequence"] = sequence
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


class SessionEventSink:
    """Send ordered, UI-safe JSON events to one LiveKit participant."""

    def __init__(
        self,
        send: Callable[[str, str | None], Awaitable[None]],
        participant_id: str,
        *,
        session_id: str | None = None,
    ) -> None:
        self._send = send
        self._participant_id = participant_id
        self._session_id = session_id
        self._sequences: defaultdict[str, int] = defaultdict(int)
        self._auth_sequence = 0

    async def send_auth(
        self,
        decision: AuthDecision,
        challenge: AuthChallengeStatus | None = None,
    ) -> None:
        self._auth_sequence += 1
        await self._send(
            auth_status_payload(
                decision,
                challenge=challenge,
                session_id=self._session_id,
                sequence=self._auth_sequence,
            ),
            self._participant_id,
        )

    async def send_event(self, event_type: str, *, turn_id: str | None = None, **values) -> None:
        if turn_id is not None:
            self._sequences[turn_id] += 1
            values["sequence"] = self._sequences[turn_id]
        await self._send(event_payload(event_type=event_type, turn_id=turn_id, **values), self._participant_id)
