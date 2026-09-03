"""Typed calendar provider boundary for mock and gateway-backed calendars."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

import httpx

from app.assistant_gateway.security import sign_internal_request
from app.assistant_gateway.store import CalendarEvent, GatewayStore


@dataclass(frozen=True)
class CalendarResult:
    provider: str
    demo: bool
    user_id: str
    events: tuple[CalendarEvent, ...]


class CalendarProvider(Protocol):
    async def list_events(self, *, user_id: str, start: datetime, end: datetime) -> CalendarResult: ...

    async def create_event(self, *, user_id: str, title: str, start: datetime, end: datetime) -> CalendarResult: ...


class CalendarProviderError(RuntimeError):
    """A calendar integration failed without changing voice authorization."""


class MockCalendarProvider:
    """Deterministic local data. It never contacts Google or an MCP server."""

    def __init__(self, store: GatewayStore) -> None:
        self._store = store

    async def list_events(self, *, user_id: str, start: datetime, end: datetime) -> CalendarResult:
        events = self._store.list_mock_events(user_id, start=start, end=end)
        if not events:
            demo_start = datetime(2026, 8, 24, 14, 0, tzinfo=UTC)
            demo_end = demo_start + timedelta(hours=1)
            events = [CalendarEvent("demo-event-1", user_id, "Demo meeting", demo_start, demo_end)]
        return CalendarResult("mock", True, user_id, tuple(events))

    async def create_event(self, *, user_id: str, title: str, start: datetime, end: datetime) -> CalendarResult:
        if not title.strip() or end <= start:
            raise ValueError("A calendar event needs a title and a valid time range.")
        event = self._store.create_mock_event(user_id=user_id, title=title, start=start, end=end)
        return CalendarResult("mock", True, user_id, (event,))


class GatewayCalendarProvider:
    """Agent-side facade; user identity is resolved by the trusted gateway."""

    def __init__(
        self,
        *,
        gateway_url: str,
        session_id: str,
        room_name: str,
        internal_secret: str,
        account_id: str,
        timeout_seconds: float = 8.0,
        client=None,
    ) -> None:
        self._gateway_url = gateway_url.rstrip("/")
        self._session_id = session_id
        self._room_name = room_name
        self._internal_secret = internal_secret
        self._account_id = account_id
        self._timeout_seconds = timeout_seconds
        self._client = client

    async def list_events(self, *, user_id: str, start: datetime, end: datetime) -> CalendarResult:
        self._check_account(user_id)
        payload = {
            "session_id": self._session_id,
            "room_name": self._room_name,
            "start": start.isoformat(),
            "end": end.isoformat(),
        }
        response = await self._post("/internal/calendar/list-events", payload)
        return self._result(response)

    async def create_event(self, *, user_id: str, title: str, start: datetime, end: datetime) -> CalendarResult:
        self._check_account(user_id)
        payload = {
            "session_id": self._session_id,
            "room_name": self._room_name,
            "title": title,
            "start": start.isoformat(),
            "end": end.isoformat(),
        }
        response = await self._post("/internal/calendar/create-event", payload)
        return self._result(response)

    def _check_account(self, user_id: str) -> None:
        if user_id != self._account_id:
            raise CalendarProviderError("Calendar account context does not match the signed session.")

    async def _post(self, path: str, payload: dict[str, object]) -> dict:
        signature = sign_internal_request(payload, self._internal_secret)
        headers = {"Content-Type": "application/json", "X-Internal-Auth": signature}
        try:
            if self._client is not None:
                response = await self._client.post(
                    f"{self._gateway_url}{path}", json=payload, headers=headers
                )
            else:
                async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
                    response = await client.post(
                        f"{self._gateway_url}{path}", json=payload, headers=headers
                    )
            if response.status_code >= 400:
                detail = ""
                try:
                    raw_detail = response.json().get("detail")
                    if isinstance(raw_detail, str):
                        detail = raw_detail
                except (AttributeError, TypeError, ValueError):
                    detail = ""
                if response.status_code in {401, 403}:
                    raise CalendarProviderError("Calendar access is not available for this voice session.")
                if response.status_code == 409 or "Reconnect Google Calendar" in detail or "permissions are incomplete" in detail:
                    raise CalendarProviderError("Connect Google Calendar before asking about personal events.")
                # Gateway errors are deliberately allowlisted at the gateway
                # boundary. Preserve those safe diagnostics instead of
                # collapsing every MCP failure into the same opaque sentence.
                if response.status_code == 503 and detail.startswith("Google Calendar"):
                    raise CalendarProviderError(detail)
                raise CalendarProviderError("Calendar service is temporarily unavailable.")
            data = response.json()
        except CalendarProviderError:
            raise
        except (httpx.HTTPError, ValueError) as error:
            raise CalendarProviderError("Calendar service is temporarily unavailable.") from error
        if not isinstance(data, dict):
            raise CalendarProviderError("Calendar service returned an invalid response.")
        return data

    def _result(self, payload: dict) -> CalendarResult:
        provider = payload.get("provider")
        demo = payload.get("demo")
        if provider not in {"mock", "google_mcp"} or not isinstance(demo, bool):
            raise CalendarProviderError("Calendar service returned an unsafe provider marker.")
        if (provider == "mock") != demo:
            raise CalendarProviderError("Calendar service returned an inconsistent provider marker.")
        raw_events = payload.get("events", [])
        if not isinstance(raw_events, list):
            raise CalendarProviderError("Calendar service returned invalid events.")
        events: list[CalendarEvent] = []
        for raw in raw_events[:50]:
            if not isinstance(raw, dict):
                continue
            try:
                event_id = str(raw["id"])[:200]
                title = str(raw["title"])[:200]
                event_start = _parse_calendar_datetime(str(raw["start"]))
                event_end = _parse_calendar_datetime(str(raw["end"]))
            except (KeyError, TypeError, ValueError):
                continue
            if not event_id or not title or event_end <= event_start:
                continue
            events.append(CalendarEvent(event_id, self._account_id, title, event_start, event_end, str(raw.get("location", ""))[:200] or None))
        return CalendarResult(str(provider), demo, self._account_id, tuple(events))


def _parse_calendar_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed
