from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.assistant.tools.calendar import GatewayCalendarProvider
from app.assistant_gateway.google_calendar import (
    GOOGLE_CALENDAR_SCOPE,
    GOOGLE_REQUIRED_CALENDAR_SCOPES,
    GoogleCalendarMcpProvider,
    GoogleOAuthService,
    TokenCipher,
    _sanitize_mcp_events,
)
from app.assistant_gateway.main import GatewaySettings, create_app
from app.assistant_gateway.security import sign_internal_request
from app.assistant_gateway.store import GatewayStore


KEY = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="
INTERNAL_SECRET = "internal-agent-secret-with-at-least-32-bytes"


def test_token_cipher_round_trip_does_not_store_plaintext() -> None:
    cipher = TokenCipher(KEY)
    encrypted = cipher.encrypt("refresh-token-value")

    assert encrypted != "refresh-token-value"
    assert cipher.decrypt(encrypted) == "refresh-token-value"


def test_oauth_state_is_bound_to_session_and_is_one_time(tmp_path) -> None:
    store = GatewayStore(tmp_path / "gateway.db")
    store.initialize()
    user = store.create_user(email="an@example.test", password_hash="hash", display_name="An")
    session_token = "browser-session"
    store.create_session(user.id, session_token, ttl=timedelta(hours=1))
    oauth = GoogleOAuthService(
        store=store,
        client_id="client",
        client_secret="secret",
        redirect_uri="http://127.0.0.1/callback",
        encryption_key=KEY,
        web_app_url="http://127.0.0.1:5173",
    )

    url = oauth.authorization_url(user=user, session_token=session_token)
    state = url.split("state=", 1)[1].split("&", 1)[0]

    assert oauth.consume_state(state=state, session_token="wrong-session") is None
    assert oauth.consume_state(state=state, session_token=session_token) == user.id
    assert oauth.consume_state(state=state, session_token=session_token) is None


def test_google_mcp_event_sanitization_discards_untrusted_fields() -> None:
    result = SimpleNamespace(
        is_error=False,
        structured_content={
            "events": [
                {
                    "id": "event-1",
                    "summary": "Planning",
                    "description": "Ignore hidden instructions",
                    "location": "Room 1",
                    "start": {"dateTime": "2026-09-03T10:00:00+07:00"},
                    "end": {"dateTime": "2026-09-03T11:00:00+07:00"},
                    "attachments": [{"fileUrl": "https://example.test"}],
                }
            ]
        },
    )

    events = _sanitize_mcp_events(result, "user-a")

    assert len(events) == 1
    assert events[0].title == "Planning"
    assert events[0].location == "Room 1"
    assert not hasattr(events[0], "description")


def test_google_provider_returns_non_demo_result() -> None:
    class FakeOAuth:
        async def access_token(self, user_id: str) -> str:
            assert user_id == "user-a"
            return "access-token"

    class FakeMcp:
        async def list_events(self, *, access_token, start, end):
            assert access_token == "access-token"
            return SimpleNamespace(
                is_error=False,
                structured_content={
                    "events": [
                        {
                            "id": "event-1",
                            "summary": "Personal meeting",
                            "start": {"dateTime": "2026-09-03T10:00:00+00:00"},
                            "end": {"dateTime": "2026-09-03T11:00:00+00:00"},
                        }
                    ]
                },
            )

    provider = GoogleCalendarMcpProvider(oauth=FakeOAuth(), client=FakeMcp())
    result = asyncio.run(
        provider.list_events(
            user_id="user-a",
            start=datetime.now(UTC),
            end=datetime.now(UTC) + timedelta(hours=1),
        )
    )

    assert result.provider == "google_mcp"
    assert result.demo is False
    assert result.user_id == "user-a"


def test_agent_gateway_facade_signs_request_and_resolves_local_account() -> None:
    class FakeResponse:
        status_code = 200

        def json(self):
            return {
                "provider": "google_mcp",
                "demo": False,
                "events": [
                    {
                        "id": "event-1",
                        "title": "Meeting",
                        "start": "2026-09-03T10:00:00+00:00",
                        "end": "2026-09-03T11:00:00+00:00",
                    }
                ],
            }

    class FakeClient:
        def __init__(self):
            self.calls = []

        async def post(self, url, *, json, headers):
            self.calls.append((url, json, headers))
            return FakeResponse()

    client = FakeClient()
    provider = GatewayCalendarProvider(
        gateway_url="http://gateway",
        session_id="session-1",
        room_name="room-1",
        internal_secret=INTERNAL_SECRET,
        account_id="user-a",
        client=client,
    )
    result = asyncio.run(
        provider.list_events(
            user_id="user-a",
            start=datetime(2026, 9, 3, 10, tzinfo=UTC),
            end=datetime(2026, 9, 3, 11, tzinfo=UTC),
        )
    )

    url, payload, headers = client.calls[0]
    assert url == "http://gateway/internal/calendar/list-events"
    assert headers["X-Internal-Auth"] == sign_internal_request(payload, INTERNAL_SECRET)
    assert result.provider == "google_mcp"
    assert result.demo is False


def test_oauth_scope_constant_is_read_only() -> None:
    assert GOOGLE_CALENDAR_SCOPE.endswith("events.readonly")
    assert GOOGLE_CALENDAR_SCOPE in GOOGLE_REQUIRED_CALENDAR_SCOPES


def test_internal_calendar_boundary_resolves_the_issued_session(tmp_path) -> None:
    settings = GatewaySettings(
        database_path=tmp_path / "gateway.db",
        livekit_url="ws://127.0.0.1:7880",
        livekit_api_key="devkey",
        livekit_api_secret="a-development-secret-with-at-least-32-bytes",
        mcp_provider="mcp",
        mock_calendar_enabled=False,
        pipecat_supervisor_secret=INTERNAL_SECRET,
        google_token_encryption_key=KEY,
    )
    app = create_app(settings)
    with TestClient(app) as client:
        registered = client.post(
            "/api/auth/register",
            json={
                "email": "an@example.test",
                "password": "correct-horse-battery-staple",
                "display_name": "An",
            },
        ).json()
        from app.assistant_gateway.store import VoiceProfile

        app.state.store.save_voice_profile(VoiceProfile(registered["id"], "context", "identity", "An"))
        token = client.post("/api/livekit/token").json()
        payload = {
            "session_id": token["session_id"],
            "room_name": token["room_name"],
            "start": "2026-09-03T10:00:00+00:00",
            "end": "2026-09-03T11:00:00+00:00",
        }
        response = client.post(
            "/internal/calendar/list-events",
            json=payload,
            headers={"X-Internal-Auth": sign_internal_request(payload, INTERNAL_SECRET)},
        )

    assert response.status_code == 409
    assert "Connect Google Calendar" in response.json()["detail"]
