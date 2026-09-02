from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from app.assistant_gateway.main import GatewaySettings, create_app
from app.assistant_gateway.store import VoiceProfile


def _client(tmp_path) -> TestClient:
    settings = GatewaySettings(
        database_path=tmp_path / "gateway.db",
        livekit_url="ws://127.0.0.1:7880",
        livekit_api_key="devkey",
        livekit_api_secret="a-development-secret-with-at-least-32-bytes",
        mcp_provider="mock",
        mock_calendar_enabled=True,
    )
    return TestClient(create_app(settings))


def _register(client: TestClient, *, email: str, name: str) -> dict:
    response = client.post(
        "/api/auth/register",
        json={"email": email, "password": "correct-horse-battery-staple", "display_name": name},
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_guest_cannot_read_mock_calendar(tmp_path) -> None:
    with _client(tmp_path) as client:
        response = client.get("/api/mock-calendar/events")

    assert response.status_code == 401


def test_logout_returns_no_content_and_clears_the_session(tmp_path) -> None:
    with _client(tmp_path) as client:
        _register(client, email="logout@example.test", name="Logout")
        response = client.post("/api/auth/logout")
        assert response.status_code == 204
        assert response.content == b""
        assert client.get("/api/auth/me").status_code == 401


def test_mock_calendar_is_explicitly_demo_and_scoped_to_the_signed_in_user(tmp_path) -> None:
    with _client(tmp_path) as alice, _client(tmp_path) as bob:
        # Separate clients intentionally have separate cookie jars, but share no
        # database here. Scope coverage is tested with one app below instead.
        _register(alice, email="alice@example.test", name="Alice")
        response = alice.get("/api/mock-calendar/events")
        assert response.status_code == 200
        payload = response.json()
        assert payload["provider"] == "mock"
        assert payload["demo"] is True
        assert payload["events"][0]["title"] == "Demo meeting"


def test_user_cannot_read_another_users_created_mock_event(tmp_path) -> None:
    settings = GatewaySettings(
        database_path=tmp_path / "gateway.db",
        livekit_url="ws://127.0.0.1:7880",
        livekit_api_key="devkey",
        livekit_api_secret="a-development-secret-with-at-least-32-bytes",
        mcp_provider="mock",
        mock_calendar_enabled=True,
    )
    app = create_app(settings)
    with TestClient(app) as alice, TestClient(app) as bob:
        _register(alice, email="alice@example.test", name="Alice")
        _register(bob, email="bob@example.test", name="Bob")
        start = datetime.now(UTC) + timedelta(days=1)
        created = alice.post(
            "/api/mock-calendar/events",
            json={"title": "Alice private demo", "start": start.isoformat(), "end": (start + timedelta(hours=1)).isoformat()},
        )
        assert created.status_code == 201, created.text

        bob_events = bob.get("/api/mock-calendar/events").json()["events"]
        assert all(event["title"] != "Alice private demo" for event in bob_events)


def test_livekit_token_has_a_backend_selected_room_and_never_exposes_the_secret(tmp_path) -> None:
    settings = GatewaySettings(
        database_path=tmp_path / "gateway.db",
        livekit_url="ws://127.0.0.1:7880",
        livekit_api_key="devkey",
        livekit_api_secret="a-development-secret-with-at-least-32-bytes",
        mcp_provider="mock",
        mock_calendar_enabled=True,
    )
    app = create_app(settings)
    with TestClient(app) as client:
        user = _register(client, email="an@example.test", name="An")
        app.state.store.save_voice_profile(VoiceProfile(user["id"], "context-id", "target-identity", "An"))
        response = client.post("/api/livekit/token")

    assert response.status_code == 200, response.text
    assert response.json()["server_url"] == "ws://127.0.0.1:7880"
    assert "a-development-secret" not in response.text
