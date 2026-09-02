"""Local FastAPI gateway: account sessions, enrollment, and LiveKit tokens."""

from __future__ import annotations

import base64
import json
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from fastapi import Cookie, Depends, FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.assistant.tools.calendar import MockCalendarProvider
from app.assistant.voice_service import build_account_voice_auth_gate
from app.assistant_gateway.security import hash_password, new_session_token, verify_password
from app.assistant_gateway.store import GatewayStore, User, VoiceProfile
from voiceauth.audio import decode_recording
from voiceauth.errors import VoiceAuthError

SESSION_COOKIE = "who_speak_session"
SESSION_TTL = timedelta(hours=8)
AGENT_NAME = "voice-auth-gate"


class GatewaySettings(BaseModel):
    database_path: Path
    livekit_url: str
    livekit_api_key: str
    livekit_api_secret: str
    mcp_provider: str
    mock_calendar_enabled: bool
    agent_runtime: str = "livekit"
    pipecat_supervisor_url: str = "http://127.0.0.1:8021"
    pipecat_supervisor_secret: str = ""

    @classmethod
    def from_environment(cls) -> "GatewaySettings":
        workspace = Path(__file__).resolve().parents[2]
        raw_database = os.getenv("WHO_SPEAK_GATEWAY_DATABASE", "app/voice_verification/data/who_speak_gateway.db")
        provider = os.getenv("MCP_PROVIDER", "mock").strip().lower()
        if provider not in {"mock", "mcp"}:
            raise ValueError("MCP_PROVIDER must be either 'mock' or 'mcp'.")
        runtime = os.getenv("VOICE_AGENT_RUNTIME", "livekit").strip().lower()
        if runtime not in {"livekit", "pipecat"}:
            raise ValueError("VOICE_AGENT_RUNTIME must be either 'livekit' or 'pipecat'.")
        return cls(
            database_path=(workspace / raw_database).resolve(),
            livekit_url=os.getenv("LIVEKIT_URL", "ws://127.0.0.1:7880").rstrip("/"),
            livekit_api_key=os.getenv("LIVEKIT_API_KEY", "").strip(),
            livekit_api_secret=os.getenv("LIVEKIT_API_SECRET", "").strip(),
            mcp_provider=provider,
            mock_calendar_enabled=os.getenv("MOCK_CALENDAR_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"},
            agent_runtime=runtime,
            pipecat_supervisor_url=os.getenv("PIPECAT_SUPERVISOR_URL", "http://127.0.0.1:8021").rstrip("/"),
            pipecat_supervisor_secret=os.getenv("PIPECAT_SUPERVISOR_SECRET", "").strip(),
        )


class RegisterRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=10, max_length=256)
    display_name: str = Field(min_length=1, max_length=120)


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=256)


class EnrollRequest(BaseModel):
    samples_b64: list[str] = Field(min_length=3, max_length=3)
    display_name: str | None = Field(default=None, max_length=120)


class CreateDemoEventRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    start: datetime
    end: datetime


def _user_view(user: User, profile: VoiceProfile | None) -> dict:
    return {
        "id": user.id,
        "email": user.email,
        "display_name": user.display_name,
        "voice_enrolled": profile is not None,
    }


def _decode_sample(value: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except (ValueError, UnicodeError) as error:
        raise ValueError("Voice samples must be valid base64 audio payloads.") from error


def _normalize_email(value: str) -> str:
    email = value.strip().lower()
    if "@" not in email or email.startswith("@") or email.endswith("@"):
        raise ValueError("Enter a valid email address.")
    return email


def _require_livekit(settings: GatewaySettings) -> None:
    if not settings.livekit_api_key or not settings.livekit_api_secret:
        raise ValueError("LIVEKIT_API_KEY and LIVEKIT_API_SECRET are required to create room tokens.")


def create_app(settings: GatewaySettings | None = None) -> FastAPI:
    active = settings or GatewaySettings.from_environment()
    store = GatewayStore(active.database_path)
    calendar = MockCalendarProvider(store)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        store.initialize()
        yield

    app = FastAPI(title="Who Speak AI Gateway", version="0.2.0", lifespan=lifespan)
    app.state.settings = active
    app.state.store = store
    app.state.calendar = calendar
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
    )

    def current_user(session_token: str | None = Cookie(default=None, alias=SESSION_COOKIE)) -> User:
        if not session_token:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Sign in is required.")
        user = store.session_user(session_token)
        if user is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Your session has expired. Sign in again.")
        return user

    def set_session(response: Response, token: str) -> None:
        response.set_cookie(
            key=SESSION_COOKIE,
            value=token,
            httponly=True,
            samesite="lax",
            secure=False,
            max_age=int(SESSION_TTL.total_seconds()),
            path="/",
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/auth/register", status_code=status.HTTP_201_CREATED)
    def register(payload: RegisterRequest, response: Response) -> dict:
        try:
            user = store.create_user(
                email=_normalize_email(payload.email), password_hash=hash_password(payload.password), display_name=payload.display_name
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except Exception as error:
            if "UNIQUE constraint failed" in str(error):
                raise HTTPException(status_code=409, detail="An account with this email already exists.") from error
            raise
        token = new_session_token()
        store.create_session(user.id, token, ttl=SESSION_TTL)
        set_session(response, token)
        return _user_view(user, None)

    @app.post("/api/auth/login")
    def login(payload: LoginRequest, response: Response) -> dict:
        try:
            email = _normalize_email(payload.email)
        except ValueError:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password.") from None
        result = store.user_by_email(email)
        if result is None or not verify_password(payload.password, result[1]):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password.")
        user = result[0]
        token = new_session_token()
        store.create_session(user.id, token, ttl=SESSION_TTL)
        set_session(response, token)
        return _user_view(user, store.voice_profile(user.id))

    @app.post("/api/auth/logout", status_code=status.HTTP_204_NO_CONTENT)
    def logout(session_token: str | None = Cookie(default=None, alias=SESSION_COOKIE)) -> Response:
        if session_token:
            store.delete_session(session_token)
        response = Response(status_code=status.HTTP_204_NO_CONTENT)
        response.delete_cookie(SESSION_COOKIE, path="/")
        return response

    @app.get("/api/auth/me")
    def me(user: User = Depends(current_user)) -> dict:
        return _user_view(user, store.voice_profile(user.id))

    @app.post("/api/voice/enroll", status_code=status.HTTP_201_CREATED)
    def enroll_voice(payload: EnrollRequest, user: User = Depends(current_user)) -> dict:
        gate = None
        try:
            recordings = [decode_recording(_decode_sample(sample)) for sample in payload.samples_b64]
            gate = build_account_voice_auth_gate(user.id)
            result = gate.enroll(payload.display_name or user.display_name, recordings)
            store.save_voice_profile(
                VoiceProfile(user.id, gate.context_id, result.identity_id, result.display_name)
            )
        except (ValueError, VoiceAuthError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        finally:
            if gate is not None:
                gate.close()
        return {"identity_id": result.identity_id, "display_name": result.display_name, "enrolled": True}

    @app.post("/api/livekit/token")
    async def livekit_token(user: User = Depends(current_user)) -> dict:
        profile = store.voice_profile(user.id)
        if profile is None:
            raise HTTPException(status_code=409, detail="Enroll three voice samples before joining the assistant.")
        try:
            _require_livekit(active)
            from livekit import api

            room_name = f"voice-agent-{uuid4()}"
            participant_identity = f"web-{uuid4()}"
            session_id = str(uuid4())
            access_token = (
                api.AccessToken(active.livekit_api_key, active.livekit_api_secret)
                .with_identity(participant_identity)
                .with_name(user.display_name)
                .with_grants(
                    api.VideoGrants(
                        room_join=True,
                        room=room_name,
                        can_publish=True,
                        can_publish_data=True,
                        can_subscribe=True,
                        can_publish_sources=["microphone"],
                    )
                )
            )
            if active.agent_runtime == "livekit":
                metadata = json.dumps(
                    {
                        "user_id": user.id,
                        "voice_identity_id": profile.matcher_identity_id,
                        "display_name": profile.display_name,
                    },
                    separators=(",", ":"),
                )
                access_token = access_token.with_room_config(
                    api.RoomConfiguration(
                        agents=[api.RoomAgentDispatch(agent_name=AGENT_NAME, metadata=metadata)]
                    )
                )
            else:
                if len(active.pipecat_supervisor_secret) < 32:
                    raise ValueError(
                        "PIPECAT_SUPERVISOR_SECRET must be at least 32 characters in Pipecat mode."
                    )
                from app.assistant.pipecat_runtime.session import (
                    PipecatSessionDescriptor,
                    sign_session_descriptor,
                )

                descriptor = PipecatSessionDescriptor(
                    session_id=session_id,
                    room_name=room_name,
                    participant_identity=participant_identity,
                    user_id=user.id,
                    voice_identity_id=profile.matcher_identity_id,
                    display_name=profile.display_name,
                    issued_at=int(datetime.now(UTC).timestamp()),
                )
                descriptor_token = sign_session_descriptor(
                    descriptor, active.pipecat_supervisor_secret
                )
                try:
                    import httpx

                    async with httpx.AsyncClient(timeout=5.0) as client:
                        supervisor_response = await client.post(
                            f"{active.pipecat_supervisor_url}/internal/pipecat/sessions",
                            json={"descriptor": descriptor_token},
                        )
                        supervisor_response.raise_for_status()
                except ImportError as error:
                    raise HTTPException(
                        status_code=503,
                        detail="Install the gateway dependencies before using Pipecat mode.",
                    ) from error
                except httpx.HTTPError as error:
                    raise HTTPException(
                        status_code=503,
                        detail="Local Pipecat supervisor is unavailable. Start it before joining.",
                    ) from error
            return {
                "server_url": active.livekit_url,
                "participant_token": access_token.to_jwt(),
                "room_name": room_name,
                "session_id": session_id,
                "participant_identity": participant_identity,
                "agent_identity": (
                    AGENT_NAME
                    if active.agent_runtime == "livekit"
                    else f"pipecat-agent-{session_id[:8]}"
                ),
                "runtime": active.agent_runtime,
            }
        except ValueError as error:
            raise HTTPException(status_code=500, detail=str(error)) from error

    @app.get("/api/agent/capabilities")
    async def capabilities(user: User = Depends(current_user)) -> dict:
        # Voice authorization is runtime/session-scoped in the Agent. The UI can
        # show connected integrations but must not infer private capability.
        return {
            "user_id": user.id,
            "mcp_provider": active.mcp_provider,
            "mock_calendar": active.mcp_provider == "mock" and active.mock_calendar_enabled,
        }

    @app.get("/api/mock-calendar/events")
    async def list_demo_events(user: User = Depends(current_user)) -> dict:
        if active.mcp_provider != "mock" or not active.mock_calendar_enabled:
            raise HTTPException(status_code=503, detail="Mock calendar is disabled.")
        now = datetime.now(UTC)
        result = await calendar.list_events(user_id=user.id, start=now, end=now + timedelta(days=7))
        return {
            "provider": result.provider,
            "demo": result.demo,
            "events": [
                {"id": event.id, "title": event.title, "start": event.start.isoformat(), "end": event.end.isoformat()}
                for event in result.events
            ],
        }

    @app.post("/api/mock-calendar/events", status_code=status.HTTP_201_CREATED)
    async def create_demo_event(payload: CreateDemoEventRequest, user: User = Depends(current_user)) -> dict:
        if active.mcp_provider != "mock" or not active.mock_calendar_enabled:
            raise HTTPException(status_code=503, detail="Mock calendar is disabled.")
        try:
            result = await calendar.create_event(user_id=user.id, title=payload.title, start=payload.start, end=payload.end)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        event = result.events[0]
        return {
            "provider": result.provider,
            "demo": result.demo,
            "event": {"id": event.id, "title": event.title, "start": event.start.isoformat(), "end": event.end.isoformat()},
        }

    return app


app = create_app()
