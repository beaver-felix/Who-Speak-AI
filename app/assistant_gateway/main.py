"""Local FastAPI gateway: account sessions, enrollment, and LiveKit tokens."""

from __future__ import annotations

import base64
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from fastapi import Cookie, Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from app.assistant.tools.calendar import MockCalendarProvider
from app.assistant_gateway.google_calendar import (
    GoogleCalendarMcpClient,
    GoogleCalendarMcpProvider,
    GoogleCalendarError,
    GoogleOAuthService,
    LocalGoogleCalendarMcpClient,
    LocalGoogleCalendarMcpProvider,
)
from app.assistant.voice_service import build_account_voice_auth_gate
from app.assistant_gateway.security import (
    digest_session_token,
    hash_password,
    new_session_token,
    verify_internal_request,
    verify_password,
)
from app.assistant_gateway.store import GatewayStore, User, VoiceProfile
from voiceauth.audio import decode_recording
from voiceauth.errors import VoiceAuthError

SESSION_COOKIE = "who_speak_session"
SESSION_TTL = timedelta(hours=8)
AGENT_NAME = "voice-auth-gate"
logger = logging.getLogger(__name__)


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
    gateway_url: str = "http://127.0.0.1:8020"
    web_app_url: str = "http://127.0.0.1:5173"
    google_oauth_client_id: str = ""
    google_oauth_client_secret: str = ""
    google_oauth_redirect_uri: str = "http://127.0.0.1:8020/api/integrations/google-calendar/callback"
    google_token_encryption_key: str = ""
    google_mcp_endpoint: str = "https://calendarmcp.googleapis.com/mcp/v1"
    google_mcp_timeout_seconds: float = 8.0

    @classmethod
    def from_environment(cls) -> "GatewaySettings":
        workspace = Path(__file__).resolve().parents[2]
        raw_database = os.getenv("WHO_SPEAK_GATEWAY_DATABASE", "app/voice_verification/data/who_speak_gateway.db")
        provider = os.getenv("MCP_PROVIDER", "mock").strip().lower()
        if provider not in {"mock", "mcp", "local"}:
            raise ValueError("MCP_PROVIDER must be either 'mock', 'mcp', or 'local'.")
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
            gateway_url=os.getenv("WHO_SPEAK_GATEWAY_URL", "http://127.0.0.1:8020").rstrip("/"),
            web_app_url=os.getenv("WHO_SPEAK_WEB_URL", "http://127.0.0.1:5173").rstrip("/"),
            google_oauth_client_id=os.getenv("GOOGLE_OAUTH_CLIENT_ID", "").strip(),
            google_oauth_client_secret=os.getenv("GOOGLE_OAUTH_CLIENT_SECRET", "").strip(),
            google_oauth_redirect_uri=os.getenv(
                "GOOGLE_OAUTH_REDIRECT_URI",
                "http://127.0.0.1:8020/api/integrations/google-calendar/callback",
            ).strip(),
            google_token_encryption_key=os.getenv("GOOGLE_TOKEN_ENCRYPTION_KEY", "").strip(),
            google_mcp_endpoint=os.getenv(
                "GOOGLE_MCP_ENDPOINT", "https://calendarmcp.googleapis.com/mcp/v1"
            ).strip(),
            google_mcp_timeout_seconds=float(os.getenv("GOOGLE_MCP_TIMEOUT_SECONDS", "8")),
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


class InternalCalendarRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    room_name: str = Field(min_length=1, max_length=256)
    start: datetime
    end: datetime


class InternalCreateCalendarRequest(InternalCalendarRequest):
    title: str = Field(min_length=1, max_length=200)


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
    google_oauth = GoogleOAuthService(
        store=store,
        client_id=active.google_oauth_client_id,
        client_secret=active.google_oauth_client_secret,
        redirect_uri=active.google_oauth_redirect_uri,
        encryption_key=active.google_token_encryption_key,
        web_app_url=active.web_app_url,
    )
    if active.mcp_provider == "local":
        google_calendar = LocalGoogleCalendarMcpProvider(
            oauth=google_oauth,
            client=LocalGoogleCalendarMcpClient(
                endpoint=active.google_mcp_endpoint,
                timeout_seconds=active.google_mcp_timeout_seconds,
            ),
        )
    else:
        google_calendar = GoogleCalendarMcpProvider(
            oauth=google_oauth,
            client=GoogleCalendarMcpClient(
                endpoint=active.google_mcp_endpoint,
                timeout_seconds=active.google_mcp_timeout_seconds,
            ),
        )
    # This registry contains only an account-session digest. It is not a
    # bearer token and lets a local Agent prove which room it belongs to.
    agent_sessions: dict[str, dict[str, str]] = {}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        store.initialize()
        yield

    app = FastAPI(title="Who Speak AI Gateway", version="0.2.0", lifespan=lifespan)
    app.state.settings = active
    app.state.store = store
    app.state.calendar = calendar
    app.state.google_oauth = google_oauth
    app.state.google_calendar = google_calendar
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "X-Internal-Auth"],
    )

    def current_session(
        session_token: str | None = Cookie(default=None, alias=SESSION_COOKIE),
    ) -> tuple[User, str]:
        if not session_token:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Sign in is required.")
        user = store.session_user(session_token)
        if user is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Your session has expired. Sign in again.")
        return user, session_token

    def current_user(session: tuple[User, str] = Depends(current_session)) -> User:
        return session[0]

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
    async def livekit_token(session: tuple[User, str] = Depends(current_session)) -> dict:
        user = session[0]
        session_token = session[1]
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
                        "session_id": session_id,
                        "room_name": room_name,
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
            agent_sessions[session_id] = {
                "user_id": user.id,
                "room_name": room_name,
                "session_digest": digest_session_token(session_token),
            }
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
            "google_calendar": google_oauth.status(user.id),
        }

    @app.get("/api/integrations/google-calendar/start")
    async def start_google_calendar_oauth(
        session: tuple[User, str] = Depends(current_session),
    ) -> RedirectResponse:
        if active.mcp_provider not in {"mcp", "local"}:
            raise HTTPException(status_code=409, detail="Google Calendar is disabled while MCP_PROVIDER=mock.")
        try:
            url = google_oauth.authorization_url(user=session[0], session_token=session[1])
        except GoogleCalendarError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return RedirectResponse(url=url, status_code=status.HTTP_303_SEE_OTHER)

    @app.get("/api/integrations/google-calendar/callback")
    async def finish_google_calendar_oauth(
        request: Request,
        code: str | None = None,
        state: str | None = None,
        error: str | None = None,
        session_token: str | None = Cookie(default=None, alias=SESSION_COOKIE),
    ) -> RedirectResponse:
        result = "error"
        if session_token and state and not error and code:
            user_id = google_oauth.consume_state(state=state, session_token=session_token)
            user = store.user_by_id(user_id) if user_id else None
            current = store.session_user(session_token)
            if user is not None and current is not None and current.id == user.id:
                try:
                    await google_oauth.finish(user=user, code=code, session_token=session_token)
                except GoogleCalendarError:
                    # Do not reflect provider error text into a redirect URL.
                    result = "error"
                else:
                    result = "connected"
        target = f"{google_oauth.web_app_url}/?google_calendar={result}"
        return RedirectResponse(url=target, status_code=status.HTTP_303_SEE_OTHER)

    @app.get("/api/integrations/google-calendar/status")
    async def google_calendar_status(user: User = Depends(current_user)) -> dict[str, object]:
        return google_oauth.status(user.id)

    @app.post("/api/integrations/google-calendar/disconnect", status_code=status.HTTP_204_NO_CONTENT)
    async def disconnect_google_calendar(user: User = Depends(current_user)) -> Response:
        google_oauth.disconnect(user.id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    def resolve_agent_session(payload: dict[str, object], request: Request) -> User:
        signature = request.headers.get("X-Internal-Auth", "")
        if not verify_internal_request(payload, signature, active.pipecat_supervisor_secret):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid internal agent context.")
        session_id = payload.get("session_id")
        room_name = payload.get("room_name")
        if not isinstance(session_id, str) or not isinstance(room_name, str):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid internal agent context.")
        binding = agent_sessions.get(session_id)
        if binding is None or binding.get("room_name") != room_name:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Agent session is no longer active.")
        user = store.session_user_by_digest(binding["session_digest"])
        if user is None or user.id != binding.get("user_id"):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Who Speak session has expired.")
        return user

    def internal_calendar_payload(payload: InternalCalendarRequest) -> dict[str, object]:
        values: dict[str, object] = {
            "session_id": payload.session_id,
            "room_name": payload.room_name,
            "start": payload.start.isoformat(),
            "end": payload.end.isoformat(),
        }
        if isinstance(payload, InternalCreateCalendarRequest):
            values["title"] = payload.title
        return values

    def calendar_result_view(result) -> dict[str, object]:
        return {
            "provider": result.provider,
            "demo": result.demo,
            "events": [
                {
                    "id": event.id,
                    "title": event.title,
                    "start": event.start.isoformat(),
                    "end": event.end.isoformat(),
                    **({"location": event.location} if event.location else {}),
                }
                for event in result.events[:50]
            ],
        }

    @app.post("/internal/calendar/list-events")
    async def internal_list_events(payload: InternalCalendarRequest, request: Request) -> dict[str, object]:
        user = resolve_agent_session(internal_calendar_payload(payload), request)
        if active.mcp_provider not in {"mcp", "local"}:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Google Calendar MCP is not active.")
        if payload.end <= payload.start:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid calendar time range.")
        try:
            result = await google_calendar.list_events(user_id=user.id, start=payload.start, end=payload.end)
        except GoogleCalendarError as error:
            detail = str(error)
            # Keep the browser/LLM message safe, but retain enough metadata in
            # the gateway logs to distinguish OAuth, MCP, and provider failures.
            logger.warning(
                "calendar_list_events_failed user_id=%s error_type=%s",
                user.id,
                type(error).__name__,
            )
            code = status.HTTP_409_CONFLICT if "Connect Google" in detail or "authorization expired" in detail else status.HTTP_503_SERVICE_UNAVAILABLE
            raise HTTPException(status_code=code, detail=detail) from error
        return calendar_result_view(result)

    @app.post("/internal/calendar/create-event")
    async def internal_create_event(payload: InternalCreateCalendarRequest, request: Request) -> dict[str, object]:
        user = resolve_agent_session(internal_calendar_payload(payload), request)
        if active.mcp_provider not in {"mcp", "local"}:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Google Calendar MCP is not active.")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Creating Google Calendar events is not enabled in this rollout.")

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
