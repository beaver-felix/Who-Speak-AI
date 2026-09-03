"""Google OAuth and the allowlisted remote Calendar MCP provider.

The gateway owns refresh tokens and creates a short-lived MCP client per call.
The Agent only receives sanitized CalendarResult values through its internal
gateway facade.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlencode
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

from app.assistant.tools.calendar import CalendarResult, CalendarProviderError, _parse_calendar_datetime
from app.assistant_gateway.store import CalendarEvent, GatewayStore, GoogleCalendarConnection, User

GOOGLE_AUTHORIZATION_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_ENDPOINT = "https://openidconnect.googleapis.com/v1/userinfo"
GOOGLE_READ_SCOPES = (
    "openid",
    "email",
    "profile",
    "https://www.googleapis.com/auth/calendar.calendarlist.readonly",
    "https://www.googleapis.com/auth/calendar.events.freebusy",
    "https://www.googleapis.com/auth/calendar.events.readonly",
)
GOOGLE_CALENDAR_SCOPE = "https://www.googleapis.com/auth/calendar.events.readonly"
GOOGLE_REQUIRED_CALENDAR_SCOPES = frozenset(
    {
        "https://www.googleapis.com/auth/calendar.calendarlist.readonly",
        "https://www.googleapis.com/auth/calendar.events.freebusy",
        GOOGLE_CALENDAR_SCOPE,
    }
)
logger = logging.getLogger(__name__)


class GoogleCalendarError(RuntimeError):
    """A safe, user-facing Google Calendar integration error."""


class TokenCipher:
    """AES-GCM wrapper for refresh tokens at rest."""

    def __init__(self, encoded_key: str) -> None:
        if not encoded_key.strip():
            raise GoogleCalendarError("GOOGLE_TOKEN_ENCRYPTION_KEY is required for Google Calendar.")
        raw = encoded_key.encode("utf-8")
        try:
            decoded = base64.urlsafe_b64decode(encoded_key + "=" * (-len(encoded_key) % 4))
        except (ValueError, base64.binascii.Error):
            decoded = b""
        key = decoded if len(decoded) == 32 else raw if len(raw) == 32 else b""
        if len(key) != 32:
            raise GoogleCalendarError(
                "GOOGLE_TOKEN_ENCRYPTION_KEY must be a URL-safe base64-encoded 32-byte key."
            )
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError as error:
            raise GoogleCalendarError("Install cryptography before enabling Google Calendar.") from error
        self._aes = AESGCM(key)

    def encrypt(self, value: str) -> str:
        nonce = secrets.token_bytes(12)
        encrypted = self._aes.encrypt(nonce, value.encode("utf-8"), None)
        return base64.urlsafe_b64encode(nonce + encrypted).decode("ascii")

    def decrypt(self, value: str) -> str:
        try:
            encoded = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
            plaintext = self._aes.decrypt(encoded[:12], encoded[12:], None)
            return plaintext.decode("utf-8")
        except Exception as error:
            raise GoogleCalendarError("Stored Google Calendar credentials are invalid.") from error


class GoogleOAuthService:
    def __init__(
        self,
        *,
        store: GatewayStore,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        encryption_key: str,
        web_app_url: str,
    ) -> None:
        self._store = store
        self._client_id = client_id
        self._client_secret = client_secret
        self._redirect_uri = redirect_uri
        self._encryption_key = encryption_key
        self._web_app_url = web_app_url.rstrip("/")

    @property
    def configured(self) -> bool:
        return bool(self._client_id and self._client_secret and self._redirect_uri and self._encryption_key)

    def authorization_url(self, *, user: User, session_token: str) -> str:
        if not self.configured:
            raise GoogleCalendarError("Google Calendar OAuth is not configured on the gateway.")
        state = secrets.token_urlsafe(32)
        now = datetime.now(UTC)
        self._store.create_google_oauth_state(
            state_hash=hashlib.sha256(state.encode("utf-8")).hexdigest(),
            user_id=user.id,
            session_binding=hashlib.sha256(session_token.encode("utf-8")).hexdigest(),
            expires_at=now + timedelta(minutes=10),
        )
        return GOOGLE_AUTHORIZATION_ENDPOINT + "?" + urlencode(
            {
                "client_id": self._client_id,
                "redirect_uri": self._redirect_uri,
                "response_type": "code",
                "scope": " ".join(GOOGLE_READ_SCOPES),
                "access_type": "offline",
                "include_granted_scopes": "true",
                "prompt": "consent",
                "state": state,
            }
        )

    def consume_state(self, *, state: str, session_token: str) -> str | None:
        if not state or not session_token:
            return None
        return self._store.consume_google_oauth_state(
            state_hash=hashlib.sha256(state.encode("utf-8")).hexdigest(),
            session_binding=hashlib.sha256(session_token.encode("utf-8")).hexdigest(),
        )

    async def finish(self, *, user: User, code: str, session_token: str) -> None:
        if not self.configured:
            raise GoogleCalendarError("Google Calendar OAuth is not configured on the gateway.")
        async with httpx.AsyncClient(timeout=8.0) as client:
            try:
                token_response = await client.post(
                    GOOGLE_TOKEN_ENDPOINT,
                    data={
                        "code": code,
                        "client_id": self._client_id,
                        "client_secret": self._client_secret,
                        "redirect_uri": self._redirect_uri,
                        "grant_type": "authorization_code",
                    },
                )
                token_response.raise_for_status()
                token_payload = token_response.json()
                access_token = token_payload.get("access_token")
                refresh_token = token_payload.get("refresh_token")
                if not isinstance(access_token, str) or not access_token:
                    raise GoogleCalendarError("Google did not return an access token.")
                identity_response = await client.get(
                    GOOGLE_USERINFO_ENDPOINT,
                    headers={"Authorization": f"Bearer {access_token}"},
                )
                identity_response.raise_for_status()
                identity = identity_response.json()
            except GoogleCalendarError:
                raise
            except (httpx.HTTPError, ValueError) as error:
                raise GoogleCalendarError("Google OAuth could not be completed. Please try again.") from error

        google_subject = identity.get("sub")
        google_email = identity.get("email")
        if identity.get("email_verified") is not True or not isinstance(google_subject, str) or not isinstance(google_email, str):
            raise GoogleCalendarError("Google did not provide a verified account identity.")
        if google_email.strip().casefold() != user.email.strip().casefold():
            raise GoogleCalendarError("Use the Google account with the same email as your Who Speak account.")
        if not isinstance(refresh_token, str) or not refresh_token:
            previous = self._store.google_calendar_connection(user.id)
            if previous is None:
                raise GoogleCalendarError("Google did not return a refresh token. Reconnect and approve offline access.")
            refresh_token = TokenCipher(self._encryption_key).decrypt(previous.encrypted_refresh_token)

        scopes = tuple(
            item for item in str(token_payload.get("scope", "")).split() if item
        ) or GOOGLE_READ_SCOPES
        if not GOOGLE_REQUIRED_CALENDAR_SCOPES.issubset(scopes):
            raise GoogleCalendarError(
                "Google Calendar read permissions are incomplete. Reconnect Google Calendar and approve all Calendar read permissions."
            )
        now = datetime.now(UTC)
        expires_in = token_payload.get("expires_in")
        expires_at = now + timedelta(seconds=int(expires_in)) if isinstance(expires_in, (int, float)) else None
        try:
            self._store.save_google_calendar_connection(
                GoogleCalendarConnection(
                    user_id=user.id,
                    google_subject=google_subject,
                    google_email=google_email.strip().lower(),
                    encrypted_refresh_token=TokenCipher(self._encryption_key).encrypt(refresh_token),
                    granted_scopes=scopes,
                    status="active",
                    token_expires_at=expires_at,
                    created_at=now,
                    updated_at=now,
                )
            )
        except ValueError:
            raise GoogleCalendarError("This Google account is already linked to another Who Speak account.") from None

    def status(self, user_id: str) -> dict[str, object]:
        connection = self._store.google_calendar_connection(user_id)
        if connection is None:
            return {"provider": "google_mcp", "configured": self.configured, "connected": False, "status": "not_connected"}
        return {
            "provider": "google_mcp",
            "configured": self.configured,
            "connected": connection.status == "active",
            "status": connection.status,
            "email": connection.google_email,
        }

    def disconnect(self, user_id: str) -> None:
        self._store.disconnect_google_calendar(user_id)

    async def access_token(self, user_id: str) -> str:
        connection = self._store.google_calendar_connection(user_id)
        if connection is None or connection.status != "active":
            raise GoogleCalendarError("Connect Google Calendar before asking about personal events.")
        if not GOOGLE_REQUIRED_CALENDAR_SCOPES.issubset(set(connection.granted_scopes)):
            self._store.mark_google_calendar_status(user_id, "needs_reconnect")
            raise GoogleCalendarError(
                "Google Calendar permissions are incomplete. Reconnect Google Calendar before asking about personal events."
            )
        refresh_token = TokenCipher(self._encryption_key).decrypt(connection.encrypted_refresh_token)
        async with httpx.AsyncClient(timeout=8.0) as client:
            try:
                response = await client.post(
                    GOOGLE_TOKEN_ENDPOINT,
                    data={
                        "client_id": self._client_id,
                        "client_secret": self._client_secret,
                        "refresh_token": refresh_token,
                        "grant_type": "refresh_token",
                    },
                )
                if response.status_code in {400, 401}:
                    self._store.mark_google_calendar_status(user_id, "needs_reconnect")
                    raise GoogleCalendarError("Google Calendar authorization expired. Reconnect it in Settings.")
                response.raise_for_status()
                payload = response.json()
            except GoogleCalendarError:
                raise
            except (httpx.HTTPError, ValueError) as error:
                raise GoogleCalendarError("Google Calendar authorization could not be refreshed.") from error
        access_token = payload.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise GoogleCalendarError("Google did not return a usable access token.")
        return access_token

    @property
    def web_app_url(self) -> str:
        return self._web_app_url


class GoogleCalendarMcpClient:
    """Short-lived MCP SDK client for one access token and one call."""

    def __init__(self, *, endpoint: str, timeout_seconds: float = 8.0) -> None:
        self._endpoint = endpoint
        self._timeout_seconds = timeout_seconds

    async def list_events(self, *, access_token: str, start: datetime, end: datetime) -> Any:
        try:
            from mcp import ClientSession
            from mcp.client.streamable_http import streamable_http_client
        except ImportError as error:
            raise GoogleCalendarError("Install the MCP Python SDK before enabling Google Calendar.") from error
        for attempt in range(2):
            try:
                logger.info(
                    "google_mcp_call_started tool=list_events attempt=%s start=%s end=%s",
                    attempt + 1,
                    start.isoformat(),
                    end.isoformat(),
                )
                async with httpx.AsyncClient(
                    timeout=self._timeout_seconds,
                    headers={"Authorization": f"Bearer {access_token}"},
                ) as http_client:
                    async with streamable_http_client(self._endpoint, http_client=http_client) as (read, write, _):
                        async with ClientSession(read, write) as session:
                            await session.initialize()
                            return await session.call_tool(
                                "list_events",
                                arguments={
                                    "calendarId": "primary",
                                    "startTime": start.isoformat(),
                                    "endTime": end.isoformat(),
                                    "pageSize": 50,
                                    "orderBy": "startTime",
                                },
                            )
            except (httpx.TimeoutException, httpx.NetworkError) as error:
                logger.warning(
                    "google_mcp_call_network_error tool=list_events attempt=%s error_type=%s",
                    attempt + 1,
                    type(error).__name__,
                )
                if attempt == 0:
                    continue
                raise GoogleCalendarError("Google Calendar MCP is temporarily unavailable.") from error
            except GoogleCalendarError:
                raise
            except Exception as error:
                response = getattr(error, "response", None)
                status_code = getattr(response, "status_code", None)
                logger.exception(
                    "google_mcp_call_failed tool=list_events attempt=%s error_type=%s status_code=%s",
                    attempt + 1,
                    type(error).__name__,
                    status_code if isinstance(status_code, int) else "none",
                )
                if status_code in {401, 403}:
                    raise GoogleCalendarError(
                        "Google Calendar authorization was rejected by MCP. Reconnect Google Calendar and approve the Calendar read permissions."
                    ) from error
                if status_code == 404:
                    raise GoogleCalendarError(
                        "Google Calendar MCP endpoint was not found. Check that calendarmcp.googleapis.com is enabled."
                    ) from error
                raise GoogleCalendarError("Google Calendar MCP is temporarily unavailable.") from error
        raise GoogleCalendarError("Google Calendar MCP is temporarily unavailable.")


class LocalGoogleCalendarMcpClient:
    """Client for the self-hosted nspady MCP server.

    The self-hosted server manages its own OAuth token store and exposes the
    MCP endpoint at the HTTP server root.  This adapter deliberately resolves
    an account by email before calling a tool so a local multi-account store
    cannot silently serve the wrong Google account to Who Speak.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        timeout_seconds: float = 8.0,
        timezone_name: str | None = None,
    ) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._timeout_seconds = timeout_seconds
        selected_timezone = (timezone_name or os.getenv("VOICE_DEFAULT_TIMEZONE", "Asia/Ho_Chi_Minh")).strip()
        try:
            self._timezone = ZoneInfo(selected_timezone)
        except ZoneInfoNotFoundError as error:
            raise ValueError(f"Unknown local Google Calendar MCP timezone: {selected_timezone}") from error
        self._timezone_name = selected_timezone

    async def account_for_email(self, email: str) -> str | None:
        try:
            async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
                response = await client.get(f"{self._endpoint}/api/accounts")
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise GoogleCalendarError("Local Google Calendar MCP server is unavailable.") from error

        accounts = payload.get("accounts") if isinstance(payload, dict) else None
        if not isinstance(accounts, list):
            raise GoogleCalendarError("Local Google Calendar MCP returned invalid account data.")
        normalized = email.strip().casefold()
        matches = [
            item for item in accounts
            if isinstance(item, dict)
            and isinstance(item.get("id"), str)
            and isinstance(item.get("email"), str)
            and item["email"].strip().casefold() == normalized
            and item.get("status") == "active"
        ]
        if len(matches) != 1:
            return None
        return matches[0]["id"]

    async def list_events(
        self,
        *,
        account: str,
        start: datetime,
        end: datetime,
    ) -> Any:
        try:
            from mcp import ClientSession
            from mcp.client.streamable_http import streamable_http_client
        except ImportError as error:
            raise GoogleCalendarError("Install the MCP Python SDK before enabling Google Calendar.") from error

        arguments = {
            "account": account,
            "calendarId": "primary",
            # The self-hosted nspady server validates this field without
            # fractional seconds.  It receives the IANA timezone separately,
            # so send a local wall-clock value rather than Agent-side UTC.
            "timeMin": self._mcp_datetime(start),
            "timeMax": self._mcp_datetime(end),
            "timeZone": self._timezone_name,
        }
        try:
            logger.info(
                "local_google_mcp_call_started tool=list-events account=%s start=%s end=%s",
                account,
                start.isoformat(),
                end.isoformat(),
            )
            async with httpx.AsyncClient(timeout=self._timeout_seconds) as http_client:
                async with streamable_http_client(self._endpoint, http_client=http_client) as (read, write, _):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        return await session.call_tool("list-events", arguments=arguments)
        except (httpx.TimeoutException, httpx.NetworkError) as error:
            raise GoogleCalendarError("Local Google Calendar MCP server timed out.") from error
        except GoogleCalendarError:
            raise
        except Exception as error:
            logger.exception(
                "local_google_mcp_call_failed tool=list-events account=%s error_type=%s",
                account,
                type(error).__name__,
            )
            raise GoogleCalendarError("Local Google Calendar MCP could not read your events.") from error

    def _mcp_datetime(self, value: datetime) -> str:
        """Format an aware instant for nspady's local-time MCP schema."""
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Calendar tool timestamps must be timezone-aware.")
        return value.astimezone(self._timezone).replace(tzinfo=None, microsecond=0).isoformat()


class LocalGoogleCalendarMcpProvider:
    """Gateway provider for a local nspady MCP server.

    VoiceAuth and the Who Speak Google connection remain required.  The local
    MCP token is used only after its account email matches that connection.
    """

    def __init__(self, *, oauth: GoogleOAuthService, client: LocalGoogleCalendarMcpClient) -> None:
        self._oauth = oauth
        self._client = client

    async def list_events(self, *, user_id: str, start: datetime, end: datetime) -> CalendarResult:
        connection = self._oauth.status(user_id)
        email = connection.get("email")
        if connection.get("status") != "active" or not isinstance(email, str):
            raise GoogleCalendarError("Connect Google Calendar before asking about personal events.")
        account = await self._client.account_for_email(email)
        if account is None:
            raise GoogleCalendarError(
                "The local Google Calendar MCP account does not match your connected Google account. "
                "Authenticate the same account with the local MCP server."
            )
        result = await self._client.list_events(account=account, start=start, end=end)
        events = _sanitize_mcp_events(result, user_id)
        return CalendarResult("google_mcp", False, user_id, tuple(events))

    async def create_event(self, *, user_id: str, title: str, start: datetime, end: datetime) -> CalendarResult:
        raise CalendarProviderError("Creating Google Calendar events is not enabled in this rollout.")


class GoogleCalendarMcpProvider:
    """Gateway provider exposing only sanitized read-only Calendar data."""

    def __init__(self, *, oauth: GoogleOAuthService, client: GoogleCalendarMcpClient) -> None:
        self._oauth = oauth
        self._client = client

    async def list_events(self, *, user_id: str, start: datetime, end: datetime) -> CalendarResult:
        access_token = await self._oauth.access_token(user_id)
        result = await self._client.list_events(access_token=access_token, start=start, end=end)
        events = _sanitize_mcp_events(result, user_id)
        return CalendarResult("google_mcp", False, user_id, tuple(events))

    async def create_event(self, *, user_id: str, title: str, start: datetime, end: datetime) -> CalendarResult:
        raise CalendarProviderError("Creating Google Calendar events is not enabled in this rollout.")


def _structured_result(result: Any) -> dict[str, Any]:
    structured = getattr(result, "structured_content", None)
    if isinstance(structured, dict):
        return structured
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        return structured
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            try:
                parsed = __import__("json").loads(text)
            except ValueError:
                continue
            if isinstance(parsed, dict):
                return parsed
    raise GoogleCalendarError("Google Calendar MCP returned an invalid response.")


def _sanitize_mcp_events(result: Any, user_id: str) -> list[CalendarEvent]:
    if getattr(result, "is_error", False) or getattr(result, "isError", False):
        raise GoogleCalendarError("Google Calendar could not read your events.")
    payload = _structured_result(result)
    raw_events = payload.get("events", [])
    if not isinstance(raw_events, list):
        raise GoogleCalendarError("Google Calendar MCP returned invalid events.")
    sanitized: list[CalendarEvent] = []
    for raw in raw_events[:50]:
        if not isinstance(raw, dict):
            continue
        start_raw = raw.get("start") or {}
        end_raw = raw.get("end") or {}
        if not isinstance(start_raw, dict) or not isinstance(end_raw, dict):
            continue
        start_value = start_raw.get("dateTime") or start_raw.get("date")
        end_value = end_raw.get("dateTime") or end_raw.get("date")
        if not isinstance(start_value, str) or not isinstance(end_value, str):
            continue
        try:
            event_start = _parse_calendar_datetime(start_value)
            event_end = _parse_calendar_datetime(end_value)
        except ValueError:
            continue
        if event_end <= event_start:
            continue
        event_id = str(raw.get("id", ""))[:200]
        title = str(raw.get("summary", ""))[:200]
        if not event_id or not title:
            continue
        location = str(raw.get("location", ""))[:200] or None
        sanitized.append(CalendarEvent(event_id, user_id, title, event_start, event_end, location))
    return sanitized
