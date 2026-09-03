"""Local, gateway-managed Pipecat session supervisor."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from app.assistant.pipecat_runtime.runtime import run_pipecat_session
from app.assistant.pipecat_runtime.session import (
    PipecatSessionDescriptor,
    SessionDescriptorError,
    verify_session_descriptor,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipecatSupervisorSettings:
    livekit_url: str
    livekit_api_key: str
    livekit_api_secret: str
    descriptor_secret: str
    host: str = "127.0.0.1"
    port: int = 8021

    @classmethod
    def from_environment(cls) -> "PipecatSupervisorSettings":
        secret = os.getenv("PIPECAT_SUPERVISOR_SECRET", "").strip()
        if len(secret) < 32:
            raise ValueError("PIPECAT_SUPERVISOR_SECRET must be at least 32 characters.")
        key = os.getenv("LIVEKIT_API_KEY", "").strip()
        api_secret = os.getenv("LIVEKIT_API_SECRET", "").strip()
        if not key or not api_secret:
            raise ValueError("LIVEKIT_API_KEY and LIVEKIT_API_SECRET are required by Pipecat supervisor.")
        return cls(
            livekit_url=os.getenv("LIVEKIT_URL", "ws://127.0.0.1:7880").rstrip("/"),
            livekit_api_key=key,
            livekit_api_secret=api_secret,
            descriptor_secret=secret,
            host=os.getenv("PIPECAT_SUPERVISOR_HOST", "127.0.0.1"),
            port=int(os.getenv("PIPECAT_SUPERVISOR_PORT", "8021")),
        )


class StartSessionRequest(BaseModel):
    descriptor: str = Field(min_length=32, max_length=4096)


class PipecatSessionSupervisor:
    def __init__(self, settings: PipecatSupervisorSettings) -> None:
        self._settings = settings
        self._tasks: dict[str, asyncio.Task] = {}

    @property
    def active_sessions(self) -> int:
        return len(self._tasks)

    async def start(self, signed_descriptor: str) -> PipecatSessionDescriptor:
        try:
            descriptor = verify_session_descriptor(
                signed_descriptor, self._settings.descriptor_secret
            )
        except SessionDescriptorError as error:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(error)) from error
        if descriptor.session_id in self._tasks:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Pipecat session already exists.")

        try:
            from livekit import api

            token = (
                api.AccessToken(self._settings.livekit_api_key, self._settings.livekit_api_secret)
                .with_identity(f"pipecat-agent-{descriptor.session_id[:8]}")
                .with_name("Who Speak AI Agent")
                .with_grants(
                    api.VideoGrants(
                        room_join=True,
                        room=descriptor.room_name,
                        can_publish=True,
                        can_subscribe=True,
                        can_publish_data=True,
                    )
                )
                .to_jwt()
            )
        except ImportError as error:
            raise HTTPException(status_code=503, detail="LiveKit Python dependencies are unavailable.") from error

        task = asyncio.create_task(
            run_pipecat_session(descriptor, token),
            name=f"pipecat-session-{descriptor.session_id}",
        )
        self._tasks[descriptor.session_id] = task

        def finished(done: asyncio.Task) -> None:
            self._tasks.pop(descriptor.session_id, None)
            if done.cancelled():
                return
            error = done.exception()
            if error:
                logger.error(
                    "pipecat_session_failed session_id=%s: %s",
                    descriptor.session_id,
                    error,
                    exc_info=(type(error), error, error.__traceback__),
                )
            else:
                logger.warning(
                    "pipecat_session_finished_without_error session_id=%s",
                    descriptor.session_id,
                )

        task.add_done_callback(finished)
        logger.info(
            "pipecat_session_starting session_id=%s room=%s",
            descriptor.session_id,
            descriptor.room_name,
        )
        return descriptor

    async def stop(self, session_id: str) -> bool:
        task = self._tasks.pop(session_id, None)
        if task is None:
            return False
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        return True

    async def close(self) -> None:
        tasks = list(self._tasks)
        for session_id in tasks:
            await self.stop(session_id)


def create_supervisor_app(settings: PipecatSupervisorSettings | None = None) -> FastAPI:
    active = settings or PipecatSupervisorSettings.from_environment()
    supervisor = PipecatSessionSupervisor(active)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        await supervisor.close()

    app = FastAPI(title="Who Speak AI Pipecat Supervisor", version="0.1.0", lifespan=lifespan)
    app.state.supervisor = supervisor

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {"status": "ok", "runtime": "pipecat", "sessions": supervisor.active_sessions}

    @app.post("/internal/pipecat/sessions", status_code=status.HTTP_202_ACCEPTED)
    async def start_session(payload: StartSessionRequest) -> dict[str, str]:
        if os.getenv("VOICE_AGENT_CONVERSATION_ENABLED", "false").strip().lower() not in {"1", "true", "yes", "on"}:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Set VOICE_AGENT_CONVERSATION_ENABLED=true before starting Pipecat.",
            )
        if not os.getenv("OPENAI_API_KEY", "").strip():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="OPENAI_API_KEY is required before starting the conversational Pipecat runtime.",
            )
        provider = os.getenv("MCP_PROVIDER", "mock").strip().lower()
        if provider not in {"mock", "mcp", "local"}:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="MCP_PROVIDER must be either mock, mcp, or local.",
            )
        if provider == "mock" and os.getenv("MOCK_CALENDAR_ENABLED", "true").strip().lower() not in {"1", "true", "yes", "on"}:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Mock calendar is disabled.",
            )
        if provider in {"mcp", "local"} and len(os.getenv("PIPECAT_SUPERVISOR_SECRET", "").strip()) < 32:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="PIPECAT_SUPERVISOR_SECRET is required for MCP tool calls.",
            )
        descriptor = await supervisor.start(payload.descriptor)
        return {"session_id": descriptor.session_id, "status": "starting"}

    @app.delete("/internal/pipecat/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def stop_session(session_id: str) -> None:
        await supervisor.stop(session_id)

    return app


# Keep importing the module safe for tests and tooling that do not load the
# user's environment. The executable entry point below still fails closed with
# a precise configuration error, and a real uvicorn deployment must provide a
# valid environment to expose ``app``.
try:
    app: FastAPI | None = create_supervisor_app()
except ValueError:
    app = None


def main() -> None:
    import uvicorn

    active = PipecatSupervisorSettings.from_environment()
    uvicorn.run(create_supervisor_app(active), host=active.host, port=active.port, reload=False)


if __name__ == "__main__":
    main()
