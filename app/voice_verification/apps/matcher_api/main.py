"""FastAPI application for the public-context-only voice matcher."""

from __future__ import annotations

import os
from dataclasses import dataclass
from fastapi import Depends, FastAPI, Header, HTTPException, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from apps.matcher_api.schemas import ContextCreate, ContextView, HealthResponse, IdentityCount, IdentityCreate, IdentityView, MatchRequest, MatchResponse
from apps.matcher_api.service import MatcherService
from apps.matcher_api.store import MatcherStore


@dataclass(frozen=True)
class MatcherSettings:
    database_path: str
    token: str
    max_identities: int = 100

    @classmethod
    def from_environment(cls) -> "MatcherSettings":
        token = os.getenv("VOICE_MATCHER_TOKEN", "")
        if len(token) < 8:
            raise RuntimeError("VOICE_MATCHER_TOKEN must contain at least 8 characters.")
        return cls(
            database_path=os.getenv("VOICE_MATCHER_DATABASE", "./data/voice_matcher.db"),
            token=token,
            max_identities=int(os.getenv("VOICE_MAX_IDENTITIES", "100")),
        )


def create_app(settings: MatcherSettings | None = None) -> FastAPI:
    active = settings or MatcherSettings.from_environment()
    service = MatcherService(MatcherStore(active.database_path), max_identities=active.max_identities)
    app = FastAPI(title="Who Speak Voice HE Matcher", version="0.1.0")

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request, _error):
        return JSONResponse(status_code=422, content={"error": {"code": "INVALID_REQUEST", "message": "The request did not match the matcher contract."}})

    @app.exception_handler(HTTPException)
    async def http_error(_request, error: HTTPException):
        if isinstance(error.detail, dict) and {"code", "message"} <= error.detail.keys():
            payload = error.detail
        else:
            payload = {"code": "REQUEST_FAILED", "message": "The matcher request could not be completed."}
        return JSONResponse(status_code=error.status_code, content={"error": payload}, headers=error.headers)

    def require_token(x_voice_matcher_token: str | None = Header(default=None)) -> None:
        if x_voice_matcher_token != active.token:
            raise HTTPException(status_code=401, detail={"code": "AUTHENTICATION_FAILED", "message": "A valid matcher token is required."})

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse()

    @app.post("/v1/contexts", response_model=ContextView, dependencies=[Depends(require_token)])
    def create_context(payload: ContextCreate) -> ContextView:
        return service.create_context(payload)

    @app.get("/v1/contexts/{context_id}", response_model=ContextView, dependencies=[Depends(require_token)])
    def get_context(context_id: str) -> ContextView:
        return service.context(context_id)

    @app.post("/v1/identities", response_model=IdentityView, status_code=status.HTTP_201_CREATED, dependencies=[Depends(require_token)])
    def create_identity(payload: IdentityCreate) -> IdentityView:
        return service.create_identity(payload)

    @app.get("/v1/contexts/{context_id}/identities/count", response_model=IdentityCount, dependencies=[Depends(require_token)])
    def identity_count(context_id: str) -> IdentityCount:
        return service.identity_count(context_id)

    @app.post("/v1/match", response_model=MatchResponse, dependencies=[Depends(require_token)])
    def match(payload: MatchRequest) -> MatchResponse:
        return service.match(payload)

    @app.delete("/v1/contexts/{context_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(require_token)])
    def delete_context(context_id: str) -> Response:
        service.delete_context(context_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return app


app = create_app()
