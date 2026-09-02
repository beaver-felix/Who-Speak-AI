"""Strict API contracts for the voice matcher."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from voiceauth.config import EMBEDDING_DIMENSION, HE_PROFILE, MODEL_PROFILE, PREPROCESSING_PROFILE


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ContextCreate(APIModel):
    context_id: UUID
    public_context_b64: str = Field(min_length=4, max_length=60_000_000)
    public_context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_profile: Literal[MODEL_PROFILE]
    preprocessing_profile: Literal[PREPROCESSING_PROFILE]
    embedding_dimension: Literal[EMBEDDING_DIMENSION]
    he_profile: Literal[HE_PROFILE]


class ContextView(APIModel):
    context_id: UUID
    public_context_sha256: str
    created_at: datetime


class IdentityCreate(APIModel):
    identity_id: UUID
    context_id: UUID
    display_name: str = Field(min_length=1, max_length=120)
    encrypted_template_b64: str = Field(min_length=4, max_length=2_000_000)


class IdentityView(APIModel):
    identity_id: UUID
    display_name: str
    created_at: datetime


class IdentityCount(APIModel):
    context_id: UUID
    identity_count: int


class MatchRequest(APIModel):
    context_id: UUID
    encrypted_query_b64: str = Field(min_length=4, max_length=2_000_000)
    target_identity_id: UUID | None = None


class EncryptedComparison(APIModel):
    identity_id: UUID
    display_name: str
    encrypted_distance_b64: str


class MatchResponse(APIModel):
    context_id: UUID
    candidate_count: int
    comparisons: list[EncryptedComparison]
    server_compute_ms: float


class HealthResponse(APIModel):
    status: Literal["ok"] = "ok"
    service: Literal["voice-he-matcher"] = "voice-he-matcher"
