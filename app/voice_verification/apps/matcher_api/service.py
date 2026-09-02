"""Ciphertext-only matcher orchestration."""

from __future__ import annotations

import time

from fastapi import HTTPException, status

from apps.matcher_api.schemas import ContextCreate, ContextView, EncryptedComparison, IdentityCount, IdentityCreate, IdentityView, MatchRequest, MatchResponse
from apps.matcher_api.store import MatcherStore
from voiceauth.errors import HEContextError
from voiceauth.he import VoiceHEMatcher


class MatcherService:
    def __init__(self, store: MatcherStore, *, max_identities: int = 100) -> None:
        self.store = store
        self.max_identities = max_identities

    def context(self, context_id: str) -> ContextView:
        stored = self.store.get_context(context_id)
        if stored is None:
            raise HTTPException(status_code=404, detail=self._error("CONTEXT_NOT_FOUND", "The HE context was not found."))
        return ContextView(context_id=stored.context_id, public_context_sha256=stored.public_context_sha256, created_at=stored.created_at)

    def create_context(self, payload: ContextCreate) -> ContextView:
        try:
            VoiceHEMatcher.validate_public_context(payload.public_context_b64)
            actual_sha256 = VoiceHEMatcher.public_context_sha256(payload.public_context_b64)
        except HEContextError as error:
            raise HTTPException(status_code=422, detail=self._error("INVALID_CONTEXT", str(error))) from error
        if actual_sha256 != payload.public_context_sha256:
            raise HTTPException(status_code=422, detail=self._error("CONTEXT_HASH_MISMATCH", "The public context hash does not match its bytes."))
        existing = self.store.get_context(str(payload.context_id))
        if existing is not None:
            if existing.public_context_sha256 != payload.public_context_sha256:
                raise HTTPException(status_code=409, detail=self._error("CONTEXT_CONFLICT", "This context ID has a different public key."))
            return ContextView(context_id=existing.context_id, public_context_sha256=existing.public_context_sha256, created_at=existing.created_at)
        stored = self.store.create_context(
            context_id=str(payload.context_id), public_context_b64=payload.public_context_b64,
            public_context_sha256=payload.public_context_sha256, model_profile=payload.model_profile,
            preprocessing_profile=payload.preprocessing_profile, embedding_dimension=payload.embedding_dimension,
            he_profile=payload.he_profile,
        )
        return ContextView(context_id=stored.context_id, public_context_sha256=stored.public_context_sha256, created_at=stored.created_at)

    def create_identity(self, payload: IdentityCreate) -> IdentityView:
        context = self.store.get_context(str(payload.context_id))
        if context is None:
            raise HTTPException(status_code=404, detail=self._error("CONTEXT_NOT_FOUND", "The HE context was not found."))
        if not payload.display_name.strip():
            raise HTTPException(status_code=422, detail=self._error("INVALID_DISPLAY_NAME", "Display name cannot be blank."))
        if len(self.store.list_identities(str(payload.context_id))) >= self.max_identities:
            raise HTTPException(status_code=409, detail=self._error("IDENTITY_LIMIT", "This context reached its identity limit."))
        try:
            VoiceHEMatcher.validate_encrypted_vector(
                context.public_context_b64, payload.encrypted_template_b64,
            )
        except HEContextError as error:
            raise HTTPException(status_code=422, detail=self._error("INVALID_CIPHERTEXT", str(error))) from error
        try:
            stored = self.store.create_identity(
                identity_id=str(payload.identity_id), context_id=str(payload.context_id),
                display_name=payload.display_name.strip(), encrypted_template_b64=payload.encrypted_template_b64,
            )
        except Exception as error:
            raise HTTPException(status_code=409, detail=self._error("IDENTITY_CONFLICT", "The identity could not be stored.")) from error
        return IdentityView(identity_id=stored.identity_id, display_name=stored.display_name, created_at=stored.created_at)

    def identity_count(self, context_id: str) -> IdentityCount:
        if self.store.get_context(context_id) is None:
            raise HTTPException(status_code=404, detail=self._error("CONTEXT_NOT_FOUND", "The HE context was not found."))
        return IdentityCount(context_id=context_id, identity_count=self.store.count_identities(context_id))

    def match(self, payload: MatchRequest) -> MatchResponse:
        context = self.store.get_context(str(payload.context_id))
        if context is None:
            raise HTTPException(status_code=404, detail=self._error("CONTEXT_NOT_FOUND", "The HE context was not found."))
        identities = self.store.list_identities(str(payload.context_id), str(payload.target_identity_id) if payload.target_identity_id else None)
        started = time.perf_counter()
        try:
            VoiceHEMatcher.validate_encrypted_vector(
                context.public_context_b64, payload.encrypted_query_b64,
            )
            encrypted_distances = VoiceHEMatcher.compare_many(
                context.public_context_b64, payload.encrypted_query_b64,
                [identity.encrypted_template_b64 for identity in identities],
            )
        except HEContextError as error:
            raise HTTPException(status_code=422, detail=self._error("INVALID_CIPHERTEXT", str(error))) from error
        comparisons = [
            EncryptedComparison(identity_id=identity.identity_id, display_name=identity.display_name, encrypted_distance_b64=distance)
            for identity, distance in zip(identities, encrypted_distances, strict=True)
        ]
        return MatchResponse(context_id=payload.context_id, candidate_count=len(comparisons), comparisons=comparisons, server_compute_ms=(time.perf_counter() - started) * 1000)

    def delete_context(self, context_id: str) -> None:
        self.store.delete_context(context_id)

    @staticmethod
    def _error(code: str, message: str) -> dict[str, str]:
        return {"code": code, "message": message}
