"""Client for the ciphertext-only voice matcher API."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from voiceauth.config import EMBEDDING_DIMENSION, HE_PROFILE, MODEL_PROFILE, PREPROCESSING_PROFILE
from voiceauth.errors import MatcherError
from voiceauth.he import VoiceHEContext


@dataclass(frozen=True)
class EncryptedCandidate:
    identity_id: str
    display_name: str
    encrypted_distance_b64: str


class MatcherClient:
    def __init__(self, base_url: str, token: str, timeout_seconds: float = 60.0) -> None:
        try:
            import httpx
        except ImportError as error:
            raise MatcherError("Install this app with the [matcher] extra.") from error
        self.base_url = base_url.rstrip("/")
        self.token = token
        self._client = httpx.Client(timeout=timeout_seconds, http2=True)

    def close(self) -> None:
        self._client.close()

    def health(self) -> bool:
        return self._request("GET", "/health", authenticated=False).json().get("status") == "ok"

    def register_context(self, context: VoiceHEContext) -> None:
        response = self._request("GET", f"/v1/contexts/{context.context_id}", allow_status={404})
        if response.status_code != 404:
            if response.json().get("public_context_sha256") != context.public_context_sha256:
                raise MatcherError("The matcher has a different context for this session. Reset the session.")
            return
        self._request(
            "POST",
            "/v1/contexts",
            json={
                "context_id": str(context.context_id),
                "public_context_b64": context.public_context_b64,
                "public_context_sha256": context.public_context_sha256,
                "model_profile": MODEL_PROFILE,
                "preprocessing_profile": PREPROCESSING_PROFILE,
                "embedding_dimension": EMBEDDING_DIMENSION,
                "he_profile": HE_PROFILE,
            },
        )

    def enroll_identity(self, context_id: UUID, identity_id: UUID, display_name: str, encrypted_template_b64: str) -> None:
        self._request(
            "POST",
            "/v1/identities",
            json={
                "identity_id": str(identity_id),
                "context_id": str(context_id),
                "display_name": display_name,
                "encrypted_template_b64": encrypted_template_b64,
            },
        )

    def identity_count(self, context_id: UUID) -> int:
        response = self._request("GET", f"/v1/contexts/{context_id}/identities/count")
        try:
            return int(response.json()["identity_count"])
        except (KeyError, TypeError, ValueError) as error:
            raise MatcherError("The matcher returned an invalid identity-count response.") from error

    def match(self, context_id: UUID, encrypted_query_b64: str, target_identity_id: UUID | None = None) -> list[EncryptedCandidate]:
        response = self._request(
            "POST",
            "/v1/match",
            json={
                "context_id": str(context_id),
                "encrypted_query_b64": encrypted_query_b64,
                "target_identity_id": str(target_identity_id) if target_identity_id else None,
            },
        )
        try:
            return [
                EncryptedCandidate(
                    identity_id=str(item["identity_id"]),
                    display_name=str(item["display_name"]),
                    encrypted_distance_b64=str(item["encrypted_distance_b64"]),
                )
                for item in response.json()["comparisons"]
            ]
        except (KeyError, TypeError, ValueError) as error:
            raise MatcherError("The matcher returned an invalid comparison response.") from error

    def delete_context(self, context_id: UUID) -> None:
        self._request("DELETE", f"/v1/contexts/{context_id}", allow_status={404})

    def _request(self, method: str, path: str, *, authenticated: bool = True, json: dict | None = None, allow_status: set[int] | None = None):
        try:
            response = self._client.request(
                method,
                f"{self.base_url}{path}",
                headers={"X-Voice-Matcher-Token": self.token} if authenticated else {},
                json=json,
            )
        except Exception as error:
            raise MatcherError("The voice matcher is unavailable.") from error
        if response.is_success or response.status_code in (allow_status or set()):
            return response
        try:
            message = str(response.json()["error"]["message"])
        except (KeyError, TypeError, ValueError):
            message = f"Voice matcher request failed with HTTP {response.status_code}."
        raise MatcherError(message)
