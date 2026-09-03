"""Signed, non-sensitive hand-off from the local gateway to Pipecat."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import asdict, dataclass


class SessionDescriptorError(ValueError):
    """Raised when a Pipecat session hand-off cannot be trusted."""


@dataclass(frozen=True)
class PipecatSessionDescriptor:
    """The minimum account/room context needed by one local Pipecat worker.

    This descriptor deliberately contains no audio, embedding, matcher token,
    HE material, OpenAI key, or model path. It is authenticated with an HMAC
    shared only by the gateway and the local supervisor.
    """

    session_id: str
    room_name: str
    participant_identity: str
    user_id: str
    voice_identity_id: str
    display_name: str
    issued_at: int


def _require_secret(secret: str) -> bytes:
    value = secret.encode("utf-8")
    if len(value) < 32:
        raise SessionDescriptorError("PIPECAT_SUPERVISOR_SECRET must be at least 32 characters.")
    return value


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def sign_session_descriptor(descriptor: PipecatSessionDescriptor, secret: str) -> str:
    """Return an opaque ``payload.signature`` hand-off token."""

    key = _require_secret(secret)
    payload = json.dumps(asdict(descriptor), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    signature = hmac.new(key, payload, hashlib.sha256).digest()
    return f"{_encode(payload)}.{_encode(signature)}"


def verify_session_descriptor(
    token: str,
    secret: str,
    *,
    max_age_seconds: int = 120,
    now: int | None = None,
) -> PipecatSessionDescriptor:
    """Verify signature, shape, and freshness before starting a worker."""

    key = _require_secret(secret)
    try:
        encoded_payload, encoded_signature = token.split(".", 1)
        payload = _decode(encoded_payload)
        supplied_signature = _decode(encoded_signature)
    except (ValueError, UnicodeError, base64.binascii.Error) as error:
        raise SessionDescriptorError("The Pipecat session descriptor is malformed.") from error

    expected_signature = hmac.new(key, payload, hashlib.sha256).digest()
    if not hmac.compare_digest(supplied_signature, expected_signature):
        raise SessionDescriptorError("The Pipecat session descriptor signature is invalid.")
    try:
        values = json.loads(payload)
        descriptor = PipecatSessionDescriptor(**values)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise SessionDescriptorError("The Pipecat session descriptor payload is invalid.") from error

    required = (
        descriptor.session_id,
        descriptor.room_name,
        descriptor.participant_identity,
        descriptor.user_id,
        descriptor.voice_identity_id,
        descriptor.display_name,
    )
    if not all(isinstance(value, str) and value.strip() for value in required):
        raise SessionDescriptorError("The Pipecat session descriptor is incomplete.")
    current = int(time.time()) if now is None else now
    if descriptor.issued_at > current + 30 or current - descriptor.issued_at > max_age_seconds:
        raise SessionDescriptorError("The Pipecat session descriptor has expired.")
    return descriptor
