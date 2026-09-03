"""Local password and opaque-session primitives.

These values never leave the FastAPI process except for the opaque session
token in an HttpOnly cookie. SQLite stores only derived values.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets


def hash_password(password: str) -> str:
    if len(password) < 10:
        raise ValueError("Password must contain at least 10 characters.")
    salt = secrets.token_bytes(16)
    derived = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1)
    return "scrypt$16384$8$1${}${}".format(
        base64.b64encode(salt).decode("ascii"),
        base64.b64encode(derived).decode("ascii"),
    )


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, n, r, p, salt_b64, digest_b64 = encoded.split("$", 5)
        if algorithm != "scrypt":
            return False
        salt = base64.b64decode(salt_b64, validate=True)
        expected = base64.b64decode(digest_b64, validate=True)
        actual = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=int(n), r=int(r), p=int(p))
    except (TypeError, ValueError):
        return False
    return hmac.compare_digest(actual, expected)


def new_session_token() -> str:
    return secrets.token_urlsafe(32)


def digest_session_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def sign_internal_request(payload: dict[str, object], secret: str) -> str:
    """Sign an Agent-to-gateway request without putting a user id in trust."""
    key = secret.encode("utf-8")
    if len(key) < 32:
        raise ValueError("PIPECAT_SUPERVISOR_SECRET must be at least 32 characters.")
    message = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hmac.new(key, message, hashlib.sha256).hexdigest()


def verify_internal_request(payload: dict[str, object], supplied_signature: str, secret: str) -> bool:
    try:
        expected = sign_internal_request(payload, secret)
    except ValueError:
        return False
    return hmac.compare_digest(expected, supplied_signature)
