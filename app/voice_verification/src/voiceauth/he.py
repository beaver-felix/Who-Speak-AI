"""CKKS operations for 256-D RawNet3 embeddings.

The matcher receives only serialized public context and ciphertext. All
decryption and cosine threshold decisions remain in the trusted client.
"""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

import numpy as np

from voiceauth.config import EMBEDDING_DIMENSION, HE_PROFILE
from voiceauth.errors import HEContextError
from voiceauth.matching import normalize_embedding


def encode_bytes(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def decode_bytes(value: str) -> bytes:
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except (ValueError, UnicodeEncodeError) as error:
        raise HEContextError("Encrypted data is not valid Base64.") from error


def _tenseal() -> Any:
    try:
        import tenseal as ts
    except ImportError as error:
        raise HEContextError("TenSEAL is required. Install this app with the [he] extra.") from error
    return ts


@dataclass(frozen=True)
class VoiceHEContext:
    context_id: UUID
    public_context_b64: str
    public_context_sha256: str
    client: "VoiceHEClient"


class VoiceHEClient:
    def __init__(self, context: Any) -> None:
        self._context = context

    @classmethod
    def create(cls) -> "VoiceHEClient":
        ts = _tenseal()
        try:
            context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=8192,
                coeff_mod_bit_sizes=[60, 40, 40, 60],
            )
            context.generate_galois_keys()
            context.global_scale = 2**40
            return cls(context)
        except Exception as error:
            raise HEContextError("Could not create the local HE context.") from error

    @classmethod
    def from_private_context_b64(cls, value: str) -> "VoiceHEClient":
        ts = _tenseal()
        try:
            context = ts.context_from(decode_bytes(value))
        except Exception as error:
            raise HEContextError("Could not load the private HE context.") from error
        if not context.has_secret_key():
            raise HEContextError("The private HE context has no secret key.")
        return cls(context)

    @classmethod
    def from_public_context_b64(cls, value: str) -> "VoiceHEClient":
        ts = _tenseal()
        try:
            context = ts.context_from(decode_bytes(value))
        except Exception as error:
            raise HEContextError("Could not load the public HE context.") from error
        if context.has_secret_key():
            raise HEContextError("The matcher accepts public HE contexts only.")
        return cls(context)

    def export_private_context_b64(self) -> str:
        if not self._context.has_secret_key():
            raise HEContextError("Cannot export a private context without a secret key.")
        return encode_bytes(self._context.serialize(save_secret_key=True))

    def export_public_context_b64(self) -> str:
        return encode_bytes(
            self._context.serialize(
                save_public_key=True,
                save_secret_key=False,
                save_galois_keys=True,
                save_relin_keys=True,
            )
        )

    def encrypt_embedding(self, embedding: np.ndarray) -> str:
        ts = _tenseal()
        vector = normalize_embedding(embedding)
        try:
            return encode_bytes(ts.ckks_vector(self._context, vector.tolist()).serialize())
        except Exception as error:
            raise HEContextError("Could not encrypt the voice embedding.") from error

    def decrypt_distance(self, encrypted_distance_b64: str) -> float:
        if not self._context.has_secret_key():
            raise HEContextError("A public HE context cannot decrypt a distance.")
        ts = _tenseal()
        try:
            encrypted = ts.ckks_vector_from(self._context, decode_bytes(encrypted_distance_b64))
            return float(encrypted.decrypt()[0])
        except Exception as error:
            raise HEContextError("Could not decrypt the matcher distance.") from error


class VoiceHEMatcher:
    """Ciphertext-only squared Euclidean matcher used by the API service."""

    @staticmethod
    def public_context_sha256(public_context_b64: str) -> str:
        return hashlib.sha256(decode_bytes(public_context_b64)).hexdigest()

    @staticmethod
    def validate_public_context(public_context_b64: str) -> None:
        VoiceHEClient.from_public_context_b64(public_context_b64)

    @staticmethod
    def validate_encrypted_vector(public_context_b64: str, encrypted_vector_b64: str) -> None:
        """Validate ciphertext at the boundary without decrypting it."""
        client = VoiceHEClient.from_public_context_b64(public_context_b64)
        ts = _tenseal()
        try:
            vector = ts.ckks_vector_from(client._context, decode_bytes(encrypted_vector_b64))
            if vector.size() != EMBEDDING_DIMENSION:
                raise HEContextError(
                    f"Encrypted embeddings must contain {EMBEDDING_DIMENSION} values."
                )
        except HEContextError:
            raise
        except Exception as error:
            raise HEContextError("The encrypted embedding is invalid for this context.") from error

    @staticmethod
    def compare_many(
        public_context_b64: str, query_b64: str, template_b64s: list[str]
    ) -> list[str]:
        client = VoiceHEClient.from_public_context_b64(public_context_b64)
        ts = _tenseal()
        try:
            query = ts.ckks_vector_from(client._context, decode_bytes(query_b64))
            if query.size() != EMBEDDING_DIMENSION:
                raise HEContextError("The encrypted query has an invalid dimension.")
            results: list[str] = []
            for template_b64 in template_b64s:
                template = ts.ckks_vector_from(client._context, decode_bytes(template_b64))
                if template.size() != EMBEDDING_DIMENSION:
                    raise HEContextError("An encrypted template has an invalid dimension.")
                difference = query - template
                results.append(encode_bytes(difference.dot(difference).serialize()))
            return results
        except HEContextError:
            raise
        except Exception as error:
            raise HEContextError("The encrypted embeddings could not be compared.") from error


def create_session_context() -> VoiceHEContext:
    """Create one memory-only private context for a trusted Streamlit session."""
    client = VoiceHEClient.create()
    public = client.export_public_context_b64()
    return VoiceHEContext(
        context_id=uuid4(),
        public_context_b64=public,
        public_context_sha256=hashlib.sha256(decode_bytes(public)).hexdigest(),
        client=client,
    )


__all__ = ["HE_PROFILE", "VoiceHEClient", "VoiceHEContext", "VoiceHEMatcher", "create_session_context"]
