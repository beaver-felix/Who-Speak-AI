"""macOS Keychain persistence for the private HE context.

The private context is deliberately never written to SQLite or to a fallback
file.  ``keyring`` delegates to macOS Keychain on the supported local setup.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from uuid import UUID, uuid4

from voiceauth.errors import HEContextError
from voiceauth.he import VoiceHEClient, VoiceHEContext, decode_bytes


@dataclass(frozen=True)
class KeychainContextRecord:
    context_id: UUID
    private_context_b64: str
    # Persist the exact public serialization used by the matcher. Re-exporting
    # it from TenSEAL in separate processes is not a safe session contract.
    public_context_b64: str | None = None


class KeychainHEContextStore:
    """Persist exactly one local-private context under an explicit account."""

    def __init__(self, *, service_name: str, account_name: str, backend=None) -> None:
        if not service_name.strip() or not account_name.strip():
            raise ValueError("Keychain service and account names are required.")
        self._service_name = service_name
        self._account_name = account_name
        self._backend = backend

    def _keyring(self):
        if self._backend is not None:
            return self._backend
        try:
            import keyring
        except ImportError as error:
            raise HEContextError("macOS Keychain support is required; install the [agent] extra.") from error
        return keyring

    def load(self) -> KeychainContextRecord | None:
        try:
            serialized = self._keyring().get_password(self._service_name, self._account_name)
        except Exception as error:
            raise HEContextError("macOS Keychain is unavailable; refusing to load a private HE context.") from error
        if serialized is None:
            return None
        try:
            payload = json.loads(serialized)
            context_id = UUID(payload["context_id"])
            private_context_b64 = payload["private_context_b64"]
            if not isinstance(private_context_b64, str) or not private_context_b64:
                raise ValueError("missing private context")
            decode_bytes(private_context_b64)
            public_context_b64 = payload.get("public_context_b64")
            if public_context_b64 is not None:
                if not isinstance(public_context_b64, str) or not public_context_b64:
                    raise ValueError("invalid public context")
                decode_bytes(public_context_b64)
        except (KeyError, TypeError, ValueError, HEContextError) as error:
            raise HEContextError("The macOS Keychain HE context is invalid; re-enrollment is required.") from error
        return KeychainContextRecord(context_id, private_context_b64, public_context_b64)

    def save(self, record: KeychainContextRecord) -> None:
        payload = json.dumps(
            {
                "context_id": str(record.context_id),
                "private_context_b64": record.private_context_b64,
                "public_context_b64": record.public_context_b64,
            },
            separators=(",", ":"),
        )
        try:
            self._keyring().set_password(self._service_name, self._account_name, payload)
        except Exception as error:
            raise HEContextError("macOS Keychain is unavailable; refusing to persist a private HE context.") from error

    def load_or_create(self) -> VoiceHEContext:
        record = self.load()
        if record is None:
            client = VoiceHEClient.create()
            private_context_b64 = client.export_private_context_b64()
            public_context_b64 = client.export_public_context_b64()
            record = KeychainContextRecord(uuid4(), private_context_b64, public_context_b64)
            self.save(record)
        else:
            client = VoiceHEClient.from_private_context_b64(record.private_context_b64)
            if record.public_context_b64 is None:
                # One-time migration for records created before the public
                # serialization was persisted. Future processes reuse the
                # exact same bytes instead of re-exporting them.
                record = KeychainContextRecord(
                    record.context_id,
                    record.private_context_b64,
                    client.export_public_context_b64(),
                )
                self.save(record)

        # ``client`` is created in either branch above; keep the private client
        # for local encryption/decryption while sharing the persisted public
        # bytes with the matcher.
        public_context_b64 = record.public_context_b64
        assert public_context_b64 is not None
        import hashlib

        return VoiceHEContext(
            context_id=record.context_id,
            public_context_b64=public_context_b64,
            public_context_sha256=hashlib.sha256(decode_bytes(public_context_b64)).hexdigest(),
            client=client,
        )
