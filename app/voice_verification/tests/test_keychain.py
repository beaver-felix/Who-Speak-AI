from __future__ import annotations

import pytest
from uuid import uuid4

from voiceauth.errors import HEContextError
from voiceauth.keychain import KeychainContextRecord, KeychainHEContextStore


class FakeKeyring:
    def __init__(self) -> None:
        self.value = None

    def get_password(self, service_name, account_name):
        return self.value

    def set_password(self, service_name, account_name, value):
        self.value = value


def test_keychain_store_round_trips_context_metadata_without_file_fallback() -> None:
    keyring = FakeKeyring()
    store = KeychainHEContextStore(service_name="who-speak", account_name="local-owner", backend=keyring)
    record = KeychainContextRecord(uuid4(), "c2VjcmV0", "cHVibGlj")

    store.save(record)

    assert store.load() == record
    assert "private_context_b64" in keyring.value
    assert "public_context_b64" in keyring.value


def test_keychain_unavailable_fails_closed() -> None:
    class BrokenKeyring:
        def get_password(self, *_args):
            raise RuntimeError("locked")

    store = KeychainHEContextStore(service_name="who-speak", account_name="local-owner", backend=BrokenKeyring())

    with pytest.raises(HEContextError, match="unavailable"):
        store.load()
