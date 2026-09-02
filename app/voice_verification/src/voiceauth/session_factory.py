"""One construction path for Streamlit and the local LiveKit Agent."""

from __future__ import annotations

from voiceauth.config import VoiceSettings
from voiceauth.he import create_session_context
from voiceauth.keychain import KeychainHEContextStore
from voiceauth.matcher_client import MatcherClient
from voiceauth.process_engine import ProcessSpeakerEngine
from voiceauth.rawnet3 import RawNet3Runtime
from voiceauth.service import VoiceVerificationService


class VoiceAuthSessionFactory:
    """Construct a service with either ephemeral or macOS-Keychain HE context."""

    def __init__(self, settings: VoiceSettings) -> None:
        self._settings = settings

    def create(
        self,
        *,
        persistent_context: bool,
        keychain_account: str | None = None,
    ) -> VoiceVerificationService:
        """Create a voice service with an optional account-scoped HE context.

        The user-facing gateway always passes a stable account-derived Keychain
        account. Streamlit can omit it for its internal debug workflow.
        """
        context = (
            KeychainHEContextStore(
                service_name=self._settings.keychain_service,
                account_name=keychain_account or self._settings.keychain_account,
            ).load_or_create()
            if persistent_context
            else create_session_context()
        )
        service = VoiceVerificationService(
            ProcessSpeakerEngine(
                RawNet3Runtime(
                    str(self._settings.model_checkpoint),
                    str(self._settings.model_cache_dir),
                    self._settings.model_device,
                )
            ),
            MatcherClient(self._settings.matcher_url, self._settings.matcher_token),
            context,
            threshold=self._settings.threshold,
        )
        service.initialize()
        return service
