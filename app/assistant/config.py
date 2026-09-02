"""Configuration owned by the trusted local Python Agent process."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from uuid import UUID


def _as_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class LocalAgentSettings:
    livekit_url: str
    livekit_api_key: str
    livekit_api_secret: str
    livekit_dev_mode: bool
    owner_identity_id: UUID | None
    auth_ttl: timedelta
    keychain_service: str
    keychain_account: str
    conversation_enabled: bool
    whisper_model: str
    whisper_device: str
    tts_provider: str
    tts_fallback_provider: str | None
    zerotts_model: str
    zerotts_voice: str
    zerotts_cache_dir: str | None
    zerotts_warmup: bool
    tts_queue_max_chunks: int
    tts_max_sentence_chars: int
    edge_tts_voice: str
    tts_timeout_seconds: float
    participant_join_timeout_seconds: float
    vad_min_speech_seconds: float
    vad_silence_seconds: float
    vad_maximum_seconds: float
    smart_turn_model_path: str | None
    smart_turn_cpu_count: int
    smart_turn_pre_speech_ms: float
    smart_turn_max_wait_seconds: float

    @classmethod
    def from_environment(cls) -> "LocalAgentSettings":
        owner_value = os.getenv("VOICE_OWNER_ID", "").strip()
        if owner_value:
            try:
                owner_identity_id = UUID(owner_value)
            except ValueError as error:
                raise ValueError("VOICE_OWNER_ID must be a UUID from the enrolled local identity.") from error
        else:
            # Compatibility fallback for the legacy Streamlit-only local flow.
            # The unified gateway supplies a target through signed dispatch metadata.
            owner_identity_id = None
        ttl_seconds = int(os.getenv("VOICE_AUTH_TTL_SECONDS", "300"))
        if ttl_seconds <= 0:
            raise ValueError("VOICE_AUTH_TTL_SECONDS must be positive.")
        vad_min_speech_seconds = float(os.getenv("VOICE_VAD_MIN_SPEECH_SECONDS", "0.3"))
        vad_silence_seconds = float(os.getenv("VOICE_VAD_SILENCE_SECONDS", "0.45"))
        vad_maximum_seconds = float(os.getenv("VOICE_VAD_MAXIMUM_SECONDS", "15"))
        smart_turn_model_path = os.getenv("VOICE_SMART_TURN_MODEL_PATH", "").strip() or None
        smart_turn_cpu_count = int(os.getenv("VOICE_SMART_TURN_CPU_COUNT", "1"))
        smart_turn_pre_speech_ms = float(os.getenv("VOICE_SMART_TURN_PRE_SPEECH_MS", "500"))
        smart_turn_max_wait_seconds = float(os.getenv("VOICE_SMART_TURN_MAX_WAIT_SECONDS", "2"))
        tts_timeout_seconds = float(os.getenv("VOICE_TTS_TIMEOUT_SECONDS", "10"))
        tts_provider = os.getenv("VOICE_TTS_PROVIDER", "zerotts").strip().lower() or "zerotts"
        tts_fallback_value = os.getenv("VOICE_TTS_FALLBACK_PROVIDER", "edge").strip().lower()
        tts_fallback_provider = tts_fallback_value or None
        zerotts_model = os.getenv("VOICE_ZEROTTS_MODEL", "zeroweight-ai/ZeroTTS").strip()
        zerotts_voice = os.getenv("VOICE_ZEROTTS_VOICE", "maichi").strip()
        zerotts_cache_dir = os.getenv("VOICE_ZEROTTS_CACHE_DIR", "").strip() or None
        zerotts_warmup = _as_bool("VOICE_ZEROTTS_WARMUP", True)
        tts_queue_max_chunks = int(os.getenv("VOICE_TTS_QUEUE_MAX_CHUNKS", "8"))
        tts_max_sentence_chars = int(os.getenv("VOICE_TTS_MAX_SENTENCE_CHARS", "120"))
        participant_join_timeout_seconds = float(
            os.getenv("VOICE_AGENT_JOIN_TIMEOUT_SECONDS", "45")
        )
        if vad_min_speech_seconds <= 0 or vad_silence_seconds <= 0 or vad_maximum_seconds <= vad_min_speech_seconds:
            raise ValueError("VOICE_VAD timing values must be positive and maximum must exceed minimum speech.")
        if tts_timeout_seconds <= 0:
            raise ValueError("VOICE_TTS_TIMEOUT_SECONDS must be positive.")
        if tts_provider not in {"zerotts", "edge"}:
            raise ValueError("VOICE_TTS_PROVIDER must be one of: zerotts, edge.")
        if tts_fallback_provider not in {None, "zerotts", "edge"}:
            raise ValueError("VOICE_TTS_FALLBACK_PROVIDER must be empty, zerotts, or edge.")
        if tts_fallback_provider == tts_provider:
            raise ValueError("VOICE_TTS_FALLBACK_PROVIDER must differ from VOICE_TTS_PROVIDER.")
        if not zerotts_model or not zerotts_voice:
            raise ValueError("VOICE_ZEROTTS_MODEL and VOICE_ZEROTTS_VOICE must not be empty.")
        if tts_queue_max_chunks <= 0 or tts_max_sentence_chars < 8:
            raise ValueError("TTS queue size must be positive and sentence limit must be at least 8.")
        if smart_turn_cpu_count <= 0 or smart_turn_pre_speech_ms < 0 or smart_turn_max_wait_seconds <= 0:
            raise ValueError("Smart Turn configuration values are invalid.")
        if participant_join_timeout_seconds <= 0:
            raise ValueError("VOICE_AGENT_JOIN_TIMEOUT_SECONDS must be positive.")
        api_key = os.getenv("LIVEKIT_API_KEY", "").strip()
        api_secret = os.getenv("LIVEKIT_API_SECRET", "").strip()
        dev_mode = _as_bool("LIVEKIT_DEV_MODE")
        if not api_key or not api_secret:
            raise ValueError("LIVEKIT_API_KEY and LIVEKIT_API_SECRET are required by the local agent.")
        if not dev_mode and (api_key == "devkey" or api_secret == "secret"):
            raise ValueError("Refusing development LiveKit credentials unless LIVEKIT_DEV_MODE=true.")
        return cls(
            livekit_url=os.getenv("LIVEKIT_URL", "ws://127.0.0.1:7880").rstrip("/"),
            livekit_api_key=api_key,
            livekit_api_secret=api_secret,
            livekit_dev_mode=dev_mode,
            owner_identity_id=owner_identity_id,
            auth_ttl=timedelta(seconds=ttl_seconds),
            keychain_service=os.getenv("VOICE_HE_KEYCHAIN_SERVICE", "who-speak.voice-he"),
            keychain_account=os.getenv("VOICE_HE_KEYCHAIN_ACCOUNT", "local-owner"),
            conversation_enabled=_as_bool("VOICE_AGENT_CONVERSATION_ENABLED"),
            whisper_model=os.getenv("VOICE_WHISPER_MODEL", "base").strip() or "base",
            whisper_device=os.getenv("VOICE_WHISPER_DEVICE", "cpu").strip() or "cpu",
            tts_provider=tts_provider,
            tts_fallback_provider=tts_fallback_provider,
            zerotts_model=zerotts_model,
            zerotts_voice=zerotts_voice,
            zerotts_cache_dir=zerotts_cache_dir,
            zerotts_warmup=zerotts_warmup,
            tts_queue_max_chunks=tts_queue_max_chunks,
            tts_max_sentence_chars=tts_max_sentence_chars,
            edge_tts_voice=os.getenv("VOICE_EDGE_TTS_VOICE", "vi-VN-HoaiMyNeural").strip() or "vi-VN-HoaiMyNeural",
            tts_timeout_seconds=tts_timeout_seconds,
            participant_join_timeout_seconds=participant_join_timeout_seconds,
            vad_min_speech_seconds=vad_min_speech_seconds,
            vad_silence_seconds=vad_silence_seconds,
            vad_maximum_seconds=vad_maximum_seconds,
            smart_turn_model_path=smart_turn_model_path,
            smart_turn_cpu_count=smart_turn_cpu_count,
            smart_turn_pre_speech_ms=smart_turn_pre_speech_ms,
            smart_turn_max_wait_seconds=smart_turn_max_wait_seconds,
        )


@dataclass(frozen=True)
class OpenAISettings:
    api_key: str
    model: str

    @classmethod
    def from_environment(cls) -> "OpenAISettings":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required only when the LLM provider is enabled.")
        model = os.getenv("OPENAI_MODEL", "gpt-5").strip()
        if not model:
            raise ValueError("OPENAI_MODEL must not be empty.")
        return cls(api_key=api_key, model=model)
