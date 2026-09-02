"""Stable model and decision constants for the voice-verification profile."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

MODEL_PROFILE = "rawnet3-vimd-best-epoch-2"
PREPROCESSING_PROFILE = "rawnet3-vimd-16khz-v1"
HE_PROFILE = "ckks-128-defensive-v1"
EMBEDDING_DIMENSION = 256
TARGET_SAMPLE_RATE = 16_000
EVALUATION_SAMPLES = 64_240
SV_THRESHOLD = 0.6565317440180312
SELECTED_CHECKPOINT_SHA256 = "0b06fd3c4644d6c1cf4e7f9c087cf7fba8be493952589ab4321f8319d4215386"
SELECTED_CONFIG_SHA256 = "08392e8146fb4ae42cc7b971641d4339c9f98f517bd6f3ece1d5b2bf2fdafa94"


@dataclass(frozen=True)
class VoiceSettings:
    """Runtime settings deliberately sourced only from process configuration."""

    matcher_url: str
    matcher_token: str
    matcher_database: str
    model_checkpoint: Path
    model_cache_dir: Path
    model_device: str
    threshold: float
    show_debug_score: bool
    he_context_mode: str
    keychain_service: str
    keychain_account: str

    @classmethod
    def from_environment(cls, *, root: Path | None = None) -> "VoiceSettings":
        root = (root or Path.cwd()).resolve()
        threshold = float(os.getenv("VOICE_SV_THRESHOLD", str(SV_THRESHOLD)))
        if not 0.0 < threshold <= 1.0:
            raise ValueError("VOICE_SV_THRESHOLD must be greater than 0 and at most 1.")
        token = os.getenv("VOICE_MATCHER_TOKEN", "")
        if len(token) < 8:
            raise ValueError("VOICE_MATCHER_TOKEN must contain at least 8 characters.")
        context_mode = os.getenv("VOICE_HE_CONTEXT_MODE", "session").strip().lower()
        if context_mode not in {"session", "keychain"}:
            raise ValueError("VOICE_HE_CONTEXT_MODE must be either 'session' or 'keychain'.")
        return cls(
            matcher_url=os.getenv("VOICE_MATCHER_URL", "http://127.0.0.1:8011").rstrip("/"),
            matcher_token=token,
            matcher_database=os.getenv("VOICE_MATCHER_DATABASE", "./data/voice_matcher.db"),
            model_checkpoint=(root / os.getenv("VOICE_MODEL_CHECKPOINT", "")).resolve(),
            model_cache_dir=(root / os.getenv("VOICE_MODEL_CACHE_DIR", "./data/model-cache")).resolve(),
            model_device=os.getenv("VOICE_MODEL_DEVICE", "cpu"),
            threshold=threshold,
            show_debug_score=os.getenv("VOICE_SHOW_DEBUG_SCORE", "false").strip().lower() in {"1", "true", "yes", "on"},
            he_context_mode=context_mode,
            keychain_service=os.getenv("VOICE_HE_KEYCHAIN_SERVICE", "who-speak.voice-he"),
            keychain_account=os.getenv("VOICE_HE_KEYCHAIN_ACCOUNT", "local-owner"),
        )
