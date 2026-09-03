"""Privacy-preserving voice verification domain package."""

from voiceauth.config import EMBEDDING_DIMENSION, SV_THRESHOLD

__all__ = ["EMBEDDING_DIMENSION", "SV_THRESHOLD"]
"""Privacy-preserving speaker-verification core.

Import concrete modules directly (for example ``voiceauth.gate``).  Keeping
this package initializer dependency-free lets the matcher load only its HE
components, not audio/model dependencies.
"""
