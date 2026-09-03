"""Shared model contracts; concrete adapters have optional dependencies."""

from speaker_recognition.models.base import (
    ModelAdapterError,
    ModelAdapterMetadata,
    SpeakerEmbeddingAdapter,
    count_parameters,
)

__all__ = [
    "ModelAdapterError",
    "ModelAdapterMetadata",
    "SpeakerEmbeddingAdapter",
    "count_parameters",
]
