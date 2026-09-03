"""Tests for the framework-neutral speaker-model adapter contract."""

from dataclasses import FrozenInstanceError

import pytest

from speaker_recognition.models.base import (
    ModelAdapterError,
    ModelAdapterMetadata,
    SpeakerEmbeddingAdapter,
    count_parameters,
)


def test_adapter_metadata_records_immutable_model_identity() -> None:
    """Checkpoint identity must not change after adapter construction."""
    metadata = ModelAdapterMetadata(
        name="fixture",
        source_id="owner/model",
        revision="0123456789abcdef",
        embedding_dim=192,
    )

    assert metadata.sample_rate == 16000
    with pytest.raises(FrozenInstanceError):
        metadata.revision = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "metadata",
    [
        {"name": "", "source_id": "owner/model", "revision": "abc", "embedding_dim": 1},
        {"name": "x", "source_id": "", "revision": "abc", "embedding_dim": 1},
        {"name": "x", "source_id": "owner/model", "revision": "", "embedding_dim": 1},
        {"name": "x", "source_id": "owner/model", "revision": "abc", "embedding_dim": 0},
        {"name": "x", "source_id": "owner/model", "revision": "abc", "embedding_dim": 1, "sample_rate": 8000},
    ],
)
def test_reject_invalid_adapter_metadata(metadata: dict[str, object]) -> None:
    """Incomplete identity or incompatible audio contracts must fail."""
    with pytest.raises(ModelAdapterError):
        ModelAdapterMetadata(**metadata)  # type: ignore[arg-type]


def test_runtime_protocol_accepts_structurally_complete_adapter() -> None:
    """Training code can validate adapters without concrete inheritance."""
    adapter = _FakeAdapter()

    assert isinstance(adapter, SpeakerEmbeddingAdapter)


def test_count_parameters_supports_trainable_filtering() -> None:
    """Parameter accounting should work without importing PyTorch locally."""
    parameters = (
        _FakeParameter(10, requires_grad=True),
        _FakeParameter(7, requires_grad=False),
    )

    assert count_parameters(parameters) == 17
    assert count_parameters(parameters, trainable_only=True) == 10


class _FakeParameter:
    """Minimal PyTorch-like parameter used for dependency-free tests."""

    def __init__(self, count: int, *, requires_grad: bool) -> None:
        self._count = count
        self.requires_grad = requires_grad

    def numel(self) -> int:
        """Return the configured element count."""
        return self._count


class _FakeAdapter:
    """Minimal structurally compatible adapter fixture."""

    metadata = ModelAdapterMetadata(
        name="fake",
        source_id="test/fake",
        revision="0123456789abcdef",
        embedding_dim=2,
    )

    def __call__(self, waveforms, lengths=None, *, normalize=True):
        """Return the provided fake batch."""
        return waveforms

    def parameters(self, recurse=True):
        """Return no parameters."""
        return iter(())

    def train(self, mode=True):
        """Return self for fluent mode changes."""
        return self

    def eval(self):
        """Return self for fluent mode changes."""
        return self

    def state_dict(self, *args, **kwargs):
        """Return empty fake state."""
        return {}

    def set_encoder_trainable(self, trainable):
        """Accept the requested fake trainability state."""
        return None
