"""Framework-neutral contract shared by all speaker-embedding adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


class ModelAdapterError(ValueError):
    """Raised when a model adapter violates the shared embedding contract."""


@dataclass(frozen=True, slots=True)
class ModelAdapterMetadata:
    """Record immutable architecture and checkpoint identity.

    Parameters
    ----------
    name:
        Stable local adapter name.
    source_id:
        Upstream model or architecture repository identifier.
    revision:
        Full immutable source revision.
    embedding_dim:
        Width of one output speaker embedding.
    sample_rate:
        Required canonical waveform rate. The accepted pipeline fixes this at
        16 kHz for all three architectures.
    """

    name: str
    source_id: str
    revision: str
    embedding_dim: int
    sample_rate: int = 16000

    def __post_init__(self) -> None:
        """Reject incomplete or comparison-incompatible metadata."""
        for field_name in ("name", "source_id", "revision"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ModelAdapterError(
                    f"{field_name} must be a non-empty string."
                )
        if (
            isinstance(self.embedding_dim, bool)
            or not isinstance(self.embedding_dim, int)
            or self.embedding_dim <= 0
        ):
            raise ModelAdapterError("embedding_dim must be a positive integer.")
        if self.sample_rate != 16000:
            raise ModelAdapterError(
                "sample_rate must equal the accepted canonical 16000 Hz."
            )


@runtime_checkable
class SpeakerEmbeddingAdapter(Protocol):
    """Structural interface required by shared training and evaluation code.

    Tensor types remain ``Any`` here so importing the data/configuration package
    never imports PyTorch. Concrete model modules provide precise tensor types
    and require the Kaggle model dependencies.
    """

    metadata: ModelAdapterMetadata

    def __call__(
        self,
        waveforms: Any,
        lengths: Any | None = None,
        *,
        normalize: bool = True,
    ) -> Any:
        """Return one ``[batch, embedding_dim]`` tensor."""
        ...

    def parameters(self, recurse: bool = True) -> Any:
        """Yield trainable module parameters."""
        ...

    def train(self, mode: bool = True) -> Any:
        """Set training or evaluation mode."""
        ...

    def eval(self) -> Any:
        """Set evaluation mode."""
        ...

    def state_dict(self, *args: Any, **kwargs: Any) -> Any:
        """Return serializable adapter state."""
        ...

    def set_encoder_trainable(self, trainable: bool) -> None:
        """Freeze or unfreeze the speaker encoder explicitly."""
        ...


def count_parameters(parameters: Any, *, trainable_only: bool = False) -> int:
    """Count tensor elements from any PyTorch-like parameter iterable."""
    total = 0
    for parameter in parameters:
        if trainable_only and not bool(parameter.requires_grad):
            continue
        total += int(parameter.numel())
    return total
