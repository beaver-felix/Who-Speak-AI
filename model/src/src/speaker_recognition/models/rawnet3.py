"""Pinned RawNet3 speaker-embedding adapter.

This concrete module requires the optional RawNet3 dependencies. It is kept
separate from the dependency-free adapter protocol and data pipeline.
"""

from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping

try:
    import torch
    import torch.nn.functional as functional
    from huggingface_hub import snapshot_download
except ModuleNotFoundError as error:  # pragma: no cover - Kaggle dependency gate.
    raise ModuleNotFoundError(
        "RawNet3 requires the optional 'rawnet3' dependencies and Kaggle's "
        "existing CUDA-matched PyTorch build. Install with "
        "`python -m pip install -e '.[data,rawnet3]'` on Kaggle."
    ) from error

from speaker_recognition.models.base import (
    ModelAdapterError,
    ModelAdapterMetadata,
    count_parameters,
)
from speaker_recognition.third_party.clova_rawnet3 import build_rawnet3


RAWNET3_MODEL_ID = "jungjee/RawNet3"
RAWNET3_CHECKPOINT_REVISION = "c89102eea20c3f96917c434de673c0ace0caddc0"
RAWNET3_ARCHITECTURE_REVISION = "f51bab870672a9b0b50fa158b4e30f329e7866d7"
RAWNET3_CHECKPOINT_FILENAME = "model.pt"
RAWNET3_CHECKPOINT_SHA256 = (
    "1ab283bcdf776bfceceea18240e56a8756835b1911b04f9c44f347d47c09f90c"
)
RAWNET3_EMBEDDING_DIM = 256
RAWNET3_PARAMETER_COUNT = 16_280_322
RAWNET3_TRAIN_SAMPLES = 48_240
RAWNET3_EVALUATION_SAMPLES = 64_240


class RawNet3Adapter(torch.nn.Module):
    """Expose the official pretrained RawNet3 through the shared contract."""

    def __init__(
        self,
        *,
        encoder: torch.nn.Module,
        source_path: str | Path,
        checkpoint_sha256: str,
        revision: str = RAWNET3_CHECKPOINT_REVISION,
    ) -> None:
        """Register an already constructed and strictly loaded encoder."""
        super().__init__()
        self.encoder = encoder
        self.source_path = Path(source_path).expanduser().resolve()
        self.checkpoint_sha256 = checkpoint_sha256
        self.architecture_revision = RAWNET3_ARCHITECTURE_REVISION
        self.metadata = ModelAdapterMetadata(
            name="rawnet3",
            source_id=RAWNET3_MODEL_ID,
            revision=revision,
            embedding_dim=RAWNET3_EMBEDDING_DIM,
        )

        parameter_count = count_parameters(self.parameters())
        if parameter_count != RAWNET3_PARAMETER_COUNT:
            raise ModelAdapterError(
                "RawNet3 parameter count differs from the audited "
                f"architecture: expected {RAWNET3_PARAMETER_COUNT:,}, "
                f"received {parameter_count:,}."
            )

    @classmethod
    def from_pretrained(
        cls,
        *,
        cache_dir: str | Path,
        device: str = "cpu",
        model_id: str = RAWNET3_MODEL_ID,
        revision: str = RAWNET3_CHECKPOINT_REVISION,
        local_files_only: bool = False,
        token: str | None = None,
    ) -> "RawNet3Adapter":
        """Resolve, authenticate, safely load, and strictly apply the checkpoint."""
        _validate_runtime_versions()
        if model_id != RAWNET3_MODEL_ID:
            raise ModelAdapterError(
                f"RawNet3 model_id must remain pinned to {RAWNET3_MODEL_ID!r}."
            )
        if revision != RAWNET3_CHECKPOINT_REVISION:
            raise ModelAdapterError(
                "RawNet3 revision must remain pinned to "
                f"{RAWNET3_CHECKPOINT_REVISION!r}."
            )

        root = Path(cache_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        snapshot_path = Path(
            snapshot_download(
                repo_id=model_id,
                revision=revision,
                cache_dir=root / "huggingface",
                allow_patterns=[RAWNET3_CHECKPOINT_FILENAME],
                local_files_only=local_files_only,
                token=token,
            )
        ).resolve()
        if snapshot_path.name != revision:
            raise ModelAdapterError(
                "Resolved RawNet3 snapshot differs from the pinned revision: "
                f"{snapshot_path.name!r}."
            )

        checkpoint_path = snapshot_path / RAWNET3_CHECKPOINT_FILENAME
        if not checkpoint_path.is_file():
            raise ModelAdapterError(
                f"RawNet3 snapshot is missing {RAWNET3_CHECKPOINT_FILENAME!r}."
            )
        checkpoint_sha256 = _sha256_file(checkpoint_path)
        if checkpoint_sha256 != RAWNET3_CHECKPOINT_SHA256:
            raise ModelAdapterError(
                "RawNet3 checkpoint SHA-256 mismatch: expected "
                f"{RAWNET3_CHECKPOINT_SHA256}, received {checkpoint_sha256}."
            )

        # weights_only=True blocks arbitrary pickle globals. The audited file
        # contains a plain dictionary whose "model" value is a tensor state map.
        checkpoint: Any = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        state = _extract_state_dictionary(checkpoint)
        encoder = build_rawnet3(
            output_dimensions=RAWNET3_EMBEDDING_DIM,
            encoder_type="ECA",
            sinc_stride=10,
        )
        incompatible = encoder.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ModelAdapterError(
                "Strict RawNet3 checkpoint load reported incompatible keys: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}."
            )

        adapter = cls(
            encoder=encoder,
            source_path=snapshot_path,
            checkpoint_sha256=checkpoint_sha256,
            revision=revision,
        )
        adapter.to(device)
        return adapter

    def forward(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor | None = None,
        *,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Encode equal-length canonical waveforms into 256-D speaker vectors."""
        if waveforms.ndim != 2 or waveforms.shape[0] == 0 or waveforms.shape[1] < 2:
            raise ModelAdapterError(
                "RawNet3 waveforms must have non-empty shape [batch, time] "
                "with at least two samples."
            )
        model_device = next(self.encoder.parameters()).device
        if waveforms.device != model_device:
            raise ModelAdapterError(
                f"Waveforms are on {waveforms.device}, but RawNet3 is on "
                f"{model_device}."
            )
        if lengths is not None:
            if lengths.ndim != 1 or lengths.shape[0] != waveforms.shape[0]:
                raise ModelAdapterError("RawNet3 lengths must have shape [batch].")
            if lengths.device != model_device:
                raise ModelAdapterError(
                    f"Lengths are on {lengths.device}, but RawNet3 is on "
                    f"{model_device}."
                )
            if not bool(torch.isfinite(lengths).all()) or not bool(
                torch.allclose(lengths.float(), torch.ones_like(lengths.float()))
            ):
                raise ModelAdapterError(
                    "RawNet3 requires equal fixed crops; every relative "
                    "length must equal 1."
                )

        embeddings = self.encoder(waveforms.float())
        expected_shape = (waveforms.shape[0], self.metadata.embedding_dim)
        if tuple(embeddings.shape) != expected_shape:
            raise ModelAdapterError(
                "RawNet3 produced an unexpected embedding shape: "
                f"{tuple(embeddings.shape)}."
            )
        if normalize:
            embeddings = functional.normalize(embeddings, p=2, dim=1)
        return embeddings

    def set_encoder_trainable(self, trainable: bool) -> None:
        """Freeze or unfreeze every RawNet3 encoder parameter."""
        if not isinstance(trainable, bool):
            raise ModelAdapterError("trainable must be a boolean.")
        for parameter in self.parameters():
            parameter.requires_grad = trainable

    @property
    def parameter_count(self) -> int:
        """Return the audited total encoder parameter count."""
        return count_parameters(self.parameters())

    @property
    def trainable_parameter_count(self) -> int:
        """Return parameters currently enabled for gradient updates."""
        return count_parameters(self.parameters(), trainable_only=True)


def _extract_state_dictionary(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    """Validate the restricted checkpoint's expected tensor-only structure."""
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != {"model"}:
        raise ModelAdapterError(
            "RawNet3 checkpoint must contain exactly the top-level 'model' key."
        )
    state = checkpoint["model"]
    if not isinstance(state, Mapping) or not state:
        raise ModelAdapterError("RawNet3 'model' value must be a non-empty mapping.")
    non_tensor_keys = [
        str(key) for key, value in state.items() if not isinstance(value, torch.Tensor)
    ]
    if non_tensor_keys:
        raise ModelAdapterError(
            "RawNet3 state dictionary contains non-tensor values at keys: "
            f"{non_tensor_keys[:5]}."
        )
    return state


def _sha256_file(path: Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    """Hash a checkpoint incrementally without loading it twice into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_runtime_versions() -> None:
    """Require the Asteroid Filterbanks release validated on Kaggle."""
    try:
        installed_version = version("asteroid-filterbanks")
    except PackageNotFoundError as error:  # pragma: no cover - dependency gate.
        raise ModelAdapterError("Asteroid Filterbanks is not installed.") from error
    if installed_version != "0.4.0":
        raise ModelAdapterError(
            "RawNet3 requires the audited Asteroid Filterbanks 0.4.0, "
            f"received {installed_version}."
        )
