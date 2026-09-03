"""Pinned SpeechBrain ECAPA-TDNN embedding adapter.

This module is intentionally imported only by ECAPA runtime code. It requires
the optional Kaggle model dependencies and does not affect local data/config
tests that run without PyTorch.
"""

from __future__ import annotations

from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from threading import RLock
from typing import Any

try:
    import torch
    import torch.nn.functional as functional
    from huggingface_hub import snapshot_download
    from speechbrain.inference.classifiers import EncoderClassifier
except ModuleNotFoundError as error:  # pragma: no cover - Kaggle dependency gate.
    raise ModuleNotFoundError(
        "ECAPA-TDNN requires the optional 'ecapa' dependencies and Kaggle's "
        "existing CUDA-matched PyTorch build. Install with "
        "`python -m pip install -e '.[data,ecapa]'` on Kaggle."
    ) from error

from speaker_recognition.models.base import (
    ModelAdapterError,
    ModelAdapterMetadata,
    count_parameters,
)


ECAPA_MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"
ECAPA_REVISION = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
ECAPA_EMBEDDING_DIM = 192
ECAPA_PARAMETER_COUNT = 20_767_552
_REQUIRED_ARTIFACTS = (
    "hyperparams.yaml",
    "embedding_model.ckpt",
    "mean_var_norm_emb.ckpt",
    "classifier.ckpt",
    "label_encoder.txt",
    "example1.wav",
)
_FUNCTIONAL_PAD_LOCK = RLock()


def deterministic_reflection_pad1d(
    inputs: torch.Tensor,
    padding: tuple[int, int] | list[int],
) -> torch.Tensor:
    """Apply reflection padding with slice/flip/concatenate operations.

    PyTorch 2.10's native CUDA reflection-pad backward is nondeterministic and
    therefore raises under strict deterministic algorithms. This expression is
    forward-equivalent but its gradient is composed only of deterministic
    slicing, flipping, concatenation, and elementwise accumulation.
    """
    if len(padding) != 2:
        raise ModelAdapterError("ECAPA reflection padding must be one-dimensional.")
    left, right = padding
    if (
        isinstance(left, bool)
        or isinstance(right, bool)
        or not isinstance(left, int)
        or not isinstance(right, int)
        or left < 0
        or right < 0
    ):
        raise ModelAdapterError("ECAPA reflection padding values must be non-negative.")
    input_length = int(inputs.shape[-1])
    if left >= input_length or right >= input_length:
        raise ModelAdapterError(
            "ECAPA reflection padding must be smaller than its input length."
        )
    if left == 0 and right == 0:
        return inputs
    pieces = []
    if left:
        pieces.append(inputs[..., 1 : left + 1].flip(-1))
    pieces.append(inputs)
    if right:
        pieces.append(inputs[..., -(right + 1) : -1].flip(-1))
    return torch.cat(pieces, dim=-1)


@contextmanager
def _deterministic_reflection_padding() -> Any:
    """Scope the equivalent pad implementation to one ECAPA forward call.

    SpeechBrain calls ``torch.nn.functional.pad`` from both its feature and
    TDNN layers. The adapter is intentionally single-device and single-threaded;
    a process lock prevents overlapping calls from observing the temporary
    function replacement.
    """
    with _FUNCTIONAL_PAD_LOCK:
        native_pad = functional.pad

        def deterministic_pad(
            inputs: torch.Tensor,
            padding: tuple[int, ...] | list[int],
            mode: str = "constant",
            value: float | None = None,
        ) -> torch.Tensor:
            """Route only one-dimensional reflection mode to the safe form."""
            if mode == "reflect" and len(padding) == 2:
                return deterministic_reflection_pad1d(inputs, padding)
            return native_pad(inputs, padding, mode=mode, value=value)

        functional.pad = deterministic_pad
        try:
            yield
        finally:
            functional.pad = native_pad


class EcapaTdnnAdapter(torch.nn.Module):
    """Expose differentiable ECAPA embeddings through the shared contract."""

    def __init__(
        self,
        *,
        compute_features: torch.nn.Module,
        mean_var_norm: torch.nn.Module,
        embedding_model: torch.nn.Module,
        source_path: str | Path,
        revision: str = ECAPA_REVISION,
    ) -> None:
        """Register only modules needed for embedding extraction.

        The upstream VoxCeleb classifier is deliberately excluded because its
        7,205 source classes do not match either target dataset. A new training
        objective/head will be owned by the shared training layer.
        """
        super().__init__()
        self.compute_features = compute_features
        self.mean_var_norm = mean_var_norm
        self.embedding_model = embedding_model
        self.source_path = Path(source_path).expanduser().resolve()
        self.metadata = ModelAdapterMetadata(
            name="ecapa_tdnn",
            source_id=ECAPA_MODEL_ID,
            revision=revision,
            embedding_dim=ECAPA_EMBEDDING_DIM,
        )

        parameter_count = count_parameters(self.parameters())
        if parameter_count != ECAPA_PARAMETER_COUNT:
            raise ModelAdapterError(
                "ECAPA parameter count differs from the audited checkpoint: "
                f"expected {ECAPA_PARAMETER_COUNT:,}, received "
                f"{parameter_count:,}."
            )

    @classmethod
    def from_pretrained(
        cls,
        *,
        cache_dir: str | Path,
        device: str = "cpu",
        model_id: str = ECAPA_MODEL_ID,
        revision: str = ECAPA_REVISION,
        local_files_only: bool = False,
        token: str | None = None,
    ) -> "EcapaTdnnAdapter":
        """Download an immutable snapshot and load trainable encoder modules."""
        _validate_runtime_versions()
        if model_id != ECAPA_MODEL_ID:
            raise ModelAdapterError(
                f"ECAPA model_id must remain pinned to {ECAPA_MODEL_ID!r}."
            )
        if revision != ECAPA_REVISION:
            raise ModelAdapterError(
                f"ECAPA revision must remain pinned to {ECAPA_REVISION!r}."
            )

        root = Path(cache_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        snapshot_path = Path(
            snapshot_download(
                repo_id=model_id,
                revision=revision,
                cache_dir=root / "huggingface",
                allow_patterns=list(_REQUIRED_ARTIFACTS),
                local_files_only=local_files_only,
                token=token,
            )
        ).resolve()
        if snapshot_path.name != revision:
            raise ModelAdapterError(
                "Resolved ECAPA snapshot revision differs from the pinned "
                f"revision: {snapshot_path.name!r}."
            )
        missing_artifacts = [
            filename
            for filename in _REQUIRED_ARTIFACTS
            if not (snapshot_path / filename).is_file()
        ]
        if missing_artifacts:
            raise ModelAdapterError(
                f"ECAPA snapshot is missing artifacts: {missing_artifacts}"
            )

        # SpeechBrain's inference wrapper is used only as the official,
        # checkpoint-compatible hyperparameter loader. freeze_params=False is
        # essential: the project will fine-tune the target speaker encoder.
        classifier = EncoderClassifier.from_hparams(
            source=str(snapshot_path),
            savedir=str(root / "speechbrain_interface"),
            run_opts={"device": device},
            freeze_params=False,
        )
        adapter = cls(
            compute_features=classifier.mods.compute_features,
            mean_var_norm=classifier.mods.mean_var_norm,
            embedding_model=classifier.mods.embedding_model,
            source_path=snapshot_path,
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
        """Encode canonical ``[batch, time]`` waveforms into speaker vectors."""
        if waveforms.ndim != 2 or waveforms.shape[0] == 0 or waveforms.shape[1] == 0:
            raise ModelAdapterError(
                "ECAPA waveforms must have non-empty shape [batch, time]."
            )
        model_device = next(self.embedding_model.parameters()).device
        if waveforms.device != model_device:
            raise ModelAdapterError(
                f"Waveforms are on {waveforms.device}, but ECAPA is on "
                f"{model_device}."
            )
        if lengths is None:
            lengths = torch.ones(
                waveforms.shape[0],
                device=model_device,
                dtype=torch.float32,
            )
        if lengths.ndim != 1 or lengths.shape[0] != waveforms.shape[0]:
            raise ModelAdapterError(
                "ECAPA lengths must have shape [batch]."
            )
        if lengths.device != model_device:
            raise ModelAdapterError(
                f"Lengths are on {lengths.device}, but ECAPA is on "
                f"{model_device}."
            )
        if not bool(torch.isfinite(lengths).all()) or not bool(
            ((lengths > 0.0) & (lengths <= 1.0)).all()
        ):
            raise ModelAdapterError(
                "ECAPA relative lengths must be finite values in (0, 1]."
            )

        # This reproduces SpeechBrain 1.1.0 EncoderClassifier.encode_batch
        # without the source VoxCeleb classifier or inference-only wrapper.
        with _deterministic_reflection_padding():
            features = self.compute_features(waveforms.float())
            features = self.mean_var_norm(features, lengths)
            embeddings = self.embedding_model(features, lengths)
        if embeddings.ndim != 3 or embeddings.shape[1:] != (
            1,
            self.metadata.embedding_dim,
        ):
            raise ModelAdapterError(
                "ECAPA produced an unexpected embedding shape: "
                f"{tuple(embeddings.shape)}."
            )
        embeddings = embeddings[:, 0, :]
        if normalize:
            embeddings = functional.normalize(embeddings, p=2, dim=1)
        return embeddings

    def set_encoder_trainable(self, trainable: bool) -> None:
        """Freeze or unfreeze every feature and embedding parameter."""
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


def _validate_runtime_versions() -> None:
    """Require the SpeechBrain version validated on Kaggle."""
    try:
        speechbrain_version = version("speechbrain")
    except PackageNotFoundError as error:  # pragma: no cover - dependency gate.
        raise ModelAdapterError("SpeechBrain is not installed.") from error
    if speechbrain_version != "1.1.0":
        raise ModelAdapterError(
            "ECAPA adapter requires the audited SpeechBrain 1.1.0, received "
            f"{speechbrain_version}."
        )
