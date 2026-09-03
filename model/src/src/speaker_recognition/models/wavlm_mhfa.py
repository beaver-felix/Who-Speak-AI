"""Pinned fine-tuned WavLM+MHFA speaker-embedding adapter."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import types
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping

try:
    import torch
    import torch.nn.functional as functional
    from huggingface_hub import snapshot_download
except ModuleNotFoundError as error:  # pragma: no cover - Kaggle dependency gate.
    raise ModuleNotFoundError(
        "WavLM+MHFA requires the optional 'wavlm_mhfa' dependencies and "
        "Kaggle's existing CUDA-matched PyTorch build. Install with "
        "`python -m pip install -e '.[data,wavlm_mhfa]'` on Kaggle."
    ) from error

from speaker_recognition.models.base import (
    ModelAdapterError,
    ModelAdapterMetadata,
    count_parameters,
)
from speaker_recognition.third_party.theolepage_wavlm_mhfa import MHFA


WAVLM_MHFA_MODEL_ID = "theolepage/wavlm_ssl_sv"
WAVLM_MHFA_REVISION = "bfb8527de83b5347fb81b1e9e31be241656ca103"
WAVLM_MHFA_CHECKPOINT_FILENAME = "model000000018.model"
WAVLM_MHFA_CHECKPOINT_FILE_ID = "1RabuRETASqhh39K8weSoNkBa5DGRvgyx"
WAVLM_MHFA_CHECKPOINT_SHA256 = (
    "0178a115dc0a43a94a71287e51d1df5016c2aeefc04169548dad40ac8a6e67da"
)
WAVLM_PARAMETER_COUNT = 94_381_936
MHFA_PARAMETER_COUNT = 2_302_554
WAVLM_MHFA_PARAMETER_COUNT = 96_684_490
WAVLM_MHFA_EMBEDDING_DIM = 256
WAVLM_MHFA_REPRESENTATION_LEVELS = 13
WAVLM_MHFA_TRAIN_SAMPLES = 48_240
WAVLM_MHFA_EVALUATION_SAMPLES = 64_240

_SOURCE_HASHES = {
    "models/Baseline/WavLM.py": (
        "7cc0837302ff032d048c0f43ebdafdf0f009f72a10d78aa82f486df33c39aa63"
    ),
    "models/Baseline/modules.py": (
        "7a06a14a7dc95c5f65cd6b09ed126013821512489dcfec2e58bd8b544ce46656"
    ),
    "models/Baseline/Spk_Encoder.py": (
        "fc4638d657a3ad09953e54ee76ae4877904b6a92c03d2f91ed6691b7f770d40f"
    ),
    "LICENSE.md": (
        "04a05562ba5e9841452b4e7209d226e543b975d0809794452c9e301f457a183a"
    ),
}

# Exact configuration recovered from the safely inspected official Microsoft
# WavLM-Base+ checkpoint. The fine-tuned speaker checkpoint contains all model
# tensors but not this architecture configuration.
_WAVLM_BASE_PLUS_CONFIG: dict[str, object] = {
    "activation_dropout": 0.0,
    "activation_fn": "gelu",
    "attention_dropout": 0.1,
    "conv_bias": False,
    "conv_feature_layers": "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
    "conv_pos": 128,
    "conv_pos_groups": 16,
    "dropout": 0.1,
    "dropout_features": 0.1,
    "dropout_input": 0.1,
    "encoder_attention_heads": 12,
    "encoder_embed_dim": 768,
    "encoder_ffn_embed_dim": 3072,
    "encoder_layerdrop": 0.05,
    "encoder_layers": 12,
    "extractor_mode": "default",
    "feature_grad_mult": 0.1,
    "gru_rel_pos": True,
    "layer_norm_first": False,
    "mask_channel_length": 10,
    "mask_channel_min_space": 1,
    "mask_channel_other": 0.0,
    "mask_channel_prob": 0.0,
    "mask_channel_selection": "static",
    "mask_length": 10,
    "mask_min_space": 1,
    "mask_other": 0.0,
    "mask_prob": 0.8,
    "mask_selection": "static",
    "max_distance": 800,
    "no_mask_channel_overlap": False,
    "no_mask_overlap": False,
    "normalize": False,
    "num_buckets": 320,
    "relative_position_embedding": True,
}


class WavlmMhfaAdapter(torch.nn.Module):
    """Expose the official fine-tuned WavLM+MHFA embedding path."""

    def __init__(
        self,
        *,
        wavlm: torch.nn.Module,
        mhfa: MHFA,
        source_path: str | Path,
        checkpoint_path: str | Path,
        checkpoint_sha256: str,
    ) -> None:
        """Register strictly loaded WavLM and MHFA modules only."""
        super().__init__()
        self.wavlm = wavlm
        self.mhfa = mhfa
        self.source_path = Path(source_path).expanduser().resolve()
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.checkpoint_sha256 = checkpoint_sha256
        self.metadata = ModelAdapterMetadata(
            name="wavlm_mhfa",
            source_id=WAVLM_MHFA_MODEL_ID,
            revision=WAVLM_MHFA_REVISION,
            embedding_dim=WAVLM_MHFA_EMBEDDING_DIM,
        )

        wavlm_count = count_parameters(self.wavlm.parameters())
        mhfa_count = count_parameters(self.mhfa.parameters())
        total_count = count_parameters(self.parameters())
        if wavlm_count != WAVLM_PARAMETER_COUNT:
            raise ModelAdapterError(
                f"WavLM expected {WAVLM_PARAMETER_COUNT:,} parameters, "
                f"received {wavlm_count:,}."
            )
        if mhfa_count != MHFA_PARAMETER_COUNT:
            raise ModelAdapterError(
                f"MHFA expected {MHFA_PARAMETER_COUNT:,} parameters, "
                f"received {mhfa_count:,}."
            )
        if total_count != WAVLM_MHFA_PARAMETER_COUNT:
            raise ModelAdapterError(
                f"WavLM+MHFA expected {WAVLM_MHFA_PARAMETER_COUNT:,} "
                f"parameters, received {total_count:,}."
            )

    @classmethod
    def from_pretrained(
        cls,
        *,
        cache_dir: str | Path,
        device: str = "cpu",
        checkpoint_path: str | Path | None = None,
        local_files_only: bool = False,
        token: str | None = None,
    ) -> "WavlmMhfaAdapter":
        """Verify pinned source and weights before strict construction."""
        _validate_runtime_versions()
        root = Path(cache_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)

        source_path = Path(
            snapshot_download(
                repo_id=WAVLM_MHFA_MODEL_ID,
                revision=WAVLM_MHFA_REVISION,
                cache_dir=root / "huggingface",
                allow_patterns=list(_SOURCE_HASHES),
                local_files_only=local_files_only,
                token=token,
            )
        ).resolve()
        if source_path.name != WAVLM_MHFA_REVISION:
            raise ModelAdapterError(
                "Resolved WavLM+MHFA source differs from the pinned revision: "
                f"{source_path.name!r}."
            )
        _verify_source_files(source_path)

        resolved_checkpoint = (
            Path(checkpoint_path).expanduser().resolve()
            if checkpoint_path is not None
            else root / "checkpoint" / WAVLM_MHFA_CHECKPOINT_FILENAME
        )
        if not resolved_checkpoint.is_file():
            if checkpoint_path is not None or local_files_only:
                raise ModelAdapterError(
                    f"WavLM+MHFA checkpoint does not exist: {resolved_checkpoint}"
                )
            _download_checkpoint(resolved_checkpoint)
        checkpoint_sha256 = _sha256_file(resolved_checkpoint)
        if checkpoint_sha256 != WAVLM_MHFA_CHECKPOINT_SHA256:
            raise ModelAdapterError(
                "WavLM+MHFA checkpoint SHA-256 mismatch: expected "
                f"{WAVLM_MHFA_CHECKPOINT_SHA256}, received "
                f"{checkpoint_sha256}."
            )

        checkpoint: Any = torch.load(
            resolved_checkpoint,
            map_location="cpu",
            weights_only=True,
        )
        wavlm_state, mhfa_state = _split_checkpoint_state(checkpoint)
        wavlm_module = _load_verified_wavlm_module(source_path)
        config = wavlm_module.WavLMConfig(dict(_WAVLM_BASE_PLUS_CONFIG))
        wavlm = wavlm_module.WavLM(config)
        mhfa = MHFA(
            head_nb=64,
            inputs_dim=768,
            compression_dim=128,
            outputs_dim=WAVLM_MHFA_EMBEDDING_DIM,
            representation_levels=WAVLM_MHFA_REPRESENTATION_LEVELS,
        )
        _strict_load(wavlm, wavlm_state, component="WavLM")
        _strict_load(mhfa, mhfa_state, component="MHFA")

        adapter = cls(
            wavlm=wavlm,
            mhfa=mhfa,
            source_path=source_path,
            checkpoint_path=resolved_checkpoint,
            checkpoint_sha256=checkpoint_sha256,
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
        """Encode fixed 16 kHz crops into 256-D speaker embeddings."""
        if waveforms.ndim != 2 or waveforms.shape[0] == 0 or waveforms.shape[1] == 0:
            raise ModelAdapterError(
                "WavLM+MHFA waveforms must have non-empty shape [batch, time]."
            )
        model_device = next(self.wavlm.parameters()).device
        if waveforms.device != model_device:
            raise ModelAdapterError(
                f"Waveforms are on {waveforms.device}, but WavLM+MHFA is on "
                f"{model_device}."
            )
        _validate_fixed_lengths(lengths, waveforms.shape[0], model_device)

        _, layer_results = self.wavlm.extract_features(
            waveforms.float(),
            output_layer=WAVLM_MHFA_REPRESENTATION_LEVELS,
        )
        if len(layer_results) != WAVLM_MHFA_REPRESENTATION_LEVELS:
            raise ModelAdapterError(
                "WavLM produced an unexpected representation-level count: "
                f"{len(layer_results)}."
            )
        layer_representations = [
            representation.transpose(0, 1)
            for representation, _ in layer_results
        ]
        # [levels, batch, frames, dimension] -> [batch, dimension, frames, levels]
        stacked = torch.stack(layer_representations).permute(1, 3, 2, 0)
        embeddings = self.mhfa(stacked)
        expected_shape = (waveforms.shape[0], self.metadata.embedding_dim)
        if tuple(embeddings.shape) != expected_shape:
            raise ModelAdapterError(
                "WavLM+MHFA produced an unexpected embedding shape: "
                f"{tuple(embeddings.shape)}."
            )
        if normalize:
            embeddings = functional.normalize(embeddings, p=2, dim=1)
        return embeddings

    def set_encoder_trainable(self, trainable: bool) -> None:
        """Freeze or unfreeze both Transformer and MHFA parameters."""
        if not isinstance(trainable, bool):
            raise ModelAdapterError("trainable must be a boolean.")
        for parameter in self.parameters():
            parameter.requires_grad = trainable

    @property
    def parameter_count(self) -> int:
        """Return the audited WavLM plus MHFA parameter count."""
        return count_parameters(self.parameters())

    @property
    def trainable_parameter_count(self) -> int:
        """Return parameters currently enabled for gradient updates."""
        return count_parameters(self.parameters(), trainable_only=True)


def _validate_fixed_lengths(
    lengths: torch.Tensor | None,
    batch_size: int,
    model_device: torch.device,
) -> None:
    """Require full equal crops because the official path has no padding mask."""
    if lengths is None:
        return
    if lengths.ndim != 1 or lengths.shape[0] != batch_size:
        raise ModelAdapterError("WavLM+MHFA lengths must have shape [batch].")
    if lengths.device != model_device:
        raise ModelAdapterError(
            f"Lengths are on {lengths.device}, but WavLM+MHFA is on "
            f"{model_device}."
        )
    float_lengths = lengths.float()
    if not bool(torch.isfinite(float_lengths).all()) or not bool(
        torch.allclose(float_lengths, torch.ones_like(float_lengths))
    ):
        raise ModelAdapterError(
            "WavLM+MHFA requires equal fixed crops; every relative length "
            "must equal 1."
        )


def _verify_source_files(source_path: Path) -> None:
    """Authenticate every executable or licensing source artifact."""
    for relative_path, expected_hash in _SOURCE_HASHES.items():
        path = source_path / relative_path
        if not path.is_file():
            raise ModelAdapterError(
                f"Pinned WavLM+MHFA source is missing {relative_path!r}."
            )
        observed_hash = _sha256_file(path)
        if observed_hash != expected_hash:
            raise ModelAdapterError(
                f"WavLM+MHFA source hash mismatch for {relative_path!r}: "
                f"expected {expected_hash}, received {observed_hash}."
            )


def _download_checkpoint(destination: Path) -> None:
    """Download the official Google Drive checkpoint transactionally."""
    try:
        import gdown
    except ModuleNotFoundError as error:  # pragma: no cover - dependency gate.
        raise ModelAdapterError("gdown is required to download WavLM+MHFA.") from error

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_path = destination.with_suffix(destination.suffix + ".part")
    downloaded = gdown.download(
        id=WAVLM_MHFA_CHECKPOINT_FILE_ID,
        output=str(partial_path),
        quiet=False,
    )
    if downloaded is None or not partial_path.is_file():
        raise ModelAdapterError("WavLM+MHFA checkpoint download failed.")
    observed_hash = _sha256_file(partial_path)
    if observed_hash != WAVLM_MHFA_CHECKPOINT_SHA256:
        raise ModelAdapterError(
            "Downloaded WavLM+MHFA checkpoint SHA-256 mismatch: expected "
            f"{WAVLM_MHFA_CHECKPOINT_SHA256}, received {observed_hash}."
        )
    os.replace(partial_path, destination)


def _split_checkpoint_state(
    checkpoint: Any,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Validate and separate WavLM/MHFA tensors from the source classifier."""
    if not isinstance(checkpoint, Mapping) or len(checkpoint) != 259:
        raise ModelAdapterError(
            "WavLM+MHFA checkpoint must be the audited 259-entry state mapping."
        )
    if any(not isinstance(value, torch.Tensor) for value in checkpoint.values()):
        raise ModelAdapterError(
            "WavLM+MHFA checkpoint state must contain tensors only."
        )

    wavlm_prefix = "__S__.model."
    mhfa_prefix = "__S__.backend."
    loss_prefix = "__L__."
    unknown_keys = [
        str(key)
        for key in checkpoint
        if not str(key).startswith((wavlm_prefix, mhfa_prefix, loss_prefix))
    ]
    if unknown_keys:
        raise ModelAdapterError(
            f"WavLM+MHFA checkpoint has unknown keys: {unknown_keys[:5]}."
        )

    wavlm_state = {
        str(key)[len(wavlm_prefix) :]: value
        for key, value in checkpoint.items()
        if str(key).startswith(wavlm_prefix)
    }
    mhfa_state = {
        str(key)[len(mhfa_prefix) :]: value
        for key, value in checkpoint.items()
        if str(key).startswith(mhfa_prefix)
    }
    source_loss = {
        str(key): value
        for key, value in checkpoint.items()
        if str(key).startswith(loss_prefix)
    }
    if len(wavlm_state) != 248 or len(mhfa_state) != 10:
        raise ModelAdapterError(
            "WavLM+MHFA checkpoint component counts differ from the audit."
        )
    if set(source_loss) != {"__L__.weight"} or tuple(
        source_loss["__L__.weight"].shape
    ) != (7500, 256):
        raise ModelAdapterError(
            "WavLM+MHFA source loss tensor differs from the audited classifier."
        )
    return wavlm_state, mhfa_state


def _strict_load(
    module: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
    *,
    component: str,
) -> None:
    """Load one component and reject every incompatible state key."""
    incompatible = module.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ModelAdapterError(
            f"Strict {component} load failed: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}."
        )


def _load_verified_wavlm_module(source_path: Path) -> types.ModuleType:
    """Import the exact authenticated upstream WavLM implementation."""
    package_name = "_speaker_recognition_pinned_wavlm_bfb8527"
    wavlm_module_name = f"{package_name}.WavLM"
    existing = sys.modules.get(wavlm_module_name)
    if existing is not None:
        return existing

    baseline_path = source_path / "models" / "Baseline"
    package = types.ModuleType(package_name)
    package.__path__ = [str(baseline_path)]  # type: ignore[attr-defined]
    package.__package__ = package_name
    sys.modules[package_name] = package

    _execute_module(
        f"{package_name}.modules",
        baseline_path / "modules.py",
    )
    return _execute_module(
        wavlm_module_name,
        baseline_path / "WavLM.py",
    )


def _execute_module(module_name: str, path: Path) -> types.ModuleType:
    """Execute one already hashed Python file under an isolated package name."""
    specification = importlib.util.spec_from_file_location(module_name, path)
    if specification is None or specification.loader is None:
        raise ModelAdapterError(f"Cannot import authenticated source: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    try:
        specification.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _sha256_file(path: Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    """Hash a large source or checkpoint artifact incrementally."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_runtime_versions() -> None:
    """Require a gdown release compatible with the audited Kaggle runtime."""
    try:
        gdown_version = version("gdown")
    except PackageNotFoundError as error:  # pragma: no cover - dependency gate.
        raise ModelAdapterError("gdown is not installed.") from error
    components = gdown_version.split(".")
    if len(components) < 2 or components[:2] != ["5", "2"]:
        raise ModelAdapterError(
            "WavLM+MHFA requires audited gdown 5.2.x, received "
            f"{gdown_version}."
        )
