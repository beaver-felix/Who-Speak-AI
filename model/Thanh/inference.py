"""Unified speaker-embedding inference for the virtual assistant backend."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as torch_functional

THANH_ROOT = Path(__file__).resolve().parent
if str(THANH_ROOT) not in sys.path:
    sys.path.insert(0, str(THANH_ROOT))

from RawNet3.model import (  # noqa: E402
    build_model as build_rawnet3,
    extract_speaker_embedding as extract_rawnet3_embedding,
)


def _load_audio(path: Path, target_sample_rate: int) -> torch.Tensor:
    """Decode and resample an audio file into a mono float32 tensor."""
    if not path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {path}")
    try:
        import torchaudio

        waveform, sample_rate = torchaudio.load(str(path))
        if sample_rate != target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, target_sample_rate
            )
    except (ImportError, RuntimeError):
        try:
            import soundfile as sound_file
        except ImportError as exc:
            raise RuntimeError(
                "Install torchaudio or soundfile to decode audio"
            ) from exc
        samples, sample_rate = sound_file.read(
            str(path), dtype="float32", always_2d=True
        )
        waveform = torch.from_numpy(samples.T.copy())
        if sample_rate != target_sample_rate:
            new_length = round(waveform.shape[-1] * target_sample_rate / sample_rate)
            waveform = torch_functional.interpolate(
                waveform.unsqueeze(0),
                size=new_length,
                mode="linear",
                align_corners=False,
            ).squeeze(0)

    waveform = waveform.to(dtype=torch.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)
    waveform = torch.nan_to_num(waveform.reshape(-1))
    if waveform.numel() == 0:
        raise ValueError(f"Audio file is empty: {path}")
    return waveform


def _prepare_audio(
    path: Path,
    sample_rate: int,
    duration_seconds: float,
    peak_normalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Center-crop or right-pad audio and return its valid-sample mask."""
    target_length = round(sample_rate * duration_seconds)
    if sample_rate <= 0 or target_length <= 0:
        raise ValueError("sample rate and duration must be positive")
    waveform = _load_audio(path, sample_rate)
    valid_length = min(waveform.numel(), target_length)
    if waveform.numel() > target_length:
        start = (waveform.numel() - target_length) // 2
        waveform = waveform[start : start + target_length]
    elif waveform.numel() < target_length:
        waveform = torch_functional.pad(
            waveform, (0, target_length - waveform.numel())
        )
    if peak_normalize:
        peak = waveform.abs().amax()
        if peak > 1e-8:
            waveform = waveform / peak
    attention_mask = torch.arange(target_length) < valid_length
    return waveform.contiguous(), attention_mask


def _load_torch_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    """Load a project checkpoint while remaining compatible with older PyTorch."""
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint must contain a model_state_dict")
    return checkpoint


@lru_cache(maxsize=2)
def _load_model_cached(
    model_path: str,
    model_type: str,
    device_name: str,
) -> tuple[torch.nn.Module, int, float, bool]:
    """Load and cache a model plus its audio preprocessing settings."""
    device = torch.device(device_name)
    normalized_type = model_type.lower()
    checkpoint_path = Path(model_path).expanduser()
    checkpoint: dict[str, Any] | None = None
    if checkpoint_path.is_file():
        checkpoint = _load_torch_checkpoint(checkpoint_path.resolve(), device)

    if normalized_type == "rawnet3":
        if checkpoint is None:
            raise FileNotFoundError(
                "RawNet3 model_path must point to a local best_model.pt checkpoint"
            )
        model_config = checkpoint.get("model_config")
        if not isinstance(model_config, dict):
            raise ValueError("RawNet3 checkpoint lacks model_config")
        model = build_rawnet3(**model_config)
    elif normalized_type in {"wavlm", "wavlm_ssl_sv"}:
        # Keep Transformers optional for deployments that use RawNet3 only.
        from WavLM.model import build_model as build_wavlm

        if checkpoint is None:
            # A Hugging Face model ID or local from_pretrained directory is valid.
            model = build_wavlm(
                pretrained_model_name=model_path,
                num_speakers=None,
                pooling="xvector",
                embedding_dim=512,
                freeze_backbone=True,
            )
        else:
            model_config = checkpoint.get("model_config")
            if not isinstance(model_config, dict):
                raise ValueError("WavLM checkpoint lacks model_config")
            model = build_wavlm(**model_config)
    else:
        raise ValueError("model_type must be 'rawnet3' or 'wavlm'")

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        run_config = checkpoint.get("config", {})
    else:
        run_config = {}
    audio_config = run_config.get("audio", {}) if isinstance(run_config, dict) else {}
    sample_rate = int(audio_config.get("sample_rate", 16_000))
    duration_seconds = float(audio_config.get("duration_seconds", 3.0))
    peak_normalize = bool(audio_config.get("peak_normalize", False))
    model = model.to(device).eval()
    return model, sample_rate, duration_seconds, peak_normalize


def extract_embedding(
    audio_path: str,
    model_path: str,
    model_type: str = "rawnet3",
) -> np.ndarray:
    """Extract an L2-normalized speaker embedding vector from an audio file.

    Args:
        audio_path: Path to a supported audio file.
        model_path: Local project checkpoint. For WavLM, this may alternatively
            be a Hugging Face model ID such as ``microsoft/wavlm-base-plus-sv``.
        model_type: Either ``rawnet3`` or ``wavlm``.

    Returns:
        A one-dimensional float32 NumPy embedding with unit L2 norm.
    """
    normalized_type = model_type.lower()
    if normalized_type not in {"rawnet3", "wavlm", "wavlm_ssl_sv"}:
        raise ValueError("model_type must be 'rawnet3' or 'wavlm'")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path_candidate = Path(model_path).expanduser()
    cache_path = str(path_candidate.resolve()) if path_candidate.exists() else model_path
    model, sample_rate, duration_seconds, peak_normalize = _load_model_cached(
        cache_path, normalized_type, str(device)
    )
    waveform, attention_mask = _prepare_audio(
        Path(audio_path).expanduser(),
        sample_rate,
        duration_seconds,
        peak_normalize,
    )

    with torch.inference_mode():
        if normalized_type == "rawnet3":
            embedding = extract_rawnet3_embedding(model, waveform)
        else:
            from WavLM.model import (
                extract_speaker_embedding as extract_wavlm_embedding,
            )

            embedding = extract_wavlm_embedding(model, waveform, attention_mask)
    result = embedding.detach().float().cpu().numpy().reshape(-1).astype(np.float32)
    norm = float(np.linalg.norm(result))
    if not np.isfinite(norm) or norm <= 0:
        raise RuntimeError("The model produced an invalid zero/non-finite embedding")
    return result / norm


def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between two finite embedding vectors."""
    first = np.asarray(emb1, dtype=np.float32).reshape(-1)
    second = np.asarray(emb2, dtype=np.float32).reshape(-1)
    if first.shape != second.shape or first.size == 0:
        raise ValueError("embeddings must be non-empty vectors with equal shape")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("embeddings must contain only finite values")
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 0:
        raise ValueError("cosine similarity is undefined for a zero vector")
    similarity = float(np.dot(first, second) / denominator)
    return float(np.clip(similarity, -1.0, 1.0))


def clear_model_cache() -> None:
    """Release cached model references, for example before switching devices."""
    _load_model_cached.cache_clear()
