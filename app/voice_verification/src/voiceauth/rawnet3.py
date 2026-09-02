"""Authenticated, in-memory inference wrapper for the selected RawNet3 model."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from voiceauth.config import (
    EVALUATION_SAMPLES,
    SELECTED_CHECKPOINT_SHA256,
    SELECTED_CONFIG_SHA256,
    TARGET_SAMPLE_RATE,
)
from voiceauth.errors import ModelInitializationError
from voiceauth.matching import normalize_embedding


@dataclass(frozen=True)
class RawNet3Runtime:
    checkpoint_path: str
    cache_dir: str
    device: str = "cpu"


def sha256_file(path: Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


class RawNet3SpeakerEncoder:
    """Load the fine-tuned ViMD checkpoint behind the exact training contract."""

    def __init__(self, runtime: RawNet3Runtime) -> None:
        self.runtime = runtime
        self._adapter = None
        self._torch = None

    def prepare(self) -> None:
        if self._adapter is not None:
            return
        checkpoint = Path(self.runtime.checkpoint_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise ModelInitializationError(
                "RawNet3 best.pt was not found. Set VOICE_MODEL_CHECKPOINT to the verified artifact."
            )
        if sha256_file(checkpoint) != SELECTED_CHECKPOINT_SHA256:
            raise ModelInitializationError("RawNet3 best.pt SHA-256 does not match the selected artifact.")
        try:
            import torch
            import torch.nn.functional as functional
            from speaker_recognition.data.audio import canonicalize_audio
            from speaker_recognition.data.segments import evenly_spaced_segments
            from speaker_recognition.models.rawnet3 import RawNet3Adapter
        except ImportError as error:
            raise ModelInitializationError(
                "Install PyTorch and model/Thanh2 with `python -m pip install -e '.[data,rawnet3]'`."
            ) from error
        try:
            adapter = RawNet3Adapter.from_pretrained(
                cache_dir=self.runtime.cache_dir,
                device=self.runtime.device,
            )
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if not isinstance(payload, Mapping):
                raise ValueError("checkpoint root is not a mapping")
            expected_identity = {
                "model_name": "rawnet3",
                "dataset_name": "vimd",
                "config_sha256": SELECTED_CONFIG_SHA256,
                "manifest_sha256": "ed7b764c6aaab2ba2c4ec95edadab19fd640ebca72aa06da3d36cbf93fc4747f",
                "seed": 42,
            }
            if payload.get("identity") != expected_identity:
                raise ValueError("checkpoint experiment identity does not match")
            state = payload.get("adapter_state")
            if not isinstance(state, Mapping):
                raise ValueError("checkpoint has no adapter_state mapping")
            adapter.load_state_dict(state, strict=True)
            adapter.to(self.runtime.device)
            adapter.eval()
            adapter.requires_grad_(False)
        except Exception as error:
            raise ModelInitializationError("The selected RawNet3 checkpoint could not be loaded safely.") from error
        self._adapter = adapter
        self._torch = torch
        self._functional = functional
        self._canonicalize_audio = canonicalize_audio
        self._segments = evenly_spaced_segments

    def embed(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        self.prepare()
        assert self._adapter is not None and self._torch is not None
        canonical = self._canonicalize_audio(
            waveform,
            sample_rate=int(sample_rate),
            target_sample_rate=TARGET_SAMPLE_RATE,
        )
        segments = self._segments(
            canonical,
            num_samples=EVALUATION_SAMPLES,
            segment_count=1,
        )
        batch = self._torch.from_numpy(segments).to(self.runtime.device)
        with self._torch.inference_mode():
            embeddings = self._adapter(batch)
            pooled = self._functional.normalize(
                embeddings.float().mean(dim=0, keepdim=True), p=2, dim=1
            )
        return normalize_embedding(pooled.squeeze(0).cpu().numpy())
