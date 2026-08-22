"""Run pinned ECAPA adapter inference and gradient checks on Kaggle GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from speaker_recognition.data.audio import load_audio_file


def parse_arguments() -> argparse.Namespace:
    """Parse artifact cache, CUDA device, and structured output path."""
    parser = argparse.ArgumentParser(
        description="Smoke-test the pinned differentiable ECAPA adapter."
    )
    parser.add_argument(
        "--cache-dir",
        required=True,
        type=Path,
        help="Writable cache for Hugging Face and SpeechBrain artifacts.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def write_json(payload: dict[str, Any], output_path: Path) -> None:
    """Write stable smoke-test evidence."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    """Load pinned artifacts and verify inference plus gradient flow."""
    arguments = parse_arguments()

    # Delayed imports keep local data/config tests independent of PyTorch.
    import torch

    from speaker_recognition.models.ecapa_tdnn import EcapaTdnnAdapter

    if not torch.cuda.is_available() and arguments.device.startswith("cuda"):
        raise RuntimeError("CUDA was requested but is not available.")

    adapter = EcapaTdnnAdapter.from_pretrained(
        cache_dir=arguments.cache_dir,
        device=arguments.device,
    )
    sample_path = adapter.source_path / "example1.wav"
    audio = load_audio_file(sample_path, target_sample_rate=16000)
    waveform = torch.from_numpy(audio.waveform).unsqueeze(0).to(arguments.device)

    adapter.eval()
    with torch.no_grad():
        first = adapter(waveform)
        repeated = adapter(waveform)
    cosine = torch.nn.functional.cosine_similarity(first, repeated).item()
    norms = torch.linalg.vector_norm(first, ord=2, dim=1)

    adapter.train()
    adapter.zero_grad(set_to_none=True)
    # Two examples keep ECAPA's final BatchNorm valid in training mode; a
    # single pooled example would provide only one value per channel.
    gradient_waveform = (
        waveform.detach().clone().repeat(2, 1).requires_grad_(True)
    )
    train_embedding = adapter(gradient_waveform)
    weights = torch.linspace(
        0.5,
        1.5,
        train_embedding.shape[1],
        device=train_embedding.device,
    )
    loss = (train_embedding * weights).sum()
    loss.backward()
    parameter_gradients = [
        parameter.grad
        for parameter in adapter.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]

    checks = {
        "embedding_shape": list(first.shape) == [1, 192],
        "embedding_finite": bool(torch.isfinite(first).all()),
        "embedding_l2_normalized": bool(
            torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
        ),
        "repeat_cosine_at_least_0_99999": cosine >= 0.99999,
        "input_gradient_finite": (
            gradient_waveform.grad is not None
            and bool(torch.isfinite(gradient_waveform.grad).all())
        ),
        "input_gradient_nonzero": (
            gradient_waveform.grad is not None
            and bool(torch.count_nonzero(gradient_waveform.grad))
        ),
        "encoder_parameter_gradient_present": bool(parameter_gradients),
        "encoder_parameter_gradients_finite": bool(parameter_gradients)
        and all(bool(torch.isfinite(gradient).all()) for gradient in parameter_gradients),
        "encoder_parameter_gradient_nonzero": bool(parameter_gradients)
        and any(bool(torch.count_nonzero(gradient)) for gradient in parameter_gradients),
    }
    if not all(checks.values()):
        raise RuntimeError(f"ECAPA adapter smoke checks failed: {checks}")

    waveform_sha256 = hashlib.sha256(
        np.ascontiguousarray(audio.waveform).tobytes()
    ).hexdigest()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "model": {
            "name": adapter.metadata.name,
            "source_id": adapter.metadata.source_id,
            "revision": adapter.metadata.revision,
            "embedding_dim": adapter.metadata.embedding_dim,
            "parameter_count": adapter.parameter_count,
            "trainable_parameter_count": adapter.trainable_parameter_count,
        },
        "runtime": {
            "device": arguments.device,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        "sample": {
            "path": sample_path.name,
            "samples": int(audio.waveform.size),
            "duration_seconds": audio.duration_seconds,
            "waveform_sha256": waveform_sha256,
        },
        "result": {
            "embedding_shape": list(first.shape),
            "embedding_norm": float(norms.item()),
            "repeat_cosine_similarity": cosine,
            "checks": checks,
        },
    }
    write_json(payload, arguments.output)

    print("ECAPA ADAPTER GPU SMOKE TEST PASSED")
    print(f"source: {adapter.metadata.source_id}")
    print(f"revision: {adapter.metadata.revision}")
    print(f"parameters: {adapter.parameter_count:,}")
    print(f"embedding shape: {tuple(first.shape)}")
    print(f"repeat cosine similarity: {cosine:.8f}")
    print(f"input gradient finite: {checks['input_gradient_finite']}")
    print(
        "encoder parameter gradient present: "
        f"{checks['encoder_parameter_gradient_present']}"
    )
    print(f"output: {arguments.output.resolve()}")


if __name__ == "__main__":
    main()
