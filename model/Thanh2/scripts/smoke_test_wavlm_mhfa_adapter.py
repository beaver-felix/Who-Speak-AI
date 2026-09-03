"""Run the pinned fine-tuned WavLM+MHFA gate on a Kaggle GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from speaker_recognition.data.audio import load_audio_file
from speaker_recognition.data.segments import evenly_spaced_segments


def parse_arguments() -> argparse.Namespace:
    """Parse real speech, artifact cache, optional checkpoint, and output."""
    parser = argparse.ArgumentParser(
        description="Smoke-test the pinned fine-tuned WavLM+MHFA adapter."
    )
    parser.add_argument("--audio-file", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def write_json(payload: dict[str, Any], output_path: Path) -> None:
    """Write stable structured compatibility evidence."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def gradients_for(parameters: Iterable[Any]) -> list[Any]:
    """Collect existing gradients from enabled parameters."""
    return [
        parameter.grad
        for parameter in parameters
        if parameter.requires_grad and parameter.grad is not None
    ]


def gradients_are_finite(gradients: list[Any], torch: Any) -> bool:
    """Return whether a non-empty gradient group contains only finite values."""
    return bool(gradients) and all(
        bool(torch.isfinite(gradient).all()) for gradient in gradients
    )


def gradients_are_nonzero(gradients: list[Any], torch: Any) -> bool:
    """Return whether any gradient tensor contains a non-zero value."""
    return bool(gradients) and any(
        bool(torch.count_nonzero(gradient)) for gradient in gradients
    )


def main() -> None:
    """Verify strict pretrained inference and the official gradient boundary."""
    arguments = parse_arguments()

    # Delayed imports preserve lightweight local tests without PyTorch.
    import torch

    from speaker_recognition.models.wavlm_mhfa import (
        WAVLM_MHFA_CHECKPOINT_SHA256,
        WAVLM_MHFA_EVALUATION_SAMPLES,
        WAVLM_MHFA_TRAIN_SAMPLES,
        WavlmMhfaAdapter,
    )

    if not torch.cuda.is_available() and arguments.device.startswith("cuda"):
        raise RuntimeError("CUDA was requested but is not available.")

    audio = load_audio_file(arguments.audio_file, target_sample_rate=16000)
    evaluation_segment = evenly_spaced_segments(
        audio.waveform,
        num_samples=WAVLM_MHFA_EVALUATION_SAMPLES,
        segment_count=1,
    )[0]
    gradient_segment = evenly_spaced_segments(
        audio.waveform,
        num_samples=WAVLM_MHFA_TRAIN_SAMPLES,
        segment_count=1,
    )[0]

    adapter = WavlmMhfaAdapter.from_pretrained(
        cache_dir=arguments.cache_dir,
        device=arguments.device,
        checkpoint_path=arguments.checkpoint,
    )
    evaluation_waveform = torch.from_numpy(evaluation_segment).unsqueeze(0).to(
        arguments.device
    )

    adapter.eval()
    with torch.no_grad():
        first = adapter(evaluation_waveform)
        repeated = adapter(evaluation_waveform)
    cosine = torch.nn.functional.cosine_similarity(first, repeated).item()
    norms = torch.linalg.vector_norm(first, ord=2, dim=1)

    adapter.train()
    adapter.zero_grad(set_to_none=True)
    if arguments.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(arguments.device)
    gradient_waveform = (
        torch.from_numpy(gradient_segment)
        .unsqueeze(0)
        .to(arguments.device)
        .requires_grad_(True)
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

    transformer_gradients = gradients_for(adapter.wavlm.encoder.layers.parameters())
    mhfa_gradients = gradients_for(adapter.mhfa.parameters())
    feature_gradients = gradients_for(adapter.wavlm.feature_extractor.parameters())
    peak_memory_bytes = (
        int(torch.cuda.max_memory_allocated(arguments.device))
        if arguments.device.startswith("cuda")
        else None
    )

    checks = {
        "checkpoint_sha256_matches": (
            adapter.checkpoint_sha256 == WAVLM_MHFA_CHECKPOINT_SHA256
        ),
        "embedding_shape": list(first.shape) == [1, 256],
        "embedding_finite": bool(torch.isfinite(first).all()),
        "embedding_l2_normalized": bool(
            torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
        ),
        "repeat_cosine_at_least_0_99999": cosine >= 0.99999,
        "transformer_gradient_present": bool(transformer_gradients),
        "transformer_gradients_finite": gradients_are_finite(
            transformer_gradients, torch
        ),
        "transformer_gradient_nonzero": gradients_are_nonzero(
            transformer_gradients, torch
        ),
        "mhfa_gradient_present": bool(mhfa_gradients),
        "mhfa_gradients_finite": gradients_are_finite(mhfa_gradients, torch),
        "mhfa_gradient_nonzero": gradients_are_nonzero(mhfa_gradients, torch),
        # This is the official architecture boundary: WavLM.py wraps its
        # convolutional feature extractor in torch.no_grad().
        "feature_extractor_gradient_absent_as_designed": not feature_gradients,
        "waveform_gradient_absent_as_designed": gradient_waveform.grad is None,
    }
    if not all(checks.values()):
        raise RuntimeError(f"WavLM+MHFA adapter smoke checks failed: {checks}")

    waveform_sha256 = hashlib.sha256(
        np.ascontiguousarray(audio.waveform).tobytes()
    ).hexdigest()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "model": {
            "name": adapter.metadata.name,
            "source_id": adapter.metadata.source_id,
            "revision": adapter.metadata.revision,
            "checkpoint_filename": adapter.checkpoint_path.name,
            "checkpoint_sha256": adapter.checkpoint_sha256,
            "embedding_dim": adapter.metadata.embedding_dim,
            "parameter_count": adapter.parameter_count,
            "trainable_parameter_count": adapter.trainable_parameter_count,
            "source_classifier_included": False,
        },
        "runtime": {
            "device": arguments.device,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "peak_gradient_gate_memory_bytes": peak_memory_bytes,
        },
        "sample": {
            "path": str(arguments.audio_file),
            "original_samples": int(audio.waveform.size),
            "duration_seconds": audio.duration_seconds,
            "waveform_sha256": waveform_sha256,
            "evaluation_crop_samples": WAVLM_MHFA_EVALUATION_SAMPLES,
            "gradient_crop_samples": WAVLM_MHFA_TRAIN_SAMPLES,
        },
        "gradient_policy": {
            "optimized_components": ["wavlm_transformer_layers", "mhfa"],
            "feature_extractor": "official_no_grad_boundary",
        },
        "result": {
            "embedding_shape": list(first.shape),
            "embedding_norm": float(norms.item()),
            "repeat_cosine_similarity": cosine,
            "checks": checks,
        },
    }
    write_json(payload, arguments.output)

    print("WAVLM+MHFA ADAPTER GPU SMOKE TEST PASSED")
    print(f"source: {adapter.metadata.source_id}")
    print(f"revision: {adapter.metadata.revision}")
    print(f"checkpoint SHA-256: {adapter.checkpoint_sha256}")
    print(f"parameters: {adapter.parameter_count:,}")
    print(f"embedding shape: {tuple(first.shape)}")
    print(f"repeat cosine similarity: {cosine:.8f}")
    print(f"Transformer gradient present: {bool(transformer_gradients)}")
    print(f"MHFA gradient present: {bool(mhfa_gradients)}")
    print(
        "feature extractor gradient absent by official design: "
        f"{not feature_gradients}"
    )
    if peak_memory_bytes is not None:
        print(f"peak gate memory: {peak_memory_bytes / 1024**3:.3f} GiB")
    print(f"output: {arguments.output.resolve()}")


if __name__ == "__main__":
    main()
