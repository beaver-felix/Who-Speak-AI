"""Run pinned RawNet3 inference and gradient checks on Kaggle GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from speaker_recognition.data.audio import load_audio_file
from speaker_recognition.data.segments import evenly_spaced_segments


def parse_arguments() -> argparse.Namespace:
    """Parse the real-speech input, cache, CUDA device, and output paths."""
    parser = argparse.ArgumentParser(
        description="Smoke-test the pinned differentiable RawNet3 adapter."
    )
    parser.add_argument(
        "--audio-file",
        required=True,
        type=Path,
        help="A real TidyVoice WAV file mounted in Kaggle.",
    )
    parser.add_argument(
        "--cache-dir",
        required=True,
        type=Path,
        help="Writable cache for the pinned Hugging Face checkpoint.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def write_json(payload: dict[str, Any], output_path: Path) -> None:
    """Write deterministic, human-readable smoke-test evidence."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    """Verify checkpoint identity, real-speech inference, and gradient flow."""
    arguments = parse_arguments()

    # Delayed imports preserve the lightweight local test environment.
    import torch

    from speaker_recognition.models.rawnet3 import (
        RAWNET3_ARCHITECTURE_REVISION,
        RAWNET3_CHECKPOINT_SHA256,
        RAWNET3_EVALUATION_SAMPLES,
        RAWNET3_TRAIN_SAMPLES,
        RawNet3Adapter,
    )

    if not torch.cuda.is_available() and arguments.device.startswith("cuda"):
        raise RuntimeError("CUDA was requested but is not available.")

    audio = load_audio_file(arguments.audio_file, target_sample_rate=16000)
    evaluation_segment = evenly_spaced_segments(
        audio.waveform,
        num_samples=RAWNET3_EVALUATION_SAMPLES,
        segment_count=1,
    )[0]
    training_segments = evenly_spaced_segments(
        audio.waveform,
        num_samples=RAWNET3_TRAIN_SAMPLES,
        segment_count=2,
    )
    if training_segments.shape[0] != 2:
        raise RuntimeError(
            "RawNet3 gradient smoke test requires a recording long enough "
            "to provide two distinct 48,240-sample crops."
        )

    adapter = RawNet3Adapter.from_pretrained(
        cache_dir=arguments.cache_dir,
        device=arguments.device,
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
    # Two distinct endpoint crops ensure the pooled BatchNorm receives real
    # between-example variation. Duplicating one crop can make the symmetric
    # loss's input gradient cancel exactly even though parameter gradients are
    # valid. The crop length is the official 300-frame recipe:
    # 300 * 160 + 240 = 48,240 waveform samples.
    gradient_waveform = (
        torch.from_numpy(training_segments)
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
    parameter_gradients = [
        parameter.grad
        for parameter in adapter.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]

    checks = {
        "checkpoint_sha256_matches": (
            adapter.checkpoint_sha256 == RAWNET3_CHECKPOINT_SHA256
        ),
        "embedding_shape": list(first.shape) == [1, 256],
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
        and all(
            bool(torch.isfinite(gradient).all())
            for gradient in parameter_gradients
        ),
        "encoder_parameter_gradient_nonzero": bool(parameter_gradients)
        and any(
            bool(torch.count_nonzero(gradient))
            for gradient in parameter_gradients
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"RawNet3 adapter smoke checks failed: {checks}")

    waveform_sha256 = hashlib.sha256(
        np.ascontiguousarray(audio.waveform).tobytes()
    ).hexdigest()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "model": {
            "name": adapter.metadata.name,
            "source_id": adapter.metadata.source_id,
            "checkpoint_revision": adapter.metadata.revision,
            "architecture_revision": RAWNET3_ARCHITECTURE_REVISION,
            "checkpoint_sha256": adapter.checkpoint_sha256,
            "embedding_dim": adapter.metadata.embedding_dim,
            "parameter_count": adapter.parameter_count,
            "trainable_parameter_count": adapter.trainable_parameter_count,
            "encoder_type": "ECA",
            "sinc_stride": 10,
        },
        "runtime": {
            "device": arguments.device,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "asteroid_filterbanks_version": version("asteroid-filterbanks"),
        },
        "sample": {
            "path": str(arguments.audio_file),
            "original_samples": int(audio.waveform.size),
            "duration_seconds": audio.duration_seconds,
            "waveform_sha256": waveform_sha256,
            "evaluation_crop_samples": RAWNET3_EVALUATION_SAMPLES,
            "gradient_crop_samples": RAWNET3_TRAIN_SAMPLES,
        },
        "result": {
            "embedding_shape": list(first.shape),
            "embedding_norm": float(norms.item()),
            "repeat_cosine_similarity": cosine,
            "checks": checks,
        },
    }
    write_json(payload, arguments.output)

    print("RAWNET3 ADAPTER GPU SMOKE TEST PASSED")
    print(f"checkpoint source: {adapter.metadata.source_id}")
    print(f"checkpoint revision: {adapter.metadata.revision}")
    print(f"architecture revision: {RAWNET3_ARCHITECTURE_REVISION}")
    print(f"checkpoint SHA-256: {adapter.checkpoint_sha256}")
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
