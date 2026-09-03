"""Evaluate a fine-tuned WavLM model on speaker-verification pairs."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

THANH_ROOT = Path(__file__).resolve().parent.parent
if str(THANH_ROOT) not in sys.path:
    sys.path.insert(0, str(THANH_ROOT))

from RawNet3.dataset import (  # noqa: E402
    VerificationPairDataset,
    build_dataloader,
    load_verification_pairs,
)
from WavLM.metrics import (  # noqa: E402
    compute_verification_metrics,
    save_evaluation_csv,
    save_metrics,
)
from WavLM.model import build_model, extract_speaker_embedding  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse WavLM evaluation command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML configuration mapping."""
    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a YAML mapping")
    return config


def resolve_device(config: dict[str, Any], override: str | None) -> torch.device:
    """Select CUDA when available and otherwise permit CPU evaluation."""
    requested = override or config.get("runtime", {}).get("device", "cuda")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA is unavailable; evaluating on CPU")
        return torch.device("cpu")
    return device


def resolve_checkpoint_path(
    config: dict[str, Any], config_path: Path, override: Path | None
) -> Path:
    """Resolve the checkpoint path relative to the YAML file."""
    if override is not None:
        return override.expanduser().resolve()
    directory = Path(config["outputs"]["checkpoint_directory"])
    if not directory.is_absolute():
        directory = config_path.resolve().parent / directory
    return directory / "best_model.pt"


def resolve_output_path(config_path: Path, configured_path: str) -> Path:
    """Resolve an output path relative to its YAML configuration file."""
    output_path = Path(configured_path)
    if not output_path.is_absolute():
        output_path = config_path.resolve().parent / output_path
    return output_path


def load_checkpoint_model(path: Path, device: torch.device) -> torch.nn.Module:
    """Restore a WavLM wrapper using constructor metadata in the checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    if "model_state_dict" not in checkpoint or "model_config" not in checkpoint:
        raise ValueError("Checkpoint lacks model_state_dict or model_config")
    model = build_model(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


def main() -> None:
    """Score all verification trials and write structured JSON metrics."""
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(config, args.device)
    checkpoint_path = resolve_checkpoint_path(config, args.config, args.checkpoint)
    model = load_checkpoint_model(checkpoint_path, device)

    dataset_config = config["dataset"]
    audio_config = config["audio"]
    runtime = config.get("runtime", {})
    training = config["training"]
    pairs = load_verification_pairs(
        dataset_config["test_pairs"],
        dataset_config.get("enrollment_path_column", "enrollment_path"),
        dataset_config.get("test_path_column", "test_path"),
        dataset_config.get("label_column", "label"),
    )
    pair_dataset = VerificationPairDataset(
        pairs,
        sample_rate=int(audio_config["sample_rate"]),
        duration_seconds=float(audio_config["duration_seconds"]),
        peak_normalize=bool(audio_config.get("peak_normalize", False)),
    )
    pair_loader = build_dataloader(
        pair_dataset,
        batch_size=int(training.get("evaluation_batch_size", training["batch_size"])),
        shuffle=False,
        num_workers=int(runtime.get("num_workers", 0)),
        pin_memory=bool(runtime.get("pin_memory", False)),
        persistent_workers=bool(runtime.get("persistent_workers", True)),
        prefetch_factor=int(runtime.get("prefetch_factor", 1)),
        drop_last=False,
    )

    precision = str(training.get("mixed_precision", "fp16")).lower()
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    amp_enabled = bool(training.get("use_amp", True)) and device.type == "cuda"
    scores: list[float] = []
    labels: list[int] = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in pair_loader:
            enrollment = batch["enrollment_waveform"].to(device, non_blocking=True)
            test = batch["test_waveform"].to(device, non_blocking=True)
            enrollment_mask = batch["enrollment_attention_mask"].to(
                device, non_blocking=True
            )
            test_mask = batch["test_attention_mask"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
                enrollment_embedding = extract_speaker_embedding(
                    model, enrollment, enrollment_mask
                )
                test_embedding = extract_speaker_embedding(model, test, test_mask)
                similarity = torch.sum(enrollment_embedding * test_embedding, dim=-1)
            scores.extend(similarity.float().cpu().tolist())
            labels.extend(batch["label"].long().cpu().tolist())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    metrics = compute_verification_metrics(
        np.asarray(scores),
        np.asarray(labels),
        target_prior=float(config.get("evaluation", {}).get("target_prior", 0.01)),
    )
    metrics.update(
        {
            "model_type": "wavlm",
            "pretrained_model_name": model.pretrained_model_name,
            "pooling": model.pooling,
            "checkpoint": str(checkpoint_path),
            "device": str(device),
            "total_inference_seconds": elapsed,
            "latency_ms_per_trial": elapsed * 1000.0 / len(scores),
        }
    )
    output_path = args.output or resolve_output_path(
        args.config,
        config["outputs"]["metrics_json"],
    )
    saved_path = save_metrics(metrics, output_path)
    csv_output_path = resolve_output_path(
        args.config,
        config["outputs"]["metrics_csv"],
    )
    saved_csv_path = save_evaluation_csv(
        metrics,
        model_id=config["project"]["name"],
        output_path=csv_output_path,
    )
    print(f"EER={metrics['eer']:.6f}")
    print(f"FAR={metrics['far']:.6f}")
    print(f"FRR={metrics['frr']:.6f}")
    print(f"minDCF={metrics['min_dcf']:.6f}")
    print(f"accuracy={metrics['accuracy']:.6f}")
    print(f"saved_metrics={saved_path}")
    print(f"saved_csv={saved_csv_path}")


if __name__ == "__main__":
    main()
