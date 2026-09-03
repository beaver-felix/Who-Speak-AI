"""Evaluate a RawNet3 checkpoint on enrollment/test verification pairs."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from dataset import VerificationPairDataset, build_dataloader, load_verification_pairs
from metrics import compute_verification_metrics, save_evaluation_csv, save_metrics
from model import build_model, extract_speaker_embedding


def parse_args() -> argparse.Namespace:
    """Parse evaluation command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    """Load and validate a YAML configuration mapping."""
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a YAML mapping")
    return config


def resolve_device(config: dict[str, Any], override: str | None) -> torch.device:
    """Resolve evaluation device, allowing explicit CPU smoke tests."""
    requested = override or config.get("runtime", {}).get("device", "cuda")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA is unavailable; evaluating on CPU")
        device = torch.device("cpu")
    return device


def load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Build a model from checkpoint metadata and load its state dictionary."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint.get("model_config", {})
    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint must contain model_state_dict")
    model = build_model(
        num_speakers=model_config.get("num_speakers"),
        embedding_dim=int(model_config.get("embedding_dim", 192)),
        base_channels=int(model_config.get("base_channels", 64)),
        am_scale=float(model_config.get("am_scale", 30.0)),
        am_margin=float(model_config.get("am_margin", 0.20)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


def resolve_output_path(config_path: Path, configured_path: str) -> Path:
    """Resolve an output path relative to its YAML configuration file."""
    output_path = Path(configured_path)
    if not output_path.is_absolute():
        output_path = config_path.resolve().parent / output_path
    return output_path


def main() -> None:
    """Compute verification scores and save EER/minDCF/accuracy metrics."""
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(config, args.device)
    dataset_config = config["dataset"]
    audio_config = config["audio"]
    runtime = config.get("runtime", {})
    pairs_path = dataset_config["test_pairs"]
    pairs = load_verification_pairs(
        pairs_path,
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
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(runtime.get("num_workers", 0)),
        pin_memory=bool(runtime.get("pin_memory", False)),
        persistent_workers=bool(runtime.get("persistent_workers", True)),
        prefetch_factor=int(runtime.get("prefetch_factor", 1)),
        drop_last=False,
    )
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = Path(config["outputs"]["checkpoint_directory"]) / "best_model.pt"
        if not checkpoint_path.is_absolute():
            checkpoint_path = args.config.resolve().parent / checkpoint_path
    model = load_checkpoint_model(checkpoint_path, device)
    scores: list[float] = []
    labels: list[int] = []
    start_time = time.perf_counter()
    with torch.inference_mode():
        for batch in pair_loader:
            enrollment = batch["enrollment_waveform"].to(device, non_blocking=True)
            test = batch["test_waveform"].to(device, non_blocking=True)
            enrollment_embedding = extract_speaker_embedding(model, enrollment)
            test_embedding = extract_speaker_embedding(model, test)
            similarity = torch.sum(enrollment_embedding * test_embedding, dim=-1)
            scores.extend(similarity.float().cpu().numpy().tolist())
            labels.extend(batch["label"].long().cpu().numpy().tolist())
    elapsed = time.perf_counter() - start_time

    metric_output = compute_verification_metrics(
        np.asarray(scores),
        np.asarray(labels),
        target_prior=float(config.get("evaluation", {}).get("target_prior", 0.01)),
    )
    metric_output.update(
        {
            "checkpoint": str(checkpoint_path),
            "device": str(device),
            "total_inference_seconds": elapsed,
            "latency_ms_per_trial": elapsed * 1000.0 / len(scores),
        }
    )
    output_path = args.output
    if output_path is None:
        output_path = resolve_output_path(
            args.config,
            config["outputs"]["metrics_json"],
        )
    saved_path = save_metrics(metric_output, output_path)
    csv_output_path = resolve_output_path(
        args.config,
        config["outputs"]["metrics_csv"],
    )
    saved_csv_path = save_evaluation_csv(
        metric_output,
        model_id=config["project"]["name"],
        output_path=csv_output_path,
    )
    print(f"EER={metric_output['eer']:.6f}")
    print(f"FAR={metric_output['far']:.6f}")
    print(f"FRR={metric_output['frr']:.6f}")
    print(f"minDCF={metric_output['min_dcf']:.6f}")
    print(f"accuracy={metric_output['accuracy']:.6f}")
    print(f"saved_metrics={saved_path}")
    print(f"saved_csv={saved_csv_path}")


if __name__ == "__main__":
    main()
