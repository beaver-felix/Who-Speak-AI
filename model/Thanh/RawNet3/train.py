"""Kaggle-friendly supervised training entry point for the RawNet3 encoder."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

from dataset import SpeakerAudioDataset, build_dataloader, load_speaker_records
from model import build_model


def parse_args() -> argparse.Namespace:
    """Parse command-line options for a training run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--device", type=str, default=None, help="Override config device")
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    """Load a YAML configuration file into a dictionary."""
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a YAML mapping")
    return config


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible data ordering."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(config: dict[str, Any], override: str | None) -> torch.device:
    """Resolve the execution device and enforce the Kaggle GPU requirement."""
    runtime = config.get("runtime", {})
    requested = override or runtime.get("device", "cuda")
    device = torch.device(requested)
    require_cuda = bool(runtime.get("require_cuda", False)) and override != "cpu"
    if require_cuda and device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by config, but no GPU is available")
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU for a debug run")
        device = torch.device("cpu")
    return device


def make_amp_settings(config: dict[str, Any], device: torch.device) -> tuple[bool, torch.dtype]:
    """Select CUDA AMP settings without enabling AMP on CPU."""
    training = config.get("training", {})
    enabled = bool(training.get("use_amp", True)) and device.type == "cuda"
    precision = str(training.get("mixed_precision", "fp16")).lower()
    if precision not in {"fp16", "bf16"}:
        raise ValueError("mixed_precision must be either 'fp16' or 'bf16'")
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return enabled, dtype


def build_datasets(config: dict[str, Any]) -> tuple[SpeakerAudioDataset, SpeakerAudioDataset | None]:
    """Create training and known-speaker validation datasets."""
    dataset_config = config["dataset"]
    audio_config = config["audio"]
    train_records = load_speaker_records(
        dataset_config["train_manifest"],
        dataset_config.get("path_column", "path"),
        dataset_config.get("speaker_column", "speaker_id"),
    )
    train_dataset = SpeakerAudioDataset(
        train_records,
        sample_rate=int(audio_config["sample_rate"]),
        duration_seconds=float(audio_config["duration_seconds"]),
        training=True,
        peak_normalize=bool(audio_config.get("peak_normalize", False)),
    )

    validation_path = dataset_config.get("validation_manifest")
    if not validation_path:
        return train_dataset, None
    validation_records = load_speaker_records(
        validation_path,
        dataset_config.get("path_column", "path"),
        dataset_config.get("speaker_column", "speaker_id"),
    )
    known_records = [
        record for record in validation_records if record.speaker_id in train_dataset.speaker_to_index
    ]
    if not known_records:
        print("No validation speakers overlap the training speaker set; validation disabled")
        return train_dataset, None
    validation_dataset = SpeakerAudioDataset(
        known_records,
        sample_rate=int(audio_config["sample_rate"]),
        duration_seconds=float(audio_config["duration_seconds"]),
        training=False,
        peak_normalize=bool(audio_config.get("peak_normalize", False)),
    )
    # Validation labels must use the classifier's training-speaker index map.
    validation_dataset.speaker_to_index = train_dataset.speaker_to_index
    return train_dataset, validation_dataset


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    gradient_accumulation_steps: int = 1,
    gradient_clip_norm: float = 5.0,
) -> dict[str, float]:
    """Run one training or validation epoch and return aggregate statistics."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    if training:
        optimizer.zero_grad(set_to_none=True)

    for batch_index, batch in enumerate(loader):
        waveform = batch["waveform"].to(device, non_blocking=True)
        labels = batch["speaker_index"].to(device, non_blocking=True).long()
        with torch.set_grad_enabled(training):
            with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
                logits = model(waveform, labels)
                loss = criterion(logits, labels)
            if training:
                scaled_loss = loss / gradient_accumulation_steps
                scaler.scale(scaled_loss).backward()
                should_step = (
                    (batch_index + 1) % gradient_accumulation_steps == 0
                    or batch_index + 1 == len(loader)
                )
                if should_step:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

        batch_size = labels.shape[0]
        total_loss += float(loss.detach()) * batch_size
        total_correct += int((logits.detach().argmax(dim=1) == labels).sum())
        total_examples += batch_size

    if total_examples == 0:
        raise RuntimeError("DataLoader produced no examples")
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    config: dict[str, Any],
    speaker_to_index: dict[str, int],
) -> None:
    """Save model, optimizer, configuration, and speaker-index metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    model_config = {
        "num_speakers": model.num_speakers,
        "embedding_dim": model.embedding_dim,
        "base_channels": model.frontend[0].out_channels,
        "am_scale": model.classifier.scale if model.classifier else 30.0,
        "am_margin": model.classifier.margin if model.classifier else 0.20,
    }
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": config,
            "model_config": model_config,
            "speaker_to_index": speaker_to_index,
        },
        path,
    )


def main() -> None:
    """Run the configured RawNet3 training job."""
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("project", {}).get("seed", 42)))
    device = resolve_device(config, args.device)
    amp_enabled, amp_dtype = make_amp_settings(config, device)
    train_dataset, validation_dataset = build_datasets(config)
    runtime = config.get("runtime", {})
    training_config = config["training"]

    train_loader = build_dataloader(
        train_dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=True,
        num_workers=int(runtime.get("num_workers", 0)),
        pin_memory=bool(runtime.get("pin_memory", False)),
        persistent_workers=bool(runtime.get("persistent_workers", True)),
        prefetch_factor=int(runtime.get("prefetch_factor", 1)),
        drop_last=bool(runtime.get("drop_last", False)),
    )
    validation_loader = None
    if validation_dataset is not None:
        validation_loader = build_dataloader(
            validation_dataset,
            batch_size=int(training_config["batch_size"]),
            shuffle=False,
            num_workers=int(runtime.get("num_workers", 0)),
            pin_memory=bool(runtime.get("pin_memory", False)),
            persistent_workers=bool(runtime.get("persistent_workers", True)),
            prefetch_factor=int(runtime.get("prefetch_factor", 1)),
            drop_last=False,
        )

    model_config = config.get("model", {})
    model = build_model(
        num_speakers=len(train_dataset.speaker_to_index),
        embedding_dim=int(model_config.get("embedding_dim", 192)),
        base_channels=int(model_config.get("base_channels", 64)),
        am_scale=float(model_config.get("am_scale", 30.0)),
        am_margin=float(model_config.get("am_margin", 0.20)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    criterion = nn.CrossEntropyLoss()
    start_epoch = 0
    best_metric = float("inf")
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_metric = float(checkpoint.get("metrics", {}).get("loss", best_metric))

    checkpoint_directory = Path(config["outputs"]["checkpoint_directory"])
    if not checkpoint_directory.is_absolute():
        checkpoint_directory = args.config.resolve().parent / checkpoint_directory
    checkpoint_path = checkpoint_directory / "best_model.pt"
    for epoch in range(start_epoch, int(training_config["epochs"])):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            amp_enabled,
            amp_dtype,
            int(training_config.get("gradient_accumulation_steps", 1)),
            float(training_config.get("gradient_clip_norm", 5.0)),
        )
        validation_metrics = (
            run_epoch(
                model,
                validation_loader,
                criterion,
                None,
                scaler,
                device,
                amp_enabled,
                amp_dtype,
            )
            if validation_loader is not None
            else train_metrics
        )
        print(
            f"epoch={epoch + 1}/{training_config['epochs']} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={validation_metrics['loss']:.4f} "
            f"val_acc={validation_metrics['accuracy']:.4f}"
        )
        if validation_metrics["loss"] < best_metric:
            best_metric = validation_metrics["loss"]
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch,
                validation_metrics,
                config,
                train_dataset.speaker_to_index,
            )
            print(f"saved_best_checkpoint={checkpoint_path}")


if __name__ == "__main__":
    main()
