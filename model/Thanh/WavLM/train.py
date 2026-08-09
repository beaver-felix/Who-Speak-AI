"""Memory-conscious WavLM speaker-classification fine-tuning for Kaggle."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

THANH_ROOT = Path(__file__).resolve().parent.parent
if str(THANH_ROOT) not in sys.path:
    sys.path.insert(0, str(THANH_ROOT))

from RawNet3.dataset import (  # noqa: E402
    SpeakerAudioDataset,
    build_dataloader,
    load_speaker_records,
)
from WavLM.model import WavLMSpeakerModel, build_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse WavLM training command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML configuration mapping."""
    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a YAML mapping")
    return config


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(config: dict[str, Any], override: str | None) -> torch.device:
    """Select a device and enforce the configured Kaggle GPU requirement."""
    runtime = config.get("runtime", {})
    requested = override or runtime.get("device", "cuda")
    device = torch.device(requested)
    require_cuda = bool(runtime.get("require_cuda", False)) and override != "cpu"
    if require_cuda and device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by config, but no GPU is available")
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU for a debug run")
        return torch.device("cpu")
    return device


def amp_settings(config: dict[str, Any], device: torch.device) -> tuple[bool, torch.dtype]:
    """Resolve FP16/BF16 CUDA autocast settings."""
    training = config["training"]
    precision = str(training.get("mixed_precision", "fp16")).lower()
    if precision not in {"fp16", "bf16"}:
        raise ValueError("mixed_precision must be 'fp16' or 'bf16'")
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    enabled = bool(training.get("use_amp", True)) and device.type == "cuda"
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

    validation_manifest = dataset_config.get("validation_manifest")
    if not validation_manifest:
        return train_dataset, None
    validation_records = load_speaker_records(
        validation_manifest,
        dataset_config.get("path_column", "path"),
        dataset_config.get("speaker_column", "speaker_id"),
    )
    known_records = [
        record
        for record in validation_records
        if record.speaker_id in train_dataset.speaker_to_index
    ]
    if not known_records:
        print("No validation speakers overlap training speakers; validation disabled")
        return train_dataset, None
    validation_dataset = SpeakerAudioDataset(
        known_records,
        sample_rate=int(audio_config["sample_rate"]),
        duration_seconds=float(audio_config["duration_seconds"]),
        training=False,
        peak_normalize=bool(audio_config.get("peak_normalize", False)),
    )
    validation_dataset.speaker_to_index = train_dataset.speaker_to_index
    return train_dataset, validation_dataset


def build_optimizer(
    model: WavLMSpeakerModel, config: dict[str, Any]
) -> torch.optim.Optimizer:
    """Use a lower learning rate for trainable backbone parameters."""
    training = config["training"]
    backbone_parameters: list[nn.Parameter] = []
    head_parameters: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("backbone."):
            backbone_parameters.append(parameter)
        else:
            head_parameters.append(parameter)

    groups: list[dict[str, Any]] = []
    if backbone_parameters:
        groups.append(
            {
                "params": backbone_parameters,
                "lr": float(training["backbone_learning_rate"]),
            }
        )
    if head_parameters:
        groups.append(
            {"params": head_parameters, "lr": float(training["head_learning_rate"])}
        )
    if not groups:
        raise RuntimeError("The model has no trainable parameters")
    return torch.optim.AdamW(
        groups, weight_decay=float(training.get("weight_decay", 0.0))
    )


def run_epoch(
    model: WavLMSpeakerModel,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    accumulation_steps: int = 1,
    gradient_clip_norm: float = 1.0,
) -> dict[str, float]:
    """Run one training or validation epoch."""
    if accumulation_steps <= 0:
        raise ValueError("gradient accumulation steps must be positive")
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    if training:
        optimizer.zero_grad(set_to_none=True)

    for batch_index, batch in enumerate(loader):
        waveform = batch["waveform"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["speaker_index"].to(device, non_blocking=True).long()
        with torch.set_grad_enabled(training):
            with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
                logits = model(waveform, attention_mask=attention_mask, labels=labels)
                loss = criterion(logits, labels)
            if training:
                scaler.scale(loss / accumulation_steps).backward()
                should_step = (
                    (batch_index + 1) % accumulation_steps == 0
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
        total_correct += int((logits.detach().argmax(dim=-1) == labels).sum())
        total_examples += batch_size

    if total_examples == 0:
        raise RuntimeError("DataLoader produced no examples")
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def checkpoint_model_config(model: WavLMSpeakerModel) -> dict[str, Any]:
    """Serialize constructor settings needed to restore the model."""
    classifier = model.classifier
    return {
        "pretrained_model_name": model.pretrained_model_name,
        "num_speakers": model.num_speakers,
        "pooling": model.pooling,
        "embedding_dim": model.embedding_dim,
        "freeze_backbone": model.freeze_backbone_requested,
        "unfreeze_last_n_layers": model.unfreeze_last_n_layers,
        "gradient_checkpointing": False,
        "am_scale": classifier.scale if classifier is not None else 30.0,
        "am_margin": classifier.margin if classifier is not None else 0.20,
    }


def save_checkpoint(
    path: Path,
    model: WavLMSpeakerModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    config: dict[str, Any],
    speaker_to_index: dict[str, int],
) -> None:
    """Save the best model and metadata required by evaluation/inference."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": config,
            "model_config": checkpoint_model_config(model),
            "speaker_to_index": speaker_to_index,
        },
        path,
    )


def main() -> None:
    """Run the WavLM fine-tuning job."""
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("project", {}).get("seed", 42)))
    device = resolve_device(config, args.device)
    amp_enabled, amp_dtype = amp_settings(config, device)
    train_dataset, validation_dataset = build_datasets(config)
    training = config["training"]
    runtime = config.get("runtime", {})

    common_loader_options = {
        "num_workers": int(runtime.get("num_workers", 0)),
        "pin_memory": bool(runtime.get("pin_memory", False)),
        "persistent_workers": bool(runtime.get("persistent_workers", True)),
        "prefetch_factor": int(runtime.get("prefetch_factor", 1)),
    }
    train_loader = build_dataloader(
        train_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        drop_last=bool(runtime.get("drop_last", True)),
        **common_loader_options,
    )
    validation_loader = (
        build_dataloader(
            validation_dataset,
            batch_size=int(training.get("evaluation_batch_size", training["batch_size"])),
            shuffle=False,
            drop_last=False,
            **common_loader_options,
        )
        if validation_dataset is not None
        else None
    )

    model_settings = config["model"]
    model = build_model(
        pretrained_model_name=model_settings["pretrained_model_name"],
        num_speakers=len(train_dataset.speaker_to_index),
        pooling=model_settings.get("pooling", "xvector"),
        embedding_dim=int(model_settings.get("embedding_dim", 512)),
        freeze_backbone=bool(model_settings.get("freeze_backbone", True)),
        unfreeze_last_n_layers=int(model_settings.get("unfreeze_last_n_layers", 0)),
        gradient_checkpointing=bool(model_settings.get("gradient_checkpointing", False)),
        am_scale=float(model_settings.get("am_scale", 30.0)),
        am_margin=float(model_settings.get("am_margin", 0.20)),
    ).to(device)
    optimizer = build_optimizer(model, config)
    scaler = torch.cuda.amp.GradScaler(
        enabled=amp_enabled and amp_dtype == torch.float16
    )
    criterion = nn.CrossEntropyLoss()
    start_epoch = 0
    best_loss = float("inf")

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_loss = float(checkpoint.get("metrics", {}).get("loss", best_loss))

    checkpoint_directory = Path(config["outputs"]["checkpoint_directory"])
    if not checkpoint_directory.is_absolute():
        checkpoint_directory = args.config.resolve().parent / checkpoint_directory
    checkpoint_path = checkpoint_directory / "best_model.pt"
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(f"device={device} trainable_parameters={trainable_parameters:,}")

    for epoch in range(start_epoch, int(training["epochs"])):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            amp_enabled,
            amp_dtype,
            int(training.get("gradient_accumulation_steps", 1)),
            float(training.get("gradient_clip_norm", 1.0)),
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
            f"epoch={epoch + 1}/{training['epochs']} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={validation_metrics['loss']:.4f} "
            f"val_acc={validation_metrics['accuracy']:.4f}"
        )
        if validation_metrics["loss"] < best_loss:
            best_loss = validation_metrics["loss"]
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
