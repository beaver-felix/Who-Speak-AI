"""Shared single-GPU epoch trainer for all three speaker adapters.

The engine owns optimization mechanics but delegates speaker-verification
evaluation to a callback. This keeps Test data outside training and allows the
same lifecycle to serve TidyVoice and ViMD.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

try:
    import torch
except ModuleNotFoundError as error:  # pragma: no cover - Kaggle dependency gate.
    raise ModuleNotFoundError(
        "Epoch training requires Kaggle's CUDA-matched PyTorch build."
    ) from error

from speaker_recognition.data.dataset import (
    TrainingSpeakerDataset,
    collate_training_samples,
)
from speaker_recognition.training.checkpointing import (
    restore_training_checkpoint,
    save_training_checkpoint,
)
from speaker_recognition.training.lifecycle import (
    DeterministicEpochBatchSampler,
    DeterministicGroupedEpochBatchSampler,
    EarlyStoppingPolicy,
    EpochSummary,
    TrainingCursor,
    TrainingRunIdentity,
    complete_epoch,
    record_training_batch,
)
from speaker_recognition.training.logging import NullRunLogger, RunLogger
from speaker_recognition.training.optimization import OptimizerBundle


ValidationFunction = Callable[[torch.nn.Module, int], Mapping[str, float]]


class TrainingEngineError(RuntimeError):
    """Raised when an epoch produces unsafe or inconsistent runtime state."""


@dataclass(frozen=True, slots=True)
class TrainerSettings:
    """Record runtime controls that are shared by one complete training run."""

    batch_size: int
    max_epochs: int
    gradient_clip_norm: float
    checkpoint_every_steps: int = 1000
    num_workers: int = 2
    pin_memory: bool = True
    initial_loss_scale: float = 1024.0
    group_by_audio_path: bool = False

    def __post_init__(self) -> None:
        """Reject settings that could create an empty or unsafe run."""
        for value, field_name in (
            (self.batch_size, "batch_size"),
            (self.max_epochs, "max_epochs"),
            (self.checkpoint_every_steps, "checkpoint_every_steps"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer.")
        if (
            isinstance(self.num_workers, bool)
            or not isinstance(self.num_workers, int)
            or self.num_workers < 0
        ):
            raise ValueError("num_workers must be a non-negative integer.")
        for value, field_name in (
            (self.gradient_clip_norm, "gradient_clip_norm"),
            (self.initial_loss_scale, "initial_loss_scale"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{field_name} must be finite and positive.")
        if not isinstance(self.pin_memory, bool):
            raise ValueError("pin_memory must be boolean.")
        if not isinstance(self.group_by_audio_path, bool):
            raise ValueError("group_by_audio_path must be boolean.")


@dataclass(frozen=True, slots=True)
class TrainingOutcome:
    """Summarize a completed or early-stopped call to ``fit``."""

    cursor: TrainingCursor
    history: tuple[EpochSummary, ...]
    stopped_early: bool


class SpeakerTrainingEngine:
    """Train one adapter/head pair with deterministic, exact-resume epochs."""

    def __init__(
        self,
        *,
        adapter: torch.nn.Module,
        objective: torch.nn.Module,
        optimizer_bundle: OptimizerBundle,
        dataset: TrainingSpeakerDataset,
        identity: TrainingRunIdentity,
        settings: TrainerSettings,
        early_stopping: EarlyStoppingPolicy,
        device: str = "cuda:0",
    ) -> None:
        """Bind all stateful components before the first optimizer step."""
        if not device.startswith("cuda") or not torch.cuda.is_available():
            raise TrainingEngineError("Accepted epoch training requires CUDA.")
        metadata = getattr(adapter, "metadata", None)
        if getattr(metadata, "name", None) != identity.model_name:
            raise TrainingEngineError(
                "Adapter metadata does not match the run identity."
            )
        if settings.batch_size <= 0 or len(dataset) <= 0:
            raise TrainingEngineError("Training dataset and batch must be non-empty.")

        self.adapter = adapter
        self.objective = objective
        self.optimizer_bundle = optimizer_bundle
        self.dataset = dataset
        self.identity = identity
        self.settings = settings
        self.early_stopping = early_stopping
        self.device = device
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=True,
            init_scale=settings.initial_loss_scale,
            growth_interval=100_000,
        )
        self.cursor = TrainingCursor()
        self.history: tuple[EpochSummary, ...] = ()

    def resume_from(self, checkpoint_path: str | Path) -> str:
        """Restore model, optimizer, scaler, RNG, cursor, and metric history."""
        restored = restore_training_checkpoint(
            checkpoint_path,
            adapter=self.adapter,
            objective=self.objective,
            optimizer=self.optimizer_bundle.optimizer,
            scaler=self.scaler,
            expected_identity=self.identity,
            expected_optimizer_group_names=self.optimizer_bundle.group_names,
            device=self.device,
        )
        self.cursor = restored.cursor
        self.history = restored.history
        return restored.checkpoint_sha256

    def fit(
        self,
        *,
        validation_function: ValidationFunction,
        last_checkpoint_path: str | Path,
        best_checkpoint_path: str | Path,
        logger: RunLogger | None = None,
    ) -> TrainingOutcome:
        """Run until the epoch budget or Validation-only early stopping."""
        active_logger = logger or NullRunLogger()
        stopped_early = False
        try:
            while self.cursor.epoch_index < self.settings.max_epochs:
                self._train_remaining_epoch(
                    checkpoint_path=last_checkpoint_path,
                    logger=active_logger,
                )
                validation = _validated_validation_metrics(
                    validation_function(
                        self.adapter,
                        self.cursor.epoch_index,
                    )
                )
                self.cursor, summary, stopped_early = complete_epoch(
                    self.cursor,
                    validation_eer=validation["eer"],
                    validation_min_dcf=validation["min_dcf"],
                    policy=self.early_stopping,
                )
                self.history = (*self.history, summary)
                epoch_metrics = {
                    "epoch": float(summary.epoch_index),
                    "train/epoch_loss": summary.training_loss,
                    "train/epoch_accuracy": summary.training_accuracy,
                    **{
                        f"validation/{name}": value
                        for name, value in validation.items()
                    },
                    "selection/improved": float(summary.improved),
                    "selection/stale_validations": float(
                        self.cursor.stale_validation_count
                    ),
                }
                active_logger.log(epoch_metrics, step=self.cursor.global_step)

                # Save best after cursor advancement so resuming a selected
                # checkpoint never repeats its completed Validation epoch.
                if summary.improved:
                    self._save(best_checkpoint_path)
                self._save(last_checkpoint_path)
                if stopped_early:
                    break
        finally:
            active_logger.finish()

        return TrainingOutcome(
            cursor=self.cursor,
            history=self.history,
            stopped_early=stopped_early,
        )

    def _train_remaining_epoch(
        self,
        *,
        checkpoint_path: str | Path,
        logger: RunLogger,
    ) -> None:
        """Train the deterministic suffix identified by the current cursor."""
        self.dataset.set_epoch(self.cursor.epoch_index)
        if self.settings.group_by_audio_path:
            groups_by_path: dict[str, list[int]] = {}
            for index, record in enumerate(self.dataset.epoch_records):
                groups_by_path.setdefault(record.audio_path, []).append(index)
            sampler = DeterministicGroupedEpochBatchSampler(
                index_groups=tuple(
                    tuple(groups_by_path[path])
                    for path in sorted(groups_by_path)
                ),
                batch_size=self.settings.batch_size,
                seed=self.identity.seed,
                epoch_index=self.cursor.epoch_index,
                start_batch_index=self.cursor.next_batch_index,
                drop_last=False,
            )
        else:
            sampler = DeterministicEpochBatchSampler(
                dataset_size=len(self.dataset),
                batch_size=self.settings.batch_size,
                seed=self.identity.seed,
                epoch_index=self.cursor.epoch_index,
                start_batch_index=self.cursor.next_batch_index,
                drop_last=False,
            )
        # DataLoader iterator construction otherwise consumes global torch RNG,
        # perturbing dropout after a mid-epoch resume. A private generator keeps
        # worker seeding separate from model RNG state.
        loader_generator = torch.Generator()
        loader_generator.manual_seed(
            _loader_seed(self.identity.seed, self.cursor.epoch_index)
        )
        data_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_sampler=sampler,
            num_workers=self.settings.num_workers,
            collate_fn=collate_training_samples,
            pin_memory=self.settings.pin_memory,
            persistent_workers=False,
            generator=loader_generator,
        )

        self.adapter.train()
        self.objective.train()
        for batch in data_loader:
            batch_metrics = self._train_batch(batch)
            self.cursor = record_training_batch(
                self.cursor,
                batch_size=int(batch.speaker_indices.shape[0]),
                loss=batch_metrics["loss"],
                accuracy=batch_metrics["accuracy"],
            )
            logger.log(
                {
                    "epoch": float(self.cursor.epoch_index),
                    "train/batch_loss": batch_metrics["loss"],
                    "train/batch_accuracy": batch_metrics["accuracy"],
                    "train/gradient_norm_before_clipping": batch_metrics[
                        "gradient_norm_before_clipping"
                    ],
                    "train/loss_scale": batch_metrics["loss_scale"],
                    "train/fp32_fallback": batch_metrics[
                        "fp32_fallback"
                    ],
                    "train/learning_rate_min": min(
                        group["lr"]
                        for group in self.optimizer_bundle.optimizer.param_groups
                    ),
                    "train/learning_rate_max": max(
                        group["lr"]
                        for group in self.optimizer_bundle.optimizer.param_groups
                    ),
                },
                step=self.cursor.global_step,
            )
            if (
                self.cursor.global_step
                % self.settings.checkpoint_every_steps
                == 0
            ):
                self._save(checkpoint_path)

    def _train_batch(self, batch: object) -> dict[str, float]:
        """Apply one fail-closed FP16 optimizer step."""
        waveforms = torch.from_numpy(batch.waveforms).to(
            self.device,
            non_blocking=self.settings.pin_memory,
        )
        labels = torch.from_numpy(batch.speaker_indices).to(
            self.device,
            non_blocking=self.settings.pin_memory,
        )
        if not bool(torch.isfinite(waveforms).all()):
            raise TrainingEngineError("Training audio contains non-finite values.")

        optimizer = self.optimizer_bundle.optimizer
        optimizer.zero_grad(set_to_none=True)
        scale_before = float(self.scaler.get_scale())
        model_name = getattr(getattr(self.adapter, "metadata", None), "name", None)
        rng_snapshot = (
            _capture_rng_state(self.device)
            if model_name == "wavlm_mhfa"
            else None
        )
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=True,
        ):
            embeddings = self.adapter(waveforms)

        used_fp32_fallback = False
        if not bool(torch.isfinite(embeddings).all()):
            if rng_snapshot is None:
                raise TrainingEngineError(
                    "Embedding became non-finite for utterances: "
                    f"{_bounded_utterance_ids(batch)}."
                )
            # WavLM attempt 1 produced its first non-finite FP16 embedding at
            # deterministic batch 579 after 578 valid updates. Restore every
            # stochastic state before recomputing so dropout/layerdrop masks
            # are identical; only arithmetic precision changes.
            _restore_rng_state(rng_snapshot, self.device)
            with torch.autocast(device_type="cuda", enabled=False):
                embeddings = self.adapter(waveforms.float())
            used_fp32_fallback = True
            if not bool(torch.isfinite(embeddings).all()):
                raise TrainingEngineError(
                    "WavLM embedding remained non-finite after FP32 fallback "
                    f"for utterances: {_bounded_utterance_ids(batch)}."
                )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=not used_fp32_fallback,
        ):
            output = self.objective(embeddings, labels)
        if not bool(torch.isfinite(output.loss)):
            raise TrainingEngineError(
                "Loss became non-finite for utterances: "
                f"{_bounded_utterance_ids(batch)}."
            )

        self.scaler.scale(output.loss).backward()
        self.scaler.unscale_(optimizer)
        enabled_parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            enabled_parameters,
            max_norm=self.settings.gradient_clip_norm,
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise TrainingEngineError("Gradient norm became non-finite.")
        self.scaler.step(optimizer)
        self.scaler.update()
        scale_after = float(self.scaler.get_scale())
        if scale_after < scale_before:
            raise TrainingEngineError(
                "GradScaler backed off and skipped an unsafe optimizer step."
            )

        return {
            "loss": float(output.loss.detach().cpu()),
            "accuracy": float(output.accuracy.detach().cpu()),
            "gradient_norm_before_clipping": float(
                gradient_norm.detach().cpu()
            ),
            "loss_scale": scale_after,
            "fp32_fallback": float(used_fp32_fallback),
        }

    def _save(self, path: str | Path) -> str:
        """Persist all exact-resume state through the atomic checkpoint layer."""
        return save_training_checkpoint(
            path,
            adapter=self.adapter,
            objective=self.objective,
            optimizer=self.optimizer_bundle.optimizer,
            scaler=self.scaler,
            identity=self.identity,
            cursor=self.cursor,
            history=self.history,
            optimizer_group_names=self.optimizer_bundle.group_names,
            device=self.device,
        )


def _capture_rng_state(device: str) -> tuple[object, object, torch.Tensor, torch.Tensor]:
    """Snapshot every stochastic source used by WavLM dropout/layerdrop."""
    return (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        torch.cuda.get_rng_state(device),
    )


def _restore_rng_state(
    state: tuple[object, object, torch.Tensor, torch.Tensor],
    device: str,
) -> None:
    """Restore a pre-forward RNG snapshot before precision-only recompute."""
    python_state, numpy_state, cpu_state, cuda_state = state
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(cpu_state)
    torch.cuda.set_rng_state(cuda_state, device=device)


def _bounded_utterance_ids(batch: object, *, limit: int = 6) -> list[str]:
    """Return bounded batch provenance for actionable numerical failures."""
    values = getattr(batch, "utterance_ids", ())
    return [str(value) for value in tuple(values)[:limit]]


def _validated_validation_metrics(
    values: Mapping[str, float],
) -> dict[str, float]:
    """Require Validation EER/minDCF and finite optional diagnostic metrics."""
    if "eer" not in values or "min_dcf" not in values:
        raise TrainingEngineError("Validation must return EER and minDCF.")
    normalized: dict[str, float] = {}
    for name, value in values.items():
        if not isinstance(name, str) or not name.strip():
            raise TrainingEngineError("Validation metric names must be non-empty.")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise TrainingEngineError(
                f"Validation metric {name!r} must be finite."
            )
        normalized[name] = float(value)
    if not 0.0 <= normalized["eer"] <= 1.0:
        raise TrainingEngineError("Validation EER must be in [0, 1].")
    if normalized["min_dcf"] < 0.0:
        raise TrainingEngineError("Validation minDCF must be non-negative.")
    return normalized


def _loader_seed(seed: int, epoch_index: int) -> int:
    """Derive a bounded private worker seed without using model RNG state."""
    return (seed * 1_000_003 + epoch_index * 97_409 + 17) % (2**63 - 1)
