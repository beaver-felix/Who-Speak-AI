"""Deterministic epoch planning and framework-neutral training state.

This module deliberately avoids PyTorch so resume arithmetic and early-stopping
policy can be tested locally. Model tensors and RNG state are persisted by the
separate Kaggle-only checkpoint module.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import asdict, dataclass
from typing import Iterator, Mapping, Sequence


class TrainingLifecycleError(ValueError):
    """Raised when epoch, resume, or selection state is inconsistent."""


@dataclass(frozen=True, slots=True)
class TrainingRunIdentity:
    """Bind a checkpoint to one immutable experiment and data manifest."""

    model_name: str
    dataset_name: str
    config_sha256: str
    manifest_sha256: str
    seed: int

    def __post_init__(self) -> None:
        """Reject identities that cannot safely guard checkpoint resume."""
        _non_empty_text(self.model_name, "model_name")
        _non_empty_text(self.dataset_name, "dataset_name")
        _sha256(self.config_sha256, "config_sha256")
        _sha256(self.manifest_sha256, "manifest_sha256")
        _non_negative_integer(self.seed, "seed")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe identity mapping."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TrainingCursor:
    """Describe the exact next batch and accumulated current-epoch metrics."""

    epoch_index: int = 0
    next_batch_index: int = 0
    global_step: int = 0
    epoch_loss_sum: float = 0.0
    epoch_accuracy_sum: float = 0.0
    epoch_example_count: int = 0
    best_validation_eer: float | None = None
    best_validation_min_dcf: float | None = None
    best_epoch_index: int | None = None
    stale_validation_count: int = 0

    def __post_init__(self) -> None:
        """Reject cursor values that could silently skip or repeat data."""
        for field_name in (
            "epoch_index",
            "next_batch_index",
            "global_step",
            "epoch_example_count",
            "stale_validation_count",
        ):
            _non_negative_integer(getattr(self, field_name), field_name)
        _finite_non_negative(self.epoch_loss_sum, "epoch_loss_sum")
        _finite_non_negative(self.epoch_accuracy_sum, "epoch_accuracy_sum")
        if self.epoch_example_count == 0 and (
            self.epoch_loss_sum != 0.0 or self.epoch_accuracy_sum != 0.0
        ):
            raise TrainingLifecycleError(
                "Empty epoch accumulation must have zero metric sums."
            )
        if self.best_validation_eer is not None:
            _rate(self.best_validation_eer, "best_validation_eer")
        if self.best_validation_min_dcf is not None:
            _finite_non_negative(
                self.best_validation_min_dcf,
                "best_validation_min_dcf",
            )
        if (self.best_validation_eer is None) != (
            self.best_validation_min_dcf is None
        ):
            raise TrainingLifecycleError(
                "Best Validation EER and minDCF must be recorded together."
            )
        if self.best_epoch_index is not None:
            _non_negative_integer(self.best_epoch_index, "best_epoch_index")
            if self.best_validation_eer is None:
                raise TrainingLifecycleError(
                    "best_epoch_index requires best_validation_eer."
                )

    def to_dict(self) -> dict[str, object]:
        """Return a checkpoint-safe cursor mapping."""
        return asdict(self)

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "TrainingCursor":
        """Restore a cursor while applying all constructor invariants."""
        try:
            return cls(**dict(values))
        except TypeError as error:
            raise TrainingLifecycleError(
                "Checkpoint cursor fields do not match schema 1."
            ) from error


@dataclass(frozen=True, slots=True)
class EarlyStoppingPolicy:
    """Select checkpoints by validation EER without consulting Test data."""

    patience: int
    minimum_eer_improvement: float = 0.0

    def __post_init__(self) -> None:
        """Validate an explicit and bounded stopping rule."""
        _positive_integer(self.patience, "patience")
        _finite_non_negative(
            self.minimum_eer_improvement,
            "minimum_eer_improvement",
        )
        if self.minimum_eer_improvement >= 1.0:
            raise TrainingLifecycleError(
                "minimum_eer_improvement must be smaller than one."
            )


@dataclass(frozen=True, slots=True)
class EpochSummary:
    """Store auditable train and Validation metrics for one completed epoch."""

    epoch_index: int
    global_step: int
    training_loss: float
    training_accuracy: float
    validation_eer: float
    validation_min_dcf: float
    improved: bool

    def __post_init__(self) -> None:
        """Require finite metrics and valid probability-like rates."""
        _non_negative_integer(self.epoch_index, "epoch_index")
        _positive_integer(self.global_step, "global_step")
        _finite_non_negative(self.training_loss, "training_loss")
        _rate(self.training_accuracy, "training_accuracy")
        _rate(self.validation_eer, "validation_eer")
        _finite_non_negative(self.validation_min_dcf, "validation_min_dcf")
        if not isinstance(self.improved, bool):
            raise TrainingLifecycleError("improved must be boolean.")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe epoch record."""
        return asdict(self)


class DeterministicEpochBatchSampler:
    """Yield a stable shuffled epoch, optionally resuming at one batch.

    The sampler owns its private ``random.Random`` instance. It never consumes
    global Python, NumPy, or PyTorch RNG state, so reconstructing it after a
    restart produces the exact same index order.
    """

    def __init__(
        self,
        *,
        dataset_size: int,
        batch_size: int,
        seed: int,
        epoch_index: int,
        start_batch_index: int = 0,
        drop_last: bool = False,
    ) -> None:
        """Validate the epoch plan before any samples are loaded."""
        _positive_integer(dataset_size, "dataset_size")
        _positive_integer(batch_size, "batch_size")
        _non_negative_integer(seed, "seed")
        _non_negative_integer(epoch_index, "epoch_index")
        _non_negative_integer(start_batch_index, "start_batch_index")
        if not isinstance(drop_last, bool):
            raise TrainingLifecycleError("drop_last must be boolean.")

        total_batches = (
            dataset_size // batch_size
            if drop_last
            else math.ceil(dataset_size / batch_size)
        )
        if total_batches == 0:
            raise TrainingLifecycleError(
                "drop_last removed the complete dataset from the epoch."
            )
        if start_batch_index > total_batches:
            raise TrainingLifecycleError(
                "start_batch_index exceeds the epoch batch count."
            )

        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.seed = seed
        self.epoch_index = epoch_index
        self.start_batch_index = start_batch_index
        self.drop_last = drop_last
        self.total_batches = total_batches

    def __len__(self) -> int:
        """Return remaining rather than total batches for DataLoader."""
        return self.total_batches - self.start_batch_index

    def __iter__(self) -> Iterator[list[int]]:
        """Yield the deterministic resume suffix as index lists."""
        indices = list(range(self.dataset_size))
        generator = random.Random(
            _epoch_shuffle_seed(self.seed, self.epoch_index)
        )
        generator.shuffle(indices)
        stop = (
            self.total_batches * self.batch_size
            if self.drop_last
            else self.dataset_size
        )
        for batch_index in range(
            self.start_batch_index,
            self.total_batches,
        ):
            start = batch_index * self.batch_size
            yield indices[start : min(start + self.batch_size, stop)]


def record_training_batch(
    cursor: TrainingCursor,
    *,
    batch_size: int,
    loss: float,
    accuracy: float,
) -> TrainingCursor:
    """Advance one successful optimizer step and preserve weighted metrics."""
    _positive_integer(batch_size, "batch_size")
    _finite_non_negative(loss, "loss")
    _rate(accuracy, "accuracy")
    return TrainingCursor(
        epoch_index=cursor.epoch_index,
        next_batch_index=cursor.next_batch_index + 1,
        global_step=cursor.global_step + 1,
        epoch_loss_sum=cursor.epoch_loss_sum + loss * batch_size,
        epoch_accuracy_sum=(
            cursor.epoch_accuracy_sum + accuracy * batch_size
        ),
        epoch_example_count=cursor.epoch_example_count + batch_size,
        best_validation_eer=cursor.best_validation_eer,
        best_validation_min_dcf=cursor.best_validation_min_dcf,
        best_epoch_index=cursor.best_epoch_index,
        stale_validation_count=cursor.stale_validation_count,
    )


def complete_epoch(
    cursor: TrainingCursor,
    *,
    validation_eer: float,
    validation_min_dcf: float,
    policy: EarlyStoppingPolicy,
) -> tuple[TrainingCursor, EpochSummary, bool]:
    """Select the best epoch by Validation EER and reset epoch accumulation."""
    if cursor.epoch_example_count == 0:
        raise TrainingLifecycleError("Cannot complete an epoch with no examples.")
    _rate(validation_eer, "validation_eer")
    _finite_non_negative(validation_min_dcf, "validation_min_dcf")

    best_eer = cursor.best_validation_eer
    best_min_dcf = cursor.best_validation_min_dcf
    improved = (
        best_eer is None
        or validation_eer
        < best_eer - policy.minimum_eer_improvement
        or (
            abs(validation_eer - best_eer)
            <= policy.minimum_eer_improvement
            and best_min_dcf is not None
            and validation_min_dcf < best_min_dcf
        )
    )
    next_best_eer = validation_eer if improved else best_eer
    next_best_min_dcf = validation_min_dcf if improved else best_min_dcf
    next_best_epoch = (
        cursor.epoch_index if improved else cursor.best_epoch_index
    )
    stale_count = 0 if improved else cursor.stale_validation_count + 1
    summary = EpochSummary(
        epoch_index=cursor.epoch_index,
        global_step=cursor.global_step,
        training_loss=cursor.epoch_loss_sum / cursor.epoch_example_count,
        training_accuracy=(
            cursor.epoch_accuracy_sum / cursor.epoch_example_count
        ),
        validation_eer=validation_eer,
        validation_min_dcf=validation_min_dcf,
        improved=improved,
    )
    next_cursor = TrainingCursor(
        epoch_index=cursor.epoch_index + 1,
        next_batch_index=0,
        global_step=cursor.global_step,
        best_validation_eer=next_best_eer,
        best_validation_min_dcf=next_best_min_dcf,
        best_epoch_index=next_best_epoch,
        stale_validation_count=stale_count,
    )
    return next_cursor, summary, stale_count >= policy.patience


def validate_epoch_history(
    history: Sequence[Mapping[str, object]],
) -> tuple[EpochSummary, ...]:
    """Restore a strictly ordered sequence of completed epoch summaries."""
    restored: list[EpochSummary] = []
    for expected_epoch, values in enumerate(history):
        try:
            summary = EpochSummary(**dict(values))
        except TypeError as error:
            raise TrainingLifecycleError(
                "Checkpoint epoch history fields do not match schema 1."
            ) from error
        if summary.epoch_index != expected_epoch:
            raise TrainingLifecycleError(
                "Checkpoint epoch history must be contiguous from zero."
            )
        restored.append(summary)
    return tuple(restored)


def _epoch_shuffle_seed(seed: int, epoch_index: int) -> int:
    """Derive a stable private shuffle seed from run and epoch identity."""
    payload = f"{seed}\0{epoch_index}\0epoch-order-v1".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _non_empty_text(value: object, field_name: str) -> None:
    """Require one non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise TrainingLifecycleError(f"{field_name} must be non-empty text.")


def _sha256(value: object, field_name: str) -> None:
    """Require one lowercase hexadecimal SHA-256 digest."""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value.lower() != value
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TrainingLifecycleError(
            f"{field_name} must be a lowercase SHA-256 digest."
        )


def _positive_integer(value: object, field_name: str) -> None:
    """Require a strict positive integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TrainingLifecycleError(f"{field_name} must be a positive integer.")


def _non_negative_integer(value: object, field_name: str) -> None:
    """Require a strict non-negative integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TrainingLifecycleError(
            f"{field_name} must be a non-negative integer."
        )


def _finite_non_negative(value: object, field_name: str) -> None:
    """Require a finite number greater than or equal to zero."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise TrainingLifecycleError(
            f"{field_name} must be finite and non-negative."
        )


def _rate(value: object, field_name: str) -> None:
    """Require a finite rate in the closed unit interval."""
    _finite_non_negative(value, field_name)
    if float(value) > 1.0:
        raise TrainingLifecycleError(f"{field_name} must be in [0, 1].")
