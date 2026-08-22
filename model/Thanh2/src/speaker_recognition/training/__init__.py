"""Shared supervised-training contracts for all speaker encoders."""

from speaker_recognition.training.lifecycle import (
    DeterministicEpochBatchSampler,
    EarlyStoppingPolicy,
    EpochSummary,
    TrainingCursor,
    TrainingLifecycleError,
    TrainingRunIdentity,
    complete_epoch,
    record_training_batch,
)
from speaker_recognition.training.specification import (
    AamSoftmaxSpec,
    BatchSpec,
    OptimizationSpec,
    TrainingSpecification,
    TrainingSpecificationError,
)

__all__ = [
    "AamSoftmaxSpec",
    "BatchSpec",
    "DeterministicEpochBatchSampler",
    "EarlyStoppingPolicy",
    "EpochSummary",
    "OptimizationSpec",
    "TrainingCursor",
    "TrainingLifecycleError",
    "TrainingRunIdentity",
    "TrainingSpecification",
    "TrainingSpecificationError",
    "complete_epoch",
    "record_training_batch",
]
