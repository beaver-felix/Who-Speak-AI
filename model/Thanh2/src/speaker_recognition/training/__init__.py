"""Shared supervised-training contracts for all speaker encoders."""

from speaker_recognition.training.lifecycle import (
    DeterministicGroupedEpochBatchSampler,
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
    EpochSamplingSpec,
    EvaluationSpec,
    LifecycleSpec,
    OptimizationSpec,
    TrainingSpecification,
    TrainingSpecificationError,
)

__all__ = [
    "DeterministicGroupedEpochBatchSampler",
    "AamSoftmaxSpec",
    "BatchSpec",
    "DeterministicEpochBatchSampler",
    "EarlyStoppingPolicy",
    "EpochSamplingSpec",
    "EpochSummary",
    "EvaluationSpec",
    "LifecycleSpec",
    "OptimizationSpec",
    "TrainingCursor",
    "TrainingLifecycleError",
    "TrainingRunIdentity",
    "TrainingSpecification",
    "TrainingSpecificationError",
    "complete_epoch",
    "record_training_batch",
]
