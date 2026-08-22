"""Shared supervised-training contracts for all speaker encoders."""

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
    "OptimizationSpec",
    "TrainingSpecification",
    "TrainingSpecificationError",
]
