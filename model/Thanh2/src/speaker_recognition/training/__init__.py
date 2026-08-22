"""Shared supervised-training contracts for all speaker encoders."""

from speaker_recognition.training.specification import (
    AamSoftmaxSpec,
    OptimizationSpec,
    TrainingSpecification,
    TrainingSpecificationError,
)

__all__ = [
    "AamSoftmaxSpec",
    "OptimizationSpec",
    "TrainingSpecification",
    "TrainingSpecificationError",
]
