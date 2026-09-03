"""Dependency-free validation of shared and model-specific training policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


class TrainingSpecificationError(ValueError):
    """Raised when a resolved training configuration is incomplete or unsafe."""


@dataclass(frozen=True, slots=True)
class AamSoftmaxSpec:
    """Describe the comparison-invariant AAM-Softmax objective."""

    margin: float
    scale: float
    easy_margin: bool

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "AamSoftmaxSpec":
        """Build the accepted shared objective from a configuration table."""
        if values.get("name") != "aam_softmax":
            raise TrainingSpecificationError(
                "training.objective.name must equal 'aam_softmax'."
            )
        margin = _positive_float(values, "margin")
        scale = _positive_float(values, "scale")
        easy_margin = values.get("easy_margin")
        if not isinstance(easy_margin, bool):
            raise TrainingSpecificationError(
                "training.objective.easy_margin must be boolean."
            )
        if margin >= 1.5707963267948966:
            raise TrainingSpecificationError(
                "AAM-Softmax margin must be smaller than pi/2 radians."
            )
        return cls(margin=margin, scale=scale, easy_margin=easy_margin)


@dataclass(frozen=True, slots=True)
class BatchSpec:
    """Describe a memory- and multi-batch-validated training batch."""

    size: int
    calibrated_largest_passing_size: int
    selection_status: str
    calibration_artifact: str
    validation_artifact: str

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "BatchSpec":
        """Validate a conservative candidate selected from a passing grid."""
        size = _positive_integer(values, "size")
        largest = _positive_integer(values, "calibrated_largest_passing_size")
        if size > largest:
            raise TrainingSpecificationError(
                "Batch size cannot exceed its calibrated passing maximum."
            )
        status = _non_empty_text(values, "selection_status")
        if status != "multibatch_validated_ready_for_epoch_training":
            raise TrainingSpecificationError(
                "Batch selection status must prove multi-batch validation."
            )
        return cls(
            size=size,
            calibrated_largest_passing_size=largest,
            selection_status=status,
            calibration_artifact=_non_empty_text(
                values,
                "calibration_artifact",
            ),
            validation_artifact=_non_empty_text(
                values,
                "validation_artifact",
            ),
        )


@dataclass(frozen=True, slots=True)
class OptimizationSpec:
    """Describe one model's optimizer groups and candidate learning rates."""

    optimizer: str
    trainable_scope: str
    head_learning_rate: float
    weight_decay: float
    encoder_learning_rate: float | None = None
    transformer_learning_rate: float | None = None
    backend_learning_rate: float | None = None
    layerwise_learning_rate_decay: float = 1.0
    selection_status: str = ""

    @classmethod
    def from_mapping(
        cls,
        model_name: str,
        values: Mapping[str, Any],
    ) -> "OptimizationSpec":
        """Validate the policy shape required by one supported architecture."""
        optimizer = values.get("name")
        if optimizer not in {"adam", "adamw"}:
            raise TrainingSpecificationError(
                "optimization.name must be 'adam' or 'adamw'."
            )
        trainable_scope = _non_empty_text(values, "trainable_scope")
        head_learning_rate = _positive_float(values, "head_learning_rate")
        weight_decay = _non_negative_float(values, "weight_decay")
        selection_status = _non_empty_text(values, "selection_status")

        if model_name in {"ecapa_tdnn", "rawnet3"}:
            if trainable_scope != "complete_encoder_and_new_head":
                raise TrainingSpecificationError(
                    f"{model_name} must train the complete encoder and new head."
                )
            return cls(
                optimizer=optimizer,
                trainable_scope=trainable_scope,
                encoder_learning_rate=_positive_float(
                    values,
                    "encoder_learning_rate",
                ),
                head_learning_rate=head_learning_rate,
                weight_decay=weight_decay,
                selection_status=selection_status,
            )

        if model_name == "wavlm_mhfa":
            if trainable_scope != "transformer_layers_mhfa_and_new_head":
                raise TrainingSpecificationError(
                    "wavlm_mhfa must optimize Transformer layers, MHFA, and the "
                    "new head only."
                )
            decay = _positive_float(values, "layerwise_learning_rate_decay")
            if decay > 1.0:
                raise TrainingSpecificationError(
                    "layerwise_learning_rate_decay must be in (0, 1]."
                )
            return cls(
                optimizer=optimizer,
                trainable_scope=trainable_scope,
                transformer_learning_rate=_positive_float(
                    values,
                    "transformer_learning_rate",
                ),
                backend_learning_rate=_positive_float(
                    values,
                    "backend_learning_rate",
                ),
                head_learning_rate=head_learning_rate,
                weight_decay=weight_decay,
                layerwise_learning_rate_decay=decay,
                selection_status=selection_status,
            )

        raise TrainingSpecificationError(
            f"Unsupported model optimization policy: {model_name!r}."
        )


@dataclass(frozen=True, slots=True)
class EpochSamplingSpec:
    """Describe the deterministic per-speaker epoch exposure policy."""

    max_utterances_per_speaker: int
    max_speakers_per_epoch: int | None

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "EpochSamplingSpec":
        """Require the accepted rotating SHA-256 speaker cap."""
        if values.get("name") != "deterministic_speaker_cap":
            raise TrainingSpecificationError(
                "training.epoch_sampling.name must be "
                "'deterministic_speaker_cap'."
            )
        if values.get("selection_status") != (
            "accepted_balance_and_runtime_control"
        ):
            raise TrainingSpecificationError(
                "Epoch sampling must retain its accepted status."
            )
        max_speakers = _non_negative_integer(
            values,
            "max_speakers_per_epoch",
        )
        return cls(
            max_utterances_per_speaker=_positive_integer(
                values,
                "max_utterances_per_speaker",
            ),
            # Zero is the explicit configuration representation of no cap.
            max_speakers_per_epoch=max_speakers or None,
        )


@dataclass(frozen=True, slots=True)
class LifecycleSpec:
    """Describe one bounded, checkpointed training lifecycle."""

    max_epochs: int
    early_stopping_patience: int
    minimum_eer_improvement: float
    checkpoint_every_steps: int
    num_workers: int
    pin_memory: bool
    initial_loss_scale: float

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "LifecycleSpec":
        """Validate runtime limits and exact-resume controls."""
        return cls(
            max_epochs=_positive_integer(values, "max_epochs"),
            early_stopping_patience=_positive_integer(
                values,
                "early_stopping_patience",
            ),
            minimum_eer_improvement=_non_negative_float(
                values,
                "minimum_eer_improvement",
            ),
            checkpoint_every_steps=_positive_integer(
                values,
                "checkpoint_every_steps",
            ),
            num_workers=_non_negative_integer(values, "num_workers"),
            pin_memory=_boolean(values, "pin_memory"),
            initial_loss_scale=_positive_float(values, "initial_loss_scale"),
        )


@dataclass(frozen=True, slots=True)
class EvaluationSpec:
    """Describe deterministic Validation crops and extraction loading."""

    segment_samples: int
    segment_count: int
    utterance_batch_size: int
    num_workers: int
    pin_memory: bool

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "EvaluationSpec":
        """Require the accepted FP16 every-epoch Validation boundary."""
        if values.get("mixed_precision") != "fp16":
            raise TrainingSpecificationError(
                "evaluation.mixed_precision must equal 'fp16'."
            )
        if values.get("validation_frequency_epochs") != 1:
            raise TrainingSpecificationError(
                "Validation must run after every completed epoch."
            )
        return cls(
            segment_samples=_positive_integer(values, "segment_samples"),
            segment_count=_positive_integer(values, "segment_count"),
            utterance_batch_size=_positive_integer(
                values,
                "utterance_batch_size",
            ),
            num_workers=_non_negative_integer(values, "num_workers"),
            pin_memory=_boolean(values, "pin_memory"),
        )


@dataclass(frozen=True, slots=True)
class TrainingSpecification:
    """Join the shared loss contract to one architecture update policy."""

    objective: AamSoftmaxSpec
    batch: BatchSpec
    optimization: OptimizationSpec
    mixed_precision: str
    gradient_clip_norm: float
    training_segment_samples: int
    epoch_sampling: EpochSamplingSpec
    lifecycle: LifecycleSpec
    evaluation: EvaluationSpec
    group_by_audio_path: bool

    @classmethod
    def from_resolved_config(
        cls,
        config: Mapping[str, Any],
    ) -> "TrainingSpecification":
        """Parse one fully resolved experiment mapping."""
        training = _mapping(config, "training")
        model = _mapping(config, "model")
        optimization = _mapping(config, "optimization")
        model_name = _non_empty_text(model, "name")
        mixed_precision = training.get("mixed_precision")
        if mixed_precision not in {"disabled", "fp16", "bf16"}:
            raise TrainingSpecificationError(
                "training.mixed_precision must be disabled, fp16, or bf16."
            )
        return cls(
            objective=AamSoftmaxSpec.from_mapping(
                _mapping(training, "objective")
            ),
            batch=BatchSpec.from_mapping(_mapping(training, "batch")),
            optimization=OptimizationSpec.from_mapping(
                model_name,
                optimization,
            ),
            mixed_precision=str(mixed_precision),
            gradient_clip_norm=_positive_float(
                training,
                "gradient_clip_norm",
            ),
            training_segment_samples=_positive_integer(
                _mapping(training, "audio"),
                "segment_samples",
            ),
            epoch_sampling=EpochSamplingSpec.from_mapping(
                _mapping(training, "epoch_sampling")
            ),
            lifecycle=LifecycleSpec.from_mapping(
                _mapping(training, "lifecycle")
            ),
            evaluation=EvaluationSpec.from_mapping(
                _mapping(config, "evaluation")
            ),
            group_by_audio_path=_boolean(
                _mapping(config, "loader"),
                "group_by_audio_path",
            ),
        )


def _mapping(values: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Read one required nested mapping."""
    value = values.get(key)
    if not isinstance(value, Mapping):
        raise TrainingSpecificationError(f"{key} must be a configuration table.")
    return value


def _non_empty_text(values: Mapping[str, Any], key: str) -> str:
    """Read one required non-empty string."""
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TrainingSpecificationError(f"{key} must be non-empty text.")
    return value


def _positive_float(values: Mapping[str, Any], key: str) -> float:
    """Read one finite numeric value strictly greater than zero."""
    value = _numeric(values, key)
    if value <= 0.0:
        raise TrainingSpecificationError(f"{key} must be positive.")
    return value


def _non_negative_float(values: Mapping[str, Any], key: str) -> float:
    """Read one finite numeric value greater than or equal to zero."""
    value = _numeric(values, key)
    if value < 0.0:
        raise TrainingSpecificationError(f"{key} must be non-negative.")
    return value


def _numeric(values: Mapping[str, Any], key: str) -> float:
    """Normalize one finite integer or float while rejecting booleans."""
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingSpecificationError(f"{key} must be numeric.")
    normalized = float(value)
    if not (-float("inf") < normalized < float("inf")):
        raise TrainingSpecificationError(f"{key} must be finite.")
    return normalized


def _positive_integer(values: Mapping[str, Any], key: str) -> int:
    """Read one strict integer greater than zero."""
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TrainingSpecificationError(f"{key} must be a positive integer.")
    return value


def _non_negative_integer(values: Mapping[str, Any], key: str) -> int:
    """Read one strict integer greater than or equal to zero."""
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TrainingSpecificationError(
            f"{key} must be a non-negative integer."
        )
    return value


def _boolean(values: Mapping[str, Any], key: str) -> bool:
    """Read one required boolean value."""
    value = values.get(key)
    if not isinstance(value, bool):
        raise TrainingSpecificationError(f"{key} must be boolean.")
    return value
