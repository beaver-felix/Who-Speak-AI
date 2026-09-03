"""Kaggle validation callback for the shared restart-safe training engine."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

try:
    import torch
except ModuleNotFoundError as error:  # pragma: no cover - Kaggle dependency gate.
    raise ModuleNotFoundError(
        "Validation evaluation requires Kaggle's CUDA-matched PyTorch build."
    ) from error

from speaker_recognition.data.dataset import EvaluationSpeakerDataset
from speaker_recognition.data.manifest import Split
from speaker_recognition.evaluation.protocol import evaluate_embedding_table
from speaker_recognition.evaluation.runtime import (
    ExtractionSettings,
    extract_utterance_embeddings,
)
from speaker_recognition.evaluation.trials import (
    VerificationTrial,
    trial_list_sha256,
)


class ValidationCallbackError(RuntimeError):
    """Raised when Validation could be incomplete, leaky, or unauditable."""


class VerificationValidationCallback:
    """Extract, score, persist, and return one epoch's Validation metrics."""

    def __init__(
        self,
        *,
        dataset: EvaluationSpeakerDataset,
        trials: Sequence[VerificationTrial],
        expected_trial_sha256: str,
        extraction_settings: ExtractionSettings,
        output_directory: str | Path,
        evidence_context: Mapping[str, object],
        device: str = "cuda:0",
    ) -> None:
        """Validate protocol coverage before any expensive GPU work."""
        if dataset.split is not Split.VALIDATION:
            raise ValidationCallbackError(
                "Training callbacks may evaluate Validation only, never Test."
            )
        snapshot = tuple(trials)
        if not snapshot:
            raise ValidationCallbackError("Validation trials must not be empty.")
        if trial_list_sha256(snapshot) != expected_trial_sha256:
            raise ValidationCallbackError(
                "Validation trial fingerprint differs from the declaration."
            )
        trial_utterance_ids = {
            utterance_id
            for trial in snapshot
            for utterance_id in (
                trial.left_utterance_id,
                trial.right_utterance_id,
            )
        }
        dataset_utterance_ids = {
            record.utterance_id for record in dataset.records
        }
        if dataset_utterance_ids != trial_utterance_ids:
            raise ValidationCallbackError(
                "Validation dataset must contain exactly the trial utterances."
            )
        context = dict(evidence_context)
        if not context or any(
            not isinstance(name, str) or not name.strip()
            for name in context
        ):
            raise ValidationCallbackError(
                "Evidence context requires non-empty string keys."
            )
        try:
            json.dumps(context, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValidationCallbackError(
                "Evidence context must be finite and JSON-compatible."
            ) from error

        self._dataset = dataset
        self._trials = snapshot
        self._expected_trial_sha256 = expected_trial_sha256
        self._extraction_settings = extraction_settings
        self._output_directory = Path(output_directory)
        self._evidence_context = context
        self._device = device

    def __call__(
        self,
        adapter: torch.nn.Module,
        epoch_index: int,
    ) -> Mapping[str, float]:
        """Evaluate one epoch and atomically persist complete JSON evidence."""
        if (
            isinstance(epoch_index, bool)
            or not isinstance(epoch_index, int)
            or epoch_index < 0
        ):
            raise ValidationCallbackError(
                "epoch_index must be a non-negative integer."
            )
        table, extraction = extract_utterance_embeddings(
            adapter,
            self._dataset,
            settings=self._extraction_settings,
            device=self._device,
        )
        result = evaluate_embedding_table(
            table,
            self._trials,
            expected_trial_sha256=self._expected_trial_sha256,
        )
        artifact = {
            **result.to_artifact(),
            "evaluation": {
                "partition": Split.VALIDATION.value,
                "epoch_index": epoch_index,
                "segment_samples": self._dataset.segment_samples,
                "segment_count": self._dataset.segment_count,
                "embedding_extraction": extraction.to_dict(),
            },
            "context": self._evidence_context,
        }
        artifact_path = self._output_directory / (
            f"validation_epoch_{epoch_index:04d}.json"
        )
        _atomic_write_json(artifact_path, artifact)

        returned = {
            name: float(value)
            for name, value in result.metrics.to_flat_dict().items()
        }
        returned.update(
            {
                f"extraction/{name}": float(value)
                for name, value in extraction.to_dict().items()
            }
        )
        if not all(math.isfinite(value) for value in returned.values()):
            raise ValidationCallbackError(
                "Validation callback produced a non-finite metric."
            )
        return returned


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Write strict JSON to a sibling partial file before atomic promotion."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.part")
    serialized = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    try:
        with partial.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, path)
    finally:
        if partial.exists():
            partial.unlink()
