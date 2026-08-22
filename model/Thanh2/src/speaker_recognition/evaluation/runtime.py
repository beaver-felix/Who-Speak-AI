"""Kaggle-only embedding extraction and model-latency measurement."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
import numpy as np

try:
    import torch
except ModuleNotFoundError as error:  # pragma: no cover - Kaggle dependency gate.
    raise ModuleNotFoundError(
        "Embedding extraction requires Kaggle's CUDA-matched PyTorch build."
    ) from error

from speaker_recognition.data.dataset import (
    EvaluationSpeakerDataset,
    collate_evaluation_samples,
)
from speaker_recognition.evaluation.embeddings import (
    EmbeddingTable,
    aggregate_crop_embeddings,
)


class EvaluationRuntimeError(RuntimeError):
    """Raised when GPU extraction or latency measurement is unsafe."""


@dataclass(frozen=True, slots=True)
class ExtractionSettings:
    """Describe deterministic, model-independent evaluation loading."""

    utterance_batch_size: int
    num_workers: int = 2
    pin_memory: bool = True
    mixed_precision: str = "fp16"

    def __post_init__(self) -> None:
        """Validate bounded evaluation runtime controls."""
        if (
            isinstance(self.utterance_batch_size, bool)
            or not isinstance(self.utterance_batch_size, int)
            or self.utterance_batch_size <= 0
        ):
            raise ValueError("utterance_batch_size must be positive.")
        if (
            isinstance(self.num_workers, bool)
            or not isinstance(self.num_workers, int)
            or self.num_workers < 0
        ):
            raise ValueError("num_workers must be non-negative.")
        if not isinstance(self.pin_memory, bool):
            raise ValueError("pin_memory must be boolean.")
        if self.mixed_precision != "fp16":
            raise ValueError("The accepted T4 evaluation baseline uses fp16.")


@dataclass(frozen=True, slots=True)
class ExtractionStatistics:
    """Report one-pass cache coverage and evaluation throughput."""

    utterance_count: int
    crop_count: int
    batch_count: int
    wall_seconds: float
    model_seconds: float
    fp32_fallback_batch_count: int = 0

    def __post_init__(self) -> None:
        """Reject incomplete or non-finite extraction evidence."""
        for value, field_name in (
            (self.utterance_count, "utterance_count"),
            (self.crop_count, "crop_count"),
            (self.batch_count, "batch_count"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be positive.")
        if (
            isinstance(self.fp32_fallback_batch_count, bool)
            or not isinstance(self.fp32_fallback_batch_count, int)
            or self.fp32_fallback_batch_count < 0
            or self.fp32_fallback_batch_count > self.batch_count
        ):
            raise ValueError(
                "fp32_fallback_batch_count must lie within [0, batch_count]."
            )
        for value, field_name in (
            (self.wall_seconds, "wall_seconds"),
            (self.model_seconds, "model_seconds"),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and positive.")

    def to_dict(self) -> dict[str, float | int]:
        """Return raw counts plus clearly named derived rates."""
        return {
            **asdict(self),
            "wall_utterances_per_second": (
                self.utterance_count / self.wall_seconds
            ),
            "model_crops_per_second": self.crop_count / self.model_seconds,
            "mean_model_ms_per_crop": (
                1000.0 * self.model_seconds / self.crop_count
            ),
        }


@dataclass(frozen=True, slots=True)
class LatencyStatistics:
    """Summarize batch-one model latency on a preloaded canonical crop."""

    warmup_iterations: int
    measured_iterations: int
    mean_ms: float
    median_ms: float
    p95_ms: float

    def to_dict(self) -> dict[str, float | int]:
        """Return JSON-ready latency evidence."""
        return asdict(self)


def extract_utterance_embeddings(
    adapter: torch.nn.Module,
    dataset: EvaluationSpeakerDataset,
    *,
    settings: ExtractionSettings,
    device: str = "cuda:0",
) -> tuple[EmbeddingTable, ExtractionStatistics]:
    """Encode every evaluation utterance once and cache one aggregate vector."""
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise EvaluationRuntimeError("Accepted evaluation requires CUDA.")
    metadata = getattr(adapter, "metadata", None)
    embedding_dim = getattr(metadata, "embedding_dim", None)
    if (
        isinstance(embedding_dim, bool)
        or not isinstance(embedding_dim, int)
        or embedding_dim <= 0
    ):
        raise EvaluationRuntimeError(
            "Adapter metadata must define a positive embedding dimension."
        )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=settings.utterance_batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
        collate_fn=collate_evaluation_samples,
        pin_memory=settings.pin_memory,
        drop_last=False,
        persistent_workers=False,
    )
    cached = np.empty((len(dataset), embedding_dim), dtype=np.float32)
    utterance_ids: list[str] = []
    utterance_offset = 0
    crop_count = 0
    batch_count = 0
    model_milliseconds = 0.0
    fp32_fallback_batch_count = 0
    was_training = bool(adapter.training)
    adapter.eval()
    torch.cuda.synchronize(device)
    wall_started = time.perf_counter()
    try:
        with torch.inference_mode():
            for batch in data_loader:
                waveforms = torch.from_numpy(batch.waveforms).to(
                    device,
                    non_blocking=settings.pin_memory,
                )
                if not bool(torch.isfinite(waveforms).all()):
                    raise EvaluationRuntimeError(
                        "Evaluation audio contains non-finite values."
                    )
                started = torch.cuda.Event(enable_timing=True)
                finished = torch.cuda.Event(enable_timing=True)
                started.record()
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=True,
                ):
                    crop_embeddings = adapter(waveforms)
                if not bool(torch.isfinite(crop_embeddings).all()):
                    model_name = getattr(metadata, "name", None)
                    if model_name != "wavlm_mhfa":
                        raise EvaluationRuntimeError(
                            "Evaluation embedding became non-finite for "
                            f"utterances: {list(batch.utterance_ids[:6])}."
                        )
                    # WavLM may overflow for isolated real-audio batches under
                    # FP16. Evaluation is stochastic-free, so recomputing only
                    # the affected batch in FP32 preserves exact membership,
                    # ordering, crops, weights, and scoring.
                    with torch.autocast(device_type="cuda", enabled=False):
                        crop_embeddings = adapter(waveforms.float())
                    fp32_fallback_batch_count += 1
                    if not bool(torch.isfinite(crop_embeddings).all()):
                        raise EvaluationRuntimeError(
                            "WavLM evaluation embedding remained non-finite "
                            "after FP32 fallback for utterances: "
                            f"{list(batch.utterance_ids[:6])}."
                        )
                finished.record()
                finished.synchronize()
                model_milliseconds += float(started.elapsed_time(finished))

                crop_values = crop_embeddings.float().cpu().numpy()
                aggregate = aggregate_crop_embeddings(
                    crop_values,
                    batch.segment_offsets,
                )
                batch_utterances = len(batch.utterance_ids)
                if aggregate.shape != (batch_utterances, embedding_dim):
                    raise EvaluationRuntimeError(
                        "Aggregate embedding shape differs from adapter metadata."
                    )
                end = utterance_offset + batch_utterances
                cached[utterance_offset:end] = aggregate
                utterance_ids.extend(batch.utterance_ids)
                utterance_offset = end
                crop_count += int(batch.waveforms.shape[0])
                batch_count += 1
    finally:
        adapter.train(was_training)
    torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - wall_started

    if utterance_offset != len(dataset) or len(utterance_ids) != len(dataset):
        raise EvaluationRuntimeError(
            "Embedding extraction did not cover the complete dataset."
        )
    table = EmbeddingTable(
        utterance_ids=tuple(utterance_ids),
        embeddings=np.ascontiguousarray(cached, dtype=np.float32),
    )
    statistics = ExtractionStatistics(
        utterance_count=len(dataset),
        crop_count=crop_count,
        batch_count=batch_count,
        wall_seconds=wall_seconds,
        model_seconds=model_milliseconds / 1000.0,
        fp32_fallback_batch_count=fp32_fallback_batch_count,
    )
    return table, statistics


def benchmark_single_crop_latency(
    adapter: torch.nn.Module,
    waveform: np.ndarray,
    *,
    warmup_iterations: int = 10,
    measured_iterations: int = 50,
    device: str = "cuda:0",
) -> LatencyStatistics:
    """Measure model-only batch-one latency on preloaded canonical audio."""
    for value, field_name in (
        (warmup_iterations, "warmup_iterations"),
        (measured_iterations, "measured_iterations"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field_name} must be positive.")
    values = np.asarray(waveform, dtype=np.float32)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("waveform must be one non-empty finite float vector.")
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise EvaluationRuntimeError("Latency measurement requires CUDA.")

    tensor = torch.from_numpy(np.ascontiguousarray(values))[None, :].to(device)
    was_training = bool(adapter.training)
    adapter.eval()
    timings: list[float] = []
    try:
        with torch.inference_mode():
            for _ in range(warmup_iterations):
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=True,
                ):
                    adapter(tensor)
            torch.cuda.synchronize(device)
            for _ in range(measured_iterations):
                started = torch.cuda.Event(enable_timing=True)
                finished = torch.cuda.Event(enable_timing=True)
                started.record()
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=True,
                ):
                    adapter(tensor)
                finished.record()
                finished.synchronize()
                timings.append(float(started.elapsed_time(finished)))
    finally:
        adapter.train(was_training)

    timing_array = np.asarray(timings, dtype=np.float64)
    if not np.isfinite(timing_array).all() or np.any(timing_array <= 0.0):
        raise EvaluationRuntimeError("Latency timings must be finite and positive.")
    return LatencyStatistics(
        warmup_iterations=warmup_iterations,
        measured_iterations=measured_iterations,
        mean_ms=float(timing_array.mean()),
        median_ms=float(np.median(timing_array)),
        p95_ms=float(np.percentile(timing_array, 95)),
    )
