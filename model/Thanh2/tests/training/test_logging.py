"""Tests for dependency-neutral structured training logging."""

from __future__ import annotations

import json

import pytest

from speaker_recognition.training.logging import (
    JsonlRunLogger,
    NullRunLogger,
    TrainingLoggerError,
)


def test_jsonl_logger_appends_finite_step_records(tmp_path) -> None:
    """Offline metrics should remain machine-readable across process restarts."""
    path = tmp_path / "metrics.jsonl"
    logger = JsonlRunLogger(path)
    logger.log({"train/loss": 2.0}, step=1)
    logger.log({"validation/eer": 0.1}, step=2)
    logger.finish()

    events = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert events == [
        {"metrics": {"train/loss": 2.0}, "step": 1},
        {"metrics": {"validation/eer": 0.1}, "step": 2},
    ]


def test_logger_rejects_nonfinite_metrics() -> None:
    """NaN must not enter local evidence or W&B history."""
    with pytest.raises(TrainingLoggerError, match="finite"):
        NullRunLogger().log({"train/loss": float("nan")}, step=1)


def test_jsonl_logger_rejects_decreasing_step(tmp_path) -> None:
    """A resumed run must retain one monotonic global-step axis."""
    logger = JsonlRunLogger(tmp_path / "metrics.jsonl")
    logger.log({"train/loss": 1.0}, step=2)

    with pytest.raises(TrainingLoggerError, match="decrease"):
        logger.log({"train/loss": 0.9}, step=1)


def test_jsonl_resume_discards_events_newer_than_checkpoint(tmp_path) -> None:
    """A crash after logging must not leave duplicate future history."""
    path = tmp_path / "metrics.jsonl"
    logger = JsonlRunLogger(path)
    logger.log({"train/loss": 2.0}, step=1)
    logger.log({"train/loss": 1.5}, step=2)

    resumed = JsonlRunLogger(path, resume_step=1)
    resumed.log({"train/loss": 1.4}, step=2)
    events = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]

    assert events == [
        {"metrics": {"train/loss": 2.0}, "step": 1},
        {"metrics": {"train/loss": 1.4}, "step": 2},
    ]


def test_existing_jsonl_requires_explicit_resume_step(tmp_path) -> None:
    """A fresh run must not silently append to another experiment."""
    path = tmp_path / "metrics.jsonl"
    JsonlRunLogger(path).log({"train/loss": 1.0}, step=1)

    with pytest.raises(TrainingLoggerError, match="resume_step"):
        JsonlRunLogger(path)
