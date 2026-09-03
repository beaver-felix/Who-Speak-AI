"""Structured local and optional W&B logging behind one small interface."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


class TrainingLoggerError(ValueError):
    """Raised when a metric record is malformed or tracking cannot start."""


class RunLogger(Protocol):
    """Minimal logging boundary consumed by the shared training engine."""

    def log(self, metrics: Mapping[str, float], *, step: int) -> None:
        """Record finite scalar metrics at one monotonically increasing step."""
        ...

    def finish(self) -> None:
        """Flush and close the run."""
        ...


class NullRunLogger:
    """Discard metrics while preserving the engine's logging contract."""

    def log(self, metrics: Mapping[str, float], *, step: int) -> None:
        """Validate metrics even when external tracking is disabled."""
        _validate_record(metrics, step)

    def finish(self) -> None:
        """Require no cleanup."""


class JsonlRunLogger:
    """Append an offline, machine-readable metric event stream."""

    def __init__(
        self,
        path: str | Path,
        *,
        resume_step: int | None = None,
    ) -> None:
        """Prepare a new log or trim an interrupted log to its checkpoint."""
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if resume_step is not None and (
            isinstance(resume_step, bool)
            or not isinstance(resume_step, int)
            or resume_step < 0
        ):
            raise TrainingLoggerError(
                "resume_step must be a non-negative integer."
            )
        self._last_step = -1 if resume_step is None else resume_step
        if self.path.exists():
            events = _read_jsonl_events(self.path)
            if resume_step is None and events:
                raise TrainingLoggerError(
                    "Existing JSONL history requires an explicit resume_step."
                )
            if resume_step is not None:
                retained = [
                    event
                    for event in events
                    if int(event["step"]) <= resume_step
                ]
                _rewrite_jsonl(self.path, retained)

    def log(self, metrics: Mapping[str, float], *, step: int) -> None:
        """Append one finite JSON event and reject decreasing step numbers."""
        normalized = _validate_record(metrics, step)
        if step < self._last_step:
            raise TrainingLoggerError("Logging steps must not decrease.")
        event = {"step": step, "metrics": normalized}
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(
                json.dumps(
                    event,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            )
            stream.flush()
        self._last_step = step

    def finish(self) -> None:
        """Require no persistent file handle cleanup."""


class WandbRunLogger:
    """Forward the same structured metrics to one resumable W&B run."""

    def __init__(
        self,
        *,
        project: str,
        run_id: str,
        run_name: str,
        config: Mapping[str, object],
        directory: str | Path,
        resume: bool,
        resume_step: int | None = None,
        mode: str = "online",
    ) -> None:
        """Start or strictly resume one explicitly identified W&B run."""
        for value, field_name in (
            (project, "project"),
            (run_id, "run_id"),
            (run_name, "run_name"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise TrainingLoggerError(f"{field_name} must be non-empty.")
        if mode not in {"online", "offline", "disabled"}:
            raise TrainingLoggerError(
                "W&B mode must be online, offline, or disabled."
            )
        if not isinstance(resume, bool):
            raise TrainingLoggerError("resume must be boolean.")
        if resume and (
            isinstance(resume_step, bool)
            or not isinstance(resume_step, int)
            or resume_step < 0
        ):
            raise TrainingLoggerError(
                "A resumed W&B run requires a non-negative resume_step."
            )
        if not resume and resume_step is not None:
            raise TrainingLoggerError(
                "A fresh W&B run must not define resume_step."
            )

        try:
            import wandb
        except ModuleNotFoundError as error:  # pragma: no cover - Kaggle gate.
            raise ModuleNotFoundError(
                "W&B tracking requires `python -m pip install -e '.[tracking]'`."
            ) from error

        tracking_directory = Path(directory).expanduser().resolve()
        tracking_directory.mkdir(parents=True, exist_ok=True)
        self._run: Any = wandb.init(
            project=project,
            id=run_id,
            name=run_name,
            config=dict(config),
            dir=str(tracking_directory),
            mode=mode,
            resume="must" if resume else "never",
        )
        if self._run is None:
            raise TrainingLoggerError("wandb.init did not return a run.")
        self._last_step = -1 if resume_step is None else resume_step

    def log(self, metrics: Mapping[str, float], *, step: int) -> None:
        """Log finite scalars while preserving a single global-step axis."""
        normalized = _validate_record(metrics, step)
        if step < self._last_step:
            raise TrainingLoggerError("Logging steps must not decrease.")
        self._run.log(normalized, step=step)
        self._last_step = step

    def finish(self) -> None:
        """Flush the W&B run explicitly."""
        self._run.finish()


class CompositeRunLogger:
    """Send identical records to local JSONL and W&B without engine coupling."""

    def __init__(self, loggers: Sequence[RunLogger]) -> None:
        """Require at least one concrete logging destination."""
        self._loggers = tuple(loggers)
        if not self._loggers:
            raise TrainingLoggerError("At least one logger is required.")

    def log(self, metrics: Mapping[str, float], *, step: int) -> None:
        """Forward one event to every logger in deterministic order."""
        for logger in self._loggers:
            logger.log(metrics, step=step)

    def finish(self) -> None:
        """Finish every logger even if a later one raises."""
        first_error: Exception | None = None
        for logger in self._loggers:
            try:
                logger.finish()
            except Exception as error:  # pragma: no cover - integration path.
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error


def _validate_record(
    metrics: Mapping[str, float],
    step: int,
) -> dict[str, float]:
    """Normalize a non-empty collection of finite scalar metrics."""
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise TrainingLoggerError("step must be a non-negative integer.")
    if not metrics:
        raise TrainingLoggerError("metrics must not be empty.")
    normalized: dict[str, float] = {}
    for key, value in metrics.items():
        if not isinstance(key, str) or not key.strip():
            raise TrainingLoggerError("Metric names must be non-empty strings.")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise TrainingLoggerError(f"Metric {key!r} must be finite.")
        normalized[key] = float(value)
    return normalized


def _read_jsonl_events(path: Path) -> list[dict[str, object]]:
    """Read and validate an existing local metric stream before resume."""
    events: list[dict[str, object]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise TrainingLoggerError(
                f"Invalid JSONL event at line {line_number}."
            ) from error
        if (
            not isinstance(event, dict)
            or not isinstance(event.get("step"), int)
            or not isinstance(event.get("metrics"), dict)
        ):
            raise TrainingLoggerError(
                f"Invalid JSONL event structure at line {line_number}."
            )
        _validate_record(event["metrics"], event["step"])
        events.append(event)
    return events


def _rewrite_jsonl(path: Path, events: Sequence[Mapping[str, object]]) -> None:
    """Atomically remove metric events newer than an authoritative checkpoint."""
    partial = path.with_name(path.name + ".part")
    try:
        with partial.open("w", encoding="utf-8", newline="\n") as stream:
            for event in events:
                stream.write(
                    json.dumps(
                        event,
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n"
                )
            stream.flush()
        partial.replace(path)
    finally:
        if partial.exists():
            partial.unlink()
