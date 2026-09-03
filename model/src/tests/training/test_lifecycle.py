"""Tests for deterministic epoch planning, resume, and model selection."""

from __future__ import annotations

import pytest

from speaker_recognition.training.lifecycle import (
    DeterministicEpochBatchSampler,
    DeterministicGroupedEpochBatchSampler,
    EarlyStoppingPolicy,
    TrainingCursor,
    TrainingLifecycleError,
    TrainingRunIdentity,
    complete_epoch,
    record_training_batch,
    validate_epoch_history,
)


def _sampler(
    *,
    epoch_index: int = 0,
    start_batch_index: int = 0,
    drop_last: bool = False,
) -> DeterministicEpochBatchSampler:
    """Return one small reproducible batch plan."""
    return DeterministicEpochBatchSampler(
        dataset_size=10,
        batch_size=4,
        seed=42,
        epoch_index=epoch_index,
        start_batch_index=start_batch_index,
        drop_last=drop_last,
    )


def test_epoch_plan_is_deterministic_and_complete() -> None:
    """The same run/epoch must visit every item exactly once."""
    first = list(_sampler())
    second = list(_sampler())

    assert first == second
    assert [len(batch) for batch in first] == [4, 4, 2]
    flattened = [index for batch in first for index in batch]
    assert sorted(flattened) == list(range(10))


def test_epoch_changes_order_without_using_global_rng() -> None:
    """A new epoch should reshuffle while remaining independently repeatable."""
    assert list(_sampler(epoch_index=1)) != list(_sampler(epoch_index=0))
    assert list(_sampler(epoch_index=1)) == list(_sampler(epoch_index=1))


def test_mid_epoch_resume_yields_exact_unprocessed_suffix() -> None:
    """A cursor should neither repeat nor skip batches after interruption."""
    complete = list(_sampler())
    resumed = list(_sampler(start_batch_index=2))

    assert resumed == complete[2:]
    assert len(_sampler(start_batch_index=2)) == 1


def test_drop_last_is_explicit_and_deterministic() -> None:
    """Dropping the final partial batch must be an auditable policy choice."""
    batches = list(_sampler(drop_last=True))

    assert [len(batch) for batch in batches] == [4, 4]
    assert len({index for batch in batches for index in batch}) == 8


def test_grouped_epoch_keeps_storage_groups_contiguous() -> None:
    """Shard-local ordering should preserve full coverage and cache locality."""
    groups = ((0, 1, 2), (3, 4), (5, 6, 7, 8, 9))
    sampler = DeterministicGroupedEpochBatchSampler(
        index_groups=groups,
        batch_size=4,
        seed=42,
        epoch_index=0,
    )

    batches = list(sampler)
    flattened = [index for batch in batches for index in batch]

    assert sorted(flattened) == list(range(10))
    assert [len(batch) for batch in batches] == [4, 4, 2]
    for group in groups:
        positions = sorted(flattened.index(index) for index in group)
        assert positions == list(range(positions[0], positions[-1] + 1))


def test_grouped_epoch_resume_matches_complete_suffix() -> None:
    """Shard-aware ordering must retain the same exact checkpoint cursor rule."""
    settings = {
        "index_groups": ((0, 1, 2), (3, 4), (5, 6, 7, 8, 9)),
        "batch_size": 4,
        "seed": 42,
        "epoch_index": 3,
    }
    complete = list(DeterministicGroupedEpochBatchSampler(**settings))
    resumed = list(
        DeterministicGroupedEpochBatchSampler(
            **settings,
            start_batch_index=1,
        )
    )

    assert resumed == complete[1:]


def test_grouped_epoch_rejects_incomplete_index_partition() -> None:
    """Missing or duplicate indexes could silently corrupt an epoch."""
    with pytest.raises(TrainingLifecycleError, match="partition"):
        DeterministicGroupedEpochBatchSampler(
            index_groups=((0, 1), (1, 2)),
            batch_size=2,
            seed=42,
            epoch_index=0,
        )


def test_cursor_accumulates_weighted_batch_metrics() -> None:
    """Unequal final batches must contribute by their real example count."""
    cursor = record_training_batch(
        TrainingCursor(),
        batch_size=2,
        loss=2.0,
        accuracy=0.5,
    )
    cursor = record_training_batch(
        cursor,
        batch_size=1,
        loss=4.0,
        accuracy=1.0,
    )
    next_cursor, summary, should_stop = complete_epoch(
        cursor,
        validation_eer=0.2,
        validation_min_dcf=0.3,
        policy=EarlyStoppingPolicy(patience=2),
    )

    assert summary.training_loss == pytest.approx(8.0 / 3.0)
    assert summary.training_accuracy == pytest.approx(2.0 / 3.0)
    assert summary.improved is True
    assert next_cursor.epoch_index == 1
    assert next_cursor.next_batch_index == 0
    assert next_cursor.best_validation_eer == pytest.approx(0.2)
    assert next_cursor.best_validation_min_dcf == pytest.approx(0.3)
    assert should_stop is False


def test_min_dcf_breaks_validation_eer_tie() -> None:
    """The documented minDCF tie-breaker should replace an equal-EER model."""
    cursor = TrainingCursor(
        epoch_index=1,
        next_batch_index=1,
        global_step=2,
        epoch_loss_sum=1.0,
        epoch_accuracy_sum=0.5,
        epoch_example_count=1,
        best_validation_eer=0.2,
        best_validation_min_dcf=0.4,
        best_epoch_index=0,
    )
    next_cursor, summary, _ = complete_epoch(
        cursor,
        validation_eer=0.2,
        validation_min_dcf=0.35,
        policy=EarlyStoppingPolicy(
            patience=2,
            minimum_eer_improvement=0.001,
        ),
    )

    assert summary.improved is True
    assert next_cursor.best_epoch_index == 1
    assert next_cursor.best_validation_min_dcf == pytest.approx(0.35)


def test_patience_stops_after_consecutive_non_improvements() -> None:
    """Only Validation history may trigger early stopping."""
    policy = EarlyStoppingPolicy(patience=2)
    cursor = TrainingCursor(
        epoch_index=1,
        next_batch_index=1,
        global_step=2,
        epoch_loss_sum=1.0,
        epoch_accuracy_sum=0.5,
        epoch_example_count=1,
        best_validation_eer=0.1,
        best_validation_min_dcf=0.2,
        best_epoch_index=0,
    )
    cursor, _, first_stop = complete_epoch(
        cursor,
        validation_eer=0.11,
        validation_min_dcf=0.21,
        policy=policy,
    )
    cursor = record_training_batch(
        cursor,
        batch_size=1,
        loss=1.0,
        accuracy=0.5,
    )
    _, _, second_stop = complete_epoch(
        cursor,
        validation_eer=0.12,
        validation_min_dcf=0.22,
        policy=policy,
    )

    assert first_stop is False
    assert second_stop is True


def test_run_identity_rejects_non_sha_fingerprint() -> None:
    """Resume guards must not accept informal or truncated fingerprints."""
    with pytest.raises(TrainingLifecycleError, match="SHA-256"):
        TrainingRunIdentity(
            model_name="ecapa_tdnn",
            dataset_name="tidyvoice",
            config_sha256="short",
            manifest_sha256="a" * 64,
            seed=42,
        )


def test_epoch_history_must_be_contiguous() -> None:
    """A missing epoch record must fail checkpoint restoration."""
    history = [
        {
            "epoch_index": 1,
            "global_step": 2,
            "training_loss": 1.0,
            "training_accuracy": 0.5,
            "validation_eer": 0.2,
            "validation_min_dcf": 0.3,
            "improved": True,
        }
    ]

    with pytest.raises(TrainingLifecycleError, match="contiguous"):
        validate_epoch_history(history)
