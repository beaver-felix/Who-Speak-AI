"""Tests for deterministic, metadata-balanced group splitting."""

from collections import OrderedDict

import pytest

from speaker_recognition.data.manifest import Split
from speaker_recognition.data.splitting import (
    BalancedSplitError,
    make_balanced_validation_test_split,
)


@pytest.fixture
def complementary_profiles() -> dict[str, dict[str, int]]:
    """Provide speakers whose optimal partition has perfect balance."""
    return {
        "speaker_1": {"en": 90, "vi": 10},
        "speaker_2": {"en": 10, "vi": 90},
        "speaker_3": {"en": 80, "vi": 20},
        "speaker_4": {"en": 20, "vi": 80},
    }


def test_split_assigns_every_speaker_once(
    complementary_profiles: dict[str, dict[str, int]],
) -> None:
    """Every speaker must belong exclusively to validation or test."""
    result = make_balanced_validation_test_split(
        complementary_profiles,
        seed=42,
    )

    assert set(result.assignments) == set(complementary_profiles)
    assert set(result.assignments.values()) == {
        Split.VALIDATION,
        Split.TEST,
    }
    assert result.validation_group_count == 2
    assert result.test_group_count == 2


def test_split_finds_balanced_complementary_partition(
    complementary_profiles: dict[str, dict[str, int]],
) -> None:
    """The optimization should balance utterances and label proportions."""
    result = make_balanced_validation_test_split(
        complementary_profiles,
        seed=42,
    )

    assert result.validation_item_count == 200
    assert result.test_item_count == 200
    assert result.item_imbalance == pytest.approx(0.0)
    assert result.max_label_proportion_difference == pytest.approx(0.0)
    assert result.objective == pytest.approx(0.0)


def test_split_is_independent_of_mapping_insertion_order(
    complementary_profiles: dict[str, dict[str, int]],
) -> None:
    """Reordering input metadata must not change a seeded result."""
    forward = make_balanced_validation_test_split(
        complementary_profiles,
        seed=42,
    )
    reversed_profiles = OrderedDict(
        reversed(tuple(complementary_profiles.items()))
    )
    reversed_result = make_balanced_validation_test_split(
        reversed_profiles,
        seed=42,
    )

    assert reversed_result.assignments == forward.assignments
    assert reversed_result.objective == forward.objective


def test_split_rejects_fewer_than_two_groups() -> None:
    """A validation/test partition requires at least two speakers."""
    with pytest.raises(BalancedSplitError, match="at least two"):
        make_balanced_validation_test_split(
            {"only_speaker": {"en": 10}},
            seed=42,
        )


@pytest.mark.parametrize("invalid_count", [0, -1])
def test_split_rejects_non_positive_label_counts(
    invalid_count: int,
) -> None:
    """Observed label counts must represent actual utterances."""
    profiles = {
        "speaker_1": {"en": invalid_count},
        "speaker_2": {"vi": 10},
    }

    with pytest.raises(BalancedSplitError, match="positive"):
        make_balanced_validation_test_split(profiles, seed=42)