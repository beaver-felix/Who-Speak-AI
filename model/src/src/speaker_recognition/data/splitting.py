"""Deterministic metadata-balanced group splitting."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from random import Random

from speaker_recognition.data.manifest import Split


class BalancedSplitError(ValueError):
    """Raised when group metadata cannot form a valid balanced split."""


@dataclass(frozen=True, slots=True)
class BalancedSplitResult:
    """Store split assignments and directly auditable diagnostics."""

    assignments: dict[str, Split]
    validation_group_count: int
    test_group_count: int
    validation_item_count: int
    test_item_count: int
    item_imbalance: float
    max_label_proportion_difference: float
    objective: float


def make_balanced_validation_test_split(
    label_counts_by_group: Mapping[str, Mapping[str, int]],
    *,
    seed: int,
    validation_fraction: float = 0.5,
    restarts: int = 64,
    max_swap_passes: int = 8,
) -> BalancedSplitResult:
    """Partition groups while balancing items and label proportions.

    The procedure uses seeded greedy multi-start search followed by pairwise
    swap refinement. Group capacity is fixed before optimization, so speakers
    cannot leak across validation and test.

    The minimized objective is::

        absolute item imbalance
        + maximum label-proportion difference

    Parameters
    ----------
    label_counts_by_group:
        Mapping from speaker ID to observed language counts.
    seed:
        Seed controlling reproducible candidate order and tie-breaking.
    validation_fraction:
        Requested fraction of speakers assigned to validation.
    restarts:
        Number of seeded greedy candidates.
    max_swap_passes:
        Maximum local-improvement passes for the best candidate.

    Returns
    -------
    BalancedSplitResult
        Speaker assignments and balance diagnostics.

    Raises
    ------
    BalancedSplitError
        If profiles or optimization parameters are invalid.
    """
    (
        group_names,
        label_names,
        vectors_by_group,
        total_label_counts,
    ) = _prepare_profiles(label_counts_by_group)

    if not 0.0 < validation_fraction < 1.0:
        raise BalancedSplitError(
            "validation_fraction must be strictly between zero and one."
        )
    if restarts <= 0:
        raise BalancedSplitError("restarts must be positive.")
    if max_swap_passes < 0:
        raise BalancedSplitError("max_swap_passes must be non-negative.")

    group_count = len(group_names)
    validation_group_target = int(
        group_count * validation_fraction + 0.5
    )
    validation_group_target = max(
        1,
        min(group_count - 1, validation_group_target),
    )
    effective_fraction = validation_group_target / group_count

    best_validation_groups: set[str] | None = None
    best_validation_counts: list[int] | None = None
    best_rank: tuple[float, float, float, tuple[str, ...]] | None = None

    for restart_index in range(restarts):
        # Each restart receives a separate deterministic pseudo-random stream.
        rng = Random(seed + restart_index * 1_000_003)
        ordered_groups = list(group_names)
        rng.shuffle(ordered_groups)

        validation_groups: set[str] = set()
        validation_counts = [0] * len(label_names)
        processed_counts = [0] * len(label_names)

        for index, group_name in enumerate(ordered_groups):
            vector = vectors_by_group[group_name]
            processed_counts = [
                current + added
                for current, added in zip(
                    processed_counts,
                    vector,
                    strict=True,
                )
            ]

            groups_remaining = len(ordered_groups) - index
            validation_groups_needed = (
                validation_group_target - len(validation_groups)
            )

            if validation_groups_needed == groups_remaining:
                assign_to_validation = True
            elif validation_groups_needed == 0:
                assign_to_validation = False
            else:
                candidate_validation_counts = [
                    current + added
                    for current, added in zip(
                        validation_counts,
                        vector,
                        strict=True,
                    )
                ]
                validation_score = _partial_balance_score(
                    candidate_validation_counts,
                    processed_counts,
                    total_label_counts,
                    effective_fraction,
                )
                test_score = _partial_balance_score(
                    validation_counts,
                    processed_counts,
                    total_label_counts,
                    effective_fraction,
                )

                if validation_score == test_score:
                    assign_to_validation = rng.random() < 0.5
                else:
                    assign_to_validation = validation_score < test_score

            if assign_to_validation:
                validation_groups.add(group_name)
                validation_counts = [
                    current + added
                    for current, added in zip(
                        validation_counts,
                        vector,
                        strict=True,
                    )
                ]

        objective, item_imbalance, label_difference = _score_counts(
            validation_counts,
            total_label_counts,
        )
        candidate_rank = (
            objective,
            item_imbalance,
            label_difference,
            tuple(sorted(validation_groups)),
        )

        if best_rank is None or candidate_rank < best_rank:
            best_rank = candidate_rank
            best_validation_groups = validation_groups
            best_validation_counts = validation_counts

    if best_validation_groups is None or best_validation_counts is None:
        raise RuntimeError("Balanced split search produced no candidate.")

    (
        best_validation_groups,
        best_validation_counts,
    ) = _improve_with_group_swaps(
        best_validation_groups,
        best_validation_counts,
        group_names=group_names,
        vectors_by_group=vectors_by_group,
        total_label_counts=total_label_counts,
        max_passes=max_swap_passes,
    )

    objective, item_imbalance, label_difference = _score_counts(
        best_validation_counts,
        total_label_counts,
    )
    total_items = sum(total_label_counts)
    validation_items = sum(best_validation_counts)
    test_items = total_items - validation_items

    assignments = {
        group_name: (
            Split.VALIDATION
            if group_name in best_validation_groups
            else Split.TEST
        )
        for group_name in group_names
    }

    return BalancedSplitResult(
        assignments=assignments,
        validation_group_count=len(best_validation_groups),
        test_group_count=group_count - len(best_validation_groups),
        validation_item_count=validation_items,
        test_item_count=test_items,
        item_imbalance=item_imbalance,
        max_label_proportion_difference=label_difference,
        objective=objective,
    )


def _prepare_profiles(
    profiles: Mapping[str, Mapping[str, int]],
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    dict[str, tuple[int, ...]],
    tuple[int, ...],
]:
    """Validate profiles and convert them to aligned count vectors."""
    if len(profiles) < 2:
        raise BalancedSplitError(
            "Balanced splitting requires at least two groups."
        )

    normalized_profiles: dict[str, dict[str, int]] = {}
    all_labels: set[str] = set()

    for group_name, label_counts in profiles.items():
        if not group_name or not group_name.strip():
            raise BalancedSplitError("Group names must be non-empty.")
        if not label_counts:
            raise BalancedSplitError(
                f"Group {group_name!r} has no observed labels."
            )

        normalized_counts: dict[str, int] = {}
        for label, count in label_counts.items():
            if not label or not label.strip():
                raise BalancedSplitError("Label names must be non-empty.")
            if (
                isinstance(count, bool)
                or not isinstance(count, Integral)
                or count <= 0
            ):
                raise BalancedSplitError(
                    "Every observed label count must be a positive integer; "
                    f"received {count!r} for {group_name!r}/{label!r}."
                )

            normalized_counts[label] = int(count)
            all_labels.add(label)

        normalized_profiles[group_name] = normalized_counts

    group_names = tuple(sorted(normalized_profiles))
    label_names = tuple(sorted(all_labels))
    vectors_by_group = {
        group_name: tuple(
            normalized_profiles[group_name].get(label, 0)
            for label in label_names
        )
        for group_name in group_names
    }
    total_label_counts = tuple(
        sum(vectors_by_group[group_name][index] for group_name in group_names)
        for index in range(len(label_names))
    )

    return (
        group_names,
        label_names,
        vectors_by_group,
        total_label_counts,
    )


def _partial_balance_score(
    validation_counts: list[int],
    processed_counts: list[int],
    total_label_counts: tuple[int, ...],
    target_fraction: float,
) -> float:
    """Score balance against the target at an intermediate greedy step."""
    processed_items = sum(processed_counts)
    validation_items = sum(validation_counts)
    total_items = sum(total_label_counts)

    item_deviation = abs(
        validation_items - target_fraction * processed_items
    ) / total_items
    label_deviation = max(
        abs(
            validation_count
            - target_fraction * processed_count
        )
        / total_count
        for validation_count, processed_count, total_count in zip(
            validation_counts,
            processed_counts,
            total_label_counts,
            strict=True,
        )
    )

    return item_deviation + label_deviation


def _score_counts(
    validation_counts: list[int],
    total_label_counts: tuple[int, ...],
) -> tuple[float, float, float]:
    """Calculate the final auditable split objective."""
    total_items = sum(total_label_counts)
    validation_items = sum(validation_counts)
    test_items = total_items - validation_items

    if validation_items == 0 or test_items == 0:
        return float("inf"), float("inf"), float("inf")

    item_imbalance = abs(validation_items - test_items) / total_items
    max_label_difference = max(
        abs(
            validation_count / validation_items
            - (total_count - validation_count) / test_items
        )
        for validation_count, total_count in zip(
            validation_counts,
            total_label_counts,
            strict=True,
        )
    )
    objective = item_imbalance + max_label_difference

    return objective, item_imbalance, max_label_difference


def _improve_with_group_swaps(
    validation_groups: set[str],
    validation_counts: list[int],
    *,
    group_names: tuple[str, ...],
    vectors_by_group: dict[str, tuple[int, ...]],
    total_label_counts: tuple[int, ...],
    max_passes: int,
) -> tuple[set[str], list[int]]:
    """Apply deterministic best-improvement swaps without changing group count."""
    current_groups = set(validation_groups)
    current_counts = list(validation_counts)

    for _ in range(max_passes):
        current_score = _score_counts(
            current_counts,
            total_label_counts,
        )
        best_score = current_score
        best_swap: tuple[str, str] | None = None
        best_counts: list[int] | None = None

        test_groups = tuple(
            group_name
            for group_name in group_names
            if group_name not in current_groups
        )

        for validation_group in sorted(current_groups):
            validation_vector = vectors_by_group[validation_group]

            for test_group in test_groups:
                test_vector = vectors_by_group[test_group]
                candidate_counts = [
                    count - removed + added
                    for count, removed, added in zip(
                        current_counts,
                        validation_vector,
                        test_vector,
                        strict=True,
                    )
                ]
                candidate_score = _score_counts(
                    candidate_counts,
                    total_label_counts,
                )

                if candidate_score < best_score:
                    best_score = candidate_score
                    best_swap = (validation_group, test_group)
                    best_counts = candidate_counts

        if best_swap is None or best_counts is None:
            break

        validation_group, test_group = best_swap
        current_groups.remove(validation_group)
        current_groups.add(test_group)
        current_counts = best_counts

    return current_groups, current_counts