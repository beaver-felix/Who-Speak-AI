"""Generate a reproducible TidyVoice Dev validation/test protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from speaker_recognition.data.splitting import (
    make_balanced_validation_test_split,
)
from speaker_recognition.data.tidyvoice import (
    collect_tidyvoice_speaker_language_counts,
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line settings for protocol generation."""
    parser = argparse.ArgumentParser(
        description=(
            "Create a deterministic speaker-disjoint and language-balanced "
            "validation/test split from TidyVoice Dev."
        )
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        type=Path,
        help="Path to the TidyVoiceX_ASV directory.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination JSON protocol file.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.5,
    )
    parser.add_argument("--restarts", type=int, default=64)
    parser.add_argument("--max-swap-passes", type=int, default=8)
    return parser.parse_args()


def calculate_profiles_sha256(
    profiles: dict[str, dict[str, int]],
) -> str:
    """Hash canonical profile metadata to identify the audited input."""
    canonical_json = json.dumps(
        profiles,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def write_json_atomically(
    payload: dict[str, Any],
    output_path: Path,
) -> None:
    """Write JSON through a temporary file to avoid partial artifacts."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)


def main() -> None:
    """Generate the protocol artifact and print concise diagnostics."""
    arguments = parse_arguments()

    profiles = collect_tidyvoice_speaker_language_counts(
        arguments.dataset_root,
        source_split="dev",
    )
    result = make_balanced_validation_test_split(
        profiles,
        seed=arguments.seed,
        validation_fraction=arguments.validation_fraction,
        restarts=arguments.restarts,
        max_swap_passes=arguments.max_swap_passes,
    )

    languages = sorted(
        {
            language
            for counts in profiles.values()
            for language in counts
        }
    )
    utterance_count = sum(
        count
        for counts in profiles.values()
        for count in counts.values()
    )
    profiles_sha256 = calculate_profiles_sha256(profiles)

    payload: dict[str, Any] = {
        "schema_version": 1,
        "dataset": {
            "id": "dullahn/mozzila-tidyvoice",
            "source_split": "dev",
            "speaker_count": len(profiles),
            "utterance_count": utterance_count,
            "language_count": len(languages),
            "profiles_sha256": profiles_sha256,
        },
        "method": {
            "name": "seeded_greedy_multistart_with_pairwise_swaps",
            "objective": (
                "item_imbalance + "
                "maximum_language_proportion_difference"
            ),
            "seed": arguments.seed,
            "validation_fraction": arguments.validation_fraction,
            "restarts": arguments.restarts,
            "max_swap_passes": arguments.max_swap_passes,
        },
        "diagnostics": {
            "validation_speakers": result.validation_group_count,
            "test_speakers": result.test_group_count,
            "validation_utterances": result.validation_item_count,
            "test_utterances": result.test_item_count,
            "item_imbalance": result.item_imbalance,
            "max_language_proportion_difference": (
                result.max_label_proportion_difference
            ),
            "objective": result.objective,
        },
        "assignments": {
            speaker_id: split.value
            for speaker_id, split in result.assignments.items()
        },
    }
    write_json_atomically(payload, arguments.output)

    print("TIDYVOICE DEV PROTOCOL GENERATED")
    print(f"speakers: {len(profiles):,}")
    print(f"utterances: {utterance_count:,}")
    print(
        "validation/test speakers: "
        f"{result.validation_group_count:,}/"
        f"{result.test_group_count:,}"
    )
    print(
        "validation/test utterances: "
        f"{result.validation_item_count:,}/"
        f"{result.test_item_count:,}"
    )
    print(f"item imbalance: {result.item_imbalance:.6%}")
    print(
        "maximum language-proportion difference: "
        f"{result.max_label_proportion_difference:.6%}"
    )
    print(f"objective: {result.objective:.8f}")
    print(f"profiles SHA-256: {profiles_sha256}")
    print(f"output: {arguments.output.resolve()}")


if __name__ == "__main__":
    main()