"""Build and fingerprint canonical verification trial protocols.

The script reads metadata only. It does not decode waveforms or generate
embeddings, so the resulting trial definitions are model-independent and can
be reused by all six experiments.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from speaker_recognition.data.manifest import ManifestRecord, Split, validate_manifest
from speaker_recognition.data.tidyvoice import iter_tidyvoice_dev_records
from speaker_recognition.data.vimd import iter_vimd_source_records
from speaker_recognition.evaluation.trials import (
    VerificationTrial,
    build_verification_trials,
    trial_list_sha256,
)


def parse_arguments() -> argparse.Namespace:
    """Parse dataset paths and fixed trial-construction settings."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic Validation/Test verification-protocol "
            "summaries for TidyVoice and ViMD."
        )
    )
    parser.add_argument(
        "--tidyvoice-root",
        required=True,
        type=Path,
        help="TidyVoiceX_ASV root containing TidyVoiceX_Dev/.",
    )
    parser.add_argument(
        "--tidyvoice-protocol",
        required=True,
        type=Path,
        help="Committed TidyVoice Dev speaker-assignment JSON.",
    )
    parser.add_argument(
        "--vimd-root",
        required=True,
        type=Path,
        help="ViMD dataset root containing data/.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination JSON summary.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic sampling seed (default: 42).",
    )
    parser.add_argument(
        "--max-genuine-per-speaker",
        type=int,
        default=20,
        help="Maximum genuine pairs retained per speaker (default: 20).",
    )
    parser.add_argument(
        "--impostor-trials",
        type=int,
        default=100_000,
        help="Unique impostor trials per dataset and split (default: 100000).",
    )
    parser.add_argument(
        "--vimd-batch-size",
        type=int,
        default=4096,
        help="ViMD metadata rows per Parquet batch (default: 4096).",
    )
    return parser.parse_args()


def load_tidyvoice_assignments(protocol_path: Path) -> dict[str, Split]:
    """Load canonical TidyVoice Dev speaker assignments from JSON."""
    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    raw_assignments = payload.get("assignments")
    if not isinstance(raw_assignments, dict) or not raw_assignments:
        raise ValueError(
            "TidyVoice protocol must contain a non-empty assignments object."
        )

    try:
        return {
            str(speaker_id): Split(split_name)
            for speaker_id, split_name in raw_assignments.items()
        }
    except (TypeError, ValueError) as error:
        raise ValueError(
            "TidyVoice assignments must map speaker IDs to canonical split names."
        ) from error


def summarize_trials(
    records: tuple[ManifestRecord, ...],
    trials: tuple[VerificationTrial, ...],
    *,
    split: Split,
) -> dict[str, Any]:
    """Summarize one deterministic split-level verification protocol."""
    split_records = tuple(record for record in records if record.split is split)
    label_counts = Counter(trial.label for trial in trials)
    trial_utterance_ids = {
        utterance_id
        for trial in trials
        for utterance_id in (
            trial.left_utterance_id,
            trial.right_utterance_id,
        )
    }
    impostor_count = label_counts[0]

    return {
        "split": split.value,
        "manifest_utterances": len(split_records),
        "manifest_speakers": len(
            {record.speaker_id for record in split_records}
        ),
        "trial_utterances": len(trial_utterance_ids),
        "genuine_trials": label_counts[1],
        "impostor_trials": impostor_count,
        "total_trials": len(trials),
        "impostor_far_resolution_fraction": 1.0 / impostor_count,
        "impostor_far_resolution_percent": 100.0 / impostor_count,
        "trial_list_sha256": trial_list_sha256(trials),
    }


def build_dataset_summary(
    records: tuple[ManifestRecord, ...],
    *,
    seed: int,
    max_genuine_per_speaker: int,
    impostor_trial_count: int,
) -> dict[str, dict[str, Any]]:
    """Build Validation and Test protocols for one validated dataset."""
    summary: dict[str, dict[str, Any]] = {}
    for split in (Split.VALIDATION, Split.TEST):
        trials = build_verification_trials(
            records,
            split=split,
            seed=seed,
            max_genuine_per_speaker=max_genuine_per_speaker,
            impostor_trial_count=impostor_trial_count,
        )
        summary[split.value] = summarize_trials(
            records,
            trials,
            split=split,
        )
    return summary


def write_json(payload: dict[str, Any], output_path: Path) -> None:
    """Write a stable, human-readable UTF-8 evidence artifact."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Validate manifests, construct trials, and write protocol evidence."""
    arguments = parse_arguments()

    tidyvoice_assignments = load_tidyvoice_assignments(
        arguments.tidyvoice_protocol
    )
    tidyvoice_records = validate_manifest(
        tuple(
            iter_tidyvoice_dev_records(
                arguments.tidyvoice_root,
                dev_assignments=tidyvoice_assignments,
            )
        )
    )
    vimd_records = validate_manifest(
        tuple(
            record
            for source_split in ("valid", "test")
            for record in iter_vimd_source_records(
                arguments.vimd_root,
                source_split=source_split,
                batch_size=arguments.vimd_batch_size,
            )
        )
    )

    settings = {
        "seed": arguments.seed,
        "max_genuine_per_speaker": arguments.max_genuine_per_speaker,
        "impostor_trials_per_dataset_split": arguments.impostor_trials,
        "pair_order": "canonical_utterance_id_order",
        "genuine_sampling": "uniform_unique_pairs_capped_per_speaker",
        "impostor_sampling": (
            "uniform_speaker_pair_then_uniform_utterance_pair"
        ),
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "settings": settings,
        "protocols": {
            "tidyvoice": build_dataset_summary(
                tidyvoice_records,
                seed=arguments.seed,
                max_genuine_per_speaker=arguments.max_genuine_per_speaker,
                impostor_trial_count=arguments.impostor_trials,
            ),
            "vimd": build_dataset_summary(
                vimd_records,
                seed=arguments.seed,
                max_genuine_per_speaker=arguments.max_genuine_per_speaker,
                impostor_trial_count=arguments.impostor_trials,
            ),
        },
        "methodology_notes": [
            "Validation trials select thresholds; Test trials report final metrics.",
            "All three models use identical model-independent trial lists.",
            "Genuine-pair caps limit domination by high-resource speakers.",
            "Uniform speaker-pair sampling limits utterance-count bias.",
            "One hundred thousand impostors give 0.001 percentage-point FAR resolution.",
            "Trial lists are regenerated deterministically and verified by SHA-256.",
        ],
    }
    write_json(payload, arguments.output)

    print("VERIFICATION TRIAL PROTOCOLS GENERATED")
    print(
        "settings: "
        f"seed={arguments.seed}, "
        f"genuine cap={arguments.max_genuine_per_speaker}, "
        f"impostors={arguments.impostor_trials:,} per dataset/split"
    )
    for dataset_name, dataset_protocols in payload["protocols"].items():
        for split_name, statistics in dataset_protocols.items():
            print(
                f"{dataset_name}/{split_name}: "
                f"{statistics['genuine_trials']:,} genuine, "
                f"{statistics['impostor_trials']:,} impostor, "
                f"SHA-256 {statistics['trial_list_sha256']}"
            )
    print(f"output: {arguments.output.resolve()}")


if __name__ == "__main__":
    main()
