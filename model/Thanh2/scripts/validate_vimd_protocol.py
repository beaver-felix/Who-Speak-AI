"""Validate and fingerprint the canonical ViMD experiment protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from speaker_recognition.data.manifest import Split, validate_manifest
from speaker_recognition.data.vimd import (
    VIMD_EXCLUDED_VALIDATION_SPEAKERS,
    iter_vimd_source_records,
)


_EXPECTED_SPLIT_STATISTICS = {
    "train": {
        "utterances": 15023,
        "speakers": 10291,
        "genuine_pairs": 7044,
    },
    "validation": {
        "utterances": 1898,
        "speakers": 1318,
        "genuine_pairs": 879,
    },
    "test": {
        "utterances": 2026,
        "speakers": 1344,
        "genuine_pairs": 1046,
    },
}


def parse_arguments() -> argparse.Namespace:
    """Parse ViMD validation settings."""
    parser = argparse.ArgumentParser(
        description=(
            "Stream ViMD metadata, apply the canonical exclusion policy, "
            "validate leakage constraints, and write an evidence artifact."
        )
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        type=Path,
        help="Path to the ViMD dataset root containing data/.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination JSON summary.",
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser.parse_args()


def calculate_manifest_sha256(records: tuple[Any, ...]) -> str:
    """Hash model-relevant identities and physical audio references."""
    digest = hashlib.sha256()

    for record in records:
        canonical_fields = (
            record.dataset,
            record.source_split,
            record.split.value,
            record.utterance_id,
            record.speaker_id,
            record.recording_id,
            record.audio_storage.value,
            record.audio_path,
            record.audio_row_index,
        )
        encoded = json.dumps(
            canonical_fields,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(encoded)
        digest.update(b"\n")

    return digest.hexdigest()


def summarize_protocol(
    records: tuple[Any, ...],
) -> dict[str, dict[str, int]]:
    """Calculate utterance, speaker, and genuine-pair statistics."""
    speaker_counts = {
        split: Counter(
            record.speaker_id
            for record in records
            if record.split is split
        )
        for split in Split
    }

    return {
        split.value: {
            "utterances": sum(speaker_counts[split].values()),
            "speakers": len(speaker_counts[split]),
            "genuine_pairs": sum(
                count * (count - 1) // 2
                for count in speaker_counts[split].values()
            ),
        }
        for split in Split
    }


def write_json(
    payload: dict[str, Any],
    output_path: Path,
) -> None:
    """Write a stable UTF-8 JSON artifact."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Build, validate, fingerprint, and report the ViMD protocol."""
    arguments = parse_arguments()

    records = tuple(
        record
        for source_split in ("train", "valid", "test")
        for record in iter_vimd_source_records(
            arguments.dataset_root,
            source_split=source_split,
            batch_size=arguments.batch_size,
        )
    )
    validated_records = validate_manifest(records)
    statistics = summarize_protocol(validated_records)

    if statistics != _EXPECTED_SPLIT_STATISTICS:
        raise RuntimeError(
            "ViMD inventory differs from the audited protocol. "
            f"Expected {_EXPECTED_SPLIT_STATISTICS}, received {statistics}."
        )

    manifest_sha256 = calculate_manifest_sha256(validated_records)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "dataset": {
            "id": "dullahn/vimd-dataset",
            "canonical_utterances": len(validated_records),
            "manifest_sha256": manifest_sha256,
        },
        "policy": {
            "source_train": "canonical_train",
            "source_valid": "canonical_validation",
            "source_test": "canonical_test",
            "excluded_validation_speakers": sorted(
                VIMD_EXCLUDED_VALIDATION_SPEAKERS
            ),
            "excluded_validation_utterances": 2,
            "reason": (
                "Preserve source Test and remove Validation speaker overlap."
            ),
        },
        "split_statistics": statistics,
        "methodology_notes": [
            "Parquet audio bytes were not read during metadata validation.",
            "The official source Test partition remains unchanged.",
            "The two excluded Validation rows contribute no genuine pairs.",
            "Gender is retained as metadata but is not a trusted target.",
        ],
    }
    write_json(payload, arguments.output)

    print("VIMD CANONICAL PROTOCOL VALIDATED")
    for split_name, split_statistics in statistics.items():
        print(
            f"{split_name}: "
            f"{split_statistics['utterances']:,} utterances, "
            f"{split_statistics['speakers']:,} speakers, "
            f"{split_statistics['genuine_pairs']:,} genuine pairs"
        )
    print(f"total utterances: {len(validated_records):,}")
    print(f"manifest SHA-256: {manifest_sha256}")
    print(f"output: {arguments.output.resolve()}")


if __name__ == "__main__":
    main()