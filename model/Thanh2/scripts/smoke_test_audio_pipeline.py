"""Smoke-test canonical audio loading on real TidyVoice and ViMD data."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np

from speaker_recognition.data.audio import (
    CanonicalAudio,
    ParquetAudioReader,
    load_audio_file,
)
from speaker_recognition.data.tidyvoice import iter_tidyvoice_audio_paths
from speaker_recognition.data.vimd import iter_vimd_source_records


def parse_arguments() -> argparse.Namespace:
    """Parse real-dataset smoke-test settings."""
    parser = argparse.ArgumentParser(
        description=(
            "Load one real TidyVoice file and one ViMD Parquet row through "
            "the shared canonical audio pipeline."
        )
    )
    parser.add_argument(
        "--tidyvoice-root",
        required=True,
        type=Path,
        help="Path to TidyVoiceX_ASV.",
    )
    parser.add_argument(
        "--vimd-root",
        required=True,
        type=Path,
        help="Path to the ViMD dataset root containing data/.",
    )
    parser.add_argument(
        "--target-sample-rate",
        type=int,
        default=16000,
    )
    return parser.parse_args()


def waveform_sha256(audio: CanonicalAudio) -> str:
    """Fingerprint canonical samples and their target sample rate."""
    digest = hashlib.sha256()
    digest.update(audio.sample_rate.to_bytes(4, "big", signed=False))
    digest.update(audio.waveform.tobytes(order="C"))
    return digest.hexdigest()


def print_audio_summary(
    label: str,
    audio: CanonicalAudio,
    *,
    elapsed_seconds: float,
) -> None:
    """Print bounded diagnostics for one canonical waveform."""
    print(f"\n=== {label} ===")
    print(f"original sample rate: {audio.original_sample_rate}")
    print(f"original channels: {audio.original_channels}")
    print(f"canonical sample rate: {audio.sample_rate}")
    print(f"canonical samples: {audio.waveform.size}")
    print(f"duration seconds: {audio.duration_seconds:.6f}")
    print(f"finite: {bool(np.isfinite(audio.waveform).all())}")
    print(f"waveform SHA-256: {waveform_sha256(audio)}")
    print(f"load seconds: {elapsed_seconds:.6f}")


def main() -> None:
    """Run real standalone-file and embedded-Parquet smoke tests."""
    arguments = parse_arguments()

    tidyvoice_path = next(
        iter_tidyvoice_audio_paths(
            arguments.tidyvoice_root,
            source_split="train",
        )
    )
    start = perf_counter()
    tidyvoice_audio = load_audio_file(
        tidyvoice_path,
        target_sample_rate=arguments.target_sample_rate,
    )
    tidyvoice_elapsed = perf_counter() - start
    print(f"TidyVoice source: {tidyvoice_path}")
    print_audio_summary(
        "TIDYVOICE CANONICAL AUDIO",
        tidyvoice_audio,
        elapsed_seconds=tidyvoice_elapsed,
    )

    vimd_record = next(
        iter_vimd_source_records(
            arguments.vimd_root,
            source_split="test",
        )
    )
    parquet_reader = ParquetAudioReader(
        arguments.vimd_root,
        target_sample_rate=arguments.target_sample_rate,
    )
    start = perf_counter()
    vimd_audio = parquet_reader.load(vimd_record)
    vimd_elapsed = perf_counter() - start

    # A repeated read must come from the row-group cache and reproduce exactly
    # the same canonical signal.
    repeated_vimd_audio = parquet_reader.load(vimd_record)
    if parquet_reader.row_group_reads != 1:
        raise RuntimeError("ViMD row-group cache was not reused.")
    if not np.array_equal(
        vimd_audio.waveform,
        repeated_vimd_audio.waveform,
    ):
        raise RuntimeError("Repeated ViMD decoding changed the waveform.")

    print(
        "ViMD source: "
        f"{vimd_record.audio_path} row {vimd_record.audio_row_index}"
    )
    print_audio_summary(
        "VIMD CANONICAL AUDIO",
        vimd_audio,
        elapsed_seconds=vimd_elapsed,
    )
    print(f"Parquet row-group reads after two loads: {parquet_reader.row_group_reads}")

    if tidyvoice_audio.sample_rate != arguments.target_sample_rate:
        raise RuntimeError("TidyVoice canonical sample rate is incorrect.")
    if vimd_audio.sample_rate != arguments.target_sample_rate:
        raise RuntimeError("ViMD canonical sample rate is incorrect.")

    print("\nREAL DATASET AUDIO PIPELINE SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
