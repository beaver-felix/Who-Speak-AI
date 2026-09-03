"""Low-memory audio datasets for RawNet3 training and verification.

Manifests are read into small lists of file paths only; waveforms are decoded
on demand in ``__getitem__``. This avoids loading a complete speech dataset
into the local 8GB machine's RAM.

Training manifest format (CSV or JSONL)::

    path,speaker_id
    audio/speaker_a/utt_001.wav,speaker_a

Verification-pair manifest format::

    enrollment_path,test_path,label
    audio/speaker_a/enroll.wav,audio/speaker_a/test.wav,1

Relative paths are resolved relative to the manifest's directory. Absolute
paths are accepted for Kaggle dataset mounts.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as torch_functional
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class AudioRecord:
    """One utterance entry in a speaker-training manifest."""

    path: Path
    speaker_id: str


@dataclass(frozen=True)
class VerificationPair:
    """One enrollment/test trial in a verification manifest."""

    enrollment_path: Path
    test_path: Path
    label: int


def _resolve_path(raw_path: str, manifest_path: Path) -> Path:
    """Resolve a manifest path relative to its manifest file."""
    candidate = Path(raw_path).expanduser()
    return candidate if candidate.is_absolute() else (manifest_path.parent / candidate)


def _read_manifest_rows(manifest_path: str | Path) -> tuple[list[dict[str, Any]], Path]:
    """Read CSV or JSONL rows without decoding any audio files."""
    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {path}")

    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as manifest_file:
            rows = [json.loads(line) for line in manifest_file if line.strip()]
    elif path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as manifest_file:
            parsed = json.load(manifest_file)
        rows = parsed if isinstance(parsed, list) else parsed["data"]
    else:
        with path.open("r", encoding="utf-8", newline="") as manifest_file:
            rows = list(csv.DictReader(manifest_file))

    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows, path


def _first_present(row: dict[str, Any], names: Iterable[str]) -> Any:
    """Return the first non-empty value among candidate column names."""
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return value
    return None


def load_speaker_records(
    manifest_path: str | Path,
    path_column: str = "path",
    speaker_column: str = "speaker_id",
) -> list[AudioRecord]:
    """Load speaker utterance metadata from a CSV, JSON, or JSONL manifest."""
    rows, resolved_manifest = _read_manifest_rows(manifest_path)
    records: list[AudioRecord] = []
    for row_number, row in enumerate(rows, start=2):
        raw_path = _first_present(row, (path_column, "audio_path", "filepath", "file"))
        speaker_id = _first_present(row, (speaker_column, "speaker", "speaker_label"))
        if raw_path is None or speaker_id is None:
            raise ValueError(
                f"Row {row_number} must contain an audio path and speaker ID"
            )
        records.append(
            AudioRecord(
                path=_resolve_path(str(raw_path), resolved_manifest),
                speaker_id=str(speaker_id),
            )
        )
    return records


def _parse_binary_label(value: Any, row_number: int) -> int:
    """Convert common textual or numeric binary labels to ``0`` or ``1``."""
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "target", "genuine", "same", "yes"}:
            return 1
        if normalized in {"0", "false", "impostor", "nontarget", "different", "no"}:
            return 0
    try:
        label = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid verification label on row {row_number}") from exc
    if label not in (0, 1):
        raise ValueError(f"Verification labels must be 0 or 1 on row {row_number}")
    return label


def load_verification_pairs(
    manifest_path: str | Path,
    enrollment_path_column: str = "enrollment_path",
    test_path_column: str = "test_path",
    label_column: str = "label",
) -> list[VerificationPair]:
    """Load enrollment/test audio pairs and binary labels for evaluation."""
    rows, resolved_manifest = _read_manifest_rows(manifest_path)
    pairs: list[VerificationPair] = []
    for row_number, row in enumerate(rows, start=2):
        enrollment = _first_present(
            row, (enrollment_path_column, "enroll_path", "enrollment", "reference_path")
        )
        test = _first_present(row, (test_path_column, "trial_path", "test", "query_path"))
        label = _first_present(row, (label_column, "target", "is_same_speaker"))
        if enrollment is None or test is None or label is None:
            raise ValueError(
                f"Row {row_number} must contain enrollment_path, test_path, and label"
            )
        pairs.append(
            VerificationPair(
                enrollment_path=_resolve_path(str(enrollment), resolved_manifest),
                test_path=_resolve_path(str(test), resolved_manifest),
                label=_parse_binary_label(label, row_number),
            )
        )
    return pairs


def _load_audio(path: Path, target_sample_rate: int) -> torch.Tensor:
    """Decode one audio file into a mono ``float32`` tensor at the target rate.

    ``torchaudio`` is preferred. A ``soundfile`` fallback keeps the dataset
    usable in lightweight environments where the torchaudio backend is absent.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {path}")

    try:
        import torchaudio

        waveform, sample_rate = torchaudio.load(str(path))
        if sample_rate != target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, target_sample_rate
            )
    except (ImportError, RuntimeError):
        try:
            import soundfile as sound_file
        except ImportError as exc:
            raise RuntimeError(
                "Install torchaudio or soundfile to load audio files"
            ) from exc

        samples, sample_rate = sound_file.read(str(path), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(samples.T.copy())
        if sample_rate != target_sample_rate:
            new_length = round(waveform.shape[-1] * target_sample_rate / sample_rate)
            waveform = torch_functional.interpolate(
                waveform.unsqueeze(0), size=new_length, mode="linear", align_corners=False
            ).squeeze(0)

    # Models expect [time]. Averaging channels is deterministic and avoids
    # silently selecting only the first channel of stereo recordings.
    waveform = waveform.to(dtype=torch.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)
    waveform = waveform.reshape(-1)
    return torch.nan_to_num(waveform)


def _fit_length(
    waveform: torch.Tensor,
    target_length: int,
    training: bool,
    peak_normalize: bool,
) -> torch.Tensor:
    """Randomly crop or zero-pad a waveform to exactly ``target_length``."""
    if target_length <= 0:
        raise ValueError("target_length must be positive")
    current_length = waveform.numel()
    if current_length > target_length:
        if training:
            start = int(torch.randint(0, current_length - target_length + 1, (1,)))
        else:
            start = (current_length - target_length) // 2
        waveform = waveform[start : start + target_length]
    elif current_length < target_length:
        waveform = torch_functional.pad(waveform, (0, target_length - current_length))

    if peak_normalize:
        peak = waveform.abs().amax()
        if peak > 1e-8:
            waveform = waveform / peak
    return waveform.contiguous()


class SpeakerAudioDataset(Dataset[dict[str, Any]]):
    """On-demand fixed-length utterance dataset for speaker classification."""

    def __init__(
        self,
        records: Sequence[AudioRecord],
        sample_rate: int = 16_000,
        duration_seconds: float = 3.0,
        training: bool = True,
        peak_normalize: bool = False,
    ) -> None:
        """Initialize the dataset without loading waveform data into memory."""
        if not records:
            raise ValueError("records must not be empty")
        if sample_rate <= 0 or duration_seconds <= 0:
            raise ValueError("sample_rate and duration_seconds must be positive")
        self.records = list(records)
        self.sample_rate = sample_rate
        self.target_length = round(sample_rate * duration_seconds)
        self.training = training
        self.peak_normalize = peak_normalize
        speakers = sorted({record.speaker_id for record in self.records})
        self.speaker_to_index = {speaker: index for index, speaker in enumerate(speakers)}

    def __len__(self) -> int:
        """Return the number of manifest entries."""
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Decode, resample, and crop/pad one training utterance."""
        record = self.records[index]
        decoded_waveform = _load_audio(record.path, self.sample_rate)
        valid_length = min(decoded_waveform.numel(), self.target_length)
        waveform = _fit_length(
            decoded_waveform,
            self.target_length,
            training=self.training,
            peak_normalize=self.peak_normalize,
        )
        attention_mask = torch.arange(self.target_length) < valid_length
        return {
            "waveform": waveform,
            "attention_mask": attention_mask,
            "speaker_index": self.speaker_to_index[record.speaker_id],
            "speaker_id": record.speaker_id,
            "path": str(record.path),
        }


class VerificationPairDataset(Dataset[dict[str, Any]]):
    """On-demand fixed-length enrollment/test pair dataset for SV trials."""

    def __init__(
        self,
        pairs: Sequence[VerificationPair],
        sample_rate: int = 16_000,
        duration_seconds: float = 3.0,
        peak_normalize: bool = False,
    ) -> None:
        """Initialize pair metadata while keeping audio on disk."""
        if not pairs:
            raise ValueError("pairs must not be empty")
        self.pairs = list(pairs)
        self.sample_rate = sample_rate
        self.target_length = round(sample_rate * duration_seconds)
        self.peak_normalize = peak_normalize

    def __len__(self) -> int:
        """Return the number of verification trials."""
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Decode and center-crop/pad both files in one verification trial."""
        pair = self.pairs[index]
        decoded_enrollment = _load_audio(pair.enrollment_path, self.sample_rate)
        decoded_test = _load_audio(pair.test_path, self.sample_rate)
        enrollment_length = min(decoded_enrollment.numel(), self.target_length)
        test_length = min(decoded_test.numel(), self.target_length)
        enrollment = _fit_length(
            decoded_enrollment,
            self.target_length,
            training=False,
            peak_normalize=self.peak_normalize,
        )
        test = _fit_length(
            decoded_test,
            self.target_length,
            training=False,
            peak_normalize=self.peak_normalize,
        )
        return {
            "enrollment_waveform": enrollment,
            "test_waveform": test,
            "enrollment_attention_mask": torch.arange(self.target_length)
            < enrollment_length,
            "test_attention_mask": torch.arange(self.target_length) < test_length,
            "label": torch.tensor(pair.label, dtype=torch.float32),
            "enrollment_path": str(pair.enrollment_path),
            "test_path": str(pair.test_path),
        }


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 1,
    drop_last: bool = False,
) -> DataLoader:
    """Build a memory-conscious DataLoader for local or Kaggle execution.

    ``prefetch_factor`` and persistent workers are applied only when worker
    processes are enabled, which keeps local debugging compatible with
    ``num_workers=0``.
    """
    if batch_size <= 0 or num_workers < 0 or prefetch_factor <= 0:
        raise ValueError("batch_size, num_workers, and prefetch_factor are invalid")

    loader_options: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
    }
    if num_workers > 0:
        loader_options["persistent_workers"] = persistent_workers
        loader_options["prefetch_factor"] = prefetch_factor
    return DataLoader(**loader_options)
