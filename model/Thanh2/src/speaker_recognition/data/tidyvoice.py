"""TidyVoice path parsing for the canonical speaker manifest."""

from __future__ import annotations
from collections.abc import Iterator

from pathlib import Path

from speaker_recognition.data.manifest import (
    AudioStorage,
    ManifestRecord,
    Split,
)


class TidyVoicePathError(ValueError):
    """Raised when a TidyVoice path violates the documented layout."""


_BRANCH_TO_SOURCE_SPLIT = {
    ("TidyVoiceX_Train", "TidyVoiceX_Train"): "train",
    ("TidyVoiceX_Dev", "TidyVoiceX_Dev"): "dev",
}

_SOURCE_SPLIT_TO_BRANCH = {
    source_split: branch
    for branch, source_split in _BRANCH_TO_SOURCE_SPLIT.items()
}

_ALLOWED_CANONICAL_SPLITS = {
    "train": frozenset({Split.TRAIN}),
    "dev": frozenset({Split.VALIDATION, Split.TEST}),
}

def iter_tidyvoice_audio_paths(
    dataset_root: str | Path,
    *,
    source_split: str,
) -> Iterator[Path]:
    """Yield TidyVoice WAV paths in deterministic lexical order.

    The function validates the documented speaker/language/file hierarchy but
    does not decode audio. This keeps discovery inexpensive for more than
    300,000 files.

    Parameters
    ----------
    dataset_root:
        Root containing the TidyVoice train and dev branches.
    source_split:
        Source partition name: ``train`` or ``dev``.

    Yields
    ------
    pathlib.Path
        Absolute WAV paths ordered by speaker, language, and filename.

    Raises
    ------
    TidyVoicePathError
        If the source split, branch, or nested file layout is invalid.
    """
    try:
        branch = _SOURCE_SPLIT_TO_BRANCH[source_split]
    except KeyError as error:
        raise TidyVoicePathError(
            f"Unsupported TidyVoice source split: {source_split!r}"
        ) from error

    branch_root = (
        Path(dataset_root)
        .expanduser()
        .resolve()
        .joinpath(*branch)
    )
    if not branch_root.is_dir():
        raise TidyVoicePathError(
            f"TidyVoice branch directory does not exist: {branch_root}"
        )

    for speaker_path in sorted(
        branch_root.iterdir(),
        key=lambda path: path.name,
    ):
        if not speaker_path.is_dir():
            raise TidyVoicePathError(
                f"Unexpected entry in speaker directory: {speaker_path}"
            )

        for language_path in sorted(
            speaker_path.iterdir(),
            key=lambda path: path.name,
        ):
            if not language_path.is_dir():
                raise TidyVoicePathError(
                    f"Unexpected entry in language directory: {language_path}"
                )

            for audio_path in sorted(
                language_path.iterdir(),
                key=lambda path: path.name,
            ):
                if not audio_path.is_file() or audio_path.suffix.lower() != ".wav":
                    raise TidyVoicePathError(
                        f"Unexpected file in TidyVoice audio directory: "
                        f"{audio_path}"
                    )

                yield audio_path

def parse_tidyvoice_audio_path(
    audio_path: str | Path,
    *,
    dataset_root: str | Path,
    split: Split,
) -> ManifestRecord:
    """Convert one TidyVoice WAV path into a canonical manifest record.

    TidyVoice uses the following dataset-relative layout::

        <source-branch>/<source-branch>/<speaker>/<language>/<utterance>.wav

    The parser validates this structure instead of inferring labels from an
    unexpected path. This prevents silent speaker-label and split corruption.

    Parameters
    ----------
    audio_path:
        Path to one TidyVoice WAV file.
    dataset_root:
        Root directory containing ``TidyVoiceX_Train`` and
        ``TidyVoiceX_Dev``.
    split:
        Canonical experiment split assigned to the utterance.

    Returns
    -------
    ManifestRecord
        Canonical metadata without loading the waveform.

    Raises
    ------
    TidyVoicePathError
        If the path is outside the dataset, malformed, not WAV, or assigned to
        an invalid canonical split.
    """
    root = Path(dataset_root).expanduser().resolve()
    path = Path(audio_path).expanduser().resolve()

    try:
        relative_path = path.relative_to(root)
    except ValueError as error:
        raise TidyVoicePathError(
            f"Audio path is outside dataset root: {audio_path}"
        ) from error

    parts = relative_path.parts
    if len(parts) != 5:
        raise TidyVoicePathError(
            "Expected five path components below the TidyVoice root, "
            f"but received {len(parts)}: {relative_path.as_posix()}"
        )

    branch = (parts[0], parts[1])
    try:
        source_split = _BRANCH_TO_SOURCE_SPLIT[branch]
    except KeyError as error:
        raise TidyVoicePathError(
            f"Unsupported TidyVoice branch: {'/'.join(branch)}"
        ) from error

    if path.suffix.lower() != ".wav":
        raise TidyVoicePathError(
            f"TidyVoice audio must use the WAV format: {path.name}"
        )

    if split not in _ALLOWED_CANONICAL_SPLITS[source_split]:
        if source_split == "train":
            expected = "train"
        else:
            expected = "validation or test"

        raise TidyVoicePathError(
            f"TidyVoice source split {source_split!r} must map to canonical "
            f"{expected}, not {split.value!r}."
        )

    speaker_name = parts[2]
    language = parts[3]
    utterance_stem = path.stem

    if not speaker_name.startswith("id"):
        raise TidyVoicePathError(
            f"Invalid TidyVoice speaker directory: {speaker_name!r}"
        )
    if not language.strip():
        raise TidyVoicePathError("TidyVoice language must not be empty.")
    if not utterance_stem.strip():
        raise TidyVoicePathError("TidyVoice utterance name must not be empty.")

    # Include every identity-bearing path component so IDs remain globally
    # stable even if filenames repeat across speakers, languages, or branches.
    utterance_id = ":".join(
        (
            "tidyvoice",
            source_split,
            speaker_name,
            language,
            utterance_stem,
        )
    )

    return ManifestRecord(
        dataset="tidyvoice",
        source_split=source_split,
        split=split,
        utterance_id=utterance_id,
        speaker_id=f"tidyvoice:{speaker_name}",
        recording_id=utterance_id,
        audio_storage=AudioStorage.FILE,
        # Store a dataset-relative POSIX path so manifests remain portable
        # between local Windows development and Kaggle Linux execution.
        audio_path=relative_path.as_posix(),
        language=language,
    )