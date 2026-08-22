# Decision 006: Layered Configuration and Shared Dataset Interface

Status: Accepted
Date: 2026-08-22

## Context

The benchmark has six dataset/model combinations. Comparison-critical controls
must be identical, while source provenance and later validated training recipes
must remain model-specific. TidyVoice and ViMD also use incompatible physical
storage: standalone WAV files versus embedded Parquet audio bytes.

The Kaggle image already supplies a CUDA-matched PyTorch build. Declaring or
installing a different PyTorch package from the local project could replace
that build and break GPU compatibility.

## Configuration Decision

Use ordered TOML layers:

```text
configs/base.toml
    + configs/datasets/<dataset>.toml
    + configs/models/<model>.toml
    + explicit existing-key overrides
    -> validated resolved JSON + SHA-256
```

The resolver uses Python's standard-library `tomllib`. It recursively merges
tables, rejects table/scalar structure conflicts, rejects unknown override
paths, validates accepted scientific controls, and fingerprints canonical JSON.

The base configuration locks only accepted shared decisions:

- mono 16 kHz input
- no amplitude normalization
- arithmetic channel downmixing
- SciPy polyphase resampling
- Decision 005 verification-trial settings
- Decision 004 FAR targets and minDCF parameters
- non-persistent workers until epoch state is worker-synchronized

Dataset layers record audited source IDs, Kaggle roots, storage formats, split
protocols, and canonical counts. Model layers record pinned revisions,
embedding dimensions, checkpoint evidence, and architecture facts.

Untested crop duration, batch size, optimizer, learning rate, freezing policy,
augmentation, scheduler, and epoch budget are deliberately not activated.
Reference values remain labeled as reference-only or pending Validation and
Kaggle profiling evidence.

## Dataset and Batch Decision

Use one framework-neutral map-style dataset boundary for all models:

```text
canonical ManifestRecord
    -> dataset-specific relative locator
    -> standalone WAV loader OR worker-local Parquet reader
    -> canonical mono float32 16 kHz waveform
    -> deterministic training crop OR evaluation crops
    -> contiguous NumPy batch
    -> torch.from_numpy in the training/evaluation layer
```

The interface implements `__len__` and `__getitem__`, so it can be used by a
PyTorch DataLoader without importing PyTorch in the package's data layer.

Training behavior:

- Select canonical Train records only.
- Sort by stable utterance ID.
- Build an exact contiguous speaker-to-class mapping.
- Derive the crop from global seed, epoch, and utterance ID.
- Stack waveforms as `[batch, samples]` and labels as integer indexes.

Evaluation behavior:

- Select canonical Validation or Test records only.
- Produce deterministic timeline crops for each utterance.
- Flatten crops into one `[total_crops, samples]` array.
- Preserve utterance boundaries with cumulative segment offsets.
- Allow short utterances to contribute one crop while long utterances
  contribute the configured crop count.

File paths are resolved below their configured dataset root. Paths escaping the
root are rejected. Parquet readers are created lazily per dataset and therefore
maintain one row-group cache per DataLoader worker.

## Worker-Lifecycle Constraint

`TrainingSpeakerDataset.set_epoch()` changes deterministic crop seeds. A
persistent DataLoader worker owns a copied dataset object and would not observe
the main process's changed epoch without explicit synchronization. Therefore,
`persistent_workers` is locked to `false` until an epoch-aware shared-state or
sampler implementation is tested. Correct epoch-specific crops take priority
over worker startup performance.

## Verification

- All six base/dataset/model layer combinations resolve successfully.
- Equivalent override mappings produce identical fingerprints.
- Unknown overrides and structure conflicts fail.
- Attempts to change accepted trial controls fail.
- Standalone WAV and embedded Parquet records produce the same training sample
  and batch interface.
- Training crops repeat within an epoch and change across epochs.
- Evaluation batches preserve variable crop boundaries.
- Escaping file paths and invalid speaker mappings fail.
- The complete local regression suite contains 88 passing tests.

## Advantages

- Every run can save its complete configuration and provenance fingerprint.
- Accepted comparison controls cannot drift silently between models.
- No configuration dependency or PyTorch reinstall threatens Kaggle CUDA.
- One data interface prevents six model/dataset-specific loader paths.
- Contiguous NumPy arrays support zero-copy CPU tensor conversion.
- Evaluation crop offsets enable encode-once, aggregate-by-utterance scoring.

## Disadvantages and Limitations

- TOML has no native null, so unresolved choices use explicit status fields.
- New override keys must first be added to a reviewed configuration layer.
- NumPy batches still require conversion in the future PyTorch runtime layer.
- Persistent workers are disabled, adding worker startup cost per epoch.
- Random record order would reduce Parquet cache effectiveness; a shard-aware,
  speaker-balanced training sampler remains to be implemented and measured.
- Crop sizes and evaluation crop counts are still pending Validation evidence.
