# Decision 014: Initial Training Experiment Matrix

Date: 2026-08-22
Status: accepted methodology; empirical pilots pending

## Question

How should the first real runs cover three architectures and two datasets
quickly, reproducibly, and fairly without presenting untested settings as
optimal?

## Decision

Use one shared runner for the six model-by-dataset experiments and require a
bounded pilot before any full run.

### Shared controls

- Seed: `42`.
- Canonical audio: 16 kHz, arithmetic channel mean, no amplitude
  normalization, and the accepted deterministic resampler.
- Objective: AAM-Softmax with margin `0.2`, scale `30`, and a newly initialized
  classifier containing every canonical Train speaker.
- Precision: FP16 with initial loss scale `1024` and fail-closed scale backoff.
- Gradient clipping: global norm `5.0`.
- Schedule: constant learning rate for the initial controlled comparison.
- Validation: immutable dataset-specific trials after every completed epoch;
  EER selects the best checkpoint and minDCF breaks a tie.
- Reproducibility: deterministic PyTorch algorithms, cuDNN benchmark disabled,
  and `CUBLAS_WORKSPACE_CONFIG=:4096:8` set before importing PyTorch.
- Tracking: local strict JSONL plus a stable W&B run ID. Pilot W&B is offline;
  full W&B is online.
- Test data never enters training, early stopping, checkpoint selection, or
  threshold selection.

### Architecture-specific controls

| Model | Train crop | Validation crop | Batch | Optimizer policy |
|---|---:|---:|---:|---|
| ECAPA-TDNN | 48,000 | 48,000 | 24 | Adam, encoder/head `1e-4` |
| RawNet3 | 48,240 | 64,240 | 24 | Adam, encoder/head `1e-4` |
| WavLM+MHFA | 48,240 | 64,240 | 6 | AdamW, Transformer `2e-5`, MHFA/head `5e-3` |

These values are source-informed candidates that passed memory and real
multi-batch integration gates. They are not claimed as target-optimal.

### Pilot stage

- Deterministically select 512 Train speakers and one utterance from each.
- Train for one epoch and evaluate the complete immutable Validation protocol
  with one deterministic crop per utterance.
- Save every 100 successful optimizer steps.
- Require finite training/Validation evidence, `last.pt`, `best.pt`, resolved
  config, JSONL metrics, and a strict run summary before promotion.

The pilot deliberately retains the full classifier class count even though
only 512 classes are sampled. It tests the true memory shape and optimizer
boundary while bounding runtime. Pilot metrics are integration evidence, not
dataset-level model-quality results.

### Full stage

- Include every canonical Train speaker.
- Select up to four utterances per speaker per epoch with SHA-256 ranking based
  on seed, epoch, speaker, and utterance ID. High-resource-speaker exposure
  therefore rotates reproducibly across epochs.
- Train for at most 15 epochs with patience 3 and minimum EER improvement
  `0.001`.
- Evaluate with two evenly spaced crops and checkpoint every 500 steps.
- Run only after the corresponding pilot passes.

### ViMD storage policy

ViMD records are ordered by deterministic shuffled Parquet shard groups and
shuffled rows within each shard. Groups remain contiguous before batching, so
each DataLoader worker can reuse its one-row-group cache instead of repeatedly
reloading large shards. TidyVoice retains the ordinary deterministic epoch
shuffle because its audio is stored as separate WAV files.

## Evidence and implementation

- `scripts/prepare_experiment_configs.py` creates six authenticated configs for
  one declared stage.
- `scripts/train_experiment.py` authenticates the resolved config, reconstructs
  and fingerprints the complete canonical manifest, regenerates and verifies
  the immutable Validation trial list, and assembles the shared runtime.
- `scripts/validate_training_run.py` authenticates downloaded summaries,
  per-epoch Validation artifacts, checkpoint sidecar hashes, and local logs
  without deserializing a checkpoint.
- Fresh and resumed runs fail closed on ambiguous state. Checkpoints bind model,
  dataset, config SHA-256, manifest SHA-256, seed, and optimizer groups.
- Each final run summary records deterministic per-epoch membership hashes.
- The local regression suite contains 219 passing tests. Real pilot evidence is
  still required before any full run begins.

The first ECAPA-TDNN/TidyVoice pilot attempt failed closed before its first
optimizer step because native CUDA reflection-padding backward is unavailable
under strict deterministic algorithms. Decision 007 records the
forward-equivalent deterministic padding correction. This rejected attempt
contains no accepted training or evaluation result.

## Advantages

- One runner prevents six experiments from drifting into different methods.
- The 512-speaker pilot exposes integration defects at bounded cost.
- Speaker-capped full epochs reduce TidyVoice imbalance and runtime while
  rotating data exposure exactly across epochs.
- Authenticated configs, manifest/trial hashes, and epoch membership hashes
  make results explainable and reproducible.
- Shard-aware ViMD order prevents avoidable Parquet I/O amplification.

## Disadvantages

- A pilot does not estimate final quality or convergence.
- A constant schedule and initial learning rates may be suboptimal; changing
  them requires a documented Validation-only experiment.
- The full speaker cap does not consume every TidyVoice utterance in one epoch.
- Complete per-epoch Validation is computationally expensive, especially for
  WavLM+MHFA, but is retained for comparable early stopping evidence.
