# Decision 013: Cached Validation Evaluation and Latency Protocol

Date: 2026-08-22
Status: accepted

## Question

How should all three architectures be evaluated during training without
re-encoding frequently reused utterances, changing trial lists, leaking Test
data, or reporting an ambiguous inference-time number?

## Decision

Use one shared evaluation boundary with the following invariants:

- A Validation dataset contains exactly the unique utterances referenced by
  its immutable verification trial list. Test partitions are rejected by the
  training callback.
- Every Validation utterance is loaded and encoded once per epoch. The model
  receives deterministic, evenly spaced crops in chronological order.
- Crop embeddings are averaged per utterance and then L2-normalized, producing
  exactly one cached embedding per utterance.
- An utterance shorter than the fixed crop contributes one repeated crop;
  longer utterances contribute the requested evenly spaced crops. Evidence
  therefore records the observed crop count instead of assuming every
  utterance is long enough to produce the requested maximum.
- Trials use cosine similarity between cached utterance vectors. No trial may
  invoke the audio loader or model independently.
- The expected trial-list SHA-256 must match before GPU work and again before
  metric computation. Missing or extra dataset utterances fail closed.
- The embedding table receives a SHA-256 over its ordered utterance IDs,
  dimensions, dtype, and raw normalized vectors.
- Each epoch reports EER, minDCF, FAR, FRR, TAR, accuracy, and TAR at FAR 5%,
  1%, 0.1%, and 0.01% through the existing shared metric implementation.
- Validation's interpolated EER threshold is used only for diagnostic
  FAR/FRR/TAR/accuracy. It is explicitly not claimed as the final security
  threshold.
- Final Test evaluation must supply a security threshold selected and frozen
  using Validation only. Test never selects or changes a threshold.
- Extraction runs in evaluation and inference mode with FP16 autocast on one
  CUDA device, restores the adapter's prior train/eval state, and reports both
  wall-clock utterance throughput and CUDA-event model throughput.
- Model latency uses one preloaded canonical crop, batch size one, 10 warm-up
  iterations, 50 measured iterations, synchronized CUDA events, and reports
  mean, median, and p95 milliseconds. This metric excludes disk I/O,
  preprocessing, trial scoring, and application/network latency.
- Per-epoch JSON evidence is strict (`allow_nan=False`) and is atomically
  promoted from a sibling partial file.

## Evidence Schema

Each Validation artifact contains:

1. exact trial-list, embedding-table, and contextual fingerprints;
2. trial, genuine, impostor, utterance, crop, and batch counts;
3. crop/embedding/scoring and threshold policies;
4. the complete shared metric set and minDCF assumptions;
5. extraction wall/model timing and derived throughput; and
6. canonical split, epoch, segment duration, and segment count.

The local regression suite contains 198 passing tests. It validates numerical
aggregation, cache integrity, cosine scoring, trial fingerprints, complete
metrics, threshold provenance, strict serialization, Validation-only source
boundaries, deterministic extraction settings, and the CUDA-event latency
protocol. A real T4 execution is still required before this runtime is marked
fully accepted. The bounded gate and its dependency-free evidence validator are
implemented by `scripts/smoke_test_evaluation_runtime.py` and
`scripts/validate_evaluation_runtime.py` respectively.

## Accepted T4 Evidence

All three adapters passed the complete callback, cache, scoring, metric,
strict-JSON, and batch-one latency path on a Tesla T4 with PyTorch
`2.10.0+cu128` and CUDA `12.8`. The shared fixture contained four speakers,
eight utterances, four genuine trials, twelve impostor trials, and thirteen
observed crops. Thirteen is correct: three short utterances contributed one
repeated crop, while five longer utterances contributed two endpoint crops.

| Model | EER | minDCF | Median latency | p95 latency | Peak CUDA allocation |
|---|---:|---:|---:|---:|---:|
| ECAPA-TDNN | 0.000000 | 0.000000 | 12.023 ms | 12.912 ms | 450,069,504 bytes |
| RawNet3 | 0.000000 | 0.000000 | 8.840 ms | 12.884 ms | 770,340,352 bytes |
| WavLM+MHFA | 0.000000 | 0.000000 | 35.260 ms | 36.692 ms | 821,150,720 bytes |

The zero error values prove only that the tiny bounded fixture is internally
separable. They are not estimates of dataset-level performance and must not be
used to rank the architectures. The latency numbers are model-only measurements
under the declared batch-one protocol and may be compared only under that
scope.

Accepted artifact SHA-256 values:

- `ecapa_tdnn_t4.json`:
  `d53bf18519bc59939d22cccf7ab1bda20ede0822c2110c952c2b524dc981827b`;
- `rawnet3_t4.json`:
  `86b09b00fc9d73e7b9dffe3e34c86b76adbe8b6ec003dce635d0f70629d8651c`;
- `wavlm_mhfa_t4.json`:
  `16f5ba7e2f6c6b23a072bf6d99383e3089664edc2a26e7c71e6376e34061bed1`;
- downloaded archive:
  `adb3a88e005668f8d9cfcef2fca08607a372a0509bf926d6ee3b007c96c1bdd8`.

## Advantages

- Runtime scales with unique trial utterances rather than number of trials.
- All six model/dataset experiments use identical scoring and metric code.
- Exact protocol and embedding fingerprints make later tables auditable.
- Test isolation and threshold provenance prevent optimistic leakage.
- Separate extraction and model-only timing make performance claims precise.

## Disadvantages

- Holding all normalized utterance embeddings in RAM increases host-memory
  use, although even 30,000 vectors of width 256 require only about 29 MiB.
- Validation still requires a full model pass after every epoch and can be
  expensive for WavLM+MHFA.
- Model-only CUDA latency is reproducible but does not represent end-to-end
  virtual-assistant response time; that must be measured separately.
- Segment duration, crop count, and Validation frequency remain experiment
  variables that must be selected before final training.
