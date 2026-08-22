# Decision 011: Real Multi-Batch Training Gate

Date: 2026-08-22
Status: protocol implemented; Kaggle evidence pending

## Purpose

Memory calibration used deterministic shifts of one real utterance. It proved
capacity but did not exercise real multi-record loading or successive optimizer
updates. Before building the epoch trainer, every model must pass a bounded gate
through the actual canonical TidyVoice loader.

## Protocol

- Dataset: canonical TidyVoice source Train.
- Data selection: the first utterance from each earliest lexical speaker.
- Speaker policy: one utterance per speaker and no speaker reuse within or
  across batches.
- Steps: three consecutive optimizer steps.
- Candidate batches: ECAPA-TDNN `24`, RawNet3 `24`, WavLM+MHFA `6`.
- Crops: ECAPA `48,000` samples; RawNet3 and WavLM+MHFA `48,240` samples,
  retaining each accepted adapter calibration shape.
- Class head: all `3,666` canonical TidyVoice training speakers, even though
  the gate touches only a deterministic subset.
- Mixed precision: FP16 with initial dynamic scale `1024`.
- Shared objective and architecture-specific optimizer groups: Decision 010.
- DataLoader workers: zero for this gate, isolating model/data correctness from
  multiprocessing performance.

Each step must prove:

1. finite canonical audio, embeddings, loss, and pre-clipping gradient norm;
2. the configured number of distinct speakers and utterances;
3. one finite non-zero gradient probe in every active optimizer group;
4. a parameter change in every probed optimizer group after the step;
5. no dynamic-loss-scale backoff; and
6. strict finite JSON evidence with step timing and peak CUDA memory.

A standalone dependency-free validator rechecks model identity, batch size,
speaker counts, runtime, finite scalars, exact optimizer groups, CUDA peaks,
and all aggregate checks before any downloaded artifact is accepted.

For WavLM+MHFA, MHFA and the new AAM-Softmax head must update on every step.
The official WavLM encoder retains layerdrop `0.05`, so one Transformer layer
may intentionally be inactive in an individual step. Every one of the 12
Transformer groups must update at least once across the complete three-step
gate. Frozen WavLM front-end components remain outside the optimizer by
Decision 010.

## Interpretation

Passing proves that the canonical loader, deterministic cropper, NumPy-to-torch
boundary, adapter, AAM objective, gradient clipping, mixed precision, and every
optimizer group work together for consecutive real batches. It is not evidence
of convergence, generalization, EER, minDCF, or optimal hyperparameters.

## Advantages

- Detects integration failures hidden by a repeated-record memory gate.
- Proves encoder/backend updates rather than only classifier updates.
- Uses bounded compute while preserving auditable real-data identities.
- Fails closed on silent GradScaler skips and non-finite JSON values.

## Disadvantages

- Three batches cannot establish a useful learning curve.
- Lexical speaker selection is deterministic but not demographically
  representative.
- Single-process loading does not measure production DataLoader throughput.
- Crop lengths remain slightly architecture-specific and require explicit
  treatment in the later comparison methodology.
