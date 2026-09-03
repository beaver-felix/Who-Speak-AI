# Decision 011: Real Multi-Batch Training Gate

Date: 2026-08-22
Status: accepted

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

## Accepted Evidence

All three gates ran on a Tesla T4 with PyTorch `2.10.0+cu128` and CUDA `12.8`.
The standalone validator accepted the downloaded artifacts on 2026-08-22.

| Model | Batch | Speakers | Losses | Maximum allocated CUDA memory |
|---|---:|---:|---|---:|
| ECAPA-TDNN | 24 | 72 | 17.0302, 16.5165, 16.2269 | 2,115,798,528 bytes |
| RawNet3 | 24 | 72 | 15.9232, 15.8536, 15.7804 | 4,014,210,560 bytes |
| WavLM+MHFA | 6 | 18 | 15.6114, 15.7050, 16.2116 | 2,097,213,952 bytes |

Every loss and pre-clipping gradient norm was finite, every GradScaler scale
remained `1024`, and all required groups changed parameters. WavLM Transformer
layer 11 was inactive only in step 2, consistent with retained layerdrop
`0.05`; it updated in steps 1 and 3, so all Transformer groups updated across
the gate.

The evidence configuration fingerprints, captured before promoting the batch
status, are:

- ECAPA-TDNN: `df84325195e8aaa3f8c4fa55aeefd567fa299a9df70e8784c2a01a90efabbd39`;
- RawNet3: `2b406d42e42759ad7b2ba2a590ba9cd990cf19cc864fc483bc0e6d9d9d568255`;
- WavLM+MHFA: `7bcbd77b64f66a2b73c032e5dd321cbd4b0f4fe91290db926408a89448720534`.

Accepted artifact SHA-256 values:

- `results/model_audit/multibatch_training/ecapa_tdnn_tidyvoice_t4.json`:
  `1899114632aaaf484d6e1d8ecc24c4d6e26d4f9f4b70b0178c938fe4abf8b118`;
- `results/model_audit/multibatch_training/rawnet3_tidyvoice_t4.json`:
  `2298171eafcf1a564fee2935527571c649700953bb6e14c2f67992139a221079`;
- `results/model_audit/multibatch_training/wavlm_mhfa_tidyvoice_t4.json`:
  `8f1ba20bf9baf2b089f27f9ca3f6c524f54c231d6addebf3e277d777813048c2`.

The downloaded archive SHA-256 was
`b34b7f486e8973f1d89568665bd2305362690e6a2e16a25568202757112a150d`.
The archive is not committed because ZIP files are generated transport
artifacts; the validated JSON contents are committed individually.

The accepted epoch-training batch sizes are therefore ECAPA-TDNN `24`,
RawNet3 `24`, and WavLM+MHFA `6`. Learning rates and full recipes remain
screening candidates until validation-metric experiments are complete.

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
