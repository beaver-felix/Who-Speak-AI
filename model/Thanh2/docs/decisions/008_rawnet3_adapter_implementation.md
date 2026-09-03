# Decision 008: RawNet3 Adapter Implementation

Status: Accepted
Date: 2026-08-22

## Context

RawNet3 operates directly on waveform samples through a parameterized Sinc
filterbank and a deep residual encoder. Unlike ECAPA-TDNN, its public
checkpoint repository contains weights but not the architecture source. The
checkpoint and architecture therefore require separate immutable provenance.

The audited checkpoint has 234 tensor state entries and loads strictly into
the official Clova architecture. The official configuration selects ECA
attentive pooling, a 256-dimensional output, and Sinc stride 10. Its dataset
loader converts a frame count to waveform samples with:

```text
samples = frames * 160 + 240
```

Consequently, its 300-frame training reference is 48,240 samples and its
400-frame evaluation reference is 64,240 samples at 16 kHz.

## Decision

- Pin checkpoint repository `jungjee/RawNet3`.
- Pin checkpoint revision
  `c89102eea20c3f96917c434de673c0ace0caddc0`.
- Require checkpoint `model.pt` SHA-256
  `1ab283bcdf776bfceceea18240e56a8756835b1911b04f9c44f347d47c09f90c`.
- Pin architecture repository `clovaai/voxceleb_trainer` at
  `f51bab870672a9b0b50fa158b4e30f329e7866d7`.
- Vendor the two required MIT-licensed architecture files with the original
  license and an explicit provenance record.
- Preserve ECA pooling, output dimension 256, Sinc stride 10, model scale 8,
  context aggregation, summed residual path, log-Sinc transformation, and
  mean Sinc normalization.
- Require `asteroid-filterbanks==0.4.0`, the version already validated against
  the Kaggle PyTorch runtime.
- Download only `model.pt` from the immutable snapshot.
- Verify its SHA-256 before deserialization.
- Load with `torch.load(..., weights_only=True)` and require the exact
  tensor-only top-level structure `{"model": state_dictionary}`.
- Apply the state dictionary with strict loading and assert 16,280,322 model
  parameters.
- Return the shared `[batch, 256]` contract and apply explicit L2
  normalization for model-independent cosine scoring.
- Require fixed, equal waveform lengths inside each RawNet3 batch. Any supplied
  relative lengths must all equal 1; segmentation owns crop construction.
- Keep the encoder trainable and expose an explicit freeze/unfreeze operation.

The 48,240- and 64,240-sample crops are used only for this architecture
compatibility gate because they are traceable to the official recipe. The
final shared experiment segment duration and crop aggregation count remain
pending controlled Validation and GPU-memory experiments.

## Source Adaptation Boundary

The redistributed implementation changes only package-relative imports, local
names, annotations, docstrings, formatting, explicit input validation, and one
constructor print removal. Layer topology, checkpoint-bearing attribute names,
parameter shapes, and forward mathematical operations remain compatible with
the pinned source. `SOURCE.md` records upstream file hashes, and `LICENSE.md`
retains NAVER's MIT terms.

## Dependency Decision

The optional group installs:

```text
asteroid-filterbanks==0.4.0
huggingface-hub>=1.11,<1.12
```

PyTorch remains supplied by Kaggle and is not declared by this project, which
prevents pip from replacing the working CUDA-matched build.

## Acceptance Gate

Run `scripts/smoke_test_rawnet3_adapter.py` on a Kaggle Tesla T4 with a real
TidyVoice recording and require:

- resolved checkpoint revision equals the full pinned commit
- checkpoint SHA-256 matches before restricted loading
- strict state loading has no missing or unexpected keys
- parameter count equals 16,280,322
- a 64,240-sample real-speech crop produces shape `[1, 256]`
- every embedding value is finite and the L2 norm equals 1 within tolerance
- repeated evaluation cosine similarity is at least 0.99999
- two distinct endpoint 48,240-sample crops produce finite, non-zero input
  gradients without symmetric BatchNorm cancellation
- at least one encoder parameter gradient is present, finite, and non-zero
- structured JSON evidence is saved and reviewed

The local regression suite contains 101 passing tests, including five
dependency-free RawNet3 syntax, architecture-surface, license, provenance, and
checkpoint-identity checks.

## Kaggle GPU Evidence

The acceptance gate passed on `cuda:0` with:

- GPU class: Tesla T4
- PyTorch: `2.10.0+cu128`
- CUDA build: `12.8`
- Asteroid Filterbanks: `0.4.0`
- Checkpoint revision:
  `c89102eea20c3f96917c434de673c0ace0caddc0`
- Architecture revision:
  `f51bab870672a9b0b50fa158b4e30f329e7866d7`
- Checkpoint SHA-256:
  `1ab283bcdf776bfceceea18240e56a8756835b1911b04f9c44f347d47c09f90c`
- Adapter parameters: 16,280,322
- Trainable adapter parameters: 16,280,322
- Real TidyVoice sample: 122,496 samples, 7.656 seconds
- Evaluation crop: 64,240 samples
- Gradient crops: two distinct 48,240-sample endpoints
- Output shape: `[1, 256]`
- Output L2 norm: 1.0
- Repeat cosine similarity: 1.0
- Checkpoint identity, embedding shape, finiteness, and normalization: passed
- Input gradient finite and non-zero: passed
- Encoder gradient present, finite, and non-zero: passed

Canonical waveform SHA-256:
`9d608813bc66beb3d0b0bf72b23e7776abbf7121337c4b3beb737da5a627ac92`

Evidence artifact:
`results/model_audit/rawnet3_adapter_smoke.json`

Artifact SHA-256:
`37e5ff5ec506f3fcc185e82465823f2fa9325246d5c20eb76bdeb0957ee8411d`

This establishes pinned-source loading, adapter shape compatibility,
deterministic inference, and fine-tuning gradient flow. It does not establish
TidyVoice or ViMD verification accuracy and must not be reported as a
benchmark result.

## Advantages

- Immutable architecture and checkpoint identities make the result auditable.
- Pre-deserialization hashing plus restricted loading reduces checkpoint risk.
- Strict state loading detects any architecture or artifact drift.
- The exact official architecture avoids unsupported reimplementations.
- The common normalized output supports the same downstream scorer and
  verification metrics used by ECAPA and WavLM+MHFA.

## Disadvantages and Limitations

- Vendoring creates a maintenance obligation when upstream code changes; the
  pinned hashes and license record make that obligation explicit.
- Raw waveform training can consume substantial activation memory, so final
  crop, batch, precision, and accumulation settings still require measurement.
- Fixed-length batches do not directly use arbitrary relative-length masks.
- The compatibility gate demonstrates correct loading and gradient flow, not
  TidyVoice or ViMD verification accuracy.
- The official reference hyperparameters are hypotheses, not automatically
  selected settings for the cross-model comparison.

## Primary Sources

- Official Clova VoxCeleb trainer:
  <https://github.com/clovaai/voxceleb_trainer>
- Official RawNet3 documentation:
  <https://github.com/Jungjee/RawNet/tree/master/python/RawNet3>
- Pinned RawNet3 checkpoint:
  <https://huggingface.co/jungjee/RawNet3/blob/c89102eea20c3f96917c434de673c0ace0caddc0/model.pt>
