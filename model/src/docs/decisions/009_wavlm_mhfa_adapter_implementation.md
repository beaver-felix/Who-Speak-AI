# Decision 009: WavLM+MHFA Adapter Implementation

Status: Accepted
Date: 2026-08-22

## Context

The pinned `theolepage/wavlm_ssl_sv` repository contains architecture and
training code but no packaged weights. It links two external initialization
options:

1. Microsoft's generic WavLM-Base+ checkpoint, followed by a randomly
   initialized MHFA backend.
2. The project's published `model000000018.model`, where both WavLM and MHFA
   have already been trained for speaker verification.

The second option is the stronger and more relevant initialization for our
speaker task. A restricted Kaggle audit established that it is a tensor-only
state dictionary with:

- 248 WavLM tensors containing 94,381,936 elements
- 10 MHFA tensors containing 2,302,554 elements
- one 7,500-by-256 source AAM-Softmax class weight containing 1,920,000
  elements

The source classifier represents upstream pseudo-labeled VoxCeleb identities,
not TidyVoice or ViMD speakers, so it cannot be reused as a target head.

## Decision

- Pin source repository `theolepage/wavlm_ssl_sv` at full revision
  `bfb8527de83b5347fb81b1e9e31be241656ca103`.
- Select the official published checkpoint `model000000018.model` from Google
  Drive file ID `1RabuRETASqhh39K8weSoNkBa5DGRvgyx`.
- Require the observed checkpoint SHA-256
  `0178a115dc0a43a94a71287e51d1df5016c2aeefc04169548dad40ac8a6e67da`.
- Treat that digest as our reproducibility fingerprint because the publisher
  does not provide an independent checksum.
- Download transactionally to a partial file, hash it, and rename it only
  after successful verification.
- Load with `torch.load(..., weights_only=True)` and require the audited
  259-entry tensor-only structure.
- Strictly load the 248 WavLM and 10 MHFA entries after removing only their
  known upstream prefixes.
- Deliberately exclude the one 7,500-class source-loss tensor.
- Return a shared L2-normalized `[batch, 256]` embedding.
- Assert 94,381,936 WavLM parameters, 2,302,554 MHFA parameters, and
  96,684,490 combined adapter parameters.

## Architecture Source Policy

The required upstream WavLM files total approximately 61 KiB and are tightly
coupled. Rather than manually rewriting them, the adapter downloads the exact
pinned source snapshot, checks `WavLM.py`, `modules.py`, `Spk_Encoder.py`, and
the license against audited SHA-256 values, then imports the authenticated
WavLM source under an isolated module namespace.

The much smaller MHFA class is adapted locally with its upstream MIT license
and provenance record. Its parameter-bearing names and computation remain
strict-checkpoint compatible.

The fine-tuned checkpoint does not carry WavLM's configuration dictionary.
Construction therefore uses the exact 35-field configuration recovered from
the safely inspected official Microsoft WavLM-Base+ artifact. Strict loading
then verifies that this configuration reproduces the checkpoint architecture.

## Gradient Boundary

The pinned upstream `WavLM.extract_features` executes its convolutional feature
extractor inside `torch.no_grad()`. Its trainer creates optimization groups for
the 12 Transformer encoder layers and the MHFA backend, not the convolutional
front end. The adapter preserves this official behavior.

Therefore, the acceptance gate requires finite, non-zero Transformer and MHFA
gradients. It also requires convolutional-feature and input-waveform gradients
to be absent as designed. Claiming full end-to-end waveform gradient flow
would contradict the selected implementation.

The final target-training optimizer, learning rates, freezing schedule,
precision, batch size, and accumulation remain pending controlled Validation
and memory experiments.

## Dependency Decision

The optional group installs:

```text
gdown>=5.2,<5.3
huggingface-hub>=1.11,<1.12
```

PyTorch remains supplied by Kaggle and is intentionally not declared by the
project, preventing replacement of its CUDA-compatible build.

## Acceptance Gate

Run `scripts/smoke_test_wavlm_mhfa_adapter.py` on a Kaggle Tesla T4 with real
TidyVoice speech and require:

- all executable upstream source hashes match before import
- checkpoint SHA-256 matches before restricted deserialization
- WavLM and MHFA strict loads have no missing or unexpected keys
- the upstream source classifier is excluded
- parameter count equals 96,684,490
- a 64,240-sample crop produces finite shape `[1, 256]`
- the output L2 norm equals 1 within tolerance
- repeated evaluation cosine similarity is at least 0.99999
- Transformer gradients are present, finite, and non-zero
- MHFA gradients are present, finite, and non-zero
- convolutional feature and waveform gradients are absent as designed
- peak gradient-gate CUDA allocation is recorded
- structured JSON evidence is saved and reviewed

The local regression suite contains 107 passing tests after adding six
dependency-free WavLM+MHFA syntax, checkpoint-surface, license, provenance,
identity, and audit-artifact checks.

## Kaggle GPU Evidence

The acceptance gate passed on `cuda:0` with:

- GPU class: Tesla T4
- PyTorch: `2.10.0+cu128`
- CUDA build: `12.8`
- Source revision: `bfb8527de83b5347fb81b1e9e31be241656ca103`
- Checkpoint SHA-256:
  `0178a115dc0a43a94a71287e51d1df5016c2aeefc04169548dad40ac8a6e67da`
- WavLM+MHFA parameters: 96,684,490
- Upstream source classifier included: no
- Real TidyVoice sample: 122,496 samples, 7.656 seconds
- Evaluation crop: 64,240 samples
- Gradient crop: 48,240 samples
- Output shape: `[1, 256]`
- Output L2 norm: 1.0
- Repeat cosine similarity: 1.0
- Transformer gradient present, finite, and non-zero: passed
- MHFA gradient present, finite, and non-zero: passed
- Feature-extractor gradient absent under the official boundary: passed
- Waveform gradient absent under the official boundary: passed
- Peak CUDA allocation during the gradient gate: 838,415,872 bytes
  (approximately 0.781 GiB)

Canonical waveform SHA-256:
`9d608813bc66beb3d0b0bf72b23e7776abbf7121337c4b3beb737da5a627ac92`

Evidence artifact:
`results/model_audit/wavlm_mhfa_adapter_smoke.json`

Artifact SHA-256:
`edcc83a454a652c87301d4c0ac6c957f8f0f0c544a41097fbad9f0da52b44b70`

The 0.781 GiB observation covers one forward/backward compatibility gate. It
does not include optimizer states, a production batch, gradient accumulation,
DataLoader memory, or cached evaluation embeddings, so it must not be used as
the final training-memory estimate.

This establishes authenticated-source loading, complete pretrained WavLM and
MHFA compatibility, deterministic inference, and gradient flow through the
officially optimized components. It does not establish TidyVoice or ViMD
verification accuracy and must not be reported as a benchmark result.

## Advantages

- Both WavLM and MHFA start from speaker-verification-trained weights.
- The incompatible upstream classifier is cleanly separated from the encoder.
- Source and checkpoint hashes prevent silent architecture or artifact drift.
- Exact source import avoids a large, error-prone manual WavLM rewrite.
- The common normalized output supports identical downstream scoring and
  verification metrics across all three model families.

## Disadvantages and Limitations

- The checkpoint is hosted externally on Google Drive and lacks a publisher
  checksum; availability depends on that service.
- Authenticated upstream Python source is executed at runtime, creating a
  network/cache dependency even though hashes are checked first.
- The convolutional feature extractor cannot be fine-tuned without changing
  upstream behavior.
- WavLM is the largest of the three adapters, so its training settings need a
  dedicated T4 memory experiment.
- The published VoxCeleb result cannot be presented as our dataset result.

## Primary Sources

- Pinned WavLM+MHFA repository:
  <https://huggingface.co/theolepage/wavlm_ssl_sv/tree/bfb8527de83b5347fb81b1e9e31be241656ca103>
- Official Microsoft WavLM repository:
  <https://github.com/microsoft/unilm/tree/master/wavlm>
- WavLM paper:
  <https://arxiv.org/abs/2110.13900>
- WavLM+MHFA paper:
  <https://arxiv.org/abs/2406.02285>
