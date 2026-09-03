# Decision 007: ECAPA-TDNN Adapter Implementation

Status: Accepted
Date: 2026-08-22

## Context

The shared pipeline requires every architecture to accept canonical
`[batch, time]` 16 kHz waveforms and return one `[batch, embedding_dim]`
speaker vector. The pinned ECAPA source is SpeechBrain's pretrained VoxCeleb
model, whose inference package also includes a 7,205-class source classifier.
That classifier does not represent TidyVoice or ViMD identities and must not be
reused as the target training head.

SpeechBrain 1.1.0 `EncoderClassifier.encode_batch` performs:

```text
waveform
    -> compute_features
    -> sentence mean/variance normalization
    -> ECAPA embedding_model
    -> optional source embedding normalization
```

The official method is differentiable and does not add an inference-only
`no_grad` context. The pretrained interface defaults to freezing parameters,
so the loader must explicitly use `freeze_params=False` for fine-tuning.

## Decision

- Pin model ID `speechbrain/spkrec-ecapa-voxceleb`.
- Pin full revision
  `0f99f2d0ebe89ac095bcc5903c4dd8f72b367286`.
- Require the Kaggle-validated SpeechBrain version `1.1.0`.
- Resolve the immutable Hugging Face snapshot before SpeechBrain loading.
- Require all official hyperparameter, encoder, normalization, classifier, and
  label artifacts needed by SpeechBrain's official loader.
- Use `EncoderClassifier.from_hparams(..., freeze_params=False)` only to
  instantiate the official architecture and load compatible checkpoints.
- Extract and register only `compute_features`, `mean_var_norm`, and
  `embedding_model` in the project adapter.
- Exclude the upstream 7,205-class VoxCeleb classifier from the adapter state
  and optimizer parameters.
- Reproduce the official feature/normalization/embedding path directly so
  gradients remain visible to the shared training layer.
- Convert upstream `[batch, 1, 192]` output into the shared `[batch, 192]`
  contract.
- Apply explicit L2 normalization for model-independent cosine scoring.
- Assert the audited encoder parameter count of 20,767,552.
- Provide an explicit freeze/unfreeze method; the actual fine-tuning policy is
  still pending controlled Validation and Kaggle memory evidence.

## Dependency Decision

PyTorch is not listed in the optional ECAPA dependency group. Kaggle's existing
CUDA-matched PyTorch must remain installed. The optional group adds only the
audited SpeechBrain and Hugging Face client versions:

```text
speechbrain==1.1.0
huggingface-hub>=1.11,<1.12
```

The concrete ECAPA module imports these dependencies only when requested.
Local configuration, data, metric, and adapter-contract tests therefore remain
usable without installing a large CPU-only or incompatible PyTorch wheel.

## Acceptance Gate

Run `scripts/smoke_test_ecapa_adapter.py` on a Kaggle Tesla T4 and require:

- resolved revision equals the full pinned commit
- parameter count equals 20,767,552
- real 16 kHz speech produces shape `[1, 192]`
- every embedding value is finite
- explicit L2 norm equals 1 within tolerance
- repeated evaluation cosine similarity is at least 0.99999
- input waveform gradients are finite and non-zero
- at least one encoder parameter gradient is finite and non-zero
- structured JSON evidence is saved and reviewed

The local dependency-free regression suite contains 96 passing tests. Concrete
PyTorch/SpeechBrain behavior was evaluated separately on the pinned Kaggle
runtime.

## Kaggle GPU Evidence

The acceptance gate passed on `cuda:0` with:

- GPU class: Tesla T4
- PyTorch: `2.10.0+cu128`
- CUDA build: `12.8`
- SpeechBrain: `1.1.0`
- Resolved model revision:
  `0f99f2d0ebe89ac095bcc5903c4dd8f72b367286`
- Adapter parameters: 20,767,552
- Trainable adapter parameters: 20,767,552
- Real sample: official `example1.wav`, 52,173 samples, 3.2608125 seconds
- Output shape: `[1, 192]`
- Output L2 norm: 1.0
- Repeat cosine similarity: 1.0
- Embedding shape and finiteness: passed
- Input gradient finite and non-zero: passed
- Encoder gradient present, finite, and non-zero: passed

Waveform SHA-256:
`48aedc3a10b14b49ebe8da2efd1dd91cbe7dbbaf58278732e7fdb04f6d6cc1e9`

Evidence artifact:
`results/model_audit/ecapa_adapter_smoke.json`

Artifact SHA-256:
`2409df41e4d7e1fde356cb1bc5da3ee1d4330754ab420f3e4b7a1287d73baa2b`

This establishes pinned-source loading, adapter shape compatibility,
deterministic inference, and fine-tuning gradient flow. It does not establish
TidyVoice or ViMD accuracy and must not be reported as a benchmark result.

## Strict-Determinism Training Correction

The first real TidyVoice pilot failed before its first optimizer step on
2026-08-22. PyTorch `2.10.0+cu128` reported that
`reflection_pad1d_backward_out_cuda` has no deterministic implementation while
strict deterministic algorithms were enabled. Earlier adapter and multi-batch
gates had not enabled that strict runtime flag, so they proved gradient flow but
did not exercise this boundary.

The accepted correction does not use `warn_only=True` and does not change the
padding mode. During one adapter forward call, one-dimensional reflection
padding is expressed with forward-equivalent tensor slices, flips, and
concatenation. Native `torch.nn.functional.pad` remains in use for every other
mode and dimensionality, and the original function is restored in a `finally`
block. A process lock protects the temporary scoped replacement. This preserves
the pretrained reflection-padding semantics while providing a deterministic
autograd graph on CUDA.

The corrected adapter must rerun its real GPU gradient gate with
`torch.use_deterministic_algorithms(True)` before the pilot is resumed. Until
that gate passes, the correction is implementation-ready but not empirically
accepted.

## Advantages

- Uses the official pinned architecture and checkpoint loader.
- Removes a source classifier that would silently mismatch target identities.
- Preserves full gradient flow for later target-domain fine-tuning.
- Produces the same normalized embedding shape expected from future adapters.
- Prevents an optional dependency install from replacing Kaggle PyTorch.

## Disadvantages and Limitations

- Loading still depends on SpeechBrain's trusted HyperPyYAML configuration
  execution, so only the pinned official snapshot is accepted.
- The official source classifier must be downloaded for loader compatibility
  even though it is discarded afterward.
- Concrete behavior cannot be tested in the lightweight local environment.
- The deterministic reflection implementation temporarily replaces PyTorch's
  functional pad entry point under a process lock; the runner therefore keeps
  one model-forward thread per process.
- Full fine-tuning may exceed a selected memory budget; freezing policy,
  precision, crop duration, batch size, and accumulation remain unresolved.
- L2-normalized embeddings are suitable for cosine/AAM-style objectives, but
  the target loss and head still require a separate documented decision.

## Primary Sources

- SpeechBrain 1.1.0 `EncoderClassifier` source:
  <https://github.com/speechbrain/speechbrain/blob/v1.1.0/speechbrain/inference/classifiers.py>
- SpeechBrain 1.1.0 pretrained interface source:
  <https://github.com/speechbrain/speechbrain/blob/v1.1.0/speechbrain/inference/interfaces.py>
- Pinned ECAPA model repository:
  <https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/tree/0f99f2d0ebe89ac095bcc5903c4dd8f72b367286>
