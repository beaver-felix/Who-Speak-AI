# Decision 016: WavLM Non-Finite Precision Fallback

Date: 2026-08-22

## Status

Accepted for the corrected WavLM+MHFA resource-constrained run. The first
WavLM attempt is rejected and must not be resumed.

## Observed evidence

The first Cường WavLM+MHFA/TidyVoice run completed 578 finite optimizer steps
and failed at deterministic batch 579 of epoch zero. Its latest complete
checkpoint was step 500. All 578 recorded losses were finite:

- minimum loss: `13.319659`;
- maximum loss: `20.937525`;
- mean loss: `16.027702`;
- maximum pre-clipping gradient norm: `192.13385` at step 288;
- gradient clipping remained active;
- AMP loss scale remained `1024`;
- learning rates remained within the configured `2e-5` to `5e-3` range.

The failure occurred before the classification objective because the WavLM
embedding for one batch contained a non-finite value. The evidence does not
support general optimization divergence or a corrupt dataset. It supports an
isolated, real-audio FP16 numerical failure. This is a diagnosis, not proof of
the exact internal operator that overflowed.

## Decision

Training remains FP16 by default. Only the WavLM+MHFA adapter receives a
bounded recovery path:

1. Save Python, NumPy, CPU-Torch, and CUDA-Torch random states immediately
   before the FP16 forward pass.
2. If and only if the resulting embedding is non-finite, restore every saved
   random state and recompute that same batch with autocast disabled.
3. Continue the step only if the FP32 embedding and objective are finite.
4. Record `train/fp32_fallback = 1`; ordinary batches record zero.
5. Fail closed with bounded utterance identifiers if FP32 is also non-finite.

Restoring all random states is necessary because the pinned upstream WavLM
implementation uses NumPy randomness for layerdrop in addition to Torch
randomness for neural-network dropout. The retry therefore keeps batch
membership, crop, order, weights, and stochastic masks fixed; arithmetic
precision is the only intentional change.

Evaluation uses the same model-scoped policy. A non-finite WavLM FP16
evaluation batch is recomputed once in FP32 and the fallback batch count is
included in extraction evidence. ECAPA-TDNN and RawNet3 retain their original
fail-closed behavior.

Before starting the two real WavLM experiments, the one-click worker runs a
three-step, batch-six, real-TidyVoice WavLM training gate entirely in FP32.
This proves that the fallback path fits on one Tesla T4 and produces finite
gradients and parameter updates. Passing the gate does not claim model quality.

## Restart and checkpoint policy

The corrected checkpoint format includes a weights-only-safe serialization of
NumPy's MT19937 state. This is required for exact WavLM layerdrop continuity
after interruption. A checkpoint without that state is rejected explicitly.

Cường must import the corrected notebook into a new Kaggle notebook/session
and use Save & Run All once. No checkpoint or output from attempt one may be
copied into the corrected run.

## Advantages

- Preserves fast FP16 execution for all ordinary batches.
- Changes precision only for a demonstrated WavLM failure mode.
- Preserves stochastic equivalence during the retry.
- Produces an auditable fallback count and actionable failure provenance.
- Proves FP32 fallback feasibility before spending hours on the full run.

## Disadvantages

- A fallback batch executes its forward pass twice and is slower.
- Capturing random states adds small WavLM-only overhead to every train batch.
- Exact restart checkpoints are intentionally incompatible with the rejected
  first attempt.
- The root numerical operator is not isolated; the correction mitigates the
  observed boundary rather than changing the upstream architecture.

## Acceptance criteria

- The complete local regression suite passes.
- The WavLM-only FP32 preflight gate passes on Kaggle T4 with real audio.
- Both corrected experiments finish with finite training and evaluation
  evidence, or fail closed with bounded provenance.
- The final archive retains the preflight evidence and fallback metrics.
