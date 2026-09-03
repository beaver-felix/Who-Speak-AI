# Decision 010: Shared AAM-Softmax and Model-Specific Update Policies

Date: 2026-08-22
Status: objective and memory candidates accepted; effectiveness pending

## Question

How should ECAPA-TDNN, RawNet3, and WavLM+MHFA be optimized so that the six
dataset/model experiments remain scientifically comparable without ignoring
real architectural differences?

## Decision

All models use one supervised AAM-Softmax objective with margin `0.2`, scale
`30`, non-easy-margin thresholding, mean cross-entropy, and a newly initialized
target-dataset class matrix. TidyVoice and ViMD use their own training-speaker
counts, so classifier heads are never transferred between datasets.

The loss implementation:

1. L2-normalizes embeddings and class weights;
2. computes cosine logits;
3. applies the angular margin only to each target logit;
4. uses a target-column scatter instead of a dense one-hot allocation; and
5. performs angular operations in float32 under FP16 mixed-precision training.

The optimizer scope is architecture-specific:

| Model | Optimizer | Trainable scope | Initial learning rates | Weight decay |
|---|---|---|---|---:|
| ECAPA-TDNN | Adam | complete pretrained encoder + new head | encoder/head `1e-4` | `2e-6` |
| RawNet3 | Adam | complete pretrained encoder + new head | encoder/head `1e-4` | `5e-5` |
| WavLM+MHFA | AdamW | 12 Transformer layers + MHFA + new head | Transformer `2e-5`; MHFA/head `5e-3` | `0` |

The WavLM convolutional feature extractor, input projection, positional
convolution, and other non-Transformer WavLM parameters remain outside the
optimizer. This preserves the audited official `torch.no_grad()` front-end
boundary and the pinned source optimizer structure. Layer-wise learning-rate
decay is represented explicitly and starts at `1.0`, matching the source base
configuration; later experiments may test values below one.

Every enabled parameter must appear in exactly one optimizer group. Construction
fails if a parameter is missing or duplicated.

## Evidence and Rationale

- The SpeechBrain VoxCeleb ECAPA recipe uses AAM margin `0.2`, scale `30`, Adam,
  and weight decay `2e-6`. Its from-scratch learning rate is `1e-3`; a
  SpeechBrain pretrained-ECAPA transfer recipe uses `1e-4`. Because this project
  starts from pretrained weights on new target datasets, `1e-4` is the safer
  screening hypothesis.
- The pinned RawNet3 upstream trainer uses AAM-Softmax and Adam; its audited
  model configuration specifies weight decay `5e-5`. The source trains at
  `1e-3`, but this project begins with a strong pretrained checkpoint, so
  `1e-4` is the conservative first fine-tuning candidate.
- The pinned WavLM+MHFA base configuration specifies AAM margin `0.2`, scale
  `30`, Transformer learning rate `2e-5`, MHFA learning rate `5e-3`, AdamW, and
  layer-wise decay `1.0`. These values are retained as the first source-aligned
  candidate.
- The source WavLM system's Loss-Gated Learning and label correction are not
  adopted. They address iteratively refined pseudo-labels, whereas TidyVoice
  and ViMD provide supervised speaker labels. Adding them would introduce an
  unneeded model-specific objective and weaken comparison clarity.
- Local dependency-free tests validate configuration structure, numerical
  safeguards by source inspection, explicit freezing, and exact optimizer
  partition logic. Concrete tensor behavior and memory still require Kaggle.

Primary source records:

- SpeechBrain ECAPA VoxCeleb recipe:
  <https://github.com/speechbrain/speechbrain/blob/develop/recipes/VoxCeleb/SpeakerRec/hparams/train_ecapa_tdnn.yaml>
- SpeechBrain pretrained-ECAPA transfer example:
  <https://github.com/speechbrain/speechbrain/blob/develop/recipes/CommonLanguage/lang_id/hparams/train_ecapa_tdnn.yaml>
- Pinned RawNet3 trainer:
  <https://github.com/clovaai/voxceleb_trainer/tree/f51bab870672a9b0b50fa158b4e30f329e7866d7>
- Pinned WavLM+MHFA trainer:
  <https://github.com/theolepage/wavlm_ssl_sv/tree/bfb8527de83b5347fb81b1e9e31be241656ca103>

## Validation Plan

1. Run the committed real-audio T4 calibration for each model with the ViMD
   class count (`10,291`), the larger classifier and therefore the conservative
   memory case.
2. Accept a calibration size only when loss and pre-clipping gradient norm are
   finite and a target-head parameter is proven to change after the optimizer
   call. Dynamic-loss-scale overflow or a skipped optimizer step fails closed.
3. Select no more than 80% of the largest passing batch size.
4. Confirm the candidate using multiple distinct real batches, including loss,
   finite gradients, peak memory, and optimizer updates.
5. Run a short validation screen. Compare EER first, then minDCF and FAR/FRR.
6. If learning is unstable or validation stalls, test one predeclared change at
   a time. Record the hypothesis, result artifact, and acceptance reason.
7. Test AAM margin `0.3` only after the `0.2` control completes; do not change
   margin for only one architecture without reporting it as a separate ablation.

## Acceptance Boundary

The shared objective and exact update scopes are accepted methodology. Current
learning rates and weight decay values remain candidates until target-validation
artifacts justify them. The calibration script measures capacity only; repeated
one-sample batches are not convergence or accuracy evidence.

## Rejected Preliminary Calibration

The first schema-1 T4 archive was rejected before acceptance even though every
candidate fit in memory. Its SHA-256 was
`b0737eb2b921868f8867fb812309a1281fc52ff8133b3575ddaff959fb0a694e`.
ECAPA reported a `NaN` gradient norm at batch 4 and infinity at batch 8;
RawNet3 reported `NaN` at batches 4, 8, 16, and 24; WavLM+MHFA reported
infinity at batch 4. The original gate called `GradScaler.step`, but did not
prove that GradScaler actually applied the update. Consequently those records
showed allocation capacity, not a successful full optimizer step.

Schema 2 corrects this by starting from a conservative dynamic scale, backing
off for up to eight attempts, requiring finite loss and pre-clipping gradient
norm, and verifying an exact target-head parameter change. A standalone
dependency-free validator rejects old schemas and all non-finite or skipped
steps. The rejected archive remains locally recoverable but is not accepted
experiment evidence and is not committed.

## Accepted Schema-2 Calibration

The corrected archive SHA-256 is
`da0499d164e5940d50c506518407d75270e52b5c22c06515a62136600b535fc4`.
Every tested size passed on a Tesla T4 using PyTorch `2.10.0+cu128`, CUDA
`12.8`, FP16, the ViMD worst-case `10,291`-class head, and dynamic scale
`1024`. Every size passed on its first attempt with finite loss, finite
pre-clipping gradient norm, and a proven target-head update.

| Model | Passing sizes | Largest | Next candidate | Candidate peak | Artifact SHA-256 |
|---|---|---:|---:|---:|---|
| ECAPA-TDNN | 4, 8, 16, 24, 32 | 32 | 24 | 1.990 GiB | `e2b651670cb509954f0706345ec2801d61006a76ab8867033fbb495752d30397` |
| RawNet3 | 4, 8, 16, 24, 32 | 32 | 24 | 3.760 GiB | `afe8df72a95acdd5d05441a29d8d25ca22879c2b29e924423a10edbfca86113d` |
| WavLM+MHFA | 1, 2, 4, 6, 8 | 8 | 6 | 1.867 GiB | `8d0d8219fb4e816dd9ca6628a9e2fecacc18cebc283301422a9ee40102c9b30b` |

For ECAPA and RawNet3, batch 24 is the largest tested value below the 80%
boundary. For WavLM+MHFA, 80% of 8 is 6.4, so tested batch 6 is selected.
These are memory candidates, not final training settings. A real multi-record,
multi-batch mini-run remains mandatory.

The evidence configurations exactly reproduced these pre-selection hashes:

- ECAPA-TDNN: `91a50f6d2e688e8956e95ad9cc3e9db6a7ff9e69bce7333b935f405d6f6aed1d`
- RawNet3: `a58e1b175f1bbc3dea7a65986fa62ebd2daeadb4ae6ac940279a7de1537e23b6`
- WavLM+MHFA: `cda92164886e331d319cfafaed1ea15c0d69359997877b4301d75ab09fa39117`

## Advantages

- Preserves one comparison objective across all architectures.
- Makes necessary architectural differences explicit and auditable.
- Reuses strong upstream evidence without pretending source-domain settings are
  already optimal on the target datasets.
- Avoids a large one-hot tensor and stabilizes angular math under FP16.
- Prevents silent WavLM front-end updates and optimizer-group overlap.

## Disadvantages

- The first learning-rate candidates may still be suboptimal.
- A shared margin can favor one embedding geometry; the planned ablation is
  needed before claiming optimality.
- WavLM uses more parameter groups and a different optimizer, reducing strict
  optimizer equality, but forcing identical updates would violate its source
  architecture and likely reduce effectiveness.
- Memory calibration with repeated audio does not represent data-loading cost,
  convergence, or final validation performance.
