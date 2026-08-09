# Speaker Verification Resources

## Model Architectures
| Architecture | Core Concept / Mechanism | Reference Implementation |
|---|---|---|
| **ECAPA-TDNN** | Channel attention, multi-layer aggregation, D-TDNN backbone | [speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) |
| **RawNet3** | Raw audio waveform processing directly (eliminates STFT/MFCC conversion) | [Jungjee/RawNet](https://huggingface.co/jungjee/RawNet3) |
| **WavLM + MHFA** | Pre-trained SSL transformer encoder + Multi-Head Factorized Attentive Pooling | [theolepage/wavlm_ssl_sv](https://huggingface.co/theolepage/wavlm_ssl_sv) |

## Dataset Information
| Name | Focus / Language | Description | src |
|---|---|---|---|
| **TidyVoice2026** | Global | From TidyVoice 2026 Challenge | [TidyVoice](https://mozilladatacollective.com/datasets/cmihtsewu023so207xot1iqqw) |
| **ViMD** | Vietnamese | Multi-dialect dataset covering speakers across 63 Vietnamese provinces | [ViMD HF](https://huggingface.co/datasets/nguyendv02/ViMD_Dataset) |

## Expected Output

After evaluation, each model produces a CSV file with the following schema:

```csv
ID,EER,FAR,FRR,TAR@FAR 5%,TAR@FAR 1%,TAR@FAR 0.1%
```

The CSV stores rates as percentages. For example, an internal EER value of
`0.0125` is written as `1.25`.

- **EER:** Equal Error Rate; lower is better.
- **FAR:** False Acceptance Rate; lower is better.
- **FRR:** False Rejection Rate; lower is better.
- **TAR:** True Acceptance Rate; higher is better and equals `1 - FRR`.
- **TAR@FAR:** Maximum TAR obtained without exceeding the specified FAR.

## Implemented Pipelines

The resources above are research references. The pipelines implemented under
`model/Thanh/` use the following models and configurations.

### RawNet3-Style Pipeline

- **Implementation:** [`Thanh/RawNet3/`](Thanh/RawNet3/)
- **Architecture reference:** [jungjee/RawNet3](https://huggingface.co/jungjee/RawNet3)
- **Pretrained checkpoint:** None
- **Implementation type:** Native PyTorch RawNet3-style encoder
- **Input:** 16 kHz mono raw waveform
- **Audio duration:** 3 seconds
- **Embedding dimension:** 192
- **Classification loss:** Additive Margin Softmax with Cross-Entropy
- **AM-Softmax scale:** 30.0
- **AM-Softmax margin:** 0.20
- **Mixed precision:** FP16
- **Checkpoint output:** `model/Thanh/checkpoints/rawnet3/best_model.pt`

The implemented encoder follows RawNet-family concepts but is not
checkpoint-compatible with the official RawNet3 implementation. Experimental
results must therefore be reported as results from the project's native
RawNet3-style baseline.

### WavLM Speaker Verification Pipeline

- **Implementation:** [`Thanh/WavLM/`](Thanh/WavLM/)
- **Pretrained model:** [microsoft/wavlm-base-plus-sv](https://huggingface.co/microsoft/wavlm-base-plus-sv)
- **Model class:** Hugging Face `AutoModelForAudioXVector`
- **Input:** 16 kHz mono waveform
- **Audio duration:** 3 seconds
- **Default pooling:** Pretrained x-vector pooling
- **Alternative pooling:** Mean or statistics pooling
- **Embedding dimension:** 512
- **Backbone strategy:** Frozen by default
- **Backbone learning rate:** `1e-5`
- **Classification-head learning rate:** `1e-4`
- **Mixed precision:** FP16
- **Checkpoint output:** `model/Thanh/checkpoints/wavlm/best_model.pt`

The project keeps
[theolepage/wavlm_ssl_sv](https://huggingface.co/theolepage/wavlm_ssl_sv)
as an advanced research reference. It is not used by the current Hugging Face
wrapper because its MHFA architecture and externally hosted checkpoint require
a separate integration workflow.

## Dataset Plan

### Primary Dataset: ViMD

- **Source:** [nguyendv02/ViMD_Dataset](https://huggingface.co/datasets/nguyendv02/ViMD_Dataset)
- **Language:** Vietnamese
- **Available splits:** `train`, `valid`, and `test`
- **Speaker field:** `speakerID`
- **Audio field:** `audio`
- **License:** CC BY-NC-ND 4.0
- **Access strategy:** Hugging Face streaming inside a Kaggle GPU Notebook
- **Preprocessing:** Mono conversion, 16 kHz resampling, and fixed-duration
  crop/padding

The final train, validation, and test manifests will be documented after
inspecting speaker overlap and the number of recordings per speaker.

### Secondary Verification Dataset: TidyVoiceX_ASV

- **Source:** [TidyVoiceX_ASV](https://mozilladatacollective.com/datasets/cmihtsewu023so207xot1iqqw)
- **Purpose:** Cross-lingual speaker-verification evaluation
- **Restriction:** Must not be used for speaker identification or recovering
  speaker identities
- **Size:** Approximately 36.72 GB compressed
- **Status:** Optional future benchmark due to Kaggle storage constraints

## Evaluation Metrics

All verification scores use the following convention:

- Higher scores indicate stronger speaker matches.
- Genuine trials use label `1`; impostor trials use label `0`.
- A trial is accepted when `score >= threshold`.
- FAR and FRR are reported at the empirical threshold closest to EER.
- TAR@FAR uses a separate operational threshold for each requested FAR.

In addition to the required CSV fields, JSON output includes minDCF, accuracy,
decision thresholds, trial counts, and inference latency.

For target FAR `x`, TAR@FAR is defined as:

```text
TAR@FAR(x) = max TAR(t), over thresholds t whose FAR(t) <= x
TAR(t) = 1 - FRR(t)
```
