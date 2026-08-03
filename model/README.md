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
- After evaluation, expected a .csv with the following schema:
```csv
ID, EER, FAR, FRR, 'TAR@FAR 5%', 'TAR@FAR 1%', 'TAR@FAR 0.1%'
...
```
note:
 + EER = Equal Error      Rate (Lower is better)
 + FAR = False Acceptance Rate (lower is better)
 + FRR = False Rejection  Rate (lower is better)
 + TAR = True  Acceptance Rate (Higer is better)
 + FAR = False Acceptance Rate (Gauged measuring TAR@FAR)
