# Speaker Verification Resources

## Model Architectures
| Architecture | Core Concept / Mechanism | Reference Implementation |
|---|---|---|
| **ECAPA-TDNN** | Channel attention, multi-layer aggregation, D-TDNN backbone | [TaoRuijie/ECAPA-TDNN](https://github.com/TaoRuijie/ECAPA-TDNN) |
| **ERes2Net-large** | Local-global feature fusion via channel-wise multi-scale Res2Net blocks | [modelscope/3D-Speaker](https://github.com/modelscope/3D-Speaker) |
| **RawNet3** | Raw audio waveform processing directly (eliminates STFT/MFCC conversion) | [Jungjee/RawNet](https://github.com/Jungjee/RawNet) |
| **WavLM + MHFA** | Pre-trained SSL transformer encoder + Multi-Head Factorized Attentive Pooling | [theolepage/wavlm_ssl_sv](https://github.com/theolepage/wavlm_ssl_sv) |
note: only evaluate `WavLM + MHFA`, no fine-tune.

## Dataset Information
| Name | Focus / Language | Description | src |
|---|---|---|---|
| **VoxCeleb 1** | Global | De facto standard benchmark for in-the-wild speaker verification | [VoxCeleb](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) |
| **TidyVoice2026** | Global | From TidyVoice 2026 Challenge | [TidyVoice](https://mozilladatacollective.com/datasets/cmihtsewu023so207xot1iqqw) |
| **ViMD** | Vietnamese | Multi-dialect dataset covering speakers across 63 Vietnamese provinces | [ViMD HF](https://huggingface.co/datasets/nguyendv02/ViMD_Dataset) |

## Expected Output
- After evaluation, expected a .csv with the following schema:
```csv
ID, FAR, FRR, 'TAR@FAR 5%', 'TAR@FAR 1%', 'TAR@FAR 0.1%'
...
```
note:
 + FAR = False Acceptance Rate (lower is better)
 + FRR = False Rejection  Rate (lower is better)
 + TAR = True  Acceptance Rate (Higer is better)
 + FAR = False Acceptance Rate (Gauged measuring TAR@FAR)
