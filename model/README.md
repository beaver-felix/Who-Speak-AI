# Speaker Verification Resources

## Model Architectures
| Architecture | Core Concept / Mechanism | Reference Implementation |
|---|---|---|
| **ECAPA-TDNN** | Channel attention, multi-layer aggregation, D-TDNN backbone | [TaoRuijie/ECAPA-TDNN](https://github.com/TaoRuijie/ECAPA-TDNN) |
| **ERes2Net / ERes2NetV2** | Local-global feature fusion via channel-wise multi-scale Res2Net blocks | [modelscope/3D-Speaker](https://github.com/modelscope/3D-Speaker) |
| **CAM++** | Context-Aware Masking with Densely Connected TDNN (fast inference) | [modelscope/3D-Speaker](https://github.com/modelscope/3D-Speaker) |
| **RawNet2 / RawNet3** | Raw audio waveform processing directly (eliminates STFT/MFCC conversion) | [Jungjee/RawNet](https://github.com/Jungjee/RawNet) |
| **ResNet34 + ASP** | 2D Spectro-temporal CNN paired with Attentive Statistics Pooling | [clovaai/voxceleb_trainer](https://github.com/clovaai/voxceleb_trainer) |
| **WavLM + MHFA / Conformer** | Pre-trained SSL transformer encoder + Multi-Head Factorized Attentive Pooling | [theolepage/wavlm_ssl_sv](https://github.com/theolepage/wavlm_ssl_sv) |

## Dataset Information
| Name | Focus / Language | Description | src |
|---|---|---|---|
| **VoxCeleb 1 & 2** | English / Global | De facto standard benchmark for in-the-wild speaker verification | [VoxCeleb](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) |
| **CN-Celeb 1 & 2** | Chinese | Large-scale multi-genre dataset (covers 11 real-world audio genres) | [OpenSLR-82](https://www.openslr.org/82/) |
| **3D-Speaker Dataset** | Multi-dialect | Multi-device, multi-distance, and multi-dialect benchmark | [3D-Speaker](https://github.com/modelscope/3D-Speaker) |
| **DeepMine** | Persian / English | Suitable for both text-dependent and text-independent SV research | [OpenSLR-103](https://www.openslr.org/103/) |
| **Vietnam-Celeb** | Vietnamese | In-the-wild Vietnamese SV benchmark dataset (INTERSPEECH 2023) | [Vietnam-Celeb](https://github.com/thanhpv2102/Vietnam-Celeb.Interspeech) |
| **ViMD** | Vietnamese | Multi-dialect dataset covering speakers across 63 Vietnamese provinces | [ViMD HF](https://huggingface.co/datasets/nguyendv02/ViMD_Dataset) |
| **VIVOS** | Vietnamese | Read speech corpus useful for quick fine-tuning or baseline testing | [VIVOS HF](https://huggingface.co/datasets/vivos) |
