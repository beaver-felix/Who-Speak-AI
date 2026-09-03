# Decision 003: Canonical Audio Preprocessing

Status: Accepted  
Date: 2026-08-20

## Context

TidyVoice contains mono 16 kHz PCM-16 WAV files. ViMD embeds mono or stereo
44.1/48 kHz PCM-16 WAV bytes in Parquet rows. All three speaker encoders need
one consistent waveform representation so dataset format cannot become a
model-specific confounder.

## Decision

Every model receives mono `float32` waveform audio at 16 kHz.

- Multichannel audio is downmixed using the arithmetic channel mean.
- Sample-rate conversion uses SciPy polyphase anti-aliasing resampling.
- Output length is adjusted to the nearest duration-preserving sample count.
- Natural amplitude is preserved without peak, RMS, or loudness normalization.
- Invalid, empty, or non-finite signals fail before modeling.
- TidyVoice is decoded from standalone WAV files.
- ViMD is decoded lazily from embedded Parquet bytes.
- Each ViMD loader worker caches one decoded Parquet row group.

Training crops use process-independent SHA-256-derived seeds. Short recordings
repeat their samples instead of adding silence. Evaluation uses deterministic
evenly spaced crops; the duration and crop count remain Validation-selected
configuration variables.

## Real-Data Evidence

| Measure | TidyVoice | ViMD |
|---|---:|---:|
| Source sample rate | 16,000 Hz | 44,100 Hz |
| Source channels | 1 | 2 |
| Canonical sample rate | 16,000 Hz | 16,000 Hz |
| Canonical duration | 7.656 s | 9.239 s |
| Finite samples | Yes | Yes |
| Initial load time | 0.016188 s | 4.410327 s |

TidyVoice waveform SHA-256:
`4f0ed3b95b1f1bb6f339f5bcb3db4d6dc0da79120c9d27942e35598a342c5c5f`

ViMD waveform SHA-256:
`084f1e72b940643f5cf74f3d3aed09cb088bd49bedb114835ffeef5c8f0e96af`

Loading the same ViMD record twice caused one physical row-group read and
produced identical canonical samples, confirming cache reuse.

The load times are smoke-test diagnostics, not final latency benchmarks. The
ViMD result indicates that training should use shard-aware access and avoid
multiple large row-group caches per process.

Evidence artifact:
`results/data_audit/audio_pipeline_smoke.json`

## Advantages

- Removes sample-rate and channel-layout differences across datasets.
- Uses one preprocessing contract for all three model families.
- Avoids extracting or duplicating approximately 56 GiB of ViMD audio.
- Makes crop selection reproducible across local and Kaggle processes.

## Disadvantages and Limitations

- Arithmetic downmixing may cancel phase-inverted stereo content.
- Polyphase resampling adds CPU work for ViMD.
- A Parquet cache miss can be slow and memory-intensive.
- Repeating very short speech does not add new speaker information.
- Final crop duration and crop count still require Validation evidence.
