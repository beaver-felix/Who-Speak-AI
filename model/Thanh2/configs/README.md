# Layered Experiment Configuration

Configuration resolves in this order:

```text
base.toml
    + datasets/<dataset>.toml
    + models/<model>.toml
    + stages/<pilot-or-full>.toml
    + explicit existing-key overrides
```

The resolver rejects missing required fields, table/scalar structure conflicts,
unknown override paths, and changes to accepted comparison invariants. It saves
the complete resolved configuration and a SHA-256 fingerprint for experiment
tracking.

Example:

```bash
python scripts/resolve_config.py \
  --layer configs/base.toml \
  --layer configs/datasets/vimd.toml \
  --layer configs/models/rawnet3.toml \
  --layer configs/stages/pilot.toml \
  --output results/configs/vimd_rawnet3_pilot_s42.json
```

Prepare the complete six-run matrix with one command:

```bash
python scripts/prepare_experiment_configs.py \
  --stage pilot \
  --output-dir results/configs/pilot
```

The shared AAM-Softmax control and the architecture-specific optimizer groups
are explicit. Their values are initial, source-informed screening candidates,
not final selected recipes. Batch sizes `24`, `24`, and `6` for
ECAPA-TDNN, RawNet3, and WavLM+MHFA respectively passed both T4 memory
calibration and a three-step real multi-batch gate.

The pilot stage is a runtime and evidence gate, not a quality experiment. It
uses one deterministic utterance from each of 512 deterministic Train speakers,
one epoch, one Validation crop, offline W&B, and checkpointing every 100 steps.
The full stage uses every Train speaker, up to four rotating utterances per
speaker per epoch, at most 15 epochs, patience 3, two Validation crops, online
W&B, and checkpointing every 500 steps. Both stages use seed 42, deterministic
CUDA operations, a constant learning rate, FP16, and Validation after every
completed epoch.

The constant schedule is a controlled initial hypothesis. It isolates the
architecture-specific optimizer settings and avoids adding an unevaluated
scheduler difference. It may be replaced only by a predeclared Validation
experiment, with the decision and evidence recorded before Test evaluation.

## Advantages

- Uses Python's standard-library TOML parser; no configuration dependency can
  disturb Kaggle's pinned environment.
- Makes shared controls and justified model differences explicit.
- Produces a complete artifact suitable for W&B and checkpoint provenance.
- Keeps the pilot bounded while exercising real training and full Validation.
- Rotates high-resource-speaker utterances without global RNG dependence.
- Keeps ViMD Parquet shards contiguous enough to use the row-group cache.
- Unknown command-line overrides fail instead of being silently ignored.
- Candidate status is stored beside every optimizer policy, preventing a source
  recipe or plausible hypothesis from being mislabeled as an empirical result.

## Disadvantages

- TOML has no native null value, so unresolved settings are represented by
  explicit status fields rather than placeholder nulls.
- Adding a new experiment-only key requires placing it in a reviewed layer
  before it can be overridden.
- The initial learning rates can be suboptimal on TidyVoice or ViMD; validation
  experiments are intentionally required before they become accepted settings.
- The four-utterance full-stage cap does not expose every high-resource
  TidyVoice utterance in one epoch; exposure rotates deterministically.
- Full Validation remains expensive, particularly for WavLM+MHFA, because the
  immutable trial protocol references many unique utterances.
