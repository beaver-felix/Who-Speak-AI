# Layered Experiment Configuration

Configuration resolves in this order:

```text
base.toml
    + datasets/<dataset>.toml
    + models/<model>.toml
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
  --set experiment.stage=smoke \
  --output results/configs/vimd_rawnet3_smoke_s42.json
```

The shared AAM-Softmax control and the architecture-specific optimizer groups
are now explicit. Their values are initial, source-informed screening
candidates—not final selected recipes. Batch size remains unset until Kaggle
memory calibration and a multi-batch mini-run; scheduler, augmentation, and
epoch budget remain unset until validation evidence exists.

## Advantages

- Uses Python's standard-library TOML parser; no configuration dependency can
  disturb Kaggle's pinned environment.
- Makes shared controls and justified model differences explicit.
- Produces a complete artifact suitable for W&B and checkpoint provenance.
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
