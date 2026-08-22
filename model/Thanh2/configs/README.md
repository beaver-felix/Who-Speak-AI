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

The model files currently record verified provenance, architecture facts, and
reference recipes only. A reference recipe is not automatically an accepted
training recipe. Active crop duration, batch size, optimizer, learning rate,
freezing policy, augmentation, scheduler, and epoch budget will be added only
after model adapters and Kaggle memory/mini-run validation provide evidence.

## Advantages

- Uses Python's standard-library TOML parser; no configuration dependency can
  disturb Kaggle's pinned environment.
- Makes shared controls and justified model differences explicit.
- Produces a complete artifact suitable for W&B and checkpoint provenance.
- Unknown command-line overrides fail instead of being silently ignored.

## Disadvantages

- TOML has no native null value, so unresolved settings are represented by
  explicit status fields rather than placeholder nulls.
- Adding a new experiment-only key requires placing it in a reviewed layer
  before it can be overridden.
