# Decision 002: ViMD Canonical Speaker-Disjoint Protocol

Status: Accepted  
Date: 2026-08-20

## Context

ViMD provides source Train, Validation, and Test partitions. Metadata auditing
found no Train/Validation or Train/Test speaker overlap, but two speaker IDs
occurred in both Validation and Test:

- `spk_73_0186`
- `spk_76_0219`

The overlapping Validation rows were:

- `spk_73_0186`: `73_0309.wav`
- `spk_76_0219`: `76_0295.wav`

Both speakers were singletons in source Validation. Their Test recordings remain
part of the official source Test partition.

## Alternatives Considered

### Keep the source partitions unchanged

Rejected because speaker overlap would leak identities between model selection
and final evaluation.

### Move the overlapping Validation rows into Test

Rejected because it would alter the source Test inventory and reduce
comparability with the documented dataset partition.

### Exclude the two contaminated Validation rows

Accepted because it preserves source Test unchanged, removes speaker leakage,
and does not reduce Validation genuine-pair capacity.

## Decision

Use the following mapping:

- Source Train becomes canonical Train.
- Source Validation becomes canonical Validation, except the two contaminated
  speaker rows are excluded.
- Source Test becomes canonical Test unchanged.

Gender is retained only as descriptive metadata. It is not used as a training
target or trusted stratification variable because the audit found inconsistent
gender labels, including conflicts for the two overlapping speakers.

## Canonical Result

| Split | Utterances | Speakers | Genuine pairs |
|---|---:|---:|---:|
| Train | 15,023 | 10,291 | 7,044 |
| Validation | 1,898 | 1,318 | 879 |
| Test | 2,026 | 1,344 | 1,046 |
| Total | 18,947 | — | 8,969 |

Source Validation and canonical Validation both contain 879 genuine pairs.
Therefore, excluding the two singleton rows causes no verification-pair loss.

## Reproducibility Evidence

- Dataset: `dullahn/vimd-dataset`
- Canonical manifest SHA-256:
  `ed7b764c6aaab2ba2c4ec95edadab19fd640ebca72aa06da3d36cbf93fc4747f`
- Evidence artifact:
  `results/data_audit/vimd_protocol_summary.json`
- Validator:
  `scripts/validate_vimd_protocol.py`

Reproduction command:

```bash
python scripts/validate_vimd_protocol.py \
  --dataset-root /kaggle/input/datasets/dullahn/vimd-dataset \
  --output results/data_audit/vimd_protocol_summary.json \
  --batch-size 4096