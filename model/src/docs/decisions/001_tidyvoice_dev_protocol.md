# Decision 001: TidyVoice Dev Validation/Test Protocol

Status: Accepted  
Date: 2026-08-19

## Context

TidyVoice provides source-level Train and Dev partitions with no speaker
overlap. The project requires separate validation and test partitions, so
source Dev must be divided without speaker leakage.

Audited source Dev metadata:

- 808 speakers
- 59,443 utterances
- 39 languages
- Every speaker has at least four utterances

The split uses only path-derived speaker and language metadata. No waveform
content, model output, or test performance is used during optimization.

## Rejected Baseline

A seeded random 404/404 speaker split produced:

| Measure | Random baseline |
|---|---:|
| Validation speakers | 404 |
| Test speakers | 404 |
| Utterance imbalance | 11.377% |
| Maximum language-proportion difference | 4.538 percentage points |

The speaker groups were disjoint, but the metadata distributions were not
sufficiently comparable.

## Decision

Use deterministic metadata-balanced group optimization with:

- Seed: 42
- Validation fraction: 0.5
- Greedy restarts: 64
- Pairwise swap passes: 8
- Fixed speaker allocation: 404 validation / 404 test
- Objective: utterance imbalance plus maximum language-proportion difference

The greedy multi-start stage searches multiple seeded speaker orderings.
Pairwise swaps then improve the best candidate without changing speaker counts
or allowing speaker leakage.

## Result

| Measure | Random baseline | Selected protocol |
|---|---:|---:|
| Validation speakers | 404 | 404 |
| Test speakers | 404 | 404 |
| Validation utterances | 26,340 | 29,720 |
| Test utterances | 33,103 | 29,723 |
| Utterance imbalance | 11.377% | 0.005047% |
| Maximum language-proportion difference | 4.538 pp | 2.230175 pp |
| Optimization objective | Not recorded | 0.02235222 |

The selected protocol nearly eliminates utterance-count imbalance and reduces
the maximum language-distribution difference by approximately 50.9%.

## Reproducibility Evidence

- Dataset: `dullahn/mozzila-tidyvoice`
- Source partition: `dev`
- Speaker-profile SHA-256:
  `9e5c0b2502f732307a605e6ccdc7dda763cd1903badfea95ef7922bcb4800b9a`
- Assignment artifact:
  `results/data_audit/tidyvoice_dev_protocol.json`
- Generator:
  `scripts/prepare_tidyvoice_protocol.py`

Reproduction command:

```bash
python scripts/prepare_tidyvoice_protocol.py \
  --dataset-root /kaggle/input/datasets/dullahn/mozzila-tidyvoice/TidyVoiceX_ASV \
  --output results/data_audit/tidyvoice_dev_protocol.json \
  --seed 42 \
  --validation-fraction 0.5 \
  --restarts 64 \
  --max-swap-passes 8