# Decision 004: Shared Speaker-Verification Metrics

Status: Accepted
Date: 2026-08-20

## Context

All six model/dataset experiments require directly comparable verification
metrics. Inconsistent threshold semantics, score-tie handling, or Test-derived
thresholds would invalidate the comparison even if the embeddings were sound.

## Decision

Use one NumPy-based implementation for every model and dataset.

- Genuine trials use label `1`; impostor trials use label `0`.
- Larger scores indicate greater speaker similarity.
- A trial is accepted when `score >= threshold`.
- All equal scores enter the ROC curve atomically.
- EER uses linear interpolation around the FAR/FRR crossing.
- FAR, FRR, TAR, and accuracy use a decision threshold selected on Validation
  and locked before Test evaluation.
- Test EER and minDCF remain threshold-independent descriptive metrics; they do
  not set the deployed operating threshold.
- minDCF uses `P_target=0.01`, `C_miss=1`, and `C_false_alarm=1` unless an
  experiment explicitly documents a different application prior.
- minDCF is normalized by the lower cost of the always-reject and always-accept
  trivial systems.
- TAR is reported at FAR targets 5%, 1%, 0.1%, and 0.01%.
- Each TAR@FAR result also records its achieved empirical FAR and threshold.

## Threshold Flow

```text
Validation scores
    -> select operating threshold
    -> freeze threshold
    -> compute Test FAR, FRR, TAR, and accuracy

Test scores
    -> EER/minDCF/TAR@FAR analysis only
    -> never retune the deployed threshold
```

## FAR Resolution

With `N` impostor trials, the smallest non-zero empirical FAR is `1/N`. A FAR
target below this resolution can only select a zero-false-accept operating
point. Reporting achieved FAR alongside TAR prevents a requested 0.01% target
from being mistaken for a precisely observed 0.01% operating rate.

## Advantages

- All models share identical definitions and tie behavior.
- Validation/Test threshold leakage is structurally avoidable.
- Metrics are independent of trial input order.
- Flat scalar output maps directly to JSON and W&B.
- No scikit-learn version behavior is hidden in the metric implementation.

## Disadvantages and Limitations

- Interpolated EER may not correspond to a realizable discrete threshold.
- Accuracy depends strongly on the genuine/impostor trial ratio.
- Very low FAR estimates require many independent impostor trials.
- Empirical metrics do not by themselves provide confidence intervals.

## Verification

The regression suite covers perfect separation, known confusion counts,
all-tied scores, constrained TAR selection, input-order invariance, required
metric names, and invalid trial arrays. The complete suite contains 62 passing
tests.
