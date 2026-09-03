# Decision 015: Resource-Constrained Parallel Execution

## Status

Accepted on 2026-08-22 before any successful dataset-level training run.

## Constraint

Only approximately eight hours of wall-clock time remain. The estimated full
15-epoch matrix cannot be completed within that budget. Three team members can
independently use Kaggle GPU T4 x2 sessions.

## Decision

The six required model-dataset combinations are divided by architecture:

| Worker | Architecture | GPU 0 | GPU 1 |
|---|---|---|---|
| 1 | ECAPA-TDNN | TidyVoice | ViMD |
| 2 | RawNet3 | TidyVoice | ViMD |
| 3 | WavLM + MHFA | TidyVoice | ViMD |

Each person uses their own Kaggle account and runs one model-specific notebook.
No account, token, or credential is shared. Each dataset is trained in an
isolated process, run directory, cache, and CUDA device.

The predeclared resource-constrained stage uses:

- every Train speaker;
- one deterministic rotating utterance per speaker per epoch;
- at most three epochs;
- Validation-only early stopping with patience one;
- one deterministic crop per evaluation utterance;
- the previously accepted architecture-specific T4 batch sizes;
- offline W&B plus local JSONL evidence;
- full immutable Validation trial evaluation after every epoch;
- one final Test evaluation using the best Validation-selected checkpoint;
- a security threshold selected on the best Validation epoch at FAR 0.1%,
  frozen, then applied to Test without Test-driven tuning.

The notebooks run a complete evidence validator and create one ZIP per
architecture. The archives preserve resolved configurations, logs, Validation,
final Test, checkpoint sidecars and binaries, run summaries, and tracking data.
Reproducible model caches are removed before archiving.

## Performance Correction

TidyVoice was previously audited as a WAV-only read-only hierarchy. Calling
`Path.is_file()` and `Path.resolve()` for each of 321,711 utterances created
hundreds of thousands of redundant metadata operations on the Kaggle mount.
The canonical scanner now resolves roots once, validates speaker/language
directories, checks every filename extension, and constructs discovered
records lexically without per-audio filesystem stats. The public defensive
single-path parser remains unchanged.

## Interpretation

These runs are a compute-constrained controlled comparison, not an optimal
configuration search and not equivalent to the original 15-epoch full plan.
The report must disclose this limitation. Results remain useful because all
models retain the same split, preprocessing, objective, trials, metric code,
epoch membership policy, and selection boundary.

## Advantages

- Completes all six required combinations concurrently within the remaining
  wall-clock budget when runtime permits.
- Preserves speaker coverage and shared comparison controls.
- Prevents Test leakage and creates auditable downloadable evidence.
- Requires teammates only to attach the two datasets and choose **Run All**.

## Disadvantages

- Three epochs and one utterance per speaker per epoch may underfit.
- A single seed cannot estimate between-seed variance.
- Patience one can stop on noisy Validation changes.
- Independent Kaggle sessions may differ slightly in infrastructure timing.
- The resulting rankings must not be described as architecture-optimal.
