# Decision 005: Shared Verification Trial Protocol

Status: Accepted  
Date: 2026-08-22

## Context

ECAPA-TDNN, RawNet3, and WavLM+MHFA must be evaluated on identical,
model-independent verification evidence. TidyVoice has many utterances per
speaker, while ViMD contains many singleton speakers and a small number of
higher-resource speakers. Enumerating every possible pair would let
high-resource identities dominate genuine trials and would make the protocol
unnecessarily large.

The required reporting includes TAR at FAR 0.01%. The trial protocol therefore
needs enough impostor trials to measure this operating region rather than
reporting a target below the empirical resolution.

## Alternatives Considered

### Enumerate every genuine and impostor pair

Rejected because the number of pairs is unnecessarily large and speakers with
many utterances contribute disproportionately many genuine pairs.

### Sample utterance pairs uniformly

Rejected for impostor construction because speakers with more utterances would
have a greater probability of appearing in the protocol.

### Use fewer than 10,000 impostor trials

Rejected because 10,000 impostors provide only 0.01 percentage-point FAR
resolution. The required FAR 0.01% would then correspond to a single false
accept and would be too coarse for meaningful comparison.

### Cap genuine pairs and sample speaker identities uniformly

Accepted because it limits high-resource-speaker influence, keeps scoring
tractable, and provides the required low-FAR resolution.

## Decision

Use the following settings independently for each dataset and canonical split:

- Splits: Validation and Test
- Seed: 42
- Maximum genuine pairs per speaker: 20
- Unique impostor trials: 100,000
- Genuine sampling: unique within-speaker pairs, capped per speaker
- Impostor sampling: choose two distinct speaker identities uniformly, then
  choose one utterance uniformly from each identity
- Pair order: canonical utterance-ID order
- Trial order: genuine trials first, then impostor trials, sorted by stable ID
- Fingerprint: SHA-256 of the complete ordered trial representation

Validation trials are used for threshold selection. Test trials are used only
for final evaluation with the Validation-selected threshold frozen. All three
models reuse the same trial fingerprints.

## Real-Data Result

| Dataset | Split | Manifest utterances | Speakers | Genuine | Impostor | Total |
|---|---|---:|---:|---:|---:|---:|
| TidyVoice | Validation | 29,720 | 404 | 7,954 | 100,000 | 107,954 |
| TidyVoice | Test | 29,723 | 404 | 7,898 | 100,000 | 107,898 |
| ViMD | Validation | 1,898 | 1,318 | 863 | 100,000 | 100,863 |
| ViMD | Test | 2,026 | 1,344 | 1,042 | 100,000 | 101,042 |

The TidyVoice trial lists reference 25,895 Validation utterances and 25,901
Test utterances. Both ViMD lists reference every canonical utterance in their
respective split.

ViMD has theoretical genuine-pair capacities of 879 Validation pairs and 1,046
Test pairs. The accepted cap retains 863 and 1,042 respectively. The excluded
pairs come only from speakers exceeding the 20-pair cap; singleton speakers
cannot contribute genuine pairs but remain eligible for impostor trials.

## FAR Resolution

Each protocol contains 100,000 impostor trials:

```text
minimum non-zero FAR = 1 / 100,000
                     = 0.00001 fraction
                     = 0.001 percentage points
```

The required FAR 0.01% therefore corresponds to ten false accepts. Every
TAR@FAR result must still record its achieved empirical FAR and threshold.

## Trial Fingerprints

| Dataset | Split | Trial-list SHA-256 |
|---|---|---|
| TidyVoice | Validation | `0fe5a2b24c05dc89d9b49dec75b945844863987a9c51e426bc139c6ae2e5e9be` |
| TidyVoice | Test | `6a1743b1aa982505e5639a3c324e2c8d840e16d4a3d098a06b6b1e38265d45b3` |
| ViMD | Validation | `4bac0087282dd94052becc73a7a37b86d13c1995318b8737b4ee80c15c4cb096` |
| ViMD | Test | `ae5da0dac25c185ffa3930c32fff5802aee6113d8e225a54d4677a70ef1a222f` |

Evidence artifact:
`results/data_audit/verification_trial_protocols.json`

Artifact SHA-256:
`89f71c9354a6fd3760284348d9d3277a7acb68175f274e4ae07288b35eb785e5`

Generator:
`scripts/prepare_verification_protocols.py`

## Advantages

- Identical deterministic trials support direct model comparison.
- Speaker-first impostor sampling reduces utterance-count bias.
- Genuine caps prevent high-resource speakers from dominating.
- One hundred thousand impostors support the required FAR 0.01% analysis.
- Compact settings and fingerprints avoid committing approximately 418,000
  verbose trial records while preserving reproducibility.

## Disadvantages and Limitations

- Sampled trials do not represent every possible utterance pair.
- TidyVoice trial sampling does not reference every manifest utterance.
- Trials share speakers and utterances, so trial outcomes are correlated; the
  nominal pair count must not be treated as the number of independent samples.
- Very-low-FAR confidence intervals may still be wide despite adequate
  empirical resolution.
- Changing any manifest, seed, cap, or sampling implementation changes the
  fingerprint and creates a different protocol.
