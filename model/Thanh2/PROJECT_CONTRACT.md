# Speaker Recognition Project Contract

## 1. Purpose

This document is the concise source of truth for work in `model/Thanh2`. Read it before proposing or implementing project changes.

The objective is to build a reproducible, end-to-end Kaggle pipeline that trains, evaluates, compares, and exports three speaker-recognition models on two datasets. The selected model will later support speaker verification (SV) and speaker identification (SID) in a secure virtual assistant.

## 2. Authoritative Sources

Consult these files when details are needed:

1. `Note/context.md`: workspace, workflow, and hardware constraints.
2. `Note/train_evaluate_3_models_requirements.md`: models, datasets, shared configuration, metrics, and W&B requirements.
3. `Note/Secure_Virtual_Assistant_with_Speaker_Recognition.md`: course requirements and submission criteria.
4. `Note/our_app.md`: application use cases, enrollment, SID personalization, and SV-protected diary access.
5. `Note/bao_cao_tien_do_speaker_recognition_2026-08-10.md`: legacy implementation and known limitations; reference only.

If documents conflict, explicit current user instructions take precedence. The active target is `Who-Speak-AI/model/Thanh2`, even though one legacy line in `context.md` names `model/Thanh`.

## 3. Working Rules

- Use Git branch `thanhDT`.
- Create or edit project artifacts only inside `Who-Speak-AI/model/Thanh2` unless the user explicitly expands the scope.
- Work in small, reviewable batches; when the user must perform actions, up to
  three closely related tasks may be grouped to improve speed.
- The assistant may create, modify, test, commit, and push project files
  without requesting permission each time. The user remains responsible for
  Kaggle execution, downloading Kaggle outputs, and other actions unavailable
  to the assistant.
- Never invent dataset properties, API behavior, experimental results, citations, or model capabilities.
- Keep code, configuration, documentation, reports, and other deliverables in English.
- Document why every important decision was made, not only what was done.
- Do not commit datasets, checkpoints, or other large binary artifacts to Git.
- Update this contract immediately after the user approves a material change to project scope, methodology, models, datasets, metrics, infrastructure, constraints, or workflow.
- Record only approved decisions as project rules. Keep proposals and unresolved alternatives clearly labeled until the user approves them.

## 4. Required Models and Datasets

### Models

1. ECAPA-TDNN: `speechbrain/spkrec-ecapa-voxceleb`
2. RawNet3: `Jungjee/RawNet3`
3. WavLM + MHFA: `theolepage/wavlm_ssl_sv`

Use representative implementations of the named architectures. Do not label a simplified approximation as the official architecture. Verify implementation and checkpoint details against primary sources before integration.

### Datasets

1. TidyVoice: Kaggle dataset `dullahn/mozzila-tidyvoice`
2. ViMD: Kaggle dataset `dullahn/vimd-dataset`

The intended benchmark contains six primary model-dataset combinations:

| Dataset | ECAPA-TDNN | RawNet3 | WavLM + MHFA |
|---|---:|---:|---:|
| TidyVoice | Required | Required | Required |
| ViMD | Required | Required | Required |

Do not assume the two datasets have identical schemas. Normalize them through one canonical manifest interface while retaining dataset provenance.

### Observed Kaggle mounts

Dataset mounts verified by the user on 2026-08-18:

- Publisher root: `/kaggle/input/datasets/dullahn`.
- TidyVoice mount: `/kaggle/input/datasets/dullahn/mozzila-tidyvoice`.
- TidyVoice content root: `TidyVoiceX_ASV`, containing explicit `TidyVoiceX_Train` and `TidyVoiceX_Dev` directory branches.
- ViMD mount: `/kaggle/input/datasets/dullahn/vimd-dataset`.
- ViMD contains a README and a `data` directory with 130 Parquet shards.
- TidyVoice contains 321,711 WAV files and no manifest-like files: 262,268 files in its Train branch and 59,443 in its Dev branch.
- Observed TidyVoice paths follow the provisional pattern `<branch>/<branch>/<speaker_id>/<language>/<utterance>.wav`; this must be validated across every path before deriving manifest fields.
- ViMD contains 130 schema-consistent Parquet shards: 103 train, 13 validation, and 14 test.
- ViMD contains 15,023 train rows, 1,900 validation rows, and 2,026 test rows, for 18,949 total utterances.
- Observed ViMD storage is approximately 44.398 GiB train, 5.521 GiB validation, and 5.801 GiB test.
- Every ViMD split has the same fields: `region`, `province_code`, `province_name`, `filename`, `text`, `speakerID`, `gender`, and `audio`.
- ViMD `audio` is an Arrow struct containing embedded binary bytes and a path. Metadata analysis must select non-audio columns, and audio processing must stream individual examples or small batches rather than materializing shards.
- TidyVoice path validation passed for all 321,711 WAV files. Train contains 3,666 speakers across 40 language folders; Dev contains 808 speakers across 39 language folders. There is no speaker overlap between Train and Dev.
- Every observed TidyVoice speaker appears in multiple language folders. Language directory is therefore an utterance attribute, not a speaker identity component.
- TidyVoice is highly imbalanced: Train has 4 to 1,618 utterances per speaker (median 28), and Dev has 4 to 1,176 (median 25).
- ViMD supplied splits are almost speaker-disjoint, but two speaker IDs (`spk_73_0186` and `spk_76_0219`) occur in both validation and test. These overlapping identities must not remain in both final partitions.
- ViMD speaker coverage is sparse: Train has 10,291 speakers for 15,023 utterances, validation has 1,320 speakers for 1,900 utterances, and test has 1,344 speakers for 2,026 utterances. Median utterances per speaker is one in every split.
- No duplicate `(speakerID, filename)` keys or empty inspected metadata values were found in ViMD.
- ViMD region and province labels are consistent within speaker IDs, but 17 speaker IDs have more than one gender label. Gender must not be used as a trusted training target without resolving or excluding these inconsistencies.
- Every TidyVoice Train and Dev speaker has at least four utterances. Their theoretical unique genuine-pair capacities are 31,829,370 and 7,405,777 respectively.
- ViMD has many singleton speakers. Speakers with at least two utterances: 3,350 in Train, 397 in validation, and 456 in test. These speakers retain 53.80%, 51.42%, and 56.17% of the respective utterances.
- ViMD theoretical unique genuine-pair capacities are 7,044 in Train, 879 in validation, and 1,046 in test.
- Final ViMD verification trials must state the eligibility policy explicitly. Singleton identities may support classification training or impostor trials, but cannot produce a genuine enrollment/probe pair without splitting one recording, which is prohibited.
- A deterministic header sample found no errors in 256 TidyVoice Train files, 128 TidyVoice Dev files, or 20 files from each ViMD split.
- Every sampled TidyVoice file was mono, 16 kHz, WAV PCM16. Sampled median durations were 5.040 seconds in Train and 5.232 seconds in Dev.
- Every sampled ViMD file was WAV PCM16, but sample rates included 44.1 kHz and 48 kHz, and channel counts included mono and stereo. Sampled median durations were 21.634 seconds in Train, 21.478 seconds in validation, and 17.585 seconds in test.
- Canonical model input is mono 16 kHz. ViMD therefore requires deterministic channel downmixing and resampling before model-specific cropping; TidyVoice should pass through unchanged when already compliant.
- These waveform findings are sample-based, not a complete corruption or duration census. The final manifest builder must validate every consumed utterance lazily and record failures.
- A seed-42 random 50/50 speaker split of TidyVoice Dev produced 404 speakers per partition but 11.377% utterance imbalance and a maximum language-proportion difference of 4.538 percentage points. This candidate is rejected.
- The TidyVoice validation/test splitter must remain speaker-disjoint while explicitly balancing utterance counts and language distributions using metadata only. Its objective and deterministic tie-breaking must be documented and tested before the split is locked.

Kaggle currently groups these inputs under the publisher-based path `/kaggle/input/datasets/dullahn`, so counting only immediate children of `/kaggle/input` does not count individual datasets.

## 5. Scientific Comparison Contract

### Mandatory shared controls

The following must be shared within a dataset so results remain comparable:

- Dataset version and canonical manifests.
- Speaker-disjoint train, validation, and test partitions.
- Exact sample membership within each partition.
- Audio preprocessing and normalization, except documented architecture requirements.
- Verification enrollment/test protocol and genuine/impostor trial lists.
- Random seeds.
- Metric implementation and output schema.
- Test-set access policy.
- W&B logging fields.
- Resource and latency measurement procedure.

### Permitted model-specific configuration

Architecture-specific settings may differ when supported by primary sources, validation experiments, or measured hardware constraints:

- Optimizer, learning rate, scheduler, and weight decay.
- Batch size and gradient accumulation.
- Training duration and early stopping.
- Input duration required by the architecture.
- Embedding dimension and pooling.
- Freezing, unfreezing, or progressive fine-tuning policy.
- Loss type and loss-specific parameters.
- Compatible augmentation policy.

Shared settings are controls, not a requirement to use knowingly ineffective hyperparameters. Every model-specific override must be documented.

### Two result views

When feasible, report:

1. **Controlled baseline:** maximally shared training settings for architectural comparison.
2. **Optimized comparison:** the best validation-selected configuration for each model under a comparable tuning budget.

The optimized comparison measures complete training recipes, not architecture alone. State this limitation explicitly.

## 6. Experimental Methodology

Use the following decision chain:

```text
Question -> Hypothesis -> Evidence -> Experiment -> Result -> Decision
```

Evidence priority:

1. Official paper.
2. Official repository or model card.
3. Reproducible validation experiment.
4. Hardware measurement.
5. Clearly labeled engineering hypothesis.

Configuration selection protocol:

1. Define the hypothesis and selection metric before running the experiment.
2. Use training data to fit candidate configurations.
3. Use validation data only for tuning, early stopping, model selection, and threshold calibration.
4. Use one seed for inexpensive preliminary screening when appropriate.
5. Give competing models a comparable tuning budget.
6. Rerun selected configurations with multiple predefined seeds, preferably three.
7. Lock the configuration before final test evaluation.
8. Evaluate the test set only for final reporting; never tune on test results.
9. Report means, standard deviations, and confidence intervals when statistically supported.

Default configuration-selection target:

- Primary: mean validation EER across final seeds.
- Tie-breakers: validation minDCF, then TAR at low FAR.
- Deployment thresholds: calibrate on validation trials for the intended security operating point.

Any departure from this protocol requires a written reason.

## 7. Data and Trial Integrity

- Prevent speaker overlap across train, validation, and test splits.
- Prevent source-recording leakage across splits.
- Make split and trial generation deterministic and reproducible.
- Store speaker ID, recording ID, path, duration, sample rate, split, and dataset provenance in canonical manifests when available.
- Detect missing files, duplicates, corrupted audio, invalid duration, and unsupported sample rates before training.
- Build shared trial lists and reuse them across all models evaluated on the same dataset.
- Keep threshold-calibration trials separate from final test trials.
- Record the number of genuine and impostor trials.
- Report the empirical FAR resolution (`1 / number_of_impostor_trials`).
- Do not claim TAR@FAR 0.01% when the trial count cannot support that operating point. Mark it as statistically unsupported instead.
- Cache embeddings during evaluation so each unique utterance is encoded once per checkpoint.

## 8. Required Metrics and Reporting

At minimum, report:

- EER.
- FAR and FRR at a clearly named threshold.
- TAR@FAR 5%.
- TAR@FAR 1%.
- TAR@FAR 0.1%.
- TAR@FAR 0.01%, only when supported by sufficient impostor trials.
- minDCF, including its prior and cost parameters.
- Accuracy, including the threshold used.
- Inference latency, including device, batch size, warm-up, and measurement procedure.

Also record when practical:

- ROC-AUC and DET/ROC curve data.
- Thresholds associated with all operating metrics.
- Parameter counts, peak GPU memory, training time, and embedding dimension.
- Confidence intervals and multi-seed variability.

All reported values must come from saved structured artifacts, not copied approximately from console output.

## 9. Configuration and Logging

Use layered configuration:

```text
shared base configuration
    + dataset-specific configuration
    + model-specific configuration
    + explicit experiment override
```

The resolved configuration for every run must be saved and logged to W&B. Each run should record:

- Experiment ID and Git commit.
- Dataset and manifest version/hash.
- Model source and revision.
- Seed and full resolved configuration.
- Training and validation history.
- Final metrics and thresholds.
- Artifact locations.
- Hardware and software versions.

Recommended run IDs follow `{dataset}_{model}_{stage}_{seed}`; for example, `vimd_rawnet3_optimized_s42`.

## 10. Kaggle and Memory Constraints

- Target Kaggle GPU notebooks and an approximately 8 GB host-RAM-conscious pipeline.
- Load audio lazily and process it batch-wise; do not load an entire dataset into RAM.
- Use mixed precision (`fp16` or `bf16` when supported).
- Use gradient accumulation when the physical batch size is limited.
- Configure DataLoader workers and pinned memory based on measured Kaggle behavior.
- Make preprocessing resumable and avoid unnecessary dataset duplication.
- Save large checkpoints and datasets to Hugging Face Hub or Kaggle artifacts rather than Git.
- Provide clear smoke-test settings before full training.

### Observed Kaggle reference environment

Environment fingerprint supplied by the user on 2026-08-16:

- Python: `3.12.13`.
- Platform: Linux with glibc 2.35.
- GPUs: two Tesla T4 devices, each reporting 14.56 GiB VRAM and compute capability 7.5.
- PyTorch: `2.10.0+cu128`.
- torchaudio: `2.10.0+cu128`.
- CUDA build used by PyTorch: `12.8`.
- Transformers: `5.0.0`.
- W&B: `0.26.1`.
- PyYAML: `6.0.3`.
- NumPy: `2.0.2`.
- SciPy: `1.16.3`.
- scikit-learn: `1.6.1`.
- huggingface-hub: `1.11.0`.
- safetensors: `0.7.0`.
- SpeechBrain: `1.1.0`, installed and smoke-tested.
- Asteroid Filterbanks: `0.4.0`, installed and smoke-tested.
- Kaggle session configuration: `GPU T4 x2`, Internet enabled, no persistence, and environment pinned to the original notebook environment.

Treat these values as an observed reference, not permanent Kaggle guarantees. Capture and save the environment fingerprint with every final experiment. Do not replace Kaggle's PyTorch or torchaudio builds until model compatibility has been checked; an unnecessary reinstall can break CUDA compatibility.

Dependency compatibility was verified on `cuda:0` on 2026-08-16. An Asteroid `ParamSincFB` encoder and a reduced SpeechBrain ECAPA-TDNN both completed forward and backward passes with finite outputs and available input gradients. This is a dependency smoke test only; it is not evidence of pretrained-checkpoint compatibility or model quality.

The official pretrained ECAPA checkpoint was also verified on `cuda:0` on 2026-08-16:

- Model: `speechbrain/spkrec-ecapa-voxceleb`.
- Immutable revision: `0f99f2d0ebe89ac095bcc5903c4dd8f72b367286`.
- Embedding-model parameter count: `20,767,552`.
- Official 16 kHz speech sample duration: 3.261 seconds.
- Raw embedding shape: `(1, 1, 192)`.
- The extracted embedding was finite, its explicit L2-normalized norm was 1.0, and repeated inference produced cosine similarity 1.0.

This establishes checkpoint and inference compatibility only. It does not establish performance on TidyVoice or ViMD and must not be reported as an experimental benchmark result.

The official RawNet3 checkpoint artifact was verified on 2026-08-16:

- Model repository: `jungjee/RawNet3`.
- Immutable revision: `c89102eea20c3f96917c434de673c0ace0caddc0`.
- File: `model.pt`, 62.27 MiB locally.
- SHA-256: `1ab283bcdf776bfceceea18240e56a8756835b1911b04f9c44f347d47c09f90c`, matching the value published on Hugging Face.
- Safe `torch.load(..., weights_only=True)` result: a `model` state dictionary containing 234 tensor entries, no non-tensor entries, and 16,305,693 total stored tensor elements.
- Every stored tensor was finite.

This establishes checkpoint provenance, integrity, and safe deserialization only. Strict architecture loading and inference compatibility remain to be tested.

RawNet3 strict architecture and inference compatibility were subsequently verified on `cuda:0` on 2026-08-16:

- Official architecture source: `https://github.com/clovaai/voxceleb_trainer.git`.
- Immutable trainer revision: `f51bab870672a9b0b50fa158b4e30f329e7866d7`.
- Configuration: ECA encoder, 256-dimensional output, Sinc stride 10, and the official `MainModel` defaults.
- Trainable parameter count: `16,280,322`.
- Model and checkpoint both contained 234 state entries; strict loading produced no missing or unexpected keys.
- A real 16 kHz, 3.261-second speech utterance produced a finite embedding with shape `(1, 256)`.
- Explicit L2 normalization produced norm 1.0, and repeated inference produced cosine similarity 1.0.

This establishes pretrained RawNet3 checkpoint and inference compatibility. It does not establish TidyVoice or ViMD performance. The official trainer was cloned temporarily for the audit; the final project will use a documented, attributed local adapter rather than depending on the full trainer repository.

The required WavLM+MHFA repository was audited without executing its code on 2026-08-16:

- Repository: `theolepage/wavlm_ssl_sv`.
- Immutable revision: `bfb8527de83b5347fb81b1e9e31be241656ca103`.
- Repository contents: 23 files totaling approximately 860.27 KiB of known file sizes.
- Required source and configuration files are present, including `models/Baseline/Spk_Encoder.py`, its bundled WavLM implementation, and `configs/wavlm_mhfa_dlg_lc.yaml`.
- No `.pt`, `.pth`, `.ckpt`, `.bin`, or `.safetensors` weight file is packaged in the repository.
- The official base configuration selects `Baseline.Spk_Encoder`, a 256-dimensional output, WavLM-Base+ initialization, AAM-Softmax with margin 0.2 and scale 30, 300 training frames, 400 evaluation frames, batch size 120, transformer LR `2e-5`, MHFA LR `5e-3`, and 15 epochs.
- The README links WavLM-Base+, DINO embeddings, and the paper's best fine-tuned checkpoint as external artifacts. The repository also states that DINO training code is not provided.

Therefore, `theolepage/wavlm_ssl_sv` must be treated as a pinned architecture and recipe source, not as a self-contained pretrained model package. WavLM-Base+ and any fine-tuned MHFA weights require independent provenance and integrity checks. The audit output ended with an assertion caused by an incorrect test expectation (`Baseline` instead of the verified `Baseline.Spk_Encoder`); the artifact findings themselves passed and do not require a rerun.

Microsoft WavLM-Base+ initialization was verified on 2026-08-16:

- The Azure Storage link currently returned HTTP 403, so the artifact was downloaded from the official Google Drive mirror linked by Microsoft's WavLM README.
- Local artifact size: 360.11 MiB.
- Observed SHA-256: `fcbcf2a94def92e90e086bb0727275d53b75a9c0e483e2abfa560ac951986b6d`.
- PyTorch reported zero unsafe pickle globals, and restricted `torch.load(..., weights_only=True)` succeeded.
- Top-level checkpoint keys: `cfg` and `model`.
- Configuration: 12 transformer layers, 768-dimensional hidden representation, 12 attention heads, 3072-dimensional feed-forward layers, and no input waveform normalization.
- Model state: 248 tensor entries, no non-tensor entries, and 94,381,936 total tensor elements.

Microsoft's README does not publish a checksum, so the observed SHA-256 is our reproducibility fingerprint rather than an independently published integrity value. Strict loading into the pinned WavLM+MHFA architecture and inference compatibility remain to be tested.

WavLM+MHFA architecture compatibility was subsequently verified on `cuda:0` on 2026-08-17:

- Pinned architecture source revision: `bfb8527de83b5347fb81b1e9e31be241656ca103`.
- Required WavLM, MHFA, and supporting source modules imported successfully.
- All 248 WavLM-Base+ checkpoint entries loaded strictly with no missing or unexpected keys.
- WavLM parameter count: `94,381,936`.
- Randomly initialized 64-head MHFA parameter count: `2,302,554`.
- Combined parameter count: `96,684,490`.
- The 3.261-second real-speech input produced 13 representation levels with stacked shape `(1, 768, 162, 13)` and a finite output embedding with shape `(1, 256)`.
- Explicit L2 normalization produced norm 1.0, and repeated inference produced cosine similarity approximately 1.0.
- PyTorch emitted a non-blocking deprecation warning for the upstream legacy `torch.nn.utils.weight_norm` API.

This establishes base-checkpoint and architecture inference compatibility only. The MHFA backend in this smoke test was randomly initialized, so its output is not evidence of speaker-recognition quality. The fine-tuned external MHFA checkpoint or our supervised fine-tuning must supply trained backend weights.

## 11. Approved Architecture

Use a shared `src`-layout Python package and introduce components incrementally:

```text
model/Thanh2/
|-- configs/
|   |-- datasets/
|   |-- models/
|   `-- experiments/
|       |-- controlled/
|       `-- optimized/
|-- src/speaker_recognition/
|   |-- data/
|   |-- models/
|   |-- losses/
|   |-- training/
|   |-- evaluation/
|   |-- inference/
|   `-- utils/
|-- scripts/
|-- notebooks/
|-- tests/
|-- docs/decisions/
|-- results/
`-- artifacts/
```

Architecture rules:

- Dataset normalization, splitting, validation, and trial generation belong to the shared data layer, not model folders.
- Model adapters expose a common embedding-oriented interface while retaining justified architecture-specific behavior.
- Training, evaluation, metrics, calibration, W&B tracking, and latency measurement are shared services.
- CLI scripts are thin entry points and contain no core business logic.
- Kaggle notebooks perform environment setup and orchestrate package commands; they do not duplicate the pipeline.
- Maintain `notebooks/01_dataset_audit_eda.ipynb` as a reproducible Kaggle-compatible record of dataset discovery, schema validation, integrity checks, split analysis, audio-header analysis, EDA visualizations, findings, rejected alternatives, and final data decisions.
- The dataset-audit notebook should call reusable functions from `src` once those functions exist. Notebook-only exploratory cells are allowed when clearly labeled and subsequently promoted to tested package code if they become part of the pipeline.
- Keep useful, reasonably sized notebook outputs for seminar evidence, but save authoritative summary tables and figures under `results/data_audit/`. Do not embed raw audio, datasets, checkpoints, or other large artifacts in Git.
- Accompany the notebook with a concise English data-audit report suitable for the team and teacher; the report must distinguish observed facts, sample-based findings, assumptions, and final decisions.
- `results/` contains small reviewed report artifacts suitable for Git.
- `artifacts/` contains generated checkpoints, cached embeddings, and other heavy files excluded from Git.
- Use layered configuration: shared base, dataset, model, then explicit experiment override.
- Save the fully resolved configuration for every experiment.
- Do not create empty modules only to match the target tree. Add each file when its responsibility is implemented or tested.

## 12. Production Relevance

The final speaker model must expose a consistent embedding interface for:

- Enrollment from three short recordings and aggregation of normalized embeddings.
- One-to-one speaker verification for protected diary operations.
- One-to-many speaker identification for personalized responses.
- Guest fallback for unknown speakers.
- Validation-calibrated rejection thresholds.

Provide an export path such as ONNX, TorchScript, or quantized PyTorch when the selected architecture supports reliable export. Compare exported outputs against the PyTorch reference before deployment.

## 13. Documentation Record for Each Decision

For every material experiment or configuration change, record:

- Decision question.
- Hypothesis.
- Source or rationale.
- Alternatives considered.
- Controlled variables.
- Changed variables.
- W&B run IDs.
- Validation results and resource measurements.
- Chosen option and justification.
- Limitations and follow-up work.

Never present preliminary, synthetic, or smoke-test results as final model results.

## 14. Known Legacy Problems to Avoid

- The previous RawNet implementation was a project-specific approximation, not official RawNet3.
- The previous WavLM experiment trained only a classifier while evaluating unchanged pretrained embeddings.
- Previous operating thresholds were derived from test scores.
- Previous evaluation repeatedly encoded the same utterances.
- Previous trials contained correlated chunks from limited source recordings.
- Previous reporting omitted verified experimental TAR@FAR values from the main result table.

The new pipeline must address these limitations or document why a limitation remains.

## 15. Current Status

- Git branch verified: `thanhDT`.
- Working tree restored to a clean baseline before starting `Thanh2`.
- `model/Thanh2` was empty at project initialization.
- Scientific methodology contract approved by the user.
- Shared `src`-layout architecture approved by the user.
- Kaggle dependencies installed: SpeechBrain `1.1.0` and Asteroid Filterbanks `0.4.0`.
- GPU dependency smoke test passed for both packages on a Tesla T4.
- Pinned pretrained ECAPA checkpoint inference smoke test passed on real speech.
- Pinned RawNet3 checkpoint integrity and state-dictionary inspection passed.
- Pinned RawNet3 checkpoint strictly loaded into the official architecture and passed real-speech GPU inference.
- Pinned WavLM+MHFA source repository and external-artifact requirements audited.
- Official WavLM-Base+ initialization artifact downloaded, fingerprinted, and safely inspected.
- Pinned WavLM-Base+ strictly loaded into the required WavLM+MHFA architecture and passed real-speech GPU inference.
- Initial feasibility and compatibility gates have passed for all three required model families.
- The canonical manifest schema, integrity validation, deterministic
  TidyVoice scanner/splitter, and lazy ViMD metadata pipeline are implemented
  and covered by local unit tests.

## Accepted TidyVoice Dev Protocol — 2026-08-19

- Source Dev is partitioned into 404 validation and 404 test speakers.
- Speaker overlap is prohibited.
- Selected configuration: seed 42, 64 greedy restarts, and 8 swap passes.
- Validation/test utterances: 29,720 / 29,723.
- Utterance imbalance: 0.005047%.
- Maximum language-proportion difference: 2.230175 percentage points.
- Objective value: 0.02235222.
- Speaker-profile SHA-256:
  `9e5c0b2502f732307a605e6ccdc7dda763cd1903badfea95ef7922bcb4800b9a`.
- Assignment artifact:
  `results/data_audit/tidyvoice_dev_protocol.json`.
- Methodology decision:
  `docs/decisions/001_tidyvoice_dev_protocol.md`.
- This exact assignment must be shared by ECAPA-TDNN, RawNet3, and
  WavLM+MHFA experiments.

## Accepted ViMD Canonical Protocol — 2026-08-20

- Source Train maps to canonical Train unchanged.
- Source Test maps to canonical Test unchanged.
- Source Validation maps to canonical Validation after excluding
  `spk_73_0186` and `spk_76_0219`.
- Canonical counts are 15,023 Train, 1,898 Validation, and 2,026 Test
  utterances.
- Canonical speaker counts are 10,291 Train, 1,318 Validation, and 1,344 Test.
- Genuine-pair capacities are 7,044 Train, 879 Validation, and 1,046 Test.
- The exclusions remove speaker leakage without reducing genuine-pair capacity.
- Canonical manifest SHA-256:
  `ed7b764c6aaab2ba2c4ec95edadab19fd640ebca72aa06da3d36cbf93fc4747f`.
- Evidence artifact:
  `results/data_audit/vimd_protocol_summary.json`.
- Methodology decision:
  `docs/decisions/002_vimd_canonical_protocol.md`.
- This protocol must be shared by all three model architectures.

## Accepted Canonical Audio Preprocessing — 2026-08-20

- All three model families receive mono `float32` waveform audio at 16 kHz.
- Multichannel sources are downmixed using the arithmetic channel mean.
- Non-16-kHz sources are converted with polyphase anti-aliasing resampling.
- Resampled signals are adjusted to the nearest duration-preserving sample
  count to prevent source-rate-dependent one-sample inconsistencies.
- Natural waveform amplitude is preserved; shared preprocessing does not apply
  peak, RMS, or loudness normalization.
- Empty, non-finite, malformed, or invalid-rate signals fail before modeling.
- Source sample rate and channel count remain available for audit reporting.
- ViMD audio remains embedded in Parquet; a worker-local one-row-group cache
  prevents repeated physical reads for nearby records without extracting the
  approximately 56 GiB source dataset.
- TidyVoice standalone WAV files and ViMD embedded WAV bytes converge on the
  same canonical waveform representation.
- NumPy `2.0.x`, SciPy `1.16.x`, and SoundFile `0.13.x` are bounded to the
  versions validated against the Kaggle runtime.
- Canonical audio behavior, embedded decoding, and row-group cache reuse are
  covered by the 43-test regression suite.

## Accepted Temporal Segmentation Invariants — 2026-08-20

- Training uses fixed-length pseudo-random crops derived from SHA-256 of the
  global seed, epoch, and canonical utterance ID; Python process hashes are
  prohibited because they are not stable across runs.
- Short recordings repeat their speech samples to reach the requested length;
  shared preprocessing does not introduce dataset-specific silence padding.
- Evaluation crops are deterministic, evenly span the available timeline, and
  include both temporal endpoints.
- A short evaluation recording produces one repeated crop rather than several
  identical crops.
- Segment duration and evaluation crop count remain explicit configuration
  variables to be selected using Validation evidence only.
- Segmentation and the complete data foundation are covered by 51 passing
  regression tests.

## Real-Dataset Audio Evidence — 2026-08-20

- TidyVoice real-audio loading passed for mono 16 kHz input and produced a
  finite 122,496-sample canonical waveform.
- ViMD real-audio loading passed for stereo 44.1 kHz embedded input and
  produced a finite 147,824-sample mono 16 kHz waveform.
- Loading the same ViMD row twice caused exactly one physical Parquet row-group
  read and reproduced identical canonical samples.
- The observed 4.410327-second initial ViMD load is a smoke-test diagnostic,
  not a final latency result; shard-aware access is required during training.
- Evidence artifact: `results/data_audit/audio_pipeline_smoke.json`.
- Methodology decision:
  `docs/decisions/003_canonical_audio_preprocessing.md`.

## Accepted Verification Metric Definitions — 2026-08-20

- Genuine trials are labeled `1`, impostor trials are labeled `0`, and larger
  scores indicate greater similarity.
- Acceptance uses `score >= threshold`; tied scores enter ROC points atomically.
- EER is linearly interpolated around the empirical FAR/FRR crossing.
- The deployed decision threshold is selected using Validation only, frozen,
  and then used for Test FAR, FRR, TAR, and accuracy.
- Test scores must never retune the deployed threshold.
- Default normalized minDCF uses `P_target=0.01`, `C_miss=1`, and
  `C_false_alarm=1`.
- TAR is reported at FAR 5%, 1%, 0.1%, and 0.01%, together with achieved FAR
  and threshold because finite trial lists limit empirical FAR resolution.
- Shared metrics are implemented without model-specific code paths and covered
  by the 62-test regression suite.
- Methodology decision: `docs/decisions/004_verification_metrics.md`.

## Verification Trial Protocol Implementation — 2026-08-20

- Deterministic genuine and impostor trial construction is implemented for
  canonical Validation and Test partitions.
- Genuine trials are capped per speaker; impostor generation samples speaker
  identities uniformly before sampling utterances.
- The proposed fixed settings are seed 42, at most 20 genuine pairs per
  speaker, and 100,000 unique impostor pairs per dataset/split.
- One hundred thousand impostor trials provide an empirical FAR resolution of
  0.001 percentage points, supporting analysis at the required FAR 0.01%
  target with ten false-accept increments at that target.
- Trial lists receive deterministic SHA-256 fingerprints and are shared by all
  three models.
- The complete local regression suite contains 68 passing tests.
- The real-data protocol artifact and its four final fingerprints remain
  pending Kaggle execution; the settings are not considered fully accepted
  until that evidence is reviewed.
