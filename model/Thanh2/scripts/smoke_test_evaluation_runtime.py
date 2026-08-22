"""Exercise the shared evaluation runtime on real TidyVoice audio and CUDA."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

from speaker_recognition.data.dataset import EvaluationSpeakerDataset
from speaker_recognition.data.manifest import ManifestRecord, Split, validate_manifest
from speaker_recognition.data.tidyvoice import iter_tidyvoice_dev_records
from speaker_recognition.evaluation.runtime import (
    ExtractionSettings,
    benchmark_single_crop_latency,
)
from speaker_recognition.evaluation.trials import (
    VerificationTrial,
    trial_list_sha256,
)
from speaker_recognition.evaluation.validation import (
    VerificationValidationCallback,
)


MODEL_SETTINGS = {
    "ecapa_tdnn": {"segment_samples": 48_000, "batch_size": 4},
    "rawnet3": {"segment_samples": 64_240, "batch_size": 4},
    "wavlm_mhfa": {"segment_samples": 64_240, "batch_size": 2},
}
REQUIRED_METRICS = {
    "eer",
    "min_dcf",
    "far",
    "frr",
    "tar",
    "accuracy",
    "tar_at_far_5pct",
    "tar_at_far_1pct",
    "tar_at_far_0p1pct",
    "tar_at_far_0p01pct",
}


def parse_arguments() -> argparse.Namespace:
    """Parse the model, mounted data, artifact cache, and evidence paths."""
    parser = argparse.ArgumentParser(
        description="Smoke-test cached Validation evaluation on a Tesla T4."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(MODEL_SETTINGS),
    )
    parser.add_argument("--tidyvoice-root", required=True, type=Path)
    parser.add_argument("--tidyvoice-protocol", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _load_assignments(path: Path) -> dict[str, Split]:
    """Load the committed TidyVoice speaker-disjoint Dev protocol."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("assignments")
    if not isinstance(values, dict) or not values:
        raise ValueError("TidyVoice protocol assignments are missing.")
    return {
        str(speaker_id): Split(split_name)
        for speaker_id, split_name in values.items()
    }


def _select_fixture(
    records: tuple[ManifestRecord, ...],
) -> tuple[ManifestRecord, ...]:
    """Select two lexical utterances from four Validation speakers."""
    by_speaker: dict[str, list[ManifestRecord]] = defaultdict(list)
    for record in records:
        if record.split is Split.VALIDATION:
            by_speaker[record.speaker_id].append(record)
    selected: list[ManifestRecord] = []
    for speaker_id in sorted(by_speaker):
        speaker_records = sorted(
            by_speaker[speaker_id],
            key=lambda record: record.utterance_id,
        )
        if len(speaker_records) >= 2:
            selected.extend(speaker_records[:2])
        if len(selected) == 8:
            break
    if len(selected) != 8:
        raise RuntimeError(
            "The evaluation gate requires four Validation speakers with two "
            "utterances each."
        )
    return tuple(selected)


def _build_fixture_trials(
    records: tuple[ManifestRecord, ...],
) -> tuple[VerificationTrial, ...]:
    """Reference every fixture utterance in genuine and impostor trials."""
    by_speaker: dict[str, list[ManifestRecord]] = defaultdict(list)
    for record in records:
        by_speaker[record.speaker_id].append(record)
    speakers = sorted(by_speaker)
    trials: list[VerificationTrial] = []
    for speaker_index, speaker_id in enumerate(speakers):
        left, right = sorted(
            by_speaker[speaker_id],
            key=lambda record: record.utterance_id,
        )
        trials.append(
            VerificationTrial(
                trial_id=f"genuine:{speaker_index:04d}",
                label=1,
                left_utterance_id=left.utterance_id,
                right_utterance_id=right.utterance_id,
                left_speaker_id=speaker_id,
                right_speaker_id=speaker_id,
            )
        )
    for pair_index, (left_speaker, right_speaker) in enumerate(
        combinations(speakers, 2)
    ):
        left_records = sorted(
            by_speaker[left_speaker],
            key=lambda record: record.utterance_id,
        )
        right_records = sorted(
            by_speaker[right_speaker],
            key=lambda record: record.utterance_id,
        )
        for utterance_index in range(2):
            trials.append(
                VerificationTrial(
                    trial_id=(
                        f"impostor:{pair_index:04d}:{utterance_index}"
                    ),
                    label=0,
                    left_utterance_id=(
                        left_records[utterance_index].utterance_id
                    ),
                    right_utterance_id=(
                        right_records[utterance_index].utterance_id
                    ),
                    left_speaker_id=left_speaker,
                    right_speaker_id=right_speaker,
                )
            )
    return tuple(trials)


def _build_adapter(
    model_name: str,
    *,
    cache_dir: Path,
    device: str,
) -> Any:
    """Load one pinned adapter without importing the other optional stacks."""
    if model_name == "ecapa_tdnn":
        from speaker_recognition.models.ecapa_tdnn import EcapaTdnnAdapter

        return EcapaTdnnAdapter.from_pretrained(
            cache_dir=cache_dir,
            device=device,
        )
    if model_name == "rawnet3":
        from speaker_recognition.models.rawnet3 import RawNet3Adapter

        return RawNet3Adapter.from_pretrained(
            cache_dir=cache_dir,
            device=device,
        )
    if model_name == "wavlm_mhfa":
        from speaker_recognition.models.wavlm_mhfa import WavlmMhfaAdapter

        return WavlmMhfaAdapter.from_pretrained(
            cache_dir=cache_dir,
            device=device,
        )
    raise ValueError(f"Unsupported model: {model_name!r}.")


def _write_json(payload: dict[str, object], path: Path) -> None:
    """Write finite evidence after every gate check has passed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    """Run a bounded end-to-end Validation callback and latency gate."""
    arguments = parse_arguments()
    import torch

    if not torch.cuda.is_available() or not arguments.device.startswith("cuda"):
        raise RuntimeError("The accepted evaluation gate requires CUDA.")
    settings = MODEL_SETTINGS[arguments.model]
    assignments = _load_assignments(arguments.tidyvoice_protocol)
    records = validate_manifest(
        tuple(
            iter_tidyvoice_dev_records(
                arguments.tidyvoice_root,
                dev_assignments=assignments,
            )
        )
    )
    fixture_records = _select_fixture(records)
    trials = _build_fixture_trials(fixture_records)
    trial_sha256 = trial_list_sha256(trials)
    dataset = EvaluationSpeakerDataset(
        fixture_records,
        split=Split.VALIDATION,
        dataset_roots={"tidyvoice": arguments.tidyvoice_root},
        segment_samples=int(settings["segment_samples"]),
        segment_count=2,
    )
    adapter = _build_adapter(
        arguments.model,
        cache_dir=arguments.cache_dir,
        device=arguments.device,
    )
    callback = VerificationValidationCallback(
        dataset=dataset,
        trials=trials,
        expected_trial_sha256=trial_sha256,
        extraction_settings=ExtractionSettings(
            utterance_batch_size=int(settings["batch_size"]),
            num_workers=0,
        ),
        output_directory=arguments.work_dir / "epochs",
        evidence_context={
            "purpose": "bounded_real_audio_evaluation_runtime_gate",
            "model_name": arguments.model,
            "dataset_name": "tidyvoice",
            "seed": 42,
        },
        device=arguments.device,
    )

    torch.cuda.reset_peak_memory_stats(arguments.device)
    returned_metrics = dict(callback(adapter, 0))
    validation_path = arguments.work_dir / "epochs/validation_epoch_0000.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    latency = benchmark_single_crop_latency(
        adapter,
        dataset[0].waveforms[0],
        device=arguments.device,
    )
    peak_memory = int(torch.cuda.max_memory_allocated(arguments.device))
    metric_values = validation["metrics"]
    checks = {
        "trial_fingerprint_matches": (
            validation["protocol"]["trial_list_sha256"] == trial_sha256
        ),
        "exact_trial_count": validation["protocol"]["trial_count"] == 16,
        "exact_genuine_count": (
            validation["protocol"]["genuine_trial_count"] == 4
        ),
        "exact_impostor_count": (
            validation["protocol"]["impostor_trial_count"] == 12
        ),
        "exact_utterance_count": (
            validation["embeddings"]["utterance_count"] == 8
        ),
        "exact_crop_count": (
            validation["evaluation"]["embedding_extraction"]["crop_count"]
            == 16
        ),
        "complete_required_metrics": REQUIRED_METRICS <= set(metric_values),
        "metrics_finite": all(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            for value in metric_values.values()
        ),
        "security_threshold_not_selected": (
            validation["threshold_policy"]["security_threshold_selected"]
            is False
        ),
        "returned_metrics_finite": all(
            math.isfinite(float(value)) for value in returned_metrics.values()
        ),
        "latency_finite_positive": all(
            math.isfinite(value) and value > 0.0
            for value in (latency.mean_ms, latency.median_ms, latency.p95_ms)
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Evaluation runtime gate failed: {checks}")

    payload: dict[str, object] = {
        "schema_version": 1,
        "purpose": "bounded_real_audio_evaluation_runtime_gate",
        "model": {
            "name": adapter.metadata.name,
            "source_id": adapter.metadata.source_id,
            "revision": adapter.metadata.revision,
            "embedding_dim": adapter.metadata.embedding_dim,
        },
        "fixture": {
            "dataset": "tidyvoice",
            "partition": "validation",
            "speaker_count": 4,
            "utterance_count": 8,
            "trial_count": 16,
            "genuine_trial_count": 4,
            "impostor_trial_count": 12,
            "segment_samples": int(settings["segment_samples"]),
            "segment_count": 2,
            "utterance_batch_size": int(settings["batch_size"]),
            "trial_list_sha256": trial_sha256,
        },
        "runtime": {
            "device": arguments.device,
            "device_name": torch.cuda.get_device_name(arguments.device),
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "maximum_allocated_cuda_bytes": peak_memory,
        },
        "validation": validation,
        "latency": latency.to_dict(),
        "checks": checks,
    }
    _write_json(payload, arguments.output)

    print("REAL EVALUATION RUNTIME GATE PASSED")
    print(f"model: {arguments.model}")
    print("fixture: 4 speakers, 8 utterances, 16 trials, 16 crops")
    print(f"EER: {float(metric_values['eer']):.6f}")
    print(f"minDCF: {float(metric_values['min_dcf']):.6f}")
    print(f"median model latency: {latency.median_ms:.3f} ms")
    print(f"output: {arguments.output.resolve()}")


if __name__ == "__main__":
    main()
