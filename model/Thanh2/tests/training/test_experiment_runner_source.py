"""Static and dependency-free integration guards for the experiment CLI."""

import runpy
from pathlib import Path

import pytest

from speaker_recognition.configuration import read_resolved_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_runner_sets_determinism_before_importing_torch() -> None:
    """cuBLAS state must exist before CUDA-backed modules are imported."""
    source = (PROJECT_ROOT / "scripts/train_experiment.py").read_text(
        encoding="utf-8"
    )

    assert source.index('os.environ["CUBLAS_WORKSPACE_CONFIG"]') < (
        source.index("    import torch")
    )
    for required in (
        "read_resolved_config",
        "validate_manifest",
        "manifest_sha256",
        "trial_list_sha256",
        "Split.VALIDATION",
        "TrainingRunIdentity",
        "--resume requires run-dir/checkpoints/last.pt",
        "Existing run state requires --resume",
        "epoch_memberships",
        "utterance_id_sha256",
    ):
        assert required in source


def test_runner_runs_test_only_after_validation_selected_training() -> None:
    """Test must follow fit and use the best Validation checkpoint/threshold."""
    source = (PROJECT_ROOT / "scripts/train_experiment.py").read_text(
        encoding="utf-8"
    )

    assert source.index("outcome = engine.fit(") < source.index(
        "final_test = _run_final_test("
    )
    assert "split=Split.TEST" in source
    assert 'threshold_name = "threshold_at_far_0p1pct"' in source
    assert "engine.resume_from(best_checkpoint)" in source
    assert "select_threshold" not in source


def test_run_validator_checks_all_evidence_boundaries() -> None:
    """Downloaded runs must be authenticated without unsafe checkpoint loads."""
    source = (PROJECT_ROOT / "scripts/validate_training_run.py").read_text(
        encoding="utf-8"
    )

    for required in (
        "read_resolved_config",
        "parse_constant=_reject_json_constant",
        "validation_trial_list_sha256",
        "validation_epoch_",
        "checkpoint_sha256",
        "_sha256_file(checkpoint)",
        "metrics.jsonl",
        "Pilot membership must contain 512 speakers and utterances",
        "Final Test did not use the authenticated best checkpoint",
        "frozen_validation_security_threshold",
    ):
        assert required in source
    assert "torch.load" not in source


@pytest.mark.parametrize(
    ("stage", "speaker_limit", "tracking_mode"),
    (
        ("pilot", 512, "offline"),
        ("resource_constrained", 0, "offline"),
        ("full", 0, "online"),
    ),
)
def test_prepare_script_writes_six_authenticated_configs(
    tmp_path: Path,
    stage: str,
    speaker_limit: int,
    tracking_mode: str,
) -> None:
    """One command must prepare a complete model-by-dataset stage matrix."""
    namespace = runpy.run_path(
        str(PROJECT_ROOT / "scripts/prepare_experiment_configs.py")
    )
    outputs = namespace["prepare_experiment_configs"](
        project_root=PROJECT_ROOT,
        output_directory=tmp_path,
        stage=stage,
    )

    assert len(outputs) == 6
    for output in outputs:
        resolved = read_resolved_config(output)
        assert resolved.get("experiment.stage") == stage
        assert resolved.get(
            "training.epoch_sampling.max_speakers_per_epoch"
        ) == speaker_limit
        assert resolved.get("tracking.mode") == tracking_mode


def test_resource_worker_uses_one_process_per_gpu_and_archives_evidence() -> None:
    """The shareable worker must isolate datasets across both Kaggle GPUs."""
    source = (
        PROJECT_ROOT / "scripts/run_resource_constrained_worker.py"
    ).read_text(encoding="utf-8")

    for required in (
        "torch.cuda.device_count() < 2",
        'f"cuda:{gpu_index}"',
        "subprocess.Popen(",
        "--resume",
        "validate_training_run.py",
        "ZIP_STORED",
        "RESOURCE-CONSTRAINED WORKER COMPLETE",
    ):
        assert required in source
