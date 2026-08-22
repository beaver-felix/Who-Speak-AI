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


def test_runner_has_no_test_scoring_or_threshold_selection() -> None:
    """Training-time model selection must remain Validation-only."""
    source = (PROJECT_ROOT / "scripts/train_experiment.py").read_text(
        encoding="utf-8"
    )

    assert "split=Split.TEST" not in source
    assert "security_threshold" not in source
    assert "select_threshold" not in source


@pytest.mark.parametrize(
    ("stage", "speaker_limit", "tracking_mode"),
    (("pilot", 512, "offline"), ("full", 0, "online")),
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
