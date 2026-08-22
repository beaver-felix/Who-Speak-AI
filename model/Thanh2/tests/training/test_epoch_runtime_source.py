"""Static guards for Kaggle-only epoch training and checkpoint mechanics."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_engine_preserves_required_safety_boundaries() -> None:
    """The shared engine must retain the accepted fail-closed mechanics."""
    source = (
        PROJECT_ROOT
        / "src/speaker_recognition/training/engine.py"
    ).read_text(encoding="utf-8")

    for required in (
        "DeterministicEpochBatchSampler",
        "dataset.set_epoch",
        "persistent_workers=False",
        "torch.amp.GradScaler",
        "torch.autocast",
        "clip_grad_norm_",
        "scale_after < scale_before",
        "validation_function",
        "validation_eer",
        "validation_min_dcf",
        "checkpoint_every_steps",
    ):
        assert required in source


def test_checkpoint_uses_safe_load_and_atomic_replace() -> None:
    """Resume artifacts must fail closed and never overwrite partially."""
    source = (
        PROJECT_ROOT
        / "src/speaker_recognition/training/checkpointing.py"
    ).read_text(encoding="utf-8")

    assert "weights_only=True" in source
    assert "os.replace(partial, destination)" in source
    assert "os.fsync" in source
    assert "Checkpoint identity does not match" in source
    assert "torch.cuda.get_rng_state" in source
    assert "torch.cuda.set_rng_state" in source


def test_wandb_logger_uses_explicit_run_identity_and_resume_policy() -> None:
    """Kaggle restarts must reconnect to one run instead of creating duplicates."""
    source = (
        PROJECT_ROOT
        / "src/speaker_recognition/training/logging.py"
    ).read_text(encoding="utf-8")

    assert "id=run_id" in source
    assert 'resume="must" if resume else "never"' in source
    assert "allow_nan=False" in source
