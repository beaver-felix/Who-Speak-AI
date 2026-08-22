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
    assert '"torch_version": str(torch.__version__)' in source
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


def test_cuda_resume_gate_rebuilds_and_compares_complete_state() -> None:
    """The empirical gate must exercise a real interruption boundary."""
    source = (
        PROJECT_ROOT / "scripts/smoke_test_checkpoint_resume.py"
    ).read_text(encoding="utf-8")

    for required in (
        "save_training_checkpoint",
        "del adapter, objective, optimizer, scaler",
        "restore_training_checkpoint",
        "restored.cursor != saved_cursor",
        "adapter_state_exact",
        "objective_state_exact",
        "optimizer_state_exact",
        "scaler_state_exact",
        "rng_dependent_embedding_exact",
        "torch.use_deterministic_algorithms(True)",
    ):
        assert required in source
