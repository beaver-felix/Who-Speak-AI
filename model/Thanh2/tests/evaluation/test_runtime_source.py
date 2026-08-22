"""Dependency-free source gates for Kaggle-only evaluation runtime modules."""

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_PATH = (
    PROJECT_ROOT / "src/speaker_recognition/evaluation/runtime.py"
)
VALIDATION_PATH = (
    PROJECT_ROOT / "src/speaker_recognition/evaluation/validation.py"
)
SMOKE_PATH = PROJECT_ROOT / "scripts/smoke_test_evaluation_runtime.py"


def test_kaggle_evaluation_modules_are_valid_python() -> None:
    """Both CUDA-only modules must remain syntactically importable on Kaggle."""
    ast.parse(RUNTIME_PATH.read_text(encoding="utf-8"))
    ast.parse(VALIDATION_PATH.read_text(encoding="utf-8"))
    ast.parse(SMOKE_PATH.read_text(encoding="utf-8"))


def test_extraction_uses_inference_mode_fp16_and_no_shuffle() -> None:
    """Embedding extraction must be deterministic and gradient-free."""
    source = RUNTIME_PATH.read_text(encoding="utf-8")

    assert "shuffle=False" in source
    assert "torch.inference_mode()" in source
    assert "dtype=torch.float16" in source
    assert "aggregate_crop_embeddings(" in source
    assert "utterance_offset != len(dataset)" in source


def test_latency_protocol_uses_cuda_events_after_warmup() -> None:
    """Model latency must exclude loading and use synchronized GPU timing."""
    source = RUNTIME_PATH.read_text(encoding="utf-8")

    assert "warmup_iterations: int = 10" in source
    assert "measured_iterations: int = 50" in source
    assert "torch.cuda.Event(enable_timing=True)" in source
    assert "finished.synchronize()" in source


def test_validation_callback_is_validation_only_and_atomic() -> None:
    """Epoch selection must reject Test and persist strict atomic evidence."""
    source = VALIDATION_PATH.read_text(encoding="utf-8")

    assert "dataset.split is not Split.VALIDATION" in source
    assert "dataset_utterance_ids != trial_utterance_ids" in source
    assert "allow_nan=False" in source
    assert "os.replace(partial, path)" in source


def test_real_gate_is_bounded_and_uses_every_fixture_utterance() -> None:
    """The CUDA gate must test multi-crop integration without claiming quality."""
    source = SMOKE_PATH.read_text(encoding="utf-8")

    assert '"speaker_count": 4' in source
    assert '"utterance_count": 8' in source
    assert '"trial_count": 16' in source
    assert "segment_count=2" in source
    assert "security_threshold_not_selected" in source
