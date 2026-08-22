"""Static dependency gates for PyTorch-only training modules."""

from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = PROJECT_ROOT / "src" / "speaker_recognition" / "training"


def _source(filename: str) -> str:
    """Read one concrete training module without importing PyTorch."""
    return (TRAINING_ROOT / filename).read_text(encoding="utf-8")


def test_pytorch_training_modules_parse_without_optional_dependencies() -> None:
    """Both concrete modules must remain syntactically valid on local CI."""
    for filename in ("objectives.py", "optimization.py"):
        ast.parse(_source(filename), filename=filename)


def test_aam_implementation_uses_normalized_cosine_and_target_scatter() -> None:
    """The implementation must retain its stable, memory-bounded formulation."""
    source = _source("objectives.py")

    assert source.count("functional.normalize(") >= 2
    assert ".clamp(-1.0 + 1e-7, 1.0 - 1e-7)" in source
    assert "logits.scatter_" in source
    assert "functional.cross_entropy" in source
    assert "one_hot" not in source


def test_optimizer_policy_freezes_before_selective_enablement() -> None:
    """WavLM must default to frozen and enable only reviewed components."""
    source = _source("optimization.py")

    assert "_set_all_trainable(adapter, False)" in source
    assert 'getattr(adapter, "mhfa", None)' in source
    assert 'getattr(wavlm, "encoder", None)' in source
    assert "_validate_parameter_partition" in source
