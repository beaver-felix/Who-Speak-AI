"""Dependency-free guards for ECAPA deterministic training compatibility."""

from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ADAPTER_PATH = (
    PROJECT_ROOT / "src/speaker_recognition/models/ecapa_tdnn.py"
)


def test_ecapa_adapter_source_parses_without_optional_dependencies() -> None:
    """The adapter must remain syntactically valid in lightweight CI."""
    source = ADAPTER_PATH.read_text(encoding="utf-8")
    ast.parse(source, filename=str(ADAPTER_PATH))


def test_ecapa_replaces_only_reflection_pad_with_equivalent_operations() -> None:
    """Strict mode must not be weakened to tolerate nondeterministic CUDA."""
    source = ADAPTER_PATH.read_text(encoding="utf-8")

    for required in (
        "def deterministic_reflection_pad1d",
        'if mode == "reflect" and len(padding) == 2',
        ".flip(-1)",
        "torch.cat(pieces, dim=-1)",
        "with _deterministic_reflection_padding()",
        "finally:",
        "functional.pad = native_pad",
    ):
        assert required in source
    assert "warn_only=True" not in source


def test_ecapa_smoke_gate_enables_strict_determinism() -> None:
    """The real GPU gradient gate must reproduce the pilot failure boundary."""
    source = (PROJECT_ROOT / "scripts/smoke_test_ecapa_adapter.py").read_text(
        encoding="utf-8"
    )

    assert 'os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"' in source
    assert "torch.use_deterministic_algorithms(True)" in source
    assert '"strict_deterministic_algorithms_enabled"' in source
