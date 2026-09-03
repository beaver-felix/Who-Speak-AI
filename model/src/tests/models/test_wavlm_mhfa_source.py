"""Dependency-free WavLM+MHFA provenance and checkpoint-audit tests."""

from __future__ import annotations

import ast
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENDORED_ROOT = (
    PROJECT_ROOT
    / "src"
    / "speaker_recognition"
    / "third_party"
    / "theolepage_wavlm_mhfa"
)
ADAPTER_PATH = (
    PROJECT_ROOT / "src" / "speaker_recognition" / "models" / "wavlm_mhfa.py"
)
AUDIT_PATH = (
    PROJECT_ROOT
    / "results"
    / "model_audit"
    / "wavlm_mhfa_checkpoint_audit.json"
)


def _parse(path: Path) -> ast.Module:
    """Parse Python without importing optional model dependencies."""
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _string_assignments(path: Path) -> dict[str, str]:
    """Extract simple top-level string constants from a module AST."""
    result: dict[str, str] = {}
    for node in _parse(path).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        if not isinstance(node.targets[0], ast.Name):
            continue
        value = ast.literal_eval(node.value)
        if isinstance(value, str):
            result[node.targets[0].id] = value
    return result


def test_wavlm_mhfa_python_sources_parse_without_dependencies() -> None:
    """Concrete adapter and adapted MHFA must remain syntactically valid."""
    _parse(ADAPTER_PATH)
    _parse(VENDORED_ROOT / "mhfa.py")


def test_mhfa_retains_checkpoint_parameter_names() -> None:
    """The adapted backend must expose every parameter-bearing upstream name."""
    source = (VENDORED_ROOT / "mhfa.py").read_text(encoding="utf-8")

    for name in (
        "weights_k",
        "weights_v",
        "cmp_linear_k",
        "cmp_linear_v",
        "att_head",
        "pooling_fc",
    ):
        assert f"self.{name}" in source


def test_source_record_pins_all_audited_hashes() -> None:
    """Provenance must identify every executable upstream source file."""
    source = (VENDORED_ROOT / "SOURCE.md").read_text(encoding="utf-8")

    for expected in (
        "bfb8527de83b5347fb81b1e9e31be241656ca103",
        "fc4638d657a3ad09953e54ee76ae4877904b6a92c03d2f91ed6691b7f770d40f",
        "7cc0837302ff032d048c0f43ebdafdf0f009f72a10d78aa82f486df33c39aa63",
        "7a06a14a7dc95c5f65cd6b09ed126013821512489dcfec2e58bd8b544ce46656",
    ):
        assert expected in source


def test_vendored_mhfa_includes_upstream_mit_terms() -> None:
    """Adapted MHFA redistribution must retain its complete license notice."""
    license_text = (VENDORED_ROOT / "LICENSE.md").read_text(encoding="utf-8")

    assert "Copyright (c) 2024 Theo Lepage" in license_text
    assert "Permission is hereby granted, free of charge" in license_text
    assert 'THE SOFTWARE IS PROVIDED "AS IS"' in license_text


def test_adapter_pins_verified_source_and_checkpoint() -> None:
    """Adapter constants must match the inspected official artifacts."""
    constants = _string_assignments(ADAPTER_PATH)

    assert constants["WAVLM_MHFA_MODEL_ID"] == "theolepage/wavlm_ssl_sv"
    assert constants["WAVLM_MHFA_REVISION"] == (
        "bfb8527de83b5347fb81b1e9e31be241656ca103"
    )
    assert constants["WAVLM_MHFA_CHECKPOINT_FILE_ID"] == (
        "1RabuRETASqhh39K8weSoNkBa5DGRvgyx"
    )
    assert constants["WAVLM_MHFA_CHECKPOINT_SHA256"] == (
        "0178a115dc0a43a94a71287e51d1df5016c2aeefc04169548dad40ac8a6e67da"
    )


def test_downloaded_checkpoint_audit_is_complete() -> None:
    """Committed evidence must prove trained WavLM and MHFA states are present."""
    payload = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert all(payload["checks"].values())
    assert payload["checkpoint"]["sha256"] == (
        "0178a115dc0a43a94a71287e51d1df5016c2aeefc04169548dad40ac8a6e67da"
    )
    assert payload["checkpoint"]["tensor_count"] == 259
    assert payload["checkpoint"]["groups"]["wavlm"] == {
        "tensor_count": 248,
        "tensor_elements": 94_381_936,
    }
    assert payload["checkpoint"]["groups"]["mhfa"] == {
        "tensor_count": 10,
        "tensor_elements": 2_302_554,
    }
    assert payload["checkpoint"]["groups"]["source_loss"] == {
        "tensor_count": 1,
        "tensor_elements": 1_920_000,
    }
