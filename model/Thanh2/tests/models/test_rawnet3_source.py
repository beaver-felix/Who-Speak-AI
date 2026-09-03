"""Dependency-free checks for the vendored RawNet3 source and provenance."""

from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENDORED_ROOT = (
    PROJECT_ROOT
    / "src"
    / "speaker_recognition"
    / "third_party"
    / "clova_rawnet3"
)
ADAPTER_PATH = (
    PROJECT_ROOT / "src" / "speaker_recognition" / "models" / "rawnet3.py"
)


def _parse(path: Path) -> ast.Module:
    """Parse a Python file without importing optional model dependencies."""
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _string_constants(path: Path) -> dict[str, str]:
    """Extract simple module-level string assignments from an AST."""
    constants: dict[str, str] = {}
    for node in _parse(path).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name):
            value = ast.literal_eval(node.value)
            if isinstance(value, str):
                constants[target.id] = value
    return constants


def test_vendored_python_sources_parse_without_optional_dependencies() -> None:
    """Vendored modules must remain syntactically valid in lightweight CI."""
    for filename in ("blocks.py", "rawnet3.py"):
        _parse(VENDORED_ROOT / filename)


def test_vendored_architecture_exposes_required_components() -> None:
    """The adapted source must retain all checkpoint-relevant classes."""
    block_names = {
        node.name
        for node in _parse(VENDORED_ROOT / "blocks.py").body
        if isinstance(node, ast.ClassDef)
    }
    architecture_names = {
        node.name
        for node in _parse(VENDORED_ROOT / "rawnet3.py").body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }

    assert {"PreEmphasis", "AFMS", "Bottle2neck"} <= block_names
    assert {"RawNet3", "build_rawnet3"} <= architecture_names


def test_source_record_pins_audited_repository_hashes() -> None:
    """Provenance must state the immutable revision and upstream file hashes."""
    source = (VENDORED_ROOT / "SOURCE.md").read_text(encoding="utf-8")

    assert "f51bab870672a9b0b50fa158b4e30f329e7866d7" in source
    assert "8daf5e486055fceda56b0571fc70c685bf9d19a6f41463f4193b88a1f34b1636" in source
    assert "63795b5d0cbde5b1cd502fac09ea84962184abd96446e65851b3d400c2454f76" in source
    assert "86147c93a19287fef53accd4137d6474190db78c6833e8d9dc4f196331e3d5f8" in source


def test_vendored_source_includes_upstream_mit_terms() -> None:
    """Redistributed source must carry NAVER's complete license notice."""
    license_text = (VENDORED_ROOT / "LICENSE.md").read_text(encoding="utf-8")

    assert "Copyright (c) 2020-present NAVER Corp." in license_text
    assert "Permission is hereby granted, free of charge" in license_text
    assert 'THE SOFTWARE IS PROVIDED "AS IS"' in license_text


def test_adapter_pins_audited_checkpoint_identity() -> None:
    """The concrete adapter must not drift from the inspected checkpoint."""
    constants = _string_constants(ADAPTER_PATH)

    assert constants["RAWNET3_MODEL_ID"] == "jungjee/RawNet3"
    assert constants["RAWNET3_CHECKPOINT_REVISION"] == (
        "c89102eea20c3f96917c434de673c0ace0caddc0"
    )
    assert constants["RAWNET3_ARCHITECTURE_REVISION"] == (
        "f51bab870672a9b0b50fa158b4e30f329e7866d7"
    )
    assert constants["RAWNET3_CHECKPOINT_SHA256"] == (
        "1ab283bcdf776bfceceea18240e56a8756835b1911b04f9c44f347d47c09f90c"
    )
