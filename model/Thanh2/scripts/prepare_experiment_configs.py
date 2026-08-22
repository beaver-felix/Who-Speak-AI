"""Resolve the complete model-by-dataset configuration matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

from speaker_recognition.configuration import (
    resolve_layered_config,
    write_resolved_config,
)


DATASETS = ("tidyvoice", "vimd")
MODELS = ("ecapa_tdnn", "rawnet3", "wavlm_mhfa")
STAGES = ("pilot", "full")


def parse_arguments() -> argparse.Namespace:
    """Parse one explicit stage and output directory."""
    parser = argparse.ArgumentParser(
        description="Prepare six authenticated experiment configurations."
    )
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def prepare_experiment_configs(
    *,
    project_root: Path,
    output_directory: Path,
    stage: str,
) -> tuple[Path, ...]:
    """Resolve and write all six immutable experiment configurations."""
    if stage not in STAGES:
        raise ValueError(f"Unsupported experiment stage: {stage!r}.")
    root = project_root.expanduser().resolve()
    configuration_root = root / "configs"
    output_root = output_directory.expanduser().resolve()
    written: list[Path] = []
    for dataset_name in DATASETS:
        for model_name in MODELS:
            resolved = resolve_layered_config(
                (
                    configuration_root / "base.toml",
                    configuration_root / "datasets" / f"{dataset_name}.toml",
                    configuration_root / "models" / f"{model_name}.toml",
                    configuration_root / "stages" / f"{stage}.toml",
                )
            )
            output_path = (
                output_root
                / f"{dataset_name}_{model_name}_{stage}_s42.json"
            )
            write_resolved_config(resolved, output_path)
            written.append(output_path)
            print(
                f"{dataset_name}/{model_name}: {resolved.sha256} -> "
                f"{output_path}"
            )
    return tuple(written)


def main() -> None:
    """Resolve the matrix relative to the checked-out project root."""
    arguments = parse_arguments()
    project_root = Path(__file__).resolve().parents[1]
    outputs = prepare_experiment_configs(
        project_root=project_root,
        output_directory=arguments.output_dir,
        stage=arguments.stage,
    )
    print("EXPERIMENT CONFIGURATION MATRIX READY")
    print(f"stage: {arguments.stage}")
    print(f"configurations: {len(outputs)}")
    print(f"output: {arguments.output_dir.expanduser().resolve()}")


if __name__ == "__main__":
    main()
