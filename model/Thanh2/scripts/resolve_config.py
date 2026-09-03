"""Resolve layered TOML experiment configuration into fingerprinted JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from speaker_recognition.configuration import (
    resolve_layered_config,
    write_resolved_config,
)


def parse_arguments() -> argparse.Namespace:
    """Parse ordered layers, explicit overrides, and output path."""
    parser = argparse.ArgumentParser(
        description="Resolve and fingerprint an experiment configuration."
    )
    parser.add_argument(
        "--layer",
        action="append",
        required=True,
        type=Path,
        help="TOML layer in merge order; repeat for each layer.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="overrides",
        metavar="DOTTED_PATH=VALUE",
        help=(
            "Override an existing leaf. VALUE is parsed as JSON when valid, "
            "otherwise as text."
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def parse_overrides(raw_overrides: list[str]) -> dict[str, Any]:
    """Parse unique ``dotted.path=value`` command-line overrides."""
    parsed: dict[str, Any] = {}
    for raw_override in raw_overrides:
        path, separator, raw_value = raw_override.partition("=")
        if not separator or not path.strip() or not raw_value.strip():
            raise ValueError(
                f"Invalid override {raw_override!r}; expected path=value."
            )
        normalized_path = path.strip()
        if normalized_path in parsed:
            raise ValueError(f"Duplicate override: {normalized_path!r}.")
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        parsed[normalized_path] = value
    return parsed


def main() -> None:
    """Resolve requested layers and persist their canonical result."""
    arguments = parse_arguments()
    resolved = resolve_layered_config(
        arguments.layer,
        overrides=parse_overrides(arguments.overrides),
    )
    write_resolved_config(resolved, arguments.output)
    print("EXPERIMENT CONFIGURATION RESOLVED")
    print(f"layers: {len(resolved.source_paths)}")
    print(f"SHA-256: {resolved.sha256}")
    print(f"output: {arguments.output.resolve()}")


if __name__ == "__main__":
    main()
