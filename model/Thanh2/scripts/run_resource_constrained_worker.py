"""Run both dataset experiments for one model on a Kaggle T4 pair.

This is the one-command boundary used by the three shareable notebooks.  GPU 0
owns TidyVoice and GPU 1 owns ViMD; each experiment remains an independent
process, run directory, log, configuration, checkpoint lifecycle, and cache.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path


MODELS = ("ecapa_tdnn", "rawnet3", "wavlm_mhfa")
DATASET_ROOTS = {
    "tidyvoice": Path(
        "/kaggle/input/datasets/dullahn/mozzila-tidyvoice/TidyVoiceX_ASV"
    ),
    "vimd": Path("/kaggle/input/datasets/dullahn/vimd-dataset"),
}


def parse_arguments() -> argparse.Namespace:
    """Parse the fixed architecture and writable Kaggle output root."""
    parser = argparse.ArgumentParser(
        description="Run one architecture on TidyVoice and ViMD concurrently."
    )
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/kaggle/working/resource_constrained_results"),
    )
    return parser.parse_args()


def _last_non_empty_line(path: Path) -> str:
    """Return a bounded progress message from one subprocess log."""
    if not path.is_file():
        return "waiting for log output"
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "log temporarily unavailable"
    for line in reversed(lines):
        if line.strip():
            return line.strip()[-180:]
    return "process started"


def _run_ecapa_determinism_gate(
    *,
    project_root: Path,
    output_root: Path,
) -> None:
    """Fail closed before ECAPA training if strict CUDA gradients regress."""
    evidence = output_root / "ecapa_strict_determinism_gate.json"
    subprocess.run(
        (
            sys.executable,
            str(project_root / "scripts/smoke_test_ecapa_adapter.py"),
            "--cache-dir",
            str(output_root / "cache/ecapa_preflight"),
            "--device",
            "cuda:0",
            "--output",
            str(evidence),
        ),
        cwd=project_root,
        check=True,
    )


def _run_wavlm_fp32_fallback_gate(
    *,
    project_root: Path,
    output_root: Path,
) -> None:
    """Prove that a real batch-six FP32 fallback fits and updates on T4."""
    evidence = output_root / "wavlm_fp32_fallback_gate.json"
    subprocess.run(
        (
            sys.executable,
            str(project_root / "scripts/smoke_test_multibatch_training.py"),
            "--model",
            "wavlm_mhfa",
            "--dataset-root",
            str(DATASET_ROOTS["tidyvoice"]),
            "--cache-dir",
            str(output_root / "cache/wavlm_preflight"),
            "--steps",
            "3",
            "--precision",
            "fp32",
            "--device",
            "cuda:0",
            "--output",
            str(evidence),
        ),
        cwd=project_root,
        check=True,
    )


def _archive_outputs(source: Path, destination: Path) -> None:
    """Create a fast, complete ZIP without recompressing tensor checkpoints."""
    partial = destination.with_name(destination.name + ".part")
    if partial.exists():
        partial.unlink()
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(source.rglob("*"), key=lambda item: item.as_posix()):
            if path.is_file() and path != partial and path != destination:
                archive.write(path, path.relative_to(source.parent).as_posix())
    partial.replace(destination)


def main() -> None:
    """Prepare, launch, monitor, validate, and archive two isolated runs."""
    arguments = parse_arguments()
    project_root = Path(__file__).resolve().parents[1]
    output_root = arguments.output_root.expanduser().resolve() / arguments.model
    output_root.mkdir(parents=True, exist_ok=True)

    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("Kaggle PyTorch is required.") from error
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("This worker requires Kaggle GPU T4 x2.")
    for name, root in DATASET_ROOTS.items():
        if not root.is_dir():
            raise FileNotFoundError(
                f"Required {name} Kaggle input is not attached: {root}"
            )

    config_root = output_root / "configs"
    subprocess.run(
        (
            sys.executable,
            str(project_root / "scripts/prepare_experiment_configs.py"),
            "--stage",
            "resource_constrained",
            "--output-dir",
            str(config_root),
        ),
        cwd=project_root,
        check=True,
    )
    if arguments.model == "ecapa_tdnn":
        _run_ecapa_determinism_gate(
            project_root=project_root,
            output_root=output_root,
        )
    elif arguments.model == "wavlm_mhfa":
        _run_wavlm_fp32_fallback_gate(
            project_root=project_root,
            output_root=output_root,
        )

    processes: dict[str, subprocess.Popen[bytes]] = {}
    streams: dict[str, object] = {}
    logs: dict[str, Path] = {}
    try:
        for gpu_index, dataset_name in enumerate(("tidyvoice", "vimd")):
            run_directory = output_root / "runs" / dataset_name
            log_path = output_root / "logs" / f"{dataset_name}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            stream = log_path.open("ab")
            streams[dataset_name] = stream
            logs[dataset_name] = log_path
            command = [
                sys.executable,
                str(project_root / "scripts/train_experiment.py"),
                "--config",
                str(
                    config_root
                    / (
                        f"{dataset_name}_{arguments.model}_"
                        "resource_constrained_s42.json"
                    )
                ),
                "--run-dir",
                str(run_directory),
                "--cache-dir",
                str(output_root / "cache" / dataset_name),
                "--dataset-root",
                str(DATASET_ROOTS[dataset_name]),
                "--device",
                f"cuda:{gpu_index}",
            ]
            if (run_directory / "checkpoints/last.pt").is_file():
                command.append("--resume")
            processes[dataset_name] = subprocess.Popen(
                command,
                cwd=project_root,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
            print(
                f"STARTED {dataset_name.upper()} ON GPU {gpu_index}: "
                f"PID {processes[dataset_name].pid}",
                flush=True,
            )

        while any(process.poll() is None for process in processes.values()):
            for dataset_name, process in processes.items():
                status = "RUNNING" if process.poll() is None else str(process.returncode)
                print(
                    f"[{dataset_name}={status}] "
                    f"{_last_non_empty_line(logs[dataset_name])}",
                    flush=True,
                )
            time.sleep(30)
    except KeyboardInterrupt:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        raise
    finally:
        for stream in streams.values():
            stream.close()

    failures = {
        name: process.returncode
        for name, process in processes.items()
        if process.returncode != 0
    }
    if failures:
        details = {
            name: _last_non_empty_line(logs[name]) for name in failures
        }
        raise RuntimeError(
            f"Experiment process failure(s): {failures}; last lines: {details}"
        )

    for dataset_name in ("tidyvoice", "vimd"):
        subprocess.run(
            (
                sys.executable,
                str(project_root / "scripts/validate_training_run.py"),
                "--run-dir",
                str(output_root / "runs" / dataset_name),
            ),
            cwd=project_root,
            check=True,
        )

    manifest = {
        "schema_version": 1,
        "stage": "resource_constrained",
        "model": arguments.model,
        "gpu_assignment": {"tidyvoice": "cuda:0", "vimd": "cuda:1"},
        "methodology": (
            "Compute-constrained comparison; three epochs maximum, one "
            "rotating utterance per Train speaker per epoch, patience one."
        ),
    }
    (output_root / "worker_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    # Download only the architecture subtree. Model caches are deliberately
    # excluded because pinned sources can be reproduced and would inflate it.
    cache_root = output_root / "cache"
    if cache_root.exists():
        shutil.rmtree(cache_root)
    archive_path = Path("/kaggle/working") / (
        f"who_speak_ai_{arguments.model}_resource_constrained.zip"
    )
    _archive_outputs(output_root, archive_path)
    print("RESOURCE-CONSTRAINED WORKER COMPLETE")
    print(f"model: {arguments.model}")
    print(f"DOWNLOAD READY: {archive_path}")


if __name__ == "__main__":
    main()
