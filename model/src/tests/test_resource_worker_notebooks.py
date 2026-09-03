"""Static acceptance tests for the three shareable Kaggle workers."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PINNED_REVISION = "c68471a69c089cc40a5975b22362da37abcac186"
NOTEBOOKS = {
    "ecapa_tdnn": PROJECT_ROOT / "notebooks/02_run_all_ecapa_tdnn_t4x2.ipynb",
    "rawnet3": PROJECT_ROOT / "notebooks/03_run_all_rawnet3_t4x2.ipynb",
    "wavlm_mhfa": PROJECT_ROOT / "notebooks/04_run_all_wavlm_mhfa_t4x2.ipynb",
}


@pytest.mark.parametrize(("model_name", "path"), NOTEBOOKS.items())
def test_worker_notebook_is_clean_pinned_and_runnable(
    model_name: str,
    path: Path,
) -> None:
    """Every notebook must be valid, output-free, pinned, and model-specific."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["nbformat"] == 4
    assert payload["metadata"]["kernelspec"]["name"] == "python3"
    assert len(payload["cells"]) == 3

    complete_text = "\n".join(
        "".join(cell.get("source", ())) for cell in payload["cells"]
    )
    assert PINNED_REVISION in complete_text
    assert "GPU T4 x2" in complete_text
    assert "mozzila-tidyvoice" in complete_text
    assert "vimd-dataset" in complete_text
    assert f'"{model_name}"' in complete_text
    assert "run_resource_constrained_worker.py" in complete_text
    assert "resource_constrained.zip" in complete_text
    assert not re.search(
        r"(?:hf_|gh[pousr]_)[A-Za-z0-9]{20,}",
        complete_text,
    )

    for cell in payload["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            ast.parse("".join(cell["source"]))
