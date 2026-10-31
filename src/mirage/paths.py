"""Single source of truth for every file location used by the pipeline.

Stages talk to each other only through files, so all of them resolve paths here.
Set MIRAGE_ROOT to relocate everything (e.g. /kaggle/working on Kaggle).
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("MIRAGE_ROOT", Path(__file__).resolve().parents[2]))

CONFIGS = Path(os.environ.get("MIRAGE_CONFIGS", Path(__file__).resolve().parents[2] / "configs"))

ARTIFACTS = ROOT / "artifacts"
DATA = ARTIFACTS / "data"
MODELS = ARTIFACTS / "models"
EMBEDDINGS = ARTIFACTS / "emb"

RESULTS = ROOT / "results"
PREDICTIONS = RESULTS / "predictions"
EXPLAIN = RESULTS / "explain"
FIGURES = ROOT / "figures"


def model_dir(model: str, variant: str = "base", seed: int | None = None) -> Path:
    suffix = f"__seed{seed}" if seed is not None else ""
    return MODELS / f"{model}__{variant}{suffix}"


def ensure_dirs() -> None:
    for d in (DATA, MODELS, EMBEDDINGS, PREDICTIONS, EXPLAIN, FIGURES):
        d.mkdir(parents=True, exist_ok=True)
