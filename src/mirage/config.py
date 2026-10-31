"""Config loading and global seeding."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from mirage import paths


def load_config(name: str) -> dict[str, Any]:
    """Load configs/<name>.yaml as a plain dict."""
    path = paths.CONFIGS / f"{name}.yaml"
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_resolved(config: dict[str, Any], out_dir: Path) -> None:
    """Write the exact config a run used next to its outputs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seeds libraries that still use the legacy global RNG
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:  # torch is optional (see majordecisions.md D2)
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
