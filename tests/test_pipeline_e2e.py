"""End-to-end: every CPU stage of the CLI on synthetic data, in an isolated MIRAGE_ROOT.

Runs in a subprocess so paths.py picks up MIRAGE_ROOT / MIRAGE_CONFIGS at import time.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

from tests.conftest import synthetic_sst

REPO = Path(__file__).resolve().parents[1]


def _write_small_configs(dst: Path) -> None:
    shutil.copytree(REPO / "configs", dst)
    c = yaml.safe_load((dst / "classical.yaml").read_text())
    c["models"]["rf"]["n_estimators"] = 10
    c["models"]["gnb"]["svd_components"] = 5
    (dst / "classical.yaml").write_text(yaml.safe_dump(c))


def _run(*args, env):
    res = subprocess.run(
        [sys.executable, "-m", "mirage", *args], env=env, capture_output=True, text=True, cwd=REPO
    )
    assert res.returncode == 0, res.stderr[-3000:]
    return res.stdout


def test_cpu_pipeline(tmp_path, monkeypatch):
    import os

    root, cfg = tmp_path / "root", tmp_path / "configs"
    _write_small_configs(cfg)
    data = root / "artifacts" / "data"
    data.mkdir(parents=True)
    from mirage.data import build_splits

    splits = build_splits(
        synthetic_sst(500),
        synthetic_sst(150, 1),
        synthetic_sst(150, 3).assign(text=lambda d: d["text"] + " again"),  # unseen
        synthetic_sst(200, 2),
        {
            "seed": 42,
            "heldout_frac": 0.1,
            "imdb_sample_size": 100,
            "imdb_head_words": 3,
            "imdb_tail_words": 1,
        },
    )
    for name, df in splits.items():
        df.to_parquet(data / f"{name}.parquet", index=False)

    env = {
        **os.environ,
        "MIRAGE_ROOT": str(root),
        "MIRAGE_CONFIGS": str(cfg),
        "PYTHONWARNINGS": "ignore",
    }
    _run("train-classical", "--variants", "base", "negmark", env=env)
    _run("build-suite", env=env)
    _run("predict", env=env)
    _run("evaluate", "--resamples", "50", env=env)
    _run("explain", "--skip-ig", env=env)
    _run("figures", env=env)

    preds = list((root / "results" / "predictions").glob("*.parquet"))
    assert len(preds) == 5 * 2 * 6  # models x variants x eval sets
    errors = pd.read_csv(root / "results" / "errors_by_type.csv")
    assert {"sst2_test", "sst2_neg", "beh_negation", "beh_typo"} <= set(errors["test"])
    control = pd.read_csv(root / "results" / "negation_control.csv")
    assert {"neg_error_all", "neg_error_known_words"} <= set(control.columns)
    assert (root / "results" / "word_errors.csv").exists()
    direction = pd.read_csv(root / "results" / "negation_direction.csv")
    assert {"negation_dir", "negation_dir_neg"} <= set(direction["test"])
    effect = pd.read_csv(root / "results" / "negmark_effect.csv")
    assert {"p_value", "p_holm", "significant"} <= set(effect.columns)
    assert (effect["p_holm"] >= effect["p_value"]).all()
    mc = pd.read_csv(root / "results" / "mcnemar.csv")
    assert "p_holm" in mc.columns
    assert (root / "results" / "explain" / "tree_depth4.txt").exists()
    assert (root / "figures" / "error_heatmap_base.png").exists()
