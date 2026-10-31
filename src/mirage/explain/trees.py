"""Tree explanations: a readable depth-limited tree and forest importances."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.tree import export_text

from mirage.models.base import SklearnClassifier
from mirage.models.classical import build_pipeline


def fit_explain_tree(
    train: pd.DataFrame, cfg: dict, negmark: bool = False
) -> tuple[SklearnClassifier, str]:
    """Refit a depth-limited tree for the figure only; the full tree's accuracy is reported elsewhere."""
    pipe = build_pipeline("dtree", cfg, negmark, {"max_depth": cfg["explain_tree_depth"]})
    pipe.fit(train["text"].tolist(), train["label"].to_numpy())
    names = list(pipe.named_steps["vec"].get_feature_names_out())
    return SklearnClassifier("dtree_explain", pipe), export_text(
        pipe.named_steps["clf"], feature_names=names
    )


def top_importances(clf: SklearnClassifier, k: int = 25) -> pd.DataFrame:
    """Impurity-based importances for a fitted tree or forest."""
    vec, model = clf.pipeline.named_steps["vec"], clf.pipeline.named_steps["clf"]
    names, imp = vec.get_feature_names_out(), model.feature_importances_
    order = np.argsort(imp)[::-1][:k]
    return pd.DataFrame(
        {"feature": names[order], "score": imp[order], "direction": "importance", "model": clf.name}
    )
