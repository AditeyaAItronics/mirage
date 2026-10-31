"""Integrated Gradients for DistilBERT (Sundararajan et al., 2017) via Captum.

Attributions are computed over word-piece embeddings and summed back to the *normalized
whitespace words* - the same words the classical models see - by tokenizing with
is_split_into_words=True and grouping on word_ids() (majordecisions.md D14).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mirage.text import normalize


def attribute(
    clf, texts: list[str], n_steps: int = 50, internal_batch_size: int = 16
) -> pd.DataFrame:
    """One row per text: words, word attributions toward the predicted class, p_pred, delta."""
    import torch
    from captum.attr import LayerIntegratedGradients

    model, tok, device = clf.model, clf.tokenizer, clf.device

    def forward(input_ids, attention_mask):
        return torch.softmax(model(input_ids=input_ids, attention_mask=attention_mask).logits, -1)

    lig = LayerIntegratedGradients(forward, model.distilbert.embeddings.word_embeddings)
    rows = []
    for text in texts:
        words = normalize(text).split()
        enc = tok(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=clf.max_length,
            return_tensors="pt",
        )
        ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
        # Baseline: same length, [CLS] ... [SEP] kept, everything else [PAD].
        base = torch.full_like(ids, tok.pad_token_id)
        base[0, 0], base[0, -1] = tok.cls_token_id, tok.sep_token_id
        with torch.no_grad():
            probs = forward(ids, mask)[0]
        target = int(probs.argmax())
        attr, delta = lig.attribute(
            ids,
            baselines=base,
            additional_forward_args=(mask,),
            target=target,
            n_steps=n_steps,
            internal_batch_size=internal_batch_size,
            return_convergence_delta=True,
        )
        piece = attr.sum(-1)[0].detach().cpu().numpy()
        word_attr = np.zeros(len(words))
        for pos, w in enumerate(enc.word_ids(0)):
            if w is not None:
                word_attr[w] += piece[pos]
        rows.append(
            {
                "text": text,
                "words": words,
                "attributions": word_attr.tolist(),
                "pred": target,
                "p_pred": float(probs[target]),
                "delta": float(delta.abs().item()),
            }
        )
    return pd.DataFrame(rows)


def mean_abs_attribution(attr_df: pd.DataFrame, min_count: int = 2) -> pd.Series:
    """Rank words by mean |attribution| across texts (for the cross-model overlap, H5)."""
    pairs = [
        (w, abs(a))
        for words, attrs in zip(attr_df["words"], attr_df["attributions"], strict=True)
        for w, a in zip(words, attrs, strict=True)
    ]
    df = pd.DataFrame(pairs, columns=["word", "abs_attr"])
    g = df.groupby("word")["abs_attr"].agg(["mean", "size"])
    return g[g["size"] >= min_count]["mean"].sort_values(ascending=False)
