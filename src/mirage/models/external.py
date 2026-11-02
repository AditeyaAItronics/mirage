"""Ready-made Hugging Face sentiment models, evaluated without training (D39).

They answer "does scale cure shortcuts?": the same eval sets and behavioral suite, the same
predictions contract, seed 0. Unlike our models they read raw text (no SST-style `normalize`),
because they were trained on ordinary English.
"""

from __future__ import annotations

import numpy as np

from mirage.models.distilbert import DistilBertClassifier

EXTERNAL_SEED = 0


def label_columns(id2label: dict) -> list[int]:
    """Output indices [negative, positive] read from the model's own config.

    Never assumes 0 = negative. Refuses anything but exactly one negative and one positive label
    (e.g. three-class models with "neutral", or unnamed LABEL_0 / LABEL_1).
    """
    names = {int(i): str(lab).lower() for i, lab in id2label.items()}
    neg = [i for i, n in names.items() if n.startswith("neg")]
    pos = [i for i, n in names.items() if n.startswith("pos")]
    if len(names) != 2 or len(neg) != 1 or len(pos) != 1:
        raise ValueError(f"need exactly one negative and one positive label, got {id2label}")
    return [neg[0], pos[0]]


class ExternalClassifier(DistilBertClassifier):
    """Any binary sequence classifier; reuses DistilBERT's batching and head + tail truncation."""

    normalize_input = False

    def __init__(
        self,
        name: str,
        model,
        tokenizer,
        max_length: int = 128,
        long_max_length: int = 512,
        batch_size: int = 32,
    ):
        super().__init__(model, tokenizer, max_length, long_max_length)
        self.name = name
        self.batch_size = batch_size
        self.columns = label_columns(model.config.id2label)

    def predict_proba(
        self, texts: list[str], long_text: bool = False, batch_size: int | None = None
    ) -> np.ndarray:
        proba = super().predict_proba(texts, long_text, batch_size or self.batch_size)
        return proba[:, self.columns]

    @classmethod
    def from_hub(
        cls, name: str, spec: dict, cfg: dict, local_only: bool = False
    ) -> ExternalClassifier:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        def fetch(local_only: bool):
            return (
                AutoModelForSequenceClassification.from_pretrained(
                    spec["hf_id"], local_files_only=local_only
                ),
                AutoTokenizer.from_pretrained(spec["hf_id"], local_files_only=local_only),
            )

        # Cache first: no network round trip, so a cached model loads offline (e.g. on stage)
        # and a flaky connection can't break it. Download only if it isn't cached yet.
        try:
            model, tokenizer = fetch(True)
        except OSError:
            if local_only:  # e.g. an optional analysis that shouldn't trigger a 1.4 GB download
                raise
            model, tokenizer = fetch(False)
        return cls(
            name,
            model,
            tokenizer,
            cfg["max_length"],
            cfg["long_max_length"],
            spec.get("batch_size", 32),
        )
