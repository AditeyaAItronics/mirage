"""DistilBERT fine-tuning (run on a Kaggle/Colab T4) and CPU/GPU inference.

All torch / transformers imports are inside functions (majordecisions.md D2).
Inputs go through `normalize` exactly like the classical models, so every model sees SST-style text.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from mirage.text import normalize


def head_tail_ids(ids: list[int], budget: int, head_frac: float = 0.5) -> list[int]:
    """Keep `budget` tokens: the first ~head_frac of them from the start, the rest from the end."""
    if len(ids) <= budget:
        return ids
    head = int(budget * head_frac)
    return ids[:head] + ids[len(ids) - (budget - head) :]


def schedule(n_examples: int, step_batch: int, epochs: int, warmup_ratio: float) -> dict:
    """Optimizer steps and warm-up steps for a run whose steps each consume `step_batch` examples.

    `step_batch` must be the *effective* batch (per-device batch x devices x accumulation): with
    2 GPUs the Trainer splits each batch across both, so one epoch has half as many steps. Using
    the per-device batch instead doubled the warm-up share on Kaggle's T4 x2 (D34, D36).
    """
    steps = math.ceil(n_examples / step_batch) * epochs
    return {"total_steps": steps, "warmup_steps": int(warmup_ratio * steps)}


def _device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DistilBertClassifier:
    name = "distilbert"

    def __init__(self, model, tokenizer, max_length: int = 128, long_max_length: int = 512):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.long_max_length = long_max_length
        self.device = _device()
        self.model.to(self.device)

    def encode(self, texts: list[str], long_text: bool = False):
        texts = [normalize(t) for t in texts]
        if not long_text:
            return self.tokenizer(
                texts,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            )
        # Long reviews: keep the first and last tokens instead of cutting the ending, where the
        # verdict often is (shortfalls.md C1).
        ids = self.tokenizer(texts, add_special_tokens=False)["input_ids"]
        ids = [head_tail_ids(x, self.long_max_length - 2) for x in ids]
        tok = self.tokenizer
        batch = [{"input_ids": [tok.cls_token_id, *x, tok.sep_token_id]} for x in ids]
        return tok.pad(batch, padding=True, return_tensors="pt")

    def predict_proba(
        self, texts: list[str], long_text: bool = False, batch_size: int = 64
    ) -> np.ndarray:
        import torch

        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = self.encode(list(texts[i : i + batch_size]), long_text).to(self.device)
                logits = self.model(**batch).logits
                out.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 2))

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path: Path, max_length: int = 128, long_max_length: int = 512):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model = AutoModelForSequenceClassification.from_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(path)
        return cls(model, tokenizer, max_length, long_max_length)


def train(
    train_df: pd.DataFrame,
    heldout_df: pd.DataFrame,
    cfg: dict,
    seed: int,
    out_dir: Path,
) -> dict:
    """Fine-tune, keep the best epoch by held-out accuracy, save weights + training log."""
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(cfg["model_name"], num_labels=2)

    def to_ds(df: pd.DataFrame) -> Dataset:
        ds = Dataset.from_dict(
            {"text": [normalize(t) for t in df["text"]], "label": df["label"].tolist()}
        )
        return ds.map(
            lambda b: tokenizer(b["text"], truncation=True, max_length=cfg["max_length"]),
            batched=True,
            remove_columns=["text"],
        )

    train_ds, heldout_ds = to_ds(train_df), to_ds(heldout_df)

    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type="linear",
        warmup_steps=0,  # set below, once the Trainer has counted the devices
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["eval_batch_size"],
        num_train_epochs=cfg["epochs"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=cfg["fp16"] and torch.cuda.is_available(),
        seed=seed,
        report_to="none",
        logging_steps=100,
    )
    # Warm-up as explicit steps (`warmup_ratio` is being deprecated across transformers versions),
    # counted from the batch each optimizer step really sees: train_batch_size is per-device x
    # GPUs (DataParallel), world_size covers multi-process runs.
    step_batch = args.train_batch_size * args.world_size * args.gradient_accumulation_steps
    plan = schedule(len(train_ds), step_batch, cfg["epochs"], cfg["warmup_ratio"])
    args.warmup_steps = plan["warmup_steps"]

    def compute_metrics(pred) -> dict:
        return {"accuracy": float((pred.predictions.argmax(-1) == pred.label_ids).mean())}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=heldout_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()

    DistilBertClassifier(trainer.model, tokenizer).save(out_dir)
    shutil.rmtree(out_dir / "checkpoints", ignore_errors=True)  # best weights are saved above
    summary = {
        "seed": seed,
        "heldout": metrics,
        "devices": max(1, args.n_gpu) * args.world_size,
        "effective_batch": step_batch,
        **plan,
        "log_history": trainer.state.log_history,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
