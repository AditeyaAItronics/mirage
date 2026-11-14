# Model card: DistilBERT (Hugging Face, ready-made)

| | |
|---|---|
| **Type** | Transformer, fine-tuned on SST-2 by Hugging Face (not by us) |
| **Input** | Raw text, as is (not normalized), word pieces up to 128 (IMDb: 512) |
| **Output** | P(positive) from its own 2-class head (label order read from its config) |
| **Size** | About 66 million parameters; same architecture as ours |
| **Source** | `distilbert/distilbert-base-uncased-finetuned-sst-2-english` |
| **Code** | `models/external.py`; predictions on Kaggle (`notebooks/kaggle_external.ipynb`) |

Part of the Mirage project ([all models](../README.md)). Every number is from `results/`.

## How it works

The same architecture as our DistilBERT (see that card), fine-tuned on SST-2 independently by
Hugging Face. It is used to check whether our DistilBERT's behaviour is ours alone or comes from the
data: an independent replication.

## Training and hyperparameters

Not trained by us; no settings chosen. Seed 0 is used in file names only.

## Performance

| measure | value |
|---|---|
| SST-2 test accuracy / macro F1 | 0.923 / 0.923 |
| Real negated sentences (519) | 0.892 |
| IMDb, 10,000 reviews (other domain) | 0.893 |
| Calibration error (ECE; 0 = perfect), SST-2 test / IMDb | 0.064 / 0.082 |
| How well low confidence flags its IMDb errors (AUROC; 0.5 = chance) | 0.845 |
| Shortcut checks passed: negation pairs / real negation / typo pairs / name pairs | 100% / 89% / 61% / 100% |
| **Robustness / Mirage Gap** (accuracy − robustness) | **0.876 / 0.047** |

*Sources: `results/main_table.csv`, `results/mirage_gap.csv`, `results/early_warning.csv`.*

## Behaviour: what the shortcut tests found

- **Replicates our DistilBERT almost exactly:** accuracy 0.923, Mirage Gap 0.047, the same
  typo default (flips 59% of positive typo pairs, finding 2) and the same "would never call"
  blind spot (99%).
- **It was trained on our held-out phrases** (its held-out accuracy is 0.989 against 0.923 on test,
  shortfall C14), so its held-out score is not a fair test; its SST-2 test score is.

## Intended use and limits

Evaluation only, as an independent second DistilBERT. All results are on English movie reviews; sarcasm and other domains are untested.
