# Model card: RoBERTa-large (SiEBERT, ready-made)

| | |
|---|---|
| **Type** | Large transformer, fine-tuned for sentiment by its authors (not by us) |
| **Input** | Raw text, as is (not normalized), word pieces up to 128 (IMDb: 512) |
| **Output** | P(positive) from its own 2-class head |
| **Size** | About 355 million parameters: 24 layers, 1,024 dimensions, 16 heads (5× DistilBERT) |
| **Source** | `siebert/sentiment-roberta-large-english`, fine-tuned on 15 sentiment datasets |
| **Code** | `models/external.py`; predictions on Kaggle |

Part of the Mirage project ([all models](../README.md)). Every number is from `results/`.

## How it works

RoBERTa is BERT's architecture trained longer, on more text and without BERT's next-sentence
task; "large" means 24 layers instead of 6. SiEBERT fine-tuned it on 15 English sentiment datasets
(reviews, tweets and more). It is used to ask whether scale removes shortcuts.

## Training and hyperparameters

Not trained by us; no settings chosen. Its training data very likely includes SST-2 and IMDb.

## Performance

| measure | value |
|---|---|
| SST-2 test accuracy / macro F1 | 0.946† / 0.946† |
| Real negated sentences (519) | 0.931† |
| IMDb, 10,000 reviews (other domain) | 0.956† |
| Calibration error (ECE; 0 = perfect), SST-2 test / IMDb | 0.052 / 0.043 |
| How well low confidence flags its IMDb errors (AUROC; 0.5 = chance) | 0.857 |
| Shortcut checks passed: negation pairs / real negation / typo pairs / name pairs | 100% / 93% / 71% / 98% |
| **Robustness / Mirage Gap** (accuracy − robustness) | **0.906 / 0.040** |

† possibly inflated: this model was very likely trained on SST-2 and IMDb (shortfall C12).

*Sources: `results/main_table.csv`, `results/mirage_gap.csv`, `results/early_warning.csv`.*

## Behaviour: what the shortcut tests found

- **The smallest Mirage Gap (0.040) and the highest robustness (0.906)**, but not zero.
- **Typos still break it, in the other direction:** it flips 42% of *negative* sentences with a
  typo (18% of positive ones); its default for an unreadable word leans positive.
- **Reads "I would never call the movie great" correctly** (fails 7%), where both DistilBERTs fail.
- **Possible contamination:** it is the only model *more* accurate on IMDb (0.956) than on SST-2
  (0.946), and again on fresh IMDb reviews (0.959); its accuracy is marked † everywhere. Its
  behaviour on our own template sentences is a fair test.

## Intended use and limits

Evaluation only, as the "does scale help?" comparison. All results are on English movie reviews; sarcasm and other domains are untested.
