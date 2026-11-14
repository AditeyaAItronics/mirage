# Model card: Random forest

| | |
|---|---|
| **Type** | Ensemble of 300 decision trees |
| **Input** | Raw text → normalized → TF-IDF of words and word pairs (75,905 features) |
| **Output** | P(positive) = average of the 300 trees' leaf probabilities |
| **Size** | 10.7 million nodes in 300 trees (average depth 3,900; 5.4 million leaves) |
| **Trained on** | SST-2 training split (laptop CPU, about 4 minutes); 822 MB on disk |
| **Code / file** | `models/classical.py` · `artifacts/models/rf__base__seed42/model.joblib` |

Part of the Mirage project ([all models](../README.md)). Every number is from `results/`.

## How it works

Many decision trees, each trained on a random resample of the training data and, at every split,
allowed to look at only a random √75,905 ≈ 275 of the features. Each tree is a poor, over-fitted
classifier, but their mistakes differ, so averaging 300 of them is far more accurate and more
stable than any one tree. The averaged probability is also smoother than a single tree's.

## Training and hyperparameters

300 trees, √features per split, seed 42; searched max depth ∈ {30, unlimited}; **selected
unlimited** (held-out 77.2% against 63.0% at depth 30).

## Performance

| measure | value |
|---|---|
| SST-2 test accuracy / macro F1 | 0.754 / 0.753 |
| Real negated sentences (519) | 0.726 |
| IMDb, 10,000 reviews (other domain) | 0.703 |
| Calibration error (ECE; 0 = perfect), SST-2 test / IMDb | 0.068 / 0.057 |
| How well low confidence flags its IMDb errors (AUROC; 0.5 = chance) | 0.702 |
| Shortcut checks passed: negation pairs / real negation / typo pairs / name pairs | 69% / 73% / 52% / 83% |
| **Robustness / Mirage Gap** (accuracy − robustness) | **0.694 / 0.060** |

*Sources: `results/main_table.csv`, `results/mirage_gap.csv`, `results/early_warning.csv`.*

## Behaviour: what the shortcut tests found

- **The smallest Mirage Gap of the classical models (0.060)**, though its accuracy is only 75.4%.
- **Names:** 83% of name pairs right, between logistic regression and Naive Bayes.
- **Typos:** it leans negative when a sentiment word is lost (flips 59% of negative-sentence typo
  pairs, 18% of positive ones).
- **Best-calibrated classical model on IMDb** (ECE 0.057).
- **Shares shortcut words** with LR and MNB (0.88–0.89, H5).
- Has only one seed (shortfall C4); a different seed could move its numbers slightly.

## Intended use and limits

A robust general-purpose classifier on tabular-style features; large on disk and slow to train
on 75,905 features. All results are on English movie reviews; sarcasm and other domains are untested.
