# Model card: Logistic regression

| | |
|---|---|
| **Type** | Linear classifier (the project's baseline) |
| **Input** | Raw text → normalized → TF-IDF of words and word pairs (75,905 features) |
| **Output** | P(positive) = sigmoid(weighted sum of the features + bias) |
| **Size** | 75,906 learned numbers: one weight per feature + a bias |
| **Trained on** | SST-2 training split, 63,971 phrases (laptop CPU, about 3 seconds) |
| **Code / file** | `models/classical.py` · `artifacts/models/logreg__base__seed42/model.joblib` |

Part of the Mirage project ([all models](../README.md)). Every number is from `results/`.

## How it works

Each word or word pair gets one learned weight: positive for words that go with positive
reviews ("masterpiece"), negative for the opposite ("tedious"). A sentence's score is the sum of
the weights of the features it contains, squashed into a probability by the sigmoid function. TF-IDF
scales each feature down if it is common everywhere ("the", "movie") and up if it is distinctive.
The model has no notion of word order beyond the word pairs: "not good" is the pair feature
`not good`, which helps only if that exact pair was common enough in training.

L2 regularization (the C setting) keeps the weights small, so a rare word can't get an extreme
weight from a handful of examples. This is why it is far less fooled by names than Naive Bayes.

## Training and hyperparameters

Searched C ∈ {0.1, 1, 10, 30, 100} (smaller C = stronger shrinking); **selected C = 10**
(held-out 84.2%). Solver `liblinear`, L2 penalty, max 1,000 iterations, seed 42.

## Performance

| measure | value |
|---|---|
| SST-2 test accuracy / macro F1 | 0.835 / 0.834 |
| Real negated sentences (519) | 0.773 |
| IMDb, 10,000 reviews (other domain) | 0.795 |
| Calibration error (ECE; 0 = perfect), SST-2 test / IMDb | 0.016 / 0.081 |
| How well low confidence flags its IMDb errors (AUROC; 0.5 = chance) | 0.751 |
| Shortcut checks passed: negation pairs / real negation / typo pairs / name pairs | 70% / 77% / 51% / 99% |
| **Robustness / Mirage Gap** (accuracy − robustness) | **0.743 / 0.091** |

*Sources: `results/main_table.csv`, `results/mirage_gap.csv`, `results/early_warning.csv`.*

## Behaviour: what the shortcut tests found

- **Its accuracy twin is Multinomial NB** (83.5% vs 83.4%), but its robustness is 19 points
  higher: it passes 99% of name pairs against Naive Bayes's 61% (the Mirage Gap's headline).
- **Typos are its weak spot:** its typo error is 29.9 points above its normal error (H1). It simply
  loses a misspelled word (outside its vocabulary) and drifts towards 50% from both sides (it flips
  42% of positive and 49% of negative typo pairs).
- **Negation:** 70% of negation pairs; with negation marking (`negmark`) this rises to 96% and its
  Mirage Gap halves (0.091 → 0.040).
- **Shared shortcut words:** its word effects agree with the random forest and Multinomial NB at
  0.89–0.92 (H5), and it has learned "IMAX" as positive (removing it drops P(positive) by 0.21).
- **Best-calibrated model on SST-2** (ECE 0.016), less so on IMDb (0.081).

## Intended use and limits

A fast, readable baseline: its weights can be printed and inspected. Not suitable where text has
typos or unusual negation. All results are on English movie reviews; sarcasm and other domains are untested.
