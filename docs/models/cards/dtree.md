# Model card: Decision tree

| | |
|---|---|
| **Type** | Single decision tree |
| **Input** | Raw text → normalized → TF-IDF of words and word pairs (75,905 features) |
| **Output** | P(positive) = share of positive training examples in the leaf the text reaches |
| **Size** | 4,089 nodes: depth 419, 2,045 leaves |
| **Trained on** | SST-2 training split (laptop CPU, about 25 seconds) |
| **Code / file** | `models/classical.py` · `artifacts/models/dtree__base__seed42/model.joblib` |

Part of the Mirage project ([all models](../README.md)). Every number is from `results/`.

## How it works

A tree of yes/no questions about single features ("does the text contain *bad* with weight above
0.1?"). Training picks, at each node, the question that best separates positive from negative
examples (Gini impurity), and keeps splitting until a leaf has too few examples. A text follows the
answers down to one leaf, and the prediction is that leaf's share of positive training examples.
The first four levels are drawn in `figures/tree_depth4.png`.

Because each question looks at one feature, a sentence is classified by a few words on one path;
most of its words are never looked at.

## Training and hyperparameters

Unlimited depth; searched minimum examples per leaf ∈ {10, 20, 50}; **selected 20** (held-out
65.6%). Leaves of 1 example were ruled out: they give only 0% / 100% probabilities and no usable
confidence (D29).

## Performance

| measure | value |
|---|---|
| SST-2 test accuracy / macro F1 | 0.653 / 0.652 |
| Real negated sentences (519) | 0.607 |
| IMDb, 10,000 reviews (other domain) | 0.612 |
| Calibration error (ECE; 0 = perfect), SST-2 test / IMDb | 0.144 / 0.200 |
| How well low confidence flags its IMDb errors (AUROC; 0.5 = chance) | 0.562 |
| Shortcut checks passed: negation pairs / real negation / typo pairs / name pairs | 37% / 61% / 45% / 62% |
| **Robustness / Mirage Gap** (accuracy − robustness) | **0.514 / 0.139** |

*Sources: `results/main_table.csv`, `results/mirage_gap.csv`, `results/early_warning.csv`.*

## Behaviour: what the shortcut tests found

- **Low accuracy (65.3%)** and a Mirage Gap of 0.139.
- **"Can't read, can't be fooled":** 10.8 points fewer errors on typo tests than normally, because
  losing one word rarely changes its path (finding 9).
- **Least reproducible word effects** of all models (split-half reliability 0.34), so it agrees only
  moderately with the others (0.44–0.64, H5).
- **Sensitive to wording:** on the second template suite its directional-negation failure rate
  fell from 20% to 3%, mostly because it gets the affirmative sentence of one new frame wrong, so
  few of those pairs are scored at all (finding 15).

## Intended use and limits

Interpretable on a small scale (a few levels can be drawn). Not accurate enough for sentiment. All results are on English movie reviews; sarcasm and other domains are untested.
