# Model card: Gaussian Naive Bayes

| | |
|---|---|
| **Type** | Probabilistic classifier on compressed features |
| **Input** | Raw text → normalized → TF-IDF (75,905) → compressed to 300 numbers by SVD |
| **Output** | P(positive) from Bayes' rule, each of the 300 numbers modelled as a bell curve |
| **Size** | 22,772,702 learned numbers, almost all the 300 × 75,905 SVD projection |
| **Trained on** | SST-2 training split (laptop CPU, about 25 seconds) |
| **Code / file** | `models/classical.py` · `artifacts/models/gnb__base__seed42/model.joblib` |

Part of the Mirage project ([all models](../README.md)). Every number is from `results/`.

## How it works

Gaussian NB needs a short list of continuous numbers, not 75,905 sparse word features. So the TF-IDF
features are first compressed by truncated SVD (a form of principal-component analysis) into 300
components: each component is a weighted mix of thousands of words. For each class the model then
learns the mean and variance of every component, assumes each follows a bell curve (a Gaussian),
and classifies with Bayes' rule.

The compression is the problem: a single word's effect is smeared across 300 mixed components, and
word-count data is far from bell-shaped. The proposal's hypothesis H6 predicted exactly this.

## Training and hyperparameters

No hyperparameters searched (Gaussian NB's defaults; 300 SVD components fixed in
`configs/classical.yaml`). Held-out accuracy 56.3%.

## Performance

| measure | value |
|---|---|
| SST-2 test accuracy / macro F1 | 0.591 / 0.589 |
| Real negated sentences (519) | 0.522 |
| IMDb, 10,000 reviews (other domain) | 0.546 |
| Calibration error (ECE; 0 = perfect), SST-2 test / IMDb | 0.365 / 0.376 |
| How well low confidence flags its IMDb errors (AUROC; 0.5 = chance) | 0.547 |
| Shortcut checks passed: negation pairs / real negation / typo pairs / name pairs | 4% / 52% / 50% / 52% |
| **Robustness / Mirage Gap** (accuracy − robustness) | **0.395 / 0.196** |

*Sources: `results/main_table.csv`, `results/mirage_gap.csv`, `results/early_warning.csv`.*

## Behaviour: what the shortcut tests found

- **The weakest model:** 59.1% on SST-2 test against Multinomial NB's 83.4% (H6: supported,
  McNemar p ≈ 10⁻⁵⁶).
- **Negation:** passes only 4% of negation pairs.
- **"Can't read, can't be fooled":** it makes 25 points *fewer* errors on typo tests than on normal
  sentences, because it barely uses the sentiment word in the first place (finding 9). This is why
  the Mirage Gap requires both sentences of a pair to be right.
- **Its word effects agree with no other model** (0.18–0.26, H5).
- **Badly calibrated** (ECE 0.365 on SST-2) and its confidence barely flags errors (0.547).

## Intended use and limits

Included as the proposal's comparison with Multinomial NB. Not recommended for text. All results are on English movie reviews; sarcasm and other domains are untested.
