# Model card: Multinomial Naive Bayes

| | |
|---|---|
| **Type** | Probabilistic (generative) classifier |
| **Input** | Raw text → normalized → counts of words and word pairs (75,905 features) |
| **Output** | P(positive) from Bayes' rule over the word probabilities |
| **Size** | 151,812 learned numbers: P(word | class) for each feature and class, + 2 class priors |
| **Trained on** | SST-2 training split (laptop CPU, about 3 seconds) |
| **Code / file** | `models/classical.py` · `artifacts/models/mnb__base__seed42/model.joblib` |

Part of the Mirage project ([all models](../README.md)). Every number is from `results/`.

## How it works

For each class it learns how often each word appears in that class's training text. To classify,
it multiplies, for every word in the sentence, the probability of that word under "positive" and
under "negative", times each class's overall frequency, and picks the larger. "Naive" because it
assumes each word is independent of the others given the class, which is never true but often
works.

The consequence: every word votes with the full strength of its training statistics. A word that
appeared mostly in negative phrases (including a name, like a director whose films were mostly
panned) pushes hard towards negative, whatever the rest of the sentence says. The smoothing setting
(alpha) adds a small count to every word so unseen combinations don't get probability zero.

## Training and hyperparameters

Searched alpha ∈ {0.1, 0.5, 1.0}; **selected alpha = 1.0** (held-out 82.0%).

## Performance

| measure | value |
|---|---|
| SST-2 test accuracy / macro F1 | 0.834 / 0.833 |
| Real negated sentences (519) | 0.761 |
| IMDb, 10,000 reviews (other domain) | 0.760 |
| Calibration error (ECE; 0 = perfect), SST-2 test / IMDb | 0.122 / 0.229 |
| How well low confidence flags its IMDb errors (AUROC; 0.5 = chance) | 0.688 |
| Shortcut checks passed: negation pairs / real negation / typo pairs / name pairs | 35% / 76% / 49% / 61% |
| **Robustness / Mirage Gap** (accuracy − robustness) | **0.554 / 0.280** |

*Sources: `results/main_table.csv`, `results/mirage_gap.csv`, `results/early_warning.csv`.*

## Behaviour: what the shortcut tests found

- **The largest Mirage Gap of all eight models (0.280)**: same accuracy as logistic regression,
  19 points less robust.
- **It judges a film by who made it:** changing only an actor's or director's name flips its
  verdict in 38% of pairs. Across 35 film people, its reaction to each name follows that name's
  reputation in the training data at ρ = 0.91 on 23 names never tested before (finding 5).
- **Negation:** passes only 35% of negation pairs; "not" can't outvote a strong sentiment word.
- **Over-confident:** its probabilities pile up near 0 and 1, so on IMDb it is far worse calibrated
  (ECE 0.229) than logistic regression, the random forest or DistilBERT, and its confidence is a
  weaker warning signal (0.688).
- **Shares shortcut words with LR and RF** (0.88–0.92, H5).

## Intended use and limits

A quick text classifier when the vocabulary is stable. Not suitable where incidental words (names,
places, brands) carry label-correlated history in the training data. All results are on English movie reviews; sarcasm and other domains are untested.
