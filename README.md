# Mirage: Exposing the Illusion of Accuracy

Detecting shortcut learning in sentiment classifiers — logistic regression, decision tree, random
forest, Multinomial and Gaussian Naive Bayes and DistilBERT — with behavioral tests, negation
marking, explanations and a Gaussian early-warning score on IMDb.

> **Status:** every experiment has run on the real data (classical models locally, DistilBERT
> on Kaggle). The numbers below are final; the hypothesis verdicts belong in the report.

## Results

Base variants; DistilBERT is the mean of 3 seeds. Full tables are in `results/`, figures in
`figures/`, and every number with its context is in [majordecisions.md](majordecisions.md)
(D35–D38).

| model | SST-2 test | real negated sentences (519) | IMDb (10,000) | ECE on IMDb |
|---|---|---|---|---|
| **DistilBERT** | **0.922** | **0.897** | **0.896** | 0.051 |
| Logistic regression | 0.835 | 0.773 | 0.795 | 0.081 |
| Multinomial NB | 0.834 | 0.761 | 0.760 | 0.229 |
| Random forest | 0.754 | 0.726 | 0.703 | 0.057 |
| Decision tree | 0.653 | 0.607 | 0.612 | 0.200 |
| Gaussian NB | 0.591 | 0.522 | 0.546 | 0.376 |

**Behavioral tests: error rate on hand-built cases**

| model | negation ("X / not X") | typo in sentiment word | typo in noun | name shortcut (Kidman ↔ Seagal) |
|---|---|---|---|---|
| DistilBERT | 0.000 | 0.394 | 0.000 | 0.000 |
| Logistic regression | 0.277 | 0.454 | 0.040 | 0.008 |
| Multinomial NB | 0.572 | 0.386 | 0.102 | 0.376 |
| Random forest | 0.247 | 0.366 | 0.114 | 0.024 |

**What the results show**
- **Hidden failures (H1):** classical models that score 65–84% on SST-2 fail 20–57% of simple
  negation pairs (Gaussian NB 96%). Naive Bayes flips its answer in 38% of sentences when only a
  film person's name changes.
- **Negation gap (H2):** every classical model is significantly worse than DistilBERT on real
  negated sentences, by 12.7–37.8 points.
- **Negation marking (H3):** `NOT_` marking closes most of the gap on templates (96% for LR) but
  significantly closes it on real sentences only for Gaussian NB. Real negations are longer-range,
  idiomatic and sometimes doubly negated (D38).
- **Early warning (H4):** low confidence flags IMDb errors well (DistilBERT AUROC 0.836). Embedding
  distance doesn't (0.53), even though it separates the two datasets (0.93, mostly via length).
- **Shared shortcuts (H5):** DistilBERT relies on almost entirely different words from the
  classical models (top-50 overlap ≤ 0.08). Its IG explanations are faithful (deleting top words
  hurts 3× more than random ones).
- **DistilBERT's own shortcut:** when it can't read the sentiment word (a typo), it defaults to
  *negative*. "brilliant" scores 0.999; "brililant" scores 0.010. A typo in the noun changes
  nothing. Fragmented words lean negative in SST-2's training data (D37).

![Error rate by model and test](figures/error_heatmap_base.png)

## Setup (Windows, Python 3.10+)

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"                                        # CPU pipeline + tests
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install -e ".[dl]"                                         # datasets, DistilBERT, IG, embeddings
```

## Reproduce

```bash
python -m mirage prepare                                  # download SST-2 (+ public test labels) + IMDb, build splits
python -m mirage train-classical --variants base negmark negmark_w3  # 5 models x 3 variants
# DistilBERT: run notebooks/kaggle_distilbert.ipynb on a free T4 GPU,
# then unzip distilbert_outputs.zip into the repo root (models + DistilBERT predictions)
python -m mirage all                                      # suite -> predict -> evaluate -> explain -> warn -> figures
```

Each stage can also run on its own: `python -m mirage --help`.

## Layout

See [ARCHITECTURE.md](ARCHITECTURE.md) for the design and [majordecisions.md](majordecisions.md)
for why each choice was made.

| Folder | Contents |
|---|---|
| `src/mirage/` | the package |
| `configs/` | every setting, one YAML per stage |
| `results/` | CSV outputs, including the reusable `behavioral_suite.csv` |
| `figures/` | report figures (PNG + PDF) |
| `notebooks/` | Kaggle notebook for DistilBERT fine-tuning |
| `tests/` | `python -m pytest` (about 30 s, no GPU needed) |

For a file-by-file tour of the code, see [codeexplanations.md](codeexplanations.md).
