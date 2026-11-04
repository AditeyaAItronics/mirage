# Mirage: Exposing the Illusion of Accuracy

Detecting shortcut learning in sentiment classifiers — logistic regression, decision tree, random
forest, Multinomial and Gaussian Naive Bayes and DistilBERT — with behavioral tests, negation
marking, explanations and a Gaussian early-warning score on IMDb. Two ready-made models
(RoBERTa-large and Hugging Face's SST-2 DistilBERT) test whether scale cures shortcuts, and a live
demo shows the shortcuts sentence by sentence.

> **Status:** every experiment has run on the real data: classical models locally, DistilBERT and
> the two external models on Kaggle. The numbers below are final.

## Results

**The results in full, with every number and confidence interval, are in two files:**
- [hypothesisanswers.md](hypothesisanswers.md): H1–H6, each with its test, tables, verdict, reasons
  and caveats;
- [interestingfindings.md](interestingfindings.md): 13 further findings, each tied to the
  hypothesis it explains or extends.

This section is the short version. Base variants; DistilBERT is the mean of 3 seeds; all
tables are in `results/` and all figures in `figures/`.

| model | SST-2 test | real negated sentences (519) | IMDb (10,000) | robustness (D40) |
|---|---|---|---|---|
| **DistilBERT** | **0.922** | **0.897** | **0.896** | **0.876** |
| Logistic regression | 0.835 | 0.773 | 0.795 | 0.743 |
| Multinomial NB | 0.834 | 0.761 | 0.760 | 0.554 |
| Random forest | 0.754 | 0.726 | 0.703 | 0.694 |
| Decision tree | 0.653 | 0.607 | 0.612 | 0.514 |
| Gaussian NB | 0.591 | 0.522 | 0.546 | 0.395 |
| *DistilBERT (Hugging Face), external* | *0.923* | *0.892* | *0.893* | *0.876* |
| *RoBERTa-large (SiEBERT), external* | *0.946†* | *0.931†* | *0.956†* | *0.906* |

† possibly contaminated: SiEBERT was very likely trained on SST-2 and IMDb (C12). Its behavioral
results are clean, because the test sentences are ours. Robustness is the share of shortcut tests
a model gets fully right; it uses DistilBERT's median seed.

| | Hypothesis | Verdict |
|---|---|---|
| H1 | Every model, DistilBERT included, errs more on negation and typos | **Partly supported:** typos yes (DistilBERT +32 points); negation only for the bag-of-words models |
| H2 | DistilBERT handles negation much better than single-word models | **Supported:** 14–35 points better on real negated sentences |
| H3 | Word pairs + negation marking close part, not all, of the gap | **Supported for logistic regression:** 40% [21, 58] on real sentences; 25–98% on templates for every model |
| H4 | Embedding distance adds information beyond confidence | **Rejected:** distance AUROC ≈ 0.5, confidence 0.55–0.86 |
| H5 | Classical models share their shortcut words | **Supported for LR, random forest, Multinomial NB** (0.88–0.92, reliability-corrected) |
| H6 | Gaussian NB is worse than Multinomial NB | **Supported:** 59.1% vs 83.4% accuracy |

**Beyond the hypotheses**, for example: logistic regression and Multinomial NB have the same
accuracy but very different behaviour; DistilBERT treats an unreadable sentiment word as
"negative" ("brilliant" 0.999 → "brililant" 0.010), and Hugging Face's DistilBERT does too; scale
shrinks shortcuts but doesn't remove them; Naive Bayes judges films by who made them (ρ = 0.91 on
23 never-tested names). No conclusion depends on DistilBERT's seed (D44).

### The Mirage Gap (D40)

Accuracy against the robustness score: every model behaves worse than it looks.

![The Mirage Gap](figures/mirage_gap.png)

Other figures: `error_heatmap_base.png` (errors by model and test), `negation_gap_sst2_neg.png`
(the H2/H3 ladder), `h5_word_effect_agreement.png` (H5) and `shortcut_fingerprint.png` (per-model
detail); the Mirage Gap and fingerprint also come in 16:9 `_slide` versions.

## Live demo: the Shortcut Explorer (D41)

Type a sentence, or pick a curated pair, and see every model's verdict, which words drove it, and
two warning flags. Curated pairs include Binoche vs Hopkins (Naive Bayes flips), brilliant vs
brililant (our DistilBERT flips) and a sarcasm example where our models fail.

```bash
.venv/Scripts/python -m pip install -e ".[dl,demo]"
```
```bash
.venv/Scripts/python -m streamlit run app/shortcut_explorer.py
```

The presenter guide is [app/PRESENTING.md](app/PRESENTING.md); fallback images are in `app/screenshots/`.

## Setup (Windows, Python 3.10+)

```bash
python -m venv .venv
# exact versions of the final results: requirements-lock.txt (see its header)
.venv/Scripts/python -m pip install -e ".[dev]"                                        # CPU pipeline + tests
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install -e ".[dl]"                                         # datasets, DistilBERT, IG, embeddings
```

## Reproduce

```bash
python -m mirage prepare                                  # download SST-2 (+ public test labels) + IMDb, build splits
python -m mirage train-classical --variants base negmark negmark_w3 unigram  # 5 models x 4 variants
# DistilBERT: run notebooks/kaggle_distilbert.ipynb on a free T4 GPU,
# then unzip distilbert_outputs.zip into the repo root (models + DistilBERT predictions)
# External models (optional, D39): run notebooks/kaggle_external.ipynb on Kaggle,
# then unzip external_outputs.zip into the repo root (predictions only)
python -m mirage all                                      # suite -> predict -> evaluate -> explain -> occlusion -> warn -> figures
```

Each stage can also run on its own: `python -m mirage --help`.

## Layout

See [proposal.md](proposal.md) for the research questions and hypotheses,
[hypothesisanswers.md](hypothesisanswers.md) and [interestingfindings.md](interestingfindings.md)
for the results in full,
[ARCHITECTURE.md](ARCHITECTURE.md) for the design and [majordecisions.md](majordecisions.md)
for why each choice was made.

| Folder | Contents |
|---|---|
| `src/mirage/` | the package |
| `configs/` | every setting, one YAML per stage |
| `results/` | CSV outputs, including the reusable `behavioral_suite.csv` |
| `figures/` | report figures (PNG + PDF) |
| `notebooks/` | Kaggle notebooks: DistilBERT fine-tuning, and the external models' predictions |
| `app/` | the live demo (Streamlit), its curated examples and presenter guide |
| `docs/archive/` | finished plans kept as a record (the original build plan, the presentation-additions design) |
| `tests/` | `python -m pytest` (136 tests, about 2–3 min, no GPU needed) |

For a file-by-file tour of the code, see [codeexplanations.md](codeexplanations.md).
