# Code explanations

A short tour of every folder and file in Mirage: what each one is for and what happens inside
it. For *why* things are built this way, see `majordecisions.md`; for the big-picture design, see
`ARCHITECTURE.md`.

**How the pieces connect:** `python -m mirage <stage>` runs one step of the pipeline. Steps talk
to each other only through files: data in `artifacts/data/`, models in `artifacts/models/`,
predictions in `results/predictions/`. Every analysis reads the prediction files, never the
models (the "predictions contract", D7).

```
prepare → train-classical / train-distilbert → build-suite → predict → evaluate → explain → warn → figures
```

---

## Top-level files

| File | What it is |
|---|---|
| `README.md` | Setup and the commands to reproduce everything. |
| `ARCHITECTURE.md` | The design: layers, interfaces, which code answers which research question. |
| `majordecisions.md` | Log of every major decision (D1–D32): what was decided, why, and what was rejected. |
| `shortfalls.md` | Known limitations, with severity, fixes and a "Resolved" list. |
| `whatyouneedtodo.md` | The steps only you can do (Kaggle, reviews, proposal fixes). |
| `MIRAGE_IMPLEMENTATION_PLAN.md` | The original week-by-week build plan. Some details are now superseded by the decision log. |
| `CLAUDE.md` | Rules for Claude Code when working in this repo. |
| `codeexplanations.md` | This file. |
| `pyproject.toml` | Package definition. Core dependencies (pandas, scikit-learn…); `[dl]` adds the deep-learning stack (torch, transformers, captum, sentence-transformers); `[dev]` adds pytest and ruff. Also holds the pytest and ruff settings. |
| `.gitignore` | Keeps the virtual environment, `artifacts/` (data and models) and prediction files out of git. |

---

## `configs/`: every setting, one YAML file per stage

| File | Controls |
|---|---|
| `data.yaml` | Dataset names (SST-2, the public SST-2 test labels, IMDb), held-out fraction (5%), IMDb sample size (10,000), the head/tail word cap (220 + 140), and the negation words. |
| `classical.yaml` | TF-IDF settings (unigrams + bigrams, min_df 2) and each classical model's tuning grid. |
| `distilbert.yaml` | Fine-tuning settings: 3 seeds, learning rate, batch size, epochs, max lengths (128 for SST-2, 512 for IMDb). |
| `behavioral.yaml` | How many cases each behavioral test gets, and the six name pairs for the name-shortcut test. |
| `warning.yaml` | Embedding model (MiniLM), distance type (Gaussian), unit (per-sentence max), CV folds, bootstrap resamples. |

---

## `src/mirage/`: the package

### Top-level modules

| File | What happens inside |
|---|---|
| `__init__.py` | Package docstring and version number. |
| `__main__.py` | Makes `python -m mirage` call `cli.main()`. |
| `cli.py` | The command line. One `stage_*` function per pipeline step: `prepare`, `train-classical`, `train-distilbert`, `build-suite`, `predict`, `evaluate`, `explain`, `warn`, `figures`, and `all`, which runs every step after training. `predict` skips models whose prediction files are newer than the model. `warn` also runs the length checks (D30). |
| `paths.py` | Every file location in one place (`artifacts/data`, `results/predictions`, `figures`…). `MIRAGE_ROOT` and `MIRAGE_CONFIGS` can move them (used by the tests and on Kaggle). |
| `config.py` | `load_config(name)` reads a YAML file from `configs/`; `save_resolved` writes the exact settings next to a model; `set_seed` seeds Python, NumPy and torch. |
| `text.py` | All text handling. `normalize` turns any text into SST-2 style (lowercase, "doesn't" → "does n't", punctuation split off, abbreviations kept whole). `negation_mark` adds `NOT_` to words after a negation, using either the original rule (up to punctuation) or the 3-word rule with exceptions. `sentences` splits a review into sentences. `Preprocessor` bundles these so they run inside the classical models. |
| `data.py` | Builds every dataset. `prepare` downloads SST-2, the public SST-2 test labels and IMDb, then: splits held-out by sentence so no phrase leaks (`heldout_mask`, D21); drops test sentences already seen in training; builds the 519-sentence negation set; samples and caps 10,000 IMDb reviews; builds `imdb_edges` (first + last sentence). Everything is saved as parquet; `load(name)` reads it back. |
| `features.py` | The TF-IDF and count vectorizers for the classical models. Stop-word removal and the default tokenizer are turned off, because both would delete "not" and "n't". |
| `predict.py` | The predictions contract. Defines the required columns, validates every prediction file, writes one parquet per (model, variant, seed, dataset), and loads them back with those four as columns. |
| `metrics.py` | Single-number metrics: accuracy, macro-F1, confidence, reliability bins, expected calibration error (ECE), AUROC. |
| `stats.py` | Uncertainty. Bootstrap confidence intervals and p-values, paired differences, the "gap closed" ratio, paired AUROC difference, Holm correction, McNemar's test. |
| `evaluate.py` | Builds every results table from the prediction files: main table, errors by test type, negation gap to DistilBERT, negation-marking effect, negation control, directional negation, IMDb error by length, length-vs-content split, name effect, McNemar, per-word errors. Writes them all to `results/*.csv`. |
| `plots.py` | Every figure (PNG + PDF): error heatmap, negation-gap bars, error vs score, error by review length, reliability diagram, explanation-overlap heatmap, the small decision tree, and word-attribution strips. Uses a fixed, colour-blind-checked palette. |

### `behavioral/`: the hand-built test sentences (CheckList-style)

These tests check *how* a model behaves, using sentences we write ourselves rather than dataset
sentences.

| File | What happens inside |
|---|---|
| `wordlists.py` | The vocabulary: 20 positive and 20 negative adjectives, 15 movie nouns, mild adjectives and "work" nouns for the name test, Indian and Western first names (paired by gender), and sentiment verbs. |
| `templates.py` | Turns word lists into test cases. Each generator writes sentences from patterns ("The {noun} was {adj}."): `simple`, `negation` ("not great" should be negative), `negation_dir` (the "X / not X" pairs), `typo`, `name_swap` (unseen names) and `name_shortcut` (Kidman vs Seagal). Each row records its test type, case id, role (original/perturbed) and the adjective used. |
| `names.py` | Picks the film people for the name test from the *training labels*: for each surname, how positive its training phrases are (smoothed log-odds). `pick_pairs` pairs the most positive names with the most negative. |
| `suite.py` | Builds the whole suite (5,250 rows, 3,250 cases) from the config, saves it to `results/behavioral_suite.csv`, and scores predictions. Single-sentence tests fail when the label is wrong; invariance pairs fail when the label changes; direction pairs fail when adding "not" doesn't change the label. |

### `models/`: the classifiers

| File | What happens inside |
|---|---|
| `__init__.py` | Re-exports the model names, the `Classifier` interface and `load_classifier`. |
| `base.py` | The one interface every model follows: `predict_proba(raw texts)` returns P(negative) and P(positive). `SklearnClassifier` wraps a scikit-learn pipeline. `load_classifier` loads either kind of saved model from its folder. |
| `classical.py` | The five classical models: logistic regression, decision tree, random forest, multinomial Naive Bayes, Gaussian Naive Bayes on 300 SVD components. `build_pipeline` assembles vectorizer + model for a variant (`base`, `negmark`, `negmark_w3`). `tune` tries every grid setting and keeps the best on held-out accuracy. `train_all` does this for every model. |
| `distilbert.py` | DistilBERT. `train` fine-tunes it with Hugging Face's Trainer (on Kaggle's GPU), keeps the best epoch and deletes checkpoints. `DistilBertClassifier` predicts. Long IMDb reviews keep their first and last tokens rather than being cut off (`head_tail_ids`). |

### `explain/`: what each model relies on

| File | What happens inside |
|---|---|
| `linear.py` | Reads explanations directly off the models: LR's largest weights, Naive Bayes's word log-ratios, and LR's per-word contribution for one sentence. |
| `trees.py` | Fits a depth-4 tree just for the figure, and lists random forest / tree feature importances. |
| `ig.py` | Integrated Gradients for DistilBERT (via Captum): how much each word pushed the prediction, summed from word pieces back to whole words. |
| `deletion.py` | Faithfulness check. Delete the top-attributed words and see whether the prediction drops more than when deleting random words. |
| `overlap.py` | Do different models rely on the same words? Compares their top-word rankings with Jaccard overlap and rank-biased overlap (H5). |

### `warning/`: can we flag a model's mistakes in advance? (H4)

| File | What happens inside |
|---|---|
| `embed.py` | Turns text into MiniLM sentence embeddings, cached on disk by item id. `split_sentences` breaks reviews into sentences. |
| `gaussian.py` | Fits a Gaussian to the SST-2 training embeddings and scores new text by Mahalanobis distance: "how unusual is this?". |
| `knn.py` | Alternative distance: 1 − average cosine similarity to the 10 nearest training phrases. |
| `distance.py` | Chooses the distance and the unit from the config. Scores each review by its most unusual sentence. Also provides the length-matched sample and the length-adjusted distance (D30). |
| `error_detection.py` | The H4 comparison. How well does distance, low confidence, the two combined, or length predict which IMDb reviews a model gets wrong (AUROC)? Each is compared head-to-head with confidence. Also the sanity check (can distance tell SST-2 from IMDb?) and error-by-decile tables for the figure. |

---

## `tests/`: 89 automated tests (`.venv/Scripts/python -m pytest`)

They run on small synthetic data, need no downloads or GPU, and finish in about 30 seconds.

| File | Checks |
|---|---|
| `conftest.py` | Shared helper that makes small fake SST-style datasets. |
| `test_text.py` | Normalization cases (contractions, abbreviations, numbers), both negation-marking rules and their exceptions. |
| `test_features.py` | The vectorizers keep "not" / "n't" and `NOT_` features, and pickle correctly. |
| `test_data.py` | The held-out split never leaks; sentences keep their phrases; test sentences seen in training are dropped; IMDb capping and edges. |
| `test_contract.py` | Prediction files are validated and broken ones are rejected. |
| `test_behavioral.py` | Suite counts, determinism, pair structure, typo and direction rules, name pairs differ only by the name, gender-matched names, name selection. |
| `test_metrics_stats.py` | ECE, AUROC, bootstrap, McNemar against statsmodels, Holm, p-values, length table, name effect. |
| `test_explain_warning.py` | Overlap measures, deletion check, Mahalanobis against scipy, kNN, sentence splitting, length adjustment, the out-of-fold combiner. |
| `test_pipeline_e2e.py` | Runs the whole CPU pipeline through the real command line on fake data, in a temporary folder, and checks every output file exists. |

---

## Other folders

| Folder | What's in it |
|---|---|
| `notebooks/` | `kaggle_distilbert.ipynb`: clones the repo on Kaggle, installs it, prepares data, fine-tunes DistilBERT (3 seeds), predicts on the GPU, and zips models + predictions for you to download. |
| `results/` | Every output table (`*.csv`), the behavioral suite, `explain/` (top features, the depth-4 tree, overlap), and `predictions/` (one parquet per model × variant × seed × dataset; git-ignored). |
| `figures/` | Every figure as PNG + PDF, redrawn by `python -m mirage figures`. |
| `artifacts/` (git-ignored) | Large regenerable files: `data/` (parquet datasets), `models/` (about 2 GB, mostly the random forests), `emb/` (embedding caches), plus scratch logs. |
| `report/` | Empty placeholder for the final report. |

---

## Bugs and issues found while writing this

1. **🔴 Kaggle notebook can silently run old code.** Cell 2 clones the repo only `if not
   os.path.isdir("mirage")`. In an interactive Kaggle session where the repo was already cloned
   (for example your earlier smoke test), re-running the notebook **keeps the old code**: the
   leaky held-out split, the 256-token IMDb cut, no `imdb_edges`. A committed run ("Save & Run
   All") starts in a fresh environment and is fine. **Fix:** `git pull` when the folder exists,
   or restart the session first.
2. **🟠 The notebook's smoke test is no longer about 2 minutes.** `SMOKE_TEST` limits training to
   2,000 examples, but cell 5 still predicts on every evaluation set, including 10,000 IMDb
   reviews at 512 tokens. That adds about 5 minutes per seed. The comment in cell 1 is out of date.
3. **🟢 ECE uses about 8 bins, not 15.** Confidence (the larger of P(pos) and P(neg)) is always at
   least 0.5, so the 7 bins below 0.5 are always empty. ECE is still computed correctly over the
   occupied bins, but "15 bins" overstates the resolution (related to shortfall G3).
4. **🟢 `set_seed` sets `PYTHONHASHSEED`, which has no effect on a running process.** Python reads
   it only at startup. Harmless, because nothing here depends on string-hash order, but the line
   doesn't do what it appears to.
5. **🟢 `paths.ROOT` assumes an editable install** (`pip install -e`). With a normal install it
   would point inside `site-packages`. The README and notebook both use `-e`, so this doesn't
   bite today.
6. **🟢 The negation gap to DistilBERT (table and figure) only covers `base` and `negmark`**, not
   the new `negmark_w3` variant. The within-model effect of both variants is in
   `negmark_effect.csv`; extending the gap table is a few lines once DistilBERT's predictions exist.
