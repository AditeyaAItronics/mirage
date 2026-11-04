# Code explanations

A short tour of every folder and file in Mirage: what each one is for and what happens inside
it. For *why* things are built this way, see `majordecisions.md`; for the big-picture design, see
`ARCHITECTURE.md`.

**How the pieces connect:** `python -m mirage <stage>` runs one step of the pipeline. Steps talk
to each other only through files: data in `artifacts/data/`, models in `artifacts/models/`,
predictions in `results/predictions/`. Every analysis reads the prediction files, never the
models (the "predictions contract", D7).

```
prepare → train-classical / train-distilbert → build-suite → predict (+ predict-external) → evaluate → explain → warn → figures
```

Next to the pipeline sits the **live demo** (`app/`), which loads the saved models directly and
never writes results.

---

## Top-level files

| File | What it is |
|---|---|
| `README.md` | Setup and the commands to reproduce everything. |
| `ARCHITECTURE.md` | The design: layers, interfaces, which code answers which research question. |
| `majordecisions.md` | Log of every major decision (D1–D41): what was decided, why, and what was rejected. |
| `shortfalls.md` | Known limitations, with severity, fixes and a "Resolved" list. |
| `whatyouneedtodo.md` | Every step for you, in order: what's left now, the report's walkthrough (yours and Claude's steps), optional extras, and a record of what's done. |
| `reportplan.md` | The final report's skeleton: each section, what goes in it, and where its numbers come from (links into the two results files). |
| `hypothesisanswers.md` | H1–H6 answered in full: the claim, how it was tested, every number with its CI, the verdict, why, caveats and robustness checks. Source material for the report. |
| `interestingfindings.md` | 13 findings beyond the hypotheses, each tied to the hypothesis it explains or extends, marked planned or exploratory. Source material for the report. |
| `proposal.md` | The project proposal: research questions, hypotheses H1–H6, datasets, models and evaluation plan. Its facts were corrected to match the code (Nov 2026); the hypotheses are as originally written. |
| `docs/archive/` | Finished plans, kept as a record: `MIRAGE_IMPLEMENTATION_PLAN.md` (the original week-by-week build plan) and `newadditionsdesign.md` (the design for the external models, the Mirage Gap and the demo, D39–D41). Superseded by the decision log where they differ. |
| `CLAUDE.md` | Rules for Claude Code when working in this repo. |
| `requirements-lock.txt` | Exact library versions of the environment that produced the final results (D46). |
| `codeexplanations.md` | This file. |
| `pyproject.toml` | Package definition. Core dependencies (pandas, scikit-learn…); `[dl]` adds the deep-learning stack (torch, transformers, captum, sentence-transformers); `[demo]` adds Streamlit for the live demo; `[dev]` adds pytest and ruff. Also holds the pytest and ruff settings. |
| `.streamlit/config.toml` | The demo's look: light theme (projector-friendly), no usage statistics. |
| `.claude/launch.json` | Lets Claude Code start the demo in its browser pane. |
| `.gitignore` | Keeps the virtual environment, `artifacts/` (data and models) and prediction files out of git. |

---

## `configs/`: every setting, one YAML file per stage

| File | Controls |
|---|---|
| `data.yaml` | Dataset names (SST-2, the public SST-2 test labels, IMDb), held-out fraction (5%), IMDb sample size (10,000), the head/tail word cap (220 + 140), and the negation words. |
| `classical.yaml` | TF-IDF settings (unigrams + bigrams, min_df 2) and each classical model's tuning grid. |
| `distilbert.yaml` | Fine-tuning settings: 3 seeds, learning rate, batch size, epochs, max lengths (128 for SST-2, 512 for IMDb). |
| `behavioral.yaml` | How many cases each behavioral test gets, the six name pairs for the name-shortcut test, and the name probe's settings (sentences per name, models, D45). |
| `explain.yaml` | The H5 settings (D43): which sentences (SST-2 test), which models, the minimum word count (5, and 10 as a check), top-k (50), reliability splits (20), bootstrap resamples and the seed. |
| `warning.yaml` | Embedding model (MiniLM), distance type (Gaussian), unit (per-sentence max), CV folds, bootstrap resamples. |
| `external.yaml` | The two ready-made Hugging Face models (RoBERTa-large, HF's SST-2 DistilBERT): their ids, batch sizes, token limits, and which eval sets they predict (D39). |
| `demo.yaml` | The live demo: which models it shows, which light mode leaves out, the two flag thresholds (0.70 confidence, 95th percentile distance) and the 60-word input limit (D41). |

---

## `src/mirage/`: the package

### Top-level modules

| File | What happens inside |
|---|---|
| `__init__.py` | Package docstring and version number. |
| `__main__.py` | Makes `python -m mirage` call `cli.main()`. |
| `cli.py` | The command line. One `stage_*` function per pipeline step: `prepare`, `train-classical`, `train-distilbert`, `build-suite`, `predict`, `evaluate`, `explain`, `warn`, `figures`, and `all`, which runs every step after training. `predict` skips models whose prediction files are newer than the model. `predict-external` runs the ready-made models and skips files that already exist (D39). `occlusion` ranks words the same way for every model for H5 (D43). `name-probe` builds and predicts the name probe (D45; external models only if cached locally). `warn` also runs the length checks (D30). `figures` also draws the Mirage Gap and the shortcut fingerprint (D40). |
| `paths.py` | Every file location in one place (`artifacts/data`, `results/predictions`, `figures`…). `MIRAGE_ROOT` and `MIRAGE_CONFIGS` can move them (used by the tests and on Kaggle). |
| `config.py` | `load_config(name)` reads a YAML file from `configs/`; `save_resolved` writes the exact settings next to a model; `set_seed` seeds Python, NumPy and torch. |
| `text.py` | All text handling. `normalize` turns any text into SST-2 style (lowercase, "doesn't" → "does n't", punctuation split off, abbreviations kept whole). `negation_mark` adds `NOT_` to words after a negation, using either the original rule (up to punctuation) or the 3-word rule with exceptions. `sentences` splits a review into sentences. `Preprocessor` bundles these so they run inside the classical models. |
| `data.py` | Builds every dataset. `prepare` downloads SST-2, the public SST-2 test labels and IMDb, then: splits held-out by sentence so no phrase leaks (`heldout_mask`, D21); drops test sentences already seen in training; builds the 519-sentence negation set; samples and caps 10,000 IMDb reviews; builds `imdb_edges` (first + last sentence). Everything is saved as parquet; `load(name)` reads it back. |
| `features.py` | The TF-IDF and count vectorizers for the classical models. Stop-word removal and the default tokenizer are turned off, because both would delete "not" and "n't". |
| `predict.py` | The predictions contract. Defines the required columns, validates every prediction file, writes one parquet per (model, variant, seed, dataset), and loads them back with those four as columns. |
| `metrics.py` | Single-number metrics: accuracy, macro-F1, confidence, reliability bins, expected calibration error (ECE), AUROC. |
| `stats.py` | Uncertainty. Bootstrap confidence intervals and p-values, paired differences, differences between two independent groups (`diff_independent_ci`, D47), the "gap closed" ratio, paired AUROC difference, Holm correction, McNemar's test. |
| `evaluate.py` | Builds every results table from the prediction files: main table, errors by test type, negation gap to DistilBERT along the ladder single words → + word pairs → + negation marking (D42), negation-marking effect, negation control, directional negation, IMDb error by length, length-vs-content split, name effect, typo direction, McNemar, per-word errors, and the Mirage Gap (`mirage_gap`: accuracy vs a robustness score built from pair accuracy on four shortcut tests, plus its leave-one-out sensitivity check; `fingerprint_table` adds calibration and early warning, D40). External models and the single-word ablation get their own Holm families (`holm_family`), so they don't change earlier p-values. Also `h1_test`, the formal H1 test: each negation and typo test against the SST-2 test sentences without negation, with a CI and Holm-corrected p (D47). Also `name_probe_effects` / `name_probe` (does a model's reaction to a name track its training reputation, with the effect size, D45), and `seed_robustness`: every result that uses DistilBERT's median seed, rerun on each of its three seeds, with flags for any change of direction or significance (D44). Writes everything to `results/*.csv`. |
| `plots.py` | Every figure (PNG + PDF): error heatmap, negation-gap bars, error vs score, error by review length, reliability diagram, explanation-overlap heatmap, the small decision tree, word-attribution strips, the Mirage Gap scatter and the shortcut-fingerprint small multiples (each also as a 16:9 `_slide` version). Uses a fixed, colour-blind-checked palette with eight model slots. |
| `demo.py` | The logic behind the live demo (D41): loads the models (DistilBERT at its median seed), scores a sentence, word importance by leaving each word out (the shared `explain.occlusion` function), which words a classical model never saw, the low-confidence and unusual-input flags, and the curated examples with a check that each still behaves as documented. |

### `behavioral/`: the hand-built test sentences (CheckList-style)

These tests check *how* a model behaves, using sentences we write ourselves rather than dataset
sentences.

| File | What happens inside |
|---|---|
| `wordlists.py` | The vocabulary: 20 positive and 20 negative adjectives, 15 movie nouns, mild adjectives and "work" nouns for the name test, Indian and Western first names (paired by gender), and sentiment verbs. |
| `templates.py` | Turns word lists into test cases. Each generator writes sentences from patterns ("The {noun} was {adj}."): `simple`, `negation` ("not great" should be negative), `negation_dir` (the "X / not X" pairs), `typo`, `typo_noun` (the control: typo in the noun, D37), `name_swap` (unseen names) and `name_shortcut` (Kidman vs Seagal). Each row records its test type, case id, role (original/perturbed) and the adjective used. |
| `names.py` | Picks the film people for the name test from the *training labels*: for each surname, how positive its training phrases are (smoothed log-odds). `pick_pairs` pairs the most positive names with the most negative. `name_probe_suite` puts all 35 such names into the same 40 neutral sentences, the separate name probe (D45). |
| `suite.py` | Builds the whole suite (6,250 rows, 3,750 cases) from the config, saves it to `results/behavioral_suite.csv`, and scores predictions. Single-sentence tests fail when the label is wrong; invariance pairs fail when the label changes; direction pairs fail when adding "not" doesn't change the label. |

### `models/`: the classifiers

| File | What happens inside |
|---|---|
| `__init__.py` | Re-exports the model names, the `Classifier` interface and `load_classifier`. |
| `base.py` | The one interface every model follows: `predict_proba(raw texts)` returns P(negative) and P(positive). `SklearnClassifier` wraps a scikit-learn pipeline. `load_classifier` loads either kind of saved model from its folder. Also holds the lightweight shared names analysis code needs: `CLASSICAL_MODELS`, `ABLATION_VARIANTS` (the single-word `unigram` variant, D42) and `external_models()`. |
| `classical.py` | The five classical models: logistic regression, decision tree, random forest, multinomial Naive Bayes, Gaussian Naive Bayes on 300 SVD components. `build_pipeline` assembles vectorizer + model for a variant: `base` (words + word pairs), `negmark` and `negmark_w3` (+ negation marking, two scope rules) and `unigram` (single words only, the first rung of the H2/H3 ladder, D42). `tune` tries every grid setting and keeps the best on held-out accuracy. `train_all` does this for every model. |
| `distilbert.py` | DistilBERT. `train` fine-tunes it with Hugging Face's Trainer (on Kaggle's GPU), keeps the best epoch and deletes checkpoints. `schedule` works out the training and warm-up steps from the effective batch, counting every GPU (D36). `DistilBertClassifier` predicts. Long IMDb reviews keep their first and last tokens rather than being cut off (`head_tail_ids`). |
| `external.py` | Ready-made Hugging Face models (D39). `ExternalClassifier` reuses DistilBERT's batching and head + tail truncation, but reads raw text and takes its label order from the model's own config (`label_columns` refuses anything but one negative and one positive label). `from_hub` loads from the local cache first, so it works offline. |

### `explain/`: what each model relies on

| File | What happens inside |
|---|---|
| `linear.py` | Reads explanations directly off the models: LR's largest weights, Naive Bayes's word log-ratios, and LR's per-word contribution for one sentence. |
| `trees.py` | Fits a depth-4 tree just for the figure, and lists random forest / tree feature importances. |
| `ig.py` | Integrated Gradients for DistilBERT (via Captum): how much each word pushed the prediction, summed from word pieces back to whole words. |
| `deletion.py` | Faithfulness check. Delete the top-attributed words and see whether the prediction drops more than when deleting random words. |
| `overlap.py` | Do different models rely on the same words? Compares their top-word rankings with Jaccard overlap and rank-biased overlap (H5). |
| `occlusion.py` | One ranking rule for every model (D43): remove each word of a sentence in turn and measure how far P(positive) moves (`occlusion_table`); rank words by the mean change (`word_importance`, words seen ≥ 5 times). H5's main measure: `word_effects` (each word's signed effect per model), `split_half_reliability` (how reproducible a model's word effects are) and `effect_agreement` (Spearman correlation between models with a bootstrap CI, corrected for both models' reliability). The top-50 overlap with a split-half ceiling and its chance level (`chance_jaccard`) is the secondary view. Used by the `occlusion` stage (which caches each model's removal table in `artifacts/occlusion/`) and by the live demo. |

### `warning/`: can we flag a model's mistakes in advance? (H4)

| File | What happens inside |
|---|---|
| `embed.py` | Turns text into MiniLM sentence embeddings, cached on disk by item id. `split_sentences` breaks reviews into sentences. |
| `gaussian.py` | Fits a Gaussian to the SST-2 training embeddings and scores new text by Mahalanobis distance: "how unusual is this?". |
| `knn.py` | Alternative distance: 1 − average cosine similarity to the 10 nearest training phrases. |
| `distance.py` | Chooses the distance and the unit from the config. Scores each review by its most unusual sentence. Also provides the length-matched sample and the length-adjusted distance (D30). |
| `error_detection.py` | The H4 comparison. How well does distance, low confidence, the two combined, or length predict which IMDb reviews a model gets wrong (AUROC)? Each is compared head-to-head with confidence. Also the sanity check (can distance tell SST-2 from IMDb?) and error-by-decile tables for the figure. |

---

## `tests/`: 136 automated tests (`.venv/Scripts/python -m pytest`)

Almost all run on small synthetic data and need no downloads or GPU. The two real-model demo
tests use the trained models when they're on disk and skip themselves otherwise. The whole suite
takes about 2 minutes.

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
| `test_external.py` | External models with a tiny random BERT built locally (no download): label mapping in both orders, rejection of 3-class and unnamed labels, raw text and long text, and the `predict-external` stage writing contract files and skipping existing ones. |
| `test_occlusion.py` | Leaving words out: correct changes, chunking changes nothing, punctuation and rare words aren't ranked, the signed direction; the split-half ceiling (a stable model scores 1, and one-off words can't swamp it); chance overlap; agreement (two models that read words alike correlate, an opposite one correlates negatively, and the reliability correction is applied). |
| `test_h1.py` | The two-group bootstrap (right estimate, CI and p; no difference gives a large p), and that H1 compares with test sentences *without* negation. |
| `test_name_probe.py` | The probe puts every name in identical sentences; a model that follows names' reputations correlates, one that ignores names gives no correlation and zero effect size. |
| `test_seed_robustness.py` | The per-seed table's flags: a sign flip, a significance flip and a changed rank are each caught; the median seed is marked; a single seed gives no table. |
| `test_ladder.py` | The single-word variant really has no word pairs; the ladder columns on hand-made error rates; adding the variant leaves every existing negation-gap column identical; `negmark_effect` skips it; its Holm family. |
| `test_mirage_gap.py` | The robustness score on hand-made numbers, "both sentences right", a model missing a component is dropped, the leave-one-out check, the fingerprint merge, and both figures render. |
| `test_demo.py` | Demo logic with a stand-in model: trimming, leaving words out, flags, the vocabulary check, the example checker, which models flip, and reading the research numbers (D49). Plus, with the real models: every curated example behaves as documented, and the app runs headless. |
| `test_pipeline_e2e.py` | Runs the whole CPU pipeline through the real command line on fake data, in a temporary folder, and checks every output file exists. |

---

## Other folders

| Folder | What's in it |
|---|---|
| `notebooks/` | `kaggle_distilbert.ipynb`: clones the repo on Kaggle, installs it, prepares data, fine-tunes DistilBERT (3 seeds), predicts on the GPU, and zips models + predictions for you to download. `kaggle_external.ipynb`: no training; runs the two ready-made models on every eval set (including all 10,000 IMDb reviews) and zips their predictions (D39). |
| `app/` | The live demo (D41). `shortcut_explorer.py` is the Streamlit page (layout only; results are cached per model and sentence, and word importance has its own section with a colour key, D48; a toggle shows each model's accuracy vs robustness and the full-test failure rate behind an example, D49). `examples.yaml` holds the curated example pairs with their measured expected behaviour and a talking point each. `make_fallbacks.py` draws a backup image of every example into `screenshots/`. `PRESENTING.md` is the presenter guide: setup, demo script and ready lines for when a live sentence goes wrong. |
| `results/` | Every output table (`*.csv`), the behavioral suite, `explain/` (top features, the depth-4 tree, overlap), and `predictions/` (one parquet per model × variant × seed × dataset; git-ignored). |
| `figures/` | Every figure as PNG + PDF, redrawn by `python -m mirage figures`. |
| `artifacts/` (git-ignored) | Large regenerable files: `data/` (parquet datasets), `models/` (about 2 GB, mostly the random forests), `emb/` (embedding caches), plus scratch logs. |
| `report/` | Empty placeholder for the final report. |

---

## Bugs and issues found while writing this

1. **✅ FIXED: the Kaggle notebook could silently run old code.** Cell 2 cloned the repo only if
   the folder was missing, so an interactive session that already had a clone (e.g. from the smoke
   test) kept the old code. It now fetches and resets an existing clone to `origin/main`, prints
   the commit it's running (`code version:`), and avoids a nested clone on Colab.
2. **✅ FIXED: the smoke test took longer than advertised.** In smoke mode, cell 5 now predicts only
   the short datasets (it skips the 10,000 IMDb reviews), and the comments say about 5 minutes
   per seed.
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
