# Mirage — Implementation Plan

Coding plan for *Mirage: Exposing the Illusion of Accuracy* (4 weeks, solo, free GPU only).
It turns the proposal into a repo layout, module contracts, a day-by-day build order, tests, and a
map of which Claude Code features to use at each step.

---

## 0. Issues in the proposal to fix before coding

These will silently corrupt results if not handled. Each one is resolved in the plan below.

| # | Issue | Fix |
|---|---|---|
| 1 | "Real negation subset = SST-2 **test** sentences", but SST-2 test labels are hidden (`-1`). | Build the negation subset from the **validation** set (872 sentences). Expect roughly 100–200 sentences; measure and report on day 1. Update the proposal wording. |
| 2 | SST-2 text is pre-tokenized and lowercased: `does n't`, `it 's`, ` , `. Templates and IMDb are not (`doesn't`). Classical models would see different tokens for the same word. | One `normalize()` function (lowercase, split `n't` / `'s` / punctuation into SST style, strip `<br />`) applied to **every** input of **every** model. |
| 3 | scikit-learn's default `token_pattern` (`\b\w\w+\b`) drops `n't`, `no`-style single-char pieces and punctuation, so negation features vanish and negation marking can't find its scope end. | `TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False)` on normalized text. **No stop-word removal** (it removes "not"). |
| 4 | SST-2 train is mostly short *phrases*, many without punctuation. | Negation scope ends at punctuation **or end of text**. |
| 5 | "not bad" has no clean expected label. | Split negation tests into *negated-positive → negative* (scored, ~100) and *negated-negative → not negative* (~50, reported separately as ambiguous). |
| 6 | Decision-tree probabilities are mostly 0/1, so confidence-based AUROC has huge ties and ECE is degenerate. | Report it, but flag it; use `predict_proba` from a depth-limited tree only for the explanation figure. |
| 7 | All IMDb reviews are "far" from SST-2 (uniform shift), so distance may barely vary *within* IMDb. | Add a sanity check: AUROC of distance for SST-val (in) vs IMDb (out) — should be near 1.0. Then the real H4 test: error detection *within* IMDb. |
| 8 | 3 DistilBERT seeds × per-item analyses = 3× the work. | Aggregate metrics over 3 seeds (mean ± std); per-item analyses (IG, failure examples, McNemar) use the **median-accuracy seed**. |

---

## 1. Architecture: the one rule

**Predictions are the contract.** Every model, on every evaluation set, writes one predictions file
with the same schema. All metrics, statistics, plots and the early-warning analysis read *only* these
files. This decouples GPU work (Kaggle) from CPU work (laptop) and makes every table reproducible
from `results/predictions/`.

```
results/predictions/{model}__{variant}__seed{seed}__{dataset}.parquet

columns: item_id | text | label | p_pos | pred | correct | (dataset-specific: test_type, pair_id, expected)
model   ∈ logreg, dtree, rf, mnb, gnb, distilbert
variant ∈ base, negmark
dataset ∈ sst2_val, sst2_neg, behavioral, imdb
```

Every classifier implements one interface:

```python
class Classifier(Protocol):
    name: str
    def fit(self, texts: list[str], labels: np.ndarray) -> "Classifier": ...
    def predict_proba(self, texts: list[str]) -> np.ndarray: ...   # shape (n, 2)
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> "Classifier": ...
```

`normalize()` (and `negation_mark()` for the `negmark` variant) live *inside* each classifier's
pipeline, so callers always pass raw text.

---

## 2. Repository layout

```
mirage/
├── CLAUDE.md                 # conventions for Claude Code (see §7)
├── README.md
├── pyproject.toml            # package + deps + ruff + pytest config
├── configs/
│   ├── data.yaml             # seeds, held-out fraction, IMDb sample size, negation words
│   ├── classical.yaml        # vectorizer settings + per-model grids
│   ├── distilbert.yaml       # lr, epochs, batch, max_len, seeds
│   ├── behavioral.yaml       # word lists, counts per test type
│   └── warning.yaml          # embedding model, covariance, CV folds
├── src/mirage/
│   ├── config.py             # YAML → dataclasses, global seed setter
│   ├── data.py               # load_sst2(), split_heldout(), negation_subset(), load_imdb_sample()
│   ├── text.py               # normalize(), negation_mark(), NEGATION_RE
│   ├── features.py           # build_tfidf(), build_counts(), build_svd()
│   ├── models/
│   │   ├── base.py           # Classifier protocol, registry, save/load helpers
│   │   ├── classical.py      # LogReg, DTree, RF, MNB, GNB wrappers + grid search on held-out
│   │   └── distilbert.py     # HF wrapper: train(), predict_proba() with batching/truncation
│   ├── behavioral/
│   │   ├── wordlists.py      # pos/neg adjectives, nouns, names, intensifiers
│   │   ├── templates.py      # MFT (simple), negation, typo-INV, name-swap-INV generators
│   │   └── suite.py          # build_suite() → DataFrame; score_suite() → failure rates
│   ├── predict.py            # run_predictions(model, dataset) → writes parquet (the contract)
│   ├── metrics.py            # accuracy, f1, ece, reliability bins, error_rate_by_group
│   ├── stats.py              # bootstrap_ci(), paired_bootstrap_diff(), mcnemar_test()
│   ├── explain/
│   │   ├── linear.py         # top ± LR weights, MNB log-prob ratios
│   │   ├── trees.py          # depth-limited tree export, RF importances
│   │   ├── ig.py             # Captum LayerIntegratedGradients + word aggregation
│   │   ├── deletion.py       # remove top-k attributed words vs random-k control
│   │   └── overlap.py        # Jaccard / rank-biased overlap of top-k shortcut words
│   ├── warning/
│   │   ├── embed.py          # MiniLM encoding with on-disk cache (.npy)
│   │   ├── gaussian.py       # LedoitWolf fit, Mahalanobis scoring
│   │   └── error_detection.py# AUROC: distance vs 1-maxprob vs CV logistic combo
│   ├── plots.py              # all figures (one function per figure)
│   └── cli.py                # `python -m mirage <stage>` entry point
├── notebooks/
│   └── kaggle_distilbert.ipynb   # thin: pip install repo, call mirage.models.distilbert.train
├── tests/                    # pytest, see §5
├── results/                  # CSVs + predictions (committed, small)
├── figures/                  # PNG/PDF (committed)
├── artifacts/                # model weights, embeddings cache (gitignored)
└── report/                   # report source + exported PDF
```

**Stages** (`python -m mirage <stage>`; Windows-friendly, no Makefile):

| Stage | Does | Output |
|---|---|---|
| `prepare` | Download SST-2 + IMDb, split held-out, build negation subset, IMDb sample | `artifacts/data/*.parquet`, `results/data_stats.csv` |
| `train-classical --variant base\|negmark` | Grid search on held-out, save best | `artifacts/models/*`, `results/tuning_classical.csv` |
| `train-distilbert --seed N` | Fine-tune (run on Kaggle) | `artifacts/models/distilbert_seedN/` |
| `build-suite` | Generate behavioral cases | `results/behavioral_suite.csv` |
| `predict --model all --dataset all` | Write the contract files | `results/predictions/*.parquet` |
| `evaluate` | Main table, per-test-type errors, ECE, CIs, McNemar | `results/main_table.csv`, `results/errors_by_type.csv`, `results/mcnemar.csv` |
| `explain` | Weights, splits, importances, IG, deletion, overlap | `results/explain/*.csv` |
| `warn` | Embeddings, Gaussian, error-detection AUROC | `results/early_warning.csv` |
| `figures` | All plots | `figures/*` |
| `all` | Everything except DistilBERT training | — |

---

## 3. Module specs (what each must do, and the gotchas)

### 3.1 `data.py`
- `load_sst2()` → `stanfordnlp/sst2` train + validation. Never load the test split's labels.
- `split_heldout(train, frac=0.05, seed=42)` → stratified split; the held-out is used for **all** tuning and DistilBERT checkpoint selection. Models are **not** refit on train+held-out (keeps classical and DistilBERT comparable).
- `negation_subset(val)` → rows matching `NEGATION_RE = r"\b(not|never|no|nothing|n't)\b"` on normalized text. Save size + label balance to `data_stats.csv`.
- `load_imdb_sample(n=2000, seed=42)` → stratified sample of `stanfordnlp/imdb` test split, strip `<br />`.

### 3.2 `text.py`
- `normalize(s)`: lowercase; `<br />` → space; `n't` → ` n't` (`can't` → `ca n't`, `won't` → `wo n't`, matching SST); split `'s 're 've 'll 'd`; pad punctuation with spaces; collapse whitespace.
- `negation_mark(s)`: after a token in the negation set, prefix following tokens with `NOT_` until a token in `{. , ! ? ; :}` or end of text. The negation word itself is unchanged. `"not good movie ."` → `"not NOT_good NOT_movie ."`.
- Unit-test both heavily (§5) — they decide H2/H3.

### 3.3 `features.py` / `models/classical.py`
| Model | Pipeline | Grid (select by held-out accuracy) |
|---|---|---|
| logreg | normalize → [negmark] → TF-IDF(1–2 grams, min_df=2, sublinear_tf) → LogisticRegression(liblinear, L2) | C ∈ {0.1, 1, 10} |
| dtree | same TF-IDF → DecisionTreeClassifier(gini) | max_depth ∈ {10, 30, None} |
| rf | same TF-IDF → RandomForest(300, max_features="sqrt", n_jobs=-1, random_state=42) | max_depth ∈ {30, None} |
| mnb | normalize → [negmark] → CountVectorizer(1–2 grams, min_df=2) → MultinomialNB | alpha ∈ {0.1, 0.5, 1} |
| gnb | same TF-IDF → TruncatedSVD(300, random_state=42) → GaussianNB | — |

Use `sklearn.pipeline.Pipeline` so the whole thing pickles via `joblib`. Log fit time per model
(useful for the "cheap model good enough?" argument). RF on ~100k sparse features is the slowest
step — run it in the background.

### 3.4 `models/distilbert.py` + Kaggle notebook
- `distilbert-base-uncased`, `Trainer`, AdamW lr 2e-5, linear schedule, `warmup_ratio=0.06`, bs 32, 3 epochs, `fp16=True`, `eval_strategy="epoch"`, `load_best_model_at_end=True` on held-out accuracy, `save_total_limit=1`.
- Inputs pass through `normalize()` (same as classical), dynamic padding, `max_length=128` for training/SST, `256` for IMDb.
- Seeds `[13, 42, 87]`. Each run ≈ 20–30 min on a T4. Save `trainer_state.json` + eval metrics alongside weights.
- The notebook is thin: clone repo → `pip install -e .` → `python -m mirage train-distilbert --seed N` → zip `artifacts/models/distilbert_seedN`. Download the three checkpoints (~260 MB each); **all inference runs locally on CPU** (2,000 IMDb reviews at 256 tokens is a few minutes).

### 3.5 `behavioral/` (≈400 cases, deterministic from a seed)
| Type | CheckList kind | Generation | Scored as failure when |
|---|---|---|---|
| simple (~100) | MFT | `"The {noun} was {pos_adj}."` / neg, 50/50 | pred ≠ label |
| negation (~100 + ~50) | MFT | `"The {noun} was not {pos_adj}."` → neg (scored); `"... not {neg_adj}."` → reported separately | pred ≠ negative (scored part) |
| typo (~100 pairs) | INV | swap / drop one char inside the sentiment word only | pred(original) ≠ pred(typo) |
| name swap (~50 pairs) | INV | `"{name} loved the {noun}."`, swap names across Indian/Western lists | pred changes |

Every row carries `test_type`, `pair_id`, `expected`. Store the suite as CSV so it is a reusable
deliverable. Report "name swap" for TF-IDF models with a note (names are OOV → trivially pass).

### 3.6 `metrics.py` / `stats.py`
- `accuracy`, `macro_f1`, `ece(p, y, n_bins=15)` (equal-width on max prob), `reliability_bins()` for diagrams.
- `error_rate_by_group(df, group_col)` → rate, n, and **relative error** = group error / overall SST-val error (the H1 number).
- `bootstrap_ci(metric_fn, *arrays, n=1000, seed=42)` → percentile 95% CI. Used for every error rate and AUROC.
- `paired_bootstrap_diff()` → CI of (classical negation error − DistilBERT negation error), resampling items jointly. This is the H2/H3 gap with uncertainty.
- `mcnemar_test(pred_a, pred_b, y)` via `statsmodels` (exact when discordant < 25).
- Negation gap closed (H3) = `(gap_base − gap_negmark) / gap_base`, with paired-bootstrap CI.

### 3.7 `explain/`
- **linear.py:** top 25 positive/negative LR coefficients; MNB `log P(w|pos) − log P(w|neg)`.
- **trees.py:** refit a `max_depth=4` tree *for the figure only* (report full tree accuracy separately); `export_text` + `plot_tree`; RF `feature_importances_` top 25.
- **ig.py:** `captum.attr.LayerIntegratedGradients` on `model.distilbert.embeddings.word_embeddings`; baseline = `[CLS] [PAD]… [SEP]`; `n_steps=50`; target = predicted class; record convergence delta (flag > 0.05). Sum sub-word attributions into words. Run on ≤100 failures (behavioral + SST-val + negation subset).
- **deletion.py:** for k ∈ {1, 2, 3}: drop top-k attributed words, measure drop in p(predicted class); compare to dropping k random words (10 draws). Faithful attributions ⇒ top-k drop ≫ random drop. Do the same for LR using coefficient × tf-idf contribution, for parity.
- **overlap.py:** top-50 word sets per model (LR weights, MNB ratio, RF importances, DTree split features, DistilBERT mean |IG| over failures). Jaccard + rank-biased overlap matrix → H5 heatmap.

### 3.8 `warning/`
- **embed.py:** `sentence-transformers/all-MiniLM-L6-v2` on raw (not normalized) text; cache to `artifacts/emb/{dataset}.npy`. SST-2 train (67k) takes minutes on CPU. Note MiniLM truncates at 256 word-pieces.
- **gaussian.py:** `LedoitWolf().fit(train_emb)`; score = `mahalanobis(x)`. Optional extra (Lee et al.): class-conditional means with shared covariance, score = min over classes.
- **error_detection.py:** for each classifier on IMDb, target = `1 − correct`; predictors:
  1. `distance`
  2. `1 − max_prob`
  3. `LogisticRegression([standardized distance, max_prob])` with 5-fold `StratifiedKFold` **out-of-fold** predictions.

  AUROC + bootstrap CI for each; plus the SST-val-vs-IMDb sanity AUROC. Plot error rate per distance decile with the confidence-decile curve overlaid.

### 3.9 `plots.py` (proposal's required figures)
1. Main results table (CSV → Markdown/LaTeX via `df.to_markdown()`).
2. Heatmap: error rate, model × test type (annotate n and CI).
3. Bar chart: negation gap to DistilBERT, base vs negmark, with CI error bars.
4. IMDb: error rate vs distance decile, confidence baseline overlaid.
5. Reliability diagrams: best classical vs DistilBERT, SST-2 and IMDb (2×2).
6. Depth-4 tree splits next to IG heatmaps for the same 3–4 sentences.
7. Shortcut-word overlap heatmap (H5).

Use one colorblind-safe palette and one style file for all figures.

---

## 4. Day-by-day build order

Each week ends at the proposal's checkpoint. Fallbacks from the proposal apply unchanged.

### Week 1 — Train all models
| Day | Build | Done when |
|---|---|---|
| 1 | Repo skeleton, `pyproject.toml`, venv, `CLAUDE.md`, `config.py`, `text.py` + tests, `data.py` + `prepare` stage | `prepare` runs; negation subset size known and logged |
| 2 | `features.py`, `models/base.py`, `classical.py`, `train-classical --variant base` | 5 classical models saved; `tuning_classical.csv` written |
| 3 | `distilbert.py` + Kaggle notebook; launch seed 42 on Kaggle, then 13 and 87 | First checkpoint trained; held-out acc ≈ 0.90+ |
| 4 | `predict.py`, `metrics.py` (acc, F1, ECE); predictions for all models on `sst2_val`, `sst2_neg` | Contract files exist for 6 models |
| 5 | `evaluate` stage v1 → main table (acc, F1, ECE); first reliability diagram; README stub | **Checkpoint:** all models trained, main table v1 |

### Week 2 — Test
| Day | Build | Done when |
|---|---|---|
| 1 | `wordlists.py`, `templates.py` | ~400 cases generated deterministically; tests pass |
| 2 | `suite.py` scoring (MFT vs INV logic), `build-suite`, predictions on `behavioral` | Failure rate per test type per model |
| 3 | `negation_mark` variant: `train-classical --variant negmark`, predictions on all sets | 5 negmark models + predictions |
| 4 | `stats.py` (bootstrap, paired diff); gap & gap-closed numbers | `errors_by_type.csv` with CIs |
| 5 | Heatmap + negation-gap bar chart; sanity-review results | **Checkpoint:** failure + negation results done (H1, H2, H3 answerable) |

### Week 3 — Explain and warn
| Day | Build | Done when |
|---|---|---|
| 1 | `explain/linear.py`, `trees.py`; top-word tables | LR/MNB/RF/DTree shortcut lists saved |
| 2 | `ig.py` on ≤100 failures (median seed) | IG attributions CSV; convergence deltas logged |
| 3 | `deletion.py`, `overlap.py` → H5 matrix | Deletion vs random table; overlap heatmap |
| 4 | `warning/` modules; IMDb predictions for all models; `warn` stage | `early_warning.csv` with 3 predictors × 6 models + sanity AUROC |
| 5 | Buffer / stretch goal (only if checkpoint met by day 4): DistilBERT + negated augmentation, on a separate branch | **Checkpoint:** explanation + early-warning results done (H4, H5) |

### Week 4 — Report
| Day | Build | Done when |
|---|---|---|
| 1 | McNemar for all model pairs; final CIs; freeze `results/` | All tables final |
| 2 | `figures` stage: all 7 figures final quality | `figures/` complete |
| 3 | Pick 8–10 failure examples; write the shortcut explanation for each | `results/failure_examples.md` |
| 4 | Write report (6–8 pages), answer RQ1–4 and H1–H6 explicitly | Draft complete |
| 5 | README (results summary + rerun commands), fresh-clone rerun of `python -m mirage all`, code review, publish repo | **Final submission** |

---

## 5. Tests (pytest, run on every edit via hook)

| Test file | Checks |
|---|---|
| `test_text.py` | `normalize("It isn't good!")` == `"it is n't good !"`; `can't`/`won't`; `<br />`; idempotent on SST text. `negation_mark` scope ends at punctuation and at end of text; multiple negations; no negation → unchanged. |
| `test_features.py` | Vectorizer vocabulary contains `n't`, `not`, `not good`, `NOT_good` (negmark). Regression guard for issue #3. |
| `test_data.py` | Held-out is stratified and disjoint; no test-split labels used; IMDb sample size and balance. |
| `test_behavioral.py` | Exact counts per type; determinism by seed; typo only alters the sentiment word; INV pairs share `pair_id`. |
| `test_metrics.py` | ECE = 0 for perfectly calibrated toy input; known value for a hand-computed case; error-by-group math. |
| `test_stats.py` | Bootstrap CI contains the point estimate and is deterministic by seed; McNemar matches statsmodels on toy table. |
| `test_warning.py` | Mahalanobis equals `scipy.spatial.distance.mahalanobis` on toy data; OOF combo never sees its own fold. |
| `test_contract.py` | Every predictions file matches the schema; `p_pos ∈ [0,1]`; `pred == (p_pos ≥ 0.5)`. |

Use a tiny fixture dataset (≈200 SST rows) so the full test run stays under ~30 s.

---

## 6. Environment

- Python 3.11, `venv` (or `uv`). Local: CPU PyTorch. Kaggle: GPU PyTorch preinstalled.
- Deps: `scikit-learn, datasets, transformers, accelerate, sentence-transformers, captum, torch, numpy, pandas, pyarrow, statsmodels, scipy, matplotlib, seaborn, pyyaml, tabulate`; dev: `pytest, ruff`.
- Pin versions in `pyproject.toml` once week 1 works; record `pip freeze` in `results/env.txt`.
- Secrets: Kaggle/HF tokens only via environment variables, never in notebooks or configs.

---

## 7. Using Claude Code across the project

Not every Claude Code skill fits a research codebase; this maps the ones that do to the moment they help.

| When | Claude Code feature | How to use it here |
|---|---|---|
| Day 1 | **`/init`** | Generate `CLAUDE.md`, then add: the predictions contract (§1), "always `normalize()` inside the pipeline", "never use SST-2 test labels", "no stop words", seed list, "run `pytest -q` before saying done". |
| Day 1 | **`update-config` skill** | Add a PostToolUse hook that runs `ruff format` + `pytest -q tests/` after edits to `src/`; allow `python`, `pytest`, `ruff` commands. |
| Start of each week | **Plan mode** | Ask Claude to plan that week's modules against this file before writing code; approve, then build. |
| Every session | **Task list** | Let Claude track the day's rows from §4 as tasks. |
| Week 1 day 3 | **NotebookEdit** | Build the thin Kaggle notebook. |
| Long jobs (RF, embeddings, IG) | **Background commands + Monitor** | Run stages in the background and get notified on finish instead of waiting. |
| Weeks 2–3 | **Subagents / parallel sessions** | `behavioral/` and `warning/` are independent: build them in parallel sessions in separate git worktrees; merge after review. |
| Week 3 day 5 | **Worktree** | Stretch goal on its own branch so main stays clean. |
| Writing `plots.py` | **`dataviz` skill** | Palette, heatmap and reliability-diagram conventions; colorblind-safe, consistent figures. |
| End of each week | **`/code-review`** then **`/simplify`** | Review the week's diff for bugs (leakage, wrong denominators), then tidy. |
| Before making repo public | **`/security-review`** | Catch committed tokens or paths. |
| After week 1 | **`fewer-permission-prompts`** | Allowlist the read-only commands you keep approving. |
| Repeated workflow | **`skill-creator`** | Make a project skill, e.g. `/mirage-stage <name>`: run a stage, verify its output files and schema, summarize the numbers. |
| Project decisions | **Memory** | Save decisions (e.g. "negation subset from validation, n=…", "median seed = 42") so later sessions don't re-decide them. |
| Week 4 | **`docx` or `pdf` skill / Docs** | Write the 6–8 page report (Word or PDF, depending on the required format). |
| Week 4 (optional) | **`pptx` skill** | Presentation deck from the figures. |
| Week 4 (optional) | **`xlsx` skill** | Results workbook if a grader wants spreadsheets. |
| Week 4 (optional) | **Artifact** | A shareable results page for a supervisor. |
| Not needed | `claude-api`, `run`, `schedule`/`loop`, `morning` | No LLM calls or app in scope. `claude-api` becomes relevant only for the future-work zero-shot LLM baseline; `run` for the future Gradio demo. |

**Suggested first prompt in the new repo:**
> Read `MIRAGE_IMPLEMENTATION_PLAN.md`. Enter plan mode and plan Week 1 Day 1–2 only: repo skeleton, `pyproject.toml`, `config.py`, `text.py` with tests, `data.py` with the `prepare` stage. Follow §0 issues 1–4 exactly.

---

## 8. Definition of done (maps to proposal deliverables)

- [ ] `python -m mirage all` on a fresh clone regenerates every CSV and figure (given the 3 DistilBERT checkpoints).
- [ ] `results/main_table.csv`: 6 classifiers × {acc, F1, ECE (SST-2, IMDb), IMDb acc drop, negation-subset error}, DistilBERT as mean ± std over 3 seeds.
- [ ] H1–H6 each have a number, a CI and a one-line verdict (supported / rejected) in the report.
- [ ] `results/behavioral_suite.csv` published as a reusable test suite.
- [ ] 7 figures, 8–10 explained failures, 6–8 page report, README with results + rerun commands.
- [ ] Tests pass; repo public; no secrets.
