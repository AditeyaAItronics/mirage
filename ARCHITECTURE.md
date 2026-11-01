# Mirage — Architecture

How the code is organised, how data moves through it, and the choices that keep it fast,
reproducible and cheap to run on a laptop plus a free T4 GPU. The *why* behind each choice is in
[majordecisions.md](majordecisions.md) (referenced as D1–D17 below).

---

## 1. The core idea: stages connected only by files

```
                  ┌──────────────────────────── GPU (Kaggle / Colab) ───────────────────────────┐
                  │  prepare ──► train-distilbert ──► artifacts/models/distilbert__base__seedN   │
                  └──────────────────────────────────────────────┬───────────────────────────────┘
                                                                 │ download zip (3 × ~260 MB)
 ┌──────────────────────────────────────── laptop CPU ───────────▼─────────────────────────────────┐
 │                                                                                                  │
 │  prepare ─► artifacts/data/*.parquet ─► train-classical ─► artifacts/models/{model}__{variant}__seed42
 │                    │                                                         │                   │
 │  build-suite ─► results/behavioral_suite.csv                                 │                   │
 │                    │                                                         │                   │
 │                    └────────────────────► predict ◄──────────────────────────┘                   │
 │                                              │                                                   │
 │                         results/predictions/{model}__{variant}__seed{s}__{dataset}.parquet       │
 │                                  ▲  THE CONTRACT (D7) — everything below reads only this  ▲      │
 │              ┌───────────────────┼─────────────────────┬─────────────────────┐                   │
 │          evaluate             explain                 warn                figures                │
 │  main_table / errors_by_type   top_features         early_warning.csv      figures/*.png|pdf     │
 │  negation_gap / mcnemar        ig_failures          (+ artifacts/emb cache)                      │
 │                                deletion / overlap                                                │
 └──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

Every stage is `python -m mirage <stage>`. A stage reads files and writes files. Nothing is passed in
memory between stages, so:

- **The GPU is needed for exactly one step** (fine-tuning). DistilBERT *inference* runs on CPU,
  because 2.8k evaluation texts take minutes.
- **Any analysis can be re-run in seconds** without re-running a model. Fixing a metric bug means
  re-running `evaluate`, not retraining.
- **Every number in the report traces back to a file** in `results/`.

---

## 2. Package layout and layers

```
src/mirage/
  paths.py  config.py                 ← foundation: where things live, what settings apply
  text.py   features.py   data.py     ← input layer: one normalization for every model
  models/   base.py classical.py distilbert.py
                                      ← model layer: one Classifier interface
  predict.py                          ← the contract: write/validate/load prediction files
  behavioral/ wordlists templates suite
                                      ← test-suite generator + MFT/INV scoring
  metrics.py stats.py evaluate.py     ← analysis layer: point metrics → uncertainty → tables
  explain/  linear trees ig deletion overlap
  warning/  embed gaussian knn distance error_detection
  plots.py                            ← presentation layer
  cli.py                              ← orchestration: one subcommand per stage
```

**Dependency rule:** each layer imports only from layers above it in this list. Analysis code never
imports a model class. It sees prediction frames only, which is why `evaluate`, `stats` and
`plots` have no torch or sklearn-model dependency.

---

## 3. The three interfaces everything depends on

### 3.1 Text in → model text (`text.py`, D5/D6/D8)

```
raw text ──normalize──► SST-style tokens ──[negation_mark]──► model input
"It isn't good."        "it is n't good ."                     "it is n't NOT_good ."
```

`normalize` is idempotent, and it runs *inside* every model: inside the sklearn vectorizer
(`Preprocessor`) and inside `DistilBertClassifier.encode`. Callers always pass raw text.

### 3.2 Classifier (`models/base.py`, D9)

```python
predict_proba(texts: list[str], long_text: bool = False) -> np.ndarray  # (n, 2), col 1 = P(pos)
save(path) / load_classifier(path)
```

Six models, one interface. `long_text=True` (IMDb) lets DistilBERT use 256 tokens instead of 128.

### 3.3 Predictions (`predict.py`, D7)

| column | meaning |
|---|---|
| `item_id` | stable id (`sst2_val-17`, `imdb-3`, `beh-212`); the negation subset reuses `sst2_val` / `sst2_test` ids |
| `text`, `label` | the input and its gold / intended label (0/1) |
| `p_pos`, `pred`, `correct` | P(positive), `p_pos ≥ 0.5`, `pred == label` |
| extras | behavioral rows also carry `test_type, kind, case_id, role, scored` |

`validate_predictions` enforces the schema on every write: unique ids, `p_pos ∈ [0, 1]`, labels in
{0, 1} (so a hidden SST-2 test label of −1 can never get in), and `pred`/`correct` consistent with
`p_pos`.

---

## 4. How each research question maps onto the code

| RQ / H | Data | Computed in | Output |
|---|---|---|---|
| RQ1 / H1 hidden failures | sst2_test, sst2_neg, behavioral | `evaluate.error_table` + `behavioral.suite.case_failures` | `errors_by_type.csv`, heatmap |
| RQ2 / H2–H3 what context buys | sst2_neg, behavioral negation, base + negmark | `evaluate.negation_gap` (paired bootstrap, D12) | `negation_gap.csv`, gap bars |
| RQ3 / H4 early warning | sst2_train embeddings, imdb | `warning.*`, `cli.stage_warn` | `early_warning.csv`, error-vs-score plot |
| RQ4 / H5 explanation | trained models, DistilBERT failures | `explain.*` | `top_features.csv`, `ig_failures.csv`, `deletion_check.csv`, `overlap.csv` |
| H6 GNB vs MNB | sst2_test | `evaluate.main_table` + McNemar | `main_table.csv`, `mcnemar.csv` |
| Calibration | sst2_test, imdb | `metrics.ece`, `reliability_bins` | ECE columns, reliability diagram |

Seed policy (D13): tables average DistilBERT over 3 seeds; per-item analyses use its median seed.

---

## 5. Performance and cost choices

| Choice | Effect |
|---|---|
| Predictions cached as parquet (D7) | All analyses re-run in seconds; models run once |
| Heavy libraries imported lazily (D2) | `import mirage` and the 85 tests need no torch, and the test suite finishes in about 20 s |
| Sparse TF-IDF end to end; RF `n_jobs=-1` | Classical training fits laptop RAM and uses all cores |
| GNB on 300-d TruncatedSVD | Dense Gaussian NB stays tractable (a dense 67k × 100k+ matrix would not) |
| DistilBERT: dynamic padding, fp16, `save_total_limit=1`, best epoch kept | ~20–30 min per seed on a T4; only one checkpoint on disk |
| IMDb truncated at 256 tokens for DistilBERT | Bounded inference cost; truncation is part of the domain shift studied |
| Sentence embeddings cached by `item_id` in `artifacts/emb/*.npz` | Embedding 67k SST sentences on CPU happens once |
| Ledoit-Wolf shrinkage covariance | Well-conditioned 384×384 precision matrix; Mahalanobis is one `einsum` |
| Per-sentence distance, max per review (D25) | Mean-pooled long reviews looked typical (sanity AUROC 0.67); the most unusual sentence separates them (0.93), but mostly through length: one random sentence per review gives 0.61 (D30). Length and a length-adjusted distance are scored as baselines |
| IMDb capped to head + tail words for every model (D23) | One text for all models; DistilBERT reads up to 512 tokens with head+tail token truncation |
| DistilBERT predicts on the Kaggle GPU (D23) | 10,000 reviews × 512 tokens × 3 seeds is slow on a CPU; local `predict` skips models whose predictions are newer than the model |
| IG only on DistilBERT failures (≤100), word-level | Bounded cost; attributions comparable with classical features (D14) |
| Bootstrap resamples item indices, not data | One generic routine for every CI; paired comparisons for free (D12) |

**Known heavy spot:** the random forest with `max_depth=None` and 300 trees on ~64k sparse rows
needs the most RAM (likely a few GB) and the longest fit, and its saved model can be large. If the
laptop struggles, train it on its own with `--models rf` and limit the grid to `max_depth: [30]`.

---

## 6. Reproducibility

- Seeds: data / classical / behavioral = 42; DistilBERT = 13, 42, 87 (`configs/*.yaml`).
- Every trained model directory holds the resolved `config.json` it was trained with.
- `tuning_classical.csv` logs every grid point with held-out accuracy and fit time.
- `MIRAGE_ROOT` relocates all outputs and `MIRAGE_CONFIGS` swaps the config folder. The
  end-to-end test uses both to run the whole CPU pipeline in a temp folder.
- Re-running from scratch: `prepare → train-classical → (Kaggle) train-distilbert → all`.

---

## 7. Testing strategy

| Level | What | Where |
|---|---|---|
| Unit | normalization, negation scope, vectorizer vocabulary (the `n't` trap), ECE, bootstrap, McNemar vs statsmodels, Mahalanobis vs scipy, overlap measures, deletion check | `tests/test_*.py` |
| Contract | every way a predictions file can be malformed is rejected | `tests/test_contract.py` |
| Generator | suite counts, determinism, INV pairing, typo changes exactly one word | `tests/test_behavioral.py` |
| End to end | all CPU stages via the real CLI in an isolated root | `tests/test_pipeline_e2e.py` |

DistilBERT, IG and embeddings are smoke-tested manually (see majordecisions.md) rather than in
pytest, so the suite stays fast and needs no torch.
