# Mirage — Architecture

How the code is organised, how data moves through it, and the choices that keep it fast,
reproducible and cheap to run on a laptop plus a free T4 GPU. The *why* behind each choice is in
[majordecisions.md](majordecisions.md) (the D-numbers below). For what each individual file and
test does, see [codeexplanations.md](codeexplanations.md); this file stays at the level of stages,
layers and interfaces.

---

## 1. The core idea: stages connected only by files

```
                  ┌──────────────────────────── GPU (Kaggle / Colab) ───────────────────────────┐
                  │  prepare ──► train-distilbert ──► predict --models distilbert               │
                  │  prepare ──► predict-external (ready-made HF models, no training, D39)      │
                  └──────────────────────────────────────────────┬───────────────────────────────┘
                                                                 │ download zips (models + predictions)
 ┌──────────────────────────────────────── laptop CPU ───────────▼─────────────────────────────────┐
 │                                                                                                  │
 │  prepare ─► artifacts/data/*.parquet ─► train-classical ─► artifacts/models/{model}__{variant}__seed42
 │                    │                                                         │                   │
 │  build-suite ─► results/behavioral_suite.csv                                 │                   │
 │                    │                                                         │                   │
 │                    └────────────────────► predict ◄──────────────────────────┘                   │
 │  name-probe (D45: its own eval set, outside EVAL_SETS) ──► predictions + name_probe*.csv          │
 │                                              │                                                   │
 │                         results/predictions/{model}__{variant}__seed{s}__{dataset}.parquet       │
 │                         (external models: {name}__base__seed0__{dataset}.parquet)                │
 │                                  ▲  THE CONTRACT (D7) — everything below reads only this  ▲      │
 │              ┌───────────────────┼─────────────────────┬─────────────────────┐                   │
 │          evaluate          explain / occlusion         warn                figures                │
 │  main_table / errors_by_type   top_features         early_warning.csv      figures/*.png|pdf     │
 │  negation_gap / mcnemar        ig_failures          (+ artifacts/emb cache)                      │
 │  h1_test / mirage_gap          deletion / overlap                                                │
 │  seed_robustness               occlusion_* (H5, cache in artifacts/occlusion)                    │
 └──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

Every stage is `python -m mirage <stage>`. A stage reads files and writes files. Nothing is passed in
memory between stages, so:

- **The GPU is needed for fine-tuning and for long-text inference.** DistilBERT and the external
  models predict IMDb (10,000 reviews × 512 tokens) on the Kaggle GPU; short texts take minutes
  on a CPU.
- **Any analysis can be re-run in seconds** without re-running a model. Fixing a metric bug means
  re-running `evaluate`, not retraining.
- **Every number in the report traces back to a file** in `results/`.

---

## 2. Package layout and layers

```
src/mirage/
  paths.py  config.py                 ← foundation: where things live, what settings apply
  text.py   features.py   data.py     ← input layer: one normalization for every model
  models/   base.py classical.py distilbert.py external.py
                                      ← model layer: one Classifier interface
  predict.py                          ← the contract: write/validate/load prediction files
  behavioral/ wordlists templates suite names
                                      ← test-suite generator + MFT/INV scoring; names = the name
                                        test's pairs and the name probe (D45)
  metrics.py stats.py evaluate.py     ← analysis layer: point metrics → uncertainty → tables
  explain/  linear trees ig deletion overlap occlusion
  warning/  embed gaussian knn distance error_detection
  plots.py                            ← presentation layer
  demo.py                             ← logic behind the live demo (scores, word importance, flags)
  cli.py                              ← orchestration: one subcommand per stage
app/        shortcut_explorer.py examples.yaml make_fallbacks.py
                                      ← the Streamlit demo: layout only, logic in demo.py
```

The demo is a side door, not a stage: it loads the saved models directly and never writes
results, so it can't change a reported number (D41).

**Dependency rule:** each layer imports only from layers above it in this list. Shared names that
analysis needs (`CLASSICAL_MODELS`, `ABLATION_VARIANTS`, `external_models()`) live in the
lightweight `models/base.py` for that reason. Analysis code never
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
The one exception is the external models (D39): they were trained on ordinary English, so
`ExternalClassifier` sets `normalize_input = False` and reads the text as stored.

### 3.2 Classifier (`models/base.py`, D9)

```python
predict_proba(texts: list[str], long_text: bool = False) -> np.ndarray  # (n, 2), col 1 = P(pos)
save(path) / load_classifier(path)
```

Eight models, one interface: five classical, our DistilBERT, and two ready-made external models
(`ExternalClassifier`, which reads its label order from the model's own config). `long_text=True`
(IMDb) lets the transformers use 512 tokens, head + tail, instead of 128.

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
| RQ1 / H1 hidden failures | sst2_test, sst2_neg, behavioral | `evaluate.error_table` + `behavioral.suite.case_failures`; formal test `evaluate.h1_test` against negation-free test sentences (D47) | `errors_by_type.csv`, `h1_test.csv`, heatmap |
| RQ2 / H2–H3 what context buys | sst2_neg, behavioral negation; the ladder unigram → base (+ pairs) → negmark (+ marking), D42 | `evaluate.negation_gap` (paired bootstrap, D12) | `negation_gap.csv`, gap bars |
| RQ3 / H4 early warning | sst2_train embeddings, imdb | `warning.*`, `cli.stage_warn` | `early_warning.csv`, error-vs-score plot |
| RQ4 / H5 explanation | trained models, DistilBERT failures; for H5, SST-2 test with one ranking rule for every model (D43) | `explain.*`, stage `occlusion` | `top_features.csv`, `ig_failures.csv`, `deletion_check.csv`, `overlap.csv`; H5 (main): `occlusion_agreement.csv` (reliability-corrected correlation of word effects), `occlusion_reliability.csv`; secondary: `overlap_occlusion.csv`, `occlusion_split_half.csv` |
| H6 GNB vs MNB | sst2_test | `evaluate.main_table` + McNemar | `main_table.csv`, `mcnemar.csv` |
| Calibration | sst2_test, imdb | `metrics.ece`, `reliability_bins` | ECE columns, reliability diagram |
| Does scale cure shortcuts? (D39) | every set, external models | the same tables; their own Holm families | new rows everywhere |
| Name shortcut beyond the 6 test pairs (D45) | all 35 film people in the same 40 sentences (`name_probe`, a separate eval set) | stage `name-probe`, `evaluate.name_probe` | `name_probe.csv`, `name_probe_by_name.csv` |
| Does any conclusion depend on DistilBERT's seed? (D44) | every per-item result, per seed | `evaluate.seed_robustness` | `seed_robustness.csv` |
| Accuracy vs behaviour (D40) | sst2_test, sst2_neg, behavioral pairs | `evaluate.mirage_gap` (pair accuracy) | `mirage_gap.csv`, Mirage Gap chart, fingerprint |

Seed policy (D13): tables average DistilBERT over 3 seeds; per-item analyses use its median seed,
and `evaluate.seed_robustness` reruns each of them on every seed to show no conclusion depends on it (D44).

---

## 5. Performance and cost choices

| Choice | Effect |
|---|---|
| Predictions cached as parquet (D7) | All analyses re-run in seconds; models run once |
| Heavy libraries imported lazily (D2) | `import mirage` needs no torch; tests that need torch, transformers or Streamlit skip themselves when those are missing |
| Sparse TF-IDF end to end; RF `n_jobs=-1` | Classical training fits laptop RAM and uses all cores |
| GNB on 300-d TruncatedSVD | Dense Gaussian NB stays tractable (a dense 67k × 100k+ matrix would not) |
| DistilBERT: dynamic padding, fp16, `save_total_limit=1`, best epoch kept | ~20–30 min per seed on a T4; only one checkpoint on disk |
| IMDb read at up to 512 tokens, head + tail, by every transformer | Bounded inference cost; the ending, where the verdict often is, is kept |
| Sentence embeddings cached by `item_id` in `artifacts/emb/*.npz` | Embedding 67k SST sentences on CPU happens once |
| Ledoit-Wolf shrinkage covariance | Well-conditioned 384×384 precision matrix; Mahalanobis is one `einsum` |
| Per-sentence distance, max per review (D25) | Mean-pooled long reviews looked typical (sanity AUROC 0.67); the most unusual sentence separates them (0.93), but mostly through length: one random sentence per review gives 0.61 (D30). Length and a length-adjusted distance are scored as baselines |
| IMDb capped to head + tail words for every model (D23) | One text for all models; DistilBERT reads up to 512 tokens with head+tail token truncation |
| DistilBERT predicts on the Kaggle GPU (D23) | 10,000 reviews × 512 tokens × 3 seeds is slow on a CPU; local `predict` skips models whose predictions are newer than the model |
| IG only on DistilBERT failures (≤100), word-level | Bounded cost; attributions comparable with classical features (D14) |
| Bootstrap resamples item indices, not data | One generic routine for every CI; paired comparisons for free (D12) |
| Single-word ablation (D42) trains in about 3 minutes | The H2/H3 ladder costs almost nothing; it gets its own Holm family, so no earlier p-value moves |
| External models load cache-first (D41) | No network round trip; the demo works offline once models are cached |
| Word-removal tables cached in `artifacts/occlusion/` (stale when the model or data is newer) | Re-running the H5 analysis takes seconds instead of ~10 min of DistilBERT on CPU |
| Word importance by leaving words out (`explain.occlusion`) | One batched call per model per sentence: works for every model and stays under a second in the demo. The H5 stage scores about 38,000 removals per model on SST-2 test in chunks; DistilBERT takes the longest on CPU. The demo and the H5 analysis share this one function |

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
- Exact library versions of the final results: `requirements-lock.txt` (`pip freeze`, D46).

---

## 7. Testing strategy

Tests are layered the same way as the code; the full list, file by file, is in
[codeexplanations.md](codeexplanations.md) (section `tests/`).

| Level | What it guards |
|---|---|
| Unit | normalization and negation scope, vectorizer vocabulary (the `n't` trap), metrics and statistics against reference implementations, word removal, explanation and warning maths |
| Contract | every way a predictions file can be malformed is rejected |
| Generator | suite counts, determinism, pairing, the typo, direction and name rules |
| Analysis | each table on hand-made numbers: the H1 test, the H2/H3 ladder, the robustness score, H5 agreement, the name probe, seed robustness; adding models or variants leaves existing columns identical |
| External models | label mapping, the CLI stage, all with a tiny local random BERT (no download) |
| Demo | its logic with stand-in models; with the real models, every curated example's expected behaviour and a headless run of the app |
| End to end | all CPU stages via the real CLI in an isolated root |

Fine-tuning, IG and embeddings are smoke-tested manually (see majordecisions.md) rather than in
pytest. The real-model demo tests skip themselves on a fresh clone without trained models.
