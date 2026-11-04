# Shortfall fixes: design plan

How to fix every open 🔴 and 🟠 item in [shortfalls.md](shortfalls.md): what each fix does, the
code and outputs it adds, how it is tested and reported, what you decide, and the effort. It
was written as the plan to approve (D51); the status table below says what has been built since.

## Status (4 Nov 2026)

The author took all six recommendations and chose to build F2 + G8 now (with the fixes they
depend on). What was built differs from the plan in a few places, noted here; the plan below is
kept as written.

| Item | Status | Where | Differs from the plan |
|---|---|---|---|
| **H8** headline seeds | ✅ done | D52, `evaluate.headline_pool` | — |
| **G7** cluster bootstrap | ✅ done | D52, `results/cluster_robustness.csv` | — |
| **G9** minimum effects | ✅ done | D52, `configs/stats.yaml` | — |
| **C12** RoBERTa † mark | ✅ done | D52 | — |
| **D10** frame set B | ✅ done | D53, `behavioral_by_frame.csv` | Claude drafted the frames, the author approved them |
| **G8** replication | ✅ done: 16 of 17 replicated | D53, `results/replication.csv` | One Kaggle notebook (`kaggle_fresh.ipynb`) for the external models only; our DistilBERT ran on the CPU (its model files aren't on GitHub), median seed only |
| **F2** own-feature distance | ✅ done | D53, `results/early_warning_own.csv` | DistilBERT's features were extracted on the CPU, not in a Kaggle notebook |
| **H7** claims registry | open | — | not built yet |
| **C3**, **C4**, **E1** | open | — | the training notebook (`kaggle_fixes_train`) and IG were not built |
| **B10** edges labels | open | — | needs the author's hand-labelling |

New items found while building: **D12** (template results depend on the frame) and **G11**
(claim 3 failed), both in shortfalls.md.

**Scope: 12 open items.**
- 🔴 **G7** confidence intervals ignore clustering (new).
- 🟠 from before: **B10** edges labels, **C12** RoBERTa contamination, **C3** DistilBERT untuned,
  **C4** few seeds, **E1** IG baseline, **F2** embedding ≠ classifier.
- 🟠 new: **D10** one template set, **G8** no confirmatory check, **G9** no minimum effect,
  **H7** hand-copied numbers, **H8** seeds/variants change headline tables.

**Closed in this change, no work needed:** F1, F6 and F7 were 🟠 but already answered (D25, D30,
D31, D46); shortfalls.md now marks them ✅.

---

## Ground rules for every fix

These come from [CLAUDE.md](CLAUDE.md) and apply to all items below.
- **Predictions are the contract.** New models, seeds or eval sets write prediction files through
  `predict.write_predictions`; analyses read only those files.
- **Additions never change earlier statistics.** New models, variants, seeds and predictors get
  their own Holm family or their own table. After each fix, `artifacts/unchanged_check.py` confirms
  every existing number in `results/*.csv` is unchanged. The one deliberate exception is G7, where
  the current intervals are wrong; that change is your decision (D1 below).
- **Side probes stay out of `EVAL_SETS`.** The fresh suite and fresh IMDb sample (G8, D10) get
  their own stage, like `name_probe`.
- **Docs in the same change:** a decision entry per fix in `majordecisions.md`, the shortfall moved
  to Resolved, new results in hypothesisanswers.md / interestingfindings.md, file tour in
  codeexplanations.md, steps for you in whatyouneedtodo.md.
- **One code review per batch**, before committing.

---

## Summary

| ID | Shortfall | Fix in one line | Who | Compute | Effort | Needs first |
|---|---|---|---|---|---|---|
| **H8** | headline tables change when seeds/variants are added | explicit headline seed per model and variant | Claude | CPU | ½ day | — |
| **H7** | numbers in docs copied by hand | claims registry checked by pytest | Claude | CPU | ½–1 day | — |
| **G7** 🔴 | CIs ignore clustering | cluster bootstrap for every behavioral and IMDb result | Claude; you decide D1 | CPU | 1 day | H7 |
| **G9** | no minimum effect | smallest effect of interest, "meaningful" columns | Claude; you approve D2 | CPU | ½ day | H7 |
| **C12** | RoBERTa-large contaminated | mark every place its accuracy appears, then 🟢 | Claude | — | 1 h | — |
| **D10** | one template set, one draw | record frames; paraphrased frames; per-frame table | Claude; you write/approve frames | CPU | 1 day | G7 |
| **G8** | no confirmatory check | pre-registered claims, tested on a fresh suite and fresh IMDb | Claude; you approve claims | CPU + Kaggle | 1 day + 1 run | D10, H8 |
| **C4** | few seeds | 2 more DistilBERT seeds; RF, DT, GNB on 3 seeds | Claude; you run Kaggle | CPU + Kaggle | ½ day + run | H8 |
| **C3** | DistilBERT untuned | small grid, a `tuned` variant as a robustness check | Claude; you run Kaggle | Kaggle | ½ day + run | H8 |
| **F2** | embedding ≠ classifier | classifiers' own-feature Mahalanobis (Lee et al.) | Claude; you run Kaggle | CPU + Kaggle | 1 day + run | — |
| **E1** | IG baseline, small sample | second baseline, all failures, IG vs occlusion | Claude; you run Kaggle | Kaggle | ½ day + run | — |
| **B10** | edges labels noisy | you hand-label 200 excerpts; re-estimate the shift | **you** label; Claude analyses | CPU | 1 h of yours + ½ day | — |

**Your time in total:** about 3–4 hours spread over the plan: the six decisions below (30 min),
approving frames and claims (1 h), two Kaggle runs (15 min of clicking, then unattended), and the
labelling (1 h).

---

## Order

```
Phase 1 (Claude, CPU)        H8 ──► H7 ──► G7 ──► G9        C12 (any time)
                              │             │
Phase 2 (Claude + you)        │             └──► D10 ──► G8 (pre-register, then build)
                              ├──► C4 classical seeds
                              └──► F2 for the classical models
Phase 3 (Kaggle, you click)   one training notebook (C3 grid + tuned seeds, C4 DistilBERT seeds)
                              one inference notebook (G8 fresh IMDb, F2 DistilBERT features, E1 IG)
Phase 4 (Claude)              integrate, replication results, unchanged check, docs, review
Any time (you)                B10 labelling
```

- **H8 first:** C3 and C4 add seeds and a variant, which would silently change the headline tables
  until H8 is fixed.
- **H7 second:** G7, G9 and the later fixes change or add many numbers; the registry catches every
  doc that needs updating.
- **G7 before D10 and G8:** the new tables should be built with the right intervals from the start.
- **Both Kaggle notebooks go in one sitting**, so you click once per notebook. They need Phase 1–2
  code to be pushed first.

---

## Decisions for you

| # | Decision | Recommendation | Why |
|---|---|---|---|
| D1 | G7: which interval becomes the reported one for behavioral results? | **The cluster interval**, with the old one kept in the robustness table | The item interval is wrong for clustered data; keeping it as the headline would report CIs known to be too narrow |
| D2 | G9: the smallest effects that matter | 2 points of error rate; 0.02 of probability shift; 0.02 of AUROC; 10% of a gap closed | Small enough not to hide real effects, large enough to drop the noise-level ones (G9's examples) |
| D3 | C3: if the tuned DistilBERT changes a conclusion, which reference counts? | **Keep the current DistilBERT as the reference**, report the tuned one beside it | Every analysis is built on it; a changed conclusion would be reported, not hidden |
| D4 | C4: headline DistilBERT on 3 seeds or 5? | **Keep 3 for the headline**, use all 5 in the seed-robustness table | Changing the headline would change every number in the docs for no expected gain |
| D5 | D10: who writes the paraphrased frames? | **You write them** (or approve Claude's draft) | Frames written by a second person are less likely to share the first set's quirks |
| D6 | G8: the list of confirmatory claims | Approve the draft Claude writes, before anything is run | A claim list written after seeing the new data isn't confirmatory |

---

## H8. Headline tables must not change when seeds or variants are added

**Problem.** `evaluate.canonical` keeps every seed of every model except DistilBERT, `main_table`
averages over all seeds it is given, and `median_seed` groups DistilBERT's validation predictions by
seed only, mixing variants. Adding RF seeds (C4) would average them into RF's headline accuracy;
adding a `tuned` DistilBERT variant (C3) would change which seed is "median".

**Done when:** adding any number of extra seeds or variants leaves every existing CSV in `results/`
byte-identical, and a test proves it.

**Design.**
- New function `evaluate.headline(preds)` replacing `canonical`, applied once at the top of
  `evaluate.run` (and wherever `canonical` is used today):
  - **classical models:** keep `seed == configs/classical.yaml seed` (42);
  - **DistilBERT:** per variant, keep the median seed among `configs/distilbert.yaml seeds`
    (the headline seeds 13, 42, 87); robustness seeds live in a new key `robustness_seeds`
    and are never headline;
  - **external models:** seed 0.
- `median_seed(preds, model, variant="base", seeds=None)` gains `variant` and `seeds`, so it ranks
  only the headline seeds of one variant.
- `main_table` takes headline seeds only: DistilBERT keeps its 3-seed mean (`n_seeds` stays 3).
- New tables that want every seed (`seed_robustness`, `classical_seed_spread`, C4) read the raw
  predictions explicitly.

**Tests** (`tests/test_headline.py`): fake predictions with an extra RF seed, an extra DistilBERT
robustness seed and a `tuned` variant; `main_table`, `negation_gap`, `h1_test` and `mirage_gap` are
identical with and without them; `median_seed` ignores robustness seeds and other variants.

**Check:** run `evaluate` and `artifacts/unchanged_check.py`: nothing changes today (only one seed
exists for each classical model).

**Effort:** half a day.

---

## H7. A claims registry so no number in the docs can go stale

**Problem.** README, hypothesisanswers.md and interestingfindings.md quote hundreds of numbers typed
from the CSVs; two have already drifted.

**Done when:** `pytest` fails if any registered number no longer matches its CSV cell, or if the
doc no longer contains it; every headline number in the three files is registered.

**Design.**
- `docs/claims.yaml`, one entry per quoted number:
  ```yaml
  - id: h1.distilbert.typo
    doc: hypothesisanswers.md
    text: "+32.0"              # exactly as it appears in the doc
    file: results/h1_test.csv
    where: {model: distilbert, test: beh_typo}
    column: diff
    format: points1            # ×100, one decimal, signed
  ```
  Formats: `pct0` (83%), `pct1`, `prob3` (0.834), `points1` (+32.0), `ci_points` ([27.6, 36.6]),
  `int`, `sci` (p-values). Ranges ("14–35") are two claims, the min and the max of a filtered set.
- `src/mirage/claims.py`: `load(path)`, `value(claim)` (reads the CSV cell; error if the filter
  matches 0 or more than 1 row), `render(value, format)`, `check(claims) -> list[str]` (problems:
  mismatch, missing in doc, ambiguous filter).
- `tests/test_claims.py` runs `check` over the registry; it uses only committed CSVs, so it also
  runs on a fresh clone.
- `python -m mirage check-claims` prints the problems, for use while editing docs.
- **Seeding the registry:** a helper script extracts the numbers from the summary tables and
  verdicts of the three files and proposes entries; Claude checks each by hand. Start with the
  ~150 headline numbers (README tables, the H1–H6 summary and verdict tables, the 13 findings'
  key numbers), and grow it as docs change.
- The report's number check (whatyouneedtodo step 10) adds the report draft as another `doc`.

**Effort:** half a day for the code, half a day to register the numbers.

---

## G7 🔴. Cluster bootstrap for every behavioral and IMDb result

**Problem.** Bootstraps resample single cases, but cases share words: each test's 250–500 cases
come from 10–39 adjectives and 6 name pairs. Measured: resampling words instead of cases widens
DistilBERT's typo CI 2.8× and Multinomial NB's name-shortcut CI 4.1× (shortfalls.md G7). IMDb
reviews come several per film.

**Done when:** every behavioral and IMDb result has a cluster interval and a cluster p-value,
Holm-adjusted within the same families; a table lists each verdict under both; the docs report
the interval chosen in D1, and any verdict that changes is stated.

**Design.**
- **Clusters.**
  - *Behavioral:* add a `cluster` column to the suite, built by `behavioral.suite`: the adjective
    (`word`) for simple, negation, direction and typo tests; the name pair for the name tests; the
    frame id is added too (D10). Columns are appended, so every existing case keeps its id, text
    and label, and the existing draw is unchanged (tested).
  - *IMDb:* the film. The Hugging Face IMDb copy has no film ids, but the original release
    (`aclImdb_v1.tar.gz`, 84 MB) lists each review's IMDb URL (`urls_pos.txt`, `urls_neg.txt`,
    line *n* = file `n_*.txt`). `data.imdb_film_ids()` matches our reviews to the original files by
    exact text and writes `artifacts/data/imdb_films.parquet` (`item_id → film`). If a review can't
    be matched it becomes its own cluster, and the match rate is reported.
  - *SST-2:* no film ids exist; sentences stay independent (stated as a limitation).
- **Statistics** (`stats.py`):
  - `cluster_bootstrap(stat, clusters, n_resamples, seed, alpha, null)`: resample cluster ids with
    replacement, take all their items, evaluate `stat`; same `_summarize` and SE-based p-value as
    `bootstrap`.
  - Paired versions for gaps and ladders (both sides resampled by the same clusters) and a
    two-group version for H1 (each group resampled by its own clusters).
- **Analyses** (`evaluate.cluster_robustness`): recompute, with clusters, every behavioral and IMDb
  result that carries a CI: H1's template and typo columns; the negation gap and ladder on
  templates; `negmark_effect` on templates; typo direction; name effect; the robustness score and
  its components (which get a CI for the first time); early-warning AUROC differences and the
  shift decomposition on IMDb.
- **Output:** `results/cluster_robustness.csv`: analysis, model, variant, test, estimate, item CI,
  cluster CI, width ratio, item and cluster Holm p, verdict under each, `changed`.
- **If you choose D1 = cluster CIs:** the existing tables' CI and p columns for behavioral and IMDb
  rows switch to the cluster version (a deliberate change, recorded in its decision entry, with the
  old values kept in `cluster_robustness.csv`), and every affected doc number is updated through the
  registry (H7).

**Tests:** on synthetic data where items within a cluster are identical, the cluster CI equals the
CI of the cluster means; with one item per cluster it equals the ordinary bootstrap; the IMDb film
matcher on a tiny fake archive; suite ids unchanged after adding the column.

**Risk:** with 6 name pairs, a cluster bootstrap over pairs is coarse; for the name tests, report
clusters by name pair *and* by adjective, and use the wider one.

**Effort:** one day (most of it the IMDb matching and re-plumbing the paired analyses).

---

## G9. A smallest effect of interest next to every significance flag

**Problem.** Tiny effects are significant (DistilBERT's −0.0001 name shift, Holm p 0.007), and H1
counts HF DistilBERT's +3.5 points as "clearly higher" on p alone.

**Done when:** every table that flags significance also flags whether the effect is at least the
smallest effect of interest, the verdicts in hypothesisanswers.md use both, and the thresholds and
their after-the-fact choice are stated.

**Design.**
- `configs/stats.yaml` (new; the D2 values): `min_effect: {error: 0.02, probability: 0.02,
  auroc: 0.02, gap_closed: 0.10}`.
- New columns, appended so existing columns don't change: `meaningful` (effect ≥ threshold) and
  `clearly` (significant after Holm *and* meaningful), in `h1_test`, `negation_gap`,
  `negmark_effect`, `name_effect`, `typo_direction`, `early_warning`, `shift_decomposition`.
- For H4, an equivalence-style statement: "combined − confidence is within ±0.02 for every model"
  turns "no evidence it helps" into "evidence it doesn't help more than 0.02".

**Tests:** threshold logic on hand-made rows; existing columns identical.

**Effort:** half a day.

---

## C12. RoBERTa-large's possible contamination: mark it everywhere, then 🟢

**Problem.** Its SST-2 and IMDb accuracy can't be trusted; there is no way to check its training
data.

**Done when:** every table, figure and slide that shows its accuracy carries the † mark and the
one-line reason; the shortfall moves to 🟢 (a stated limitation).

**Design.**
- Audit list: README table (done), hypothesisanswers H1/H4 tables, interestingfindings §1, §4, §8,
  §12, the Mirage Gap and fingerprint figures, the demo's "looks → behaves" column.
- `plots.py`: an `ACCURACY_CAVEAT = {"roberta_large": "†"}` map, appended to its label on every
  chart that plots accuracy; a footnote line on those figures.
- Demo: the same mark in the "looks → behaves" cell, with a hover note.

**Effort:** one hour.

---

## D10. Paraphrased frames and a fresh draw for the behavioral suite

**Problem.** Every template result comes from one set of frames written by us and one random draw
(seed 42); a frame's own wording could drive a failure rate.

**Done when:** every case records its frame; each test has a second frame set; a table gives each
test's failure rate per frame and per frame set, with cluster CIs, and a heterogeneity test says
whether any frame stands out.

**Design.**
- `templates.py`: every generator returns the frame id with each case; the suite gets a `frame`
  column (appended; existing cases unchanged, tested).
- **Frame set B:** for each test, 3–5 new frames with the same slots, e.g. for negation "I found
  the {noun} {adj}." / "I did not find the {noun} {adj}."; for names "{name}'s latest {noun} was
  {adj}.". Written or approved by you (D5), stored in `configs/behavioral.yaml → frames_b`.
- **Fresh suite** (`behavioral_v2`, seed 2026, frame set B): a side probe, outside `EVAL_SETS`,
  built and predicted by a new stage `replicate` (shared with G8); predictions through the contract;
  all eight models on the CPU (short sentences: minutes, RoBERTa-large included, as for the name
  probe).
- **Output:** `results/behavioral_by_frame.csv`: model, test, frame set, frame, n, failure rate,
  cluster CI; plus a permutation test across frames per model and test (Holm within test).

**Tests:** frame ids recorded; the original suite unchanged; frame set B generates the same slot
types and counts.

**Effort:** one day, plus your frames (30 min).

---

## G8. Pre-registered claims, tested on untouched data

**Problem.** Many analyses were designed after the first results, on the same data. Nothing has
been confirmed on data the analysis never touched.

**Done when:** a committed list of claims, written before the new data is run, each with its
measure, direction and pass rule; a table says which replicated, on a fresh behavioral suite and a
fresh IMDb sample; the findings files report it.

**Design.**
1. **Pre-register** `docs/confirmatory_claims.md` (D6): about 15 claims, for example:
   - H1: every transformer's typo error is ≥ 10 points above its plain-sentence error;
   - H2: every single-word classical model's directional-negation error is ≥ 10 points above
     DistilBERT's;
   - typo default: DistilBERT flips more positive than negative typo pairs, and 0–2% of noun-typo
     pairs;
   - names: Multinomial NB flips ≥ 20% of name pairs, both DistilBERTs ≤ 2%;
   - H4: on fresh IMDb, confidence AUROC exceeds distance AUROC by ≥ 0.1 for every transformer;
   - the leaderboard twins: LR and Multinomial NB within 2 points of accuracy on fresh IMDb, with
     Multinomial NB's typo + name failure rate higher;
   - shift: on fresh IMDb, full reviews are easier than their edges for every model.
   Each with its pass rule (direction and threshold, judged on the cluster CI). Committed, and the
   commit hash recorded, before step 3.
2. **Fresh data:**
   - `behavioral_v2` (D10);
   - `imdb_fresh`: 10,000 reviews from the 15,000 IMDb test reviews not in the current sample, built
     by `prepare` with the same capping and an `imdb_fresh_edges` version; a side set outside
     `EVAL_SETS`. The classical models predict locally; the transformers on Kaggle (inference
     notebook).
   - Embedding distances for `imdb_fresh` (CPU, cached).
3. **Stage `replicate`** evaluates every claim with the same functions as the original analyses →
   `results/replication.csv`: claim, original estimate, new estimate, cluster CI, pass rule,
   `replicated`.

**Reporting:** interestingfindings.md gets a "Replicated?" column in its overview;
hypothesisanswers.md a line per hypothesis. A claim that fails is reported as such.

**Effort:** one day, plus one Kaggle inference run.

---

## C4. More seeds: DistilBERT and the seed-dependent classical models

**Problem.** DistilBERT has 3 seeds; the random forest, the decision tree (tie-breaking) and
Gaussian NB (its SVD) depend on a seed but have one.

**Done when:** DistilBERT has 5 seeds in the seed-robustness table; RF, DT and GNB have 3 seeds with
their spread reported; the headline tables are unchanged (H8).

**Design.**
- **DistilBERT:** `configs/distilbert.yaml → robustness_seeds: [7, 101]`, trained on Kaggle with the
  current settings (training notebook). `seed_robustness` extends to all 5 seeds; the headline stays
  on 3 (D4).
- **Classical:** `train-classical --models rf dtree gnb --variants base unigram negmark --seeds 13 87
  --no-tune`: refit with the hyperparameters chosen for seed 42 (`tuning_classical.csv`), so only the
  seed changes. Logistic regression and Multinomial NB are deterministic and skipped.
  - Disk: RF is 822 MB per fit; save these with `joblib` compression (H5 in shortfalls), or keep
    only their predictions.
- **Output:** `results/classical_seed_spread.csv`: model, variant, metric (SST-2 test accuracy,
  real-negation accuracy, IMDb accuracy, robustness, typo and name failure rates), value per seed,
  range, and whether any verdict changes.

**Tests:** the `--seeds` option writes seed-named model dirs and predictions; H8's test covers the
headline tables.

**Effort:** half a day, plus about 1 h of Kaggle GPU and 40 min of local CPU.

---

## C3. A tuned DistilBERT as a robustness check

**Problem.** DistilBERT uses fixed settings (lr 2e-5, 3 epochs), while the classical models are
tuned.

**Done when:** a small grid has been searched on the held-out set; the best setting is trained on
the three headline seeds as a `tuned` variant; a table shows whether any H1–H3 conclusion changes
with it as the reference.

**Design.**
- **Grid:** learning rate {2e-5, 3e-5, 5e-5} × 4 epochs, evaluated on the sentence-grouped held-out
  set after every epoch (the trainer already evaluates per epoch). That gives 3 runs instead of 9,
  with epochs 2, 3 and 4 read from the same run. Caveat: the learning-rate schedule depends on the
  total number of epochs, so "epoch 2 of 4" approximates a 2-epoch run; stated in the decision entry.
- Selection by held-out accuracy (as for the classical models), never by validation or test.
- `train-distilbert --variant tuned --lr … --epochs …`; output `results/tuning_distilbert.csv`.
- **Variant `tuned`:** seeds 13, 42, 87; predictions on every eval set and the behavioral suite
  (training notebook). In `models.base`, `ROBUSTNESS_VARIANTS = {"tuned"}`, which `holm_family`
  puts in its own family, so no existing p-value moves.
- **Output:** `results/distilbert_tuning_check.csv`: every H1, H2, H3 and Mirage Gap number with the
  tuned model next to the current one, and `changed`.

**Tests:** CLI arguments reach the config; the `tuned` variant lands in its own Holm family; H8's
test covers the headline tables.

**Effort:** half a day, plus about 3 h of Kaggle GPU (3 grid runs + 3 seeds).

---

## F2. Distance in each classifier's own feature space

**Problem.** The early-warning distance uses MiniLM's embeddings for every model, so it can only
flag "unusual text", not "unusual for this model".

**Done when:** for DistilBERT and for the classical models, a Mahalanobis distance in the model's
own features is scored for IMDb error detection next to confidence, in its own table and Holm
family.

**Design.**
- **DistilBERT** (median seed): the 768-d pre-classifier features (`pre_classifier` + ReLU, the
  layer the classification head reads) for the SST-2 training phrases (128 tokens) and the 10,000
  IMDb reviews (512 tokens, head + tail), extracted on Kaggle (inference notebook), saved as
  float16 `.npz` (about 100 MB) in the output zip.
- **Classical models:** their TF-IDF features reduced to 300 SVD components (fitted on training
  only), on the CPU.
- **Score:** class-conditional Gaussian with Ledoit-Wolf covariance, the minimum distance over the
  two classes (Lee et al., 2018), using the existing `GaussianDensity(class_conditional=True)`.
- **Output:** `results/early_warning_own.csv`: AUROC of own-feature distance, low confidence and
  both combined (out of fold), paired differences, Holm within this table.

**Tests:** the class-conditional score on a synthetic two-Gaussian set; the feature extractor on a
tiny random BERT (no download, as in `test_external.py`).

**Effort:** one day, plus about 30 min of Kaggle GPU.

---

## E1. Integrated Gradients: a second baseline, every failure, and a check against occlusion

**Problem.** IG uses one baseline (`[PAD]`) and at most 100 examples; its word claims rest on that
choice.

**Done when:** IG runs with two baselines on all of DistilBERT's SST-2 test and real-negation
failures; their agreement and IG's agreement with leave-one-word-out are reported; the convergence
delta is under 0.05 for every example.

**Design.**
- `explain.ig.attribute(..., baseline="pad" | "mask")`; `configs/explain.yaml → ig: {baselines:
  [pad, mask], n_steps: 50, max_examples: null}` (null = all failures).
- Run on Kaggle (inference notebook), median seed; output `results/explain/ig_failures_mask.csv`
  next to the existing file.
- **Outputs:**
  - `results/explain/ig_baseline_agreement.csv`: per example, the Spearman correlation of word
    attributions between baselines; overall, top-50 word overlap (Jaccard, RBO); convergence deltas;
  - `results/explain/ig_vs_occlusion.csv`: the same comparison between IG and leave-one-word-out
    for DistilBERT, on the same sentences.
- If any example's delta is ≥ 0.05, rerun it with `n_steps: 200`.

**Reporting:** since H5 now uses leave-one-word-out (D43), IG is for illustration; this tells the
report how far those illustrations can be trusted.

**Effort:** half a day, plus about 30 min of Kaggle GPU.

---

## B10. Hand-label the IMDb edges to measure their label noise

**Problem.** An excerpt of a review's first and last sentence keeps the whole review's label, but
often doesn't state it, so the "content" part of the IMDb drop is an upper bound.

**Done when:** 200 excerpts are labelled by you, blind; the noise rate has a CI; the shift
decomposition is re-estimated on the clearly labelled excerpts; interestingfindings.md §8 reports
the corrected numbers.

**Design.**
1. Claude writes `results/labelling/edges_sample.csv`: 200 excerpts from `imdb_edges`, stratified
   100 positive / 100 negative by review label, random within each (seed 42). Columns: `item_id`,
   `text`, `your_label` (empty). The review label and every model's prediction are left out, so you
   label blind.
2. **You** fill `your_label` with `pos`, `neg` or `unclear` (about 1 hour; 20 seconds each).
   Optionally a second person labels 50 of them, for an agreement score (Cohen's kappa).
3. `evaluate.edges_label_noise` → `results/edges_label_noise.csv`:
   - noise rate = share `unclear` or contradicting the review label, with a CI;
   - every model's edges error on the clear excerpts against all 200;
   - the content and length parts of the shift recomputed on the clear subset, with CIs (wider:
     n ≈ 150), and the full-sample estimate corrected for the noise rate.

**Tests:** the noise computation and the corrected decomposition on hand-made labels.

**Effort:** 1 hour of yours; half a day of Claude's.

---

## The two Kaggle notebooks (Phase 3)

Built after Phases 1–2 are pushed, so they run the new code. Each zips its outputs for you to unzip
into the repo root, as before.

| Notebook | Does | GPU time |
|---|---|---|
| `notebooks/kaggle_fixes_train.ipynb` | C3: 3 grid runs (4 epochs each), then the best setting on seeds 13 / 42 / 87 as `tuned`, with predictions; C4: base DistilBERT on seeds 7 and 101, with predictions | about 4–5 h |
| `notebooks/kaggle_fixes_infer.ipynb` | G8: every transformer (our DistilBERT's 3 seeds, HF DistilBERT, RoBERTa-large) on `imdb_fresh` and its edges; F2: DistilBERT's own features; E1: IG with both baselines on all failures | about 1–1.5 h |

Both stay well inside Kaggle's free weekly quota (about 30 GPU hours) and its 12-hour session
limit. The training notebook will save its outputs after each run, so a stopped session can resume
where it left off.

---

## After everything (Phase 4)

1. Unzip both Kaggle outputs; Claude checks them against the local data (as for D46).
2. Re-run `evaluate`, `warn`, `explain`, `replicate` and `figures`.
3. `artifacts/unchanged_check.py`: every earlier number unchanged, except the G7 intervals if D1
   says so.
4. `pytest`, including the claims registry.
5. Update hypothesisanswers.md and interestingfindings.md (new numbers, the replication column,
   any changed verdict), README's summary, shortfalls.md (each item to Resolved), majordecisions.md
   (one entry per fix), codeexplanations.md and whatyouneedtodo.md.
6. One code review of all the changes.

**Total:** about 6–7 days of Claude's work, 3–4 hours of yours, and about 6 hours of unattended
Kaggle GPU.
