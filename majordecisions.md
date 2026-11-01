# Major decisions

A running log of the major coding decisions taken while building the Mirage code skeleton, and why.
Newest decisions are appended at the bottom. Each entry says what was decided, the alternatives,
and the reason.

---

## D1 — `src/` layout with a pip-installable package

**Decision:** Code lives in `src/mirage/` and is installed with `pip install -e .`, configured in
`pyproject.toml`.

**Alternatives:** loose scripts in `scripts/`; a flat `mirage/` package at the repo root.

**Why:** the same package must run on the laptop *and* on Kaggle. On Kaggle the notebook just runs
`pip install -e .` and calls `python -m mirage train-distilbert`, with no copy-pasted code. The
`src/` layout also stops tests from accidentally importing the uncompiled working copy instead of
the installed package.

## D2 — The deep-learning stack is an optional extra, imported lazily

**Decision:** `torch`, `transformers`, `datasets`, `sentence-transformers` and `captum` sit in the
`[dl]` extra. Modules that need them import them *inside functions*, never at module top level.

**Alternatives:** one flat dependency list.

**Why:** torch alone is ~2 GB. The classical models, metrics, statistics, behavioral suite and the
whole test suite must run on a bare CPU install in seconds. Lazy imports mean `import mirage.cli`
never pulls in torch unless a DistilBERT stage is actually run.

## D3 — One YAML file per stage, loaded as plain dicts

**Decision:** `configs/{data,classical,distilbert,behavioral,warning}.yaml`, read by
`mirage.config.load_config(name)` into a plain `dict`.

**Alternatives:** Hydra / OmegaConf; typed dataclasses per config.

**Why:** the configs are small and flat, and the proposal only requires "config files for every
run". Hydra adds a framework to learn in a 4-week project. Plain dicts keep the configs readable,
and each run copies its resolved config next to its outputs, which is what reproducibility needs.

## D4 — All file locations come from one `paths.py`

**Decision:** every path (`artifacts/data`, `artifacts/models`, `results/predictions`, `figures`, ...)
is defined once in `mirage.paths`, and the root can be overridden with the `MIRAGE_ROOT` env var.

**Why:** stages hand data to each other only through files (see D7), so a single source of truth
for where those files live prevents silent mismatches. `MIRAGE_ROOT` lets the Kaggle notebook write
to `/kaggle/working` without code changes.

## D5 — Normalize every input into SST-2's own tokenization

**Decision:** `mirage.text.normalize` lowercases and rewrites text into SST-2 style:
`doesn't → does n't`, `can't → ca n't`, `it's → it 's`, punctuation split into its own tokens,
`<br />` removed, curly quotes straightened. It is idempotent on SST text.

**Alternatives:** leave SST as-is and "un-tokenize" it into normal English; or use each model's own
tokenizer with no shared normalization.

**Why:** SST-2 is the training data, so matching *its* format means nothing in the training
vocabulary changes. If templates kept `isn't`, a TF-IDF model trained on `is n't` would see an
unknown word and the negation tests would measure a formatting bug, not shortcut learning.
**Screened:** run on contractions, ellipses, `--`, `<br />` and curly quotes; output matched SST
style and `normalize(normalize(x)) == normalize(x)` held for every case.

## D6 — Negation marking: scope ends at punctuation *or end of text*, and restarts on a new negation

**Decision:** after a negation word (`not, never, no, nothing, n't`), every token gets a `NOT_`
prefix until one of `. , ! ? ; : ... --` or the end of the text. The negation word itself is not
prefixed, and a second negation word restarts the scope. `NOT_` stays uppercase because
`normalize` has already lowercased everything and vectorizers use `lowercase=False`.

**Alternatives:** a fixed window (e.g. next 3 words); ending scope at "but" as well.

**Why:** this is the Pang et al. (2002) rule the proposal specifies. "End of text" matters because
most SST-2 *training* items are short phrases with no final punctuation. A fixed window was rejected
because the proposal names the punctuation rule; "but" was left out to stay faithful to the
original method (easy to add later as a config option and report as an ablation).

## D7 — Predictions are the contract between stages

**Decision:** every (model, variant, seed, dataset) produces one parquet file in
`results/predictions/` with a fixed schema (`item_id, text, label, p_pos, pred, correct`, plus
dataset-specific columns). All metrics, statistics, plots and the early-warning analysis read only
these files. `mirage.predict.validate_predictions` checks the schema.

**Alternatives:** each analysis loads models and predicts on the fly.

**Why:** DistilBERT is trained on Kaggle and everything else runs on a laptop. With predictions as
files, a GPU is needed only for training; every table can be rebuilt in seconds; and a metric bug
can be fixed without re-running any model.

## D8 — Preprocessing lives inside the sklearn vectorizer; default token pattern replaced

**Decision:** classical pipelines use `Vectorizer(preprocessor=Preprocessor(negmark), tokenizer=tokens,
token_pattern=None, lowercase=False, stop_words=None)`.

**Alternatives:** preprocess the text before fitting; use sklearn defaults.

**Why:** inside the vectorizer means a saved model takes **raw** text, so it can't be fed
un-normalized or double-normalized input by mistake. The default pattern `\b\w\w+\b` would turn
`n't` into nothing, and stop-word lists contain `not`, so the defaults would silently erase the
very features H2/H3 measure. `Preprocessor` and `tokens` are module-level so the pipeline pickles.

## D9 — One `Classifier` interface; `long_text` flag instead of model-specific branches

**Decision:** every model exposes `predict_proba(texts, long_text=False) -> (n, 2)` with column 1 =
P(positive). sklearn models are wrapped in `SklearnClassifier` (which also forces column order to
`[0, 1]`); DistilBERT in `DistilBertClassifier`. `load_classifier(dir)` loads either kind.

**Alternatives:** separate code paths per model family in every stage.

**Why:** `predict.run_predictions` can then treat all six models the same. IMDb needs DistilBERT's
256-token limit instead of 128, so the caller passes `long_text=True` for IMDb and each model decides
what that means (sklearn ignores it). This keeps model-specific `if model == "distilbert"` checks
out of the pipeline.

## D10 — Train on 95%, select on the 5% held-out, no refit

**Decision:** classical grid search fits each grid point on the 95% split, scores held-out accuracy,
and keeps the best fitted model as-is. Ties keep the first grid point. Every run is logged to
`results/tuning_classical.csv` with fit time.

**Alternatives:** refit the winner on train + held-out; k-fold CV.

**Why:** DistilBERT picks its best epoch on the same held-out split and never trains on it.
Refitting only the classical models would give them ~3k more examples and blur the comparison.
k-fold CV over 67k examples × RF(300 trees) would cost hours for little gain. Fit times are logged
because "when is a cheap model good enough" (RQ2) needs a cost number too.
**Screened:** toy run of all 5 models with negation marking; save → load → predict on raw text
worked, and `NOT_good` appears in the vocabulary. Also dropped `penalty="l2"` (deprecated in
scikit-learn 1.8; L2 is the default).

## D11 — Behavioral suite: MFT vs INV scoring, negated negatives reported but not scored

**Decision:** each row has `kind` (MFT/INV), `case_id`, `role` and `scored`. MFT failure = wrong
label; INV failure = the prediction changes between the `original` and `perturbed` rows of one
case. Every row still carries the intended sentiment as `label`, so the suite fits the predictions
contract unchanged. "not <negative>" rows have `scored=False`. Counts are 100 simple, 100 + 50
negation, 100 typo pairs, 50 name-swap pairs = **400 cases / 550 rows**. Generation is
deterministic: sampling without replacement from the template × noun × adjective product with a
fixed seed. Typos only touch adjectives of 4+ letters.

**Why:** this is the CheckList (Ribeiro et al., 2020) split between capability tests and invariance
tests. "not bad" has no clean expected label, so counting it as a failure would inflate H1.
**Screened:** an "always positive" fake model gave exactly 100% negation failure, 50% simple, 0% INV;
two builds were identical; no duplicate MFT texts. A first version derived `case_id` from row
positions. That was fragile, so each generator now assigns `case_id` itself.

## D12 — Paired item-level bootstrap for every gap; the "gap closed" ratio is secondary

**Decision:** `stats.bootstrap(stat, n_items)` resamples item indices and recomputes any statistic.
Gaps (classical − DistilBERT negation error) resample both models' aligned items together. H3 is
reported first as the two paired gaps (before vs after negation marking) with CIs, and second as the
ratio `(gap_base − gap_negmark) / gap_base`.

**Why:** pairing removes item-difficulty variance, so the CIs are much tighter than two independent
CIs. **Screened:** on synthetic predictions the ratio's CI ran from −2.8 to 5.1 because the base gap
was small. A ratio of two noisy small numbers is unstable, so the report must lead with the gaps,
not the ratio. McNemar uses the exact binomial test when there are fewer than 25 discordant pairs,
which is expected on the small negation subset.

## D13 — Seed policy: means over seeds in tables, the median seed for per-item analyses

**Decision:** `evaluate.main_table` averages DistilBERT over its 3 seeds (mean ± std).
`evaluate.canonical()` keeps only DistilBERT's *median-accuracy* seed (on SST-2 val) for error
tables, gaps, McNemar, IG and failure examples. Classical models have one seed (42); RF is the only
randomised one, and its seed is fixed.

**Why:** McNemar and paired gaps need one prediction per item. Choosing the median seed (instead of
the best) avoids flattering DistilBERT. The std over seeds is still reported in the main table.

## D14 — IG attributions are summed back onto the *same* words the classical models see

**Decision:** `explain.ig.attribute` tokenizes `normalize(text).split()` with
`is_split_into_words=True` and sums word-piece attributions per word through `word_ids()`. The
baseline is `[CLS] [PAD]… [SEP]`, attributions target the *predicted* class, and Captum's
convergence delta is stored per sentence.

**Alternatives:** attribute raw word-pieces; let BERT's own pre-tokenizer define words (it splits
`n't` into `n ' t`).

**Why:** H5 compares DistilBERT's top words with LR/MNB/RF features, and the deletion check
removes whole words. Both only work if a "word" means the same thing for every model. The deletion
check re-joins the remaining normalized words and re-predicts, which is safe because `normalize` is
idempotent (D5). A random-k deletion control (10 draws) is built in, since "the prediction changed"
means little without a baseline.

## D15 — Figures use one validated colorblind-safe palette; models keep fixed colors

**Decision:** `plots.py` uses the validated categorical order from the `dataviz` skill (blue, orange,
aqua, yellow, magenta, green, …). Each model has a fixed slot everywhere; base vs negmark are always
blue vs orange; magnitudes (heatmaps) use one blue ramp light → dark; attribution strips use a
red ↔ gray ↔ blue diverging map with a neutral midpoint. Every figure is written as PNG (slides)
and PDF (report). The backend is `Agg` so figures render headless on Kaggle.

**Why:** "color follows the entity, never its rank": a reader who learns "orange = negmark" in one
figure can rely on it in all of them. A rainbow colormap would make error magnitudes hard to compare.
**Screened:** rendered the error heatmap and reliability diagram from a synthetic run and looked at
them. The first heatmap sorted its columns alphabetically and wasted space because of a fixed aspect
ratio. Fixed with an explicit `TEST_ORDER` (standard tests, then behavioral probes) and
`aspect="auto"`. Cell text now switches to white relative to the color scale, not at a fixed value.

## D16 — Shortcut-word overlap (H5): unigrams only, linear models ranked by |weight|

**Decision:** `explain.overlap` compares top-50 *unigram* rankings with Jaccard@50 and rank-biased
overlap (RBO, p = 0.9). LR and MNB are ranked by |score|, so strong negative words count as
shortcuts too. RF and DTree are ranked by impurity importance, and DistilBERT by mean |IG| over its
failures (words seen ≥ 2 times). GNB is excluded because its SVD components have no word identity.
`NOT_` features are kept as distinct words.

**Why:** IG produces words, not bigrams, so comparing bigram features would be unfair to DistilBERT.
Jaccard ignores rank; RBO weights the top of the lists. Reporting both shows whether models share
the *strongest* shortcuts or only the long tail.
**Screened:** on the synthetic run, Jaccard was 1.0 for every pair (tiny vocabulary) while RBO
ranged from 0.28 to 0.59. That is exactly why both are reported.

## D17 — One CLI, file-driven stages; `all` excludes data download and training

**Decision:** `python -m mirage <stage>` with stages `prepare, train-classical, train-distilbert,
build-suite, predict, evaluate, explain, warn, figures, all`. `predict` finds every saved model under
`artifacts/models/` by its directory name (`{model}__{variant}__seed{seed}`). `figures` draws only
the figures whose input CSVs exist. `all` runs everything *after* training.
`train-distilbert --limit N` runs a tiny smoke test.

**Alternatives:** numbered scripts; a Makefile (not available on Windows by default); Snakemake.

**Why:** stages are idempotent and skip what isn't ready, so you can run `figures` in week 2 with
only part of the results. Download and GPU training are deliberately separate manual steps.
**Screened:** ran `train-classical (base + negmark) → build-suite → predict → evaluate →
explain --skip-ig → figures` end to end on synthetic SST-style data in a temp `MIRAGE_ROOT`. All
outputs were produced, and negation marking cut logreg's negation-test error from 22% to 4%.

## D18 — Normalization revised after screening on real SST-2: abbreviation periods stay attached

**Decision:** periods are now split per token. A sentence-final period becomes its own token, but
initials (`j.`, `l.a.`, `p.o.w.`) and a short abbreviation list (`mr. mrs. dr. st. bros. etc. al.`
…) keep theirs, as SST-2 does. Commas inside numbers (`10,000`) stay. Glued IMDb sentences
(`great.The`) are split.

**How it was found:** after `prepare` downloaded the real data, `normalize` was run on all 63,981
training phrases and 872 validation sentences. The first version changed **544** training phrases
(`mr. taylor → mr . taylor`). That alters vocabulary, and the stray `.` would **end a negation scope
mid-phrase** ("not mr . taylor 's best"), which corrupts the negation-marking experiment (H3).
After the fix: **45 / 63,981** training phrases change (0.07%, leftovers such as `!?` → `! ?` and
`.com`, which are harmless), validation changes **0 / 872**, and `normalize` is idempotent on every
real sentence. Six regression cases were added to `tests/test_text.py`.

**Real data facts recorded at the same time** (`results/data_stats.csv`): train 63,981 / held-out
3,368 / val 872 / **negation subset 180 (36% positive)** / IMDb sample 2,000 (median 201 tokens
vs 7 for SST-2 train phrases). The negation subset is large enough for CIs, but it is skewed
negative, so its error rates must be compared against its own majority baseline (64%).

## D19 — Built for transformers 5.x, checkpoints deleted after training

**Decision:** the Trainer setup passes only arguments that are stable across transformers 4.41+ and
5.x: `eval_strategy`, explicit `warmup_steps` (not `warmup_ratio`), `DataCollatorWithPadding`
instead of the deprecated `tokenizer=` argument. After training, the best model is saved and
`checkpoints/` is deleted.

**Why / screened:** the local install resolved to transformers **5.22**. A 256-example CPU
fine-tune ran all 3 epochs, chose the best epoch, and saved `model.safetensors`, tokenizer and
`train_summary.json`. Without the cleanup, each seed's folder kept an extra full checkpoint
(~260 MB × 3 seeds) for nothing.

## D20 — Smoke test of the full GPU-side code on CPU; two fixes from it

**What ran:** with the deep-learning stack installed locally (torch 2.15 CPU, transformers 5.22,
sentence-transformers 6.2, captum 0.9), in an isolated `MIRAGE_ROOT`, on real data (training
subsampled to 3,000 phrases): DistilBERT fine-tune (256 examples) → `predict` for all eval sets →
`evaluate` → `explain` with IG on 6 failures + deletion check → `warn` with real MiniLM embeddings
→ `figures` (14 figures). IG's max convergence delta was **0.002**, well under 0.05, so 50 steps
are enough.

**Fix 1 — `warn` crashed.** It loaded IMDb predictions only and then asked for DistilBERT's median
seed, which is chosen on SST-2 val rows. `warn` now selects seeds from all predictions *before*
filtering to IMDb, and `median_seed` raises a clear error instead of an `IndexError`.

**Fix 2 — the "combined" early-warning score now uses log-odds of confidence.** For MNB, the
combined predictor scored *below* confidence alone (AUROC 0.581 vs 0.656). Naive Bayes
probabilities on long reviews pile up near 1.0, so raw confidence is almost constant and a linear
combiner can't use it. With `log(c / (1 − c))` the combined score went to 0.654, back in line with
its best input, as it should be. LR was unchanged (0.698 → 0.699).

**Also changed:** the IG sample now takes half its budget from real SST-2 failures and half from
behavioral failures. It never draws from the held-out tuning split. Previously the 100-item cap
could fill up with behavioral items because files are read in alphabetical order.

## D21 — Held-out split groups phrases by their sentence (fixes shortfalls.md B1)

**Problem found:** SST-2's training set has one row per parse-tree *phrase*, so a sentence and all
its sub-phrases are separate rows. The old split (`train_test_split` on rows) left the parent
sentence in training for most held-out phrases. On the real data, **89.6%** of held-out phrases
appeared verbatim inside a training phrase. Held-out accuracy was inflated, and since it drives
every classical grid search and DistilBERT's best-epoch choice, model selection favoured
memorization: deeper trees, weaker regularization, later epochs.

**Decision:** `data.heldout_mask` finds the *roots* (phrases contained in no other phrase, on token
boundaries), shuffles them with the data seed, and holds out the first *k*. A phrase is held out
only if **every** root containing it is held out. Common phrases such as "the film" therefore stay
in training, and by construction no held-out phrase occurs inside any training phrase. *k* is the
smallest number of roots that reaches 5% of rows. Containment is found with a token posting-list
index (intersect the posting lists of a phrase's tokens, then substring-check the few candidates),
which takes about 2 s on 67k phrases.

**Alternatives rejected:**
- Holding out only full sentences: the held-out set would be all long text, unlike training.
- Recovering the original SST trees from the PTB files: needs another download, and SST-2's
  phrase strings don't map one-to-one onto them.
- `GroupShuffleSplit` with one group per phrase's *first* containing root: a phrase inside two
  sentences could still leak.

**Screened on real data:** leakage dropped from **89.6% to 0 of 3,378** held-out phrases. Size
3,378 vs 3,368 before. Positive share 0.565 vs 0.557 in training. Mean length 9.8 vs 9.4 words, so
the held-out set still looks like the training phrases. Validation (the reported numbers) never
leaked, so already-reported accuracies don't move; only which model settings get chosen can
change. A new test builds nested phrases and checks all-or-nothing per sentence plus zero leakage
over 20 seeds.

**Consequence:** expect held-out accuracy to drop a little compared with the old split. That's the
honest number. Any DistilBERT run made before this change (the Kaggle smoke test) used the old
split; the full run must use this commit or later.

## D22 — Scaled up the evaluation; SST-2 test sentences added from the public Stanford labels

**Problem:** several results rested on small samples (shortfalls.md B2, B3, B5, D2, F3). The worst
was the 180-sentence negation subset: about ±7 points per model, too wide to test H3 (does
negation marking close the gap?). Validation was also both the set used for choices
(DistilBERT's median seed) and the reported test set.

**Decision:**
- **SST-2 test set added** from `SetFit/sst2`, which carries the original public Stanford labels
  that GLUE hides. Screened before use:
  - Its validation sentences match ours with **100%** label agreement (823 of 872 matched text
    exactly; the rest differ only in encoding).
  - Its test split is balanced: 912 negative, 909 positive.
  - One sentence also appears in training and is dropped by `drop_seen`, leaving 1,820.
- `sst2_val` is now for **choices only**: DistilBERT's median seed and picking the best classical
  model for the reliability figure. `sst2_test` is the **reported** in-distribution set
  (`evaluate.INDIST`).
- **Negation subset pools validation and test:** 180 → **519** sentences (36% positive). The
  margin of error per model falls from about ±7 to about ±4 points.
- **IMDb sample:** 2,000 → **10,000** reviews (config only), so H4 has about 5× more errors.
- **Behavioral suite:** 5× larger, from 400 to **2,000 cases** (2,750 rows). Every generator
  still samples distinct combinations; the largest request (500 of 1,200) is well inside the pool.

**Rejected:** writing new negated sentences by hand (slow, and our own bias, see D1 in
shortfalls). Pulling negated sentences from SST-2 training was also rejected: models are trained on
them.

## D23 — IMDb text is capped to head + tail words for every model; DistilBERT reads up to 512 tokens (C1)

**Problem:** DistilBERT read only the first 256 tokens of each IMDb review, while the TF-IDF models
read the whole review. Reviews often give the verdict at the end, so the IMDb comparison mixed
"shortcut failure" with "DistilBERT saw less text".

**Decision:**
- **One rule for everyone:** `prepare` caps each IMDb review to its first 220 and last 140 words
  (`head_tail`, configured in `data.yaml`). All models read that same text. `n_words` and
  `capped` record what was cut. On the real 10,000-review sample, **15.9%** of reviews are capped
  (median length 173 words).
- **DistilBERT's IMDb limit goes from 256 to 512 tokens.** 360 words fit in about 512 word pieces.
  For the rare review that still doesn't fit, `encode` keeps the first and last tokens
  (`head_tail_ids`) instead of cutting the ending. Screened with a 905-token review: the batch
  keeps `[CLS] start …` and `… the ending was awful [SEP]`.
- **Prediction moved to the Kaggle GPU:** 10,000 reviews at 512 tokens × 3 seeds is slow on a
  laptop CPU. The notebook now runs `build-suite` and `predict --models distilbert` and zips the
  models and predictions together. Locally, `predict` skips any model whose prediction files are
  newer than its model files (`--overwrite` forces a redo). Retraining makes the model newer, so
  stale predictions are never kept.

**Rejected:**
- A separate truncated IMDb control set: one more set to explain, when making the text identical
  answers the question directly.
- A sliding window over the whole review: several forward passes per review, and its
  probabilities aren't comparable with a single pass.

## D24 — Behavioral control for template artefacts (D1)

**Problem:** we wrote the templates and word lists ourselves. A model could fail "The movie was not
brilliant." because it misreads *brilliant* or the template itself, not because of negation.

**Decision:** every behavioral row now carries `word` (the sentiment adjective it's built on).
`evaluate` writes two new tables:
- `word_errors.csv`: failure rate per model × test × word, so one odd adjective can't drive a
  result, and the word lists can be audited.
- `negation_control.csv`: negation error on **all** positive adjectives, and on **known words**
  only (adjectives the model gets right in *every* simple-template case). If the error stays high
  on known words, the failure is about negation. If it vanishes, it was about the words or the
  templates.

The human review of the word lists stays on the user's list (`whatyouneedtodo.md`). Code can make
the bias visible, but can't remove it.

## D25 — Early-warning distance is scored per sentence (max), with review length as a baseline (F1)

**Problem:** the smoke test's sanity check, which asks whether distance tells SST-2 validation
from IMDb, gave AUROC **0.646**. If distance can't detect a change of domain that obvious, H4's
"distance vs confidence" comparison is set up to fail, and a negative result would reflect the
method, not the hypothesis.

**Experiment** (full 64k-phrase SST-2 training embeddings, 2,000 IMDb reviews split into 24,052
sentences, MiniLM). Sanity AUROC, SST-2 val vs IMDb:

| distance | whole review | sentence mean | **sentence max** |
|---|---|---|---|
| Gaussian (Mahalanobis, Ledoit-Wolf) | 0.668 | 0.714 | **0.950** |
| Gaussian on L2-normalised embeddings | 0.668 | 0.714 | 0.950 |
| class-conditional Gaussian | 0.668 | 0.713 | 0.950 |
| kNN cosine, k=10 | 0.688 | 0.718 | 0.953 |

**What it shows:**
- **Fitting on the full training set didn't help** (0.646 → 0.668).
- **The choice of distance doesn't matter** (a spread of 0.02). MiniLM already outputs unit-norm
  vectors, so L2 normalisation changes nothing.
- **What matters is what gets embedded.** Mean-pooling a 200-word review averages away its
  unusual parts, leaving a generic "movie talk" vector near SST-2's centre. Scoring each sentence
  and taking the review's **most unusual sentence** separates the domains (0.95).

**Decision:** `configs/warning.yaml` sets `distance: gaussian` (the method the proposal names;
kNN is no better) and `unit: sentence_max`. Both are switchable, and `warning/distance.py` and
`warning/knn.py` implement the alternatives.

**Caveat, built into the analysis:** the maximum over more sentences is larger by chance, so
sentence-max distance partly measures **length**. `compare_predictors` now also reports
**length** (word count) as an error predictor. H4 only counts as supported if distance beats
both confidence *and* length. The sanity check compares single SST-2 sentences with
multi-sentence reviews, so part of its 0.95 is also length. It shows the detector *can* see the
domain, not that it sees nothing else.

## D26 — First full end-to-end run on real data (H1); three fixes from screening it

**What ran:** `prepare` → `train-classical --variants base negmark` (5 models × 2 variants, about
15 min; the RF is the slow one at about 4 min per fit) → `all` (predict on all 6 evaluation sets,
evaluate, explain without IG, warn, figures). No crashes. DistilBERT is the only piece not yet run
at full scale; it needs the Kaggle run.

**Fix 1: LR's C grid was too narrow.** On the leak-free split, LR chose C=10, the edge of the grid
{0.1, 1, 10}. The grid is now {0.1, 1, 10, 30, 100}, and 10 is an interior optimum
(held-out 0.842 vs 0.837 at C=30).

**Fix 2: the "combined" early-warning score lost to confidence alone for MNB.**
- Diagnosis: **50.6%** of MNB's IMDb probabilities are exactly 1.0 in float64, and 78.6% are
  above 1 − 10⁻⁶. Clipping at 10⁻⁹ before log-odds merged thousands of distinct values, and the
  standardised linear fit let a pure-noise feature outweigh confidence. With random noise as the
  "distance", combined scored 0.637 against 0.688 for confidence alone.
- Fix: clip at 10⁻¹⁵ and rank-transform both features (`QuantileTransformer`, normal output)
  before the logistic combiner. With noise it now scores 0.676, within one CI half-width. With
  the real distance MNB scores 0.700 vs 0.688, and every model's combined score is at least its
  confidence score.

**Fix 3: `predict` skips models whose predictions are newer than the model files** (D23), so
Kaggle-made DistilBERT predictions aren't redone on a CPU.

**First results, classical models only (to be confirmed once DistilBERT is in):**
- **SST-2 test accuracy:** LR 0.835, MNB 0.834, RF 0.754, DT 0.661, GNB 0.591.
- **IMDb drop** (test − IMDb): 2–9 points.
- **Negation marking helps on real negated sentences:** LR 0.773 → 0.802, MNB 0.761 → 0.784.
  On templates the effect is much larger: LR's templated negation error falls from 0.262 to 0.008.
- **H4 (classical): distance does not flag errors.** With the new detector:
  - Sanity check (SST-2 test vs IMDb): AUROC **0.926**.
  - IMDb error-detection AUROC for distance: 0.49–0.53 for every model.
  - Low confidence: 0.67–0.77 for LR, MNB and RF.
  - Length: 0.50–0.54.
  - The detector clearly sees the domain change, but *within* IMDb, unusual reviews aren't the
    ones classical models get wrong.
- **Degenerate confidence:** the unpruned decision tree outputs only 0/1 probabilities (ECE 0.34,
  low_conf AUROC exactly 0.500). GNB is near chance (SST-2 test 0.591, majority rate 0.50).

## D27 — Directional negation test (fixes shortfalls.md D5)

**Problem:** the negation template ("The movie was not brilliant." → negative) gives a model that
leans "negative" a free pass. On real data the decision tree failed only 10% of negation cases
but 29% of plain ones. GNB was the mirror image.

**Decision:** a CheckList-style **DIR** test (`templates.negation_dir`). Each case is a pair of
twins with the same frame, noun and adjective, where only the negation differs:
"The movie was wonderful." / "The movie was not wonderful." It uses 4 paired frames:
not / isn't / don't think / always→never.

The question is whether adding "not" moves the model's *own* answer:
- **flip_failure**: among pairs whose affirmative sentence the model gets right
  (`applicable`), the share where the negated twin keeps the same label. A model that always says
  "negative" gets *no* applicable pairs instead of a perfect score. A unit test pins exactly
  this: the old test scores it 0% error, and the new one has nothing to score.
- **no_move**: share of all pairs where p_pos didn't move in the expected direction at all.
- **mean_shift**: average movement of p_pos toward the expected label.

500 positive-adjective pairs are scored. 250 negative-adjective pairs ("bad" / "not bad") are an
unscored probe, as in D11. The table is `negation_direction.csv`, and `beh_negation_dir` also
appears in the error heatmap and the negation-gap table.

**Result on the classical models (base):**

| model | old negation error | flip_failure | applicable | no_move | mean_shift |
|---|---|---|---|---|---|
| logreg | 0.262 | 0.266 | 0.96 | 0.00 | 0.58 |
| rf | 0.202 | 0.240 | 0.91 | 0.02 | 0.28 |
| mnb | 0.486 | 0.569 | 0.82 | 0.00 | 0.32 |
| dtree | 0.096 | **0.183** | **0.49** | **0.60** | 0.39 |
| gnb | 0.928 | 0.955 | 0.90 | 0.83 | −0.04 |

- **The decision tree's good negation score was mostly its negative lean.** Only half its pairs
  are testable, its error nearly doubles once measured fairly, and for 60% of pairs "not" doesn't
  change its probability at all.
- **LR's number barely moves** (it wasn't benefiting). "not" always pushes its probability the
  right way, but in about a quarter of pairs not far enough to flip the label.
- With `NOT_` marking, LR's flip_failure falls to 0.010 and MNB's to 0.214.

**Side effect:** the new generator draws from the suite's random stream before `typo` and
`name_swap`, so those cases were resampled. Their rates moved within noise (e.g. LR typo
0.478 → 0.45). The suite is now 2,750 cases (4,250 rows).

## D28 — Multiple-comparison correction (Holm) and p-values for every paired comparison (G1)

**Problem:** dozens of comparisons were judged one at a time at α = 0.05 (McNemar pairs, negation
gaps, AUROCs). By chance, about 1 in 20 true-null comparisons looks significant. H4 also had no
direct test: distance and confidence each had their own CI, but nothing compared them on the
same reviews.

**Decision:**
- **p-values for every paired comparison:** `bootstrap(..., null=0)` returns a two-sided p-value
  from the bootstrap standard error. Used by `paired_diff_ci`, `gap_closed_ci` and the new
  `paired_auroc_diff`. CIs stay percentile.
- **Holm–Bonferroni within each family** (`stats.holm`), one family per question:

| family | where | size now |
|---|---|---|
| McNemar model pairs | per dataset, `mcnemar.csv` → `p_holm`, `significant` | 10 per dataset |
| H2 base gaps / H3 gap-closed | per column, `negation_gap.csv` → `*_p_holm` | needs DistilBERT |
| H3 within each model: base − negmark error | `negmark_effect.csv` (new) → `p_holm` | 20 |
| H4: predictor − low_conf AUROC | per predictor across models, `early_warning.csv` → `diff_p_holm`, `beats_conf` | 10 |

- **New table, `negmark_effect.csv`:** H3 can now be tested before DistilBERT arrives. It
  reports base − negmark error on the same items for sst2_neg and the directional negation test,
  with sst2_test and IMDb as controls (marking shouldn't cost accuracy elsewhere).

**Bug caught while screening:** the first version counted resamples past the null. With 1,000
resamples that caps p at 2/1001 ≈ 0.002, and Holm over 20 tests puts the floor at **0.04**: every
strong effect showed exactly 0.0400. In a family of 25 or more, nothing could ever be
significant. The standard-error p-value has no floor, and a unit test now checks for one.

**What the correction changes (classical models):**
- **Negation marking works on templates, not clearly on real sentences.**
  - Directional test: significant for every model (LR −25.7 points, MNB −39.5).
  - Real negated sentences (sst2_neg, n = 519): significant only for GNB (−9.3).
  - LR −2.9 points [−0.2, +5.8], raw p 0.064, Holm 0.64.
  - The templates overstate the benefit (see shortfalls C10).
- **Side effects:** marking costs the decision tree 2.3 points on IMDb (significant). LR's
  +1.2 points on SST-2 test has a raw p of 0.013 but isn't significant after Holm, which is
  exactly the kind of claim the correction exists to stop.
- **H4:** distance is significantly *worse* than confidence for LR, MNB, RF and GNB, and no
  better for DT. "Combined" beats confidence only for MNB base (+0.012 AUROC, Holm p 0.003).
- **McNemar:** 35 of 40 model pairs differ significantly. LR vs MNB doesn't on any dataset.

**Rejected:** Bonferroni (needlessly conservative next to Holm); Benjamini–Hochberg FDR (it
controls a different error rate, and the report makes per-claim statements).

## D29 — Decision tree with minimum leaf size ≥ 10 (C9)

**Problem:** the tuned tree (`max_depth: null`, leaves of 1 example) was **2,956 levels deep**
and output only two probabilities, 0 and 1. Its calibration error was 0.339, and its confidence
carried no ranking information (early-warning AUROC exactly 0.500). Every calibration and
confidence-based result for the tree was meaningless.

**Experiment** (base variant, unpruned depth, varying `min_samples_leaf`):

| min leaf | held-out acc | test acc | test ECE | confidence AUROC (test errors) | distinct probabilities |
|---|---|---|---|---|---|
| 1 (old) | 0.686 | 0.661 | 0.339 | 0.500 | 2 |
| 10 | 0.662 | 0.659 | 0.182 | 0.599 | 114 |
| 20 | 0.660 | 0.653 | 0.144 | 0.617 | 285 |
| 50 | 0.656 | 0.640 | 0.109 | 0.638 | 408 |

**Decision:** the tree's grid is now `min_samples_leaf: [10, 20, 50]` with unlimited depth, still
**selected by held-out accuracy** like every other model. Accuracy picked 20 for both variants.
A tree whose leaves hold a single example isn't a probabilistic classifier, so leaf size 1 is
left out of the search rather than out-voted. The cost is 0.8 points of SST-2 test accuracy,
within noise for 1,820 sentences.

**Rejected:**
- Selecting the tree by log-loss: one model with a different selection rule would be harder to
  defend than one model with a narrower grid.
- Calibrating after the fact (Platt or isotonic): 0/1 outputs have no ranking to calibrate.

**Result (base):**
- SST-2 test 0.661 → 0.653.
- **IMDb 0.592 → 0.612**: the pruned tree generalises better.
- ECE 0.339 → 0.144.
- IMDb confidence AUROC 0.500 → 0.562. Distance is now significantly worse than confidence for
  the tree too (Holm p < 0.001), so H4's pattern holds for all five classical models.
- Directional negation: "not" leaves the probability unmoved in 27% of pairs (was 60%).
  flip_failure is 0.199 on 42% testable pairs.

**Not changed:** GNB's near-chance accuracy is the H6 comparison itself (GNB vs MNB), not a defect
to tune away. MNB's saturated probabilities come from multiplying thousands of word likelihoods,
and the early-warning combiner now handles them (D26).

## D30 — Length checks for the early-warning distance (F4) and for IMDb errors (C8)

**F4: how much of the distance is just length?** Taking the maximum over a review's sentences
rewards long reviews by chance (D25). `warn` now writes `early_warning_sanity.csv`:

| check | value |
|---|---|
| sanity AUROC, full reviews (as used) | 0.926 |
| sanity AUROC, **one random sentence per review** (length-matched) | **0.612** |
| corr(distance, log number of sentences) | 0.444 |

**Finding:** most of the 0.926 was length. Sentence for sentence, IMDb is only modestly unusual
next to SST-2. That makes sense: both are movie-review text. The SST-2 → IMDb shift is mainly
**length and format**, not sentence content. This changes how the shift should be described in
the report (shortfalls F7).

`warn` also scores a new predictor, **`distance_adj`**: distance minus what log(number of
sentences) predicts (an OLS residual that uses no labels). Its error-detection AUROC is 0.48–0.53
for every model, the same as raw distance and significantly below confidence everywhere (Holm
p < 0.001). Length wasn't hiding a useful distance signal, so the H4 conclusion stands.

**C8: does IMDb error grow with length?** `evaluate` writes `imdb_by_length.csv` (error with a 95%
CI per original-length bin: ≤100, 101–200, 201–360, >360 capped) and `figures` draws
`imdb_error_by_length_base`. The bins hold 1,257 / 4,678 / 2,472 / 1,593 reviews.

| model (base) | ≤100 | 101–200 | 201–360 | >360 (capped) |
|---|---|---|---|---|
| logreg | 0.167 | 0.197 | 0.218 | 0.236 |
| rf | 0.251 | 0.304 | 0.289 | 0.323 |
| mnb | 0.208 | 0.247 | 0.243 | 0.239 |
| dtree | 0.375 | 0.388 | 0.384 | 0.403 |
| gnb | 0.426 | 0.451 | 0.460 | 0.478 |

**Finding:** even position-blind bag-of-words models get worse on longer reviews (LR +6.9
points, RF +7.2), probably because long reviews mix praise and criticism. When DistilBERT
arrives, its curve has to be compared against these slopes, not against a flat line: only
growth *beyond* the classical slope points to its short-phrase training (C8).

## D31 — Robustness checks: test-only negation (B9), a second negation scope (C7), IMDb excerpts (B4)

**B9: does part of the negation set picking DistilBERT's seed matter?** `sst2_neg_testonly` (the
339 negated *test* sentences, never used for any choice) is now a row in `negmark_effect`,
`negation_gap` and the error heatmap. For the classical models it tells the same story as the
pooled 519 sentences: the same signs, and nothing significant after Holm except GNB with
`negmark`. The DistilBERT gap will be checked the same way when it arrives.

**C7: is the H3 result about negation marking, or about our scope rule?** A second variant,
`negmark_w3`, marks at most the **3 tokens** after a negation word (still cut at punctuation) and
skips fixed phrases where the negation doesn't negate ("not only", "not just", "no doubt",
"nothing but", "never fails"). The original `negmark` (Pang et al., scope up to punctuation) is
unchanged, and old saved models still load (`Preprocessor(negmark=True)` means "punct").
Trained for all 5 models (held-out: LR 0.843, MNB 0.822, RF 0.770, DT 0.667, GNB 0.566).

Effect on real negated sentences (sst2_neg, n = 519), base − variant error. Holm is now over 50
rows (5 models × 2 variants × 5 tests):

| model | negmark | negmark_w3 |
|---|---|---|
| logreg | +2.9 [−0.2, +5.8] | +2.9 [+0.8, +5.2], Holm 0.27 |
| mnb | +2.3 [−1.3, +6.0] | +3.9 [+1.3, +6.6], Holm 0.12 |
| rf | −0.4 [−3.5, +2.7] | +3.1 [+0.4, +6.0], Holm 0.78 |
| dtree | −1.5 | +0.2 |
| gnb | **+9.2, Holm 0.002** | +5.0, Holm 0.21 |

**Finding:** the shorter scope helps real sentences a little more consistently (every raw CI
except DT's is above 0), but nothing survives Holm except GNB with `negmark`. On templates the two
rules are similar for LR and MNB. The short scope does less for DT and GNB, whose template gains
under `negmark` came partly from marking whole clauses. So C10 holds with either scope rule:
negation marking's big template gains don't carry over to real negated sentences. Side effect: LR
`negmark_w3` improves SST-2 test by 1.4 points (Holm 0.009), the only significant test-set gain
from marking.

**B4: which part of the IMDb drop is length and which is content?** A new evaluation set,
`imdb_edges`, holds each IMDb review's **first and last sentence** (mean 44 words, close to SST-2's
19) with the review's label and *the same item_ids* as `imdb`. `shift_decomposition.csv` splits
each model's error change:
- length_format = err(imdb) − err(imdb_edges), same reviews, paired.
- content = err(imdb_edges) − err(sst2_test), independent samples.

| model (base) | SST-2 test | IMDb edges | IMDb full | length_format | content |
|---|---|---|---|---|---|
| logreg | 0.165 | 0.281 | 0.205 | **−7.7** [−8.6, −6.7] | **+11.6** [+9.7, +13.5] |
| mnb | 0.166 | 0.298 | 0.240 | −5.8 | +13.1 |
| rf | 0.246 | 0.334 | 0.297 | −3.7 | +8.8 |
| dtree | 0.347 | 0.417 | 0.388 | −2.9 | +7.0 |
| gnb | 0.409 | 0.481 | 0.454 | −2.6 | +7.1 |

All significant after Holm. **Finding:** cutting a review down to two sentences makes it *harder*
(every model), so full reviews' extra text helps more than their length hurts. Two short IMDb
sentences are much harder than two SST-2-length sentences, by 7–13 points.

**Caveat, which limits that content number:** `imdb_edges` inherits the review's label, but
its two sentences don't always state the verdict ("Caught this 1969 film on cable TCM one night.
Long live musicals." is labelled positive). That label noise inflates the edges error, so
**content is an upper bound** and length_format is a lower bound on how much text helps
(shortfalls B10). Together with D30, this means the embedding detector sees IMDb sentences as only
mildly unusual, but the classifiers do find them harder.
