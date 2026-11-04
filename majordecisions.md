# Major decisions

A running log of the major coding decisions taken while building the Mirage code skeleton, and why.
Newest decisions are appended at the bottom. Each entry says what was decided, the alternatives,
and the reason.

> **Numbers here are a dated record.** Each entry gives the results as they stood when the
> decision was made; later entries sometimes update them (e.g. when models were added). For the
> **current** results, use [hypothesisanswers.md](hypothesisanswers.md) and
> [interestingfindings.md](interestingfindings.md), which are checked against `results/*.csv`.

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

## D32 — Name tests rebuilt: a name-shortcut test from the training labels, and a gender-matched unseen-names probe

**Problem found while reviewing `wordlists.py`:**
- **None of the 10 Indian names occurs in SST-2's training data.** To the classical models they're
  invisible, so the old name-swap test effectively asked "does adding *John* change anything?"
  It was nearly impossible to fail (LR 1–2%).
- **Every Indian/Western pair also swapped gender** (Priya ↔ John, Rahul ↔ Emily, …), so a changed
  prediction couldn't be blamed on either factor.
- **Meanwhile the data is full of names with learned sentiment.** LR's weights: Spielberg +3.19,
  Seagal −1.69, and even "Sarah" −0.78 (from 7 phrases).

**Decision 1: a new `name_shortcut` test** (500 invariance pairs):
- Each pair swaps a film person whose training phrases are mostly positive for one whose phrases
  are mostly negative. The sentiment comes only from a mild adjective
  (decent/solid/watchable/pleasant/nice vs flat/uneven/forgettable/bland/weak), so the prediction
  shouldn't move.
- **Names are chosen from the training labels, not any model's weights**
  (`behavioral/names.py`), so the test is equally fair to DistilBERT. The method: smoothed
  log-odds of positive labels among training phrases containing the surname, with at least 15
  phrases. Surnames that are also everyday words (cage, stone, lee, ford…) are excluded.
- The six pairs, fixed in `configs/behavioral.yaml` (evidence in `results/name_associations.csv`):
  Kidman↔Seagal, Binoche↔Hopkins, Spielberg↔Costner, Nicholson↔Ritchie, Diesel↔Tarantino,
  Kline↔Sandler.
- The templates use nouns that name a whole work ("film", "comedy"…), because "Another Spielberg
  acting" isn't English. A first version had "and a uneven one"; it was reworded to "and it was
  {adj}." A check confirms no a/an errors remain in the suite.
- The generator runs last, so the other tests' random draws didn't change.
- `name_effect.csv` reports, per model: the flip rate, and the **shift** = mean p_pos(positive
  name) − p_pos(negative name), with CI and Holm p.

**Decision 2: the old swap is kept as an honest "unseen names" probe.** The Indian and Western
names are now paired by gender (Priya ↔ Emily, Rahul ↔ John, …).

**Result (base):**

| model | name_shortcut flips | shift [95% CI] | unseen-names shift |
|---|---|---|---|
| **mnb** | **40.4%** | **+0.380** [+0.353, +0.406] | −0.004 (n.s.) |
| logreg | 0.6% | +0.063 [+0.057, +0.069] | −0.003 (n.s.) |
| rf | 2.4% | +0.009 [+0.007, +0.011] | −0.004 (n.s.) |
| dtree | 1.2% | −0.003 (n.s.) | −0.002 (n.s.) |
| gnb | 3.0% | +0.001 (n.s.) | +0.011 (n.s.) |

- **Naive Bayes learned the names' reputations strongly.** "Another Kidman picture, and it was
  bland." and the same sentence with Seagal get very different scores (in the first version of
  the template: 0.72 vs 0.02). It multiplies word likelihoods, so one surname whose 26 training
  phrases were all positive outweighs a mild adjective.
- **LR moves every prediction the same way but rarely flips the label:** it has the shortcut, but
  the adjective usually wins.
- **Negation marking doesn't change this** (MNB shift about +0.31 under both variants), as expected.
- **Unseen names move nothing**, which confirms the old test couldn't detect anything for these
  models.

**Rejected:**
- Choosing names by LR's weights: circular, since it would favour finding an effect in LR.
- Removing the Indian/Western probe: it's still the right test for *unseen* names, and DistilBERT,
  whose pre-training may know these names, could behave differently.

## D33 — Word-list review: "messy" replaced by "sloppy" (user's decision)

**Decision (made by the project author during the word-list review):** the negative adjective
"messy" is replaced by **"sloppy"**.

**Why:**
- "messy" occurs in only 12 SST-2 training phrases.
- It's sometimes affectionate in reviews ("gloriously messy"), so a model that reads it as
  positive isn't clearly wrong.
- "sloppy" keeps the intended meaning without the ambiguity, and is common in training (76
  phrases, 12% positive).

**Shortlisted alternatives:** tiresome, pointless, lifeless, contrived, inept, shallow, clumsy,
stale, listless. **Ruled out:** unfunny (odd with most nouns), worst (ungrammatical in "The plot
was worst."), ugly (can be literal), tired (can describe the viewer), silly/dumb/laughable
(sometimes affectionate), pretentious (sometimes neutral), clichéd (encoding risk).

**Rule kept:** words are removed only when the *word* is ambiguous, never because models fail on
it. The adjectives the models misread most ("memorable", "lovely", "superb") are clearly
positive to a reader and stay. Removing them would hide exactly the failures the project measures.

**Effect:** the behavioral suite was rebuilt and the classical models' behavioral predictions
redone. The other word lists, the name pairs and the mild adjectives were left as they were.

**Review completed:** the author then reviewed the remaining lists (Indian and Western names, the
six film-person pairs, mild adjectives, nouns and verbs) and kept them unchanged. The word-list
review in `whatyouneedtodo.md` (step 7) is done.

## D34 — The DistilBERT run as it actually happened (Kaggle, Nov 2026)

**What ran:** the committed Kaggle notebook (code from commit 332bac0 or later), 3 seeds, then
GPU predictions on all 7 evaluation sets. About 40 minutes in total. Every one of the 21
prediction files matches this laptop's data item for item (ids, text and order, behavioral suite
included) and passes the predictions contract.

**Deviation from the plan, recorded rather than rerun:** Kaggle assigned **GPU T4 x2**. Hugging
Face's Trainer then splits each batch across both GPUs, so:
- **Effective batch: 64, not 32.** 3,000 steps per seed instead of 6,000.
- **Warm-up: 360 steps, which is 12% of training, not 6%.** `train()` computes warm-up from the
  per-device batch size, assuming one GPU.

Both are ordinary fine-tuning settings, and the seeds agree closely, so the results stand. The
report should state batch 64 and 12% warm-up. Future runs should either pick GPU T4 x1 or have the
code count GPUs.

| seed | held-out by epoch (1 / 2 / 3) | best | SST-2 test | negation (519) | IMDb | train time |
|---|---|---|---|---|---|---|
| 13 | 0.914 / 0.912 / **0.921** | 0.921 | 0.922 | 0.900 | 0.900 | 519 s |
| 42 | 0.916 / 0.917 / **0.920** | 0.920 | 0.920 | 0.898 | 0.898 | 522 s |
| 87 | 0.912 / **0.918** / 0.915 | 0.918 | 0.923 | 0.894 | 0.890 | 521 s |

Seed 87's best epoch was epoch 2 (`load_best_model_at_end`). The median-accuracy seed on `sst2_val`
(D13) is used for per-item analyses.

## D35 — First complete results, DistilBERT included (classical models: base variant)

Everything below comes from one `python -m mirage all` run on the real data. DistilBERT is
reported as mean ± std over 3 seeds for aggregates; per-item analyses use its median seed (13).

**Accuracy and the IMDb drop**

| model | SST-2 test | negation (519) | IMDb | drop | ECE (test / IMDb) |
|---|---|---|---|---|---|
| **distilbert** | **0.922 ± 0.002** | **0.897** | **0.896** | 0.026 | 0.046 / 0.051 |
| logreg | 0.835 | 0.773 | 0.795 | 0.039 | 0.016 / 0.081 |
| mnb | 0.834 | 0.761 | 0.760 | 0.073 | 0.122 / 0.229 |
| rf | 0.754 | 0.726 | 0.703 | 0.051 | 0.068 / 0.057 |
| dtree | 0.653 | 0.607 | 0.612 | 0.041 | 0.144 / 0.200 |
| gnb | 0.591 | 0.522 | 0.546 | 0.045 | 0.365 / 0.376 |

DistilBERT differs significantly from every classical model on every dataset (McNemar, Holm).

**H1: hidden failures on behavioral tests (error rate)**

| model | simple | negation | "X / not X" | typo | unseen names | name shortcut |
|---|---|---|---|---|---|---|
| distilbert | 0.000 | 0.000 | 0.000 | **0.394** | 0.000 | 0.000 |
| logreg | 0.040 | 0.272 | 0.277 | 0.454 | 0.016 | 0.008 |
| mnb | 0.102 | 0.492 | 0.572 | 0.386 | 0.084 | **0.376** |
| rf | 0.094 | 0.206 | 0.247 | 0.366 | 0.048 | 0.024 |

**New shortcut found in DistilBERT: a misspelled word reads as negative.** "The movie was
brilliant." scores 0.999 positive; "The movie was brililant." scores 0.010. A typo in a positive
adjective flips DistilBERT's label in **65%** of pairs (mean p_pos −0.64). A typo in a negative
adjective flips only 9% (+0.11). The classical models lose typo'd words much more symmetrically:
LR flips 42% for positive words and 49% for negative ones, because an unseen word is simply
ignored. DistilBERT breaks the misspelled word into unfamiliar word pieces and has learned
"unfamiliar-looking ⇒ negative" (shortfalls D9).

**H2 / H3: negation gap to DistilBERT, and how much `NOT_` marking closes it**

| model | gap on real negated sentences | closed by marking | gap on "X / not X" | closed |
|---|---|---|---|---|
| logreg | +12.7 pts [8.9, 16.6] | 23% [−2, 43], n.s. | +27.9 | **96%** |
| mnb | +13.9 | 17%, n.s. | +60.9 | 63% |
| rf | +17.3 | −2%, n.s. | +25.3 | 49% |
| dtree | +29.3 | −5%, n.s. | +20.9 | 74% |
| gnb | +37.8 | **25%** [13, 35] | +97.7 | 30% |

Every base gap is significant (H2 supported). Marking closes most of the *template* gap but none
of the *real-sentence* gap significantly, except for GNB (H3: supported only on templates; see
C10). The same holds on the 339 test-only sentences (B9).

**H4: does distance flag errors better than confidence?** No, for every model. For DistilBERT,
low confidence has AUROC **0.836** [0.823, 0.847] on 1,004 IMDb errors. Distance reaches 0.530,
length 0.525, and length-adjusted distance 0.531. Combining distance with confidence adds nothing
(0.835).

**C8: length.** DistilBERT's IMDb error is flat across lengths (0.098 / 0.096 / 0.094) and rises
only for capped reviews (0.126), where text was cut. Training on short phrases didn't hurt it on
long reviews.

**B4: length vs content.** First + last sentence only: DistilBERT's error goes 0.100 → 0.207 on
the same reviews. At SST length, IMDb text is harder than SST-2 by 12.9 points (an upper bound,
see B10), the same size as for LR and MNB.

**H5: do model families rely on the same words?** Hardly. Top-50 overlap between DistilBERT's IG
words and any classical model is Jaccard 0.01–0.08 (RBO ≤ 0.04). The tree and forest agree
(0.72), and LR shares a quarter of its words with them. The IG explanations are faithful:
deleting the top-1/2/3 attributed words lowers DistilBERT's confidence by 0.37/0.46/0.52, against
0.11/0.19/0.27 for the same number of random words. IG convergence delta ≤ 0.024.

## D36 — Warm-up counts every GPU (fixes the two-GPU quirk from D34)

**Bug:** `train()` computed warm-up from the *per-device* batch (32), assuming one GPU. On
Kaggle's T4 x2 the Trainer splits each batch across both GPUs, so each step consumed 64 examples
and training had 3,000 steps, not 6,000. The fixed 360-step warm-up became 12% of training
instead of the configured 6%.

**Fix:** `TrainingArguments` is built first, then warm-up is set from the batch each optimizer
step really sees: `train_batch_size` (per-device × GPUs) × `world_size` × gradient accumulation.
The arithmetic lives in a small, tested function, `schedule()`. Two GPUs give 3,000 steps and 180
warm-up steps (6%), matching the Kaggle run's step count. `train_summary.json` now also records
`devices`, `effective_batch`, `total_steps` and `warmup_steps`, so a run's real recipe is visible
without reading its logs.

**Screened:** a 64-example CPU fine-tune in a throwaway `MIRAGE_ROOT`. The plan said 6 steps and
the Trainer ran 6. The real models were untouched.

**Not changed:** the effective batch itself. Two GPUs still mean batch 64, which is a fine
setting. The existing three models (batch 64, 12% warm-up) stay as they are and as recorded in
D34. Retraining them for a warm-up difference isn't worth a GPU run.

## D37 — DistilBERT's typo shortcut, pinned down (C11)

**Question:** D35 found that a misspelled *positive* adjective flips DistilBERT to negative in 65%
of pairs ("brilliant" 0.999 → "brililant" 0.010), but a misspelled negative one in only 9%. Is
that a reaction to *any* garbled word, or specifically to an unreadable sentiment word?

**Control added:** a `typo_noun` test (500 pairs, appended last so earlier draws didn't change),
which puts the typo in the noun and leaves the adjective intact: "The movie was brilliant." →
"The moive was brilliant.". A new table, `typo_direction.csv`, splits every typo test by the
sentence's polarity, with a paired CI on the probability shift.

| model | adjective typo, positive | adjective typo, negative | noun typo, positive | noun typo, negative |
|---|---|---|---|---|
| **distilbert** | flips 65%, **−0.64** | flips 9%, +0.11 | **flips 0%, −0.00** | flips 0%, +0.00 |
| logreg | 42%, −0.37 | 49%, +0.36 | 2%, +0.04 | 6%, −0.07 |
| mnb | 50%, −0.42 | 25%, +0.28 | 19%, +0.11 | 2%, −0.03 |
| rf | 18%, −0.13 | 59%, +0.21 | 8%, +0.03 | 14%, −0.03 |

**Finding:** DistilBERT ignores a garbled noun completely, so it isn't "any odd word means
negative". The shortcut is narrower: **when it can't read the sentiment word itself, its default
answer is "negative"**, and confidently so. LR, which simply loses the unknown word, drifts
toward the middle from both sides (−0.37 / +0.36). The tree-based and Naive Bayes models fall in
between, each with its own lean.

**Where the default comes from:** in SST-2's training phrases, 16.0% of the words in negative
phrases are split into word pieces (not whole words in DistilBERT's vocabulary), against 10.9% in
positive phrases. Phrases containing any split word are 49.9% positive, against 61.4% for phrases
without one. Odd, fragmented words lean negative in the data: coinages like "unwatchable",
sarcastic spellings, and pieces like "##ible" from "horrible" and "terrible". A typo turns
"brilliant" into `br ##ili ##lan ##t`, which looks like that negative-leaning vocabulary.

**Also:** the behavioral suite is now 3,750 cases (6,250 rows). All 18 models' behavioral
predictions were redone locally, including DistilBERT's 3 seeds on the CPU (short sentences take
minutes), and the tables were regenerated.

## D38 — Why negation marking helps templates but not real sentences (C10)

**Question:** `NOT_` marking closes 96% of LR's negation gap on templates but has no significant
effect on the 519 real negated sentences. Why? Analysis of LR, `negmark` vs `base`:

**First hypothesis, rejected:** "real sentences mark words the model never saw marked". The model
knows about as many of the marked words in real sentences (78%) as in templates (74%).

**What actually differs:**
1. **Scope swallows whole clauses.** A real negation marks **6.9 words** on average, a template
   2.1. In "it 's never a good sign when a film 's star spends the entirety of the film …",
   everything after "never" becomes `NOT_a`, `NOT_film`, `NOT_the`. The signal on `NOT_good` is
   diluted by many marked function words.
2. **Real negations often don't negate the sentiment.**
   - Double negation: "it 's almost impossible not to be moved", which is positive.
   - Idioms: "it wo n't be long before you 'll spy it at a video store".
   - Negation of a non-sentiment verb: "surely can not last".

   Marking pushes these the wrong way.
3. **The learned marked weights are noisy outside the clean cases.** `NOT_good` −3.43 and
   `NOT_funny` −4.29 are sensible, but `NOT_brilliant` is +1.07 (rare and noisy) and `NOT_boring`
   is −0.08 (no signal).

**Net effect on real sentences (LR):** marking **fixes 38** of the 118 base errors but **breaks
23** previously correct sentences, a net of +15 out of 519. That's the +2.9-point,
non-significant gain (D28).

**Takeaway for the report:** the templates test negation in its easiest form, with one negation
word right before one sentiment word. Real negation is longer-range, idiomatic and sometimes
double. A token-level marking heuristic captures the first and not the second, which is the
gap a contextual model like DistilBERT closes (0% template errors, 10.3% real-sentence errors).

**Note on resampling (added with D37):** replacing "messy" with "sloppy" (D33) changed how many
random draws the typo generator uses (it depends on word length), so every test generated after it
was resampled: `name_swap`, `name_shortcut`, and now `typo_noun`. The name-shortcut numbers moved
within noise. Naive Bayes flips 37.6% of pairs (shift +0.37), against 40.4% / +0.38 in D32. Cite
the current tables (`name_effect.csv`); D32 keeps its numbers as a record of the earlier run.

## D39 — External models: "does scale cure shortcuts?"

**Decision:** two ready-made Hugging Face models go through exactly the same eval sets and
behavioral suite, with no training (`configs/external.yaml`, stage `predict-external`):

| short name | model | why |
|---|---|---|
| `roberta_large` | `siebert/sentiment-roberta-large-english` (355M) | **Scale:** about 5× DistilBERT, trained on 15 sentiment datasets |
| `distilbert_hf` | `distilbert/distilbert-base-uncased-finetuned-sst-2-english` (66M) | **Replication control:** same architecture as ours, trained on SST-2 by someone else |

**How:**
- `ExternalClassifier` reuses our DistilBERT wrapper (batching, head + tail truncation at 512 for
  IMDb), so the long-text rule matches D23.
- **The label order comes from the model's own `id2label`.** It never assumes 0 = negative, and
  it refuses anything that isn't exactly one negative and one positive label.
- **Raw text, no `normalize`.** These models were trained on ordinary English. SST-2 rows are
  already in SST's tokenized style ("does n't"), which they read slightly less well; that's a
  caveat, not a fix, since changing the text would make the comparison less like-for-like.
- Seed 0, variant `base`: there's no training run to seed.
- **Every eval set, including all 10,000 IMDb reviews, runs on the Kaggle GPU**
  (`notebooks/kaggle_external.ipynb`, one run, one zip). Every model, ours and external, is
  compared on exactly the same reviews, so the existing tables and figures need no subsample
  caveat. (A first plan ran the short sets on the laptop CPU and only IMDb on Kaggle; one
  notebook run is simpler and adds only minutes of GPU time.)

**Statistics are kept separate:** pairs involving an external model form their own Holm family in
McNemar, and external models form their own family in early warning. They're left out of
`negation_gap` (H2 is our classical models vs our DistilBERT). So adding them leaves every
earlier p-value unchanged.

**Rejected:**
- `cardiffnlp/twitter-roberta-base-sentiment-latest`: three classes (neutral) and trained on
  tweets.
- `textattack/roberta-base-SST-2`: unnamed labels, so the mapping can't be checked.
- A 2,000-review CPU subsample of IMDb: cheaper, but it either breaks like-for-like comparison or
  forces every model onto the subsample.

**Caveats written into the results:**
- SiEBERT was very likely trained on SST-2 and IMDb, so **its accuracy isn't a fair
  comparison**. Its behavioral results are clean, because the test sentences are ours.
- `distilbert_hf` was trained on GLUE SST-2 train, which contains our held-out phrases, so its
  held-out accuracy is inflated. Validation and test are fine.

## D40 — The Mirage Gap: one score for "how a model behaves"

**Decision:** a **robustness score** per model is the equally weighted mean of four core
shortcut checks. It's plotted against SST-2 test accuracy (`mirage_gap.csv`,
`figures/mirage_gap*.png`). The gap between the two is the "mirage": how much better a model
looks on a leaderboard than it behaves.

| component | source | a case counts as passed when… |
|---|---|---|
| negation | `negation_dir` pairs | both "X" and "not X" are classified correctly |
| real negation | `sst2_neg` | the real negated sentence is classified correctly |
| typos | `typo` pairs | both the clean and the misspelled sentence are correct |
| names | `name_shortcut` pairs | both name versions are correct |

Left out: `typo_noun` (a control), `name_swap` (almost impossible to fail, D32) and the unscored
`*_neg` probes.

**Changed while screening: pair accuracy, not "didn't flip".** The first version used the
invariance and direction failure rates from `errors_by_type`, and weak models came out "robust":
the decision tree scored 79% against 65% accuracy, and Gaussian NB passed 86% of typo pairs. A
model that can't read the sentiment word never changes its answer, so it never "fails" an
invariance test. Requiring **both sentences of a pair to be right** means robustness can't
exceed competence. The decision tree drops to 37% on negation and 45% on typos, and every model
now sits below the diagonal.

**Result (base variants):**

| model | SST-2 test accuracy | robustness | gap |
|---|---|---|---|
| Multinomial NB | 0.834 | 0.554 | **0.280** |
| Gaussian NB | 0.591 | 0.395 | 0.196 |
| Decision tree | 0.653 | 0.514 | 0.139 |
| Logistic regression | 0.835 | 0.743 | 0.091 |
| Random forest | 0.754 | 0.694 | 0.060 |
| DistilBERT (ours) | 0.922 | 0.876 | 0.046 |

- **The headline:** logistic regression and Multinomial NB look identical on the leaderboard
  (83.5% vs 83.4%), but NB behaves 19 points worse (names 61% vs 99%, negation 35% vs 70%).
- **Negation marking narrows the gap.** LR's gap goes from 0.091 to 0.040 (robustness 0.743 →
  0.807), mostly through template negation.

**Sensitivity:** leaving out each component in turn keeps the model ordering identical for every
variant (Spearman 1.0 in all 12 checks, `mirage_gap_sensitivity.csv`), so the ranking doesn't
hinge on the equal weights.

**Shortcut fingerprint** (`shortcut_fingerprint.csv`, `figures/shortcut_fingerprint*.png`): small
multiples, one panel per model, one bar per check, plus calibration (1 − ECE on IMDb) and
error warning (AUROC of low confidence on IMDb). Bars rather than a radar chart: radar areas
distort and are hard to read from a distance. Each figure also has a 16:9 `_slide` version.

**Rejected:**
- Weights tuned to make the story cleaner.
- A radar chart.
- Folding calibration and warning into the score (they measure something else: whether you
  can *tell* when the model is wrong).

## D41 — The Shortcut Explorer: a live demo that can't embarrass you

**Decision:** a Streamlit app (`app/shortcut_explorer.py`, optional `[demo]` extra) where you
type a sentence, or pick a curated pair, and see every model's verdict and *why*. All its logic
is in `mirage.demo` and unit-tested; the app is layout only. Settings are in `configs/demo.yaml`.

**What it shows, per model:**
- the verdict and the probability;
- **word importance by leaving each word out** (remove the word, see how far the probability
  moves). It works identically for every model, including the forest, and fits in one
  sentence on stage;
- **words a classical model never saw**, greyed out;
- two flags:
  - **low confidence** below 0.70, the signal H4 showed predicts errors;
  - **unusual input** above the 95th percentile of SST-2 test sentences' embedding distance.
    It's worded "unusual", not "probably wrong", because H4 showed distance doesn't predict
    errors.

**Pairs show the punchline first:** a table of A vs B per model with flipped verdicts highlighted.
The per-word detail sits in tabs below.

**Curated examples are measured, not assumed.** Every candidate sentence was scored on all eight
models first, and only clear-cut predictions (far from 0.5) became expectations in
`app/examples.yaml`. `tests/test_demo.py` re-checks them, so the scripted demo can't drift
silently. What the measurement found:
- **RoBERTa-large flips on a name too** (Kidman 0.98 → Seagal 0.02), so scale doesn't cure the
  name shortcut. It does resist the typo (brililant stays 1.00) and reads the sarcasm sentence
  correctly (0.00).
- "The acting was not bad at all." gets 0.03 from LR and 0.99–1.00 from every transformer.
- The sarcasm sentence is the only probe that trips "unusual input". Our DistilBERT calls it
  0.98 positive, so it's planned into the demo as the "where our models fail" example.

**Stage safety:**
- **Cache-first model loading:** `ExternalClassifier.from_hub` tries the local cache before the
  network. A flaky Hugging Face check had made RoBERTa-large fail to load once during
  development. The app now runs offline once the models are cached.
- Input is trimmed to 60 words with a message, and empty input is handled.
- Light mode leaves out the random forest and RoBERTa-large.
- A "reset to examples" button, and a light theme for projectors (`.streamlit/config.toml`).
- Fallback images of every curated example are in `app/screenshots/`
  (`app/make_fallbacks.py`).
- A presenter guide (`app/PRESENTING.md`) gives the demo script and a ready line for each kind of
  failure.

**Changed from the design:**
- **The optional Integrated Gradients "detailed view" was left out.** Leaving words out already
  explains every model, including DistilBERT, while IG would explain only one model, add about 2
  seconds per sentence and need a second explanation on stage. IG stays in the report (H5).
- **Probability bars are neutral grey.** In model colours, RoBERTa-large's red bar read as
  "negative" even at 98% positive, so colour is kept for the model dot.

**Code review of D39–D41 (before the Kaggle results).** Three issues were found and fixed:
1. `shift_decomposition` and `name_effect` pooled every model into one Holm family, so external
   models would have shifted our models' adjusted p-values. All four multi-model families now
   split by `evaluate.holm_family` (ours vs external), and a test covers it.
2. The reliability figure picked "the best classical model" as any model other than DistilBERT,
   which would have chosen RoBERTa-large. It now chooses among the five classical models.
3. `all` ran `predict-external`, which on a laptop means a 1.9 GB download and hours of CPU on
   IMDb. It's now run only explicitly (the Kaggle notebook); `all` picks up its files.

Also: the Kaggle notebook's time estimate was raised to 20–30 minutes (RoBERTa-large on 10,000
reviews at 512 tokens). Re-running `evaluate` reproduced every existing number exactly.

## D42 — A single-word variant, so H2 and H3 test what the proposal says

**Problem:** the proposal's H2 compares DistilBERT with "bag-of-words models with single-word
features", and H3 says "word-pair features and negation marking close part, but not all, of that
gap". But every classical model was built with single words *and* word pairs
(`ngram_range: [1, 2]`), so no single-word model existed. H2 was tested against the wrong rung,
and the word-pair half of H3 was never tested.

**Decision:** a new classical variant, `unigram` (single words only; everything else identical:
the same grids, held-out split, normalization and seed), completes a three-rung ladder:

| rung | variant | the classical model sees |
|---|---|---|
| 1 | `unigram` | single words |
| 2 | `base` | + word pairs |
| 3 | `negmark` | + negation marking |

`negation_gap` gains these columns:
- `gap_unigram`: H2 as worded.
- `pairs_closed`: the share of the single-word gap closed by word pairs (H3, first half).
- `ladder_closed`: the share closed by pairs and marking together (H3 as a whole).
- `gap_closed`: the marking step alone, unchanged.

The figure shows three bars per model.

**Nothing existing moves:**
- Each new comparison is aligned on its own items. On "X / not X" pairs, a model counts only
  pairs whose affirmative it got right, so joining the unigram model into the existing alignment
  would have shrunk it.
- `unigram` is an *ablation*: `evaluate.holm_family` puts it in its own Holm family in every
  multi-model table (`shift_decomposition`, `name_effect`, early warning), and `negmark_effect`
  skips it, because it isn't a marking variant.
- Every previously reported number was checked against the committed CSVs: identical.

**Results (gap to DistilBERT on real negated sentences, 519):**

| model | single words | + pairs | + marking | pairs close | pairs + marking close |
|---|---|---|---|---|---|
| Logistic regression | 0.164 | 0.127 | 0.098 | 22% [4, 39] | **40% [21, 58]**, p_holm < 0.001 |
| Multinomial NB | 0.141 | 0.139 | 0.116 | 1% | 18% (n.s.) |
| Random forest | 0.179 | 0.173 | 0.177 | 3% | 1% (n.s.) |
| Decision tree | 0.243 | 0.293 | 0.308 | −21% | −27% (n.s.) |
| Gaussian NB | 0.347 | 0.378 | 0.285 | −9% | 18% [5, 30], p_holm 0.04 |

**What it means:**
- **H2 holds as worded.** Every single-word model is significantly worse than DistilBERT on
  negation, on real sentences (14–35 points) and on every template test (Holm p < 0.001).
- **H3 holds for logistic regression, the baseline model.** Pairs and marking together close 40%
  of its real-sentence gap: part, but not all. Pairs alone help (22%), but not significantly
  after Holm.
- **On templates the ladder works for every model** (pairs + marking close 25–98%, all
  significant).
- **On real sentences the other models barely benefit**, and the decision tree gets *worse* with
  word pairs (its sparse splits latch onto pairs that don't generalize). Real negation is
  long-range and idiomatic (D38), which a two-word window rarely captures.
- **Accuracy hardly depends on word pairs:** single-word LR scores 0.822 on SST-2 test against
  0.835 with pairs. Pairs mostly matter for negation, which is the point H3 makes.

**Rejected:** rewording H2/H3 to drop the word-pair claim. The ablation was cheap (about 3
minutes of training) and tests the proposal as written.

**After `warn` and the code review (D42):**
- **Early warning (H4) is unchanged for every existing row.** The single-word models show the
  same pattern: low confidence flags IMDb errors (AUROC 0.55–0.74), distance doesn't (about 0.5).
  Single-word LR's combined score beats confidence by a negligible +0.003 AUROC (Holm p 0.02,
  ablation family), like Multinomial NB's +0.012.
- **Review fix:** `evaluate.py` had started importing `models.classical` (scikit-learn model
  classes) to read the ablation list, which breaks the layering rule that analysis never imports
  models. `ABLATION_VARIANTS` now lives in `models/base.py`.
- **Process note:** a `warn` run crashed at its last line because that fix was saved while it was
  running (lazy imports loaded the old `base.py` and the new `classical.py`). The rerun on
  consistent code completed; CLAUDE.md now says not to edit `src/` while a stage runs.

## D43 — H5 with one ranking rule for every model (leave-one-word-out)

**Problem:** D16 ranked each model's "top shortcut words" with a *different* method:
- LR by |weight|;
- Multinomial NB by its log-count ratio, which favours rare one-sided words like *devoid*;
- the trees by impurity importance, which favours common words like *the*, *a* and *.*;
- DistilBERT by Integrated Gradients.

So the top-50 overlaps mostly measured whether two *methods* favour the same kind of word:
- decision tree – random forest 0.72, from the same method and shared function words;
- Multinomial NB ≤ 0.04 with everyone.

An H5 verdict from those numbers would be about the explanation methods, not the models.

**Decision:** rank words the same way for every model.
- **Rule:** for each SST-2 test sentence (1,820) and each word in it, remove the word and measure
  how far the model's P(positive) moves. A word's importance is the mean |change| over its
  occurrences. The signed mean is kept too, to tell which way it pushes.
- **Eligible words:** a word is ranked only if it occurs at least 5 times (fewer is too noisy)
  and contains a letter (punctuation is removed and scored, but not ranked as a "shortcut
  word").
- **Models:** all five classical models (base variant), **now including Gaussian NB**, plus our
  DistilBERT at its median seed. The external models are left out: H5 is about our models, and
  RoBERTa-large on about 38,000 sentences is slow on CPU.
- **Same overlap measures as D16:** Jaccard@50 and RBO (p = 0.9), so old and new results are
  directly comparable.
- **New noise ceiling:** each model's split-half self-overlap. The sentences are split in half
  at random, and the model's two top-50 lists are compared. It shows how much overlap a model
  reaches with *itself*, so "high" and "low" overlap between models have a reference point.
- **Settings:** `configs/explain.yaml`. **Stage:** `python -m mirage occlusion` (also run by `all`).
- **Outputs:**
  - `results/explain/occlusion_importance.csv`
  - `results/explain/overlap_occlusion.csv`
  - `results/explain/occlusion_split_half.csv`
  - `figures/overlap_occlusion_{jaccard,rbo}.png`, with split-half agreement on the diagonal
- **D16's overlap stays** as a secondary view, with the caveat above.

**Why removing words:** it treats every model as a black box, so the same rule applies to a
forest, Naive Bayes and a transformer alike. It's the same measure the live demo uses (D41), so
the slides and the report agree.

**Rejected:**
- SHAP: different estimators per model family, so the problem comes back, and it's slow for a
  forest.
- Using one model's explanation method for all: not possible across these families.

**Changed while screening (D43): the main H5 measure is a reliability-corrected correlation, not
top-50 overlap.**
1. **The first split-half ceiling was wrong.** Each half was ranked with a halved threshold, so
   words seen only 2–3 times dominated, and every model "agreed with itself" (0.02–0.32) *less*
   than with other models (up to 0.49). Fixed: both halves now rank the same words (those
   eligible in the full data), with a regression test.
2. **Even fixed, top-50 overlap is too noisy at this data size.** A model's top 50 agrees with its
   own other half only 11–45% (chance: 3%). Raising the word-count threshold buys reliability but
   raises chance overlap just as fast (threshold 30: self-agreement 0.49–0.77, chance 0.29), so
   no threshold is clearly right.
3. **Main measure now:** for each pair of models, the Spearman correlation of their signed word
   effects over all 773 eligible words (bootstrap 95% CI over words), divided by
   √(reliability_a × reliability_b). Each model's reliability is its split-half correlation
   (20 random splits, Spearman-Brown corrected). This is the standard way to compare noisy
   measures: noise in one model's ranking can't make two models look different. It also uses
   the direction of each word, not just membership of a top-50 list.

The threshold was chosen on reliability alone, never on the H5 result. The conclusions are the
same at 5 and 10 (`occlusion_agreement.csv` has both). Top-50 overlap stays as a secondary view,
with its chance level (`occlusion_split_half.csv`).

**Results (threshold 5, 773 words; corrected agreement, raw Spearman with 95% CI):**

| pair | corrected | raw ρ [CI] |
|---|---|---|
| LR – Multinomial NB | **0.92** | 0.76 [0.72, 0.79] |
| LR – random forest | **0.89** | 0.79 [0.75, 0.82] |
| random forest – Multinomial NB | **0.88** | 0.70 [0.66, 0.74] |
| LR – DistilBERT | 0.77 | 0.53 [0.47, 0.58] |
| random forest – DistilBERT | 0.74 | 0.49 [0.43, 0.55] |
| Multinomial NB – DistilBERT | 0.71 | 0.44 [0.38, 0.50] |
| decision tree – others | 0.44–0.64 | 0.22–0.35 |
| Gaussian NB – anyone | 0.18–0.26 | 0.09–0.22 |

Reliability: LR 0.91, random forest 0.86, Gaussian NB 0.77, Multinomial NB 0.75, DistilBERT 0.52,
decision tree 0.34.

**What it means for H5:**
- **Supported for the three main classical models.** LR, random forest and Multinomial NB agree
  almost as much as their noise allows (0.88–0.92). They rely on the same words in the same
  direction, which points to the shortcuts living in the data. D16's top-50 method had put
  Multinomial NB at ≤ 0.04 with everyone; that was the ranking method, not the model.
- **Gaussian NB is the exception** (0.18–0.26). It reads words only through 300 compressed SVD
  components, so a single word's effect is spread out. That ties in with H6.
- **The decision tree agrees moderately** (0.44–0.64), and its own word effects are the least
  reproducible (0.34): one tree's path depends on a few splits.
- **DistilBERT shares most of the classical models' word effects** (0.71–0.77). That corrects
  D35's reading of D16 ("almost entirely different words", top-50 overlap ≤ 0.08): the earlier
  number compared IG on DistilBERT's failures with the classical weights, a method mismatch.
  DistilBERT differs where context matters (negation, D42), not in which words carry sentiment.

**Code review (D43):**
- **Fixed: a cache-naming bug.** Removal tables were cached as `{model}__{dataset}`. If
  DistilBERT's median seed changed, the other seed's older table would have been reused
  silently. Caches are now named like prediction files, `{model}__base__seed{seed}__{dataset}`.
- **Fixed:** the reliability correction is undefined (not an error) when a model's reliability
  is ≤ 0.
- **Documented in shortfalls E4:** the corrected values have no interval of their own, and the
  bootstrap resamples words, not sentences.

## D44 — Every median-seed result checked on all three DistilBERT seeds (shortfall G2)

**Problem:** the per-item analyses use DistilBERT's median seed only (D13), so their confidence
intervals cover item sampling, not training randomness. A reviewer can fairly ask: "would another
seed change the conclusion?"

**Decision:** `evaluate.seed_robustness` reruns every per-item result that involves DistilBERT,
once per seed (13, 42, 87), without retraining: the predictions for all three seeds already
exist. It covers:
- DistilBERT's error on every test;
- the full negation-gap ladder, with Holm p-values (H2/H3);
- the Mirage Gap's robustness, gap and **rank** for every model;
- DistilBERT's typo direction and name effect.

Output: `results/seed_robustness.csv`, one row per result with a column per seed. Two flags:
- `same_sign`: the estimate never changes direction (for ranks: the rank is identical);
- `same_significance`: every seed is on the same side of 0.05.

It adds about 70 seconds to `evaluate`.

**Rejected:** a two-level bootstrap (seeds, then items). With only 3 seeds the seed level can't
be estimated meaningfully, and side-by-side seeds are easier to read and to check.

**Result: no conclusion depends on the seed.** Across 285 results: 0 sign changes and
0 significance changes. The Mirage Gap ranks are identical on all three seeds. For example:

| result | seed 13 | seed 42 | seed 87 |
|---|---|---|---|
| DistilBERT SST-2 test error | 0.078 | 0.080 | 0.077 |
| DistilBERT real-negation error | 0.100 | 0.102 | 0.106 |
| LR: pairs + marking close the gap (H3) | 40.0% | 40.5% | 41.5% (all p_holm < 0.001) |
| LR: pairs alone close the gap | 22.4% | 22.6% | 23.2% (none significant) |
| DistilBERT flips on a misspelled positive word | 64.9% | 60.9% | 53.5% |
| … on a misspelled negative word | 9.2% | 14.0% | 18.3% |
| … on a misspelled noun | 0% | 0% | 0% |
| DistilBERT name-shortcut flips | 0% | 0% | 0% |

The typo asymmetry (D37) narrows on seed 87 (54% vs 18%) but holds on every seed.

**Code review:**
- Join keys are unique per seed, so no rows can be dropped or mixed.
- Significance is judged within each seed's own Holm family, as in the reported tables.
- A value of 0 on one seed and non-zero on another counts as a change, which is conservative.
- No bugs found. All previously reported numbers were re-checked: unchanged.

## D45 — The name shortcut, tested on every film person in the training data (shortfall D8)

**Problem:** the name-shortcut finding (Multinomial NB flips 37.6% of pairs; RoBERTa-large flips
on Kidman vs Seagal in the demo) rests on 6 name pairs. Six pairs could be a coincidence of
those particular names.

**Decision:** a separate **name probe**, outside the main behavioral suite, so no existing result
or prediction file changes, and the external models' Kaggle run isn't affected:
- **Names:** all **35** film people with at least 15 SST-2 training phrases
  (`name_associations`, from the training labels only). That's 12 already in the test and **23
  never tested**.
- **Sentences:** each name is placed in the **same 40 sentences**: the name-shortcut templates
  with a work noun and a mild adjective, sampled once with a fixed seed. Only the name differs
  between names, and the adjective carries the sentiment.
- **Name effect** per model and name: mean P(positive) over the 40 sentences, minus the mean over
  all names on the same sentences.
- **Test:** the Spearman correlation between a model's name effects and the names' training
  log-odds, with a bootstrap CI over names, computed on the 23 never-tested names (the
  out-of-sample check) and on all 35. Plus the difference between the most positive and most
  negative never-tested names. Holm across models, with our models and external models in
  separate families.
- **Models:** our five classical models (base), our DistilBERT (median seed), and the external
  models if they're cached locally (they are on this laptop).
- **Contract kept:** its own eval set (`name_probe`, suite in `results/name_probe_suite.csv`),
  prediction files written through `write_predictions`, and analysis reading only those files.
  It is not added to `EVAL_SETS`, so `predict` and the Kaggle notebooks never touch it.
- **Stage:** `python -m mirage name-probe`; settings in `configs/behavioral.yaml → name_probe`.

**Why a correlation over names, not more flip pairs:** it uses every name, and it asks the
sharper question. A model that learned names' reputations reacts to a name in proportion to how
positive its training phrases were. A model that ignores names shows no relationship.

**Results (D45).** The name probe ran on all eight models: 35 names × 40 sentences.
`name_probe.csv` gives each model's **effect size** (SD and largest name effect) next to its ρ.
Ranking near-zero, noise-level effects can still produce a sizeable ρ, so ρ is read only
where names actually move the model.

| model | largest name effect | ρ, 23 never-tested names [95% CI] | Holm p |
|---|---|---|---|
| **Multinomial NB** | **±0.32** | **0.91 [0.78, 0.96]** | < 0.001 |
| Logistic regression | ±0.13 | 0.33 [−0.13, 0.71] (all 35 names: 0.45, p 0.03) | 0.48 |
| RoBERTa-large | ±0.03 | 0.47 [0.06, 0.79] | 0.025 (external family) |
| Random forest / Gaussian NB / decision tree | ≤ ±0.03 | not significant | — |
| Our DistilBERT / HF DistilBERT | ≤ ±0.001 | noise (effects ≈ 0) | — |

- **The name shortcut generalizes.** Multinomial NB's reaction to 23 names it was never tested
  on tracks their training reputations almost perfectly. So the 6-pair result (37.6% flips) was
  not a coincidence of those names.
- Logistic regression has a weaker, real version across all names. The transformers ignore
  names (DistilBERT), or follow them very slightly (RoBERTa-large: systematic but tiny).

## D46 — External model results: does scale cure shortcuts? (completes D39)

The Kaggle run (all 7 eval sets, both models) was checked against the laptop's data: every
item id, text and label matches in all 14 files.

| | SST-2 test | real negation | IMDb | IMDb edges | robustness (D40) |
|---|---|---|---|---|---|
| Our DistilBERT | 0.922 | 0.897 | 0.896 | 0.788 | 0.876 |
| **HF DistilBERT** (same architecture, trained by Hugging Face) | 0.923 | 0.892 | 0.893 | 0.793 | 0.876 |
| **RoBERTa-large** (SiEBERT, 5× larger, 15 datasets) | 0.946* | 0.931* | 0.956* | 0.871* | **0.906** |

\* possibly contaminated (C12): SiEBERT was very likely trained on SST-2 and IMDb. HF
DistilBERT's held-out accuracy (0.989) confirms it was trained on our held-out phrases (C14);
its validation and test numbers are clean.

**Findings:**
1. **Replication: an independently trained DistilBERT behaves like ours.** Accuracy is within
   half a point everywhere, and **the typo shortcut reproduces**: a misspelled positive word
   flips HF's model in 59% of pairs (ours 65%), a negative one in 15% (ours 9%), a misspelled
   noun in 0%. So "unreadable sentiment word means negative" comes from SST-2's data (D37), not
   from our training run.
2. **Scale reduces shortcuts but doesn't remove them.** RoBERTa-large has the smallest Mirage Gap
   (0.946 → 0.906) and passes the negation templates perfectly. But:
   - **Typos still break it, in the opposite direction.** It flips 42% of *negative* sentences
     with a typo (positive: 18%); its default for an unreadable word leans positive.
   - **Names still move it slightly.** It flips 1% of name pairs (shift +0.010, Holm p 0.02),
     and its name effects follow the training reputations (D45). The live demo's Kidman/Seagal
     sentence (0.98 → 0.02) is an extreme case. **Correction:** D41 and the demo's talking
     point said "scale does not cure this shortcut" without that qualification; both now say
     scale *shrinks* it.
3. **H4 holds for both external models.** Low confidence flags their IMDb errors (AUROC 0.845
   HF, 0.857 RoBERTa-large); embedding distance doesn't (0.525 and 0.507).
4. **The IMDb shift has the same shape for them:** full reviews are easier than first + last
   sentences, and the content part is the larger part (RoBERTa-large: −0.084 / +0.074).

**Code review (this request: D44, D45, D46, lock file):**
- **Fixed: correlation without effect size.** `name_probe` reported ρ alone. HF DistilBERT showed
  ρ = 0.51 on name effects of at most 0.0003 (noise). Added `effect_sd` and `max_abs_effect`.
- **Fixed: constant input.** A model that ignores names entirely produced scipy warnings; it now
  returns NaN explicitly, and the test runs with warnings treated as errors.
- **Fixed: stale probe files.** If a later run skipped a model (e.g. RoBERTa-large not cached),
  its old probe predictions would have been analysed as current. The stage now removes them.
- **Fixed: overlapping chart labels.** Our and HF's DistilBERT sit on the same point of the
  Mirage Gap chart; coinciding points now share one label.
- **Checked:** external models are in their own Holm family in every multi-model table and are
  excluded from H2. The probe is never part of `EVAL_SETS`. Seed robustness touches DistilBERT
  rows only. Every previously reported number was verified unchanged (all tables and early
  warning).
- **Lock file (shortfall H2):** `requirements-lock.txt` is a `pip freeze` of the environment that
  produced the results, with the CPU torch install line.

## D47 — A formal test for H1

**Problem:** H1 ("every model, including DistilBERT, has a clearly higher error rate on negation
and sentiment-word typos than on the normal test set") was judged by eye from separate
confidence intervals. That's not a test, and it's the one hypothesis without one.

**Decision:** `evaluate.h1_test` → `results/h1_test.csv`. For every base model (our six, plus the
external models in their own Holm family):
- **Comparison set: the SST-2 test sentences that contain *no* negation word.** About 19% of test
  sentences contain negation, and most of the real-negation set comes from them, so comparing
  with the whole test set would partly compare negated sentences with themselves. The overall
  test error is reported next to it for reference.
- **Tests:**
  - real negated sentences (`sst2_neg`, 519);
  - negation templates (`beh_negation`, "not X" → negative);
  - "X / not X" pairs (`beh_negation_dir`);
  - sentiment-word typos (`beh_typo`).
- **Statistic:** error(test) − error(non-negated test sentences), with a 95% bootstrap CI that
  resamples each group separately (they're different sentences), and the same SE-based p-value as
  every other comparison (D28). The ratio to the comparison error is reported too.
- **"Clearly higher"** means a positive difference with Holm p < 0.05 across every
  (model, test) row in the family.

**Rejected:** comparing with the whole test set (overlapping items), and a pooled z-test (it
would duplicate the bootstrap the project already uses everywhere else).

**Results (D47; difference in error, percentage points, against SST-2 test sentences without
negation; ✔ = clearly higher, Holm p < 0.05):**

| model | real negation | "not X" templates | "X / not X" pairs | typos |
|---|---|---|---|---|
| Logistic regression | +7.3 ✔ | +11.7 ✔ | +12.2 ✔ | +29.9 ✔ |
| Multinomial NB | +8.4 ✔ | +33.7 ✔ | +41.8 ✔ | +23.1 ✔ |
| Gaussian NB | +8.6 ✔ | +54.2 ✔ | +56.6 ✔ | −25.4 |
| Random forest | +3.6 | −3.2 | +1.0 | +12.8 ✔ |
| Decision tree | +5.5 | −25.8 | −13.3 | −10.8 |
| DistilBERT (ours) | +2.6 (p 0.33) | −7.4 (0 errors) | −7.4 (0 errors) | +32.0 ✔ |
| DistilBERT (HF, external) | +3.5 ✔ (p 0.045) | −7.3 | −7.3 | +31.7 ✔ |
| RoBERTa-large (external) | +1.6 | −5.3 | −5.3 | +23.9 ✔ |

**H1 verdict: partly supported.**
- **Typos clearly hurt every model that reads sentiment words**, *including every transformer*
  (DistilBERT +32 points). That half of H1 holds as worded.
- **Negation clearly hurts the bag-of-words models** (LR and Multinomial NB on all three negation
  tests; Gaussian NB too).
- **It does not clearly hurt the transformers.** They make no template errors, and our DistilBERT
  is only 2.6 points worse on real negated sentences (not significant). So "including DistilBERT"
  fails for negation, which is the same finding as H2 seen from the other side.
- **The weakest models fail typo tests *less* than normal sentences** (decision tree, Gaussian NB).
  They barely react to the sentiment word anyway, the same effect D40 corrected for in the
  robustness score. The decision tree's negation results are similar: its normal error is
  already 34%.

**Code review:**
- Extracting `stats._summarize` from `bootstrap` is behaviour-preserving: every reported number
  was re-checked, unchanged.
- `diff_independent_ci` resamples the two groups separately and uses the project's SE-based
  p-value.
- The comparison set excludes exactly the negated test sentences; the validation sentences in
  the negation set were never in it.
- The pair tests count failed *pairs* and the comparison counts failed *sentences*. Both are
  failure rates, so they are comparable; this is stated in the docstring.
- No bugs found.

## D48 — Demo polish: faster reruns, readable word importance, theme-neutral colours (refines D41)

**What changed in `app/shortcut_explorer.py`** (layout only; `mirage.demo` is unchanged, so no
number the demo shows changed):
- **Caching.** Model scores, word importance and the unusual-input distance are cached per
  (model, sentence) with `st.cache_data`. Moving the low-confidence slider or a toggle no longer
  re-runs any model (before, every interaction re-ran all eight, RoBERTa-large included), and
  switching light mode on and off reuses what both modes share. The low-confidence flag is still
  applied at display time, so the slider takes effect at once.
- **Word importance gets its own full-width section** ("Which words drove each verdict?"), one
  row per model, under the verdict table. Before, it was a fifth table column that squeezed a
  sentence into one word per line and was cut off on narrow screens.
- **A colour key** above the words: blue = pushed positive, red = pushed negative, grey italic =
  never seen in training, darker = mattered more to that model (shading is scaled per model).
  Hovering a word shows its exact effect on the positive probability.
- **Theme-neutral colours.** Lines, probability-bar tracks and header text use semi-transparent
  greys or the inherited text colour instead of fixed dark-grey hex values, so the tables stay
  readable if the dark theme is chosen. Blue / red keep their project-wide meaning.

**Rejected:** hosting on Vercel. Vercel serves JavaScript frontends and short serverless
functions; the demo keeps PyTorch models (about 2-3 GB) loaded in memory. A public link, if
wanted, fits Hugging Face Spaces (Streamlit, free CPU tier with enough RAM for all eight models).
The presentation itself runs locally, with `app/screenshots/` as the fallback.

**Code review:** cache keys include the model name and the exact text tuple, so two models or two
sentences never share an entry; the unhashed `_clf` argument is safe because a name always maps to
the same loaded model. User text is still HTML-escaped everywhere it is rendered (chips, tabs,
the A/B lines). No bugs found.

## D49 — The demo shows the research numbers behind each live example (refines D41)

**Why:** a live flip on one sentence could be a lucky pick. Showing, on the same screen, how the
model does on the full test makes one example stand for the measured result: Multinomial NB flips
on "Binoche → Hopkins" *and* looks 83% / behaves 55%.

**What the app shows** (sidebar toggle "Show research numbers", on by default):
- **A summary line above the A/B table:** "1 of 8 models flip: Multinomial NB", and for a curated
  example, how often each flipping model fails the full test behind that shortcut ("In the full
  test, Multinomial NB fails 39% of name-swap pairs").
- **A "looks → behaves" column** in both tables: SST-2 test accuracy → robustness score, with the
  gap (the Mirage Gap, D40).

**Where the numbers come from:** `results/mirage_gap.csv` (base variant), read by
`demo.load_findings`; nothing is recomputed. `demo.SHORTCUT_TESTS` maps an example's `shortcut`
to its robustness tests: names → name-swap pairs, typos → typo pairs, negation → negation
template pairs and real negated sentences. Vocabulary and sarcasm have no matching test, so they
show only the flip count. "Fails" uses the robustness score's definition (a pair fails unless both
sentences are right), so it can differ slightly from a flip rate: the presenter note's "Naive
Bayes flips on 38% of pairs" and the screen's "fails 39%" are both correct.

**Also fixed:** the header said models are "59-92% right"; RoBERTa-large reaches 95%, so it now
says 59-95%. A change that rounds to zero is shown as "+0%", not "-0%".

**Code review:** the demo works without `mirage_gap.csv` (fresh clone): the column and the
full-test lines disappear. A model missing from the results shows "—". The helper is named
`shortcut_failures`, not `test_failures`, so pytest doesn't collect it from the test module's
imports. A test checks that every column the demo reads is one `evaluate` writes. No bugs found.

## D50 — One home for results, archived plans, one to-do list (documentation)

**Problem:** the same results were copied into five files (README, this log, reportplan's verdicts,
and the two results files), and they had started to drift: the README said 285 seed checks where
the current table has 291, and "the ranking holds in all 12 leave-one-out checks" stopped being
true when the external models were added (now one check swaps two neighbours). Two finished build
plans were still presented as current, and the report's steps were split between
`whatyouneedtodo.md` and `reportplan.md`.

**Decision:**
- **`hypothesisanswers.md` and `interestingfindings.md` are the one home for current results**,
  checked against `results/*.csv`. The README keeps a short summary (one table, one verdict line
  per hypothesis) and links to them; `reportplan.md` links to them instead of restating verdicts.
  This log's numbers are a dated record of each decision (a note at the top says so).
- **Finished plans archived:** `MIRAGE_IMPLEMENTATION_PLAN.md` and `newadditionsdesign.md` moved
  to `docs/archive/`, each with a note that the code and this log win where they differ. CLAUDE.md
  no longer points at the implementation plan, and its docs rule names the results files instead
  of the design document.
- **One to-do list:** every user step, including the report walkthrough (format rules, verdicts,
  main text vs appendix, failure examples, review, personal parts, submit) and Claude's steps
  between them, is in `whatyouneedtodo.md`; finished steps are condensed into a "Done" table,
  keeping the Kaggle re-run tips. `reportplan.md` is the skeleton and sources only.
- **ARCHITECTURE.md** stays at the level of stages, layers and interfaces: its diagram now shows
  the `name-probe` and `occlusion` stages and the newer tables, and its testing section summarises
  the levels and links to `codeexplanations.md` for the file-by-file list.

**Rejected:** merging ARCHITECTURE.md and codeexplanations.md (different readers: the design vs a
file tour), and deleting the old plans (they record what was planned against what was built).

**Check:** every relative markdown link in the repo resolves; the README's summary table was
re-read from `results/main_table.csv` and `results/mirage_gap.csv`.

