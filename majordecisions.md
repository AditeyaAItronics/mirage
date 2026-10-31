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
