# Shortfalls: known limitations of Mirage

Each item has four parts: **what** the limitation is, **why it matters** for the conclusions, a **possible fix**, and the
**effort** it would take. They are grouped by area and ranked within each group. Severity:

- 🔴 can change a headline conclusion or undermine its validity. Fix it, or state it clearly in the report.
- 🟠 weakens a result. Worth fixing if there is time.
- 🟢 a scope limit. Mention it in the report's Limitations section.

When one gets fixed, move it to the **Resolved** section at the bottom and add an entry to `majordecisions.md`.

---

## A. What the task can and can't measure

### A1. 🟢 Sentiment is only positive vs negative
**What:** every input is forced into one of two labels. There is no neutral, no mixed ("great acting, awful plot"),
no intensity ("okay" vs "masterpiece") and no specific emotion (anger vs sadness). SST-2 dropped its neutral
sentences when it was built.
**Why it matters:** we can only claim things about *polarity* shortcuts. A model that can't handle mixed reviews
looks fine here, because mixed reviews aren't in the data.
**Possible fix:** also evaluate on **SST-5** (five classes, from very negative to very positive). The trees are
already in the same dataset, so the loader needs little change. Add a "mixed-sentiment" behavioral template
(`<positive aspect> but <negative aspect>`).
**Effort:** medium (about 1 day).

### A2. 🟢 Confidence is not intensity
**What:** `p_pos = 0.87` means the model is 87% *sure* the text is positive, not that it's "87% positive".
**Why it matters:** readers easily misread the reliability diagrams and the confidence-based early warning.
**Possible fix:** say so explicitly in the report. If SST-5 is added (A1), correlate `p_pos` with the
fine-grained label to test whether confidence tracks intensity at all.
**Effort:** small.

### A3. 🟢 Only English, only movie reviews
**Why it matters:** the conclusions about shortcuts and the SST-2 → IMDb shift may not transfer to other domains
(product reviews, tweets) or other languages.
**Possible fix:** add a third evaluation set from another domain, such as Amazon or Yelp polarity (a 2,000-row
sample, same loader pattern).
**Effort:** small to medium.

---

## B. Data

### B1. 🔴 The held-out set leaks: 90% of held-out phrases appear inside training phrases
**What:** SST-2's training set has one row per *phrase* of each parse tree, so a sentence and its sub-phrases are
all separate rows. `split_heldout` splits rows at random, so a held-out phrase's parent sentence usually stays in
training. Measured on the real data, **89.6%** of held-out phrases appear verbatim inside a training phrase.
Validation has 0% overlap.
**Why it matters:**
- Held-out accuracy is inflated: the model has effectively seen the answers.
- All classical tuning (`tune` picks C, depth and alpha on held-out) and DistilBERT's best-epoch choice use this
  number, so they favour memorization: deeper trees and weaker regularization.
- The *reported* results use `sst2_val`, which is clean, so those numbers stay honest. But the model choices
  behind them are biased.

**Possible fix:**
- Reconstruct the groups: a phrase belongs to the longest training phrase that contains it.
- Hold out whole groups (`GroupShuffleSplit`), or simply hold out only full sentences plus all their sub-phrases.
- Re-run tuning and record the change in `majordecisions.md`.

**Effort:** medium (half a day: grouping 64k phrases by containment needs a suffix index or a sort-based trick).

### B2. 🟠 The negation test set is small (180 sentences) and imbalanced (36% positive)
**Why it matters:** confidence intervals on negation error will be wide, roughly ±7 points, so differences
between models smaller than that won't be significant. The imbalance means a model biased toward "negative"
looks better on negation than it really is.
**Possible fix:**
- Report per-class error and balanced accuracy on this subset.
- Add more negated sentences: SST-2's training *full sentences* that aren't used for tuning, or a hand-checked
  set from IMDb sentences.
- Lean more on the templated negation tests, which can be enlarged at will.

**Effort:** small (metrics) to medium (new data).

### B3. 🟠 Validation is also the test set
**What:** SST-2's test labels are hidden, so `sst2_val` (872 sentences) is our "test" set. But
`evaluate.median_seed` chooses DistilBERT's representative seed by *validation* accuracy.
**Why it matters:** it's a mild selection-on-test: the per-item analyses use the seed chosen on the same data
they're reported on.
**Possible fix:** choose the median seed by held-out accuracy (after B1 is fixed), or report all three seeds for
the per-item tables.
**Effort:** small.

### B4. 🟠 The IMDb shift mixes several changes at once
**What:** IMDb differs from SST-2 in length (about 230 words vs about 20), writing style, vocabulary (plot
summaries, actor names) and label definition (whole-review star rating vs sentence sentiment).
**Why it matters:** when accuracy drops on IMDb, we can't say *which* difference caused it, so "shortcut
failure" is only one explanation among several.
**Possible fix:**
- Bucket IMDb results by review length.
- Evaluate on IMDb's first and last sentences only, which is SST-length text from the IMDb domain. This
  separates the effect of length from the effect of domain.

**Effort:** small to medium.

### B5. 🟢 IMDb sample is 2,000 reviews
**Why it matters:** fine for accuracy (±2 points), but the error-detection AUROC for strong models rests on a
few hundred errors.
**Possible fix:** raise `imdb_sample_size` to 5,000 to 10,000. Classical inference is cheap; DistilBERT
inference on a GPU is minutes.
**Effort:** trivial (a config change), plus compute.

### B6. 🟢 Label noise in SST-2
**What:** SST-2's crowd labels are noisy, especially for short phrases, and some "errors" are actually label
mistakes.
**Possible fix:** when picking failure examples for the report, check them by hand and mark any with a
questionable label.
**Effort:** small.

---

## C. Models

### C1. 🔴 IMDb comparison is unfair because of truncation
**What:** DistilBERT reads only the first **256 tokens** of each IMDb review (`imdb_max_length`), while the
classical TF-IDF models read the **whole** review. Many IMDb reviews put the verdict at the end.
**Why it matters:** DistilBERT's IMDb drop is partly self-inflicted. Comparing it with the classical drop
overstates or understates the shortcut story.
**Possible fix:**
- Head+tail truncation: the first 128 tokens plus the last 128.
- Or sliding windows with averaged probabilities.
- At minimum, also evaluate the classical models on the same truncated text so both sides read the same thing.

**Effort:** small (the truncated-classical control) to medium (sliding windows).

### C2. 🟠 DistilBERT is not a "shortcut-free" reference
**What:** the project treats DistilBERT as the model that *doesn't* use shortcuts (`REFERENCE` in
`evaluate.py`). Transformers are known to use shortcuts too, such as lexical overlap and negation cues, and its
pre-training text may include review-like language.
**Why it matters:** "gap to DistilBERT" measures *relative* shortcut reliance, not absolute reliance.
**Possible fix:**
- Word the hypotheses relative to DistilBERT, and report DistilBERT's own failure rates prominently.
- Optionally add a stronger model (RoBERTa-base), or an off-the-shelf sentiment model, as a second reference.

**Effort:** small (wording) to medium (another model).

### C3. 🟠 DistilBERT hyperparameters are not tuned
**What:** fixed lr 2e-5, 3 epochs and batch size 32, while the classical models get a grid search.
**Why it matters:** this asymmetry slightly favours the classical models. That's conservative for our claims,
but worth stating.
**Possible fix:** a small grid (lr in {2e-5, 3e-5, 5e-5}, epochs in {2, 3, 4}) on one seed, on the GCP or
Kaggle GPU.
**Effort:** medium (GPU time).

### C4. 🟠 Few seeds
**What:** DistilBERT has 3 seeds. Classical models have one, although RF depends on its seed.
**Why it matters:** the seed-to-seed spread can't be estimated well from 3 runs, and the RF results could
change with a different seed.
**Possible fix:** 5 seeds for DistilBERT. For RF, run 3 seeds and report the spread.
**Effort:** small (compute only).

### C5. 🟢 The classical feature set is basic
**What:** TF-IDF on unigrams and bigrams only. No trigrams, character n-grams or word embeddings.
**Why it matters:** a reviewer could argue that negation failures come from the features we chose rather than
from classical models as such.
**Possible fix:**
- Add a (1, 3) n-gram variant.
- Add an averaged-GloVe + LR variant, as a "non-transformer but semantic" middle ground.

**Effort:** small to medium.

### C6. 🟢 Gaussian Naive Bayes on SVD features is a weak model
**Why it matters:** GNB will likely do much worse than the others, and readers may dismiss it as a strawman.
**Possible fix:** keep it, since the proposal lists it, but present it as a lower bound rather than a fair
competitor.
**Effort:** trivial (wording).

### C7. 🟠 Negation marking uses a simple scope rule
**What:** `negation_mark` adds `NOT_` to every word from a negation word up to the next punctuation mark.
It doesn't handle "not only … but also", double negatives, "no doubt", or scope across clauses that have no
punctuation.
**Why it matters:** H3 ("negation marking closes the gap") tests *this heuristic*, not negation handling in
general. A weak result might come from the rule rather than the idea.
**Possible fix:** add exceptions for fixed phrases ("not only", "no doubt", "nothing but"), and an alternative
scope rule (the next 3 words only); report both.
**Effort:** small.

---

## D. Behavioral tests

### D1. 🔴 The templates and word lists are hand-written by us
**What:** `wordlists.py` and `templates.py` were written by the project author (with Claude). The sentences,
such as "The movie was not brilliant.", are clean and short, and unlike SST-2's style.
**Why it matters:**
- **Author bias:** we chose the words, so results depend on which adjectives we picked.
- **Out of distribution:** a model might fail a template because templates look odd, not because of the
  shortcut being tested.

**Possible fix:**
- Have someone else review the word lists before the final run (this is in `whatyouneedtodo.md`).
- Run the "simple" templates as a control. If a model fails *those*, its failures on negation templates aren't
  evidence of a negation shortcut.
- Report error rates per word, so one odd adjective can't drive the result.

**Effort:** small.

### D2. 🟠 The behavioral suite is small
**What:** 400 test cases in total; 50 name-swap pairs.
**Possible fix:** raise the counts in `configs/behavioral.yaml`. Generation is free, and prediction is fast.
**Effort:** trivial.

### D3. 🟢 "Not bad" cases are not scored
**What:** "not bad" isn't clearly positive (it's often lukewarm), so those cases are reported separately and not
scored (decision D11).
**Why it matters:** we ignore half of the negation behaviour.
**Possible fix:** label a sample of "not <negative>" sentences by hand (or with several annotators) to decide
whether they count as positive.
**Effort:** small to medium.

### D4. 🟢 Only four kinds of behavioral test
**What:** simple, negation, typo, name swap. Missing: intensifiers ("very", "slightly"), contrast ("but"),
sarcasm, and irrelevant additions ("…, and the popcorn was salty").
**Possible fix:** add 1–2 templates. The suite design makes this a new generator function in `templates.py`.
**Effort:** small per template.

---

## E. Explanations

### E1. 🟠 Integrated Gradients depends on its baseline and covers a small sample
**What:**
- IG measures each word's contribution *relative to a baseline input*. We use all `[PAD]` tokens, and a
  different baseline, such as `[MASK]` or a blurred embedding, would give different attributions.
- It runs on at most 100 examples (`--max-ig 100`) with 50 steps.

**Why it matters:** the word overlap between models (Jaccard and RBO) and the "which words does DistilBERT use"
claims rest on one baseline choice and a small sample.
**Possible fix:**
- Run a `[MASK]` baseline too and report the overlap between the two.
- Check IG's convergence delta and raise `n_steps` if it's large.
- Raise `--max-ig` when running on a GPU.

**Effort:** small (on a GPU).

### E2. 🟠 We compare explanations that mean different things
**What:** LR weights and tree importances are *global* (about the whole model), while IG is *local* (about
one prediction, then averaged). The depth-4 tree we explain is a *separate surrogate*, not the actual RF.
**Why it matters:** low overlap between models may reflect the explanation methods rather than the models.
**Possible fix:**
- Use the same local method for every model: SHAP or IG-style occlusion on the classical models.
- For trees, use the RF's own permutation importance instead of a surrogate tree.

**Effort:** medium.

### E3. 🟢 Attribution is not causation
**What:** high attribution doesn't prove the model "relies on" a word. The deletion check helps, but deleting
words also creates unnatural sentences.
**Possible fix:** replace words with a neutral substitute instead of deleting them, and report both.
**Effort:** small.

---

## F. Early warning (H4)

### F1. 🔴 Distance barely separates SST-2 from IMDb: AUROC 0.646 in the smoke test
**What:** the sanity check, which asks whether Mahalanobis distance on MiniLM embeddings can tell SST-2
validation from IMDb, gave **0.646**. It should be near 1.0, since the two are very different. That run fitted
the Gaussian on a 3,000-phrase subsample.
**Suspected cause:** averaging the embeddings of a 230-word review pulls it toward the centre of embedding
space, so long reviews look "typical" even though they're from a different domain. Also, MiniLM reads at most
256 word pieces and silently cuts the rest of the review.
**Why it matters:** if distance can't even detect the domain change, H4's comparison of distance and confidence
for predicting errors is set up to fail, and the negative result would come from the method, not from the
hypothesis.
**Possible fix (in order):**
1. Refit on the full SST-2 training set. A background check of exactly this was running and hadn't finished
   when this file was written.
2. Embed each IMDb review sentence by sentence, then take the *maximum* distance (or the mean).
3. Use kNN cosine distance (k=10) or a class-conditional Gaussian instead of one Gaussian.
4. Normalize embeddings (L2) before fitting.

**Effort:** small to medium. The options can be added to `configs/warning.yaml` as a `distance:` setting.

### F2. 🟠 The embedding model is not the classifier
**What:** distance is measured in MiniLM's embedding space, which is the same for every classifier.
**Why it matters:** distance can't explain why *one* model fails where another doesn't. It can only flag
"unusual inputs".
**Possible fix:** for DistilBERT, also use its own `[CLS]` / last-layer features. That's the standard
Mahalanobis OOD detector (Lee et al., 2018).
**Effort:** medium.

### F3. 🟢 Few errors to detect for strong models
**What:** if DistilBERT gets about 88% of IMDb right, there are only about 240 errors in 2,000 reviews, so the
AUROC confidence intervals are wide.
**Possible fix:** a larger IMDb sample (see B5).
**Effort:** trivial.

---

## G. Statistics

### G1. 🟠 No correction for running many tests
**What:** dozens of comparisons (McNemar tests, gap CIs, AUROC comparisons) are made at α = 0.05. By chance,
about 1 in 20 true-null comparisons will look significant.
**Possible fix:** apply Holm–Bonferroni within each family of tests (one per research question). This is a few
lines in `evaluate.py`.
**Effort:** small.

### G2. 🟠 Confidence intervals ignore seed variance
**What:** the per-item analyses use DistilBERT's median seed only, so their CIs reflect item sampling, not
training randomness.
**Possible fix:** a two-level bootstrap: resample seeds, then items. Or report the analysis for each of the
3 seeds side by side.
**Effort:** small to medium.

### G3. 🟢 Expected calibration error depends on the binning
**What:** ECE uses 15 equal-width bins. With 180 negation sentences, many bins are nearly empty.
**Possible fix:** use equal-mass bins on small sets, or report the Brier score too.
**Effort:** small.

---

## H. Engineering and reproducibility

### H1. 🔴 Not yet run end to end on the full real data
**What:** the pipeline has been run on real SST-2 for `prepare`, and with smoke-sized DistilBERT runs. The full
classical training, the 3-seed DistilBERT run and the `evaluate` / `explain` / `warn` stages on real
predictions haven't run yet.
**Why it matters:** bugs that only appear at full scale are still possible: memory use (RF with 300 deep
trees), long runtimes, and NaNs in rare cases.
**Fix:** this is the plan. Run `python -m mirage all` after the Kaggle models are downloaded, then check
the results by hand.

### H2. 🟠 Dependencies aren't pinned
**What:** `pyproject.toml` gives minimum versions only. Kaggle currently runs Python 3.13, while local runs
use the `.venv`. The `transformers` 4 → 5 change already broke one argument (D19).
**Possible fix:** commit a `requirements-lock.txt` (`pip freeze`) from the run that produces the final results,
and record the Kaggle environment in `train_summary.json`.
**Effort:** small.

### H3. 🟢 No automated test runs on GitHub
**Possible fix:** a GitHub Actions workflow that runs `ruff` and `pytest` on each push. The tests use synthetic
data, so they need no downloads. About 20 lines of YAML.
**Effort:** small.

### H4. 🟢 Some docs are out of date
**What:**
- `ARCHITECTURE.md` says 54 tests; there are 60.
- The README doesn't mention that `prepare` has already been run.
- The OOD finding (F1) isn't recorded in `majordecisions.md` yet.

**Effort:** trivial.

---

## Suggested order if there's time

1. **B1** (leaky held-out): it biases every tuned model.
2. **F1** (distance sanity check): without it, H4 can't be tested properly.
3. **C1** (truncation): it makes the IMDb comparison fair.
4. **D1** (template control and per-word errors).
5. **G1** (multiple-comparison correction).
6. **B2**, **C7** and **B4**: cheap, and each makes a headline result more robust.
7. Everything 🟢: one sentence each in the report's Limitations section.

## Resolved

*(nothing yet)*
