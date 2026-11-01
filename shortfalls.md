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

### B1. ✅ RESOLVED (D21): the held-out set leaked; 90% of held-out phrases appeared inside training phrases
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

### B2. ✅ Size fixed (D22); 🟢 imbalance remains
**Was:** 180 negated sentences, about ±7 points per model.
**Now:** **519** sentences (validation + public SST-2 test), about ±4 points.
**Still true:** 36% positive. Report per-class error and compare against the 64% majority rate.

### B3. ✅ RESOLVED (D22): validation was also the test set
`sst2_val` now only makes choices (DistilBERT's median seed). `sst2_test` (1,820 sentences with the
original public labels) is the reported in-distribution set.

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

### B5. ✅ RESOLVED (D22): IMDb sample was 2,000 reviews
Now 10,000. The weakest model has 2,000 or more IMDb errors.

### B6. 🟢 Label noise in SST-2
**What:** SST-2's crowd labels are noisy, especially for short phrases, and some "errors" are actually label
mistakes.
**Possible fix:** when picking failure examples for the report, check them by hand and mark any with a
questionable label.
**Effort:** small.

### B7. 🟢 NEW: SST-2 test labels come from a third-party copy (`SetFit/sst2`)
**What:** GLUE hides the test labels, so we use a public copy of the original Stanford labels.
**Checked:** its validation labels match GLUE's on all 823 sentences that match word for word, and
its test split is balanced.
**Residual risk:** test labels can't be checked against GLUE. State the source in the report.
**Effort:** none (wording).

### B8. 🟢 NEW: IMDb text is capped, so IMDb results are on capped reviews
**What:** to give every model the same text (C1), reviews over 360 words keep their first 220
and last 140 words. That's **15.9%** of the sample.
**Why it matters:** for those reviews, every model loses the middle of the text. The comparison
is fair, but it isn't "full IMDb".
**Possible fix:** report accuracy for capped vs uncapped reviews (`capped` column); raise the cap
if DistilBERT's 512-token limit allows.
**Effort:** small.

### B9. 🟢 NEW: part of the negation set also picks DistilBERT's seed
**What:** 180 of the 519 negated sentences come from validation, which is also where DistilBERT's
median seed is chosen.
**Why it matters:** a tiny selection effect on per-item negation results.
**Possible fix:** report the negation gap on the 339 test-only sentences as a robustness check.
**Effort:** small.

---

## C. Models

### C1. ✅ RESOLVED (D23): IMDb comparison was unfair because of truncation
Every model now reads the same capped text (first 220 + last 140 words). DistilBERT's IMDb limit
went from 256 to 512 tokens, and anything still too long keeps its head and tail tokens, not just
the start.

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

### C8. 🟠 NEW: DistilBERT is trained on short phrases but tested on long reviews
**What:** training phrases average about 9 words (128-token limit); IMDb reviews are up to 512 tokens.
**Why it matters:** part of DistilBERT's IMDb drop may be length generalisation, not domain shift.
**Possible fix:** bucket IMDb accuracy by length (`n_words`); compare with the classical models'
curves, which don't depend on position.
**Effort:** small (analysis only).

### C9. ✅ RESOLVED for the tree (D29); 🟢 GNB and MNB remain as notes
**Tree:** it now keeps at least 10 examples per leaf (accuracy picked 20). Its probabilities are
usable: ECE 0.339 → 0.144, IMDb confidence AUROC 0.500 → 0.562, and IMDb accuracy *rose*
0.592 → 0.612. Cost: SST-2 test 0.661 → 0.653.
**GNB** stays near chance (SST-2 test 0.591). That's the H6 comparison, so present it as a lower
bound (C6), not a fault to tune away.
**MNB** pushes half its IMDb probabilities to exactly 1.0. The early-warning combiner handles this
(D26). Note it next to MNB's calibration numbers.

### C10. 🟠 NEW (first result): negation marking helps templates, not clearly real sentences
**What:** after Holm, `NOT_` marking cuts the directional-test error significantly for every model
(LR −25.7 points, MNB −39.5). On real negated SST-2 sentences (n = 519) it's significant only
for GNB. LR's gain is −2.9 points [−0.2, +5.8], Holm p 0.64.
**Why it matters:** H3 rests on real negation. The templates are short and have a single, clear
negation scope, which is the case `negation_mark` handles best (C7).
**Possible fix:** check which real negated sentences change prediction under marking. Try the
scope variants from C7. Report the template and real-sentence effects side by side, not just the
template one.
**Effort:** small.

---

## D. Behavioral tests

### D1. 🟠 MITIGATED (D24): templates and word lists are hand-written by us
**Done:**
- Every behavioral case records its adjective.
- `word_errors.csv` shows failures per word.
- `negation_control.csv` re-scores negation on only the positive words the model gets right in
  every simple case. For LR: 0.262 on all words, 0.284 on known words, so its negation failure is
  real and not caused by the words.

**Still open:** someone other than the author should review `wordlists.py` (in
`whatyouneedtodo.md`). Code can show the bias, but can't remove it.

### D2. ✅ RESOLVED (D22): behavioral suite was small
5× larger: 2,000 cases (2,750 rows); with the directional test (D27), 2,750 cases (4,250 rows).

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

### D5. ✅ RESOLVED (D27): the negation template rewarded models that lean negative
A directional test now pairs each "X" with "not X" and asks whether the model's *own* answer
flips. Only pairs whose affirmative the model gets right count, so label bias can't help.
The decision tree's negation error went from 0.096 (template) to **0.183**, on only 49% testable
pairs; for 60% of its pairs "not" doesn't move its probability at all. LR's barely changed
(0.262 → 0.266). See `negation_direction.csv`.

### D6. 🟢 NEW: the "known words" control is strict
**What:** a word counts as known only if the model gets *every* simple case with it right (about 12
cases each). Weak models have few known words (MNB negmark: 3 of 20), so their controlled
estimate rests on about 70 cases.
**Possible fix:** a softer threshold (≥ 90% right), reported alongside the strict one.
**Effort:** trivial.

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

### F1. 🟠 MITIGATED (D25): distance barely separated SST-2 from IMDb
**Was:** AUROC 0.646 with whole-review embeddings.
**Now:** each sentence is scored and the review keeps its most unusual one: AUROC **0.926**
(SST-2 test vs 10,000 IMDb reviews). The distance type barely mattered (0.67–0.69 for whole
reviews whichever we used). See F4 for the caveat.

### F2. 🟠 The embedding model is not the classifier
**What:** distance is measured in MiniLM's embedding space, which is the same for every classifier.
**Why it matters:** distance can't explain why *one* model fails where another doesn't. It can only flag
"unusual inputs".
**Possible fix:** for DistilBERT, also use its own `[CLS]` / last-layer features. That's the standard
Mahalanobis OOD detector (Lee et al., 2018).
**Effort:** medium.

### F3. ✅ RESOLVED (D22): few errors to detect
With 10,000 IMDb reviews, LR makes about 2,000 errors and the others make more. AUROC CIs are now about ±0.01.

### F4. 🟠 NEW: sentence-max distance is partly a length measure
**What:** the maximum over more sentences is larger by chance, so long reviews look more unusual.
**Done:** length is reported as its own error predictor. In the first run, distance (0.49–0.53)
and length (0.50–0.54) were both near chance for the classical models, so there's nothing yet for
length to explain away.
**Possible fix:** a length-matched sanity check (SST-2 test vs single IMDb sentences); distance
residualised on the number of sentences.
**Effort:** small.

### F5. 🟢 NEW: sentences are split with a regex
**What:** reviews are split on `. ! ?` followed by whitespace. IMDb text has "..." runs and
sentences with no space after the full stop, so some "sentences" are merged or broken.
**Possible fix:** a proper sentence splitter (`nltk` punkt or `pysbd`).
**Effort:** small.

### F6. 🟠 NEW (first result): distance does not flag classical models' IMDb errors
**What:** with a detector that clearly sees the domain change (0.926), error-detection AUROC for
distance is 0.49–0.53 for every classical model, while low confidence reaches 0.67–0.77 (LR,
MNB, RF).
**Why it matters:** H4 looks unsupported for the classical models. This is a *result*, not a
bug, but it should be confirmed with DistilBERT and with F4's length-matched check before it
goes in the report.
**Possible fix:** none needed. Report it as is, and try DistilBERT's own features (F2) before
concluding.

---

## G. Statistics

### G1. ✅ RESOLVED (D28): no correction for running many tests
Every paired comparison now has a p-value, adjusted with Holm–Bonferroni within its family (McNemar
per dataset, negation gaps, negation-marking effects, early-warning AUROC differences). H4 now has
a direct paired test of distance against confidence. The correction already changes a conclusion:
LR's +1.2-point gain from negation marking on SST-2 test has a raw p of 0.013 but isn't significant
after Holm.

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

### G4. 🟢 NEW: p-values use a normal approximation
**What:** bootstrap p-values come from the standard error, assuming a roughly normal bootstrap
distribution. Counting resamples instead had a floor that Holm made useless (D28).
**Why it matters:** fine for differences of means and AUROCs at our sample sizes. The H3
gap-closed *ratio* can be skewed when the base gap is small.
**Possible fix:** a BCa interval, or a permutation test for the ratio.
**Effort:** small.

### G5. 🟢 NEW: the families are a design choice
**What:** Holm is applied per family (McNemar per dataset; one family per table elsewhere). Larger
families make every test stricter.
**Possible fix:** state the families in the report's methods section (they're listed in D28), and
show raw p-values next to adjusted ones.
**Effort:** none (wording).

---

## H. Engineering and reproducibility

### H1. 🟠 MOSTLY RESOLVED (D26): not yet run end to end on the full data
The whole classical pipeline has run on the real data:
- `prepare`, then both variants of all 5 models;
- predictions on all 6 evaluation sets (10,000 IMDb reviews included);
- `evaluate`, `explain` (without IG), `warn` and `figures`.

Screening it found and fixed three problems: LR's grid edge, the early-warning combiner, and
redundant predictions.
**Still open:** DistilBERT at full scale (3 seeds on Kaggle), which also unlocks IG and the
negation-gap tables.

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

### H4. ✅ RESOLVED: docs out of date
ARCHITECTURE, README, CLAUDE.md and whatyouneedtodo were updated with D21–D26.

### H5. 🟢 NEW: model files are large
The random forest saves at **822 MB** per variant (300 unpruned trees) and GNB at 176 MB, about
2 GB in `artifacts/models`. They're git-ignored, but a laptop with little disk or RAM will
struggle.
**Possible fix:** `joblib.dump(..., compress=3)`, or cap RF depth if accuracy holds (depth 30 was
much worse: 0.630 vs 0.772).
**Effort:** trivial (compression).

### H6. 🟢 NEW: skipping predictions relies on file timestamps
**What:** `predict` skips a model whose prediction files are newer than its model files. Unzip
tools normally keep the zip's timestamps. One that doesn't would only cause a slow CPU redo, not
wrong results.
**Effort:** none.

---

## Suggested order if there's time

All 🔴 items from the first pass are resolved or mitigated (B1, C1, D1, F1, H1). Next, in order:
1. ~~**D5** (directional negation test)~~: done, see D27.
2. ~~**G1** (multiple-comparison correction)~~: done, see D28.
3. ~~**C9** (degenerate DT probabilities)~~: done, see D29.
4. **F4** and **C8** (length checks): both are analyses on columns that already exist.
5. **B4**, **C7** and **B9**: cheap robustness checks for the headline results.
6. Everything 🟢: one sentence each in the report's Limitations section.

## Resolved

- **B1, leaky held-out split.** Fixed by a sentence-grouped split (`data.heldout_mask`). Leakage went from 89.6% to 0 of 3,378 phrases. See `majordecisions.md` D21.
- **B2 (size), B3, B5, D2, F3: too little data, and validation reused as test.** Fixed by adding 1,820 SST-2 test sentences, pooling the negation set (519 sentences), using 10,000 IMDb reviews and a 2,000-case behavioral suite (D22).
- **C1, unequal IMDb truncation.** Fixed by giving every model the same capped text, plus head+tail tokens at 512 for DistilBERT (D23).
- **D1, template bias.** Mitigated by per-word errors and a known-words negation control (D24); the human review of the word lists is still open.
- **F1, a distance blind to the domain change.** Mitigated by scoring the most unusual sentence: sanity AUROC 0.646 → 0.926 (D25); the length caveat is tracked in F4.
- **H1, never run on full data.** The classical side has run end to end, with three fixes (D26); the DistilBERT run on Kaggle is still pending.
- **H4, stale docs.** Updated.
- **D5, negation template rewards negative bias.** Fixed with a directional "X / not X" test scored on the model's own answers (D27).
- **G1, no multiple-comparison correction.** Fixed with Holm-adjusted p-values on every paired comparison, plus a paired AUROC test for H4 (D28).
- **C9, degenerate tree probabilities.** Fixed by a minimum leaf size of at least 10, still tuned on accuracy (D29).
