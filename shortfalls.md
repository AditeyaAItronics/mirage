# Shortfalls: known limitations of Mirage

Each item has four parts: **what** the limitation is, **why it matters** for the conclusions, a **possible fix**, and the
**effort** it would take. They are grouped by area and ranked within each group. Severity:

- 🔴 can change a headline conclusion or undermine its validity. Fix it, or state it clearly in the report.
- 🟠 weakens a result. Worth fixing if there is time.
- 🟢 a scope limit. Mention it in the report's Limitations section.

When one gets fixed, move it to the **Resolved** section at the bottom and add an entry to `majordecisions.md`.

**How to fix the open 🔴 and 🟠 items:** the detailed design for each is in
[shortfallfix.md](shortfallfix.md) (order, code, outputs, who does what, effort).

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

### B4. ✅ RESOLVED (D31): the IMDb shift mixed several changes at once
`imdb_edges` (each review's first + last sentence, same item_ids) splits the drop. Cutting a review
to two sentences makes it *harder* (LR −7.7 points for full reviews against edges). At SST
length, IMDb text is harder than SST-2 (LR +11.6). `shift_decomposition.csv` has every model. See
B10 for the label-noise caveat.

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

### B9. ✅ RESOLVED (D31): part of the negation set also picks DistilBERT's seed
`sst2_neg_testonly` (339 test-only sentences) is reported next to the pooled set in
`negmark_effect`, `negation_gap` and the heatmap. For the classical models it gives the same
conclusions. Re-check the DistilBERT gap on it when it arrives.

### B10. 🟠 NEW: `imdb_edges` inherits whole-review labels
**What:** a review's first and last sentences don't always state its verdict ("Caught this 1969
film on cable TCM one night. Long live musicals." is labelled positive).
**Why it matters:** label noise raises the edges error. So the "content" part of the IMDb drop
(D31) is an upper bound, and the "length/format" part understates how much full text helps.
**Possible fix:** hand-label about 200 edges excerpts as positive, negative or unclear, measure the
noise rate, and re-estimate on the clear ones. A sentence-level sentiment check is another option.
**Effort:** small to medium (about 1 hour of labelling).

---

## C. Models

### C1. ✅ RESOLVED (D23): IMDb comparison was unfair because of truncation
Every model now reads the same capped text (first 220 + last 140 words). DistilBERT's IMDb limit
went from 256 to 512 tokens, and anything still too long keeps its head and tail tokens, not just
the start.

### C2. ✅ ADDRESSED (D46): DistilBERT is not a "shortcut-free" reference
**What:** the project treats DistilBERT as the model that *doesn't* use shortcuts (`REFERENCE` in
`evaluate.py`). Transformers are known to use shortcuts too, such as lexical overlap and negation cues, and its
pre-training text may include review-like language.
**Why it matters:** "gap to DistilBERT" measures *relative* shortcut reliance, not absolute reliance.
**Possible fix:**
- Word the hypotheses relative to DistilBERT, and report DistilBERT's own failure rates prominently.
- Optionally add a stronger model (RoBERTa-base), or an off-the-shelf sentiment model, as a second reference.

**Done (D46):** two ready-made models go through every test: RoBERTa-large (SiEBERT) and Hugging
Face's SST-2 DistilBERT. HF DistilBERT shares our DistilBERT's typo shortcut; RoBERTa-large shrinks
the shortcuts but keeps some (interestingfindings.md §3, §4).

**Effort:** small (wording) to medium (another model).

### C12. ✅ ADDRESSED (D52): RoBERTa-large's accuracy may be contaminated (CONFIRMED LIKELY, D46: IMDb 0.956)
**Done:** every accuracy number of RoBERTa-large carries a † and a footnote (docs, Mirage Gap chart, demo).
On fresh IMDb reviews it scores 0.959, again above its SST-2 accuracy (claim 16, D53).
**What:** SiEBERT (`roberta_large`, D39) was fine-tuned on 15 sentiment datasets, very likely
including SST-2 and IMDb. It may have seen our test sentences and reviews.
**Why it matters:** its SST-2 and IMDb **accuracy** isn't a fair comparison with our models.
Its **behavioral** results are clean, because the test sentences are ours.
**Fix:** none possible without its training data. Lead with its behavioral results, and mark its
accuracy as "possibly contaminated" in every table and on the slides. More evidence: it is the only
model that is *better* on IMDb than on SST-2 (0.956 vs 0.946; interestingfindings.md §12).
**Effort:** none (wording).

### C13. 🟢 NEW: external models read SST-2's tokenized text as is
**What:** SST-2 rows are stored in SST's tokenized style ("it 's", "does n't"). Our models are
trained on that style; the external models were trained on ordinary English and read it
unchanged (D39).
**Why it matters:** a small handicap for the external models on SST-2 sets. It doesn't affect
the behavioral suite or IMDb, which are ordinary English.
**Effort:** small, if wanted: a detokenized copy of the SST-2 sets as a robustness check.

### C14. 🟢 CONFIRMED (D46: held-out 0.989 vs test 0.923): `distilbert_hf` was trained on our held-out phrases
**What:** Hugging Face's SST-2 DistilBERT was trained on all of GLUE SST-2 train, which contains
our held-out phrases. Its held-out accuracy is inflated; validation and test are unaffected.
**Effort:** none (don't report its held-out accuracy).

### C15. ✅ RESOLVED (D42): H2/H3 didn't test the features the proposal names
**What:** the proposal compares DistilBERT with *single-word* models (H2) and asks whether *word
pairs* and marking close the gap (H3), but every classical model already used word pairs.
**Fix:** a `unigram` variant completes the ladder single words → + pairs → + marking. Result:
pairs and marking together close 40% of LR's real-sentence gap (CI 21–58%).

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

### C7. ✅ RESOLVED (D31): negation marking used one simple scope rule
A second variant, `negmark_w3` (3-token scope with fixed-phrase exceptions: "not only", "no doubt"
…), is trained for every model. On real negated sentences it helps slightly more consistently than
the original rule, but no gain survives Holm (except GNB with `negmark`). The C10 conclusion
doesn't depend on the scope rule.

### C8. ✅ ANSWERED (D35): short-phrase training vs long reviews
DistilBERT's IMDb error is flat across lengths (about 0.095) and rises only for the capped
reviews (0.126). Training on short phrases didn't hurt it on long reviews; cutting text did.

### C9. ✅ RESOLVED for the tree (D29); 🟢 GNB and MNB remain as notes
**Tree:** it now keeps at least 10 examples per leaf (accuracy picked 20). Its probabilities are
usable: ECE 0.339 → 0.144, IMDb confidence AUROC 0.500 → 0.562, and IMDb accuracy *rose*
0.592 → 0.612. Cost: SST-2 test 0.661 → 0.653.
**GNB** stays near chance (SST-2 test 0.591). That's the H6 comparison, so present it as a lower
bound (C6), not a fault to tune away.
**MNB** pushes half its IMDb probabilities to exactly 1.0. The early-warning combiner handles this
(D26). Note it next to MNB's calibration numbers.

### C10. ✅ EXPLAINED (D38): negation marking helps templates, not clearly real sentences
After Holm, `NOT_` marking cuts template negation errors for every model but has no significant
effect on real negated sentences (except GNB). The mechanism:
- Real negations mark 6.9 words against 2.1 in templates, diluting the signal.
- Many real negations don't negate the sentiment (double negation, idioms).
- For LR, marking fixes 38 real sentences and breaks 23 (net +15 of 519).

It isn't a vocabulary gap (78% vs 74% of marked words known). Report the template and
real-sentence effects side by side, with this explanation.

### C11. ✅ PINNED DOWN (D37): DistilBERT has a shortcut of its own
**What:** DistilBERT is perfect on simple, negation, direction and both name tests, but when it
can't read the *sentiment word* (a typo in it), it defaults to **negative**. A positive adjective
with a typo flips it in 65% of pairs (mean −0.64); a negative one in 9%. A typo in the *noun*
changes nothing (0 of 500 flips).
**Why:** in SST-2's training phrases, fragmented, out-of-vocabulary words lean negative: 16.0% of
words in negative phrases against 10.9% in positive ones are split into word pieces.
**Why it matters:** DistilBERT is a stronger reference, not a shortcut-free one (C2). Its shortcut
is a data-learned default, different in kind from the classical models' word and name shortcuts.
**Still open (optional):** a nonce-word test ("The movie was flurbish.") would show the default
directly, without a corrupted real word.

---

## D. Behavioral tests

### D1. ✅ RESOLVED (D24, D33): templates and word lists are hand-written by us
**Done:**
- Every behavioral case records its adjective.
- `word_errors.csv` shows failures per word.
- `negation_control.csv` re-scores negation on only the positive words the model gets right in
  every simple case. For LR: 0.262 on all words, 0.284 on known words, so its negation failure is
  real and not caused by the words.

**Human review done (D33):** the project author reviewed every list (adjectives, mild adjectives,
nouns, names) and the six film-person pairs. One word changed ("messy" → "sloppy"); the rest were
kept. The lists remain author-chosen, so say so in the report's limitations section.

### D2. ✅ RESOLVED (D22): behavioral suite was small
5× larger: 2,000 cases (2,750 rows); with the directional test (D27), 2,750 cases (4,250 rows).

### D3. 🟢 "Not bad" cases are not scored
**What:** "not bad" isn't clearly positive (it's often lukewarm), so those cases are reported separately and not
scored (decision D11).
**Why it matters:** we ignore half of the negation behaviour.
**Possible fix:** label a sample of "not <negative>" sentences by hand (or with several annotators) to decide
whether they count as positive.
**Effort:** small to medium.

### D4. 🟢 A limited set of behavioral tests
**What:** simple, negation (template and directional), typo, name swap, name shortcut. Missing: intensifiers ("very", "slightly"), contrast ("but"),
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

### D7. ✅ RESOLVED (D32): the name-swap test couldn't fail and was confounded with gender
None of its Indian names occur in SST-2 training, so the classical models never saw them, and
every pair also swapped gender. It's now a gender-matched "unseen names" probe, and a new
`name_shortcut` test swaps film people whose training labels lean positive vs negative.

### D8. ✅ CONFIRMED BEYOND 6 PAIRS (D45): Naive Bayes judges films by who made them (DistilBERT: 0 flips)
**What:** swapping Kidman for Seagal (etc.) in an otherwise identical mild sentence flips Naive
Bayes's label in **37.6%** of pairs (mean probability shift +0.37; 40.4% / +0.38 on the sentence
sample before the D33 word swap reshuffled the suite). LR shifts every prediction
(+0.063) but rarely flips (0.6%). Unseen names move no model.
**Why it matters:** this is the clearest shortcut in the project so far. The prediction depends on
*who* is mentioned rather than *what* is said. It's real, learned from data, and (in this test)
wrong.
**Caveats:**
- These names stand for real track records, so the shortcut is often right on SST-2 itself.
- Six pairs and four templates. More pairs from `results/name_associations.csv` would show
  whether the effect holds beyond these.

**Effort:** small to extend (add pairs to `configs/behavioral.yaml`).

### D9. 🟢 NEW: the live demo illustrates; it isn't evidence
**What:** the Shortcut Explorer (D41) shows hand-picked sentences, and its word importance comes
from leaving one word out at a time. Removing a word can make a sentence ungrammatical, and it
misses interactions between words. Its two flag thresholds (0.70 and the 95th percentile) are
chosen, not tuned.
**Why it matters:** the slides must not quote demo sentences as results. The numbers come from
the suite and the tables; the demo makes them visible.
**Effort:** none (wording on the slide: "illustration").

### D10. ✅ RESOLVED (D53): every behavioral result rests on one set of templates and one random draw
**Done:** every case records its frame (`templates.frame_of`); a second suite with four new frames per
test and seed 2026 (`behavioral_v2`, 6,250 rows); per-frame failure rates in
`results/behavioral_by_frame.csv`. Frames differ significantly in 29 of 72 cells (interestingfindings §15);
the new weakness this found is D12.
**What:** each test uses a handful of sentence frames ("The movie was X.", "Another N comedy, and
it was X."), written by us, filled from word lists with seed 42. The suite doesn't record which
frame a case came from, and no second wording or second draw was ever tested.
**Why it matters:** a failure rate could partly reflect one frame's wording (e.g. "Another …
comedy" may itself sound negative) rather than the model's handling of negation, typos or names.
Combined with G7, the template results are less certain than their CIs suggest.
**Possible fix:** record the frame of every case; add a second, paraphrased frame set and a fresh
draw (seed 2026); report each test's failure rate per frame and on the new draw (shortfallfix.md).
**Effort:** small to medium.

### D11. 🟢 NEW: the name tests attach reputations to real people
**What:** the name-shortcut test and probe use real directors and actors, chosen because their
SST-2 training phrases lean positive or negative.
**Why it matters:** the tables can read as "X is a bad filmmaker". The reputation is the training
data's label balance, not a judgement of the person.
**Possible fix:** say so in the report and on any slide that names them; use "names whose training
phrases lean negative" rather than "bad names".
**Effort:** none (wording).

---

### D12. 🟠 NEW (D53): template results depend strongly on the sentence frame
**What:** the per-frame split shows failure rates moving by up to 100 points between frames of the same
test (LR, directional negation: 0% for "I don't think the movie was excellent", 100% for "The movie
didn't feel great"). Both DistilBERTs fail one new frame almost always ("I would never call the movie
great": 96% and 99%) while getting every other new frame right.
**Why it matters:** a template test's failure rate is an average over 3–4 frames, so its size depends on
which frames were written. Verdicts that hold on both suites are robust; exact template numbers are not.
**Possible fix:** report template effects per frame or pooled over both suites; add more frames per test
(e.g. 10) so no single frame dominates; test *never* in more constructions to see how general the
DistilBERT blind spot is.
**Effort:** small (pooling) to medium (more frames, a new suite run).

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

### E2. ✅ RESOLVED for H5 (D43); 🟢 remains for the per-model explanations: we compare explanations that mean different things
**What:** LR weights and tree importances are *global* (about the whole model), while IG is *local* (about
one prediction, then averaged). The depth-4 tree we explain is a *separate surrogate*, not the actual RF.
**Why it matters:** low overlap between models may reflect the explanation methods rather than the models.
**Possible fix:**
- Use the same local method for every model: SHAP or IG-style occlusion on the classical models.
- For trees, use the RF's own permutation importance instead of a surrogate tree.

**Effort:** medium.

### E4. 🟢 NEW: removing words is a rough measure of a word's effect
**What:** H5's ranking (D43) removes one word at a time. That can leave ungrammatical text, and it
misses words that only matter together ("not … good"). The decision tree's and DistilBERT's word
effects are also only moderately reproducible (split-half reliability 0.34 and 0.52), so their
agreement figures rest on noisier rankings, even after the correction.
**Why it matters:** H5's numbers describe single-word effects on SST-2 test sentences, not
everything a model relies on.
**Also:** the 95% intervals cover the raw correlation only, resampling words; the
reliability-corrected values have no interval of their own, and sentence-to-sentence variation is
reflected only through the reliabilities.
**Effort:** none (state it); medium to add pair removal.

### E3. 🟢 Attribution is not causation
**What:** high attribution doesn't prove the model "relies on" a word. The deletion check helps, but deleting
words also creates unnatural sentences.
**Possible fix:** replace words with a neutral substitute instead of deleting them, and report both.
**Effort:** small.

---

## F. Early warning (H4)

### F1. ✅ ANSWERED (D25, D30, D31): distance barely separated SST-2 from IMDb
**Was:** AUROC 0.646 with whole-review embeddings.
**Now:** each sentence is scored and the review keeps its most unusual one, giving 0.926. But
the length check (D30) shows most of that is length. With one random sentence per review, it's
**0.612**. The detector works; the shift itself is mostly length and format, not sentence content
(see F7).

### F2. ✅ RESOLVED (D53): The embedding model is not the classifier
**Done:** `extract-features` + `own-distance`: distance in DistilBERT's own [CLS] features detects its
errors (AUROC 0.822) but adds nothing to confidence; classical models' own TF-IDF distance stays at
chance (`results/early_warning_own.csv`, hypothesisanswers H4).
**What:** distance is measured in MiniLM's embedding space, which is the same for every classifier.
**Why it matters:** distance can't explain why *one* model fails where another doesn't. It can only flag
"unusual inputs".
**Possible fix:** for DistilBERT, also use its own `[CLS]` / last-layer features. That's the standard
Mahalanobis OOD detector (Lee et al., 2018).
**Effort:** medium.

### F3. ✅ RESOLVED (D22): few errors to detect
With 10,000 IMDb reviews, LR makes about 2,000 errors and the others make more. AUROC CIs are now about ±0.01.

### F4. ✅ RESOLVED (D30): sentence-max distance is partly a length measure
Measured, then removed:
- A length-matched sanity check (one random sentence per review) gives 0.612, against 0.926 for
  full reviews. Distance correlates 0.44 with log(number of sentences).
- A length-adjusted distance (`distance_adj`, an OLS residual on log(number of sentences)) is
  scored in H4. It does no better than raw distance (0.48–0.53) and is significantly below
  confidence for every model, so H4's conclusion doesn't depend on length.

### F5. 🟢 NEW: sentences are split with a regex
**What:** reviews are split on `. ! ?` followed by whitespace. IMDb text has "..." runs and
sentences with no space after the full stop, so some "sentences" are merged or broken.
**Possible fix:** a proper sentence splitter (`nltk` punkt or `pysbd`).
**Effort:** small.

### F6. ✅ CONFIRMED (D46): distance does not flag any model's IMDb errors
**What:** with a detector that clearly sees the domain change (0.926), error-detection AUROC for
distance is 0.49–0.53 for every classical model, while low confidence reaches 0.67–0.77 (LR,
MNB, RF).
**Why it matters:** H4 looks unsupported for the classical models. This is a *result*, not a
bug, but it should be confirmed with DistilBERT and with F4's length-matched check before it
goes in the report.
**Confirmed:** DistilBERT (0.530), HF DistilBERT (0.525) and RoBERTa-large (0.507) show the same,
and the length-adjusted distance does no better (F4). H4 is rejected for all eight models. Whether
a classifier's *own* features do better was F2 (D53): DistilBERT's own-feature distance detects
its errors (0.82) but adds nothing beyond confidence, so H4 stays rejected.

### F7. ✅ ANSWERED (D31): the SST-2 → IMDb shift is mostly length and format
**What:** sentence for sentence, an embedding detector separates SST-2 from IMDb only modestly
(0.612). Whole reviews look different (0.926) mainly because they're long and multi-sentence.
**Why it matters:**
- The report shouldn't describe IMDb as a "new domain" in content. Both are movie reviews.
- IMDb errors are better explained by length and by mixed opinions inside a review: even
  bag-of-words models get 7 points worse from short to long reviews (C8).
- This narrows what H4 can show: a sentence-level familiarity score has little to work with.

**Answered:** the shift decomposition (D31) shows the length part *helps* every model and the
errors come from content (interestingfindings.md §8; B10 is the remaining label-noise caveat).

### F8. 🟢 NEW: the early warning is scored only by AUROC
**What:** AUROC summarises every threshold at once. A deployment needs one threshold: "check the
10% of predictions with the lowest confidence; how many errors does that catch?"
**Possible fix:** report recall and precision of errors at a fixed review budget (5%, 10%, 20%) per
model, from the existing `early_warning_deciles.csv`.
**Effort:** small.

---

## G. Statistics

### G1. ✅ RESOLVED (D28): no correction for running many tests
Every paired comparison now has a p-value, adjusted with Holm–Bonferroni within its family (McNemar
per dataset, negation gaps, negation-marking effects, early-warning AUROC differences). H4 now has
a direct paired test of distance against confidence. The correction already changes a conclusion:
LR's +1.2-point gain from negation marking on SST-2 test has a raw p of 0.013 but isn't significant
after Holm.

### G2. ✅ RESOLVED (D44): confidence intervals ignore seed variance
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

### G6. 🟢 NEW: the robustness score is a design choice
**What:** the Mirage Gap's robustness score (D40) is an equal-weight mean of four tests, and
"pass" means getting *both* sentences of a pair right. That mixes competence (reading the word)
with robustness (not being fooled), on purpose: otherwise a model that can't read the word
looks robust.
**Why it matters:** another definition gives other numbers. The model **ordering** barely depends
on it: leaving out each component in turn keeps the order identical in 14 of 16 checks; the other
two swap one pair of neighbours (Spearman 0.976 and 0.900) with the same top and bottom model
(`mirage_gap_sensitivity.csv`).
**Effort:** none (report the definition and the sensitivity check in the caption).

### G7. ✅ RESOLVED (D52): confidence intervals treat clustered items as independent
**Done:** every interval resamples clusters (behavioral: the adjective; IMDb: the film; all 20,000
reviews matched, 2,796 films in the main sample). 961 intervals compared in `results/cluster_robustness.csv`: median
width ratio 1.0–1.5×, 13 significance flags changed, no H1–H6 verdict changed; RoBERTa-large's name
shift stopped being significant (interestingfindings §4 softened). The table below is the original
measurement.
**What:** every bootstrap resamples single items. But behavioral cases are built from a few
words: each test's 250–500 cases share only 10–39 adjectives (and 6 name pairs), so cases with the
same word fail or pass together. IMDb reviews also come several per film, and SST-2 sentences
several per film.
**Measured (behavioral, resampling whole words instead of cases):**

| model, test | failure rate | CI resampling cases | CI resampling words | width ratio |
|---|---|---|---|---|
| DistilBERT, typo pairs (37 words) | 0.394 | [0.352, 0.436] | [0.278, 0.510] | 2.8× |
| Multinomial NB, name pairs (10 words) | 0.386 | [0.344, 0.432] | [0.209, 0.569] | 4.1× |
| LR, directional negation (19 words) | 0.300 | [0.260, 0.342] | [0.251, 0.348] | 1.2× |

**Why it matters:** the behavioral CIs and p-values in every table (H1's template columns, H2 and
H3 on templates, typo direction, name effects, the robustness score) are too narrow, in some cases
about 3–4×. The large effects survive any reasonable widening, but borderline template results
(e.g. GNB's 25% gap closed, small name shifts) may stop being significant. Until it is checked, a
verdict that leans on a template test could change.
**Possible fix:** a cluster bootstrap (resample words, or name pairs, then their cases) as a
robustness table for every behavioral result; movie-level clusters for IMDb from the original
review URLs. Report both CIs and flag any verdict that changes (shortfallfix.md).
**Effort:** medium.

### G8. ✅ RESOLVED (D53): many analyses were designed after seeing results, with no confirmatory check
**Done:** 17 claims committed before scoring (`docs/confirmatory_claims.md`, commit fbdf48e), tested on
10,000 fresh IMDb reviews and the second suite: 16 replicated, claim 3 (H2) did not
(`results/replication.csv`).
**What:** the H1 test (D47), the robustness score (D40), the typo control (D37), the name probe
(D45) and several thresholds were designed after the first results. The findings files label them
"exploratory", but every number, exploratory or not, was computed on the same data that suggested
the analysis.
**Why it matters:** with enough choices, some patterns appear by chance ("forking paths"). An
examiner can fairly ask which results would hold on data the analysis never touched.
**Possible fix:** write down the key claims with their expected direction *before* running them
again, commit that list, then test them on untouched data: a fresh behavioral draw with paraphrased
frames (D10) and a fresh IMDb sample from the 40,000 unused reviews. Report a "replicated / not
replicated" column (shortfallfix.md).
**Effort:** medium (one Kaggle run for the transformers on IMDb).

### G9. ✅ RESOLVED (D52): significance is reported without a minimum effect that matters
**Done:** `configs/stats.yaml` sets the minimum effects; every table has `*_meaningful` columns and H1
has `clearly_higher_meaningful`. The thresholds were set after the results (stated in D52).
**What:** with thousands of items, tiny effects become significant. DistilBERT's name shift is
−0.0001 (Holm p 0.007) and the random forest's +0.008 (Holm p < 0.001); HF DistilBERT's +3.5-point
real-negation difference counts as "clearly higher" for H1 at Holm p 0.044.
**Why it matters:** "significant" can be read as "important". Verdicts should rest on effects
large enough to matter, not only on p-values.
**Possible fix:** state a smallest effect of interest per kind of measure (e.g. 2 points of error,
0.02 of probability, 0.02 of AUROC), add a "meaningful" column next to each significance flag, and
use both in the verdict wording. The thresholds are chosen after the results, so say so (G8).
**Effort:** small.

### G11. 🟠 NEW (D53): one pre-registered claim failed on new sentence frames
**What:** claim 3 (every single-word classical model ≥ 10 points worse than DistilBERT on directional
negation) failed on the second suite: the decision tree's gap is +0.4 points, the random forest's lower
bound +4.5. Our DistilBERT fails "I would never call the movie great" in 96% of cases (D12).
**Why it matters:** H2's "much better at negation" is true for the negations SST-2 shows most, not for
every construction. The report must say so; the pre-registered result can't be re-scored.
**Fix:** none for the claim. State it in H2's verdict (done in hypothesisanswers.md).
**Effort:** none (wording).

### G10. 🟢 NEW: the Mirage Gap's two axes share sentences
**What:** "looks" is SST-2 test accuracy (1,820 sentences); one of the four "behaves" checks is the
real negated sentences, 339 of which are SST-2 test sentences.
**Why it matters:** the two axes are slightly correlated by construction. The effect is small (one
check of four, 19% of the test set), and the ordering survives leaving that check out (G6).
**Effort:** none (state it); small to compute the gap against negation-free test accuracy.

---

## H. Engineering and reproducibility

### H1. ✅ RESOLVED (D26, D34, D35): not yet run end to end on the full data
Everything has now run on the real data: 5 classical models × 3 variants, DistilBERT × 3 seeds
(trained and predicted on Kaggle), and every stage of `all`, IG included. Results are in D35.

### H2. ✅ RESOLVED (D46): dependencies aren't pinned
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

### H7. 🟠 NEW: numbers in the docs are copied by hand
**What:** the results in README, hypothesisanswers.md and interestingfindings.md are typed from
the CSVs. They have already drifted twice (285 vs 291 seed checks; a sensitivity claim that stopped
being true when models were added, D50).
**Why it matters:** the report will be written from these files. A stale number in the report is
the error an examiner is most likely to catch.
**Possible fix:** a claims registry: each quoted number listed once with the CSV cell it comes
from, checked by a test, so `pytest` fails when a result changes and a doc isn't updated
(shortfallfix.md). The report's number check (whatyouneedtodo step 10) reuses it.
**Effort:** small to medium.

### H8. ✅ RESOLVED (D52): adding seeds or variants would silently change the headline tables
**Done:** `evaluate.headline_pool` keeps classical seed 42 and DistilBERT's configured seeds;
`median_seed` ranks per variant (`tests/test_headline.py`).
**What:** `evaluate.canonical` keeps *every* seed of every model except DistilBERT, and
`main_table` averages over seeds; `median_seed` assumes DistilBERT has one variant. Training the
random forest on more seeds (C4) or a tuned DistilBERT (C3) would therefore change existing
numbers without any error, against the rule that additions never change earlier statistics.
**Why it matters:** it blocks C3 and C4 until fixed, and a careless re-run could change reported
numbers unnoticed.
**Possible fix:** make the headline seed explicit per model family (classical 42; DistilBERT its
median among the configured headline seeds, per variant), keep extra seeds in robustness tables
only, and add a test that adding a seed or variant leaves `main_table` unchanged
(shortfallfix.md).
**Effort:** small.

---

## Suggested order

The design of every fix is in [shortfallfix.md](shortfallfix.md); its status table says what
is done. Done (D52, D53): H8, G7, G9, C12, D10, G8, F2. Still open, in a sensible order:
- **H7** (a test that the docs' numbers match the CSVs): protects the report; CPU only.
- **G11** (wording only) and **D12** (pool the two suites, or more frames per test).
- **B10** (hand-label 200 IMDb excerpts: your time).
- **C3, C4, E1** (tuned DistilBERT, more seeds, IG baselines): Kaggle GPU runs.

No 🔴 item is open. Everything 🟢 gets one sentence in the report's Limitations section.

## Resolved

- **G7, H8, G9, C12 (D52):** cluster bootstrap, explicit headline seeds, minimum effects, the RoBERTa caveat mark.
- **G8, D10, F2 (D53):** pre-registered replication (16 of 17 claims), frame set B and the per-frame table, own-feature distance.

- **B1, leaky held-out split.** Fixed by a sentence-grouped split (`data.heldout_mask`). Leakage went from 89.6% to 0 of 3,378 phrases. See `majordecisions.md` D21.
- **B2 (size), B3, B5, D2, F3: too little data, and validation reused as test.** Fixed by adding 1,820 SST-2 test sentences, pooling the negation set (519 sentences), using 10,000 IMDb reviews and a 2,000-case behavioral suite (D22).
- **C1, unequal IMDb truncation.** Fixed by giving every model the same capped text, plus head+tail tokens at 512 for DistilBERT (D23).
- **D1, template bias.** Per-word errors and a known-words negation control (D24), plus the author's review of every word list (D33).
- **F1, a distance blind to the domain change.** Mitigated by scoring the most unusual sentence: sanity AUROC 0.646 → 0.926 (D25); the length caveat is tracked in F4.
- **H1, never run on full data.** The classical side has run end to end, with three fixes (D26); DistilBERT followed (D34, D35).
- **H4, stale docs.** Updated.
- **D5, negation template rewards negative bias.** Fixed with a directional "X / not X" test scored on the model's own answers (D27).
- **G1, no multiple-comparison correction.** Fixed with Holm-adjusted p-values on every paired comparison, plus a paired AUROC test for H4 (D28).
- **C9, degenerate tree probabilities.** Fixed by a minimum leaf size of at least 10, still tuned on accuracy (D29).
- **F4, length inside the distance.** Measured with a length-matched check (0.612 vs 0.926) and removed with a length-adjusted distance; H4 is unchanged (D30).
- **B4, B9, C7: robustness of the headline results.** An IMDb excerpt set splits length from content, test-only negation rows were added, and a second negation-scope variant was trained (D31).
- **D7, name test that couldn't fail.** Replaced by a name-shortcut test built from the training labels, plus a gender-matched unseen-names probe (D32).
- **H1 judged by eye.** A formal test against negation-free test sentences (D47): partly supported.
- **C2, no second reference model.** HF DistilBERT and RoBERTa-large tested on everything (D46).
- **D8, six name pairs.** All 35 film people: Multinomial NB's reaction tracks 23 unseen names at ρ = 0.91 (D45).
- **H2, unpinned dependencies.** `requirements-lock.txt` (D46).
- **G2, seed variance.** Every median-seed result rerun on all three seeds: no conclusion changes (D44).
- **E2 (for H5), explanations that mean different things.** One ranking rule for every model and a reliability-corrected agreement measure (D43).
- **C15, H2/H3 tested on the wrong features.** A single-word variant completes the ladder (D42).
- **H1, C8: full run.** Pipeline complete with DistilBERT (D34, D35); the length question is answered.
- **F1, F6, F7: early warning and the IMDb shift.** Answered by the sentence-max distance and its length checks (D25, D30), the shift decomposition (D31) and the external models (D46): distance detects the shift but not the errors, and the shift is mostly length and format.
